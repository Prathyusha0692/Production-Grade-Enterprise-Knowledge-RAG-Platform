# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Embedding Pipeline
# MAGIC Embeds all eligible chunks using the configured embedding model via REST API.
# MAGIC
# MAGIC **Features:**
# MAGIC - Reads API key from **Databricks Secrets** (falls back to env var locally)
# MAGIC - **Idempotent re-runs**: skips chunk_ids already present in the embeddings table
# MAGIC - **Deterministic batching**: sorted chunk_id order ensures reproducible batches
# MAGIC - **Exponential-backoff retries** on transient failures (429, 5xx)
# MAGIC - **Quarantine**: chunks that fail after all retries are written to a quarantine table
# MAGIC - **Dimension validation**: rejects embeddings with wrong vector length
# MAGIC - **Run metrics**: every run writes a metrics row (batch count, success, quarantined, elapsed)

# COMMAND ----------

import os, json, time, math, hashlib
from datetime import datetime
from typing import Optional

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import *
    spark = SparkSession.builder.getOrCreate()
    DATABRICKS = True
except ImportError:
    DATABRICKS = False

try:
    from config.config import (
        SECRET_SCOPE, OPENAI_API_KEY_NAME,
        EMBEDDING_MODEL, EMBEDDING_ENDPOINT, EMBEDDING_DIM,
        EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_RETRIES, EMBEDDING_RETRY_BACKOFF,
        CHUNKS_TABLE, EMBEDDINGS_TABLE, QUARANTINE_TABLE, EMBEDDING_METRICS_TABLE,
        EMBEDDINGS_PATH, DB_NAME, LOCAL_DATA_PATH
    )
except ImportError:
    SECRET_SCOPE           = "rag-secrets"
    OPENAI_API_KEY_NAME    = "openai-api-key"
    EMBEDDING_MODEL        = "text-embedding-3-small"
    EMBEDDING_ENDPOINT     = "https://api.openai.com/v1/embeddings"
    EMBEDDING_DIM          = 1536
    EMBEDDING_BATCH_SIZE   = 64
    EMBEDDING_MAX_RETRIES  = 3
    EMBEDDING_RETRY_BACKOFF = 2.0
    CHUNKS_TABLE           = "rag_platform.chunks"
    EMBEDDINGS_TABLE       = "rag_platform.embeddings"
    QUARANTINE_TABLE       = "rag_platform.quarantine"
    EMBEDDING_METRICS_TABLE = "rag_platform.embedding_metrics"
    EMBEDDINGS_PATH        = "./embeddings_delta"
    DB_NAME                = "rag_platform"
    LOCAL_DATA_PATH        = "./data"

RUN_ID  = f"embed-run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
RUN_AT  = datetime.utcnow()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Retrieve API key

def get_api_key() -> str:
    """Fetch from Databricks Secrets, fall back to environment variable."""
    if DATABRICKS:
        try:
            return dbutils.secrets.get(scope=SECRET_SCOPE, key=OPENAI_API_KEY_NAME)
        except Exception as e:
            print(f"[WARN] Databricks secret retrieval failed: {e}. Trying env var.")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "No API key found. Set OPENAI_API_KEY env var or configure Databricks Secrets."
        )
    return key

API_KEY = get_api_key()
print(f"API key loaded: {'*' * 8}{API_KEY[-4:] if len(API_KEY) >= 4 else '****'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. REST embedding caller with retries

import urllib.request
import urllib.error

def call_embedding_api(texts: list[str], api_key: str, model: str,
                       max_retries: int, backoff: float) -> dict:
    """
    Calls the OpenAI embeddings endpoint.
    Returns the full response dict on success.
    Raises RuntimeError after max_retries exhausted.
    """
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    headers = {
        "Content-Type" : "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(EMBEDDING_ENDPOINT, data=payload, headers=headers, method="POST")

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            status = e.code
            body   = e.read().decode("utf-8", errors="replace")
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = backoff ** attempt
                print(f"    [RETRY {attempt}/{max_retries}] HTTP {status} — waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Embedding API error HTTP {status}: {body}")
        except Exception as e:
            if attempt < max_retries:
                wait = backoff ** attempt
                print(f"    [RETRY {attempt}/{max_retries}] {type(e).__name__}: {e} — waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Embedding API call failed after {max_retries} retries: {e}")

    raise RuntimeError("Unreachable: retry loop exhausted without raising")


def validate_embedding(vector: list, dim: int) -> tuple[bool, str]:
    """Returns (valid, reason). Checks dimension, type, and NaN/Inf."""
    if not isinstance(vector, list):
        return False, f"not_a_list:{type(vector)}"
    if len(vector) != dim:
        return False, f"wrong_dim:{len(vector)}!={dim}"
    if not all(isinstance(v, (int, float)) for v in vector):
        return False, "non_numeric_value"
    if any(math.isnan(v) or math.isinf(v) for v in vector):
        return False, "nan_or_inf_in_vector"
    return True, ""

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load eligible chunks — skip already embedded (idempotent)

if DATABRICKS:
    try:
        chunks_df = (spark.table(CHUNKS_TABLE)
                         .filter(F.col("eligible") == True))
        chunk_records = [row.asDict() for row in chunks_df.collect()]
    except Exception as e:
        raise RuntimeError(f"Cannot read {CHUNKS_TABLE}: {e}. Run notebook 02 first.")

    # Load already-embedded chunk IDs
    try:
        existing_ids = set(
            spark.table(EMBEDDINGS_TABLE)
                 .select("chunk_id")
                 .rdd.flatMap(lambda r: [r[0]])
                 .collect()
        )
        print(f"Already embedded chunk IDs: {len(existing_ids)}")
    except Exception:
        existing_ids = set()
        print("Embeddings table does not exist yet — full embedding run")
else:
    import json as _json
    cpath = os.path.join(LOCAL_DATA_PATH, "chunks.json")
    if not os.path.exists(cpath):
        raise FileNotFoundError("chunks.json not found. Run notebook 02 first.")
    with open(cpath) as f:
        all_chunks = _json.load(f)
    chunk_records = [c for c in all_chunks if c.get("eligible")]
    existing_ids  = set()

# Deterministic order: sort by chunk_id for reproducible batches
chunk_records.sort(key=lambda r: r["chunk_id"])
# Filter out already-embedded
pending = [r for r in chunk_records if r["chunk_id"] not in existing_ids]

print(f"Eligible chunks: {len(chunk_records)}  |  Already embedded: {len(existing_ids)}  |  Pending: {len(pending)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Run embedding in deterministic batches

embedding_rows   = []
quarantine_rows  = []
batch_count      = 0
success_count    = 0
quarantine_count = 0
t_start          = time.time()

# Chunk pending into batches
batches = [pending[i:i+EMBEDDING_BATCH_SIZE]
           for i in range(0, len(pending), EMBEDDING_BATCH_SIZE)]

total_batches = len(batches)
print(f"\nRunning {total_batches} batches of ≤ {EMBEDDING_BATCH_SIZE} chunks each...\n")

for b_idx, batch in enumerate(batches):
    batch_count += 1
    texts     = [c["chunk_text"] for c in batch]
    chunk_ids = [c["chunk_id"]   for c in batch]

    print(f"  Batch {b_idx+1:>4}/{total_batches}  chunks={len(batch)}", end="  ")

    try:
        response = call_embedding_api(texts, API_KEY, EMBEDDING_MODEL,
                                      EMBEDDING_MAX_RETRIES, EMBEDDING_RETRY_BACKOFF)
        # Parse response
        data = response.get("data", [])
        if len(data) != len(batch):
            raise RuntimeError(f"Response length mismatch: expected {len(batch)}, got {len(data)}")

        usage = response.get("usage", {})
        batch_tokens = usage.get("total_tokens", 0)
        print(f"tokens={batch_tokens}", end="  ")

        batch_ok = 0
        for item, chunk in zip(data, batch):
            vector  = item.get("embedding", [])
            valid, reason = validate_embedding(vector, EMBEDDING_DIM)

            if valid:
                embedding_rows.append({
                    "chunk_id"    : chunk["chunk_id"],
                    "knowledge_id": chunk["knowledge_id"],
                    "domain"      : chunk["domain"],
                    "source_id"   : chunk["source_id"],
                    "embedding"   : vector,
                    "model"       : EMBEDDING_MODEL,
                    "dim"         : EMBEDDING_DIM,
                    "run_id"      : RUN_ID,
                    "embedded_at" : RUN_AT.isoformat(),
                })
                success_count += 1
                batch_ok      += 1
            else:
                quarantine_rows.append({
                    "chunk_id"    : chunk["chunk_id"],
                    "knowledge_id": chunk["knowledge_id"],
                    "domain"      : chunk["domain"],
                    "reason"      : f"invalid_embedding:{reason}",
                    "run_id"      : RUN_ID,
                    "quarantined_at": RUN_AT.isoformat(),
                })
                quarantine_count += 1

        print(f"ok={batch_ok}  quarantined={len(batch)-batch_ok}")

    except RuntimeError as e:
        print(f"FAILED — quarantining {len(batch)} chunks: {e}")
        for chunk in batch:
            quarantine_rows.append({
                "chunk_id"    : chunk["chunk_id"],
                "knowledge_id": chunk["knowledge_id"],
                "domain"      : chunk["domain"],
                "reason"      : f"api_error:{str(e)[:200]}",
                "run_id"      : RUN_ID,
                "quarantined_at": RUN_AT.isoformat(),
            })
            quarantine_count += 1

elapsed = time.time() - t_start
print(f"\nEmbedding complete: {success_count} success | {quarantine_count} quarantined | {elapsed:.1f}s elapsed")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Persist embeddings, quarantine, and run metrics

def write_json(path, rows):
    import json as _j
    with open(path, "w") as f:
        _j.dump(rows, f, indent=2)
    print(f"  Written {len(rows)} rows → {path}")

if DATABRICKS:
    EMBEDDING_SCHEMA = StructType([
        StructField("chunk_id",     StringType(), False),
        StructField("knowledge_id", StringType(), False),
        StructField("domain",       StringType(), False),
        StructField("source_id",    StringType(), False),
        StructField("embedding",    ArrayType(FloatType()), False),
        StructField("model",        StringType(), True),
        StructField("dim",          IntegerType(), True),
        StructField("run_id",       StringType(), True),
        StructField("embedded_at",  StringType(), True),
    ])

    if embedding_rows:
        df_emb = spark.createDataFrame(embedding_rows, schema=EMBEDDING_SCHEMA)
        (df_emb.write.format("delta").mode("append").save(EMBEDDINGS_PATH))
        spark.sql(f"CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} USING DELTA LOCATION '{EMBEDDINGS_PATH}'")
        print(f"Appended {len(embedding_rows)} embeddings to {EMBEDDINGS_TABLE}")

    QUARANTINE_SCHEMA = StructType([
        StructField("chunk_id",       StringType(), True),
        StructField("knowledge_id",   StringType(), True),
        StructField("domain",         StringType(), True),
        StructField("reason",         StringType(), True),
        StructField("run_id",         StringType(), True),
        StructField("quarantined_at", StringType(), True),
    ])
    if quarantine_rows:
        df_q = spark.createDataFrame(quarantine_rows, schema=QUARANTINE_SCHEMA)
        qpath = EMBEDDINGS_PATH.replace("embeddings", "quarantine")
        (df_q.write.format("delta").mode("append").save(qpath))
        spark.sql(f"CREATE TABLE IF NOT EXISTS {QUARANTINE_TABLE} USING DELTA LOCATION '{qpath}'")
        print(f"Appended {len(quarantine_rows)} quarantine rows to {QUARANTINE_TABLE}")

else:
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
    write_json(os.path.join(LOCAL_DATA_PATH, "embeddings.json"), embedding_rows)
    if quarantine_rows:
        write_json(os.path.join(LOCAL_DATA_PATH, "quarantine.json"), quarantine_rows)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Log run metrics

metrics_row = {
    "run_id"            : RUN_ID,
    "run_at"            : RUN_AT.isoformat(),
    "model"             : EMBEDDING_MODEL,
    "embedding_dim"     : EMBEDDING_DIM,
    "batch_size"        : EMBEDDING_BATCH_SIZE,
    "total_pending"     : len(pending),
    "batch_count"       : batch_count,
    "success_count"     : success_count,
    "quarantine_count"  : quarantine_count,
    "elapsed_seconds"   : round(elapsed, 2),
    "chunks_per_second" : round(success_count / max(elapsed, 0.001), 2),
}

if DATABRICKS:
    METRICS_SCHEMA = StructType([StructField(k, StringType(), True) for k in metrics_row])
    df_m = spark.createDataFrame([{k: str(v) for k, v in metrics_row.items()}],
                                  schema=METRICS_SCHEMA)
    mpath = EMBEDDINGS_PATH.replace("embeddings", "embedding_metrics")
    (df_m.write.format("delta").mode("append").save(mpath))
    spark.sql(f"CREATE TABLE IF NOT EXISTS {EMBEDDING_METRICS_TABLE} USING DELTA LOCATION '{mpath}'")
else:
    metrics_path = os.path.join(LOCAL_DATA_PATH, "embedding_metrics.json")
    existing_metrics = []
    if os.path.exists(metrics_path):
        import json as _j
        with open(metrics_path) as f:
            existing_metrics = _j.load(f)
    existing_metrics.append(metrics_row)
    import json as _j
    with open(metrics_path, "w") as f:
        _j.dump(existing_metrics, f, indent=2)

print("\n── Run metrics ──────────────────────────────────────────────────")
for k, v in metrics_row.items():
    print(f"  {k:<25} {v}")
