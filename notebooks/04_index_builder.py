# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Index Builder
# MAGIC Joins approved chunks with successful embeddings and builds a versioned FAISS index.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Join integrity check — every embedding must have a matching approved chunk
# MAGIC 2. Strict embedding dimension validation (all vectors must be exactly `EMBEDDING_DIM`)
# MAGIC 3. L2-normalise vectors → cosine similarity becomes inner product (IndexFlatIP)
# MAGIC 4. Build `IndexFlatIP` (exact, no approximation) — production can swap to IVF/HNSW
# MAGIC 5. Save: FAISS index binary, ID mapping JSON, metadata snapshot JSON, manifest Delta row
# MAGIC 6. Increment version number deterministically (v1 → v2 → …)

# COMMAND ----------

import os, json, math, struct, hashlib
from datetime import datetime

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss / numpy not installed. Install with: pip install faiss-cpu numpy")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    spark = SparkSession.builder.getOrCreate()
    DATABRICKS = True
except ImportError:
    DATABRICKS = False

try:
    from config.config import (
        EMBEDDING_DIM, CHUNKS_TABLE, EMBEDDINGS_TABLE, INDEX_MANIFEST_TABLE,
        INDEX_PATH as _INDEX_PATH_ADLS, LOCAL_INDEX_PATH, LOCAL_DATA_PATH,
        DB_NAME, INDEX_VERSION_PREFIX
    )
    INDEX_PATH = _INDEX_PATH_ADLS if DATABRICKS else LOCAL_INDEX_PATH
except ImportError:
    EMBEDDING_DIM          = 1536
    CHUNKS_TABLE           = "rag_platform.chunks"
    EMBEDDINGS_TABLE       = "rag_platform.embeddings"
    INDEX_MANIFEST_TABLE   = "rag_platform.index_manifest"
    INDEX_PATH             = "./index"
    LOCAL_DATA_PATH        = "./data"
    DB_NAME                = "rag_platform"
    INDEX_VERSION_PREFIX   = "v"

BUILD_AT = datetime.utcnow()
os.makedirs(INDEX_PATH, exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Determine next version number

def get_next_version(index_dir: str, prefix: str) -> str:
    """Scans index directory for existing versioned folders and increments."""
    existing = []
    if os.path.isdir(index_dir):
        for d in os.listdir(index_dir):
            if d.startswith(prefix) and d[len(prefix):].isdigit():
                existing.append(int(d[len(prefix):]))
    next_v = max(existing, default=0) + 1
    return f"{prefix}{next_v}"

VERSION    = get_next_version(INDEX_PATH, INDEX_VERSION_PREFIX)
VER_PATH   = os.path.join(INDEX_PATH, VERSION)
os.makedirs(VER_PATH, exist_ok=True)
print(f"Building index version: {VERSION}  →  {VER_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load chunks and embeddings

if DATABRICKS:
    try:
        chunks_df = (spark.table(CHUNKS_TABLE)
                         .filter(F.col("eligible") == True)
                         .select("chunk_id", "knowledge_id", "domain", "source_id",
                                 "chunk_text", "token_estimate", "metadata"))
        chunk_map = {row["chunk_id"]: row.asDict() for row in chunks_df.collect()}
    except Exception as e:
        raise RuntimeError(f"Cannot read {CHUNKS_TABLE}: {e}")

    try:
        emb_df   = spark.table(EMBEDDINGS_TABLE).select("chunk_id", "embedding", "model", "dim")
        emb_rows = [row.asDict() for row in emb_df.collect()]
    except Exception as e:
        raise RuntimeError(f"Cannot read {EMBEDDINGS_TABLE}: {e}")
else:
    import json as _j
    cpath = os.path.join(LOCAL_DATA_PATH, "chunks.json")
    epath = os.path.join(LOCAL_DATA_PATH, "embeddings.json")
    if not os.path.exists(cpath):
        raise FileNotFoundError("chunks.json not found. Run notebook 02.")
    if not os.path.exists(epath):
        raise FileNotFoundError("embeddings.json not found. Run notebook 03.")
    with open(cpath) as f:
        all_chunks = _j.load(f)
    with open(epath) as f:
        emb_rows = _j.load(f)
    chunk_map = {c["chunk_id"]: c for c in all_chunks if c.get("eligible")}

print(f"Eligible chunks loaded : {len(chunk_map)}")
print(f"Embedding rows loaded  : {len(emb_rows)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Join integrity check

orphan_embeddings = [e for e in emb_rows if e["chunk_id"] not in chunk_map]
if orphan_embeddings:
    print(f"[WARN] {len(orphan_embeddings)} embeddings have no matching approved chunk — skipping")

joined = []
for e in emb_rows:
    if e["chunk_id"] in chunk_map:
        entry = dict(chunk_map[e["chunk_id"]])
        entry["embedding"] = e["embedding"]
        entry["model"]     = e.get("model", "")
        joined.append(entry)

missing_embeddings = [cid for cid in chunk_map if cid not in {e["chunk_id"] for e in emb_rows}]
if missing_embeddings:
    print(f"[WARN] {len(missing_embeddings)} approved chunks have no embedding — excluded from index")

print(f"Join result: {len(joined)} chunks will enter the index")
assert len(joined) > 0, "No joined records — cannot build index"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Embedding dimension validation

dim_errors = []
for row in joined:
    vec = row["embedding"]
    if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
        dim_errors.append((row["chunk_id"], len(vec) if isinstance(vec, list) else type(vec)))

if dim_errors:
    print(f"[ERROR] {len(dim_errors)} dimension mismatches found — aborting index build:")
    for cid, dim in dim_errors[:10]:
        print(f"  chunk_id={cid}  dim={dim}")
    raise ValueError(f"Embedding dimension validation failed ({len(dim_errors)} errors). "
                     f"Expected dim={EMBEDDING_DIM}.")

print(f"Dimension validation PASSED — all {len(joined)} vectors have dim={EMBEDDING_DIM}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Build FAISS index (IndexFlatIP + L2-normalisation for cosine)

if not FAISS_AVAILABLE:
    raise RuntimeError("faiss not installed. Run: pip install faiss-cpu numpy")

import numpy as np

# Deterministic order: sort by chunk_id for reproducible index → id mapping
joined.sort(key=lambda r: r["chunk_id"])

# Stack vectors into numpy matrix
vectors = np.array([r["embedding"] for r in joined], dtype=np.float32)

# L2-normalise each row → inner product == cosine similarity
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
vectors_normalised = vectors / norms

print(f"Vector matrix shape: {vectors_normalised.shape}")
print(f"Norm range after normalisation: [{vectors_normalised.dot(vectors_normalised.T).diagonal().min():.4f}, "
      f"{vectors_normalised.dot(vectors_normalised.T).diagonal().max():.4f}]  (should be ~1.0)")

# Build FAISS IndexFlatIP (exact inner product)
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(vectors_normalised)
print(f"FAISS index built: ntotal={index.ntotal}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Save index artifacts

# 6a. FAISS binary
faiss_path = os.path.join(VER_PATH, "index.faiss")
faiss.write_index(index, faiss_path)
faiss_size = os.path.getsize(faiss_path)
print(f"Saved FAISS index: {faiss_path}  ({faiss_size/1024:.1f} KB)")

# 6b. ID mapping: FAISS integer index → chunk_id
id_map = {str(i): r["chunk_id"] for i, r in enumerate(joined)}
id_map_path = os.path.join(VER_PATH, "id_map.json")
with open(id_map_path, "w") as f:
    json.dump(id_map, f)
print(f"Saved ID map:      {id_map_path}  ({len(id_map)} entries)")

# 6c. Metadata snapshot — enough to reconstruct context without hitting Delta
meta_snapshot = []
for i, r in enumerate(joined):
    meta_snapshot.append({
        "faiss_idx"    : i,
        "chunk_id"     : r["chunk_id"],
        "knowledge_id" : r["knowledge_id"],
        "domain"       : r["domain"],
        "source_id"    : r["source_id"],
        "chunk_text"   : r["chunk_text"],
        "token_estimate": r.get("token_estimate", 0),
        "metadata"     : r.get("metadata", {}),
    })
meta_path = os.path.join(VER_PATH, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta_snapshot, f, indent=2)
meta_size = os.path.getsize(meta_path)
print(f"Saved metadata:    {meta_path}  ({meta_size/1024:.1f} KB)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Write manifest

# Checksum of FAISS binary for integrity verification
with open(faiss_path, "rb") as f:
    faiss_checksum = hashlib.sha256(f.read()).hexdigest()

manifest = {
    "version"        : VERSION,
    "built_at"       : BUILD_AT.isoformat(),
    "model"          : joined[0].get("model", "") if joined else "",
    "embedding_dim"  : EMBEDDING_DIM,
    "index_type"     : "IndexFlatIP_cosine_normalised",
    "ntotal"         : index.ntotal,
    "faiss_path"     : faiss_path,
    "id_map_path"    : id_map_path,
    "metadata_path"  : meta_path,
    "faiss_size_bytes": faiss_size,
    "faiss_sha256"   : faiss_checksum,
    "domain_counts"  : {},
}
for r in joined:
    d = r["domain"]
    manifest["domain_counts"][d] = manifest["domain_counts"].get(d, 0) + 1

manifest_path = os.path.join(VER_PATH, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Saved manifest:    {manifest_path}")

# Persist manifest to Delta / JSON log
if DATABRICKS:
    from pyspark.sql.types import StructType, StructField, StringType
    man_flat = {k: str(v) for k, v in manifest.items() if not isinstance(v, dict)}
    man_flat["domain_counts"] = json.dumps(manifest["domain_counts"])
    schema = StructType([StructField(k, StringType(), True) for k in man_flat])
    df_m = spark.createDataFrame([man_flat], schema=schema)
    mpath = os.path.join(INDEX_PATH, "manifest_delta")
    (df_m.write.format("delta").mode("append").save(mpath))
    spark.sql(f"CREATE TABLE IF NOT EXISTS {INDEX_MANIFEST_TABLE} USING DELTA LOCATION '{mpath}'")
else:
    log_path = os.path.join(INDEX_PATH, "manifest_log.json")
    history  = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = json.load(f)
    history.append(manifest)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Manifest log updated: {log_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Build summary

print("\n── Index build summary ──────────────────────────────────────────")
print(f"  Version         : {VERSION}")
print(f"  Index type      : IndexFlatIP (cosine via L2-normalisation)")
print(f"  Total vectors   : {index.ntotal}")
print(f"  Embedding dim   : {EMBEDDING_DIM}")
print(f"  FAISS size      : {faiss_size/1024:.1f} KB")
print(f"  SHA-256         : {faiss_checksum[:16]}…")
print(f"\n  Domain breakdown:")
for domain, count in sorted(manifest["domain_counts"].items()):
    print(f"    {domain:<15} {count:>5} chunks")
print(f"\n  Index artifacts saved to: {VER_PATH}")
print(f"  Built at: {BUILD_AT.isoformat()}")
print("─" * 65)
print(f"\n[OK] Index {VERSION} is ready for retrieval (notebook 05).")
