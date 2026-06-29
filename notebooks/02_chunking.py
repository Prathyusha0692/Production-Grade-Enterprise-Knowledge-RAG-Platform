# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Chunking Layer
# MAGIC Converts Knowledge records into smaller, semantically coherent chunks ready for embedding.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. **Eligibility gates** — drop records too short, too sparse, or flagged empty
# MAGIC 2. **SOP section splitting** — SOPs split on markdown / numbered headings
# MAGIC 3. **Sliding-window chunking** — all other domains split by token count with overlap
# MAGIC 4. **Token estimation** — whitespace-approximate token count stored on each chunk
# MAGIC 5. **Quality scoring** — reject chunks that fail minimum quality thresholds
# MAGIC 6. **Output** — approved chunks written to `chunks` Delta table

# COMMAND ----------

import os, re, json, hashlib, uuid
from datetime import datetime

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import *
    spark = SparkSession.builder.getOrCreate()
    DATABRICKS = True
except ImportError:
    DATABRICKS = False
    import pandas as pd

try:
    from config.config import (
        CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_CHARS, CHUNK_MIN_WORDS,
        SOP_SECTION_PATTERN, KNOWLEDGE_TABLE, CHUNKS_TABLE, CHUNKS_PATH,
        DB_NAME, LOCAL_DATA_PATH
    )
except ImportError:
    CHUNK_MAX_TOKENS    = 400
    CHUNK_OVERLAP_TOKENS = 40
    CHUNK_MIN_CHARS     = 80
    CHUNK_MIN_WORDS     = 10
    SOP_SECTION_PATTERN = r"(?m)^#{1,3}\s+|^\d+\.\s+"
    KNOWLEDGE_TABLE     = "rag_platform.knowledge"
    CHUNKS_TABLE        = "rag_platform.chunks"
    CHUNKS_PATH         = "./chunks_delta"
    DB_NAME             = "rag_platform"
    LOCAL_DATA_PATH     = "./data"

CHUNKED_AT = datetime.utcnow()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Define chunk schema

CHUNK_SCHEMA = StructType([
    StructField("chunk_id",       StringType(),  False),
    StructField("knowledge_id",   StringType(),  False),
    StructField("domain",         StringType(),  False),
    StructField("source_id",      StringType(),  False),
    StructField("chunk_index",    IntegerType(), False),
    StructField("chunk_text",     StringType(),  False),
    StructField("token_estimate", IntegerType(), False),
    StructField("char_count",     IntegerType(), False),
    StructField("word_count",     IntegerType(), False),
    StructField("quality_score",  FloatType(),   True),
    StructField("eligible",       BooleanType(), False),
    StructField("rejection_reason", StringType(), True),
    StructField("chunk_type",     StringType(),  True),   # "section" | "sliding_window"
    StructField("metadata",       MapType(StringType(), StringType()), True),
    StructField("chunked_at",     TimestampType(), False),
]) if DATABRICKS else None

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Core chunking utilities

def estimate_tokens(text: str) -> int:
    """Whitespace-based token estimate (approx 1 token ≈ 0.75 words for English)."""
    words = text.split()
    return max(1, int(len(words) / 0.75))


def _chunk_id(knowledge_id: str, chunk_index: int) -> str:
    namespace = uuid.UUID("abcdef12-abcd-1234-abcd-abcdef123456")
    return str(uuid.uuid5(namespace, f"{knowledge_id}:{chunk_index}"))


def eligibility_gate(text: str) -> tuple[bool, str]:
    """Returns (eligible, rejection_reason). Empty reason means eligible."""
    if not text or not text.strip():
        return False, "empty_text"
    chars = len(text.strip())
    words = len(text.split())
    if chars < CHUNK_MIN_CHARS:
        return False, f"too_short_chars:{chars}<{CHUNK_MIN_CHARS}"
    if words < CHUNK_MIN_WORDS:
        return False, f"too_sparse_words:{words}<{CHUNK_MIN_WORDS}"
    # Reject chunks that are ≥ 80% numeric (likely raw data tables)
    alpha_num = sum(1 for c in text if c.isalpha())
    if alpha_num / max(chars, 1) < 0.20:
        return False, "too_few_alpha_chars"
    return True, ""


def quality_score(text: str) -> float:
    """
    Simple quality score 0-1 based on:
    - sentence count (more = better, up to 5)
    - unique word ratio (lexical diversity)
    - average sentence length
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words     = text.split()
    unique    = set(w.lower() for w in words)

    sentence_score  = min(len(sentences) / 5.0, 1.0)
    diversity_score = len(unique) / max(len(words), 1)
    avg_sent_len    = len(words) / max(len(sentences), 1)
    length_score    = min(avg_sent_len / 20.0, 1.0)   # 20 words/sentence = ideal

    return round((sentence_score + diversity_score + length_score) / 3.0, 4)


def sliding_window_chunks(text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Splits text into chunks of approximately max_tokens with overlap_tokens overlap.
    Splits on word boundaries; never breaks mid-word.
    """
    words = text.split()
    if not words:
        return []

    # Approximate: 1 word ≈ 0.75 tokens → convert token limits to word limits
    max_words     = int(max_tokens * 0.75)
    overlap_words = int(overlap * 0.75)

    if len(words) <= max_words:
        return [text]

    chunks = []
    start  = 0
    while start < len(words):
        end      = min(start + max_words, len(words))
        chunk    = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def sop_section_split(text: str) -> list[tuple[str, str]]:
    """
    Splits SOP text on markdown headings or numbered section headings.
    Returns list of (section_title, section_body) tuples.
    Falls back to sliding window if no headings found.
    """
    # Split on heading patterns while keeping the delimiter
    pattern = r'(?m)(^#{1,3}\s+[^\n]+|^\d+\.\s+[^\n]+)'
    parts   = re.split(pattern, text)

    sections = []
    i = 0
    # parts[0] is any text before the first heading
    if parts[0].strip():
        sections.append(("preamble", parts[0].strip()))

    while i + 2 <= len(parts) - 1:
        heading = parts[i + 1].strip() if i + 1 < len(parts) else ""
        body    = parts[i + 2].strip() if i + 2 < len(parts) else ""
        if heading or body:
            sections.append((heading, body))
        i += 2

    if not sections:
        # No headings found — treat as single section
        sections = [("full_document", text.strip())]

    return sections


# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Per-record chunking dispatcher

def chunk_record(record: dict) -> list[dict]:
    """
    Chunks a single Knowledge record. Returns a list of chunk dicts.
    """
    knowledge_id = record["knowledge_id"]
    domain       = record["domain"]
    source_id    = record["source_id"]
    body         = record.get("body", "") or ""
    meta         = record.get("metadata", {}) or {}

    raw_chunks = []   # list of (chunk_type, text)

    if domain == "sop":
        # Section-based splitting for SOPs
        sections = sop_section_split(body)
        for heading, section_body in sections:
            # Sub-chunk if section is too large
            sub_chunks = sliding_window_chunks(section_body, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
            for sc in sub_chunks:
                if heading and heading != "preamble" and heading != "full_document":
                    text = f"{heading}\n{sc}"
                else:
                    text = sc
                raw_chunks.append(("section", text))
    else:
        # Sliding window for all other domains
        for text in sliding_window_chunks(body, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS):
            raw_chunks.append(("sliding_window", text))

    # Build chunk rows
    chunk_rows = []
    for idx, (chunk_type, text) in enumerate(raw_chunks):
        eligible, rejection = eligibility_gate(text)
        tok  = estimate_tokens(text)
        chars = len(text)
        words = len(text.split())
        score = quality_score(text) if eligible else 0.0

        # Further reject chunks below quality threshold (only for sliding_window)
        if eligible and chunk_type == "sliding_window" and score < 0.20:
            eligible   = False
            rejection  = f"low_quality_score:{score}"

        chunk_rows.append({
            "chunk_id"        : _chunk_id(knowledge_id, idx),
            "knowledge_id"    : knowledge_id,
            "domain"          : domain,
            "source_id"       : source_id,
            "chunk_index"     : idx,
            "chunk_text"      : text,
            "token_estimate"  : tok,
            "char_count"      : chars,
            "word_count"      : words,
            "quality_score"   : score,
            "eligible"        : eligible,
            "rejection_reason": rejection if not eligible else None,
            "chunk_type"      : chunk_type,
            "metadata"        : {k: str(v) for k, v in meta.items()},
            "chunked_at"      : CHUNKED_AT,
        })

    return chunk_rows


# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Load Knowledge records and run chunker

if DATABRICKS:
    try:
        knowledge_df = spark.table(KNOWLEDGE_TABLE)
        knowledge_records = [row.asDict() for row in knowledge_df.collect()]
    except Exception as e:
        raise RuntimeError(f"Cannot read {KNOWLEDGE_TABLE}: {e}. Run notebook 01 first.")
else:
    import json as _json
    kpath = os.path.join(LOCAL_DATA_PATH, "knowledge.json")
    if not os.path.exists(kpath):
        raise FileNotFoundError(f"knowledge.json not found at {kpath}. Run notebook 01 first.")
    with open(kpath) as f:
        knowledge_records = _json.load(f)

print(f"Loaded {len(knowledge_records)} Knowledge records")

# COMMAND ----------

all_chunks  = []
stats       = {"total": 0, "eligible": 0, "rejected": 0, "by_domain": {}}

for record in knowledge_records:
    chunks = chunk_record(record)
    all_chunks.extend(chunks)

    domain = record["domain"]
    if domain not in stats["by_domain"]:
        stats["by_domain"][domain] = {"total": 0, "eligible": 0, "rejected": 0}

    for c in chunks:
        stats["total"] += 1
        stats["by_domain"][domain]["total"] += 1
        if c["eligible"]:
            stats["eligible"] += 1
            stats["by_domain"][domain]["eligible"] += 1
        else:
            stats["rejected"] += 1
            stats["by_domain"][domain]["rejected"] += 1

print(f"\nChunking complete: {stats['total']} total | {stats['eligible']} eligible | {stats['rejected']} rejected")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Write chunks to Delta / JSON

if DATABRICKS:
    # Convert timestamps for Spark
    spark_rows = []
    for c in all_chunks:
        row = dict(c)
        row["chunked_at"] = row["chunked_at"].isoformat()
        spark_rows.append(row)

    df = spark.createDataFrame(spark_rows, schema=CHUNK_SCHEMA)
    (df.write
       .format("delta")
       .mode("overwrite")          # overwrite = full re-chunk on each run
       .option("overwriteSchema", "true")
       .save(CHUNKS_PATH))

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE}
        USING DELTA LOCATION '{CHUNKS_PATH}'
    """)
    print(f"Written {len(all_chunks)} chunks to {CHUNKS_TABLE}")

else:
    import json as _json
    out_path = os.path.join(LOCAL_DATA_PATH, "chunks.json")
    serial   = []
    for c in all_chunks:
        row = dict(c)
        row["chunked_at"] = row["chunked_at"].isoformat()
        serial.append(row)
    with open(out_path, "w") as f:
        _json.dump(serial, f, indent=2)
    print(f"Written {len(all_chunks)} chunks to {out_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Chunking summary report

print("\n── Chunking summary ─────────────────────────────────────────────")
print(f"{'Domain':<15} {'Total':>7} {'Eligible':>9} {'Rejected':>9} {'Elig%':>7}")
print("─" * 52)
for domain, s in stats["by_domain"].items():
    pct = 100 * s["eligible"] / max(s["total"], 1)
    print(f"  {domain:<13} {s['total']:>7} {s['eligible']:>9} {s['rejected']:>9} {pct:>6.1f}%")
print("─" * 52)
tot = stats["total"]
eli = stats["eligible"]
rej = stats["rejected"]
print(f"  {'TOTAL':<13} {tot:>7} {eli:>9} {rej:>9} {100*eli/max(tot,1):>6.1f}%")

# Rejection reason breakdown
from collections import Counter
reasons = Counter(c["rejection_reason"] for c in all_chunks if not c["eligible"])
if reasons:
    print("\nRejection reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason:<45} {count:>5}")

# Eligible chunk token stats
eligible_tokens = [c["token_estimate"] for c in all_chunks if c["eligible"]]
if eligible_tokens:
    print(f"\nEligible chunk token distribution:")
    print(f"  min={min(eligible_tokens)}  max={max(eligible_tokens)}  "
          f"avg={sum(eligible_tokens)/len(eligible_tokens):.1f}  "
          f"p50={sorted(eligible_tokens)[len(eligible_tokens)//2]}")
