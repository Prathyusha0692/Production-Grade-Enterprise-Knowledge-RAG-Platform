# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Retrieval
# MAGIC Given a user query, retrieves the most relevant evidence chunks from the FAISS index.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Load the latest (or specified) versioned FAISS index + metadata snapshot
# MAGIC 2. Embed the query with the **same model** used during indexing
# MAGIC 3. L2-normalise the query vector (required for cosine similarity via IndexFlatIP)
# MAGIC 4. Search FAISS for top-K candidates
# MAGIC 5. Apply **hard filters** (domain filter, similarity threshold floor)
# MAGIC 6. Enforce **token budget** — select as many high-scoring chunks as fit
# MAGIC 7. Return a ranked `EvidenceSet` list ready for prompt assembly

# COMMAND ----------

import os, json, math
from typing import Optional

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss/numpy not installed: pip install faiss-cpu numpy")

try:
    from config.config import (
        EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_ENDPOINT,
        RETRIEVAL_TOP_K, RETRIEVAL_FINAL_K, SIMILARITY_THRESHOLD,
        EVIDENCE_TOKEN_BUDGET, HARD_FILTER_DOMAIN_KEY,
        SECRET_SCOPE, OPENAI_API_KEY_NAME,
        INDEX_PATH, INDEX_VERSION_PREFIX, LOCAL_DATA_PATH
    )
except ImportError:
    EMBEDDING_DIM            = 1536
    EMBEDDING_MODEL          = "text-embedding-3-small"
    EMBEDDING_ENDPOINT       = "https://api.openai.com/v1/embeddings"
    RETRIEVAL_TOP_K          = 20
    RETRIEVAL_FINAL_K        = 5
    SIMILARITY_THRESHOLD     = 0.70
    EVIDENCE_TOKEN_BUDGET    = 2000
    HARD_FILTER_DOMAIN_KEY   = "domain"
    SECRET_SCOPE             = "rag-secrets"
    OPENAI_API_KEY_NAME      = "embedding-api-key"
    INDEX_PATH               = "./index"
    LOCAL_INDEX_PATH         = "./index"
    INDEX_VERSION_PREFIX     = "v"
    LOCAL_DATA_PATH          = "./data"

try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    DATABRICKS = True
except ImportError:
    DATABRICKS = False

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Index loader

def _latest_version(index_dir: str, prefix: str) -> Optional[str]:
    """Returns the highest numbered version folder, or None."""
    existing = []
    if os.path.isdir(index_dir):
        for d in os.listdir(index_dir):
            if d.startswith(prefix) and d[len(prefix):].isdigit():
                existing.append((int(d[len(prefix):]), d))
    if not existing:
        return None
    return max(existing, key=lambda x: x[0])[1]


def load_index(version: Optional[str] = None) -> dict:
    """
    Loads FAISS index, ID map, and metadata snapshot for the given version.
    Defaults to the latest version if version is None.
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss not installed.")

    ver = version or _latest_version(INDEX_PATH, INDEX_VERSION_PREFIX)
    if not ver:
        raise FileNotFoundError(f"No index versions found in {INDEX_PATH}. Run notebook 04.")

    ver_path = os.path.join(INDEX_PATH, ver)
    print(f"Loading index version: {ver}  from  {ver_path}")

    faiss_path = os.path.join(ver_path, "index.faiss")
    id_path    = os.path.join(ver_path, "id_map.json")
    meta_path  = os.path.join(ver_path, "metadata.json")

    for p in [faiss_path, id_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing index artifact: {p}")

    index = faiss.read_index(faiss_path)
    with open(id_path)   as f: id_map    = json.load(f)
    with open(meta_path) as f: meta_snap = json.load(f)

    # Build fast lookup: chunk_id → metadata row
    meta_by_chunk = {row["chunk_id"]: row for row in meta_snap}

    print(f"  Index ntotal : {index.ntotal}")
    print(f"  ID map size  : {len(id_map)}")
    print(f"  Metadata rows: {len(meta_snap)}")

    return {
        "version"       : ver,
        "index"         : index,
        "id_map"        : id_map,          # faiss_int_str → chunk_id
        "meta_by_chunk" : meta_by_chunk,   # chunk_id → metadata dict
    }


# ── Load index at import/startup ─────────────────────────────────────────────
INDEX_STATE = load_index()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Query embedding

def get_api_key() -> str:
    if DATABRICKS:
        try:
            return dbutils.secrets.get(scope=SECRET_SCOPE, key=OPENAI_API_KEY_NAME)
        except Exception:
            pass
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    return key

_API_KEY = get_api_key()


import urllib.request, urllib.error

def embed_query(query: str) -> "np.ndarray":
    """Embeds a single query text and returns an L2-normalised float32 array."""
    payload = json.dumps({"model": EMBEDDING_MODEL, "input": [query]}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_API_KEY}"}
    req     = urllib.request.Request(EMBEDDING_ENDPOINT, data=payload, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    vector = data["data"][0]["embedding"]
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(f"Query embedding dim mismatch: got {len(vector)}, expected {EMBEDDING_DIM}")

    vec = np.array(vector, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec   # shape (1, EMBEDDING_DIM)


# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Core retrieval function

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def retrieve(
    query: str,
    top_k: int               = RETRIEVAL_TOP_K,
    final_k: int             = RETRIEVAL_FINAL_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    token_budget: int        = EVIDENCE_TOKEN_BUDGET,
    domain_filter: Optional[str] = None,
    index_state: Optional[dict]  = None,
) -> list[dict]:
    """
    Main retrieval entry point.

    Returns a ranked list of evidence dicts:
        {
          chunk_id, knowledge_id, domain, source_id,
          chunk_text, token_estimate, metadata,
          faiss_idx, similarity_score, rank
        }
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss not installed.")

    state       = index_state or INDEX_STATE
    index       = state["index"]
    id_map      = state["id_map"]
    meta_by_chunk = state["meta_by_chunk"]

    # 1. Embed query
    query_vec = embed_query(query)

    # 2. FAISS search — retrieve top_k candidates
    scores, indices = index.search(query_vec, top_k)
    scores   = scores[0].tolist()
    indices  = indices[0].tolist()

    # 3. Map FAISS integer indices to chunk metadata
    candidates = []
    for faiss_idx, score in zip(indices, scores):
        if faiss_idx < 0:
            continue   # FAISS returns -1 for empty slots
        chunk_id = id_map.get(str(faiss_idx))
        if not chunk_id:
            continue
        meta_row = meta_by_chunk.get(chunk_id)
        if not meta_row:
            continue
        candidates.append({
            "chunk_id"      : chunk_id,
            "knowledge_id"  : meta_row.get("knowledge_id", ""),
            "domain"        : meta_row.get("domain", ""),
            "source_id"     : meta_row.get("source_id", ""),
            "chunk_text"    : meta_row.get("chunk_text", ""),
            "token_estimate": meta_row.get("token_estimate", estimate_tokens(meta_row.get("chunk_text",""))),
            "metadata"      : meta_row.get("metadata", {}),
            "faiss_idx"     : faiss_idx,
            "similarity_score": round(float(score), 6),
        })

    # 4. Hard filter: similarity threshold
    before_threshold = len(candidates)
    candidates = [c for c in candidates if c["similarity_score"] >= similarity_threshold]
    print(f"  Similarity filter ({similarity_threshold}): {before_threshold} → {len(candidates)}")

    # 5. Hard filter: domain
    if domain_filter:
        before_domain = len(candidates)
        candidates = [c for c in candidates if c["domain"] == domain_filter]
        print(f"  Domain filter ({domain_filter}): {before_domain} → {len(candidates)}")

    # 6. Deduplicate by knowledge_id (keep highest-scoring chunk per record)
    seen_knowledge = {}
    for c in candidates:
        kid = c["knowledge_id"]
        if kid not in seen_knowledge or c["similarity_score"] > seen_knowledge[kid]["similarity_score"]:
            seen_knowledge[kid] = c
    candidates = sorted(seen_knowledge.values(), key=lambda x: x["similarity_score"], reverse=True)

    # 7. Token budget enforcement: greedily pick top chunks until budget exhausted
    selected       = []
    tokens_used    = 0
    for c in candidates:
        tok = int(c.get("token_estimate", estimate_tokens(c["chunk_text"])))
        if tokens_used + tok <= token_budget and len(selected) < final_k:
            selected.append(c)
            tokens_used += tok
        if len(selected) >= final_k:
            break

    # 8. Assign final rank
    for rank, c in enumerate(selected, start=1):
        c["rank"] = rank

    print(f"  Final evidence: {len(selected)} chunks | {tokens_used} tokens used of {token_budget} budget")
    return selected


# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Interactive test — run a sample query

SAMPLE_QUERIES = [
    "What are the most common incident categories on Line-A?",
    "How should I perform preventive maintenance on a hydraulic press?",
    "What is the machine start-up procedure?",
    "What happens when defect rate exceeds 0.5% in a batch?",
    "What was the OEE for Widget-Alpha production runs?",
]

print("── Sample retrieval test ────────────────────────────────────────\n")
for query in SAMPLE_QUERIES[:2]:   # run 2 samples to limit API calls
    print(f"Query: \"{query}\"")
    try:
        results = retrieve(query, top_k=RETRIEVAL_TOP_K, final_k=3,
                           similarity_threshold=SIMILARITY_THRESHOLD)
        for r in results:
            print(f"  [{r['rank']}] score={r['similarity_score']:.4f}  "
                  f"domain={r['domain']}  source={r['source_id']}")
            print(f"       text[:120]: {r['chunk_text'][:120].replace(chr(10),' ')}…")
        print()
    except Exception as e:
        print(f"  [SKIP] Could not run query (likely no API key in local mode): {e}\n")

print("Retrieval module ready. Import retrieve() in notebook 06.")
