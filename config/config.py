# =============================================================================
# config/config.py  —  Central configuration for the RAG pipeline
# =============================================================================

import os

# ---------------------------------------------------------------------------
# ADLS / Storage paths
# ---------------------------------------------------------------------------
ADLS_ACCOUNT      = os.getenv("ADLS_ACCOUNT", "yourstorageaccount")
ADLS_CONTAINER    = os.getenv("ADLS_CONTAINER", "rag-data")
ADLS_BASE_PATH    = f"abfss://{ADLS_CONTAINER}@{ADLS_ACCOUNT}.dfs.core.windows.net"

RAW_DATA_PATH     = f"{ADLS_BASE_PATH}/raw"
KNOWLEDGE_PATH    = f"{ADLS_BASE_PATH}/knowledge"
CHUNKS_PATH       = f"{ADLS_BASE_PATH}/chunks"
EMBEDDINGS_PATH   = f"{ADLS_BASE_PATH}/embeddings"
INDEX_PATH        = f"{ADLS_BASE_PATH}/index"
LOGS_PATH         = f"{ADLS_BASE_PATH}/logs"

# Local fallback paths (used when running outside Databricks)
LOCAL_DATA_PATH   = "./data"
LOCAL_INDEX_PATH  = "./index"
LOCAL_LOGS_PATH   = "./logs"

# ---------------------------------------------------------------------------
# Delta table names
# ---------------------------------------------------------------------------
DB_NAME                 = "rag_platform"
KNOWLEDGE_TABLE         = f"{DB_NAME}.knowledge"
CHUNKS_TABLE            = f"{DB_NAME}.chunks"
EMBEDDINGS_TABLE        = f"{DB_NAME}.embeddings"
INDEX_MANIFEST_TABLE    = f"{DB_NAME}.index_manifest"
VALIDATION_LOG_TABLE    = f"{DB_NAME}.validation_log"
EMBEDDING_METRICS_TABLE = f"{DB_NAME}.embedding_metrics"
QUARANTINE_TABLE        = f"{DB_NAME}.quarantine"

# ---------------------------------------------------------------------------
# Databricks Secret scope / keys
# ---------------------------------------------------------------------------
SECRET_SCOPE        = "rag-secrets"
OPENAI_API_KEY_NAME = "openai-api-key"
EMBEDDING_KEY_NAME  = "embedding-api-key"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL         = "text-embedding-3-small"   # or text-embedding-ada-002
EMBEDDING_ENDPOINT      = "https://api.openai.com/v1/embeddings"
EMBEDDING_DIM           = 1536
EMBEDDING_BATCH_SIZE    = 64
EMBEDDING_MAX_RETRIES   = 3
EMBEDDING_RETRY_BACKOFF = 2.0   # seconds, exponential base

# ---------------------------------------------------------------------------
# LLM (generation)
# ---------------------------------------------------------------------------
LLM_MODEL               = "gpt-4o"
LLM_ENDPOINT            = "https://api.openai.com/v1/chat/completions"
LLM_TEMPERATURE         = 0.0
LLM_MAX_TOKENS          = 1024
LLM_TIMEOUT_SECONDS     = 60

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_MAX_TOKENS        = 400
CHUNK_OVERLAP_TOKENS    = 40
CHUNK_MIN_CHARS         = 80     # eligibility gate: reject shorter text
CHUNK_MIN_WORDS         = 10     # eligibility gate: reject sparse text
SOP_SECTION_PATTERN     = r"(?m)^#{1,3}\s+|^\d+\.\s+"  # markdown / numbered headings

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K             = 20    # FAISS candidates before filtering
RETRIEVAL_FINAL_K           = 5     # chunks returned after filters
SIMILARITY_THRESHOLD        = 0.70  # cosine similarity floor (post-normalisation)
EVIDENCE_TOKEN_BUDGET       = 2000  # max tokens of evidence passed to LLM
HARD_FILTER_DOMAIN_KEY      = "domain"   # metadata field for domain filter

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
MIN_CITATIONS_REQUIRED      = 1
CITATION_PATTERN            = r"\[SRC-\d+\]"   # e.g. [SRC-1], [SRC-2]
HALLUCINATION_BLOCK         = True   # fail responses with uncited factual claims
NO_EVIDENCE_PHRASE          = "NO_EVIDENCE"

# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------
DOMAINS = ["incidents", "quality", "maintenance", "production", "sop"]

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
INDEX_VERSION_PREFIX = "v"   # versions become v1, v2, …
