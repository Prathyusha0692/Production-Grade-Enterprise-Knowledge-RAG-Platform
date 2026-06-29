# Production-Grade Enterprise Knowledge RAG Platform

A fully reproducible, end-to-end Retrieval-Augmented Generation (RAG) pipeline that transforms messy operational datasets into a strictly grounded question-answering system. Built for Azure Databricks on ADLS Gen2 with dual local/cloud execution.

---

## Architecture

```
Raw CSVs (ADLS Gen2)
        │
        ▼
┌─────────────────────┐
│  00  Synthetic Data │  Generate realistic operational datasets
│      Generator      │  incidents · quality · maintenance · production · SOP
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  01  Ingestion &    │  Load CSVs · Normalize to fixed schema
│      Normalization  │  Deterministic knowledge_id · MD5 lineage hash
└────────┬────────────┘    Idempotent reloads via source_hash dedup
         │
         ▼
┌─────────────────────┐
│  02  Chunking       │  Eligibility gates (min chars/words/alpha ratio)
│      Layer          │  SOP section splitting on headings
└────────┬────────────┘  Sliding-window with overlap · Token estimation
         │               Quality scoring · Rejection logging
         ▼
┌─────────────────────┐
│  03  Embedding      │  OpenAI REST via Databricks Secrets
│      Pipeline       │  Idempotent re-runs (skip already-embedded IDs)
└────────┬────────────┘  Deterministic batching · Exp-backoff retries
         │               Quarantine for corrupt vectors · Run metrics log
         ▼
┌─────────────────────┐
│  04  Index          │  Join integrity check · Dim validation
│      Builder        │  FAISS IndexFlatIP + L2-normalise (cosine)
└────────┬────────────┘  Versioned: index.faiss · id_map · metadata · manifest
         │               SHA-256 checksum · Domain breakdown
         ▼
┌─────────────────────┐
│  05  Retrieval      │  Embed query with same model
│                     │  FAISS top-K search · Similarity threshold filter
└────────┬────────────┘  Domain hard-filter · Token budget enforcement
         │               Dedup by knowledge_id · Ranked EvidenceSet
         ▼
┌─────────────────────┐
│  06  Generation &   │  Deterministic prompt with [SRC-N] anchors
│      Validation     │  OpenAI GPT-4o generation
└────────┬────────────┘  Post-gen validation: citation check · anchor resolution
         │               Sentence-level grounding · NO_EVIDENCE path
         │               Delta validation log · Hallucination blocking
         ▼
┌─────────────────────┐
│  07  Orchestrator   │  run_pipeline() · query() · batch_query()
│                     │  Single entry point for full rebuild or query-only
└─────────────────────┘
```

---

## Project Structure

```
├── config/
│   └── config.py              # All paths, model names, thresholds, constants
├── notebooks/
│   ├── 00_synthetic_data_generator.py
│   ├── 01_ingestion_normalization.py
│   ├── 02_chunking.py
│   ├── 03_embedding_pipeline.py
│   ├── 04_index_builder.py
│   ├── 05_retrieval.py
│   ├── 06_generation_validation.py
│   └── 07_end_to_end_pipeline.py
├── data/                      # Generated CSVs and intermediate JSON (gitignored)
├── index/                     # Versioned FAISS index artifacts (gitignored)
├── logs/                      # Validation logs (gitignored)
└── .gitignore
```

---

## Data Domains

| Domain | Records | Description |
|---|---|---|
| `incidents` | 200 | IT/OT incident records with severity, root cause, CAPA |
| `quality` | 200 | Product inspection results, defect rates, Cp/Cpk |
| `maintenance` | 200 | Equipment PM/CM work orders, MTBF, parts replaced |
| `production` | 200 | Production run summaries, OEE, downtime |
| `sop` | 30 | Standard operating procedures with section structure |

---

## Key Design Decisions

**Strict grounding** — Every factual sentence in a generated answer must carry a `[SRC-N]` citation anchor. Responses with uncited claims, invalid anchors, or missing citations are blocked before reaching the caller.

**Idempotent pipeline** — Each stage skips work already done: ingestion deduplicates on `source_hash`, embedding skips `chunk_id`s already in the embeddings table, and the index builder increments the version number rather than overwriting.

**Cosine similarity via IndexFlatIP** — Vectors are L2-normalised before insertion so inner product equals cosine similarity. Exact search (no approximation) is used for correctness; swap to `IndexIVFFlat` or `IndexHNSWFlat` for large-scale production.

**Quarantine** — Embedding batches that fail after all retries, or produce vectors with wrong dimension / NaN / Inf values, are written to a quarantine table rather than silently dropped. Re-runs skip quarantined chunks unless explicitly cleared.

**Versioned index** — Every `run_pipeline()` call produces a new `index/vN/` directory containing `index.faiss`, `id_map.json`, `metadata.json`, and `manifest.json` (with SHA-256 checksum). The retrieval layer defaults to the latest version but can be pinned to any prior version.

---

## Setup

### Prerequisites

```bash
pip install faiss-cpu numpy openai
# On Databricks, faiss-cpu and numpy are pre-installed
```

### Configuration

Edit `config/config.py` with your environment values:

```python
ADLS_ACCOUNT   = "yourstorageaccount"   # Azure storage account name
ADLS_CONTAINER = "rag-data"             # ADLS container
```

### Databricks Secrets

Store your OpenAI API key in Databricks Secrets:

```bash
databricks secrets create-scope --scope rag-secrets
databricks secrets put --scope rag-secrets --key openai-api-key
```

### Local execution (without Databricks)

```bash
export OPENAI_API_KEY=sk-...
python -c "exec(open('notebooks/00_synthetic_data_generator.py').read())"
python -c "exec(open('notebooks/01_ingestion_normalization.py').read())"
python -c "exec(open('notebooks/02_chunking.py').read())"
python -c "exec(open('notebooks/03_embedding_pipeline.py').read())"
python -c "exec(open('notebooks/04_index_builder.py').read())"
```

---

## Usage

### Full pipeline rebuild

```python
# In notebook 07 or any Python environment
from notebooks.07_end_to_end_pipeline import run_pipeline

summary = run_pipeline(rebuild=True, generate_data=True)
```

### Single query

```python
from notebooks.07_end_to_end_pipeline import query

response = query("What are the steps to start up a production line machine safely?")

print(response["answer"])
print(response["status"])        # "success" | "no_evidence" | "blocked" | "error"
print(response["citations"])     # [{"anchor": "[SRC-1]", "domain": "sop", ...}]
print(response["validation"])    # {"passed": True, "issues": []}
```

### Domain-filtered query

```python
response = query(
    "What preventive maintenance tasks are required for conveyors?",
    domain_filter="maintenance",
    similarity_threshold=0.72,
)
```

### Batch queries

```python
from notebooks.07_end_to_end_pipeline import batch_query

results = batch_query([
    "What is the escalation matrix for P1 incidents?",
    "Which products had the highest defect rates?",
    "What is the OEE target for Line-A?",
])
```

---

## PublicResponse Schema

Every query returns a `PublicResponse` dict:

```json
{
  "request_id"    : "uuid",
  "query"         : "user question",
  "answer"        : "grounded answer with [SRC-1] citations, or null if blocked",
  "status"        : "success | no_evidence | blocked | error",
  "citations"     : [
    {"anchor": "[SRC-1]", "chunk_id": "...", "domain": "sop", "source_id": "SOP-003"}
  ],
  "evidence_count": 3,
  "validation"    : {"passed": true, "issues": []},
  "elapsed_ms"    : 842
}
```

---

## Validation Rules

| Rule | Behaviour on failure |
|---|---|
| Response is empty | Blocked |
| `NO_EVIDENCE` returned but evidence exists | Blocked |
| Answer provided but no evidence was retrieved | `NO_EVIDENCE` status |
| Fewer than 1 `[SRC-N]` citation in answer | Blocked |
| `[SRC-N]` anchor not in evidence context | Blocked |
| Factual sentence has no citation | Blocked |

All results (pass and fail) are persisted to the `rag_platform.validation_log` Delta table.

---

## Technology Stack

| Component | Technology |
|---|---|
| Compute | Azure Databricks |
| Storage | ADLS Gen2 (abfss://) |
| Data processing | PySpark, Python |
| Table format | Delta Lake |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector index | FAISS `IndexFlatIP` |
| LLM | OpenAI `gpt-4o` |
| Secrets | Databricks Secret Scopes |
| API calls | Python `urllib` (no SDK dependency) |
