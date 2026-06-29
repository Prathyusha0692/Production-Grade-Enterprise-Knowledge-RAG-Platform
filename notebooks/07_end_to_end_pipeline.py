# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — End-to-End Pipeline Orchestrator
# MAGIC Single entry point that wires together all pipeline stages:
# MAGIC
# MAGIC ```
# MAGIC  [00] Synthetic data  →  [01] Ingest & Normalize  →  [02] Chunk
# MAGIC       ↓
# MAGIC  [03] Embed  →  [04] Build FAISS Index
# MAGIC       ↓
# MAGIC  [05] Retrieve  →  [06] Generate + Validate  →  PublicResponse
# MAGIC ```
# MAGIC
# MAGIC **Usage modes:**
# MAGIC - `run_pipeline(rebuild=True)` — full rebuild: data → index
# MAGIC - `run_pipeline(rebuild=False)` — query only against existing index
# MAGIC - `query(question)` — single query, returns PublicResponse

# COMMAND ----------

import os, sys, json, time
from datetime import datetime

# Ensure notebooks/ and project root are on the path (for local execution)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) \
        if "__file__" in dir() else os.path.abspath(".")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config.config import (
        LOCAL_DATA_PATH, INDEX_PATH, INDEX_VERSION_PREFIX,
        RETRIEVAL_TOP_K, RETRIEVAL_FINAL_K, SIMILARITY_THRESHOLD,
        EVIDENCE_TOKEN_BUDGET, LLM_MODEL
    )
except ImportError:
    LOCAL_DATA_PATH      = "./data"
    INDEX_PATH           = "./index"
    INDEX_VERSION_PREFIX = "v"
    RETRIEVAL_TOP_K      = 20
    RETRIEVAL_FINAL_K    = 5
    SIMILARITY_THRESHOLD = 0.70
    EVIDENCE_TOKEN_BUDGET = 2000
    LLM_MODEL            = "gpt-4o"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Stage runner helpers

def _section(title: str):
    bar = "─" * 65
    print(f"\n{bar}\n  {title}\n{bar}")

def _run_notebook(nb_name: str, globals_dict: dict):
    """
    Executes a notebook .py file in the current interpreter context.
    In Databricks, replace this with dbutils.notebook.run(nb_name, timeout, args).
    """
    nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__))
                           if "__file__" in dir() else ".", nb_name)
    if not os.path.exists(nb_path):
        # Try relative to notebooks/
        nb_path = os.path.join("notebooks", nb_name)
    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Notebook not found: {nb_name}")
    with open(nb_path) as f:
        source = f.read()
    # Strip Databricks magic lines
    lines  = [l for l in source.splitlines()
              if not l.startswith("# MAGIC") and not l.startswith("# COMMAND")]
    exec(compile("\n".join(lines), nb_path, "exec"), globals_dict)


# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Full pipeline rebuild

def run_pipeline(
    rebuild: bool        = True,
    generate_data: bool  = True,
    skip_stages: list    = None,
) -> dict:
    """
    Runs the full ingestion → index build pipeline.

    Args:
        rebuild      : If True, re-run data gen → ingest → chunk → embed → index.
        generate_data: If True (and rebuild=True), regenerate synthetic CSVs first.
        skip_stages  : List of stage names to skip, e.g. ["00", "03"]

    Returns:
        Summary dict with stage timings and artefact paths.
    """
    skip   = set(skip_stages or [])
    summary = {"started_at": datetime.utcnow().isoformat(), "stages": {}}
    t_total = time.time()

    if not rebuild:
        print("rebuild=False — skipping data pipeline. Call query() directly.")
        return summary

    stages = [
        ("00", "00_synthetic_data_generator.py", "Data generation"),
        ("01", "01_ingestion_normalization.py",  "Ingestion & normalisation"),
        ("02", "02_chunking.py",                 "Chunking"),
        ("03", "03_embedding_pipeline.py",        "Embedding"),
        ("04", "04_index_builder.py",             "Index build"),
    ]

    if not generate_data:
        stages = [s for s in stages if s[0] != "00"]

    for stage_id, nb_file, label in stages:
        if stage_id in skip:
            print(f"[SKIP] {label} (stage {stage_id})")
            continue

        _section(f"Stage {stage_id}: {label}")
        t0 = time.time()
        try:
            _run_notebook(nb_file, globals().copy())
            elapsed = time.time() - t0
            summary["stages"][stage_id] = {"status": "ok", "elapsed_s": round(elapsed, 1)}
            print(f"\n[OK] {label} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            summary["stages"][stage_id] = {"status": "error", "error": str(e), "elapsed_s": round(elapsed, 1)}
            print(f"\n[ERROR] Stage {stage_id} failed: {e}")
            raise   # halt pipeline on failure

    summary["total_elapsed_s"] = round(time.time() - t_total, 1)
    summary["completed_at"]    = datetime.utcnow().isoformat()

    _section("Pipeline complete")
    for sid, info in summary["stages"].items():
        status  = info["status"]
        elapsed = info["elapsed_s"]
        label   = next((s[2] for s in stages if s[0] == sid), sid)
        print(f"  [{sid}] {label:<35} {status:<6}  {elapsed:.1f}s")
    print(f"\n  Total elapsed: {summary['total_elapsed_s']}s")

    return summary


# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Query function — single Q&A call

def query(
    question: str,
    domain_filter: Optional[str]    = None,
    top_k: int                      = RETRIEVAL_TOP_K,
    final_k: int                    = RETRIEVAL_FINAL_K,
    similarity_threshold: float     = SIMILARITY_THRESHOLD,
    token_budget: int               = EVIDENCE_TOKEN_BUDGET,
    index_version: Optional[str]    = None,
    verbose: bool                   = True,
) -> dict:
    """
    End-to-end Q&A: retrieves evidence then generates a grounded answer.

    Returns PublicResponse JSON dict.
    """
    from typing import Optional as _Opt   # noqa: re-import for exec scope

    # ── Import retrieval ──────────────────────────────────────────────────
    try:
        from notebooks.retrieval_module import retrieve, load_index, INDEX_STATE
        index_state = load_index(index_version) if index_version else INDEX_STATE
    except ImportError:
        # Inline import when running inside the notebook directly
        _ret_path = os.path.join("notebooks", "05_retrieval.py")
        _ctx = {}
        if os.path.exists(_ret_path):
            with open(_ret_path) as _f:
                _src = "\n".join(l for l in _f.read().splitlines()
                                 if not l.startswith("# MAGIC") and not l.startswith("# COMMAND"))
            exec(compile(_src, _ret_path, "exec"), _ctx)
        retrieve    = _ctx.get("retrieve")
        load_index  = _ctx.get("load_index")
        INDEX_STATE_LOCAL = _ctx.get("INDEX_STATE")
        index_state = load_index(index_version) if (load_index and index_version) else INDEX_STATE_LOCAL

    # ── Import generation ─────────────────────────────────────────────────
    try:
        from notebooks.generation_module import answer_query
    except ImportError:
        _gen_path = os.path.join("notebooks", "06_generation_validation.py")
        _ctx2 = {}
        if os.path.exists(_gen_path):
            with open(_gen_path) as _f:
                _src2 = "\n".join(l for l in _f.read().splitlines()
                                  if not l.startswith("# MAGIC") and not l.startswith("# COMMAND"))
            exec(compile(_src2, _gen_path, "exec"), _ctx2)
        answer_query = _ctx2.get("answer_query")

    if retrieve is None or answer_query is None:
        raise RuntimeError("Could not load retrieve() or answer_query(). "
                           "Ensure notebooks 05 and 06 are available.")

    if verbose:
        print(f'\nQuery: "{question}"')
        if domain_filter:
            print(f"Domain filter: {domain_filter}")

    # ── Retrieve ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        evidence = retrieve(
            query               = question,
            top_k               = top_k,
            final_k             = final_k,
            similarity_threshold= similarity_threshold,
            token_budget        = token_budget,
            domain_filter       = domain_filter,
            index_state         = index_state,
        )
    except Exception as e:
        if verbose:
            print(f"[WARN] Retrieval failed: {e}. Proceeding with empty evidence.")
        evidence = []

    # ── Generate ──────────────────────────────────────────────────────────
    response = answer_query(question, evidence)
    response["retrieval_elapsed_ms"] = int((time.time() - t0) * 1000 - response.get("elapsed_ms", 0))

    if verbose:
        print(f"\nStatus   : {response['status']}")
        print(f"Validated: {response['validation']['passed']}")
        print(f"Evidence : {response['evidence_count']} chunks  |  "
              f"Citations: {[c['anchor'] for c in response['citations']]}")
        print(f"Elapsed  : {response['elapsed_ms']}ms")
        if response["answer"]:
            print(f"\n{'─'*60}\n{response['answer']}\n{'─'*60}")
        elif response["status"] == "blocked":
            print("\n[BLOCKED] Response failed grounding validation.")
            if response["validation"]["issues"]:
                for issue in response["validation"]["issues"]:
                    print(f"  • {issue}")
        elif response["status"] == "no_evidence":
            print("\n[NO EVIDENCE] The knowledge base does not contain relevant information.")

    return response


# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Batch query runner

def batch_query(questions: list[str], **kwargs) -> list[dict]:
    """Run multiple questions and return all PublicResponse dicts."""
    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] ", end="")
        results.append(query(q, **kwargs))
    return results


# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Demo run

# ── Uncomment the block you want to run ──────────────────────────────────────

# FULL REBUILD (first-time setup):
# summary = run_pipeline(rebuild=True, generate_data=True)

# QUERY ONLY (index already built):
DEMO_QUESTIONS = [
    "What are the steps to start up a production line machine safely?",
    "Which incidents were classified as P1-Critical and what were their root causes?",
    "What is the corrective action when Cpk falls below 1.0?",
    "How often should bearings on conveyors be replaced?",
    "What was the OEE for Line-B during the night shift?",
]

print("── End-to-End RAG Pipeline — Demo ──────────────────────────────")
print(f"  Model      : {LLM_MODEL}")
print(f"  Index path : {INDEX_PATH}")
print(f"  Data path  : {LOCAL_DATA_PATH}")
print("\nTo run a query:\n  response = query('your question here')")
print("\nTo rebuild the full pipeline:\n  summary = run_pipeline(rebuild=True)")
print("\nDemo questions loaded in DEMO_QUESTIONS list.")
print("\nExample single query (requires OPENAI_API_KEY + built index):")
print("  response = query(DEMO_QUESTIONS[0])")
print("\n[Ready]")
