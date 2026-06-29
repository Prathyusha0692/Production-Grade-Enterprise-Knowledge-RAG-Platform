# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Generation & Validation
# MAGIC Assembles a grounded prompt from retrieved evidence, calls the LLM, validates the
# MAGIC response for hallucinations, and returns a clean `PublicResponse` JSON.
# MAGIC
# MAGIC **Validation rules (post-generation):**
# MAGIC - Every factual claim sentence must contain at least one citation anchor `[SRC-N]`
# MAGIC - Every cited anchor `[SRC-N]` must resolve to an evidence chunk in the context
# MAGIC - If no evidence was retrieved, the LLM must respond with exactly `NO_EVIDENCE`
# MAGIC - Responses failing validation are **blocked** — never returned to the caller
# MAGIC - All results (pass and fail) are persisted to the `validation_log` Delta table

# COMMAND ----------

import os, re, json, uuid
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
        LLM_MODEL, LLM_ENDPOINT, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS,
        SECRET_SCOPE, OPENAI_API_KEY_NAME,
        CITATION_PATTERN, MIN_CITATIONS_REQUIRED, HALLUCINATION_BLOCK, NO_EVIDENCE_PHRASE,
        VALIDATION_LOG_TABLE, LOGS_PATH, DB_NAME, LOCAL_DATA_PATH
    )
except ImportError:
    LLM_MODEL              = "gpt-4o"
    LLM_ENDPOINT           = "https://api.openai.com/v1/chat/completions"
    LLM_TEMPERATURE        = 0.0
    LLM_MAX_TOKENS         = 1024
    LLM_TIMEOUT_SECONDS    = 60
    SECRET_SCOPE           = "rag-secrets"
    OPENAI_API_KEY_NAME    = "openai-api-key"
    CITATION_PATTERN       = r"\[SRC-\d+\]"
    MIN_CITATIONS_REQUIRED = 1
    HALLUCINATION_BLOCK    = True
    NO_EVIDENCE_PHRASE     = "NO_EVIDENCE"
    VALIDATION_LOG_TABLE   = "rag_platform.validation_log"
    LOGS_PATH              = "./logs"
    DB_NAME                = "rag_platform"
    LOCAL_DATA_PATH        = "./data"

os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. API key and LLM caller

import urllib.request, urllib.error

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


def call_llm(messages: list[dict], model: str, temperature: float,
             max_tokens: int, timeout: int) -> str:
    """Calls the OpenAI chat completions endpoint. Returns the assistant message text."""
    payload = json.dumps({
        "model"      : model,
        "messages"   : messages,
        "temperature": temperature,
        "max_tokens" : max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_API_KEY}"}
    req     = urllib.request.Request(LLM_ENDPOINT, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API error HTTP {e.code}: {body}")


# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Prompt assembly (deterministic, stable citation anchors)

SYSTEM_PROMPT = """You are an enterprise knowledge assistant with strict grounding rules.

RULES:
1. Answer ONLY based on the provided EVIDENCE passages. Do not use outside knowledge.
2. Every factual claim you make MUST be followed immediately by a citation in the form [SRC-N]
   where N is the evidence source number.
3. You may cite multiple sources: "...as described [SRC-1][SRC-3]."
4. If the evidence does not contain enough information to answer the question, respond with
   exactly: NO_EVIDENCE
5. Do not invent details, infer beyond the evidence, or speculate.
6. Be concise. Prefer specific facts with citations over vague generalisations.
"""


def build_prompt(query: str, evidence: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """
    Builds the messages list for the LLM and returns a citation anchor map.

    Returns:
        messages      : list of {role, content} dicts
        anchor_map    : {"SRC-1": chunk_id, "SRC-2": chunk_id, ...}
    """
    if not evidence:
        # No-evidence path
        evidence_block = "(No relevant evidence found in the knowledge base.)"
        anchor_map     = {}
    else:
        lines      = []
        anchor_map = {}
        for i, ev in enumerate(evidence, start=1):
            anchor = f"SRC-{i}"
            anchor_map[anchor] = ev["chunk_id"]
            domain   = ev.get("domain", "unknown")
            src_id   = ev.get("source_id", "")
            sim      = ev.get("similarity_score", 0.0)
            text     = ev.get("chunk_text", "").strip()
            lines.append(
                f"[{anchor}] (domain={domain}, source={src_id}, similarity={sim:.4f})\n{text}"
            )
        evidence_block = "\n\n---\n\n".join(lines)

    user_content = (
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"QUESTION: {query}"
    )

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]
    return messages, anchor_map


# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Post-generation validator

def _sentence_split(text: str) -> list[str]:
    """Splits text into sentences on . ! ? without destroying citations like [SRC-1]."""
    # Temporarily replace citation brackets so they don't confuse the splitter
    protected = re.sub(r'\[SRC-\d+\]', lambda m: m.group().replace("]", "§"), text)
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    # Restore
    return [s.replace("§", "]") for s in sentences if s.strip()]


def validate_response(
    answer: str,
    anchor_map: dict[str, str],
    has_evidence: bool,
) -> tuple[bool, list[str]]:
    """
    Validates the LLM response against grounding rules.

    Returns:
        (passed: bool, issues: list[str])
    """
    issues = []

    # Rule 0: NO_EVIDENCE path
    if not has_evidence:
        if answer.strip() == NO_EVIDENCE_PHRASE:
            return True, []
        else:
            issues.append(
                f"no_evidence_violation: expected '{NO_EVIDENCE_PHRASE}' "
                f"but got non-empty answer"
            )
            return False, issues

    # Rule 1: Response must not be empty
    if not answer.strip():
        issues.append("empty_response")
        return False, issues

    # Rule 2: Response must not be NO_EVIDENCE when evidence was available
    if answer.strip() == NO_EVIDENCE_PHRASE:
        issues.append("spurious_no_evidence: answer is NO_EVIDENCE but evidence was provided")
        return False, issues

    # Rule 3: Extract all anchor references used in the answer
    used_anchors = set(re.findall(CITATION_PATTERN, answer))   # e.g. {"[SRC-1]", "[SRC-3]"}

    # Rule 4: Minimum citation count
    if len(used_anchors) < MIN_CITATIONS_REQUIRED:
        issues.append(
            f"insufficient_citations: found {len(used_anchors)}, "
            f"required ≥ {MIN_CITATIONS_REQUIRED}"
        )

    # Rule 5: Every cited anchor must resolve to a known chunk
    valid_anchors = {f"[{k}]" for k in anchor_map}
    invalid_anchors = used_anchors - valid_anchors
    if invalid_anchors:
        issues.append(f"invalid_anchor_refs: {sorted(invalid_anchors)}")

    # Rule 6: Sentence-level grounding — every sentence with a factual claim
    #         must carry at least one citation.
    sentences = _sentence_split(answer)
    FACTUAL_VERBS = re.compile(
        r'\b(is|are|was|were|has|have|had|showed|reported|confirmed|indicates?|'
        r'found|occurred|resulted|caused|prevented|reduced|increased|achieved)\b',
        re.IGNORECASE
    )
    uncited_factual = []
    for sent in sentences:
        has_citation = bool(re.search(CITATION_PATTERN, sent))
        has_factual  = bool(FACTUAL_VERBS.search(sent))
        if has_factual and not has_citation:
            uncited_factual.append(sent.strip()[:120])

    if uncited_factual:
        issues.append(
            f"uncited_factual_sentences ({len(uncited_factual)}): "
            + " | ".join(uncited_factual[:3])
        )

    passed = len(issues) == 0
    return passed, issues


# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Persistence — write validation log

def _log_validation(request_id: str, query: str, answer: str, evidence: list[dict],
                    passed: bool, issues: list[str], elapsed: float):
    log_row = {
        "request_id"    : request_id,
        "logged_at"     : datetime.utcnow().isoformat(),
        "query"         : query,
        "answer"        : answer,
        "evidence_count": len(evidence),
        "chunk_ids"     : json.dumps([e["chunk_id"] for e in evidence]),
        "domains"       : json.dumps(list({e["domain"] for e in evidence})),
        "passed"        : str(passed),
        "issues"        : json.dumps(issues),
        "elapsed_ms"    : str(round(elapsed * 1000)),
        "llm_model"     : LLM_MODEL,
    }

    if DATABRICKS:
        from pyspark.sql.types import StructType, StructField, StringType
        schema = StructType([StructField(k, StringType(), True) for k in log_row])
        df     = spark.createDataFrame([log_row], schema=schema)
        lpath  = LOGS_PATH if LOGS_PATH.startswith("abfss://") else "./logs_delta"
        (df.write.format("delta").mode("append").save(lpath))
        try:
            spark.sql(f"CREATE TABLE IF NOT EXISTS {VALIDATION_LOG_TABLE} USING DELTA LOCATION '{lpath}'")
        except Exception:
            pass
    else:
        log_file = os.path.join(LOCAL_DATA_PATH, "validation_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_row) + "\n")


# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Main `answer_query` function

import time as _time

def answer_query(
    query: str,
    evidence: list[dict],
    model: str              = LLM_MODEL,
    temperature: float      = LLM_TEMPERATURE,
    max_tokens: int         = LLM_MAX_TOKENS,
    timeout: int            = LLM_TIMEOUT_SECONDS,
    block_on_failure: bool  = HALLUCINATION_BLOCK,
) -> dict:
    """
    Full generation + validation pipeline for one query.

    Returns a PublicResponse dict:
    {
        "request_id"    : str,
        "query"         : str,
        "answer"        : str | None,
        "status"        : "success" | "no_evidence" | "blocked" | "error",
        "citations"     : [{"anchor": "SRC-1", "chunk_id": ..., "domain": ..., "source_id": ...}],
        "evidence_count": int,
        "validation"    : {"passed": bool, "issues": [str]},
        "elapsed_ms"    : int,
    }
    """
    request_id = str(uuid.uuid4())
    t0         = _time.time()

    has_evidence = bool(evidence)

    try:
        # 1. Build prompt
        messages, anchor_map = build_prompt(query, evidence)

        # 2. Generate
        raw_answer = call_llm(messages, model, temperature, max_tokens, timeout)

        # 3. Validate
        passed, issues = validate_response(raw_answer, anchor_map, has_evidence)

        elapsed = _time.time() - t0

        # 4. Persist log (always, pass or fail)
        _log_validation(request_id, query, raw_answer, evidence, passed, issues, elapsed)

        # 5. Build citation list
        used_anchors = re.findall(CITATION_PATTERN, raw_answer)
        citations    = []
        seen         = set()
        for anchor_str in used_anchors:
            anchor_key = anchor_str.strip("[]")   # "SRC-1"
            if anchor_key in anchor_map and anchor_key not in seen:
                seen.add(anchor_key)
                chunk_id  = anchor_map[anchor_key]
                ev_match  = next((e for e in evidence if e["chunk_id"] == chunk_id), {})
                citations.append({
                    "anchor"   : anchor_str,
                    "chunk_id" : chunk_id,
                    "domain"   : ev_match.get("domain", ""),
                    "source_id": ev_match.get("source_id", ""),
                })

        # 6. Determine status and final answer
        if not has_evidence or raw_answer.strip() == NO_EVIDENCE_PHRASE:
            status        = "no_evidence"
            public_answer = (
                "I was unable to find relevant information in the knowledge base "
                "to answer your question."
            )
        elif passed:
            status        = "success"
            public_answer = raw_answer
        else:
            status        = "blocked"
            if block_on_failure:
                public_answer = None   # blocked — do not expose unvalidated answer
            else:
                public_answer = raw_answer   # non-blocking mode (dev/debug)

        return {
            "request_id"    : request_id,
            "query"         : query,
            "answer"        : public_answer,
            "status"        : status,
            "citations"     : citations,
            "evidence_count": len(evidence),
            "validation"    : {"passed": passed, "issues": issues},
            "elapsed_ms"    : int((elapsed) * 1000),
        }

    except Exception as e:
        elapsed = _time.time() - t0
        _log_validation(request_id, query, "", evidence, False,
                        [f"exception:{str(e)[:300]}"], elapsed)
        return {
            "request_id"    : request_id,
            "query"         : query,
            "answer"        : None,
            "status"        : "error",
            "citations"     : [],
            "evidence_count": len(evidence),
            "validation"    : {"passed": False, "issues": [str(e)]},
            "elapsed_ms"    : int(elapsed * 1000),
        }


# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Interactive test

print("── Generation & Validation test ─────────────────────────────────\n")

# Mock evidence (used when no FAISS index is available locally)
MOCK_EVIDENCE = [
    {
        "chunk_id"       : "mock-chunk-001",
        "knowledge_id"   : "mock-kn-001",
        "domain"         : "maintenance",
        "source_id"      : "WO-123456",
        "chunk_text"     : (
            "Work order WO-123456 — Preventive maintenance on Hydraulic-Press-H3 (2024-03-15). "
            "Technician Raj Iyer replaced bearing SKF-6205 and seal kit SK-200. "
            "Duration: 3.5 hours. MTBF updated: 4200 hours. Next PM due: 2024-06-15."
        ),
        "token_estimate" : 65,
        "metadata"       : {"equipment": "Hydraulic-Press-H3", "maintenance_type": "Preventive"},
        "similarity_score": 0.87,
        "rank"           : 1,
    },
    {
        "chunk_id"       : "mock-chunk-002",
        "knowledge_id"   : "mock-kn-002",
        "domain"         : "sop",
        "source_id"      : "SOP-003",
        "chunk_text"     : (
            "# 4. Execution Steps\n"
            "Technician receives work order on mobile device. "
            "Obtain permit-to-work from Shift Supervisor before isolating equipment. "
            "Apply LOTO (Lockout/Tagout) per LOTO procedure HS-007. "
            "Perform tasks listed on work order; record actual condition of each component. "
            "Replace parts if condition is below acceptance threshold (see Part Specs TS-012). "
            "Remove LOTO; perform functional test before returning to production. "
            "Close work order in CMMS with all findings recorded."
        ),
        "token_estimate" : 90,
        "metadata"       : {"sop_id": "SOP-003", "revision": "Rev-3"},
        "similarity_score": 0.82,
        "rank"           : 2,
    },
]

test_query = "What maintenance was performed on the hydraulic press and what does the SOP say about the steps?"

try:
    result = answer_query(
        query    = test_query,
        evidence = MOCK_EVIDENCE,
    )
    print(f"Status       : {result['status']}")
    print(f"Validation   : passed={result['validation']['passed']}")
    if result['validation']['issues']:
        print(f"Issues       : {result['validation']['issues']}")
    print(f"Elapsed      : {result['elapsed_ms']} ms")
    print(f"Citations    : {[c['anchor'] for c in result['citations']]}")
    print(f"\nAnswer:\n{result['answer']}")
except Exception as e:
    print(f"[NOTE] LLM call skipped (no API key in local env): {e}")
    print("Generation module is ready — set OPENAI_API_KEY to run live queries.")

print("\n[OK] Notebook 06 loaded. Call answer_query(query, evidence) from notebook 07.")
