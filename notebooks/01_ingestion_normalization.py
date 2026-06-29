# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingestion & Normalization
# MAGIC Loads all five domain CSVs from ADLS Gen2 (or local path), normalises them into
# MAGIC a single **Knowledge** table with a fixed schema, and writes to Delta with full lineage.
# MAGIC
# MAGIC **Fixed schema:**
# MAGIC | column | type | description |
# MAGIC |---|---|---|
# MAGIC | `knowledge_id` | STRING | deterministic UUID v5 from source_id + domain |
# MAGIC | `domain` | STRING | incidents / quality / maintenance / production / sop |
# MAGIC | `source_id` | STRING | original primary key from the source CSV |
# MAGIC | `title` | STRING | short human-readable label |
# MAGIC | `body` | STRING | rich text content used for chunking |
# MAGIC | `date` | DATE | event or document date |
# MAGIC | `metadata` | MAP<STRING,STRING> | arbitrary domain key-values |
# MAGIC | `ingested_at` | TIMESTAMP | pipeline run timestamp |
# MAGIC | `source_file` | STRING | originating file path for lineage |
# MAGIC | `source_hash` | STRING | MD5 of the raw row for idempotent reloads |

# COMMAND ----------

import os, hashlib, uuid, json
from datetime import datetime, date

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import *
    spark = SparkSession.builder.getOrCreate()
    DATABRICKS = True
except ImportError:
    DATABRICKS = False
    print("[INFO] PySpark not available — running in local pandas mode")
    import pandas as pd

# ── config ───────────────────────────────────────────────────────────────────
try:
    from config.config import (LOCAL_DATA_PATH, RAW_DATA_PATH, KNOWLEDGE_PATH,
                               KNOWLEDGE_TABLE, DB_NAME, DOMAINS)
    DATA_DIR    = RAW_DATA_PATH if DATABRICKS else LOCAL_DATA_PATH
    DELTA_PATH  = KNOWLEDGE_PATH
except ImportError:
    DATA_DIR   = "./data"
    DELTA_PATH = "./knowledge_delta"
    KNOWLEDGE_TABLE = "rag_platform.knowledge"
    DOMAINS    = ["incidents", "quality", "maintenance", "production", "sop"]

INGESTED_AT = datetime.utcnow()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Define the target schema

KNOWLEDGE_SCHEMA = StructType([
    StructField("knowledge_id", StringType(), False),
    StructField("domain",       StringType(), False),
    StructField("source_id",    StringType(), False),
    StructField("title",        StringType(), True),
    StructField("body",         StringType(), False),
    StructField("event_date",   DateType(),   True),
    StructField("metadata",     MapType(StringType(), StringType()), True),
    StructField("ingested_at",  TimestampType(), False),
    StructField("source_file",  StringType(), True),
    StructField("source_hash",  StringType(), False),
]) if DATABRICKS else None

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Normalizer functions — one per domain

def _hash(row_dict: dict) -> str:
    raw = json.dumps(row_dict, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()

def _knowledge_id(source_id: str, domain: str) -> str:
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(namespace, f"{domain}:{source_id}"))

def _safe_date(val):
    if not val:
        return None
    try:
        return datetime.strptime(str(val).strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def normalize_incidents(rows, source_file):
    out = []
    for r in rows:
        sid  = r.get("incident_id", "")
        body = (
            f"Incident {sid} — {r.get('category','')} ({r.get('severity','')}) "
            f"on {r.get('line','')}, {r.get('date','')}.\n"
            f"{r.get('description','')}\n"
            f"Resolution: {r.get('resolution','')}. "
            f"Root cause: {r.get('root_cause','')}. "
            f"Preventive action: {r.get('preventive_action','')}. "
            f"Duration: {r.get('duration_minutes','')} minutes. "
            f"Recurrences this year: {r.get('recurrence_count','')}."
        )
        meta = {
            "line"             : str(r.get("line", "")),
            "category"         : str(r.get("category", "")),
            "severity"         : str(r.get("severity", "")),
            "assigned_to"      : str(r.get("assigned_to", "")),
            "duration_minutes" : str(r.get("duration_minutes", "")),
            "resolution"       : str(r.get("resolution", "")),
            "recurrence_count" : str(r.get("recurrence_count", "")),
        }
        out.append({
            "knowledge_id": _knowledge_id(sid, "incidents"),
            "domain"      : "incidents",
            "source_id"   : sid,
            "title"       : f"Incident {sid}: {r.get('category','')} on {r.get('line','')}",
            "body"        : body,
            "event_date"  : _safe_date(r.get("date")),
            "metadata"    : meta,
            "ingested_at" : INGESTED_AT,
            "source_file" : source_file,
            "source_hash" : _hash(r),
        })
    return out


def normalize_quality(rows, source_file):
    out = []
    for r in rows:
        sid  = r.get("inspection_id", "")
        body = (
            f"Quality inspection {sid} for {r.get('product','')} "
            f"(batch {r.get('batch_id','')}) on {r.get('date','')}.\n"
            f"{r.get('description','')}\n"
            f"Total units: {r.get('total_units','')}. "
            f"Defective: {r.get('defective_units','')} ({r.get('defect_rate_pct','')}%). "
            f"Defect type: {r.get('defect_type','')}. "
            f"Disposition: {r.get('disposition','')}. "
            f"Cp: {r.get('cp','')}. Cpk: {r.get('cpk','')}. "
            f"Corrective action: {r.get('corrective_action','')}."
        )
        meta = {
            "product"         : str(r.get("product", "")),
            "batch_id"        : str(r.get("batch_id", "")),
            "inspector"       : str(r.get("inspector", "")),
            "defect_type"     : str(r.get("defect_type", "")),
            "defect_rate_pct" : str(r.get("defect_rate_pct", "")),
            "disposition"     : str(r.get("disposition", "")),
            "cp"              : str(r.get("cp", "")),
            "cpk"             : str(r.get("cpk", "")),
        }
        out.append({
            "knowledge_id": _knowledge_id(sid, "quality"),
            "domain"      : "quality",
            "source_id"   : sid,
            "title"       : f"QI {sid}: {r.get('defect_type','')} in {r.get('product','')}",
            "body"        : body,
            "event_date"  : _safe_date(r.get("date")),
            "metadata"    : meta,
            "ingested_at" : INGESTED_AT,
            "source_file" : source_file,
            "source_hash" : _hash(r),
        })
    return out


def normalize_maintenance(rows, source_file):
    out = []
    for r in rows:
        sid  = r.get("work_order", "")
        body = (
            f"Work order {sid} — {r.get('maintenance_type','')} maintenance on "
            f"{r.get('equipment','')} ({r.get('date','')}).\n"
            f"{r.get('description','')}\n"
            f"Technician: {r.get('technician','')}. "
            f"Duration: {r.get('duration_hours','')} hours. "
            f"Parts replaced: {r.get('parts_replaced','')}. "
            f"MTBF: {r.get('mtbf_hours','')} hours. "
            f"Next PM due: {r.get('next_pm_date','')}."
        )
        meta = {
            "equipment"       : str(r.get("equipment", "")),
            "maintenance_type": str(r.get("maintenance_type", "")),
            "technician"      : str(r.get("technician", "")),
            "duration_hours"  : str(r.get("duration_hours", "")),
            "parts_replaced"  : str(r.get("parts_replaced", "")),
            "mtbf_hours"      : str(r.get("mtbf_hours", "")),
            "next_pm_date"    : str(r.get("next_pm_date", "")),
        }
        out.append({
            "knowledge_id": _knowledge_id(sid, "maintenance"),
            "domain"      : "maintenance",
            "source_id"   : sid,
            "title"       : f"WO {sid}: {r.get('maintenance_type','')} on {r.get('equipment','')}",
            "body"        : body,
            "event_date"  : _safe_date(r.get("date")),
            "metadata"    : meta,
            "ingested_at" : INGESTED_AT,
            "source_file" : source_file,
            "source_hash" : _hash(r),
        })
    return out


def normalize_production(rows, source_file):
    out = []
    for r in rows:
        sid  = r.get("run_id", "")
        body = (
            f"Production run {sid} on {r.get('line','')} — {r.get('shift','')} shift, "
            f"{r.get('date','')}. Product: {r.get('product','')}.\n"
            f"{r.get('description','')}\n"
            f"Target: {r.get('target_units','')} units. "
            f"Actual: {r.get('actual_units','')} units. "
            f"Rejects: {r.get('reject_units','')} units. "
            f"OEE: {r.get('oee_pct','')}%. "
            f"Downtime: {r.get('downtime_minutes','')} min ({r.get('downtime_cause','')})."
        )
        meta = {
            "line"             : str(r.get("line", "")),
            "product"          : str(r.get("product", "")),
            "shift"            : str(r.get("shift", "")),
            "operator"         : str(r.get("operator", "")),
            "oee_pct"          : str(r.get("oee_pct", "")),
            "target_units"     : str(r.get("target_units", "")),
            "actual_units"     : str(r.get("actual_units", "")),
            "reject_units"     : str(r.get("reject_units", "")),
            "downtime_minutes" : str(r.get("downtime_minutes", "")),
            "downtime_cause"   : str(r.get("downtime_cause", "")),
        }
        out.append({
            "knowledge_id": _knowledge_id(sid, "production"),
            "domain"      : "production",
            "source_id"   : sid,
            "title"       : f"Run {sid}: {r.get('product','')} on {r.get('line','')} ({r.get('shift','')})",
            "body"        : body,
            "event_date"  : _safe_date(r.get("date")),
            "metadata"    : meta,
            "ingested_at" : INGESTED_AT,
            "source_file" : source_file,
            "source_hash" : _hash(r),
        })
    return out


def normalize_sop(rows, source_file):
    out = []
    for r in rows:
        sid  = r.get("sop_id", "")
        body = (
            f"Standard Operating Procedure: {r.get('title','')} [{sid}] "
            f"({r.get('revision','')}) — Domain: {r.get('domain','')}.\n\n"
            f"{r.get('content','')}"
        )
        meta = {
            "sop_id"  : str(r.get("sop_id", "")),
            "revision": str(r.get("revision", "")),
            "sub_domain": str(r.get("domain", "")),
        }
        out.append({
            "knowledge_id": _knowledge_id(sid, "sop"),
            "domain"      : "sop",
            "source_id"   : sid,
            "title"       : f"{sid}: {r.get('title','')}",
            "body"        : body,
            "event_date"  : None,
            "metadata"    : meta,
            "ingested_at" : INGESTED_AT,
            "source_file" : source_file,
            "source_hash" : _hash(r),
        })
    return out


NORMALIZERS = {
    "incidents"  : normalize_incidents,
    "quality"    : normalize_quality,
    "maintenance": normalize_maintenance,
    "production" : normalize_production,
    "sop"        : normalize_sop,
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load, normalize, and union all domains

import csv as _csv

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(_csv.DictReader(f))

all_records = []
load_summary = {}

for domain in DOMAINS:
    csv_path = os.path.join(DATA_DIR, f"{domain}.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found — run notebook 00 first")
        continue
    raw_rows = load_csv(csv_path)
    normalizer = NORMALIZERS[domain]
    norm_rows  = normalizer(raw_rows, csv_path)
    all_records.extend(norm_rows)
    load_summary[domain] = {"raw": len(raw_rows), "normalized": len(norm_rows)}
    print(f"  {domain:<15} raw={len(raw_rows):>4}  normalized={len(norm_rows):>4}")

print(f"\nTotal normalized records: {len(all_records)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Deduplication — skip already-loaded hashes (idempotent reload)

if DATABRICKS:
    # Try to load existing hashes from Delta
    try:
        existing_hashes = set(
            spark.table(KNOWLEDGE_TABLE)
                 .select("source_hash")
                 .rdd.flatMap(lambda r: [r[0]])
                 .collect()
        )
        print(f"Existing Delta hashes: {len(existing_hashes)}")
    except Exception:
        existing_hashes = set()
        print("Knowledge table does not exist yet — first load")

    new_records = [r for r in all_records if r["source_hash"] not in existing_hashes]
else:
    existing_hashes = set()
    new_records = all_records

print(f"New records to insert: {len(new_records)}  |  Skipped (already loaded): {len(all_records) - len(new_records)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Write to Delta (Databricks) or Parquet/CSV (local)

if DATABRICKS and new_records:
    # Convert event_date back to string for Spark (Spark handles the DateType cast)
    spark_rows = []
    for r in new_records:
        row = dict(r)
        row["event_date"]  = row["event_date"].isoformat() if row["event_date"] else None
        row["ingested_at"] = row["ingested_at"].isoformat()
        spark_rows.append(row)

    df = spark.createDataFrame(spark_rows, schema=KNOWLEDGE_SCHEMA)

    # Write as Delta (append for incremental loads)
    (df.write
       .format("delta")
       .mode("append")
       .option("mergeSchema", "true")
       .save(DELTA_PATH))

    # Register as table in Unity Catalog / Hive metastore
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {KNOWLEDGE_TABLE}
        USING DELTA LOCATION '{DELTA_PATH}'
    """)
    print(f"Written {len(new_records)} rows to Delta table: {KNOWLEDGE_TABLE}")

elif not DATABRICKS and new_records:
    # Local mode: write to JSON for inspection
    import json as _json
    out_path = os.path.join("./data", "knowledge.json")
    serial = []
    for r in new_records:
        row = dict(r)
        row["event_date"]  = row["event_date"].isoformat() if row["event_date"] else None
        row["ingested_at"] = row["ingested_at"].isoformat()
        serial.append(row)
    with open(out_path, "w") as f:
        _json.dump(serial, f, indent=2)
    print(f"Written {len(new_records)} rows to {out_path}")

else:
    print("No new records to write.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Lineage summary

print("\n── Ingestion summary ────────────────────────────────────────────")
print(f"{'Domain':<15} {'Raw':>6} {'Normalized':>12} {'New':>6}")
print("─" * 45)
for domain, stats in load_summary.items():
    existing_in_domain = sum(1 for r in all_records
                             if r["domain"] == domain
                             and r["source_hash"] in existing_hashes)
    new_in_domain = stats["normalized"] - existing_in_domain
    print(f"  {domain:<13} {stats['raw']:>6} {stats['normalized']:>12} {new_in_domain:>6}")
print("─" * 45)
print(f"  {'TOTAL':<13} {sum(v['raw'] for v in load_summary.values()):>6} "
      f"{sum(v['normalized'] for v in load_summary.values()):>12} {len(new_records):>6}")
print(f"\nIngestion run at: {INGESTED_AT.isoformat()}")
