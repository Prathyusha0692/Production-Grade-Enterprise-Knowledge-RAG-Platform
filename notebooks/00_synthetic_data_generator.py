# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Synthetic Data Generator
# MAGIC Generates realistic CSV files for all five operational domains:
# MAGIC - **incidents** — IT/OT incident records
# MAGIC - **quality** — product quality inspection results
# MAGIC - **maintenance** — equipment maintenance logs
# MAGIC - **production** — production run summaries
# MAGIC - **sop** — standard operating procedure documents
# MAGIC
# MAGIC Output path: `./data/` (or ADLS raw path in Databricks)

# COMMAND ----------

import random
import csv
import os
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    _spark = SparkSession.builder.getOrCreate()
    _IN_DATABRICKS = True
except Exception:
    _IN_DATABRICKS = False

try:
    from config.config import LOCAL_DATA_PATH, RAW_DATA_PATH
    OUTPUT_DIR = RAW_DATA_PATH if _IN_DATABRICKS else LOCAL_DATA_PATH
except Exception:
    OUTPUT_DIR = "./data"               # local fallback

N_INCIDENTS   = 200
N_QUALITY     = 200
N_MAINTENANCE = 200
N_PRODUCTION  = 200
N_SOP         = 30

random.seed(42)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def random_date(start="2023-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    return (s + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")

def random_id(prefix, n=6):
    return f"{prefix}-{random.randint(10**(n-1), 10**n - 1)}"

# ── 1. INCIDENTS ─────────────────────────────────────────────────────────────

INCIDENT_CATEGORIES = ["Network", "Power", "Software", "Hardware", "Security", "Process"]
SEVERITY_LEVELS     = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]
RESOLUTION_CODES    = ["Root cause fixed", "Workaround applied", "No action needed",
                       "Escalated to vendor", "Monitoring extended"]
LINES               = ["Line-A", "Line-B", "Line-C", "Line-D"]
TECHNICIANS         = ["Alice Chen", "Bob Mehra", "Carlos Ruiz", "Diana Patel",
                       "Edward Kim", "Fatima Al-Hassan"]

INCIDENT_TEMPLATES = [
    "Unexpected shutdown observed on {line}. Operator reported loss of communication with PLC unit. "
    "Investigation revealed {cause}. Technician {tech} restored service after {duration} minutes. "
    "Root cause: {rc}. Preventive action: {action}.",

    "Alarm triggered on {line} due to {cause}. Severity assessed as {severity}. "
    "Response team dispatched at T+{response}min. Issue resolved by {tech}. "
    "System validated and returned to normal operation. Lessons learned documented.",

    "Production halted on {line}. {cause} caused cascading failures in downstream units. "
    "Containment: {action}. Full recovery achieved at {duration} minutes after detection. "
    "No product loss reported. Incident categorised as {category}.",

    "Recurring fault detected on {line}. Third occurrence this quarter. "
    "{tech} performed deep-dive analysis. {cause} identified as systemic issue. "
    "Long-term fix: {action}. Scheduled for next maintenance window.",
]

causes     = ["coolant pressure drop", "firmware mismatch", "sensor calibration drift",
              "network packet loss", "thermal runaway in drive unit", "PLC watchdog timeout",
              "unplanned power fluctuation", "lubricant contamination", "dust ingress on PCB"]
actions    = ["Updated firmware to v3.4.1", "Replaced faulty sensor module",
              "Implemented redundant network path", "Installed surge protector",
              "Scheduled bi-weekly lubrication cycle", "Adjusted PLC timeout threshold"]
root_causes = ["Manufacturing defect in sensor batch", "Configuration drift post-update",
               "Inadequate preventive maintenance interval", "Environmental factor (humidity)",
               "Vendor firmware bug — ticket raised"]

def gen_incidents(n):
    rows = []
    for i in range(n):
        line     = random.choice(LINES)
        tech     = random.choice(TECHNICIANS)
        severity = random.choice(SEVERITY_LEVELS)
        cat      = random.choice(INCIDENT_CATEGORIES)
        cause    = random.choice(causes)
        action   = random.choice(actions)
        rc       = random.choice(root_causes)
        duration = random.randint(15, 480)
        response = random.randint(5, 60)
        tpl      = random.choice(INCIDENT_TEMPLATES)
        desc     = tpl.format(line=line, cause=cause, tech=tech, duration=duration,
                              rc=rc, action=action, severity=severity,
                              response=response, category=cat)
        rows.append({
            "incident_id"      : random_id("INC"),
            "date"             : random_date(),
            "line"             : line,
            "category"         : cat,
            "severity"         : severity,
            "assigned_to"      : tech,
            "duration_minutes" : duration,
            "description"      : desc,
            "resolution"       : random.choice(RESOLUTION_CODES),
            "root_cause"       : rc,
            "preventive_action": action,
            "recurrence_count" : random.randint(0, 5),
        })
    return rows

# ── 2. QUALITY ───────────────────────────────────────────────────────────────

PRODUCTS        = ["Widget-Alpha", "Widget-Beta", "Valve-X200", "Pump-G10", "Gear-M5"]
DEFECT_TYPES    = ["Surface scratch", "Dimensional deviation", "Material void",
                   "Assembly gap", "Coating defect", "Weight out of spec"]
INSPECTORS      = ["QA-01 (John Lee)", "QA-02 (Sara Novak)", "QA-03 (Mike Torres)"]
DISPOSITIONS    = ["Accept", "Rework", "Scrap", "Hold for review"]

QUALITY_TEMPLATES = [
    "Batch {batch} of {product} inspected by {inspector}. {defect} found in {count} of {total} units "
    "({pct:.1f}%). Disposition: {disp}. Corrective action initiated: {action}. "
    "Process Cp: {cp:.2f}, Cpk: {cpk:.2f}. Control chart shows {trend}.",

    "In-process inspection at Station {station} flagged {product} batch {batch}. "
    "{count}/{total} units exhibited {defect}. Inspector {inspector} escalated to QA supervisor. "
    "Containment: {action}. Root cause investigation opened under NCR-{ncr}.",

    "Final inspection of {product} (batch {batch}). Overall yield: {yield_pct:.1f}%. "
    "{defect} accounted for majority of rejects. Measurement data logged to SPC system. "
    "Disposition: {disp}. Inspector: {inspector}.",
]

trends  = ["upward drift suggesting tool wear", "stable within control limits",
           "one-point-out-of-control signal at subgroup 12", "bimodal distribution — two machines suspected"]
q_actions = ["Adjusted feed rate on CNC-3", "Replaced worn cutting insert",
              "Retrained operator on torque spec", "Updated inspection sampling plan",
              "Calibrated CMM probe"]

def gen_quality(n):
    rows = []
    for i in range(n):
        product  = random.choice(PRODUCTS)
        inspector = random.choice(INSPECTORS)
        defect   = random.choice(DEFECT_TYPES)
        disp     = random.choice(DISPOSITIONS)
        total    = random.randint(50, 500)
        count    = random.randint(0, max(1, total // 10))
        pct      = count / total * 100
        cp       = round(random.uniform(0.8, 1.8), 2)
        cpk      = round(cp - random.uniform(0, 0.3), 2)
        batch    = random_id("BATCH", 5)
        ncr      = random_id("NCR", 5)
        station  = random.randint(1, 8)
        action   = random.choice(q_actions)
        tpl      = random.choice(QUALITY_TEMPLATES)
        desc     = tpl.format(product=product, batch=batch, inspector=inspector,
                              defect=defect, count=count, total=total, pct=pct,
                              disp=disp, action=action, cp=cp, cpk=cpk,
                              trend=random.choice(trends), station=station,
                              ncr=ncr, yield_pct=100 - pct)
        rows.append({
            "inspection_id"  : random_id("QI"),
            "date"           : random_date(),
            "product"        : product,
            "batch_id"       : batch,
            "inspector"      : inspector,
            "total_units"    : total,
            "defective_units": count,
            "defect_rate_pct": round(pct, 2),
            "defect_type"    : defect,
            "disposition"    : disp,
            "cp"             : cp,
            "cpk"            : cpk,
            "description"    : desc,
            "corrective_action": action,
        })
    return rows

# ── 3. MAINTENANCE ───────────────────────────────────────────────────────────

EQUIPMENT      = ["Compressor-C1", "Conveyor-B2", "Hydraulic-Press-H3",
                  "CNC-Lathe-L4", "Packaging-Unit-P5", "Chiller-CH6"]
MAINT_TYPES    = ["Preventive", "Corrective", "Predictive", "Breakdown"]
PARTS_REPLACED = ["Bearing SKF-6205", "Drive belt B-47", "Filter cartridge FC-10",
                  "Seal kit SK-200", "Coupling unit CU-15", "Oil pump OP-8",
                  "Relay module RM-3", "Motor capacitor MC-50"]

MAINT_TEMPLATES = [
    "{mtype} maintenance performed on {equip} by {tech}. Work order {wo}. "
    "Parts replaced: {parts}. Duration: {hours:.1f} hours. Equipment returned to service. "
    "Next scheduled PM: {next_date}. Oil sample sent to lab — results pending.",

    "Breakdown maintenance on {equip} following alarm code E{alarm:03d}. "
    "Technician {tech} diagnosed {fault}. {parts} replaced. Downtime: {hours:.1f} hours. "
    "Post-repair test: PASS. Work order {wo} closed.",

    "{mtype} inspection of {equip} completed. {tech} identified {fault} during vibration analysis. "
    "Proactive replacement of {parts} prevents unplanned downtime. "
    "MTBF updated: {mtbf} hours. Lubrication topped up to specification.",
]

faults   = ["bearing wear beyond threshold", "belt tension out of spec",
            "hydraulic pressure fluctuation", "elevated motor temperature",
            "abnormal vibration at 48 Hz", "coolant leakage from seal"]
m_techs  = ["Raj Iyer", "Sandra Bloom", "Tom Nguyen", "Priya Shah", "Kevin Osei"]

def gen_maintenance(n):
    rows = []
    for i in range(n):
        equip  = random.choice(EQUIPMENT)
        tech   = random.choice(m_techs)
        mtype  = random.choice(MAINT_TYPES)
        parts  = random.choice(PARTS_REPLACED)
        fault  = random.choice(faults)
        hours  = round(random.uniform(0.5, 12.0), 1)
        mtbf   = random.randint(500, 8000)
        alarm  = random.randint(100, 999)
        wo     = random_id("WO", 6)
        d      = random_date()
        next_d = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=90)).strftime("%Y-%m-%d")
        tpl    = random.choice(MAINT_TEMPLATES)
        desc   = tpl.format(equip=equip, tech=tech, mtype=mtype, parts=parts,
                            fault=fault, hours=hours, mtbf=mtbf, alarm=alarm,
                            wo=wo, next_date=next_d)
        rows.append({
            "work_order"       : wo,
            "date"             : d,
            "equipment"        : equip,
            "maintenance_type" : mtype,
            "technician"       : tech,
            "duration_hours"   : hours,
            "parts_replaced"   : parts,
            "fault_description": fault,
            "description"      : desc,
            "next_pm_date"     : next_d,
            "mtbf_hours"       : mtbf,
        })
    return rows

# ── 4. PRODUCTION ────────────────────────────────────────────────────────────

SHIFTS     = ["Day", "Afternoon", "Night"]
OPERATORS  = ["Op-101", "Op-102", "Op-103", "Op-104", "Op-105"]

PROD_TEMPLATES = [
    "Production run {run_id} on {line} during {shift} shift. Product: {product}. "
    "Target output: {target} units. Actual output: {actual} units (OEE: {oee:.1f}%). "
    "Downtime: {dt} minutes due to {cause}. Quality rejects: {rejects} units. "
    "Operator: {op}. Run completed without safety incidents.",

    "{shift} shift run on {line}. {product} batch {run_id}. Achieved {oee:.1f}% OEE against "
    "{target} unit target. {actual} units produced, {rejects} scrapped. "
    "Key downtime event: {cause} ({dt} min). Shift supervisor signed off.",

    "Run {run_id}: {line}, {product}, {shift} shift. Performance summary — "
    "Availability: {avail:.1f}%, Performance: {perf:.1f}%, Quality: {qual:.1f}%, "
    "OEE: {oee:.1f}%. Downtime cause: {cause}. Operator {op} reported {event}.",
]

prod_causes = ["changeover", "minor stoppages", "planned break", "unplanned jam",
               "material shortage", "quality hold", "tooling change"]
events      = ["no abnormal conditions", "slight vibration on feeder belt",
               "one label printer jam at 14:30", "temp spike in oven unit at 22:00"]

def gen_production(n):
    rows = []
    for i in range(n):
        line    = random.choice(LINES)
        product = random.choice(PRODUCTS)
        shift   = random.choice(SHIFTS)
        op      = random.choice(OPERATORS)
        target  = random.randint(200, 2000)
        avail   = round(random.uniform(75, 99), 1)
        perf    = round(random.uniform(75, 99), 1)
        qual    = round(random.uniform(90, 99.9), 1)
        oee     = round(avail * perf * qual / 10000, 1)
        actual  = int(target * avail / 100 * perf / 100)
        rejects = int(actual * (1 - qual / 100))
        dt      = random.randint(0, 120)
        cause   = random.choice(prod_causes)
        event   = random.choice(events)
        run_id  = random_id("RUN", 6)
        tpl     = random.choice(PROD_TEMPLATES)
        desc    = tpl.format(run_id=run_id, line=line, shift=shift, product=product,
                             target=target, actual=actual, oee=oee, dt=dt,
                             cause=cause, rejects=rejects, op=op, avail=avail,
                             perf=perf, qual=qual, event=event)
        rows.append({
            "run_id"         : run_id,
            "date"           : random_date(),
            "line"           : line,
            "product"        : product,
            "shift"          : shift,
            "operator"       : op,
            "target_units"   : target,
            "actual_units"   : actual,
            "reject_units"   : rejects,
            "oee_pct"        : oee,
            "downtime_minutes": dt,
            "downtime_cause" : cause,
            "description"    : desc,
        })
    return rows

# ── 5. SOPs ──────────────────────────────────────────────────────────────────

SOP_DOCUMENTS = [
    {
        "sop_id"  : "SOP-001",
        "title"   : "Machine Start-Up and Shutdown Procedure",
        "domain"  : "production",
        "revision": "Rev-4",
        "content" : textwrap.dedent("""
            # 1. Purpose
            This procedure defines the standard method for starting up and shutting down
            production line machinery to ensure operator safety and equipment longevity.

            # 2. Scope
            Applies to all production line operators on Line-A through Line-D.
            Equipment covered: conveyors, hydraulic presses, CNC lathes, and packaging units.

            # 3. Responsibilities
            - Shift Supervisor: authorises start-up; signs off on pre-start checklist.
            - Operator: performs all steps as listed; reports deviations immediately.
            - Maintenance Technician: on call during first 30 minutes of start-up.

            # 4. Pre-Start Checklist
            1. Confirm all guards are in place and safety interlocks are active.
            2. Verify that the area is clear of personnel and obstructions.
            3. Check hydraulic fluid level — must be between MIN and MAX marks.
            4. Inspect drive belts for wear or cracking; replace if wear exceeds 2 mm.
            5. Confirm lubrication points have been greased per PM schedule.
            6. Verify emergency stop buttons are functional — press and release each one.
            7. Confirm electrical panel shows no active fault codes.

            # 5. Start-Up Sequence
            1. Enable main isolator switch (key-operated — supervisor key required).
            2. Power on HMI and wait for system self-test to complete (<60 s).
            3. Select the correct recipe/programme on HMI for the current production run.
            4. Jog conveyor at 10% speed for 30 seconds; listen for abnormal noise.
            5. Gradually ramp to full production speed over 2 minutes.
            6. Monitor temperature, pressure, and vibration readings for first 5 minutes.
            7. If any alarm activates, halt and refer to the Alarm Response Matrix (ARM-01).

            # 6. Shutdown Sequence
            1. Complete the current production lot before initiating shutdown.
            2. Reduce speed to 10% on HMI; allow conveyor to clear.
            3. Press STOP on HMI; confirm all actuators return to home position.
            4. Disable main isolator (lockout if maintenance is required).
            5. Log run data (units produced, downtime, OEE) into the production system.
            6. Clean equipment surfaces per Cleaning SOP-005.

            # 7. Emergency Stop
            In case of any safety concern, press the nearest red E-STOP button immediately.
            Do not restart until the cause is investigated and signed off by the supervisor.

            # 8. Records
            Completed checklists must be filed in the Production Log binder (or digital ERP)
            within 30 minutes of shift end.
        """).strip(),
    },
    {
        "sop_id"  : "SOP-002",
        "title"   : "Incident Reporting and Escalation Procedure",
        "domain"  : "incidents",
        "revision": "Rev-6",
        "content" : textwrap.dedent("""
            # 1. Purpose
            Define how production incidents are reported, classified, escalated, and closed
            to minimise recurrence and ensure regulatory compliance.

            # 2. Definitions
            - **P1-Critical**: Plant-wide stoppage or safety hazard; immediate response.
            - **P2-High**: Line-level stoppage; response within 30 minutes.
            - **P3-Medium**: Partial degradation; response within 2 hours.
            - **P4-Low**: Minor anomaly; logged and addressed in next available slot.

            # 3. Detection and Initial Report
            1. Any employee who observes an incident must report it to the Shift Supervisor
               within 5 minutes of detection.
            2. Supervisor opens an incident ticket in the ITSM system with:
               - Affected line/equipment
               - Observed symptom
               - Time of detection
               - Initial severity estimate

            # 4. Severity Assessment
            Supervisor and on-call technician jointly assess severity within 15 minutes.
            Severity may be upgraded but not downgraded without QA approval.

            # 5. Escalation Matrix
            | Severity | Notify within | Escalate to |
            |----------|--------------|-------------|
            | P1       | 10 minutes    | Plant Manager + Safety Officer |
            | P2       | 30 minutes    | Production Manager |
            | P3       | 2 hours       | Shift Supervisor |
            | P4       | Next business day | Team Lead |

            # 6. Investigation and Root Cause Analysis
            1. For P1/P2: 5-Why analysis must be completed within 24 hours.
            2. For P3/P4: Fishbone or Pareto analysis within 5 business days.
            3. Document root cause in ITSM ticket.

            # 7. Corrective and Preventive Actions (CAPA)
            - Owner assigned for each action item.
            - Due dates set based on severity (P1: 7 days; P2: 14 days; P3/P4: 30 days).
            - CAPA effectiveness verified at 30- and 90-day checkpoints.

            # 8. Closure
            Ticket is closed only when all CAPAs are verified effective.
            QA sign-off required for P1 and P2.

            # 9. Records
            All incident records retained for minimum 5 years per regulatory requirement.
        """).strip(),
    },
    {
        "sop_id"  : "SOP-003",
        "title"   : "Preventive Maintenance Schedule and Execution",
        "domain"  : "maintenance",
        "revision": "Rev-3",
        "content" : textwrap.dedent("""
            # 1. Purpose
            Standardise the execution of preventive maintenance (PM) to maximise equipment
            reliability and minimise unplanned downtime.

            # 2. PM Frequency by Equipment Class
            | Equipment Class     | Daily | Weekly | Monthly | Quarterly | Annual |
            |---------------------|-------|--------|---------|-----------|--------|
            | Conveyors           | Visual| Belt   | Bearing | Full PM   | Overhaul|
            | Hydraulic Presses   | Oil   | Seals  | Filter  | Full PM   | Overhaul|
            | CNC Lathes          | Clean | Lube   | Calibrate| Full PM  | Overhaul|
            | Compressors         | Drain | Filter | Belts   | Full PM   | Overhaul|
            | Chillers            | Check | Coils  | Refrigerant| Full PM| Overhaul|

            # 3. Work Order Creation
            1. CMMS generates PM work orders automatically 7 days before due date.
            2. Maintenance Planner reviews and assigns technician.
            3. Required parts are kitted 2 days before scheduled date.

            # 4. Execution Steps
            1. Technician receives work order on mobile device.
            2. Obtain permit-to-work from Shift Supervisor before isolating equipment.
            3. Apply LOTO (Lockout/Tagout) per LOTO procedure HS-007.
            4. Perform tasks listed on work order; record actual condition of each component.
            5. Replace parts if condition is below acceptance threshold (see Part Specs TS-012).
            6. Take oil/vibration sample if scheduled.
            7. Remove LOTO; perform functional test before returning to production.
            8. Close work order in CMMS with all findings recorded.

            # 5. Parts and Spares Management
            Critical spares list is maintained in Stores. Minimum stock levels defined in
            Spare Parts Catalogue SPC-001. Technician must flag any stock-outs immediately.

            # 6. PM Compliance KPI
            Target: ≥ 95% PM completion on schedule. Reported monthly to Plant Manager.
            Overdue PMs trigger automatic escalation after 3 days.

            # 7. Records
            Completed work orders archived in CMMS. Retain for 7 years.
        """).strip(),
    },
    {
        "sop_id"  : "SOP-004",
        "title"   : "Quality Inspection and Non-Conformance Reporting",
        "domain"  : "quality",
        "revision": "Rev-5",
        "content" : textwrap.dedent("""
            # 1. Purpose
            Define the process for conducting quality inspections and raising non-conformance
            reports (NCRs) to prevent defective product from reaching customers.

            # 2. Inspection Types
            - **Incoming**: Raw material receipt — 100% visual + dimensional sample.
            - **In-Process**: At defined control points per Control Plan CP-003.
            - **Final**: 100% visual + AQL 1.0 attribute sampling before shipment.

            # 3. Sampling Plan
            Follow MIL-STD-1916 for attribute inspection. Sample sizes are defined
            in the Inspection and Test Plan (ITP) for each product family.

            # 4. Acceptance Criteria
            Dimensional tolerances per engineering drawing. Surface finish per RA spec.
            Any defect listed in Defect Catalogue DC-009 is automatic rejection.

            # 5. Non-Conformance Report (NCR) Process
            1. Inspector raises NCR in QMS within 1 hour of detection.
            2. Affected batch is quarantined with orange NCR tag.
            3. QA Engineer conducts disposition review within 4 hours.
            4. Disposition options: Accept As-Is / Rework / Scrap / Return to Vendor.
            5. Rework must be re-inspected before release. Scrap recorded in yield report.

            # 6. Statistical Process Control (SPC)
            Control charts (Xbar-R or IMR) maintained for critical dimensions.
            Out-of-control signals per Nelson Rules trigger immediate investigation.
            Cp/Cpk targets: Cp ≥ 1.33, Cpk ≥ 1.00 for critical characteristics.

            # 7. Corrective Actions
            NCR triggers CAPA if defect rate exceeds 0.5% in any batch or if same defect
            recurs three times in 30 days. CAPA ownership assigned to Process Engineer.

            # 8. Records
            Inspection results, NCRs, and CAPA records retained for 10 years.
        """).strip(),
    },
    {
        "sop_id"  : "SOP-005",
        "title"   : "Equipment Cleaning and Sanitisation Procedure",
        "domain"  : "production",
        "revision": "Rev-2",
        "content" : textwrap.dedent("""
            # 1. Purpose
            Prevent cross-contamination and maintain Good Manufacturing Practice (GMP)
            compliance through defined cleaning and sanitisation steps.

            # 2. Cleaning Frequencies
            - End of each production run: internal surfaces, hoppers, conveyors.
            - Weekly deep clean: all equipment, walls, floors, drains.
            - Monthly verification: ATP swab test; results logged.

            # 3. Approved Cleaning Agents
            | Agent          | Dilution | Contact Time | Application |
            |----------------|----------|--------------|-------------|
            | AlkaClean-200  | 2%       | 5 minutes    | Surfaces    |
            | SaniQuat-50    | 0.5%     | 3 minutes    | Sanitisation|
            | AcidFoam-Pro   | 1%       | 10 minutes   | CIP lines   |

            Do NOT mix cleaning agents. Rinse thoroughly between steps.

            # 4. Cleaning Procedure (End-of-Run)
            1. Isolate equipment (energy isolation per LOTO HS-007).
            2. Dismantle removable parts; soak in AlkaClean-200 solution.
            3. Rinse all surfaces with potable water.
            4. Apply SaniQuat-50 by spray or wipe; allow 3-minute contact time.
            5. Final rinse with potable water; confirm no residue.
            6. Reassemble equipment; record cleaning in log.
            7. QA verification swab on critical surfaces before next run.

            # 5. Personal Protective Equipment (PPE)
            Chemical-resistant gloves, safety goggles, and apron required.
            Face shield required when handling acid foam concentrates.

            # 6. Spill Response
            Neutralise acid/alkali spills with appropriate neutraliser from the spill kit
            located at each line entrance. Report all chemical spills as incidents per SOP-002.

            # 7. Verification
            ATP swab results must be < 100 RLU. Positive result triggers re-clean and re-test.
            Monthly summary reported to QA Manager.
        """).strip(),
    },
]

# Generate remaining SOPs programmatically to reach N_SOP
SOP_TITLES = [
    ("SOP-006", "Emergency Evacuation and Muster Procedure",          "incidents"),
    ("SOP-007", "New Product Introduction Checklist",                  "production"),
    ("SOP-008", "Calibration Management Procedure",                    "quality"),
    ("SOP-009", "Supplier Qualification and Audit Procedure",          "quality"),
    ("SOP-010", "Change Control and Management of Change Procedure",   "production"),
    ("SOP-011", "LOTO (Lockout/Tagout) Safety Procedure",              "maintenance"),
    ("SOP-012", "Spare Parts Ordering and Receiving Procedure",        "maintenance"),
    ("SOP-013", "Operator Training and Competency Assessment",         "production"),
    ("SOP-014", "Energy Monitoring and Reduction Procedure",           "production"),
    ("SOP-015", "Waste Management and Disposal Procedure",             "production"),
    ("SOP-016", "Document Control and Record Management",              "quality"),
    ("SOP-017", "Internal Audit Procedure",                            "quality"),
    ("SOP-018", "Customer Complaint Handling Procedure",               "quality"),
    ("SOP-019", "Lubrication Management Standard",                     "maintenance"),
    ("SOP-020", "Predictive Maintenance Vibration Analysis Procedure", "maintenance"),
    ("SOP-021", "Shift Handover Communication Standard",               "production"),
    ("SOP-022", "Production Scheduling and Sequencing Guideline",      "production"),
    ("SOP-023", "Material Handling and Storage Procedure",             "production"),
    ("SOP-024", "Measurement System Analysis (MSA/Gauge R&R)",         "quality"),
    ("SOP-025", "OEE Calculation and Reporting Guideline",             "production"),
    ("SOP-026", "Root Cause Analysis Toolkit",                         "incidents"),
    ("SOP-027", "IT/OT System Patch Management Procedure",             "incidents"),
    ("SOP-028", "Alarm Management and Rationalisation Procedure",      "incidents"),
    ("SOP-029", "Environmental Monitoring Programme",                  "production"),
    ("SOP-030", "Business Continuity and Disaster Recovery Plan",      "incidents"),
]

def gen_sop_body(sop_id, title, domain):
    return textwrap.dedent(f"""
        # 1. Purpose
        This document establishes the standard procedure for {title.lower()}.
        It applies to all personnel involved in {domain} activities and ensures consistency,
        compliance, and continuous improvement.

        # 2. Scope
        Applicable to all shifts and production lines. Any deviation from this procedure
        must be approved in writing by the department manager and documented in the change
        control system.

        # 3. Responsibilities
        - Department Manager: procedure owner; reviews annually.
        - Supervisors: ensure team compliance; report deviations.
        - Operators/Technicians: follow procedure as written; raise concerns immediately.
        - QA: verify compliance during audits; issue NCRs for non-conformances.

        # 4. Procedure Steps
        1. Review the applicable requirements and confirm all prerequisites are met.
        2. Gather required tools, materials, and PPE before starting.
        3. Perform the activity following the steps defined in this document.
        4. Record all observations and measurements in the designated log or system.
        5. If an out-of-specification condition is found, stop work and notify supervisor.
        6. Complete all required documentation before leaving the workstation.

        # 5. Key Performance Indicators
        Compliance rate: ≥ 95% on internal audits.
        Deviations logged and tracked monthly by QA.

        # 6. Related Documents
        Refer to the Document Master List (DML-001) for the latest revision of all
        supporting specifications, drawings, and work instructions referenced herein.

        # 7. Revision History
        Rev-1: Initial release.
        Rev-2: Incorporated audit findings from ISO 9001 gap assessment.
    """).strip()

def gen_sop_rows():
    rows = list(SOP_DOCUMENTS)  # first 5 detailed SOPs
    for sop_id, title, domain in SOP_TITLES:
        rows.append({
            "sop_id"  : sop_id,
            "title"   : title,
            "domain"  : domain,
            "revision": f"Rev-{random.randint(1, 8)}",
            "content" : gen_sop_body(sop_id, title, domain),
        })
    return rows

# ── write CSVs ───────────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"  [WARN] No rows for {path}")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):>4} rows → {path}")

# COMMAND ----------

print("Generating synthetic datasets...\n")

datasets = {
    "incidents.csv"  : gen_incidents(N_INCIDENTS),
    "quality.csv"    : gen_quality(N_QUALITY),
    "maintenance.csv": gen_maintenance(N_MAINTENANCE),
    "production.csv" : gen_production(N_PRODUCTION),
    "sop.csv"        : gen_sop_rows(),
}

for filename, rows in datasets.items():
    out_path = os.path.join(OUTPUT_DIR, filename)
    write_csv(out_path, rows)

print(f"\nAll datasets written to: {OUTPUT_DIR}")

# COMMAND ----------

# Quick validation
import csv as _csv

print("\n── Dataset summary ──────────────────────────────────────────────")
for filename in datasets:
    fpath = os.path.join(OUTPUT_DIR, filename)
    with open(fpath, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows   = list(reader)
    cols = list(rows[0].keys()) if rows else []
    print(f"  {filename:<20} rows={len(rows):>4}  cols={len(cols):>2}  {cols}")
print("─" * 65)
