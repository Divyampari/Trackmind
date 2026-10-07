"Incident Intelligence" section to the existing README.md.

Do not rewrite or remove any existing README content.

Use this exact content:

## 🚨 Incident Intelligence

The Incident Intelligence module converts detected safety-related behaviour events into structured safety incidents.

### What it does

- Receives behaviour events from the Behaviour Intelligence module.
- Creates a structured incident for each confirmed safety violation.
- Assigns a severity level to the incident.
- Records the worker ID associated with the incident.
- Records the timestamp and incident type.
- Prevents duplicate incidents for the same ongoing event.
- Stores incident information in a structured format for downstream dashboard and reporting use.
- Supports evidence/reference information when available.

### Incident Information

Each incident can contain:

- Incident ID
- Worker ID
- Incident type
- Severity
- Timestamp
- Duration
- Zone/location
- Description/explanation
- Evidence frame or reference

### Example Incident

```json
{
  "incident_id": "INC-001",
  "worker_id": 7,
  "event_type": "restricted_zone_violation",
  "severity": "critical",
  "zone": "Machine Zone A",
  "timestamp": "00:12",
  "duration": 6.2,
  "description": "Worker remained inside a restricted machine zone beyond the configured safety threshold."
}
Pipeline
Behaviour Event
      ↓
Incident Intelligence
      ↓
Incident Validation
      ↓
Severity Classification
      ↓
Duplicate Prevention
      ↓
Structured Incident
      ↓
Dashboard / Reporting
Purpose

The goal of Incident Intelligence is to transform low-level behaviour events into meaningful, structured safety incidents that can be consumed by the dashboard and used for monitoring, investigation, and reporting.

IMPORTANT:

Do not claim capabilities that are not implemented.
Do not add fake AI/LLM functionality.
Do not add screenshots.
Do not modify any other README sections.
After editing, show me exactly what was added.

### If you mean **the README code itself**

You can directly paste this into README:

```markdown
## 🚨 Incident Intelligence

The Incident Intelligence module converts detected safety-related behaviour events into structured safety incidents.

### What it does

- Receives behaviour events from the Behaviour Intelligence module.
- Creates a structured incident for each confirmed safety violation.
- Assigns a severity level to the incident.
- Records the worker ID associated with the incident.
- Records the timestamp and incident type.
- Prevents duplicate incidents for the same ongoing event.
- Stores incident information in a structured format for downstream dashboard and reporting use.
- Supports evidence/reference information when available.

### Incident Information

Each incident can contain:

- Incident ID
- Worker ID
- Incident type
- Severity
- Timestamp
- Duration
- Zone/location
- Description/explanation
- Evidence frame or reference

### Example Incident

```json
{
  "incident_id": "INC-001",
  "worker_id": 7,
  "event_type": "restricted_zone_violation",
  "severity": "critical",
  "zone": "Machine Zone A",
  "timestamp": "00:12",
  "duration": 6.2,
  "description": "Worker remained inside a restricted machine zone beyond the configured safety threshold."
}
Pipeline
Behaviour Event
      ↓
Incident Intelligence
      ↓
Incident Validation
      ↓
Severity Classification
      ↓
Duplicate Prevention
      ↓
Structured Incident
      ↓
Dashboard / Reporting
Purpose

The goal of Incident Intelligence is to transform low-level behaviour events into meaningful, structured safety incidents that can be consumed by the dashboard and used for monitoring, investigation, and reporting.


**One caution:** if your teammate's actual Incident Intelligence implementation uses different field names, don't blindly use the example JSON. The README should match the actual implementation.
Pasted markdown(6).md
File
IT IS SHOWING THE CODE RA  LIKE THIS GIVE FOR INCIDENTINTELLIGNE

Ahh, now I understand. You mean you want a normal README section, not a giant code block showing on GitHub.

Your existing README already has a Phase 3 — Evidence & Incident Intelligence section, and the actual implementation is specifically documented there.

So for Incident Intelligence alone, give Antigravity this prompt. It will add proper Markdown to the README rather than displaying the README syntax as visible code:

Update README.md.

I want ONLY the Incident Intelligence / Phase 3 section improved.

Do NOT put the Markdown syntax inside a ``` code block in the README.
The headings, bullets, tables, and JSON example should render normally on GitHub.

Use the existing implementation as the source of truth.

Replace/improve the existing Phase 3 section with:

## Phase 3 — Evidence & Incident Intelligence

Phase 3 transforms behaviour events into persistent, structured, evidence-backed safety incidents.

### Incident Intelligence

The Incident Intelligence module:

- Generates unique Incident IDs such as `INC-0001`, `INC-0002`, etc.
- Classifies incidents into `WARNING`, `HIGH`, and `CRITICAL` severity levels.
- Prevents duplicate alerts using configurable cooldown windows.
- Tracks incident duration and lifecycle status using `OPEN` and `CLOSED`.
- Supports querying incidents by severity, worker, and zone.
- Provides incident summary information.

### Evidence Capture

For detected safety violations, the system captures visual evidence:

- Automatically captures the relevant video frame.
- Highlights the configured safety zone.
- Displays the worker bounding box.
- Adds an information banner containing Incident ID, Worker ID, Zone, Timestamp, Frame, and Severity.
- Stores evidence images under `data/incidents/evidence/`.

### Structured Incident Report

The system generates a machine-readable JSON report at:

`data/incidents/incidents_report.json`

The report contains:

- Source information
- Total incident count
- Severity summary
- Zone summary
- Worker summary
- Detailed incident records

### Example Incident

Use a small JSON example showing:

- `incident_id`
- `worker_id`
- `event_type`
- `timestamp`
- `zone`
- `severity`
- `duration`
- `evidence_frame`
- `status`
- `description`

### Phase 3 Pipeline

Behaviour Event
↓
Incident Intelligence
↓
Severity Classification
↓
Duplicate Prevention
↓
Evidence Capture
↓
Structured Incident Report
↓
Dashboard / Reporting

### Running Phase 3

```text
python app_phase3.py

For a custom video and report:

python app_phase3.py --video videos/my_test.mp4 --output-report data/incidents/my_report.json

IMPORTANT:

Keep the existing Phase 1 and Phase 2 README sections unchanged.
Do not invent features.
Do not claim LLM/AI reasoning if it is not implemented.
Keep the wording professional and hackathon-friendly.
Make sure Markdown headings and bullets render normally instead of appearing as code.
Do not commit or push changes.

### One important correction

Your current README already says Phase 3 is **completed and verified**, and it documents the actual `incident_manager.py`, evidence capture, JSON reporting, severity levels, deduplication, and lifecycle. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

So **don't create a completely different Incident Intelligence description**. The prompt above tells Antigravity to cleanly present what is actually implemented.

And yes — the reason you were seeing the **code-looking format** before is that the Markdown itself was being placed inside a code fence. The README should have:

```text
## Phase 3 — Evidence & Incident Intelligence

as actual Markdown, not:

```markdown
## Phase 3 — Evidence & Incident Intelligence
```

That's the key difference.
