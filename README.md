# Trackmind

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
