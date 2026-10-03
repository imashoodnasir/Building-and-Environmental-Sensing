from pathlib import Path
import pandas as pd
import json
import glob

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
RAW = ROOT / "data" / "raw" / "opensensemap"

SENSOR_FILE = META / "opensensemap_semantically_eligible_sensors.csv"
BOX_FILE = META / "opensensemap_eligible_boxes.csv"

OUT_BOX = META / "opensensemap_recent_candidate_boxes.csv"
OUT_SENSOR = META / "opensensemap_recent_candidate_sensors.csv"
OUT_AUDIT = META / "opensensemap_recency_filter_audit.csv"

CUTOFF = pd.Timestamp("2026-09-30T00:00:00Z")

print("=" * 88)
print("OPENSENSEMAP RECENCY PRE-SCREEN")
print("=" * 88)

# ------------------------------------------------------------
# Load semantic candidates
# ------------------------------------------------------------

boxes = pd.read_csv(
    BOX_FILE,
    dtype={"box_id": str}
)

sensors = pd.read_csv(
    SENSOR_FILE,
    dtype={
        "box_id": str,
        "sensor_id": str
    }
)

# ------------------------------------------------------------
# Load frozen metadata snapshot
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        str(RAW / "boxes_metadata_*.json")
    )
)

if not files:
    raise FileNotFoundError(
        "No openSenseMap metadata snapshot found."
    )

snapshot = files[-1]

print("\nMetadata snapshot:")
print(snapshot)

with open(
    snapshot,
    "r",
    encoding="utf-8"
) as f:
    raw = json.load(f)

metadata = pd.DataFrame([
    {
        "box_id": str(b.get("_id")),
        "lastMeasurementAt": b.get("lastMeasurementAt"),
        "metadata_updated_at": b.get("updatedAt"),
        "metadata_created_at": b.get("createdAt"),
    }
    for b in raw
])

metadata["lastMeasurementAt"] = pd.to_datetime(
    metadata["lastMeasurementAt"],
    utc=True,
    errors="coerce"
)

# ------------------------------------------------------------
# Join
# ------------------------------------------------------------

audit = boxes.merge(
    metadata,
    on="box_id",
    how="left"
)

audit["has_last_measurement"] = (
    audit["lastMeasurementAt"].notna()
)

audit["active_at_target_end"] = (
    audit["lastMeasurementAt"] >= CUTOFF
)

audit["recency_reason"] = "excluded_before_target_end"

audit.loc[
    audit["lastMeasurementAt"].isna(),
    "recency_reason"
] = "missing_lastMeasurementAt"

audit.loc[
    audit["active_at_target_end"],
    "recency_reason"
] = "retained_active_at_target_end"

# ------------------------------------------------------------
# Save full audit trail
# ------------------------------------------------------------

audit.to_csv(
    OUT_AUDIT,
    index=False
)

# ------------------------------------------------------------
# Retain boxes active through target end
# ------------------------------------------------------------

recent_boxes = audit[
    audit["active_at_target_end"]
].copy()

recent_boxes.to_csv(
    OUT_BOX,
    index=False
)

recent_ids = set(
    recent_boxes["box_id"].astype(str)
)

recent_sensors = sensors[
    sensors["box_id"]
    .astype(str)
    .isin(recent_ids)
].copy()

recent_sensors.to_csv(
    OUT_SENSOR,
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nOriginal semantically eligible boxes:")
print(len(audit))

print("\nMissing lastMeasurementAt:")
print(
    audit["lastMeasurementAt"]
    .isna()
    .sum()
)

print("\nStopped before target-period end:")
print(
    (
        audit["lastMeasurementAt"].notna()
        &
        ~audit["active_at_target_end"]
    ).sum()
)

print("\nRetained boxes:")
print(len(recent_boxes))

print("\nRetained sensor candidates:")
print(len(recent_sensors))

print("\nVariable counts:")

print(
    recent_sensors[
        "canonical_variable"
    ].value_counts()
)

print("\nEarliest retained lastMeasurementAt:")
print(
    recent_boxes[
        "lastMeasurementAt"
    ].min()
)

print("\nLatest retained lastMeasurementAt:")
print(
    recent_boxes[
        "lastMeasurementAt"
    ].max()
)

print("\n" + "=" * 88)
print("RECENCY PRE-SCREEN COMPLETE")
print("=" * 88)

print("""
Generated:

  data/metadata/opensensemap_recent_candidate_boxes.csv
  data/metadata/opensensemap_recent_candidate_sensors.csv
  data/metadata/opensensemap_recency_filter_audit.csv

IMPORTANT:
Recency is only a pre-screen.

It does NOT establish:
  - 180-day temporal coverage
  - >=70% hourly completeness
  - individual sensor continuity

Those criteria will be tested from observations next.

No forecasting results were used.
""")