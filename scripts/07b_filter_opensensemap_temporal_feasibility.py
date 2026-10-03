from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

BOX_INPUT = META / "opensensemap_recent_candidate_boxes.csv"
SENSOR_INPUT = META / "opensensemap_recent_candidate_sensors.csv"

BOX_OUTPUT = META / "opensensemap_temporally_feasible_boxes.csv"
SENSOR_OUTPUT = META / "opensensemap_temporally_feasible_sensors.csv"
AUDIT_OUTPUT = META / "opensensemap_temporal_feasibility_audit.csv"

TARGET_END = pd.Timestamp("2026-09-30T23:59:59Z")
MIN_SPAN_DAYS = 180

# Latest possible creation time that could still permit
# a 180-day observation span by TARGET_END.
LATEST_CREATION = TARGET_END - pd.Timedelta(days=MIN_SPAN_DAYS)


print("=" * 88)
print("OPENSENSEMAP TEMPORAL FEASIBILITY FILTER")
print("=" * 88)

print("\nTarget end:")
print(TARGET_END)

print("\nRequired minimum span:")
print(f"{MIN_SPAN_DAYS} days")

print("\nLatest feasible box creation:")
print(LATEST_CREATION)


# ============================================================
# LOAD
# ============================================================

boxes = pd.read_csv(
    BOX_INPUT,
    dtype={"box_id": str}
)

sensors = pd.read_csv(
    SENSOR_INPUT,
    dtype={
        "box_id": str,
        "sensor_id": str,
    }
)


# ============================================================
# PARSE CREATION TIME
# ============================================================

boxes["metadata_created_at"] = pd.to_datetime(
    boxes["metadata_created_at"],
    utc=True,
    errors="coerce"
)


# ============================================================
# FEASIBILITY
# ============================================================

boxes["creation_time_available"] = (
    boxes["metadata_created_at"].notna()
)

boxes["creation_temporally_feasible"] = (
    boxes["metadata_created_at"]
    <= LATEST_CREATION
)

boxes["temporal_feasibility_reason"] = (
    "retained"
)

boxes.loc[
    boxes["metadata_created_at"].isna(),
    "temporal_feasibility_reason"
] = "missing_creation_timestamp"

boxes.loc[
    boxes["metadata_created_at"] > LATEST_CREATION,
    "temporal_feasibility_reason"
] = "created_too_late_for_180_day_span"


# ============================================================
# SAVE AUDIT
# ============================================================

boxes.to_csv(
    AUDIT_OUTPUT,
    index=False
)


# ============================================================
# RETAIN FEASIBLE BOXES
# ============================================================

feasible_boxes = boxes[
    boxes["creation_temporally_feasible"]
].copy()

feasible_boxes.to_csv(
    BOX_OUTPUT,
    index=False
)

feasible_ids = set(
    feasible_boxes["box_id"]
    .astype(str)
)


# ============================================================
# RETAIN ASSOCIATED SENSORS
# ============================================================

feasible_sensors = sensors[
    sensors["box_id"]
    .astype(str)
    .isin(feasible_ids)
].copy()

feasible_sensors.to_csv(
    SENSOR_OUTPUT,
    index=False
)


# ============================================================
# CHECK FOUR-VARIABLE STRUCTURE
# ============================================================

variable_counts = (
    feasible_sensors
    .groupby("box_id")[
        "canonical_variable"
    ]
    .nunique()
)

n_all_four = int(
    (variable_counts == 4).sum()
)


# ============================================================
# SUMMARY
# ============================================================

print("\nInput boxes:")
print(len(boxes))

print("\nInput sensors:")
print(len(sensors))

print("\nExcluded because created too late:")
print(
    (
        boxes["metadata_created_at"]
        > LATEST_CREATION
    ).sum()
)

print("\nMissing creation timestamp:")
print(
    boxes["metadata_created_at"]
    .isna()
    .sum()
)

print("\nRetained temporally feasible boxes:")
print(len(feasible_boxes))

print("\nRetained sensor candidates:")
print(len(feasible_sensors))

print("\nBoxes still containing all four variables:")
print(n_all_four)

print("\nVariable counts:")

print(
    feasible_sensors[
        "canonical_variable"
    ].value_counts()
)

print("\nSensors per box distribution:")

print(
    feasible_sensors
    .groupby("box_id")
    .size()
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 88)
print("TEMPORAL FEASIBILITY FILTER COMPLETE")
print("=" * 88)

print(
    """
Generated:

  data/metadata/opensensemap_temporally_feasible_boxes.csv
  data/metadata/opensensemap_temporally_feasible_sensors.csv
  data/metadata/opensensemap_temporal_feasibility_audit.csv

This is a necessary-condition filter only.

A retained box is NOT yet considered eligible.

Actual observations must still demonstrate:
  - >= 180-day observation span
  - >= 70% hourly completeness
  - valid measurements for all four variables

No forecasting performance was used.
"""
)