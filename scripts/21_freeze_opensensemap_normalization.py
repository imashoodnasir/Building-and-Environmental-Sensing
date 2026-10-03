from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

TARGET_FILE = (
    ROOT / "data" / "final" / "opensensemap"
    / "opensensemap_target_hourly_wide.csv"
)

COHORT_FILE = (
    ROOT / "data" / "metadata"
    / "opensensemap_final_eligible_target_cohort.csv"
)

WINDOW_MANIFEST_FILE = (
    ROOT / "data" / "metadata"
    / "opensensemap_target_window_manifest.json"
)

OUTPUT_STATS = (
    ROOT / "data" / "metadata"
    / "opensensemap_adaptation_normalization_statistics.csv"
)

OUTPUT_BOX_AUDIT = (
    ROOT / "data" / "metadata"
    / "opensensemap_adaptation_normalization_box_audit.csv"
)

OUTPUT_MANIFEST = (
    ROOT / "data" / "metadata"
    / "opensensemap_adaptation_normalization_manifest.json"
)


# =============================================================================
# FROZEN RULES
# =============================================================================

ADAPTATION_START = pd.Timestamp(
    "2026-01-01 00:00:00",
    tz="UTC",
)

ADAPTATION_END = pd.Timestamp(
    "2026-06-30 23:00:00",
    tz="UTC",
)

VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]

EXPECTED_BOXES = 17
EXPECTED_HOURS_PER_BOX = 4344


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def as_bool(series):

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# =============================================================================
# START
# =============================================================================

t0 = time.time()

banner("OPENSENSEMAP ADAPTATION-ONLY NORMALIZATION")


# =============================================================================
# VERIFY INPUT FILES
# =============================================================================

for path in [
    TARGET_FILE,
    COHORT_FILE,
    WINDOW_MANIFEST_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nRequired frozen inputs found.")


# =============================================================================
# VERIFY WINDOW MANIFEST
# =============================================================================

with open(
    WINDOW_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as f:
    window_manifest = json.load(f)

if window_manifest.get("frozen_boxes") != EXPECTED_BOXES:
    raise RuntimeError(
        "Target-window manifest does not contain "
        "the expected 17 frozen boxes."
    )

partitions = window_manifest.get("partitions", {})

if "adaptation" not in partitions:
    raise RuntimeError(
        "Adaptation partition missing from target-window manifest."
    )

print("\nTarget window protocol verified.")
print(f"Frozen boxes: {EXPECTED_BOXES}")


# =============================================================================
# LOAD FROZEN COHORT
# =============================================================================

banner("LOADING FROZEN TARGET COHORT")

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={"box_id": str},
)

frozen_boxes = (
    cohort["box_id"]
    .astype(str)
    .drop_duplicates()
    .tolist()
)

if len(frozen_boxes) != EXPECTED_BOXES:
    raise RuntimeError(
        f"Expected {EXPECTED_BOXES} boxes, "
        f"found {len(frozen_boxes)}."
    )

print(f"\nFrozen target boxes: {len(frozen_boxes)}")


# =============================================================================
# LOAD TARGET DATA
# =============================================================================

banner("LOADING TARGET HOURLY DATA")

df = pd.read_csv(
    TARGET_FILE,
    dtype={"box_id": str},
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="raise",
)

required_columns = [
    "box_id",
    "timestamp",
]

for variable in VARIABLES:
    required_columns.extend([
        variable,
        f"{variable}_observed",
    ])

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

df = df[
    df["box_id"].isin(frozen_boxes)
].copy()

for variable in VARIABLES:

    df[f"{variable}_observed"] = as_bool(
        df[f"{variable}_observed"]
    )

print(f"\nRows loaded: {len(df):,}")
print(f"Boxes: {df['box_id'].nunique()}")


# =============================================================================
# ISOLATE ADAPTATION PERIOD
# =============================================================================

banner("ISOLATING ADAPTATION PARTITION")

adapt = df[
    (df["timestamp"] >= ADAPTATION_START)
    & (df["timestamp"] <= ADAPTATION_END)
].copy()

expected_rows = (
    EXPECTED_BOXES
    * EXPECTED_HOURS_PER_BOX
)

print(f"\nAdaptation start: {ADAPTATION_START}")
print(f"Adaptation end:   {ADAPTATION_END}")
print(f"Expected rows:    {expected_rows:,}")
print(f"Actual rows:      {len(adapt):,}")

if len(adapt) != expected_rows:
    raise RuntimeError(
        "Adaptation row count does not match frozen protocol."
    )

counts = adapt.groupby("box_id").size()

if not (counts == EXPECTED_HOURS_PER_BOX).all():
    raise RuntimeError(
        "One or more boxes do not contain exactly "
        f"{EXPECTED_HOURS_PER_BOX} adaptation hours."
    )

if adapt["box_id"].nunique() != EXPECTED_BOXES:
    raise RuntimeError(
        "Not all frozen boxes are present in adaptation data."
    )

print("\nAdaptation partition verification: PASSED")


# =============================================================================
# IMPORTANT NORMALIZATION POLICY
# =============================================================================

banner("NORMALIZATION POLICY")

print(
    """
Normalization parameters will be estimated from DIRECTLY
OBSERVED adaptation-period values only.

Causally forward-filled values are NOT used to estimate
mean or standard deviation.

This prevents the frequency of imputation from influencing
the normalization parameters.

The frozen physical-unit dataset remains unchanged.
"""
)


# =============================================================================
# CALCULATE GLOBAL ADAPTATION STATISTICS
# =============================================================================

banner("CALCULATING ADAPTATION STATISTICS")

stats_rows = []

for variable in VARIABLES:

    observed_mask = (
        adapt[f"{variable}_observed"]
        & adapt[variable].notna()
    )

    values = (
        adapt.loc[
            observed_mask,
            variable,
        ]
        .astype(float)
        .to_numpy()
    )

    if len(values) == 0:
        raise RuntimeError(
            f"No directly observed values available "
            f"for {variable}."
        )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Non-finite directly observed values "
            f"found for {variable}."
        )

    mean = float(np.mean(values))

    # Population standard deviation, consistent with z-score
    # transformation of the frozen empirical adaptation distribution.
    std = float(np.std(values, ddof=0))

    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(
            f"Invalid standard deviation for {variable}: {std}"
        )

    row = {
        "variable": variable,
        "n": int(len(values)),
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
    }

    stats_rows.append(row)


stats = pd.DataFrame(stats_rows)

print()
print(stats.to_string(index=False))


# =============================================================================
# Z-SCORE SANITY CHECK
# =============================================================================

banner("Z-SCORE SANITY CHECK")

sanity_rows = []

for _, row in stats.iterrows():

    variable = row["variable"]

    observed_mask = (
        adapt[f"{variable}_observed"]
        & adapt[variable].notna()
    )

    values = (
        adapt.loc[
            observed_mask,
            variable,
        ]
        .astype(float)
        .to_numpy()
    )

    z = (
        values - row["mean"]
    ) / row["std"]

    sanity_rows.append({
        "variable": variable,
        "normalized_mean": float(np.mean(z)),
        "normalized_std": float(np.std(z, ddof=0)),
        "normalized_min": float(np.min(z)),
        "normalized_max": float(np.max(z)),
    })


sanity = pd.DataFrame(sanity_rows)

print()
print(sanity.to_string(index=False))

if not np.allclose(
    sanity["normalized_mean"],
    0.0,
    atol=1e-8,
):
    raise RuntimeError(
        "Normalized means failed sanity check."
    )

if not np.allclose(
    sanity["normalized_std"],
    1.0,
    atol=1e-8,
):
    raise RuntimeError(
        "Normalized standard deviations failed sanity check."
    )

print("\nZ-score sanity check: PASSED")


# =============================================================================
# BOX-LEVEL AUDIT
# =============================================================================

banner("BOX-LEVEL ADAPTATION AUDIT")

audit_rows = []

for box_id in frozen_boxes:

    box = adapt[
        adapt["box_id"] == box_id
    ]

    row = {
        "box_id": box_id,
        "adaptation_hours": len(box),
    }

    for variable in VARIABLES:

        observed = (
            box[f"{variable}_observed"]
            & box[variable].notna()
        )

        available = box[variable].notna()

        row[
            f"{variable}_observed_hours"
        ] = int(observed.sum())

        row[
            f"{variable}_observed_percent"
        ] = float(
            100.0 * observed.mean()
        )

        row[
            f"{variable}_available_hours"
        ] = int(available.sum())

        row[
            f"{variable}_available_percent"
        ] = float(
            100.0 * available.mean()
        )

    audit_rows.append(row)


audit = pd.DataFrame(audit_rows)

display_columns = [
    "box_id",
    "temperature_observed_percent",
    "relative_humidity_observed_percent",
    "pressure_observed_percent",
    "pm25_observed_percent",
]

print()
print(
    audit[display_columns]
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# LEAKAGE ASSERTIONS
# =============================================================================

banner("LEAKAGE ASSERTIONS")

if adapt["timestamp"].min() != ADAPTATION_START:
    raise RuntimeError(
        "Unexpected adaptation start timestamp."
    )

if adapt["timestamp"].max() != ADAPTATION_END:
    raise RuntimeError(
        "Unexpected adaptation end timestamp."
    )

if (
    adapt["timestamp"]
    > ADAPTATION_END
).any():
    raise RuntimeError(
        "Validation/test observations entered "
        "normalization calculation."
    )

print("\nAdaptation-only fitting: PASSED")
print("Validation observations used: NO")
print("Test observations used: NO")


# =============================================================================
# SAVE STATISTICS
# =============================================================================

stats.to_csv(
    OUTPUT_STATS,
    index=False,
)

audit.to_csv(
    OUTPUT_BOX_AUDIT,
    index=False,
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {
    "dataset": "openSenseMap",
    "domain_role": "target",
    "normalization_fit_partition": "adaptation",
    "adaptation_start": ADAPTATION_START.isoformat(),
    "adaptation_end": ADAPTATION_END.isoformat(),
    "frozen_boxes": EXPECTED_BOXES,
    "hours_per_box": EXPECTED_HOURS_PER_BOX,
    "variables": VARIABLES,
    "method": "global_z_score",
    "statistics_source": (
        "directly observed adaptation-period values only"
    ),
    "imputed_values_used_for_statistics": False,
    "validation_values_used_for_statistics": False,
    "test_values_used_for_statistics": False,
    "standard_deviation_ddof": 0,
    "physical_unit_data_modified": False,
    "statistics": {
        row["variable"]: {
            "n": int(row["n"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "median": float(row["median"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "q01": float(row["q01"]),
            "q99": float(row["q99"]),
        }
        for _, row in stats.iterrows()
    },
}


with open(
    OUTPUT_MANIFEST,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# =============================================================================
# FINAL
# =============================================================================

elapsed = time.time() - t0

banner("OPENSENSEMAP ADAPTATION NORMALIZATION FROZEN")

print(
    f"""
Frozen target boxes:
  {EXPECTED_BOXES}

Normalization fit period:
  {ADAPTATION_START}
  to
  {ADAPTATION_END}

Variables:
  temperature
  relative_humidity
  pressure
  pm25

Normalization:
  global z-score

Statistics estimated from:
  DIRECTLY OBSERVED adaptation-period values only

Validation used for normalization:
  NO

Test used for normalization:
  NO

Causally imputed values used to estimate statistics:
  NO

Physical-unit target dataset modified:
  NO

Processing time:
  {elapsed:.2f} seconds

Generated:

  data\\metadata\\opensensemap_adaptation_normalization_statistics.csv

  data\\metadata\\opensensemap_adaptation_normalization_box_audit.csv

  data\\metadata\\opensensemap_adaptation_normalization_manifest.json

IMPORTANT:

These target normalization parameters are now intended
to remain FROZEN.

They may be applied to adaptation, validation, and test
target observations without refitting.

The target TEST partition must never be used to modify
these parameters.

No forecasting model was trained.

No forecasting performance was inspected.
"""
)