from pathlib import Path
import pandas as pd
import json
import hashlib

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "final"
    / "opensensemap"
    / "opensensemap_target_hourly_wide.csv"
)

OUTPUT_DIR = ROOT / "data" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTITION_FILE = OUTPUT_DIR / "opensensemap_target_temporal_partitions.csv"
AUDIT_FILE = OUTPUT_DIR / "opensensemap_target_temporal_partition_audit.csv"
MANIFEST_FILE = OUTPUT_DIR / "opensensemap_target_temporal_partition_manifest.json"


# ============================================================
# FROZEN EXPERIMENTAL PROTOCOL
# ============================================================

PERIODS = {
    "adaptation": (
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-06-30 23:00:00", tz="UTC"),
    ),
    "validation": (
        pd.Timestamp("2026-07-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-08-31 23:00:00", tz="UTC"),
    ),
    "test": (
        pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-09-30 23:00:00", tz="UTC"),
    ),
}

VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("FREEZE OPENSENSEMAP TARGET TEMPORAL PROTOCOL")
print("=" * 100)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="raise",
)

df = df.sort_values(
    ["box_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# BASIC INTEGRITY CHECKS
# ============================================================

required_columns = [
    "selection_order",
    "box_id",
    "timestamp",
]

for variable in VARIABLES:
    required_columns.extend(
        [
            variable,
            f"{variable}_observed",
            f"{variable}_imputed",
        ]
    )

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing required columns: {missing_columns}"
    )


n_boxes = df["box_id"].nunique()

print()
print("Target boxes:")
print(n_boxes)

if n_boxes != 17:
    raise RuntimeError(
        f"Expected 17 frozen target boxes, found {n_boxes}"
    )


# ============================================================
# VERIFY TEMPORAL GRID
# ============================================================

expected_start = pd.Timestamp(
    "2026-01-01 00:00:00",
    tz="UTC",
)

expected_end = pd.Timestamp(
    "2026-09-30 23:00:00",
    tz="UTC",
)

if df["timestamp"].min() != expected_start:
    raise RuntimeError(
        "Unexpected target-period start."
    )

if df["timestamp"].max() != expected_end:
    raise RuntimeError(
        "Unexpected target-period end."
    )


expected_total_hours = int(
    (
        expected_end
        - expected_start
    ).total_seconds()
    / 3600
) + 1

print()
print("Expected hours per box:")
print(expected_total_hours)

counts = df.groupby("box_id").size()

if not (counts == expected_total_hours).all():
    raise RuntimeError(
        "At least one target box does not have "
        "the complete hourly timestamp grid."
    )


# ============================================================
# ASSIGN PARTITIONS
# ============================================================

df["target_partition"] = None

for partition, (start, end) in PERIODS.items():

    mask = (
        (df["timestamp"] >= start)
        & (df["timestamp"] <= end)
    )

    df.loc[
        mask,
        "target_partition"
    ] = partition


if df["target_partition"].isna().any():
    raise RuntimeError(
        "Some timestamps were not assigned "
        "to a target partition."
    )


# ============================================================
# CHECK PARTITION ORDER
# ============================================================

adapt_end = PERIODS["adaptation"][1]
val_start = PERIODS["validation"][0]
val_end = PERIODS["validation"][1]
test_start = PERIODS["test"][0]

if not adapt_end < val_start:
    raise RuntimeError(
        "Adaptation and validation periods overlap."
    )

if not val_end < test_start:
    raise RuntimeError(
        "Validation and test periods overlap."
    )


# ============================================================
# PARTITION SUMMARY
# ============================================================

partition_rows = []

for partition, (start, end) in PERIODS.items():

    x = df[
        df["target_partition"] == partition
    ]

    expected_hours = int(
        (
            end - start
        ).total_seconds()
        / 3600
    ) + 1

    row = {
        "partition": partition,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "boxes": x["box_id"].nunique(),
        "hours_per_box": expected_hours,
        "total_rows": len(x),
    }

    for variable in VARIABLES:

        observed = x[
            f"{variable}_observed"
        ].sum()

        available = x[
            variable
        ].notna().sum()

        row[
            f"{variable}_observed_percent"
        ] = (
            100.0 * observed / len(x)
        )

        row[
            f"{variable}_available_percent"
        ] = (
            100.0 * available / len(x)
        )

    partition_rows.append(row)


partition_summary = pd.DataFrame(
    partition_rows
)


# ============================================================
# PER-BOX AUDIT
# ============================================================

audit_rows = []

for partition, (start, end) in PERIODS.items():

    x = df[
        df["target_partition"] == partition
    ]

    for (
        selection_order,
        box_id
    ), g in x.groupby(
        ["selection_order", "box_id"]
    ):

        row = {
            "selection_order": selection_order,
            "box_id": box_id,
            "partition": partition,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "hours": len(g),
        }

        for variable in VARIABLES:

            observed = g[
                f"{variable}_observed"
            ].sum()

            imputed = g[
                f"{variable}_imputed"
            ].sum()

            available = g[
                variable
            ].notna().sum()

            row[
                f"{variable}_observed_hours"
            ] = int(observed)

            row[
                f"{variable}_observed_percent"
            ] = (
                100.0
                * observed
                / len(g)
            )

            row[
                f"{variable}_imputed_hours"
            ] = int(imputed)

            row[
                f"{variable}_available_hours"
            ] = int(available)

            row[
                f"{variable}_available_percent"
            ] = (
                100.0
                * available
                / len(g)
            )

        audit_rows.append(row)


audit = pd.DataFrame(audit_rows)


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

print()
print("=" * 100)
print("CONSISTENCY CHECKS")
print("=" * 100)

expected_partition_hours = {
    "adaptation": 181 * 24,
    "validation": 62 * 24,
    "test": 30 * 24,
}

for partition, expected_hours in expected_partition_hours.items():

    x = audit[
        audit["partition"] == partition
    ]

    if not (
        x["hours"] == expected_hours
    ).all():

        raise RuntimeError(
            f"Unexpected number of hours "
            f"in {partition} partition."
        )


# Every box must appear in all partitions

box_partition_counts = (
    audit.groupby("box_id")[
        "partition"
    ].nunique()
)

if not (
    box_partition_counts == 3
).all():

    raise RuntimeError(
        "Not every target box occurs "
        "in all three partitions."
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

partition_summary.to_csv(
    PARTITION_FILE,
    index=False,
)

audit.to_csv(
    AUDIT_FILE,
    index=False,
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "dataset": "openSenseMap",

    "protocol_name":
        "frozen_chronological_target_protocol",

    "target_boxes": int(n_boxes),

    "full_target_period": {
        "start": expected_start.isoformat(),
        "end": expected_end.isoformat(),
        "hours_per_box": expected_total_hours,
    },

    "partitions": {
        name: {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        for name, (start, end)
        in PERIODS.items()
    },

    "partition_roles": {

        "adaptation":
            "Chronologically available target-domain "
            "history permitted for causal adaptation.",

        "validation":
            "Chronologically subsequent target-domain "
            "period permitted for model/adaptation "
            "selection where explicitly defined.",

        "test":
            "Strictly held-out final target-domain "
            "evaluation period.",
    },

    "causality_constraints": {

        "random_temporal_sampling": False,

        "future_target_observations_used": False,

        "test_used_for_normalization": False,

        "test_used_for_model_selection": False,

        "test_used_for_early_stopping": False,

        "test_used_for_threshold_selection": False,

        "test_used_for_adaptation": False,

        "backward_fill_allowed": False,

        "future_interpolation_allowed": False,
    },

    "missing_target_policy": {

        "long_gaps_remain_missing": True,

        "evaluation_requires_available_ground_truth":
            True,

        "missing_ground_truth_is_not_imputed_for_scoring":
            True,
    },

    "cohort_policy": {

        "number_of_boxes": 17,

        "boxes_removed_based_on_partition_performance":
            False,

        "boxes_removed_based_on_monthly_pm25_coverage":
            False,

        "forecasting_results_used_for_partition_choice":
            False,
    },

    "input_file":
        str(INPUT_FILE.relative_to(ROOT)),

    "input_sha256":
        sha256_file(INPUT_FILE),

    "generated_files": [
        str(PARTITION_FILE.relative_to(ROOT)),
        str(AUDIT_FILE.relative_to(ROOT)),
        str(MANIFEST_FILE.relative_to(ROOT)),
    ],
}


with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print("PASSED")

print()
print("=" * 100)
print("TARGET PARTITION SUMMARY")
print("=" * 100)

pd.set_option(
    "display.max_columns",
    None,
)

print(
    partition_summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("PM2.5 PER-BOX COVERAGE")
print("=" * 100)

pm_audit = audit[
    [
        "selection_order",
        "box_id",
        "partition",
        "pm25_observed_percent",
        "pm25_available_percent",
    ]
]

print(
    pm_audit.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("TARGET TEMPORAL PROTOCOL FROZEN")
print("=" * 100)

print(
    """
Adaptation:
2026-01-01 00:00 UTC
to
2026-06-30 23:00 UTC

Validation:
2026-07-01 00:00 UTC
to
2026-08-31 23:00 UTC

Test:
2026-09-01 00:00 UTC
to
2026-09-30 23:00 UTC

All 17 target boxes are retained.

No target box or month was removed based on
forecasting performance.

Missing ground truth will not be manufactured
for evaluation.

The test partition is strictly held out from:

  - normalization fitting
  - model selection
  - early stopping
  - adaptation
  - threshold selection

The temporal protocol must not be changed after
examining downstream forecasting results.
"""
)