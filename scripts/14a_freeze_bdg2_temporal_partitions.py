from pathlib import Path
import json
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META_DIR = ROOT / "data" / "metadata"
DATA_DIR = ROOT / "data" / "final" / "bdg2" / "buildings"

COHORT_FILE = META_DIR / "bdg2_frozen_source_cohort.csv"

OUTPUT_FILE = META_DIR / "bdg2_temporal_partitions.csv"

AUDIT_FILE = META_DIR / "bdg2_temporal_partition_audit.csv"

MANIFEST_FILE = META_DIR / "bdg2_temporal_partition_manifest.json"


# =============================================================================
# FROZEN TEMPORAL BOUNDARIES
# =============================================================================

TRAIN_START = pd.Timestamp("2016-01-01 00:00:00")
TRAIN_END   = pd.Timestamp("2017-04-30 23:00:00")

VAL_START   = pd.Timestamp("2017-05-01 00:00:00")
VAL_END     = pd.Timestamp("2017-08-31 23:00:00")

TEST_START  = pd.Timestamp("2017-09-01 00:00:00")
TEST_END    = pd.Timestamp("2017-12-31 23:00:00")


# =============================================================================
# EXPECTED HOURS
# =============================================================================

def expected_hours(start, end):
    return len(
        pd.date_range(
            start,
            end,
            freq="h"
        )
    )


EXPECTED_TRAIN = expected_hours(
    TRAIN_START,
    TRAIN_END
)

EXPECTED_VAL = expected_hours(
    VAL_START,
    VAL_END
)

EXPECTED_TEST = expected_hours(
    TEST_START,
    TEST_END
)


# =============================================================================
# START
# =============================================================================

print("=" * 100)
print("FREEZE BDG2 CAUSAL TEMPORAL PARTITIONS")
print("=" * 100)

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={
        "building_id": str,
        "site_id": str,
    }
)

print("\nFrozen source buildings:")
print(len(cohort))

print("\nFrozen source sites:")
print(cohort["site_id"].nunique())


# =============================================================================
# CHECK SITE ACCOUNTING
# =============================================================================

all_original_sites = {
    "Bear", "Bobcat", "Bull", "Cockatoo", "Crow",
    "Eagle", "Fox", "Gator", "Hog", "Lamb",
    "Moose", "Mouse", "Panther", "Peacock",
    "Rat", "Robin", "Shrew", "Swan", "Wolf"
}

retained_sites = set(
    cohort["site_id"].unique()
)

missing_sites = sorted(
    all_original_sites - retained_sites
)

print("\nSites absent from frozen cohort:")
print(missing_sites)


# =============================================================================
# DEFINE PARTITION TABLE
# =============================================================================

partitions = pd.DataFrame([
    {
        "partition": "train",
        "start": TRAIN_START,
        "end": TRAIN_END,
        "expected_hours_per_building": EXPECTED_TRAIN,
    },
    {
        "partition": "validation",
        "start": VAL_START,
        "end": VAL_END,
        "expected_hours_per_building": EXPECTED_VAL,
    },
    {
        "partition": "test",
        "start": TEST_START,
        "end": TEST_END,
        "expected_hours_per_building": EXPECTED_TEST,
    },
])

partitions.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTemporal partitions:")
print(partitions.to_string(index=False))


# =============================================================================
# VERIFY CAUSAL ORDER
# =============================================================================

assert TRAIN_END < VAL_START
assert VAL_END < TEST_START

assert (
    VAL_START - TRAIN_END
    == pd.Timedelta(hours=1)
)

assert (
    TEST_START - VAL_END
    == pd.Timedelta(hours=1)
)

print("\nTemporal ordering:")
print("PASSED")


# =============================================================================
# AUDIT EVERY FROZEN BUILDING
# =============================================================================

print("\n" + "=" * 100)
print("AUDITING FROZEN BUILDINGS")
print("=" * 100)

audit_rows = []

for i, row in cohort.iterrows():

    building_id = str(row["building_id"])
    site_id = str(row["site_id"])

    file = DATA_DIR / f"{building_id}.parquet"

    if not file.exists():
        raise FileNotFoundError(
            f"Missing building file: {file}"
        )

    df = pd.read_parquet(
        file,
        columns=[
            "timestamp",
            "temperature",
            "relative_humidity",
            "pressure",
            "energy",
            "temperature_observed",
            "relative_humidity_observed",
            "pressure_observed",
            "energy_observed",
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise RuntimeError(
            f"Invalid timestamp in {building_id}"
        )

    if df["timestamp"].duplicated().any():
        raise RuntimeError(
            f"Duplicate timestamps in {building_id}"
        )

    df = df.sort_values("timestamp")

    partition_specs = [
        (
            "train",
            TRAIN_START,
            TRAIN_END,
            EXPECTED_TRAIN
        ),
        (
            "validation",
            VAL_START,
            VAL_END,
            EXPECTED_VAL
        ),
        (
            "test",
            TEST_START,
            TEST_END,
            EXPECTED_TEST
        ),
    ]

    for (
        partition,
        start,
        end,
        expected
    ) in partition_specs:

        x = df[
            (df["timestamp"] >= start)
            &
            (df["timestamp"] <= end)
        ].copy()

        actual = len(x)

        if actual != expected:
            raise RuntimeError(
                f"{building_id}: {partition} "
                f"contains {actual} rows; "
                f"expected {expected}."
            )

        record = {
            "building_id": building_id,
            "site_id": site_id,
            "partition": partition,
            "start": start,
            "end": end,
            "expected_hours": expected,
            "actual_hours": actual,
        }

        for variable in [
            "temperature",
            "relative_humidity",
            "pressure",
            "energy",
        ]:

            record[
                f"{variable}_available_hours"
            ] = int(
                x[variable].notna().sum()
            )

            record[
                f"{variable}_available_percent"
            ] = float(
                x[variable].notna().mean()
                * 100
            )

            observed_col = (
                f"{variable}_observed"
            )

            record[
                f"{variable}_observed_hours"
            ] = int(
                x[observed_col].sum()
            )

            record[
                f"{variable}_observed_percent"
            ] = float(
                x[observed_col].mean()
                * 100
            )

        audit_rows.append(record)

    if (
        (i + 1) % 100 == 0
        or
        i + 1 == len(cohort)
    ):
        print(
            f"Audited "
            f"{i + 1}/{len(cohort)} "
            f"buildings"
        )


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


# =============================================================================
# CROSS-PARTITION CHECKS
# =============================================================================

print("\n" + "=" * 100)
print("CROSS-PARTITION CONSISTENCY")
print("=" * 100)

counts = (
    audit.groupby("partition")
    ["building_id"]
    .nunique()
)

print(counts.to_string())

assert counts["train"] == len(cohort)
assert counts["validation"] == len(cohort)
assert counts["test"] == len(cohort)

records_per_building = (
    audit.groupby("building_id")
    ["partition"]
    .nunique()
)

assert (
    records_per_building == 3
).all()

print("\nEvery frozen building has all three partitions:")
print("PASSED")


# =============================================================================
# PARTITION SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("PARTITION SUMMARY")
print("=" * 100)

summary_rows = []

for partition in [
    "train",
    "validation",
    "test",
]:

    x = audit[
        audit["partition"] == partition
    ]

    row = {
        "partition": partition,
        "buildings": x["building_id"].nunique(),
        "sites": x["site_id"].nunique(),
        "hours_per_building": int(
            x["actual_hours"].iloc[0]
        ),
        "total_building_hours": int(
            x["actual_hours"].sum()
        ),
    }

    for variable in [
        "temperature",
        "relative_humidity",
        "pressure",
        "energy",
    ]:

        row[
            f"{variable}_mean_available_percent"
        ] = float(
            x[
                f"{variable}_available_percent"
            ].mean()
        )

        row[
            f"{variable}_min_available_percent"
        ] = float(
            x[
                f"{variable}_available_percent"
            ].min()
        )

        row[
            f"{variable}_mean_observed_percent"
        ] = float(
            x[
                f"{variable}_observed_percent"
            ].mean()
        )

        row[
            f"{variable}_min_observed_percent"
        ] = float(
            x[
                f"{variable}_observed_percent"
            ].min()
        )

    summary_rows.append(row)


summary = pd.DataFrame(
    summary_rows
)

print(summary.to_string(index=False))


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {
    "dataset":
        "Building Data Genome Project 2",

    "cohort_file":
        "bdg2_frozen_source_cohort.csv",

    "number_of_buildings":
        int(len(cohort)),

    "number_of_sites":
        int(cohort["site_id"].nunique()),

    "partitions": {
        "train": {
            "start": str(TRAIN_START),
            "end": str(TRAIN_END),
            "hours_per_building":
                EXPECTED_TRAIN,
        },
        "validation": {
            "start": str(VAL_START),
            "end": str(VAL_END),
            "hours_per_building":
                EXPECTED_VAL,
        },
        "test": {
            "start": str(TEST_START),
            "end": str(TEST_END),
            "hours_per_building":
                EXPECTED_TEST,
        },
    },

    "partition_rule":
        "strict chronological fixed boundaries",

    "shuffle_before_split":
        False,

    "future_observations_used":
        False,

    "normalization_applied":
        False,

    "normalization_statistics_computed":
        False,

    "model_training_applied":
        False,

    "model_selection_applied":
        False,

    "target_domain_used":
        False,

    "forecasting_results_used":
        False,
}

with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# =============================================================================
# FINAL
# =============================================================================

print("\n" + "=" * 100)
print("BDG2 TEMPORAL PARTITIONS FROZEN")
print("=" * 100)

print("\nTrain:")
print(
    TRAIN_START,
    "->",
    TRAIN_END,
    f"({EXPECTED_TRAIN:,} h/building)"
)

print("\nValidation:")
print(
    VAL_START,
    "->",
    VAL_END,
    f"({EXPECTED_VAL:,} h/building)"
)

print("\nTest:")
print(
    TEST_START,
    "->",
    TEST_END,
    f"({EXPECTED_TEST:,} h/building)"
)

print("\nFrozen buildings:")
print(len(cohort))

print("\nFrozen sites:")
print(cohort["site_id"].nunique())

print("\nSites absent from original 19:")
print(missing_sites)

print("\nGenerated:")
print(
    " data/metadata/"
    "bdg2_temporal_partitions.csv"
)
print(
    " data/metadata/"
    "bdg2_temporal_partition_audit.csv"
)
print(
    " data/metadata/"
    "bdg2_temporal_partition_manifest.json"
)

print("""
IMPORTANT:

The temporal boundaries are now FROZEN.

The split is strictly chronological:

    training -> validation -> test

No random temporal sampling was used.

No validation or test observation has been used to
calculate normalization statistics.

No normalization has yet been performed.

No target-domain observation, forecasting result,
adaptation result, uncertainty estimate, or model
performance was used to define these partitions.

The next stage will calculate normalization statistics
from the TRAINING partition only.
""")