from pathlib import Path
import json
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "metadata"
BUILDING_DIR = ROOT / "data" / "final" / "bdg2" / "buildings"

COHORT_FILE = META / "bdg2_final_experimental_source_cohort.csv"

OUTPUT_STATS = META / "bdg2_training_normalization_statistics.csv"
OUTPUT_SITE_STATS = META / "bdg2_training_normalization_site_audit.csv"
OUTPUT_MANIFEST = META / "bdg2_training_normalization_manifest.json"

TRAIN_START = pd.Timestamp("2016-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2017-04-30 23:00:00")

SHARED_VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
]

PRIVATE_VARIABLE = "energy"

ALL_VARIABLES = SHARED_VARIABLES + [PRIVATE_VARIABLE]


# =============================================================================
# HELPERS
# =============================================================================

def find_timestamp_column(df):
    candidates = [
        "timestamp",
        "datetime",
        "date_time",
        "time",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        f"No timestamp column found. Columns: {list(df.columns)}"
    )


def find_variable_column(df, canonical_name):

    candidates = {
        "temperature": [
            "temperature",
            "airTemperature",
            "air_temperature",
        ],

        "relative_humidity": [
            "relative_humidity",
            "humidity",
            "rh",
        ],

        "pressure": [
            "pressure",
            "seaLvlPressure",
            "sea_level_pressure",
        ],

        "energy": [
            "energy",
            "electricity",
            "meter_reading",
        ],
    }

    for c in candidates[canonical_name]:
        if c in df.columns:
            return c

    raise RuntimeError(
        f"Cannot find column for {canonical_name}. "
        f"Columns available: {list(df.columns)}"
    )


def safe_statistics(values):

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        raise RuntimeError("No valid observations available.")

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "sample_std": float(values.std(ddof=1)),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q01": float(values.quantile(0.01)),
        "q05": float(values.quantile(0.05)),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "q95": float(values.quantile(0.95)),
        "q99": float(values.quantile(0.99)),
    }


# =============================================================================
# LOAD FROZEN COHORT
# =============================================================================

print("=" * 100)
print("BDG2 TRAINING-ONLY NORMALIZATION")
print("=" * 100)

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={
        "building_id": str,
        "site_id": str,
    }
)

print("\nFrozen buildings:")
print(len(cohort))

print("\nFrozen sites:")
print(cohort["site_id"].nunique())

print("\nTraining interval:")
print(f"{TRAIN_START} -> {TRAIN_END}")


# =============================================================================
# IMPORTANT DESIGN
# =============================================================================
#
# Shared weather variables are site-level observations.
#
# They MUST NOT be pooled once for every building because that would duplicate
# identical site weather observations according to the number of buildings at
# the site.
#
# Therefore:
#
# T/RH/P:
#     one copy of each site-hour
#
# Energy:
#     one copy of each building-hour
#
# This prevents sites with many buildings from receiving artificial weight
# when estimating shared-variable normalization parameters.
#
# =============================================================================


shared_values = {
    v: []
    for v in SHARED_VARIABLES
}

energy_values = []

site_audit_rows = []

processed_sites = set()


# =============================================================================
# PROCESS BUILDINGS
# =============================================================================

for i, row in cohort.iterrows():

    building_id = str(row["building_id"])
    site_id = str(row["site_id"])

    file = BUILDING_DIR / f"{building_id}.parquet"

    if not file.exists():
        raise FileNotFoundError(
            f"Missing building file: {file}"
        )

    df = pd.read_parquet(file)

    timestamp_col = find_timestamp_column(df)

    ts = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    train_mask = (
        (ts >= TRAIN_START)
        & (ts <= TRAIN_END)
    )

    train = df.loc[train_mask].copy()

    if len(train) == 0:
        raise RuntimeError(
            f"No training rows for building {building_id}"
        )


    # =========================================================================
    # BUILDING-LEVEL ENERGY
    # =========================================================================

    energy_col = find_variable_column(
        train,
        "energy"
    )

    e = pd.to_numeric(
        train[energy_col],
        errors="coerce"
    )

    energy_values.append(
        e.dropna().to_numpy(dtype=np.float64)
    )


    # =========================================================================
    # SITE-LEVEL SHARED VARIABLES
    # =========================================================================
    #
    # Process weather only once per site.
    #
    # =========================================================================

    if site_id not in processed_sites:

        processed_sites.add(site_id)

        audit = {
            "site_id": site_id,
            "representative_building_id": building_id,
            "training_rows": int(len(train)),
        }

        for variable in SHARED_VARIABLES:

            col = find_variable_column(
                train,
                variable
            )

            x = pd.to_numeric(
                train[col],
                errors="coerce"
            )

            valid = x.dropna()

            shared_values[variable].append(
                valid.to_numpy(dtype=np.float64)
            )

            audit[f"{variable}_n"] = int(len(valid))
            audit[f"{variable}_mean"] = (
                float(valid.mean())
                if len(valid)
                else np.nan
            )

            audit[f"{variable}_std"] = (
                float(valid.std(ddof=0))
                if len(valid)
                else np.nan
            )

        site_audit_rows.append(audit)


    if (i + 1) % 100 == 0 or (i + 1) == len(cohort):

        print(
            f"Processed "
            f"{i + 1:,}/{len(cohort):,} buildings"
        )


# =============================================================================
# CONCATENATE
# =============================================================================

print("\n" + "=" * 100)
print("CALCULATING TRAINING STATISTICS")
print("=" * 100)

combined = {}

for variable in SHARED_VARIABLES:

    if not shared_values[variable]:
        raise RuntimeError(
            f"No observations for {variable}"
        )

    combined[variable] = np.concatenate(
        shared_values[variable]
    )


if not energy_values:
    raise RuntimeError(
        "No energy observations collected."
    )

combined["energy"] = np.concatenate(
    energy_values
)


# =============================================================================
# GLOBAL TRAINING STATISTICS
# =============================================================================

rows = []

for variable in ALL_VARIABLES:

    stats = safe_statistics(
        combined[variable]
    )

    row = {
        "variable": variable,
        "normalization_method": "z_score",
        "normalization_mean": stats["mean"],
        "normalization_std": stats["std"],
        **stats,
    }

    rows.append(row)


stats_df = pd.DataFrame(rows)


# =============================================================================
# VALIDATION
# =============================================================================

if stats_df["normalization_mean"].isna().any():
    raise RuntimeError(
        "NaN normalization mean detected."
    )

if stats_df["normalization_std"].isna().any():
    raise RuntimeError(
        "NaN normalization standard deviation detected."
    )

if (
    stats_df["normalization_std"] <= 0
).any():
    raise RuntimeError(
        "Zero or negative normalization standard deviation."
    )


# =============================================================================
# SAVE
# =============================================================================

stats_df.to_csv(
    OUTPUT_STATS,
    index=False
)

site_audit_df = pd.DataFrame(
    site_audit_rows
)

site_audit_df.to_csv(
    OUTPUT_SITE_STATS,
    index=False
)


# =============================================================================
# PRINT
# =============================================================================

print("\nTRAINING-ONLY NORMALIZATION PARAMETERS\n")

display_cols = [
    "variable",
    "n",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "q01",
    "q99",
]

print(
    stats_df[display_cols]
    .to_string(index=False)
)


# =============================================================================
# NORMALIZATION SANITY CHECK
# =============================================================================

print("\n" + "=" * 100)
print("Z-SCORE SANITY CHECK")
print("=" * 100)

sanity_rows = []

for _, row in stats_df.iterrows():

    variable = row["variable"]

    x = combined[variable]

    z = (
        x - row["normalization_mean"]
    ) / row["normalization_std"]

    sanity_rows.append({
        "variable": variable,
        "normalized_mean": float(
            np.mean(z)
        ),
        "normalized_std": float(
            np.std(z, ddof=0)
        ),
        "normalized_min": float(
            np.min(z)
        ),
        "normalized_max": float(
            np.max(z)
        ),
    })


sanity_df = pd.DataFrame(
    sanity_rows
)

print(
    sanity_df.to_string(
        index=False
    )
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {

    "status":
        "frozen",

    "source_dataset":
        "BDG2",

    "experimental_buildings":
        int(len(cohort)),

    "experimental_sites":
        int(cohort["site_id"].nunique()),

    "training_start":
        str(TRAIN_START),

    "training_end":
        str(TRAIN_END),

    "normalization_method":
        "z_score",

    "normalization_formula":
        "z = (x - training_mean) / training_population_std",

    "std_ddof":
        0,

    "shared_variables":
        SHARED_VARIABLES,

    "source_private_variable":
        PRIVATE_VARIABLE,

    "shared_variable_weighting":
        (
            "Each site-hour appears once when estimating "
            "temperature, relative-humidity, and pressure "
            "normalization statistics."
        ),

    "energy_weighting":
        (
            "Each valid building-hour appears once when "
            "estimating source-private energy statistics."
        ),

    "statistics_source":
        "BDG2 training partition only",

    "validation_used_for_statistics":
        False,

    "test_used_for_statistics":
        False,

    "target_domain_used_for_statistics":
        False,

    "future_target_observations_used":
        False,

    "raw_physical_unit_files_modified":
        False,
}


with open(
    OUTPUT_MANIFEST,
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
print("BDG2 TRAINING NORMALIZATION FROZEN")
print("=" * 100)

print("\nGenerated:")

print(
    " data/metadata/"
    "bdg2_training_normalization_statistics.csv"
)

print(
    " data/metadata/"
    "bdg2_training_normalization_site_audit.csv"
)

print(
    " data/metadata/"
    "bdg2_training_normalization_manifest.json"
)

print("""
IMPORTANT:

Normalization statistics were estimated exclusively from
the frozen BDG2 TRAINING partition.

Temperature, relative humidity, and pressure were counted
once per site-hour rather than duplicated for every building.

Energy was counted once per valid building-hour.

Validation and test observations did not contribute to
normalization parameters.

No openSenseMap target-domain observation contributed to
these source statistics.

The original harmonized Parquet files remain in physical
units and were not modified.

The frozen mean and standard deviation must now be reused
without refitting on BDG2 validation/test data.
""")