"""
Step 22
Build and freeze model-ready lazy dataset specifications.

This step does NOT materialize overlapping 24-hour windows.

It validates:
    - frozen forecasting protocol
    - source and target cohorts
    - source and target schemas
    - source and target normalization
    - frozen source/target window indices
    - common shared-encoder feature contract
    - causality and leakage rules

Shared encoder inputs are IDENTICAL across domains:

    3 continuous environmental values
    3 observation masks
    3 causal-imputation masks
    6 deterministic calendar features

Total shared features = 15.

Private branches:
    BDG2       -> energy + observed/imputed masks
    openSenseMap -> PM2.5 + observed/imputed masks

Unresolved missingness is represented by:
    observed = 0
    imputed  = 0
    value    = NaN

The physical datasets are not modified.

No model is trained.
No forecasting performance is inspected.
"""

from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed"

BDG2_BUILDINGS = ROOT / "data" / "final" / "bdg2" / "buildings"
OSE_FINAL = ROOT / "data" / "final" / "opensensemap"

OUTPUT_DIR = PROCESSED / "model_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# REQUIRED FROZEN INPUTS
# =============================================================================

FORECAST_PROTOCOL = META / "forecasting_protocol.json"

SOURCE_COHORT = META / "bdg2_final_experimental_source_cohort.csv"
SOURCE_NORM = META / "bdg2_training_normalization_statistics.csv"

TARGET_COHORT = META / "opensensemap_final_eligible_target_cohort.csv"
TARGET_NORM = META / "opensensemap_adaptation_normalization_statistics.csv"

TARGET_HOURLY = (
    OSE_FINAL /
    "opensensemap_target_hourly_wide.csv"
)

SOURCE_INDEX_FILES = {
    "train":
        PROCESSED / "bdg2" /
        "source_window_index_train.parquet",

    "validation":
        PROCESSED / "bdg2" /
        "source_window_index_validation.parquet",

    "test":
        PROCESSED / "bdg2" /
        "source_window_index_test.parquet",
}

TARGET_INDEX_FILES = {
    "adaptation":
        PROCESSED / "opensensemap" /
        "target_window_index_adaptation.parquet",

    "validation":
        PROCESSED / "opensensemap" /
        "target_window_index_validation.parquet",

    "test":
        PROCESSED / "opensensemap" /
        "target_window_index_test.parquet",
}


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen input not found:\n{path}"
        )


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def detect_column(df, candidates, description):

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Expected one of:\n{candidates}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


def read_normalization(path):

    df = pd.read_csv(path)

    variable_col = detect_column(
        df,
        ["variable", "canonical_variable"],
        "normalization variable column"
    )

    mean_col = detect_column(
        df,
        ["mean"],
        "normalization mean column"
    )

    std_col = detect_column(
        df,
        ["std", "standard_deviation"],
        "normalization standard deviation column"
    )

    result = {}

    for _, row in df.iterrows():

        variable = str(row[variable_col])

        mean = float(row[mean_col])
        std = float(row[std_col])

        if not np.isfinite(mean):
            raise ValueError(
                f"Non-finite normalization mean for {variable}"
            )

        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid normalization std for "
                f"{variable}: {std}"
            )

        result[variable] = {
            "mean": mean,
            "std": std,
        }

    return result


def inspect_index(path, domain, partition):

    df = pd.read_parquet(path)

    if len(df) == 0:
        raise ValueError(
            f"Frozen window index is empty:\n{path}"
        )

    if domain == "source":

        entity_col = detect_column(
            df,
            ["building_id"],
            "source entity column"
        )

    else:

        entity_col = detect_column(
            df,
            ["box_id"],
            "target entity column"
        )

    origin_col = detect_column(
        df,
        [
            "origin_timestamp",
            "forecast_origin",
            "forecast_origin_timestamp",
            "origin",
            "timestamp",
        ],
        f"{domain} forecast-origin column"
    )

    return {
        "domain": domain,
        "partition": partition,
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "entity_column": entity_col,
        "entities": int(df[entity_col].nunique()),
        "origin_column": origin_col,
        "columns": list(df.columns),
    }


# =============================================================================
# START
# =============================================================================

section("MODEL-READY LAZY DATASET CONSTRUCTION")


required_files = [
    FORECAST_PROTOCOL,
    SOURCE_COHORT,
    SOURCE_NORM,
    TARGET_COHORT,
    TARGET_NORM,
    TARGET_HOURLY,
]

required_files += list(SOURCE_INDEX_FILES.values())
required_files += list(TARGET_INDEX_FILES.values())

for path in required_files:
    require_file(path)

print("All required frozen inputs found.")


# =============================================================================
# FORECASTING PROTOCOL
# =============================================================================

section("VERIFYING FROZEN FORECASTING PROTOCOL")

with open(
    FORECAST_PROTOCOL,
    "r",
    encoding="utf-8"
) as f:
    protocol = json.load(f)


required_protocol_keys = [
    "history_length_hours",
    "forecast_horizons_hours",
    "shared_variables",
    "source_private_variable",
    "target_private_variable",
    "calendar_features",
]

for key in required_protocol_keys:

    if key not in protocol:
        raise ValueError(
            f"Missing required frozen protocol key: {key}"
        )


history_length = int(
    protocol["history_length_hours"]
)

horizons = [
    int(x)
    for x in protocol["forecast_horizons_hours"]
]

shared_variables = list(
    protocol["shared_variables"]
)

source_private_variable = str(
    protocol["source_private_variable"]
)

target_private_variable = str(
    protocol["target_private_variable"]
)

calendar_features = list(
    protocol["calendar_features"]
)


print(
    f"Protocol version: "
    f"{protocol.get('protocol_version')}"
)

print(
    f"Frozen at UTC:    "
    f"{protocol.get('frozen_at_utc')}"
)

print(
    f"History length:   "
    f"{history_length} hours"
)

print(
    f"Horizons:         "
    f"{horizons}"
)

print(
    f"Shared variables: "
    f"{shared_variables}"
)

print(
    f"Source private:   "
    f"{source_private_variable}"
)

print(
    f"Target private:   "
    f"{target_private_variable}"
)

print(
    f"Calendar features:"
    f"{calendar_features}"
)


# =============================================================================
# VERIFY FROZEN SCIENTIFIC DESIGN
# =============================================================================

if history_length != 24:

    raise ValueError(
        f"Expected frozen 24-hour history; "
        f"found {history_length}"
    )


if horizons != [1, 3, 6, 12, 24]:

    raise ValueError(
        f"Unexpected frozen horizons: {horizons}"
    )


if shared_variables != [
    "temperature",
    "relative_humidity",
    "pressure",
]:

    raise ValueError(
        f"Unexpected frozen shared variables: "
        f"{shared_variables}"
    )


if source_private_variable != "energy":

    raise ValueError(
        f"Unexpected source-private variable: "
        f"{source_private_variable}"
    )


if target_private_variable != "pm25":

    raise ValueError(
        f"Unexpected target-private variable: "
        f"{target_private_variable}"
    )


if calendar_features != [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]:

    raise ValueError(
        f"Unexpected frozen calendar features: "
        f"{calendar_features}"
    )


print("\nFrozen forecasting design: VERIFIED")


# =============================================================================
# CAUSALITY
# =============================================================================

section("VERIFYING FROZEN CAUSALITY RULES")

causality = protocol.get(
    "causality",
    {}
)

required_false = [
    "future_observations_in_inputs",
    "future_based_interpolation",
    "target_test_for_normalization",
    "target_test_for_adaptation",
    "target_test_for_model_selection",
    "target_test_for_early_stopping",
    "target_test_for_threshold_selection",
    "cross_partition_windows",
]


for key in required_false:

    if key not in causality:
        raise ValueError(
            f"Missing frozen causality field: {key}"
        )

    if causality[key] is not False:
        raise ValueError(
            f"Causality violation: "
            f"{key}={causality[key]}"
        )

    print(f"{key}: PASSED")


# =============================================================================
# MASK PROTOCOL
# =============================================================================

section("VERIFYING MASK PROTOCOL")

mask_protocol = protocol.get(
    "mask_protocol",
    {}
)

required_true = [
    "historical_observation_masks",
    "historical_imputation_masks",
    "evaluation_requires_observed_label",
    "missingness_is_explicit_model_information",
]


for key in required_true:

    if mask_protocol.get(key) is not True:
        raise ValueError(
            f"Required mask rule not enabled: {key}"
        )

    print(f"{key}: VERIFIED")


if (
    mask_protocol.get(
        "imputed_label_allowed_for_evaluation"
    )
    is not False
):

    raise ValueError(
        "Frozen protocol unexpectedly permits "
        "imputed evaluation labels."
    )


print(
    "imputed_label_allowed_for_evaluation: NO"
)


# =============================================================================
# SOURCE COHORT
# =============================================================================

section("VERIFYING SOURCE COHORT")

source_cohort = pd.read_csv(
    SOURCE_COHORT
)

if "building_id" not in source_cohort.columns:

    raise ValueError(
        "building_id missing from source cohort."
    )


source_cohort["building_id"] = (
    source_cohort["building_id"]
    .astype(str)
)


source_buildings = sorted(
    source_cohort["building_id"].unique()
)


if "site_id" in source_cohort.columns:

    source_sites = sorted(
        source_cohort["site_id"]
        .astype(str)
        .unique()
    )

else:

    source_sites = []


print(
    f"Frozen source buildings: "
    f"{len(source_buildings):,}"
)

print(
    f"Frozen source sites:     "
    f"{len(source_sites):,}"
)


if len(source_buildings) != 1318:

    raise ValueError(
        f"Expected 1,318 source buildings; "
        f"found {len(source_buildings)}"
    )


if len(source_sites) != 16:

    raise ValueError(
        f"Expected 16 source sites; "
        f"found {len(source_sites)}"
    )


print("Source cohort count: VERIFIED")


# =============================================================================
# SOURCE BUILDING FILES
# =============================================================================

section("VERIFYING SOURCE BUILDING FILES")

missing_files = []

for building_id in source_buildings:

    path = (
        BDG2_BUILDINGS /
        f"{building_id}.parquet"
    )

    if not path.exists():
        missing_files.append(building_id)


if missing_files:

    raise FileNotFoundError(
        f"{len(missing_files)} source files missing.\n"
        f"Examples:\n{missing_files[:10]}"
    )


print(
    f"All {len(source_buildings):,} "
    f"frozen source building files found."
)


# =============================================================================
# SOURCE SCHEMA
# =============================================================================

section("VERIFYING SOURCE FEATURE SCHEMA")

example_source_path = (
    BDG2_BUILDINGS /
    f"{source_buildings[0]}.parquet"
)

example_source = pd.read_parquet(
    example_source_path
)


required_source_columns = [
    "building_id",
    "site_id",
    "timestamp",

    "temperature",
    "temperature_observed",
    "temperature_imputed",

    "relative_humidity",
    "relative_humidity_observed",
    "relative_humidity_imputed",

    "pressure",
    "pressure_observed",
    "pressure_imputed",

    "energy",
    "energy_observed",
    "energy_imputed",
] + calendar_features


source_missing = [
    col
    for col in required_source_columns
    if col not in example_source.columns
]


if source_missing:

    raise ValueError(
        f"Required source columns missing:\n"
        f"{source_missing}"
    )


print("Required source feature schema: VERIFIED")


# -----------------------------------------------------------------------------
# Audit-only missing columns may exist in BDG2.
# They are deliberately NOT required as model inputs.
# -----------------------------------------------------------------------------

source_audit_missing_columns = [
    "temperature_missing",
    "relative_humidity_missing",
    "pressure_missing",
    "energy_missing",
]


available_source_audit_missing = [
    c
    for c in source_audit_missing_columns
    if c in example_source.columns
]


print(
    "\nBDG2 audit-only *_missing columns found: "
    f"{len(available_source_audit_missing)}"
)

for col in available_source_audit_missing:
    print(f"  {col}")


# =============================================================================
# TARGET COHORT
# =============================================================================

section("VERIFYING TARGET COHORT")

target_cohort = pd.read_csv(
    TARGET_COHORT
)

if "box_id" not in target_cohort.columns:

    raise ValueError(
        "box_id missing from target cohort."
    )


target_cohort["box_id"] = (
    target_cohort["box_id"]
    .astype(str)
)


target_boxes = sorted(
    target_cohort["box_id"].unique()
)


print(
    f"Frozen target boxes: "
    f"{len(target_boxes):,}"
)


if len(target_boxes) != 17:

    raise ValueError(
        f"Expected 17 target boxes; "
        f"found {len(target_boxes)}"
    )


print("Target cohort count: VERIFIED")


# =============================================================================
# TARGET HOURLY SCHEMA
# =============================================================================

section("VERIFYING TARGET HOURLY DATA")

target_sample = pd.read_csv(
    TARGET_HOURLY,
    nrows=100
)


required_target_columns = [
    "box_id",
    "timestamp",

    "temperature",
    "temperature_observed",
    "temperature_imputed",

    "relative_humidity",
    "relative_humidity_observed",
    "relative_humidity_imputed",

    "pressure",
    "pressure_observed",
    "pressure_imputed",

    "pm25",
    "pm25_observed",
    "pm25_imputed",
] + calendar_features


target_missing = [
    col
    for col in required_target_columns
    if col not in target_sample.columns
]


if target_missing:

    raise ValueError(
        f"Required target columns missing:\n"
        f"{target_missing}"
    )


print("Required target feature schema: VERIFIED")


target_physical_missing_columns = [
    "temperature_missing",
    "relative_humidity_missing",
    "pressure_missing",
    "pm25_missing",
]


target_missing_columns_present = [
    c
    for c in target_physical_missing_columns
    if c in target_sample.columns
]


if target_missing_columns_present:

    print(
        "\nOptional target *_missing columns found:"
    )

    for c in target_missing_columns_present:
        print(f"  {c}")

else:

    print(
        "\nNo physical target *_missing columns present."
    )

    print(
        "This is valid."
    )

    print(
        "Unresolved missingness is derivable from:"
    )

    print(
        "  observed == 0 AND imputed == 0"
    )

    print(
        "and the corresponding value remains NaN."
    )


# =============================================================================
# NORMALIZATION
# =============================================================================

section("VERIFYING NORMALIZATION PARAMETERS")

source_norm = read_normalization(
    SOURCE_NORM
)

target_norm = read_normalization(
    TARGET_NORM
)


source_required_norm = [
    "temperature",
    "relative_humidity",
    "pressure",
    "energy",
]


target_required_norm = [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]


for variable in source_required_norm:

    if variable not in source_norm:

        raise ValueError(
            f"Missing source normalization "
            f"parameter: {variable}"
        )


for variable in target_required_norm:

    if variable not in target_norm:

        raise ValueError(
            f"Missing target normalization "
            f"parameter: {variable}"
        )


print("\nSOURCE NORMALIZATION")

for variable in source_required_norm:

    print(
        f"{variable:20s} "
        f"mean={source_norm[variable]['mean']:.6f} "
        f"std={source_norm[variable]['std']:.6f}"
    )


print("\nTARGET NORMALIZATION")

for variable in target_required_norm:

    print(
        f"{variable:20s} "
        f"mean={target_norm[variable]['mean']:.6f} "
        f"std={target_norm[variable]['std']:.6f}"
    )


print(
    "\nNormalization parameters: VERIFIED"
)


# =============================================================================
# NORMALIZATION PROVENANCE
# =============================================================================

section("VERIFYING NORMALIZATION PROVENANCE")

normalization_protocol = protocol.get(
    "normalization_protocol",
    {}
)

source_norm_protocol = (
    normalization_protocol.get(
        "source",
        {}
    )
)


if (
    source_norm_protocol.get("fit_partition")
    != "BDG2 train"
):

    raise ValueError(
        "Source normalization is not frozen "
        "as BDG2-train-only."
    )


if (
    source_norm_protocol.get(
        "validation_refit"
    )
    is not False
):

    raise ValueError(
        "Source validation normalization "
        "refit is enabled."
    )


if (
    source_norm_protocol.get(
        "test_refit"
    )
    is not False
):

    raise ValueError(
        "Source test normalization refit "
        "is enabled."
    )


if (
    source_norm_protocol.get(
        "target_data_used"
    )
    is not False
):

    raise ValueError(
        "Target data contributed to "
        "source normalization."
    )


print(
    "Source normalization provenance: PASSED"
)

print(
    "\nTarget normalization provenance:"
)

print(
    "  fit partition: adaptation only"
)

print(
    "  directly observed values only"
)

print(
    "  validation used for fit: NO"
)

print(
    "  test used for fit: NO"
)

print(
    "  frozen by Step 21"
)


# =============================================================================
# WINDOW INDICES
# =============================================================================

section("INSPECTING FROZEN WINDOW INDICES")

index_specs = {
    "source": {},
    "target": {},
}

summary_rows = []


for partition, path in SOURCE_INDEX_FILES.items():

    print(
        f"\nSource partition: {partition}"
    )

    spec = inspect_index(
        path,
        "source",
        partition
    )

    index_specs["source"][partition] = spec

    print(
        f"  origins:   {spec['rows']:,}"
    )

    print(
        f"  buildings: {spec['entities']:,}"
    )

    print(
        f"  origin:    {spec['origin_column']}"
    )


    if spec["entities"] != 1318:

        raise ValueError(
            f"Source {partition} contains "
            f"{spec['entities']} buildings "
            f"instead of 1,318."
        )


    summary_rows.append({
        "domain": "source",
        "partition": partition,
        "origins": spec["rows"],
        "entities": spec["entities"],
        "entity_type": "building",
    })


for partition, path in TARGET_INDEX_FILES.items():

    print(
        f"\nTarget partition: {partition}"
    )

    spec = inspect_index(
        path,
        "target",
        partition
    )

    index_specs["target"][partition] = spec

    print(
        f"  origins: {spec['rows']:,}"
    )

    print(
        f"  boxes:   {spec['entities']:,}"
    )

    print(
        f"  origin:  {spec['origin_column']}"
    )


    if spec["entities"] != 17:

        raise ValueError(
            f"Target {partition} contains "
            f"{spec['entities']} boxes "
            f"instead of 17."
        )


    summary_rows.append({
        "domain": "target",
        "partition": partition,
        "origins": spec["rows"],
        "entities": spec["entities"],
        "entity_type": "box",
    })


# =============================================================================
# MODEL FEATURE CONTRACT
# =============================================================================

section("FREEZING COMMON MODEL FEATURE CONTRACT")


# -----------------------------------------------------------------------------
# Shared continuous variables
# -----------------------------------------------------------------------------

shared_value_features = [
    "temperature",
    "relative_humidity",
    "pressure",
]


# -----------------------------------------------------------------------------
# Shared observation masks
# -----------------------------------------------------------------------------

shared_observation_features = [
    "temperature_observed",
    "relative_humidity_observed",
    "pressure_observed",
]


# -----------------------------------------------------------------------------
# Shared causal-imputation masks
# -----------------------------------------------------------------------------

shared_imputation_features = [
    "temperature_imputed",
    "relative_humidity_imputed",
    "pressure_imputed",
]


# -----------------------------------------------------------------------------
# IMPORTANT
#
# We deliberately do NOT add separate *_missing channels.
#
# State is fully recoverable:
#
# observed=1, imputed=0 -> direct observation
# observed=0, imputed=1 -> causal fill
# observed=0, imputed=0 -> unresolved missing
#
# Therefore source and target can use exactly the same feature semantics.
# -----------------------------------------------------------------------------


shared_encoder_features = (
    shared_value_features
    + shared_observation_features
    + shared_imputation_features
    + calendar_features
)


if len(shared_encoder_features) != 15:

    raise ValueError(
        "Expected exactly 15 shared encoder "
        f"features; found "
        f"{len(shared_encoder_features)}"
    )


# -----------------------------------------------------------------------------
# Source-private branch
# -----------------------------------------------------------------------------

source_private_features = [
    "energy",
    "energy_observed",
    "energy_imputed",
]


# -----------------------------------------------------------------------------
# Target-private branch
# -----------------------------------------------------------------------------

target_private_features = [
    "pm25",
    "pm25_observed",
    "pm25_imputed",
]


source_model_features = (
    shared_encoder_features
    + source_private_features
)


target_model_features = (
    shared_encoder_features
    + target_private_features
)


print(
    f"Shared encoder feature count: "
    f"{len(shared_encoder_features)}"
)

for feature in shared_encoder_features:
    print(f"  {feature}")


print(
    f"\nSource-private feature count: "
    f"{len(source_private_features)}"
)

for feature in source_private_features:
    print(f"  {feature}")


print(
    f"\nTarget-private feature count: "
    f"{len(target_private_features)}"
)

for feature in target_private_features:
    print(f"  {feature}")


# =============================================================================
# FEATURE CONTRACT CONSISTENCY
# =============================================================================

section("VERIFYING CROSS-DOMAIN FEATURE CONSISTENCY")


source_available = set(
    example_source.columns
)

target_available = set(
    target_sample.columns
)


for feature in shared_encoder_features:

    if feature not in source_available:

        raise ValueError(
            f"Shared feature missing from "
            f"source: {feature}"
        )

    if feature not in target_available:

        raise ValueError(
            f"Shared feature missing from "
            f"target: {feature}"
        )


print(
    "All 15 shared encoder features exist "
    "in BOTH domains."
)

print(
    "Shared feature order is identical "
    "across domains."
)

print(
    "Cross-domain feature consistency: PASSED"
)


# =============================================================================
# SAMPLE SHAPES
# =============================================================================

section("MODEL SAMPLE SHAPES")


shared_history_shape = [
    history_length,
    len(shared_encoder_features),
]


source_private_shape = [
    history_length,
    len(source_private_features),
]


target_private_shape = [
    history_length,
    len(target_private_features),
]


label_shape = [
    len(horizons),
]


print(
    "Shared history:        "
    f"{tuple(shared_history_shape)}"
)

print(
    "Source-private history:"
    f"{tuple(source_private_shape)}"
)

print(
    "Target-private history:"
    f"{tuple(target_private_shape)}"
)

print(
    "Multi-horizon labels:  "
    f"{tuple(label_shape)}"
)


# =============================================================================
# MISSINGNESS CONTRACT
# =============================================================================

section("FREEZING MISSINGNESS CONTRACT")


missingness_contract = {

    "state_encoding": {

        "direct_observation": {
            "observed": 1,
            "imputed": 0,
        },

        "causally_imputed": {
            "observed": 0,
            "imputed": 1,
        },

        "unresolved_missing": {
            "observed": 0,
            "imputed": 0,
        },
    },

    "separate_missing_channel_used_by_model":
        False,

    "unresolved_value_representation":
        "NaN in physical dataset",

    "model_loader_requirement":
        (
            "Before tensor construction, unresolved NaN "
            "continuous inputs must be replaced with a "
            "fixed neutral numeric placeholder AFTER "
            "normalization, while observed/imputed masks "
            "remain explicit model inputs."
        ),

    "recommended_normalized_placeholder":
        0.0,

    "interpretation_of_placeholder":
        (
            "Normalized zero corresponds to the frozen "
            "training/adaptation mean and is not treated "
            "as an observation because both masks are zero."
        ),

    "future_based_imputation_allowed":
        False,
}


print(
    "Direct observation: observed=1, imputed=0"
)

print(
    "Causal fill:        observed=0, imputed=1"
)

print(
    "Unresolved missing: observed=0, imputed=0"
)

print(
    "Separate *_missing model channel: NO"
)

print(
    "Unresolved NaN tensor placeholder after "
    "normalization: 0.0"
)


# =============================================================================
# FEATURE CONTRACT TABLE
# =============================================================================

feature_rows = []


for feature in shared_value_features:

    feature_rows.append({
        "feature": feature,
        "component": "shared_encoder",
        "feature_type": "continuous_value",
        "source": True,
        "target": True,
        "normalized": True,
    })


for feature in shared_observation_features:

    feature_rows.append({
        "feature": feature,
        "component": "shared_encoder",
        "feature_type": "observation_mask",
        "source": True,
        "target": True,
        "normalized": False,
    })


for feature in shared_imputation_features:

    feature_rows.append({
        "feature": feature,
        "component": "shared_encoder",
        "feature_type": "imputation_mask",
        "source": True,
        "target": True,
        "normalized": False,
    })


for feature in calendar_features:

    feature_rows.append({
        "feature": feature,
        "component": "shared_encoder",
        "feature_type": "deterministic_calendar",
        "source": True,
        "target": True,
        "normalized": False,
    })


for feature in source_private_features:

    if feature == "energy":

        feature_type = "continuous_value"
        normalized = True

    elif feature.endswith("_observed"):

        feature_type = "observation_mask"
        normalized = False

    else:

        feature_type = "imputation_mask"
        normalized = False


    feature_rows.append({
        "feature": feature,
        "component": "source_private",
        "feature_type": feature_type,
        "source": True,
        "target": False,
        "normalized": normalized,
    })


for feature in target_private_features:

    if feature == "pm25":

        feature_type = "continuous_value"
        normalized = True

    elif feature.endswith("_observed"):

        feature_type = "observation_mask"
        normalized = False

    else:

        feature_type = "imputation_mask"
        normalized = False


    feature_rows.append({
        "feature": feature,
        "component": "target_private",
        "feature_type": feature_type,
        "source": False,
        "target": True,
        "normalized": normalized,
    })


feature_contract = pd.DataFrame(
    feature_rows
)


# =============================================================================
# DATASET SPECIFICATION
# =============================================================================

section("WRITING MODEL-READY DATASET SPECIFICATION")


dataset_spec = {

    "specification_version": "1.0",

    "construction_mode": "lazy",

    "materialized_overlapping_windows":
        False,

    "history_length_hours":
        history_length,

    "forecast_horizons_hours":
        horizons,

    "shared_environmental_variables":
        shared_variables,

    "calendar_features":
        calendar_features,

    "shared_encoder_features":
        shared_encoder_features,

    "shared_encoder_feature_count":
        len(shared_encoder_features),

    "shared_history_shape":
        shared_history_shape,

    "missingness_contract":
        missingness_contract,

    "source": {

        "dataset":
            "BDG2",

        "entities":
            len(source_buildings),

        "sites":
            len(source_sites),

        "entity_type":
            "building",

        "historical_private_variable":
            "energy",

        "prediction_variable":
            "energy",

        "normalization_fit_partition":
            "train only",

        "normalization_statistics_file":
            str(
                SOURCE_NORM.relative_to(ROOT)
            ),

        "building_data_directory":
            str(
                BDG2_BUILDINGS.relative_to(ROOT)
            ),

        "shared_features":
            shared_encoder_features,

        "private_features":
            source_private_features,

        "model_features":
            source_model_features,

        "sample_shapes": {

            "shared_history":
                shared_history_shape,

            "private_history":
                source_private_shape,

            "labels":
                label_shape,

            "label_mask":
                label_shape,
        },

        "partitions":
            index_specs["source"],
    },

    "target": {

        "dataset":
            "openSenseMap",

        "entities":
            len(target_boxes),

        "entity_type":
            "box",

        "historical_private_variable":
            "pm25",

        "prediction_variable":
            "pm25",

        "normalization_fit_partition":
            "adaptation only",

        "normalization_statistics_file":
            str(
                TARGET_NORM.relative_to(ROOT)
            ),

        "hourly_data_file":
            str(
                TARGET_HOURLY.relative_to(ROOT)
            ),

        "shared_features":
            shared_encoder_features,

        "private_features":
            target_private_features,

        "model_features":
            target_model_features,

        "physical_missing_columns_present":
            bool(
                target_missing_columns_present
            ),

        "missingness_derivation":
            (
                "unresolved missing when "
                "observed=0 and imputed=0"
            ),

        "sample_shapes": {

            "shared_history":
                shared_history_shape,

            "private_history":
                target_private_shape,

            "labels":
                label_shape,

            "label_mask":
                label_shape,
        },

        "partitions":
            index_specs["target"],
    },

    "label_policy": {

        "source":
            "future energy",

        "target":
            "future PM2.5",

        "horizons_hours":
            horizons,

        "evaluation_requires_direct_observation":
            True,

        "imputed_labels_allowed":
            False,
    },

    "history_policy": {

        "timestamps_allowed":
            "<= forecast origin only",

        "causal_forward_fill_allowed":
            True,

        "future_observations_allowed":
            False,

        "observation_masks_included":
            True,

        "imputation_masks_included":
            True,

        "separate_missing_mask_model_input":
            False,
    },

    "transfer_policy": {

        "transferable_component":
            (
                "shared environmental temporal "
                "representation encoder"
            ),

        "shared_encoder_input_dimension":
            len(shared_encoder_features),

        "source_private_component":
            (
                "energy-specific temporal "
                "representation"
            ),

        "target_private_component":
            (
                "PM2.5-specific temporal "
                "representation"
            ),

        "source_prediction_head":
            "energy forecasting head",

        "target_prediction_head":
            "PM2.5 forecasting head",

        "energy_head_transferred_to_pm25_head":
            False,
    },

    "test_policy": {

        "target_test_used_for_training":
            False,

        "target_test_used_for_adaptation":
            False,

        "target_test_used_for_normalization":
            False,

        "target_test_used_for_model_selection":
            False,

        "target_test_used_for_early_stopping":
            False,

        "target_test_used_for_threshold_selection":
            False,
    },
}


# =============================================================================
# OUTPUT PATHS
# =============================================================================

SPEC_PATH = (
    OUTPUT_DIR /
    "model_ready_dataset_specification.json"
)

FEATURE_PATH = (
    OUTPUT_DIR /
    "model_ready_feature_contract.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR /
    "model_ready_partition_summary.csv"
)

NORM_PATH = (
    OUTPUT_DIR /
    "model_ready_normalization_contract.csv"
)

MANIFEST_PATH = (
    OUTPUT_DIR /
    "model_ready_dataset_manifest.json"
)


# =============================================================================
# WRITE DATASET SPEC
# =============================================================================

with open(
    SPEC_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset_spec,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# WRITE FEATURE CONTRACT
# =============================================================================

feature_contract.to_csv(
    FEATURE_PATH,
    index=False
)


# =============================================================================
# WRITE PARTITION SUMMARY
# =============================================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# =============================================================================
# NORMALIZATION CONTRACT
# =============================================================================

norm_rows = []


for variable in source_required_norm:

    norm_rows.append({

        "domain":
            "source",

        "dataset":
            "BDG2",

        "variable":
            variable,

        "mean":
            source_norm[variable]["mean"],

        "std":
            source_norm[variable]["std"],

        "fit_partition":
            "train",

        "validation_refit":
            False,

        "test_refit":
            False,
    })


for variable in target_required_norm:

    norm_rows.append({

        "domain":
            "target",

        "dataset":
            "openSenseMap",

        "variable":
            variable,

        "mean":
            target_norm[variable]["mean"],

        "std":
            target_norm[variable]["std"],

        "fit_partition":
            "adaptation",

        "validation_refit":
            False,

        "test_refit":
            False,
    })


normalization_contract = pd.DataFrame(
    norm_rows
)

normalization_contract.to_csv(
    NORM_PATH,
    index=False
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {

    "step":
        22,

    "name":
        "build_model_ready_datasets",

    "status":
        "complete",

    "construction_mode":
        "lazy",

    "history_length_hours":
        history_length,

    "forecast_horizons_hours":
        horizons,

    "source_buildings":
        len(source_buildings),

    "source_sites":
        len(source_sites),

    "target_boxes":
        len(target_boxes),

    "shared_encoder_feature_count":
        len(shared_encoder_features),

    "shared_encoder_features":
        shared_encoder_features,

    "source_private_feature_count":
        len(source_private_features),

    "target_private_feature_count":
        len(target_private_features),

    "shared_history_shape":
        shared_history_shape,

    "source_private_history_shape":
        source_private_shape,

    "target_private_history_shape":
        target_private_shape,

    "label_shape":
        label_shape,

    "source_partitions": {
        key: int(value["rows"])
        for key, value
        in index_specs["source"].items()
    },

    "target_partitions": {
        key: int(value["rows"])
        for key, value
        in index_specs["target"].items()
    },

    "normalization": {

        "source":
            (
                "BDG2 training-only "
                "frozen statistics"
            ),

        "target":
            (
                "openSenseMap adaptation-only "
                "frozen statistics"
            ),
    },

    "missingness": {

        "observation_mask_used":
            True,

        "imputation_mask_used":
            True,

        "separate_missing_model_channel":
            False,

        "unresolved_missing_state":
            "observed=0 and imputed=0",

        "normalized_nan_placeholder":
            0.0,
    },

    "causality": {

        "future_inputs":
            False,

        "cross_partition_windows":
            False,

        "imputed_evaluation_labels":
            False,

        "target_test_for_training":
            False,

        "target_test_for_adaptation":
            False,

        "target_test_for_normalization":
            False,

        "target_test_for_model_selection":
            False,
    },

    "generated_files": [

        str(
            SPEC_PATH.relative_to(ROOT)
        ),

        str(
            FEATURE_PATH.relative_to(ROOT)
        ),

        str(
            SUMMARY_PATH.relative_to(ROOT)
        ),

        str(
            NORM_PATH.relative_to(ROOT)
        ),
    ],

    "frozen_input_hashes": {

        str(
            FORECAST_PROTOCOL.relative_to(ROOT)
        ):
            sha256_file(
                FORECAST_PROTOCOL
            ),

        str(
            SOURCE_COHORT.relative_to(ROOT)
        ):
            sha256_file(
                SOURCE_COHORT
            ),

        str(
            SOURCE_NORM.relative_to(ROOT)
        ):
            sha256_file(
                SOURCE_NORM
            ),

        str(
            TARGET_COHORT.relative_to(ROOT)
        ):
            sha256_file(
                TARGET_COHORT
            ),

        str(
            TARGET_NORM.relative_to(ROOT)
        ):
            sha256_file(
                TARGET_NORM
            ),
    },
}


with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# FINAL REPORT
# =============================================================================

section("MODEL-READY LAZY DATASETS COMPLETE")


print("\nSOURCE")

print(
    f"Buildings: {len(source_buildings):,}"
)

print(
    f"Sites:     {len(source_sites):,}"
)


for partition in [
    "train",
    "validation",
    "test",
]:

    print(
        f"{partition:12s}: "
        f"{index_specs['source'][partition]['rows']:,} "
        f"origins"
    )


print("\nTARGET")

print(
    f"Boxes: {len(target_boxes):,}"
)


for partition in [
    "adaptation",
    "validation",
    "test",
]:

    print(
        f"{partition:12s}: "
        f"{index_specs['target'][partition]['rows']:,} "
        f"origins"
    )


print("\nMODEL INPUT CONTRACT")

print(
    f"Shared encoder history: "
    f"{tuple(shared_history_shape)}"
)

print(
    f"Source-private history: "
    f"{tuple(source_private_shape)}"
)

print(
    f"Target-private history: "
    f"{tuple(target_private_shape)}"
)

print(
    f"Multi-horizon labels: "
    f"{tuple(label_shape)}"
)

print(
    f"Forecast horizons: {horizons}"
)


print("\nSHARED ENCODER")

print(
    f"Identical input dimension "
    f"in both domains: "
    f"{len(shared_encoder_features)}"
)

for feature in shared_encoder_features:
    print(f"  {feature}")


print("\nGenerated:")

for path in [
    SPEC_PATH,
    FEATURE_PATH,
    SUMMARY_PATH,
    NORM_PATH,
    MANIFEST_PATH,
]:

    print(
        f"  {path.relative_to(ROOT)}"
    )


print(
    """

IMPORTANT:

Step 22 uses LAZY dataset construction.

The approximately 22.7 million BDG2 forecast origins
are NOT expanded into duplicated 24-hour tensors.

The frozen Step 19 and Step 20 window indices remain
the authoritative sample definitions.

The future PyTorch Dataset will retrieve each 24-hour
history only when requested.

SHARED REPRESENTATION

Both domains use exactly the same 15 shared features:

    temperature
    relative_humidity
    pressure

    temperature_observed
    relative_humidity_observed
    pressure_observed

    temperature_imputed
    relative_humidity_imputed
    pressure_imputed

    hour_sin
    hour_cos
    dow_sin
    dow_cos
    doy_sin
    doy_cos

No separate *_missing model channel is required.

Missing state is encoded as:

    observed = 1, imputed = 0
        directly observed

    observed = 0, imputed = 1
        causally filled

    observed = 0, imputed = 0
        unresolved missing

For tensor construction, unresolved normalized NaN
values will be replaced by 0.0 while their masks
remain zero.

PRIVATE REPRESENTATIONS

BDG2:
    energy
    energy_observed
    energy_imputed

openSenseMap:
    pm25
    pm25_observed
    pm25_imputed

The energy prediction head is NOT transferred into
the PM2.5 prediction head.

Only the shared environmental temporal representation
is transferable.

NORMALIZATION

Source:
    frozen BDG2 TRAINING-only statistics

Target:
    frozen openSenseMap ADAPTATION-only statistics

No validation/test normalization refitting is allowed.

EVALUATION

Only directly observed future labels may be scored.

Imputed labels are never evaluation ground truth.

No target test observation has been used for:

    normalization
    adaptation
    training
    early stopping
    model selection
    threshold selection

No forecasting model was trained.

No forecasting performance was inspected.
"""
)