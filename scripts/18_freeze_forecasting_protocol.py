from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META_DIR = ROOT / "data" / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_COHORT_FILE = (
    META_DIR / "bdg2_final_experimental_source_cohort.csv"
)

SOURCE_PARTITION_FILE = (
    META_DIR / "bdg2_temporal_partitions.csv"
)

SOURCE_NORMALIZATION_FILE = (
    META_DIR / "bdg2_training_normalization_statistics.csv"
)

TARGET_COHORT_FILE = (
    META_DIR / "opensensemap_final_eligible_target_cohort.csv"
)

TARGET_DATA_FILE = (
    ROOT / "data" / "final" / "opensensemap"
    / "opensensemap_target_hourly_wide.csv"
)

TARGET_CAUSAL_MANIFEST_FILE = (
    META_DIR / "opensensemap_causal_target_manifest.json"
)

OUTPUT_JSON = (
    META_DIR / "forecasting_protocol.json"
)

OUTPUT_CSV = (
    META_DIR / "forecasting_protocol_summary.csv"
)

OUTPUT_MANIFEST = (
    META_DIR / "forecasting_protocol_manifest.json"
)


# =============================================================================
# FROZEN EXPERIMENTAL DESIGN
# =============================================================================

HISTORY_LENGTH_HOURS = 24

FORECAST_HORIZONS_HOURS = [
    1,
    3,
    6,
    12,
    24,
]

SHARED_VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
]

SOURCE_PRIVATE_VARIABLE = "energy"

TARGET_PRIVATE_VARIABLE = "pm25"

CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]

SOURCE_PARTITIONS = {
    "train": {
        "start": "2016-01-01T00:00:00",
        "end": "2017-04-30T23:00:00",
    },
    "validation": {
        "start": "2017-05-01T00:00:00",
        "end": "2017-08-31T23:00:00",
    },
    "test": {
        "start": "2017-09-01T00:00:00",
        "end": "2017-12-31T23:00:00",
    },
}

TARGET_PARTITIONS = {
    "adaptation": {
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-06-30T23:00:00+00:00",
    },
    "validation": {
        "start": "2026-07-01T00:00:00+00:00",
        "end": "2026-08-31T23:00:00+00:00",
    },
    "test": {
        "start": "2026-09-01T00:00:00+00:00",
        "end": "2026-09-30T23:00:00+00:00",
    },
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
            f"Required frozen input does not exist:\n{path}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def parse_source_timestamp(value):
    return pd.Timestamp(value)


def parse_target_timestamp(value):
    return pd.Timestamp(value)


# =============================================================================
# VERIFY INPUTS
# =============================================================================

section("FORECASTING PROTOCOL FREEZE")

required_files = [
    SOURCE_COHORT_FILE,
    SOURCE_PARTITION_FILE,
    SOURCE_NORMALIZATION_FILE,
    TARGET_COHORT_FILE,
    TARGET_DATA_FILE,
    TARGET_CAUSAL_MANIFEST_FILE,
]

for path in required_files:
    require_file(path)

print("\nRequired frozen inputs:")
for path in required_files:
    print(f"  {path.relative_to(ROOT)}")


# =============================================================================
# LOAD SOURCE INFORMATION
# =============================================================================

section("VERIFYING BDG2 SOURCE DOMAIN")

source_cohort = pd.read_csv(
    SOURCE_COHORT_FILE,
    dtype={"building_id": str},
)

print(f"\nFrozen source buildings: {len(source_cohort):,}")

if "site_id" in source_cohort.columns:
    print(
        f"Frozen source sites: "
        f"{source_cohort['site_id'].nunique():,}"
    )

if len(source_cohort) != 1318:
    raise RuntimeError(
        "Expected 1,318 frozen BDG2 source buildings, "
        f"but found {len(source_cohort):,}."
    )

source_norm = pd.read_csv(
    SOURCE_NORMALIZATION_FILE
)

required_norm_variables = {
    "temperature",
    "relative_humidity",
    "pressure",
    "energy",
}

if "variable" not in source_norm.columns:
    raise RuntimeError(
        "Normalization file does not contain 'variable'."
    )

available_norm_variables = set(
    source_norm["variable"].astype(str)
)

missing_norm = (
    required_norm_variables
    - available_norm_variables
)

if missing_norm:
    raise RuntimeError(
        "Missing source normalization parameters for: "
        + ", ".join(sorted(missing_norm))
    )

print("\nSource normalization parameters found for:")
for variable in sorted(required_norm_variables):
    print(f"  {variable}")

print(
    "\nSource normalization remains TRAINING-ONLY."
)


# =============================================================================
# VERIFY SOURCE PARTITION CHRONOLOGY
# =============================================================================

section("VERIFYING SOURCE PARTITIONS")

previous_end = None

for partition, bounds in SOURCE_PARTITIONS.items():

    start = parse_source_timestamp(bounds["start"])
    end = parse_source_timestamp(bounds["end"])

    if start > end:
        raise RuntimeError(
            f"Invalid source partition: {partition}"
        )

    if previous_end is not None:
        if start <= previous_end:
            raise RuntimeError(
                "Source partitions overlap."
            )

    previous_end = end

    hours = int(
        (end - start).total_seconds() / 3600
    ) + 1

    print(
        f"{partition:12s}: "
        f"{start} -> {end} "
        f"({hours:,} hours)"
    )


# =============================================================================
# LOAD TARGET INFORMATION
# =============================================================================

section("VERIFYING OPENSENSEMAP TARGET DOMAIN")

target_cohort = pd.read_csv(
    TARGET_COHORT_FILE,
    dtype={"box_id": str},
)

target_data = pd.read_csv(
    TARGET_DATA_FILE,
    dtype={"box_id": str},
)

target_data["timestamp"] = pd.to_datetime(
    target_data["timestamp"],
    utc=True,
    errors="raise",
)

with open(
    TARGET_CAUSAL_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as f:
    target_causal_manifest = json.load(f)

target_boxes = set(
    target_cohort["box_id"].astype(str)
)

data_boxes = set(
    target_data["box_id"].astype(str)
)

print(f"\nFrozen target boxes: {len(target_boxes):,}")
print(f"Boxes in hourly target data: {len(data_boxes):,}")

if len(target_boxes) != 17:
    raise RuntimeError(
        "Expected 17 frozen target boxes, "
        f"but found {len(target_boxes)}."
    )

if target_boxes != data_boxes:

    missing = target_boxes - data_boxes
    unexpected = data_boxes - target_boxes

    raise RuntimeError(
        "Target cohort/data mismatch.\n"
        f"Missing: {sorted(missing)}\n"
        f"Unexpected: {sorted(unexpected)}"
    )


# =============================================================================
# VERIFY TARGET VARIABLES
# =============================================================================

required_target_columns = (
    SHARED_VARIABLES
    + [TARGET_PRIVATE_VARIABLE]
)

missing_columns = [
    x for x in required_target_columns
    if x not in target_data.columns
]

if missing_columns:
    raise RuntimeError(
        "Target data missing variables: "
        + ", ".join(missing_columns)
    )

print("\nTarget variables verified:")

for variable in required_target_columns:
    print(f"  {variable}")


# =============================================================================
# VERIFY OBSERVATION MASKS
# =============================================================================

section("VERIFYING OBSERVATION MASKS")

required_masks = []

for variable in required_target_columns:
    required_masks.append(
        f"{variable}_observed"
    )

missing_masks = [
    x for x in required_masks
    if x not in target_data.columns
]

if missing_masks:
    raise RuntimeError(
        "Missing observation masks:\n"
        + "\n".join(missing_masks)
    )

for mask in required_masks:

    values = set(
        target_data[mask]
        .dropna()
        .astype(int)
        .unique()
    )

    if not values.issubset({0, 1}):
        raise RuntimeError(
            f"Non-binary mask detected: {mask}"
        )

    print(f"Verified: {mask}")


# =============================================================================
# VERIFY TARGET CAUSAL IMPUTATION POLICY
# =============================================================================

section("VERIFYING TARGET CAUSAL POLICY")

missing_policy = target_causal_manifest.get(
    "missing_value_policy",
    {}
)

future_used = missing_policy.get(
    "future_observations_used"
)

if future_used is not False:
    raise RuntimeError(
        "Target causal manifest does not explicitly "
        "state future_observations_used=false."
    )

print(
    "\nFuture target observations used for filling: "
    "NO"
)

print(
    "Missing-value method:",
    missing_policy.get("method")
)

print(
    "Maximum consecutive fill:",
    missing_policy.get(
        "maximum_consecutive_hours"
    ),
    "hours",
)


# =============================================================================
# VERIFY TARGET PARTITION CHRONOLOGY
# =============================================================================

section("VERIFYING TARGET PARTITIONS")

previous_end = None

for partition, bounds in TARGET_PARTITIONS.items():

    start = parse_target_timestamp(
        bounds["start"]
    )

    end = parse_target_timestamp(
        bounds["end"]
    )

    if start > end:
        raise RuntimeError(
            f"Invalid target partition: {partition}"
        )

    if previous_end is not None:
        if start <= previous_end:
            raise RuntimeError(
                "Target partitions overlap."
            )

    previous_end = end

    hours = int(
        (end - start).total_seconds() / 3600
    ) + 1

    print(
        f"{partition:12s}: "
        f"{start} -> {end} "
        f"({hours:,} hours/box)"
    )


# =============================================================================
# DEFINE FORECASTING RULES
# =============================================================================

section("FREEZING FORECASTING RULES")

forecasting_rules = {

    "history_length_hours":
        HISTORY_LENGTH_HOURS,

    "forecast_horizons_hours":
        FORECAST_HORIZONS_HOURS,

    "forecast_origin_definition":
        (
            "The final timestamp included in the "
            "historical input sequence."
        ),

    "history_rule":
        (
            "For forecast origin t, model inputs may "
            "contain information with timestamps <= t only."
        ),

    "future_input_rule":
        (
            "No measured observation with timestamp > t "
            "may enter the model input."
        ),

    "label_rule":
        (
            "For forecast horizon h, the supervised label "
            "is the outcome observed at timestamp t+h."
        ),

    "evaluation_ground_truth_rule":
        (
            "A forecast is scored only when the label at "
            "t+h is directly observed. Causally imputed "
            "labels are not valid evaluation ground truth."
        ),

    "history_missingness_rule":
        (
            "Causally available historical values may be "
            "used together with explicit observation and "
            "imputation masks. Long unresolved gaps remain "
            "represented as missing."
        ),

    "partition_boundary_rule":
        (
            "The complete historical input sequence, "
            "forecast origin, and supervised label must "
            "belong to the same frozen partition."
        ),

    "calendar_rule":
        (
            "Deterministic calendar features are permitted "
            "because they are known at forecast time."
        ),

    "target_test_rule":
        (
            "Target test observations cannot influence "
            "normalization fitting, model fitting, "
            "adaptation, hyperparameter selection, early "
            "stopping, threshold selection, or any other "
            "model-selection decision."
        ),
}


# =============================================================================
# DEFINE TRANSFER ARCHITECTURE
# =============================================================================

transfer_protocol = {

    "problem_type":
        "heterogeneous-label cross-domain transfer",

    "shared_environmental_variables":
        SHARED_VARIABLES,

    "source_private_variable":
        SOURCE_PRIVATE_VARIABLE,

    "target_private_variable":
        TARGET_PRIVATE_VARIABLE,

    "source_task":
        (
            "Forecast future building electricity from "
            "historical shared environmental variables "
            "and historical building electricity."
        ),

    "target_task":
        (
            "Forecast future PM2.5 from historical shared "
            "environmental variables and historical PM2.5."
        ),

    "transferable_component":
        (
            "shared environmental temporal representation "
            "encoder"
        ),

    "source_private_component":
        "energy-specific temporal representation",

    "target_private_component":
        "PM2.5-specific temporal representation",

    "source_prediction_head":
        "energy forecasting head",

    "target_prediction_head":
        "PM2.5 forecasting head",

    "explicit_non_claim":
        (
            "The energy prediction head is not transferred "
            "as a PM2.5 prediction head. Transfer concerns "
            "the representation of shared environmental "
            "temporal dynamics."
        ),
}


# =============================================================================
# NORMALIZATION POLICY
# =============================================================================

normalization_protocol = {

    "source": {

        "fit_partition":
            "BDG2 train",

        "statistics_file":
            str(
                SOURCE_NORMALIZATION_FILE.relative_to(
                    ROOT
                )
            ),

        "validation_refit":
            False,

        "test_refit":
            False,

        "target_data_used":
            False,
    },

    "target": {

        "fit_partition":
            "adaptation only",

        "validation_used_for_fit":
            False,

        "test_used_for_fit":
            False,

        "future_observations_used":
            False,

        "status":
            (
                "parameters will be estimated in a "
                "subsequent preprocessing stage"
            ),
    },
}


# =============================================================================
# MASK POLICY
# =============================================================================

mask_protocol = {

    "historical_observation_masks":
        True,

    "historical_imputation_masks":
        True,

    "evaluation_requires_observed_label":
        True,

    "imputed_label_allowed_for_evaluation":
        False,

    "missingness_is_explicit_model_information":
        True,
}


# =============================================================================
# FULL FROZEN PROTOCOL
# =============================================================================

protocol = {

    "protocol_version":
        "1.0",

    "frozen_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "history_length_hours":
        HISTORY_LENGTH_HOURS,

    "forecast_horizons_hours":
        FORECAST_HORIZONS_HOURS,

    "shared_variables":
        SHARED_VARIABLES,

    "source_private_variable":
        SOURCE_PRIVATE_VARIABLE,

    "target_private_variable":
        TARGET_PRIVATE_VARIABLE,

    "calendar_features":
        CALENDAR_FEATURES,

    "source_partitions":
        SOURCE_PARTITIONS,

    "target_partitions":
        TARGET_PARTITIONS,

    "forecasting_rules":
        forecasting_rules,

    "transfer_protocol":
        transfer_protocol,

    "normalization_protocol":
        normalization_protocol,

    "mask_protocol":
        mask_protocol,

    "causality": {

        "future_observations_in_inputs":
            False,

        "future_based_interpolation":
            False,

        "target_test_for_normalization":
            False,

        "target_test_for_adaptation":
            False,

        "target_test_for_model_selection":
            False,

        "target_test_for_early_stopping":
            False,

        "target_test_for_threshold_selection":
            False,

        "cross_partition_windows":
            False,
    },
}


# =============================================================================
# WRITE JSON
# =============================================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        protocol,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# HUMAN-READABLE SUMMARY
# =============================================================================

summary_rows = []

for horizon in FORECAST_HORIZONS_HOURS:

    summary_rows.append({

        "history_hours":
            HISTORY_LENGTH_HOURS,

        "forecast_horizon_hours":
            horizon,

        "shared_variables":
            ", ".join(
                SHARED_VARIABLES
            ),

        "source_private_target":
            SOURCE_PRIVATE_VARIABLE,

        "target_private_target":
            TARGET_PRIVATE_VARIABLE,

        "source_training_partition":
            "train",

        "target_adaptation_partition":
            "adaptation",

        "target_validation_partition":
            "validation",

        "target_test_partition":
            "test",

        "observed_labels_required":
            True,

        "future_observations_allowed":
            False,

        "cross_partition_windows_allowed":
            False,
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {

    "protocol_file":
        str(
            OUTPUT_JSON.relative_to(ROOT)
        ),

    "protocol_sha256":
        sha256_file(OUTPUT_JSON),

    "history_length_hours":
        HISTORY_LENGTH_HOURS,

    "forecast_horizons_hours":
        FORECAST_HORIZONS_HOURS,

    "source_buildings":
        len(source_cohort),

    "target_boxes":
        len(target_boxes),

    "shared_variables":
        SHARED_VARIABLES,

    "source_private_variable":
        SOURCE_PRIVATE_VARIABLE,

    "target_private_variable":
        TARGET_PRIVATE_VARIABLE,

    "future_observations_allowed":
        False,

    "cross_partition_windows_allowed":
        False,

    "frozen_input_hashes": {
        str(path.relative_to(ROOT)):
            sha256_file(path)
        for path in required_files
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
        ensure_ascii=False,
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

section("FORECASTING PROTOCOL FROZEN")

print(
    f"""
History length:
{HISTORY_LENGTH_HOURS} hours

Forecast horizons:
{", ".join(map(str, FORECAST_HORIZONS_HOURS))} hours

Shared environmental variables:
{", ".join(SHARED_VARIABLES)}

Source-private prediction variable:
{SOURCE_PRIVATE_VARIABLE}

Target-private prediction variable:
{TARGET_PRIVATE_VARIABLE}

Transfer mechanism:
Shared environmental temporal representation

Source domain:
BDG2
{len(source_cohort):,} frozen buildings

Target domain:
openSenseMap
{len(target_boxes):,} frozen boxes

CAUSALITY RULES

For forecast origin t:

    inputs <= t

For horizon h:

    label = outcome at t+h

Future measured observations:
    NOT ALLOWED as input

Evaluation labels:
    MUST BE DIRECTLY OBSERVED

Imputed evaluation labels:
    NOT ALLOWED

Cross-partition windows:
    NOT ALLOWED

Target test data may NOT be used for:

    normalization
    adaptation
    training
    model selection
    early stopping
    threshold selection

Generated:

  {OUTPUT_JSON.relative_to(ROOT)}
  {OUTPUT_CSV.relative_to(ROOT)}
  {OUTPUT_MANIFEST.relative_to(ROOT)}

IMPORTANT:

This protocol is now intended to remain fixed.

The source and target domains have different private
prediction variables.

The experiment therefore does NOT transfer an electricity
prediction head into a PM2.5 prediction head.

It transfers representations of shared environmental
temporal dynamics.

No forecasting model has been trained in this step.
No forecasting result has been inspected.
No target test observation has been used for fitting.
"""
)