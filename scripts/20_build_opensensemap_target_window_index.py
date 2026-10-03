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
    ROOT
    / "data"
    / "final"
    / "opensensemap"
    / "opensensemap_target_hourly_wide.csv"
)

COHORT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "opensensemap_final_eligible_target_cohort.csv"
)

PROTOCOL_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "forecasting_protocol.json"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "opensensemap"
META_DIR = ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FIXED EXPERIMENTAL RULES
# =============================================================================

HISTORY_LENGTH = 24
HORIZONS = [1, 3, 6, 12, 24]

PARTITIONS = {
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

SHARED_VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
]

TARGET_VARIABLE = "pm25"


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def as_bool(series):
    """
    Robust conversion of mask columns to Boolean.
    """
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

start_time = time.time()

banner("OPENSENSEMAP TARGET FORECAST-WINDOW INDEX")


# =============================================================================
# VERIFY REQUIRED FILES
# =============================================================================

for path in [TARGET_FILE, COHORT_FILE, PROTOCOL_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

print("\nRequired files found:")
print(f"  {TARGET_FILE.relative_to(ROOT)}")
print(f"  {COHORT_FILE.relative_to(ROOT)}")
print(f"  {PROTOCOL_FILE.relative_to(ROOT)}")


# =============================================================================
# VERIFY FROZEN FORECASTING PROTOCOL
# =============================================================================

with open(PROTOCOL_FILE, "r", encoding="utf-8") as f:
    protocol = json.load(f)

protocol_text = json.dumps(protocol).lower()

if str(HISTORY_LENGTH) not in protocol_text:
    raise RuntimeError(
        "Could not verify 24-hour history in forecasting protocol."
    )

for h in HORIZONS:
    if str(h) not in protocol_text:
        raise RuntimeError(
            f"Could not verify horizon {h} in forecasting protocol."
        )

print("\nFrozen forecasting rules:")
print(f"  History length: {HISTORY_LENGTH} hours")
print(f"  Horizons: {HORIZONS}")
print(f"  Target: {TARGET_VARIABLE}")


# =============================================================================
# LOAD FROZEN COHORT
# =============================================================================

banner("LOADING FROZEN TARGET COHORT")

cohort = pd.read_csv(COHORT_FILE, dtype={"box_id": str})

if "box_id" not in cohort.columns:
    raise RuntimeError("Frozen target cohort does not contain box_id.")

frozen_boxes = cohort["box_id"].astype(str).drop_duplicates().tolist()

if len(frozen_boxes) != 17:
    raise RuntimeError(
        f"Expected 17 frozen target boxes, found {len(frozen_boxes)}."
    )

print(f"\nFrozen target boxes: {len(frozen_boxes)}")


# =============================================================================
# LOAD TARGET HOURLY DATA
# =============================================================================

banner("LOADING TARGET HOURLY DATA")

df = pd.read_csv(
    TARGET_FILE,
    dtype={"box_id": str},
)

if "timestamp" not in df.columns:
    raise RuntimeError("Target dataset does not contain timestamp.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="raise",
)

required_columns = [
    "box_id",
    "timestamp",
    "temperature",
    "temperature_observed",
    "relative_humidity",
    "relative_humidity_observed",
    "pressure",
    "pressure_observed",
    "pm25",
    "pm25_observed",
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required target columns:\n"
        + "\n".join(missing_columns)
    )

df = df[df["box_id"].isin(frozen_boxes)].copy()

boxes_in_data = df["box_id"].nunique()

if boxes_in_data != 17:
    raise RuntimeError(
        f"Expected data for 17 boxes, found {boxes_in_data}."
    )

df = df.sort_values(
    ["box_id", "timestamp"]
).reset_index(drop=True)

for c in [
    "temperature_observed",
    "relative_humidity_observed",
    "pressure_observed",
    "pm25_observed",
]:
    df[c] = as_bool(df[c])

print(f"\nRows: {len(df):,}")
print(f"Boxes: {boxes_in_data}")
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")


# =============================================================================
# EXACT GRID VERIFICATION
# =============================================================================

banner("VERIFYING TARGET HOURLY GRID")

EXPECTED_HOURS = 6552

grid_counts = (
    df.groupby("box_id")["timestamp"]
    .nunique()
)

bad_grid = grid_counts[
    grid_counts != EXPECTED_HOURS
]

if not bad_grid.empty:
    raise RuntimeError(
        "One or more boxes do not contain exactly "
        f"{EXPECTED_HOURS} hourly timestamps:\n{bad_grid}"
    )

duplicates = df.duplicated(
    subset=["box_id", "timestamp"]
).sum()

if duplicates:
    raise RuntimeError(
        f"Found {duplicates} duplicate box/timestamp rows."
    )

print(f"\nEvery box contains exactly {EXPECTED_HOURS:,} hourly rows.")
print("Duplicate box/timestamp rows: 0")
print("Hourly grid verification: PASSED")


# =============================================================================
# BUILD WINDOW INDICES
# =============================================================================

banner("GENERATING TARGET WINDOW INDICES")

partition_results = {}
audit_rows = []

total_candidate_origins = 0
total_retained_origins = 0

for partition, (partition_start, partition_end) in PARTITIONS.items():

    print(f"\nProcessing partition: {partition}")

    partition_frames = []

    for box_number, box_id in enumerate(frozen_boxes, start=1):

        box = df[df["box_id"] == box_id].copy()

        box = box[
            (box["timestamp"] >= partition_start)
            & (box["timestamp"] <= partition_end)
        ].copy()

        box = box.sort_values("timestamp").reset_index(drop=True)

        if box.empty:
            raise RuntimeError(
                f"No {partition} data found for box {box_id}."
            )

        # -------------------------------------------------------------
        # Ensure partition itself is hourly and contiguous
        # -------------------------------------------------------------

        expected_partition_index = pd.date_range(
            partition_start,
            partition_end,
            freq="h",
        )

        if len(box) != len(expected_partition_index):
            raise RuntimeError(
                f"{box_id} {partition}: expected "
                f"{len(expected_partition_index)} rows, "
                f"found {len(box)}."
            )

        actual_times = pd.DatetimeIndex(box["timestamp"])

        if not actual_times.equals(expected_partition_index):
            raise RuntimeError(
                f"{box_id} {partition}: timestamps are not "
                "the exact frozen hourly grid."
            )

        n = len(box)

        # -------------------------------------------------------------
        # Forecast origin must have:
        #
        #   24-hour history ending at t
        #   all requested labels remaining inside partition
        #
        # Since max horizon = 24, valid positions are:
        #
        #   HISTORY_LENGTH - 1 ... n - MAX_HORIZON - 1
        # -------------------------------------------------------------

        max_horizon = max(HORIZONS)

        first_origin_pos = HISTORY_LENGTH - 1
        last_origin_pos = n - max_horizon - 1

        if last_origin_pos < first_origin_pos:
            raise RuntimeError(
                f"{box_id} {partition}: partition too short."
            )

        origin_positions = np.arange(
            first_origin_pos,
            last_origin_pos + 1,
            dtype=np.int32,
        )

        total_candidate_origins += len(origin_positions)

        origin_timestamps = (
            box.loc[origin_positions, "timestamp"]
            .reset_index(drop=True)
        )

        out = pd.DataFrame({
            "box_id": box_id,
            "partition": partition,
            "origin_position": origin_positions,
            "origin_timestamp": origin_timestamps,
        })

        # -------------------------------------------------------------
        # History boundaries
        # -------------------------------------------------------------

        out["history_start_position"] = (
            out["origin_position"] - (HISTORY_LENGTH - 1)
        ).astype(np.int32)

        out["history_end_position"] = (
            out["origin_position"]
        ).astype(np.int32)

        out["history_start_timestamp"] = (
            out["origin_timestamp"]
            - pd.Timedelta(hours=HISTORY_LENGTH - 1)
        )

        out["history_end_timestamp"] = out["origin_timestamp"]

        # -------------------------------------------------------------
        # History availability audit
        #
        # We DO NOT require every history value to have been directly
        # observed. Causally available/fillable values remain legitimate
        # model inputs according to the frozen preprocessing policy.
        #
        # But these flags tell us whether each shared variable is
        # numerically available across the entire 24-hour history.
        # -------------------------------------------------------------

        for variable in SHARED_VARIABLES + [TARGET_VARIABLE]:

            available = box[variable].notna().astype(np.int8)

            rolling_available = (
                available
                .rolling(
                    window=HISTORY_LENGTH,
                    min_periods=HISTORY_LENGTH,
                )
                .sum()
            )

            values = (
                rolling_available
                .iloc[origin_positions]
                .to_numpy()
            )

            out[f"{variable}_history_available_hours"] = (
                values.astype(np.int16)
            )

            out[f"{variable}_history_complete"] = (
                values == HISTORY_LENGTH
            )

        # -------------------------------------------------------------
        # Horizon-specific PM2.5 labels
        # -------------------------------------------------------------

        for horizon in HORIZONS:

            label_positions = origin_positions + horizon

            label_times = (
                box.loc[label_positions, "timestamp"]
                .reset_index(drop=True)
            )

            expected_label_times = (
                origin_timestamps
                + pd.Timedelta(hours=horizon)
            )

            if not label_times.equals(expected_label_times):
                raise RuntimeError(
                    f"{box_id} {partition}: horizon {horizon} "
                    "timestamp mismatch."
                )

            label_values = (
                box.loc[label_positions, TARGET_VARIABLE]
                .to_numpy()
            )

            label_observed = (
                box.loc[label_positions, "pm25_observed"]
                .to_numpy(dtype=bool)
            )

            out[f"label_timestamp_h{horizon}"] = label_times
            out[f"label_pm25_h{horizon}"] = label_values
            out[f"label_observed_h{horizon}"] = label_observed

            # Critical integrity condition:
            # an evaluation label is usable only if directly observed.
            out[f"label_valid_h{horizon}"] = (
                label_observed
                & np.isfinite(label_values)
            )

        # -------------------------------------------------------------
        # At least one directly observed requested horizon
        #
        # Do not require ALL five horizons. Doing so would unnecessarily
        # discard valid horizon-specific examples.
        # -------------------------------------------------------------

        label_valid_columns = [
            f"label_valid_h{h}"
            for h in HORIZONS
        ]

        out["n_valid_horizon_labels"] = (
            out[label_valid_columns]
            .sum(axis=1)
            .astype(np.int8)
        )

        out["has_any_valid_label"] = (
            out["n_valid_horizon_labels"] > 0
        )

        before = len(out)

        out = out[
            out["has_any_valid_label"]
        ].reset_index(drop=True)

        after = len(out)

        total_retained_origins += after

        partition_frames.append(out)

        audit_row = {
            "partition": partition,
            "box_id": box_id,
            "partition_hours": n,
            "candidate_origins": before,
            "retained_origins": after,
        }

        for horizon in HORIZONS:
            audit_row[f"observed_labels_h{horizon}"] = int(
                out[f"label_valid_h{horizon}"].sum()
            )

        audit_rows.append(audit_row)

        if box_number % 5 == 0 or box_number == len(frozen_boxes):
            print(
                f"  Processed "
                f"{box_number}/{len(frozen_boxes)} boxes"
            )

    partition_df = pd.concat(
        partition_frames,
        ignore_index=True,
    )

    partition_results[partition] = partition_df


# =============================================================================
# GLOBAL LEAKAGE AUDIT
# =============================================================================

banner("GLOBAL LEAKAGE AUDIT")

summary_rows = []

for partition, data in partition_results.items():

    partition_start, partition_end = PARTITIONS[partition]

    if (data["history_start_timestamp"] < partition_start).any():
        raise RuntimeError(
            f"{partition}: history crosses partition start."
        )

    if (data["history_end_timestamp"] > partition_end).any():
        raise RuntimeError(
            f"{partition}: history crosses partition end."
        )

    if (
        data["history_end_timestamp"]
        != data["origin_timestamp"]
    ).any():
        raise RuntimeError(
            f"{partition}: history end != forecast origin."
        )

    for horizon in HORIZONS:

        expected = (
            data["origin_timestamp"]
            + pd.Timedelta(hours=horizon)
        )

        actual = data[f"label_timestamp_h{horizon}"]

        if not expected.equals(actual):
            raise RuntimeError(
                f"{partition}: horizon {horizon} "
                "timestamps are incorrect."
            )

        if (actual > partition_end).any():
            raise RuntimeError(
                f"{partition}: horizon {horizon} "
                "crosses partition end."
            )

        if (actual <= data["origin_timestamp"]).any():
            raise RuntimeError(
                f"{partition}: horizon {horizon} "
                "is not strictly future."
            )

        # -------------------------------------------------------------
        # Direct ground-truth verification
        # -------------------------------------------------------------

        valid_mask = data[f"label_valid_h{horizon}"]

        if (
            valid_mask
            & ~data[f"label_observed_h{horizon}"]
        ).any():
            raise RuntimeError(
                f"{partition}: non-observed PM2.5 label "
                f"marked valid at h={horizon}."
            )

    row = {
        "partition": partition,
        "origins": len(data),
        "boxes": data["box_id"].nunique(),
    }

    for horizon in HORIZONS:
        row[f"observed_labels_h{horizon}"] = int(
            data[f"label_valid_h{horizon}"].sum()
        )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

print()
print(summary.to_string(index=False))


# =============================================================================
# VERIFY ALL FROZEN BOXES RETAINED
# =============================================================================

banner("BOX COVERAGE")

for partition, data in partition_results.items():

    n_boxes = data["box_id"].nunique()

    print(
        f"{partition:12s}: "
        f"{n_boxes}/{len(frozen_boxes)} boxes"
    )

    if n_boxes != len(frozen_boxes):
        missing = sorted(
            set(frozen_boxes)
            - set(data["box_id"].unique())
        )

        raise RuntimeError(
            f"{partition}: frozen boxes disappeared "
            f"from window index:\n{missing}"
        )


# =============================================================================
# DIRECT LABEL MASK VERIFICATION AGAINST ORIGINAL DATA
# =============================================================================

banner("DIRECT PM2.5 LABEL MASK VERIFICATION")

lookup = (
    df[
        [
            "box_id",
            "timestamp",
            "pm25",
            "pm25_observed",
        ]
    ]
    .set_index(["box_id", "timestamp"])
    .sort_index()
)

verified_labels = 0

for partition, data in partition_results.items():

    for horizon in HORIZONS:

        valid = data[
            data[f"label_valid_h{horizon}"]
        ]

        if valid.empty:
            continue

        idx = pd.MultiIndex.from_arrays(
            [
                valid["box_id"].astype(str),
                valid[f"label_timestamp_h{horizon}"],
            ],
            names=["box_id", "timestamp"],
        )

        original = lookup.loc[idx]

        if not original["pm25_observed"].all():
            raise RuntimeError(
                f"{partition} h={horizon}: "
                "a retained ground-truth label was "
                "not directly observed."
            )

        if original["pm25"].isna().any():
            raise RuntimeError(
                f"{partition} h={horizon}: "
                "a retained ground-truth label is missing."
            )

        verified_labels += len(valid)

print(
    f"\nVerified directly observed status for "
    f"{verified_labels:,} PM2.5 labels."
)

print("Direct-label mask verification: PASSED")


# =============================================================================
# WRITE OUTPUTS
# =============================================================================

banner("WRITING TARGET WINDOW INDICES")

for partition, data in partition_results.items():

    path = (
        OUTPUT_DIR
        / f"target_window_index_{partition}.parquet"
    )

    data.to_parquet(
        path,
        index=False,
        compression="snappy",
    )

    size_mb = path.stat().st_size / 1024 / 1024

    print(
        f"{partition:12s}: "
        f"{len(data):,} origins | "
        f"{size_mb:.2f} MB"
    )


# =============================================================================
# AUDIT FILE
# =============================================================================

audit = pd.DataFrame(audit_rows)

audit_file = (
    META_DIR
    / "opensensemap_target_window_audit.csv"
)

audit.to_csv(
    audit_file,
    index=False,
)


# =============================================================================
# SUMMARY FILE
# =============================================================================

summary_file = (
    META_DIR
    / "opensensemap_target_window_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {
    "dataset": "openSenseMap",
    "role": "target_domain",
    "frozen_boxes": len(frozen_boxes),
    "history_length_hours": HISTORY_LENGTH,
    "forecast_horizons_hours": HORIZONS,
    "shared_variables": SHARED_VARIABLES,
    "target_private_variable": TARGET_VARIABLE,
    "partitions": {
        name: {
            "start": str(start),
            "end": str(end),
        }
        for name, (start, end) in PARTITIONS.items()
    },
    "window_policy": {
        "history_uses_times": "<= forecast origin",
        "labels_use_times": "forecast origin + horizon",
        "cross_partition_windows": False,
        "history_may_include_causally_filled_values": True,
        "future_observations_used_as_inputs": False,
    },
    "ground_truth_policy": {
        "target": TARGET_VARIABLE,
        "directly_observed_labels_only": True,
        "imputed_labels_allowed": False,
        "missing_labels_manufactured": False,
        "horizon_specific_label_masks": True,
        "require_all_horizons_observed": False,
    },
    "test_policy": {
        "test_used_for_training": False,
        "test_used_for_adaptation": False,
        "test_used_for_normalization": False,
        "test_used_for_model_selection": False,
        "test_used_for_early_stopping": False,
    },
    "candidate_origins_before_label_filter": int(
        total_candidate_origins
    ),
    "retained_origins_with_at_least_one_observed_label": int(
        total_retained_origins
    ),
    "directly_observed_horizon_labels": int(
        verified_labels
    ),
}

manifest_file = (
    META_DIR
    / "opensensemap_target_window_manifest.json"
)

with open(
    manifest_file,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


# =============================================================================
# FINAL REPORT
# =============================================================================

elapsed = time.time() - start_time

banner("OPENSENSEMAP TARGET WINDOW INDEX COMPLETE")

print(
    f"""
Frozen target boxes:
  {len(frozen_boxes)}

History:
  {HISTORY_LENGTH} hours

Forecast horizons:
  {", ".join(map(str, HORIZONS))} hours

Candidate forecast origins:
  {total_candidate_origins:,}

Retained origins with >=1 directly observed PM2.5 label:
  {total_retained_origins:,}

Total directly observed horizon-specific PM2.5 labels:
  {verified_labels:,}

Processing time:
  {elapsed:.2f} seconds

Generated:

  data\\processed\\opensensemap\\target_window_index_adaptation.parquet
  data\\processed\\opensensemap\\target_window_index_validation.parquet
  data\\processed\\opensensemap\\target_window_index_test.parquet

  data\\metadata\\opensensemap_target_window_audit.csv
  data\\metadata\\opensensemap_target_window_summary.csv
  data\\metadata\\opensensemap_target_window_manifest.json

CAUSALITY / LEAKAGE STATUS

  24-hour history stays inside partition: PASSED
  Labels occur strictly after origin: PASSED
  Exact horizon timestamps: PASSED
  Labels stay inside partition: PASSED
  PM2.5 ground-truth labels directly observed: PASSED
  Imputed PM2.5 labels used as ground truth: NO
  Future observations used in history: NO
  Cross-partition windows: NO

No source normalization was refitted.

No target normalization was fitted.

No target test observation was used for training,
adaptation, normalization, early stopping, or model selection.

No forecasting model was trained.

No forecasting performance was inspected.
"""
)