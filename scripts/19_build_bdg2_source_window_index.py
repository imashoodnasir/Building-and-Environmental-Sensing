from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META_DIR = ROOT / "data" / "metadata"
BDG2_DIR = ROOT / "data" / "final" / "bdg2" / "buildings"
OUT_DIR = ROOT / "data" / "processed" / "bdg2"

OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_FILE = META_DIR / "bdg2_final_experimental_source_cohort.csv"
PROTOCOL_FILE = META_DIR / "forecasting_protocol.json"
NORMALIZATION_FILE = META_DIR / "bdg2_training_normalization_statistics.csv"

AUDIT_FILE = META_DIR / "bdg2_source_window_audit.csv"
SUMMARY_FILE = META_DIR / "bdg2_source_window_summary.csv"
MANIFEST_FILE = META_DIR / "bdg2_source_window_manifest.json"


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
            f"Required file does not exist:\n{path}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def normalize_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


# =============================================================================
# LOAD FROZEN CONFIGURATION
# =============================================================================

section("BDG2 SOURCE FORECAST-WINDOW INDEX - VECTORIZED")

require_file(COHORT_FILE)
require_file(PROTOCOL_FILE)
require_file(NORMALIZATION_FILE)

with open(PROTOCOL_FILE, "r", encoding="utf-8") as f:
    protocol = json.load(f)

history_hours = int(protocol["history_length_hours"])

horizons = [
    int(x)
    for x in protocol["forecast_horizons_hours"]
]

source_partitions = protocol["source_partitions"]

if history_hours != 24:
    raise RuntimeError(
        f"Expected history length 24, found {history_hours}."
    )

if horizons != [1, 3, 6, 12, 24]:
    raise RuntimeError(
        f"Unexpected frozen horizons: {horizons}"
    )

print(f"\nHistory length: {history_hours} hours")
print(f"Horizons: {horizons}")


# =============================================================================
# LOAD FROZEN SOURCE COHORT
# =============================================================================

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={
        "building_id": str,
        "site_id": str,
    },
)

if len(cohort) != 1318:
    raise RuntimeError(
        f"Expected 1,318 buildings, found {len(cohort):,}."
    )

if cohort["building_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate building IDs found in frozen cohort."
    )

print(f"Frozen buildings: {len(cohort):,}")
print(f"Frozen sites: {cohort['site_id'].nunique():,}")


# =============================================================================
# VERIFY PARQUET FILES
# =============================================================================

section("VERIFYING BUILDING PARQUET FILES")

missing_files = []

for building_id in cohort["building_id"]:
    path = BDG2_DIR / f"{building_id}.parquet"

    if not path.exists():
        missing_files.append(building_id)

if missing_files:
    raise RuntimeError(
        f"{len(missing_files)} frozen building files are missing.\n"
        f"Examples: {missing_files[:10]}"
    )

print(
    f"All {len(cohort):,} frozen building parquet files found."
)


# =============================================================================
# VERIFY SCHEMA
# =============================================================================

sample_id = cohort.iloc[0]["building_id"]
sample_file = BDG2_DIR / f"{sample_id}.parquet"

sample = pd.read_parquet(sample_file)

section("BUILDING FILE SCHEMA")

print(f"\nExample building: {sample_id}")
print("\nColumns:")

for column in sample.columns:
    print(f"  {column}")

required_columns = {
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
}

missing_columns = required_columns - set(sample.columns)

if missing_columns:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(sorted(missing_columns))
    )


# =============================================================================
# OUTPUT BUFFERS
# =============================================================================

partition_frames = {
    partition: []
    for partition in source_partitions
}

audit_rows = []

processing_start = time.time()


# =============================================================================
# PROCESS EACH BUILDING
# =============================================================================

section("GENERATING WINDOW INDICES")

for building_number, cohort_row in cohort.iterrows():

    building_id = str(cohort_row["building_id"])
    expected_site_id = str(cohort_row["site_id"])

    path = BDG2_DIR / f"{building_id}.parquet"

    # Read only what Step 19 needs.
    df = pd.read_parquet(
        path,
        columns=[
            "site_id",
            "timestamp",
            "energy",
            "energy_observed",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise RuntimeError(
            f"Empty building file: {building_id}"
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise RuntimeError(
            f"Non-monotonic timestamps: {building_id}"
        )

    df["energy_observed"] = normalize_boolean(
        df["energy_observed"]
    )

    # Verify site identity.
    file_sites = set(
        df["site_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if file_sites and file_sites != {expected_site_id}:
        raise RuntimeError(
            f"Site mismatch for {building_id}: "
            f"cohort={expected_site_id}, file={file_sites}"
        )

    # =========================================================================
    # PROCESS PARTITIONS
    # =========================================================================

    for partition_name, bounds in source_partitions.items():

        partition_start = pd.Timestamp(bounds["start"])
        partition_end = pd.Timestamp(bounds["end"])

        part = df.loc[
            (df["timestamp"] >= partition_start)
            &
            (df["timestamp"] <= partition_end),
            [
                "timestamp",
                "energy",
                "energy_observed",
            ],
        ].copy()

        part = part.reset_index(drop=True)

        expected_index = pd.date_range(
            partition_start,
            partition_end,
            freq="h",
        )

        if len(part) != len(expected_index):
            raise RuntimeError(
                f"Hourly-grid row-count mismatch for "
                f"{building_id} / {partition_name}. "
                f"Expected {len(expected_index)}, "
                f"found {len(part)}."
            )

        actual_index = pd.DatetimeIndex(
            part["timestamp"]
        )

        if not actual_index.equals(expected_index):
            raise RuntimeError(
                f"Hourly timestamp grid mismatch for "
                f"{building_id} / {partition_name}."
            )

        # =====================================================================
        # VECTORISED FUTURE LABEL GENERATION
        # =====================================================================

        for h in horizons:

            # Future physical energy value.
            future_energy = part["energy"].shift(-h)

            # Whether that future value was directly observed.
            future_observed = (
                part["energy_observed"]
                .shift(-h)
                .fillna(False)
                .astype(bool)
            )

            # Timestamp exactly h hours into future.
            future_timestamp = (
                part["timestamp"].shift(-h)
            )

            # Directly observed labels only.
            valid_label = (
                future_observed
                & future_energy.notna()
                & future_timestamp.notna()
            )

            part[
                f"label_t_plus_{h}"
            ] = future_energy.where(
                valid_label,
                np.nan,
            )

            part[
                f"observed_t_plus_{h}"
            ] = valid_label

            part[
                f"label_timestamp_t_plus_{h}"
            ] = future_timestamp.where(
                valid_label,
                pd.NaT,
            )

        # =====================================================================
        # HISTORY ELIGIBILITY
        # =====================================================================

        # Origin row position 23 corresponds to:
        #
        # [0 ... 23]
        #
        # = exactly 24 hourly observations including origin.
        #
        # Therefore rows 0..22 cannot be forecast origins.

        origin_mask = (
            np.arange(len(part))
            >= (history_hours - 1)
        )

        candidate_origins = int(
            origin_mask.sum()
        )

        # At least one horizon must have an observed label.
        any_label = np.zeros(
            len(part),
            dtype=bool,
        )

        for h in horizons:
            any_label |= (
                part[
                    f"observed_t_plus_{h}"
                ].to_numpy(dtype=bool)
            )

        retain_mask = (
            origin_mask
            & any_label
        )

        retained = part.loc[
            retain_mask
        ].copy()

        retained[
            "history_start_timestamp"
        ] = (
            retained["timestamp"]
            - pd.Timedelta(
                hours=history_hours - 1
            )
        )

        retained = retained.rename(
            columns={
                "timestamp":
                    "origin_timestamp"
            }
        )

        retained.insert(
            0,
            "partition",
            partition_name,
        )

        retained.insert(
            0,
            "site_id",
            expected_site_id,
        )

        retained.insert(
            0,
            "building_id",
            building_id,
        )

        # Energy at forecast origin is not needed in the index.
        retained = retained.drop(
            columns=[
                "energy",
                "energy_observed",
            ]
        )

        # Reorder columns.
        base_columns = [
            "building_id",
            "site_id",
            "partition",
            "history_start_timestamp",
            "origin_timestamp",
        ]

        label_columns = []

        for h in horizons:
            label_columns.extend([
                f"label_timestamp_t_plus_{h}",
                f"label_t_plus_{h}",
                f"observed_t_plus_{h}",
            ])

        retained = retained[
            base_columns + label_columns
        ]

        partition_frames[
            partition_name
        ].append(retained)

        # =====================================================================
        # AUDIT
        # =====================================================================

        audit_row = {
            "building_id":
                building_id,
            "site_id":
                expected_site_id,
            "partition":
                partition_name,
            "partition_start":
                partition_start,
            "partition_end":
                partition_end,
            "partition_hours":
                len(part),
            "candidate_origins":
                candidate_origins,
            "retained_origins":
                len(retained),
        }

        for h in horizons:

            audit_row[
                f"observed_labels_h{h}"
            ] = int(
                retained[
                    f"observed_t_plus_{h}"
                ].sum()
            )

        audit_rows.append(
            audit_row
        )

    # =========================================================================
    # PROGRESS
    # =========================================================================

    processed = building_number + 1

    if (
        processed == 1
        or processed % 25 == 0
        or processed == len(cohort)
    ):

        elapsed = (
            time.time()
            - processing_start
        )

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            len(cohort)
            - processed
        )

        eta_seconds = (
            remaining / rate
            if rate > 0
            else np.nan
        )

        print(
            f"Processed "
            f"{processed:,}/{len(cohort):,} "
            f"| {rate:.2f} buildings/s "
            f"| elapsed {elapsed/60:.1f} min "
            f"| ETA {eta_seconds/60:.1f} min",
            flush=True,
        )


# =============================================================================
# COMBINE AND SAVE
# =============================================================================

section("WRITING SOURCE WINDOW INDICES")

output_files = {}

for partition_name in source_partitions:

    frames = partition_frames[
        partition_name
    ]

    if not frames:
        raise RuntimeError(
            f"No frames generated for {partition_name}."
        )

    d = pd.concat(
        frames,
        ignore_index=True,
    )

    d = d.sort_values(
        [
            "building_id",
            "origin_timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    output_path = (
        OUT_DIR
        / f"source_window_index_{partition_name}.parquet"
    )

    d.to_parquet(
        output_path,
        index=False,
    )

    output_files[
        partition_name
    ] = output_path

    print(
        f"{partition_name:12s}: "
        f"{len(d):,} origins "
        f"| {output_path.stat().st_size / 1024**2:.2f} MB"
    )


# =============================================================================
# SAVE DETAILED AUDIT
# =============================================================================

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_FILE,
    index=False,
)


# =============================================================================
# GLOBAL LEAKAGE / INTEGRITY AUDIT
# =============================================================================

section("GLOBAL LEAKAGE AUDIT")

summary_rows = []

for partition_name, output_path in output_files.items():

    d = pd.read_parquet(
        output_path
    )

    d["history_start_timestamp"] = pd.to_datetime(
        d["history_start_timestamp"]
    )

    d["origin_timestamp"] = pd.to_datetime(
        d["origin_timestamp"]
    )

    partition_start = pd.Timestamp(
        source_partitions[
            partition_name
        ]["start"]
    )

    partition_end = pd.Timestamp(
        source_partitions[
            partition_name
        ]["end"]
    )

    # -------------------------------------------------------------------------
    # History length
    # -------------------------------------------------------------------------

    history_delta = (
        d["origin_timestamp"]
        - d["history_start_timestamp"]
    )

    expected_history_delta = pd.Timedelta(
        hours=history_hours - 1
    )

    if not (
        history_delta
        == expected_history_delta
    ).all():
        raise RuntimeError(
            f"History length violation in {partition_name}."
        )

    # -------------------------------------------------------------------------
    # History boundary
    # -------------------------------------------------------------------------

    if (
        d["history_start_timestamp"]
        < partition_start
    ).any():
        raise RuntimeError(
            f"History crosses start of {partition_name}."
        )

    if (
        d["origin_timestamp"]
        > partition_end
    ).any():
        raise RuntimeError(
            f"Origin exceeds end of {partition_name}."
        )

    # -------------------------------------------------------------------------
    # At least one valid label per retained origin
    # -------------------------------------------------------------------------

    any_observed = np.zeros(
        len(d),
        dtype=bool,
    )

    for h in horizons:

        obs_col = f"observed_t_plus_{h}"
        label_col = f"label_t_plus_{h}"
        ts_col = (
            f"label_timestamp_t_plus_{h}"
        )

        d[obs_col] = normalize_boolean(
            d[obs_col]
        )

        any_observed |= (
            d[obs_col].to_numpy(
                dtype=bool
            )
        )

        label_ts = pd.to_datetime(
            d[ts_col],
            errors="coerce",
        )

        observed = d[obs_col]

        # Observed label must have physical value.
        if (
            observed
            & d[label_col].isna()
        ).any():
            raise RuntimeError(
                f"Observed label has NaN: "
                f"{partition_name}, h={h}"
            )

        # Observed label must have timestamp.
        if (
            observed
            & label_ts.isna()
        ).any():
            raise RuntimeError(
                f"Observed label timestamp missing: "
                f"{partition_name}, h={h}"
            )

        valid = (
            observed
            & label_ts.notna()
        )

        expected_ts = (
            d.loc[
                valid,
                "origin_timestamp"
            ]
            + pd.Timedelta(
                hours=h
            )
        )

        actual_ts = (
            label_ts.loc[
                valid
            ]
        )

        if not (
            actual_ts.reset_index(
                drop=True
            )
            ==
            expected_ts.reset_index(
                drop=True
            )
        ).all():
            raise RuntimeError(
                f"Horizon timestamp mismatch: "
                f"{partition_name}, h={h}"
            )

        if (
            actual_ts
            <= d.loc[
                valid,
                "origin_timestamp"
            ]
        ).any():
            raise RuntimeError(
                f"Non-future label: "
                f"{partition_name}, h={h}"
            )

        if (
            actual_ts
            > partition_end
        ).any():
            raise RuntimeError(
                f"Cross-partition label: "
                f"{partition_name}, h={h}"
            )

    if not any_observed.all():
        raise RuntimeError(
            f"Origin without valid label in {partition_name}."
        )

    row = {
        "partition":
            partition_name,
        "origins":
            len(d),
        "buildings":
            d["building_id"].nunique(),
        "sites":
            d["site_id"].nunique(),
    }

    for h in horizons:

        row[
            f"observed_labels_h{h}"
        ] = int(
            d[
                f"observed_t_plus_{h}"
            ].sum()
        )

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)

print()
print(
    summary_df.to_string(
        index=False
    )
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =============================================================================
# VERIFY AGAINST ORIGINAL ENERGY OBSERVATION MASKS
# =============================================================================

section("DIRECT LABEL MASK VERIFICATION")

# This is intentionally performed independently from the shift-generation
# stage. It checks that labels marked observed in the final index correspond
# to energy_observed=True in the original building parquet.

verification_failures = 0
verification_count = 0

for i, cohort_row in cohort.iterrows():

    building_id = str(
        cohort_row["building_id"]
    )

    source = pd.read_parquet(
        BDG2_DIR / f"{building_id}.parquet",
        columns=[
            "timestamp",
            "energy",
            "energy_observed",
        ],
    )

    source["timestamp"] = pd.to_datetime(
        source["timestamp"]
    )

    source["energy_observed"] = normalize_boolean(
        source["energy_observed"]
    )

    source = (
        source
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .set_index("timestamp")
        .sort_index()
    )

    for partition_name, output_path in output_files.items():

        idx = pd.read_parquet(
            output_path,
            filters=[
                (
                    "building_id",
                    "==",
                    building_id,
                )
            ],
        )

        if idx.empty:
            continue

        for h in horizons:

            observed = normalize_boolean(
                idx[
                    f"observed_t_plus_{h}"
                ]
            )

            if not observed.any():
                continue

            timestamps = pd.to_datetime(
                idx.loc[
                    observed,
                    f"label_timestamp_t_plus_{h}"
                ]
            )

            expected_mask = (
                source.reindex(
                    timestamps
                )["energy_observed"]
            )

            if expected_mask.isna().any():
                verification_failures += int(
                    expected_mask.isna().sum()
                )

            expected_mask = (
                expected_mask
                .fillna(False)
                .astype(bool)
            )

            failures = int(
                (~expected_mask).sum()
            )

            verification_failures += failures
            verification_count += len(
                expected_mask
            )

    if (
        (i + 1) % 100 == 0
        or i + 1 == len(cohort)
    ):
        print(
            f"Verified "
            f"{i + 1:,}/{len(cohort):,} buildings",
            flush=True,
        )


if verification_failures != 0:
    raise RuntimeError(
        f"Direct-label verification failed for "
        f"{verification_failures:,} labels."
    )

print(
    f"\nVerified directly observed status for "
    f"{verification_count:,} labels."
)

print("Direct-label mask verification: PASSED")


# =============================================================================
# BUILDING COVERAGE
# =============================================================================

section("BUILDING COVERAGE")

expected_buildings = set(
    cohort["building_id"].astype(str)
)

for partition_name, output_path in output_files.items():

    d = pd.read_parquet(
        output_path,
        columns=[
            "building_id"
        ],
    )

    present = set(
        d["building_id"].astype(str)
    )

    missing = (
        expected_buildings
        - present
    )

    print(
        f"{partition_name:12s}: "
        f"{len(present):,}/"
        f"{len(expected_buildings):,} buildings"
    )

    if missing:
        print(
            f"  Buildings without usable origins: "
            f"{len(missing):,}"
        )


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {
    "dataset":
        "BDG2",
    "purpose":
        "source forecasting window index",
    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
    "implementation":
        "vectorized pandas shift",
    "history_length_hours":
        history_hours,
    "forecast_horizons_hours":
        horizons,
    "frozen_buildings":
        len(cohort),
    "frozen_sites":
        int(
            cohort["site_id"].nunique()
        ),
    "source_target":
        "energy",
    "history_definition":
        "[t-23, ..., t]",
    "label_definition":
        "energy at t+h",
    "label_ground_truth_policy":
        "directly observed energy only",
    "imputed_labels_used":
        False,
    "future_observations_in_history":
        False,
    "cross_partition_windows":
        False,
    "index_storage":
        (
            "one row per building/origin with "
            "horizon-specific labels"
        ),
    "input_hashes": {
        str(
            COHORT_FILE.relative_to(ROOT)
        ):
            sha256_file(COHORT_FILE),

        str(
            PROTOCOL_FILE.relative_to(ROOT)
        ):
            sha256_file(PROTOCOL_FILE),

        str(
            NORMALIZATION_FILE.relative_to(ROOT)
        ):
            sha256_file(
                NORMALIZATION_FILE
            ),
    },
    "outputs": {},
}

for partition_name, output_path in output_files.items():

    d = pd.read_parquet(
        output_path
    )

    manifest["outputs"][
        partition_name
    ] = {
        "path":
            str(
                output_path.relative_to(
                    ROOT
                )
            ),
        "origins":
            len(d),
        "buildings":
            int(
                d[
                    "building_id"
                ].nunique()
            ),
        "sha256":
            sha256_file(
                output_path
            ),
    }

manifest["audit"] = {
    "path":
        str(
            AUDIT_FILE.relative_to(ROOT)
        ),
    "sha256":
        sha256_file(
            AUDIT_FILE
        ),
}

manifest["summary"] = {
    "path":
        str(
            SUMMARY_FILE.relative_to(ROOT)
        ),
    "sha256":
        sha256_file(
            SUMMARY_FILE
        ),
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


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("BDG2 SOURCE WINDOW INDEX COMPLETE")

total_origins = int(
    summary_df["origins"].sum()
)

total_labels = 0

for h in horizons:
    total_labels += int(
        summary_df[
            f"observed_labels_h{h}"
        ].sum()
    )

elapsed_total = (
    time.time()
    - processing_start
)

print(
    f"""
Frozen source buildings:
  {len(cohort):,}

Frozen source sites:
  {cohort['site_id'].nunique():,}

History:
  {history_hours} hours

Forecast horizons:
  {", ".join(map(str, horizons))} hours

Total retained forecast origins:
  {total_origins:,}

Total directly observed horizon-specific labels:
  {total_labels:,}

Processing time:
  {elapsed_total / 60:.2f} minutes

Generated:

  {output_files['train'].relative_to(ROOT)}
  {output_files['validation'].relative_to(ROOT)}
  {output_files['test'].relative_to(ROOT)}

  {AUDIT_FILE.relative_to(ROOT)}
  {SUMMARY_FILE.relative_to(ROOT)}
  {MANIFEST_FILE.relative_to(ROOT)}

CAUSALITY / LEAKAGE STATUS

  24-hour history stays inside partition: PASSED
  Labels occur strictly after origin: PASSED
  Exact horizon timestamps: PASSED
  Labels stay inside partition: PASSED
  Ground-truth labels directly observed: PASSED
  Imputed labels used as ground truth: NO
  Future observations used in history: NO
  Cross-partition windows: NO

No target-domain data was used.

No normalization was refitted.

No model was trained.

No forecasting performance was inspected.
"""
)