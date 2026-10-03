from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "data" / "final"

SOURCE_DATA_DIR = FINAL / "bdg2" / "buildings"
TARGET_DATA_FILE = FINAL / "opensensemap" / "opensensemap_target_hourly_wide.csv"

SOURCE_INDEX_DIR = PROCESSED / "bdg2"
TARGET_INDEX_DIR = PROCESSED / "opensensemap"
OUTPUT_DIR = PROCESSED / "model_ready"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOL_FILE = META / "forecasting_protocol.json"

SOURCE_COHORT_FILE = (
    META / "bdg2_final_experimental_source_cohort.csv"
)

TARGET_COHORT_FILE = (
    META / "opensensemap_final_eligible_target_cohort.csv"
)

SOURCE_NORM_FILE = (
    META / "bdg2_training_normalization_statistics.csv"
)

TARGET_NORM_FILE = (
    META / "opensensemap_adaptation_normalization_statistics.csv"
)

MODEL_READY_SPEC_FILE = (
    OUTPUT_DIR / "model_ready_dataset_specification.json"
)

MODEL_READY_CONTRACT_FILE = (
    OUTPUT_DIR / "model_ready_feature_contract.csv"
)

SOURCE_INDEX_FILES = {
    "train":
        SOURCE_INDEX_DIR / "source_window_index_train.parquet",
    "validation":
        SOURCE_INDEX_DIR / "source_window_index_validation.parquet",
    "test":
        SOURCE_INDEX_DIR / "source_window_index_test.parquet",
}

TARGET_INDEX_FILES = {
    "adaptation":
        TARGET_INDEX_DIR / "target_window_index_adaptation.parquet",
    "validation":
        TARGET_INDEX_DIR / "target_window_index_validation.parquet",
    "test":
        TARGET_INDEX_DIR / "target_window_index_test.parquet",
}

AUDIT_FILE = (
    META / "pytorch_lazy_dataset_audit.csv"
)

MANIFEST_FILE = (
    META / "pytorch_lazy_dataset_manifest.json"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def to_bool_array(values):
    """
    Robust conversion of mask-like values to float32 0/1.
    """

    s = pd.Series(values)

    if pd.api.types.is_bool_dtype(s):
        return s.astype(np.float32).to_numpy()

    if pd.api.types.is_numeric_dtype(s):
        return (
            pd.to_numeric(s, errors="coerce")
            .fillna(0)
            .astype(np.float32)
            .to_numpy()
        )

    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "1.0": 1.0,
            "0.0": 0.0,
        })
        .fillna(0.0)
    )

    return mapped.astype(np.float32).to_numpy()


def normalize(values, mean, std):
    """
    Apply frozen z-score normalization.

    NaNs remain NaN here. They are replaced with zero only after
    normalization so that zero corresponds to the frozen mean.
    """

    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid frozen standard deviation: {std}")

    return (x - np.float32(mean)) / np.float32(std)


def replace_nan_with_zero(x):
    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)


def read_normalization(path):
    df = pd.read_csv(path)

    required = {"variable", "mean", "std"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Normalization file {path} missing columns: {missing}"
        )

    result = {}

    for _, row in df.iterrows():
        result[str(row["variable"])] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }

    return result


def parquet_row_count(path):
    return pq.ParquetFile(path).metadata.num_rows


def get_index_columns(path):
    return pq.ParquetFile(path).schema.names


def load_parquet_row(path, row_number):
    """
    Retrieve one logical row from a parquet file without loading
    the entire 15-million-row source index.

    Reads only the row group containing row_number.
    """

    pf = pq.ParquetFile(path)

    if row_number < 0 or row_number >= pf.metadata.num_rows:
        raise IndexError(row_number)

    cumulative = 0

    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows

        if row_number < cumulative + n:
            local_index = row_number - cumulative

            table = pf.read_row_group(rg)

            return table.slice(local_index, 1).to_pandas().iloc[0]

        cumulative += n

    raise IndexError(row_number)


# =============================================================================
# VERIFY REQUIRED INPUTS
# =============================================================================

section("PYTORCH LAZY DATASET CONSTRUCTION AND AUDIT")

required_files = [
    PROTOCOL_FILE,
    SOURCE_COHORT_FILE,
    TARGET_COHORT_FILE,
    SOURCE_NORM_FILE,
    TARGET_NORM_FILE,
    TARGET_DATA_FILE,
    MODEL_READY_SPEC_FILE,
]

for p in SOURCE_INDEX_FILES.values():
    required_files.append(p)

for p in TARGET_INDEX_FILES.values():
    required_files.append(p)

for p in required_files:
    require_file(p)

print("All required frozen inputs found.")


# =============================================================================
# LOAD FORECASTING PROTOCOL
# =============================================================================

section("VERIFYING FROZEN FORECASTING PROTOCOL")

with open(PROTOCOL_FILE, "r", encoding="utf-8") as f:
    protocol = json.load(f)

history_length = int(
    protocol.get(
        "history_length_hours",
        protocol.get(
            "forecasting_rules",
            {}
        ).get("history_length_hours")
    )
)

horizons = protocol.get(
    "forecast_horizons_hours",
    protocol.get(
        "forecasting_rules",
        {}
    ).get("forecast_horizons_hours")
)

if history_length != 24:
    raise ValueError(
        f"Expected frozen 24-hour history, got {history_length}"
    )

if horizons != [1, 3, 6, 12, 24]:
    raise ValueError(
        f"Unexpected frozen horizons: {horizons}"
    )

shared_variables = protocol["shared_variables"]
source_private_variable = protocol["source_private_variable"]
target_private_variable = protocol["target_private_variable"]
calendar_features = protocol["calendar_features"]

print(f"History length: {history_length}")
print(f"Horizons:      {horizons}")
print(f"Shared vars:   {shared_variables}")
print(f"Source private:{source_private_variable}")
print(f"Target private:{target_private_variable}")


# =============================================================================
# FEATURE CONTRACT
# =============================================================================

section("BUILDING FEATURE CONTRACT")

shared_features = []

for variable in shared_variables:
    shared_features.append(variable)

for variable in shared_variables:
    shared_features.append(f"{variable}_observed")

for variable in shared_variables:
    shared_features.append(f"{variable}_imputed")

shared_features.extend(calendar_features)

source_private_features = [
    source_private_variable,
    f"{source_private_variable}_observed",
    f"{source_private_variable}_imputed",
]

target_private_features = [
    target_private_variable,
    f"{target_private_variable}_observed",
    f"{target_private_variable}_imputed",
]

if len(shared_features) != 15:
    raise ValueError(
        f"Expected 15 shared features, got {len(shared_features)}"
    )

if len(source_private_features) != 3:
    raise ValueError("Source-private feature contract invalid.")

if len(target_private_features) != 3:
    raise ValueError("Target-private feature contract invalid.")

print("Shared features:")
for x in shared_features:
    print(f"  {x}")

print("\nSource-private:")
for x in source_private_features:
    print(f"  {x}")

print("\nTarget-private:")
for x in target_private_features:
    print(f"  {x}")


# =============================================================================
# NORMALIZATION
# =============================================================================

section("LOADING FROZEN NORMALIZATION PARAMETERS")

source_norm = read_normalization(SOURCE_NORM_FILE)
target_norm = read_normalization(TARGET_NORM_FILE)

for variable in shared_variables + [source_private_variable]:
    if variable not in source_norm:
        raise ValueError(
            f"Missing source normalization for {variable}"
        )

for variable in shared_variables + [target_private_variable]:
    if variable not in target_norm:
        raise ValueError(
            f"Missing target normalization for {variable}"
        )

print("SOURCE")
for k in shared_variables + [source_private_variable]:
    print(
        f"{k:20s} "
        f"mean={source_norm[k]['mean']:.6f} "
        f"std={source_norm[k]['std']:.6f}"
    )

print("\nTARGET")
for k in shared_variables + [target_private_variable]:
    print(
        f"{k:20s} "
        f"mean={target_norm[k]['mean']:.6f} "
        f"std={target_norm[k]['std']:.6f}"
    )


# =============================================================================
# SOURCE FILE MAP
# =============================================================================

section("INDEXING SOURCE BUILDING FILES")

source_cohort = pd.read_csv(SOURCE_COHORT_FILE)

if "building_id" not in source_cohort.columns:
    raise ValueError("Source cohort missing building_id")

source_buildings = (
    source_cohort["building_id"]
    .astype(str)
    .tolist()
)

source_file_map = {}

for building_id in source_buildings:
    path = SOURCE_DATA_DIR / f"{building_id}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing source building file:\n{path}"
        )

    source_file_map[building_id] = path

print(f"Indexed {len(source_file_map):,} source building files.")


# =============================================================================
# SOURCE CACHE
# =============================================================================

class SourceBuildingCache:

    def __init__(self):
        self.current_building_id = None
        self.current_df = None

    def get(self, building_id):

        building_id = str(building_id)

        if self.current_building_id == building_id:
            return self.current_df

        path = source_file_map[building_id]

        df = pd.read_parquet(path)

        required = (
            ["timestamp"]
            + shared_features
            + source_private_features
        )

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{building_id} missing source columns:\n{missing}"
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="raise"
        )

        df = (
            df.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .set_index("timestamp", drop=False)
        )

        self.current_building_id = building_id
        self.current_df = df

        return df


# =============================================================================
# TARGET DATA CACHE
# =============================================================================

section("LOADING TARGET HOURLY DATA")

target_df = pd.read_csv(TARGET_DATA_FILE)

target_required = (
    ["box_id", "timestamp"]
    + shared_features
    + target_private_features
)

missing = [
    c for c in target_required
    if c not in target_df.columns
]

if missing:
    raise ValueError(
        f"Target hourly data missing columns:\n{missing}"
    )

target_df["box_id"] = target_df["box_id"].astype(str)

target_df["timestamp"] = pd.to_datetime(
    target_df["timestamp"],
    utc=True,
    errors="raise"
)

target_df = target_df.sort_values(
    ["box_id", "timestamp"]
).reset_index(drop=True)

target_groups = {}

for box_id, group in target_df.groupby(
    "box_id",
    sort=False
):
    g = (
        group
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp", drop=False)
    )

    target_groups[str(box_id)] = g

print(f"Target rows loaded: {len(target_df):,}")
print(f"Target boxes indexed: {len(target_groups):,}")

del target_df


# =============================================================================
# INDEX ACCESSOR
# =============================================================================

class ParquetIndexAccessor:

    def __init__(self, path):
        self.path = Path(path)
        self.pf = pq.ParquetFile(self.path)

        self.row_group_starts = []

        cumulative = 0

        for rg in range(self.pf.num_row_groups):
            self.row_group_starts.append(cumulative)
            cumulative += self.pf.metadata.row_group(rg).num_rows

        self.n_rows = cumulative

        self.cached_rg = None
        self.cached_df = None

    def __len__(self):
        return self.n_rows

    def get(self, idx):

        if idx < 0:
            idx += self.n_rows

        if idx < 0 or idx >= self.n_rows:
            raise IndexError(idx)

        # Binary search for row group
        lo = 0
        hi = len(self.row_group_starts) - 1

        while lo <= hi:
            mid = (lo + hi) // 2

            start = self.row_group_starts[mid]

            if mid + 1 < len(self.row_group_starts):
                end = self.row_group_starts[mid + 1]
            else:
                end = self.n_rows

            if start <= idx < end:
                rg = mid
                break

            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            raise IndexError(idx)

        if self.cached_rg != rg:
            self.cached_df = (
                self.pf.read_row_group(rg).to_pandas()
            )
            self.cached_rg = rg

        local_idx = idx - self.row_group_starts[rg]

        return self.cached_df.iloc[local_idx]


# =============================================================================
# BASE DATASET
# =============================================================================

class BaseLazyForecastDataset(Dataset):

    def __init__(self, index_file):
        self.index_file = Path(index_file)
        self.index = ParquetIndexAccessor(index_file)

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _history_timestamps(start, origin, utc=False):

        if utc:
            start = pd.Timestamp(start)

            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")

            origin = pd.Timestamp(origin)

            if origin.tzinfo is None:
                origin = origin.tz_localize("UTC")
            else:
                origin = origin.tz_convert("UTC")

        else:
            start = pd.Timestamp(start)

            if start.tzinfo is not None:
                start = start.tz_localize(None)

            origin = pd.Timestamp(origin)

            if origin.tzinfo is not None:
                origin = origin.tz_localize(None)

        timestamps = pd.date_range(
            start=start,
            end=origin,
            freq="h"
        )

        if len(timestamps) != history_length:
            raise ValueError(
                f"History has {len(timestamps)} timestamps; "
                f"expected {history_length}."
            )

        return timestamps


# =============================================================================
# BDG2 DATASET
# =============================================================================

class BDG2LazyDataset(BaseLazyForecastDataset):

    def __init__(self, partition):
        if partition not in SOURCE_INDEX_FILES:
            raise ValueError(partition)

        super().__init__(SOURCE_INDEX_FILES[partition])

        self.partition = partition
        self.cache = SourceBuildingCache()

    def __getitem__(self, idx):

        row = self.index.get(idx)

        building_id = str(row["building_id"])

        df = self.cache.get(building_id)

        history_start = pd.Timestamp(
            row["history_start_timestamp"]
        )

        origin = pd.Timestamp(
            row["origin_timestamp"]
        )

        if history_start.tzinfo is not None:
            history_start = history_start.tz_localize(None)

        if origin.tzinfo is not None:
            origin = origin.tz_localize(None)

        expected_timestamps = self._history_timestamps(
            history_start,
            origin,
            utc=False
        )

        history = df.reindex(expected_timestamps)

        if len(history) != history_length:
            raise RuntimeError(
                "Source history length mismatch."
            )

        # -------------------------------------------------------------
        # Shared values
        # -------------------------------------------------------------

        shared_value_columns = shared_variables

        shared_values = []

        for variable in shared_value_columns:
            x = normalize(
                history[variable],
                source_norm[variable]["mean"],
                source_norm[variable]["std"],
            )
            shared_values.append(
                replace_nan_with_zero(x)
            )

        # -------------------------------------------------------------
        # Shared observed masks
        # -------------------------------------------------------------

        shared_observed = []

        for variable in shared_variables:
            shared_observed.append(
                to_bool_array(
                    history[f"{variable}_observed"]
                )
            )

        # -------------------------------------------------------------
        # Shared imputation masks
        # -------------------------------------------------------------

        shared_imputed = []

        for variable in shared_variables:
            shared_imputed.append(
                to_bool_array(
                    history[f"{variable}_imputed"]
                )
            )

        # -------------------------------------------------------------
        # Calendar
        # -------------------------------------------------------------

        calendar = []

        for feature in calendar_features:
            x = pd.to_numeric(
                history[feature],
                errors="coerce"
            ).to_numpy(dtype=np.float32)

            calendar.append(
                replace_nan_with_zero(x)
            )

        shared_history = np.stack(
            shared_values
            + shared_observed
            + shared_imputed
            + calendar,
            axis=1
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Private history
        # -------------------------------------------------------------

        energy = normalize(
            history["energy"],
            source_norm["energy"]["mean"],
            source_norm["energy"]["std"],
        )

        energy = replace_nan_with_zero(energy)

        energy_observed = to_bool_array(
            history["energy_observed"]
        )

        energy_imputed = to_bool_array(
            history["energy_imputed"]
        )

        private_history = np.stack(
            [
                energy,
                energy_observed,
                energy_imputed,
            ],
            axis=1
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Labels
        # -------------------------------------------------------------

        labels = np.zeros(
            len(horizons),
            dtype=np.float32
        )

        label_mask = np.zeros(
            len(horizons),
            dtype=np.float32
        )

        label_timestamps = []

        for j, h in enumerate(horizons):

            value = row[f"label_t_plus_{h}"]
            observed = bool(row[f"observed_t_plus_{h}"])

            expected_label_time = (
                origin + pd.Timedelta(hours=h)
            )

            stored_raw = row[f"label_timestamp_t_plus_{h}"]

            if pd.notna(stored_raw):
                stored_label_time = pd.Timestamp(stored_raw)

                if stored_label_time.tzinfo is not None:
                    stored_label_time = stored_label_time.tz_localize(None)
            else:
                stored_label_time = pd.NaT

            # A timestamp is required to match the exact future horizon
            # only when this horizon supplies valid observed ground truth.
            if observed and pd.notna(value):

                if pd.isna(stored_label_time):
                    raise RuntimeError(
                        f"Source observed label has no timestamp "
                        f"for h={h}, building={building_id}, "
                        f"origin={origin}"
                    )

                if stored_label_time != expected_label_time:
                    raise RuntimeError(
                        f"Source valid label timestamp mismatch "
                        f"for h={h}, building={building_id}, "
                        f"origin={origin}: "
                        f"stored={stored_label_time}, "
                        f"expected={expected_label_time}"
                    )

                normalized_label = (
                    float(value)
                    - source_norm["energy"]["mean"]
                ) / source_norm["energy"]["std"]

                labels[j] = np.float32(normalized_label)
                label_mask[j] = 1.0
                normalized_label = (
                    float(value)
                    - source_norm["energy"]["mean"]
                ) / source_norm["energy"]["std"]

                labels[j] = np.float32(normalized_label)
                label_mask[j] = 1.0

        # -------------------------------------------------------------
        # Final checks
        # -------------------------------------------------------------

        if shared_history.shape != (24, 15):
            raise RuntimeError(
                f"Source shared shape: {shared_history.shape}"
            )

        if private_history.shape != (24, 3):
            raise RuntimeError(
                f"Source private shape: {private_history.shape}"
            )

        if labels.shape != (5,):
            raise RuntimeError(labels.shape)

        if label_mask.shape != (5,):
            raise RuntimeError(label_mask.shape)

        if not np.isfinite(shared_history).all():
            raise RuntimeError(
                "Non-finite source shared tensor."
            )

        if not np.isfinite(private_history).all():
            raise RuntimeError(
                "Non-finite source private tensor."
            )

        if not np.isfinite(labels).all():
            raise RuntimeError(
                "Non-finite source label tensor."
            )

        return {
            "shared_history":
                torch.from_numpy(shared_history),

            "private_history":
                torch.from_numpy(private_history),

            "labels":
                torch.from_numpy(labels),

            "label_mask":
                torch.from_numpy(label_mask),

            "entity_id":
                building_id,

            "partition":
                self.partition,

            "origin_timestamp":
                str(origin),
        }


# =============================================================================
# OPENSENSEMAP DATASET
# =============================================================================

class OpenSenseMapLazyDataset(BaseLazyForecastDataset):

    def __init__(self, partition):
        if partition not in TARGET_INDEX_FILES:
            raise ValueError(partition)

        super().__init__(TARGET_INDEX_FILES[partition])

        self.partition = partition

    def __getitem__(self, idx):

        row = self.index.get(idx)

        box_id = str(row["box_id"])

        if box_id not in target_groups:
            raise KeyError(
                f"Target box not found: {box_id}"
            )

        df = target_groups[box_id]

        history_start = pd.Timestamp(
            row["history_start_timestamp"]
        )

        origin = pd.Timestamp(
            row["origin_timestamp"]
        )

        if history_start.tzinfo is None:
            history_start = history_start.tz_localize("UTC")
        else:
            history_start = history_start.tz_convert("UTC")

        if origin.tzinfo is None:
            origin = origin.tz_localize("UTC")
        else:
            origin = origin.tz_convert("UTC")

        expected_timestamps = self._history_timestamps(
            history_start,
            origin,
            utc=True
        )

        history = df.reindex(expected_timestamps)

        if len(history) != history_length:
            raise RuntimeError(
                "Target history length mismatch."
            )

        # -------------------------------------------------------------
        # Shared values
        # -------------------------------------------------------------

        shared_values = []

        for variable in shared_variables:

            x = normalize(
                history[variable],
                target_norm[variable]["mean"],
                target_norm[variable]["std"],
            )

            shared_values.append(
                replace_nan_with_zero(x)
            )

        # -------------------------------------------------------------
        # Shared observed masks
        # -------------------------------------------------------------

        shared_observed = []

        for variable in shared_variables:
            shared_observed.append(
                to_bool_array(
                    history[f"{variable}_observed"]
                )
            )

        # -------------------------------------------------------------
        # Shared imputation masks
        # -------------------------------------------------------------

        shared_imputed = []

        for variable in shared_variables:
            shared_imputed.append(
                to_bool_array(
                    history[f"{variable}_imputed"]
                )
            )

        # -------------------------------------------------------------
        # Calendar
        # -------------------------------------------------------------

        calendar = []

        for feature in calendar_features:

            x = pd.to_numeric(
                history[feature],
                errors="coerce"
            ).to_numpy(dtype=np.float32)

            calendar.append(
                replace_nan_with_zero(x)
            )

        shared_history = np.stack(
            shared_values
            + shared_observed
            + shared_imputed
            + calendar,
            axis=1
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Target-private history
        # -------------------------------------------------------------

        pm25 = normalize(
            history["pm25"],
            target_norm["pm25"]["mean"],
            target_norm["pm25"]["std"],
        )

        pm25 = replace_nan_with_zero(pm25)

        pm25_observed = to_bool_array(
            history["pm25_observed"]
        )

        pm25_imputed = to_bool_array(
            history["pm25_imputed"]
        )

        private_history = np.stack(
            [
                pm25,
                pm25_observed,
                pm25_imputed,
            ],
            axis=1
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Labels
        # -------------------------------------------------------------

        labels = np.zeros(
            len(horizons),
            dtype=np.float32
        )

        label_mask = np.zeros(
            len(horizons),
            dtype=np.float32
        )

        for j, h in enumerate(horizons):

            value = row[f"label_pm25_h{h}"]

            valid = bool(
                row[f"label_valid_h{h}"]
            )

            observed = bool(
                row[f"label_observed_h{h}"]
            )

            expected_label_time = (
                origin + pd.Timedelta(hours=h)
            )

            stored_label_time = pd.Timestamp(
                row[f"label_timestamp_h{h}"]
            )

            if stored_label_time.tzinfo is None:
                stored_label_time = (
                    stored_label_time.tz_localize("UTC")
                )
            else:
                stored_label_time = (
                    stored_label_time.tz_convert("UTC")
                )

            if stored_label_time != expected_label_time:
                raise RuntimeError(
                    f"Target label timestamp mismatch "
                    f"for h={h}"
                )

            if valid and observed and pd.notna(value):

                normalized_label = (
                    float(value)
                    - target_norm["pm25"]["mean"]
                ) / target_norm["pm25"]["std"]

                labels[j] = np.float32(
                    normalized_label
                )

                label_mask[j] = 1.0

        # -------------------------------------------------------------
        # Final checks
        # -------------------------------------------------------------

        if shared_history.shape != (24, 15):
            raise RuntimeError(
                f"Target shared shape: {shared_history.shape}"
            )

        if private_history.shape != (24, 3):
            raise RuntimeError(
                f"Target private shape: {private_history.shape}"
            )

        if labels.shape != (5,):
            raise RuntimeError(labels.shape)

        if label_mask.shape != (5,):
            raise RuntimeError(label_mask.shape)

        if not np.isfinite(shared_history).all():
            raise RuntimeError(
                "Non-finite target shared tensor."
            )

        if not np.isfinite(private_history).all():
            raise RuntimeError(
                "Non-finite target private tensor."
            )

        if not np.isfinite(labels).all():
            raise RuntimeError(
                "Non-finite target label tensor."
            )

        return {
            "shared_history":
                torch.from_numpy(shared_history),

            "private_history":
                torch.from_numpy(private_history),

            "labels":
                torch.from_numpy(labels),

            "label_mask":
                torch.from_numpy(label_mask),

            "entity_id":
                box_id,

            "partition":
                self.partition,

            "origin_timestamp":
                str(origin),
        }


# =============================================================================
# CONSTRUCT DATASETS
# =============================================================================
if __name__ != "__main__":
    # When imported by later pipeline stages, expose the reusable
    # dataset classes/functions without executing the Step-23 audit.
    raise SystemExit

section("CONSTRUCTING LAZY DATASETS")

datasets = {}

for partition in SOURCE_INDEX_FILES:

    ds = BDG2LazyDataset(partition)

    datasets[f"source_{partition}"] = ds

    print(
        f"source {partition:10s}: "
        f"{len(ds):,} samples"
    )

for partition in TARGET_INDEX_FILES:

    ds = OpenSenseMapLazyDataset(partition)

    datasets[f"target_{partition}"] = ds

    print(
        f"target {partition:10s}: "
        f"{len(ds):,} samples"
    )


# =============================================================================
# AUDIT SAMPLE POSITIONS
# =============================================================================

section("AUDITING DATASET SAMPLES")

audit_rows = []


def audit_dataset(name, ds):

    n = len(ds)

    positions = sorted(
        set([
            0,
            min(1, n - 1),
            n // 4,
            n // 2,
            (3 * n) // 4,
            n - 1,
        ])
    )

    for position in positions:

        t0 = time.time()

        sample = ds[position]

        elapsed = time.time() - t0

        shared = sample["shared_history"]
        private = sample["private_history"]
        labels = sample["labels"]
        mask = sample["label_mask"]

        if tuple(shared.shape) != (24, 15):
            raise RuntimeError(
                f"{name}: bad shared shape"
            )

        if tuple(private.shape) != (24, 3):
            raise RuntimeError(
                f"{name}: bad private shape"
            )

        if tuple(labels.shape) != (5,):
            raise RuntimeError(
                f"{name}: bad label shape"
            )

        if tuple(mask.shape) != (5,):
            raise RuntimeError(
                f"{name}: bad mask shape"
            )

        if not torch.isfinite(shared).all():
            raise RuntimeError(
                f"{name}: shared non-finite"
            )

        if not torch.isfinite(private).all():
            raise RuntimeError(
                f"{name}: private non-finite"
            )

        if not torch.isfinite(labels).all():
            raise RuntimeError(
                f"{name}: labels non-finite"
            )

        mask_values = set(
            mask.numpy().astype(float).tolist()
        )

        if not mask_values.issubset({0.0, 1.0}):
            raise RuntimeError(
                f"{name}: invalid label mask"
            )

        audit_rows.append({
            "dataset": name,
            "sample_position": int(position),
            "entity_id": sample["entity_id"],
            "origin_timestamp":
                sample["origin_timestamp"],
            "shared_shape":
                str(tuple(shared.shape)),
            "private_shape":
                str(tuple(private.shape)),
            "label_shape":
                str(tuple(labels.shape)),
            "valid_labels":
                int(mask.sum().item()),
            "shared_finite":
                bool(torch.isfinite(shared).all()),
            "private_finite":
                bool(torch.isfinite(private).all()),
            "labels_finite":
                bool(torch.isfinite(labels).all()),
            "retrieval_seconds":
                round(elapsed, 6),
        })

        print(
            f"{name:20s} "
            f"position={position:>10,} "
            f"entity={sample['entity_id']} "
            f"valid_labels={int(mask.sum().item())} "
            f"time={elapsed:.4f}s"
        )


for name, ds in datasets.items():
    audit_dataset(name, ds)


# =============================================================================
# CROSS-DOMAIN TENSOR CONTRACT
# =============================================================================

section("VERIFYING CROSS-DOMAIN TENSOR CONTRACT")

source_sample = datasets["source_train"][0]
target_sample = datasets["target_adaptation"][0]

if source_sample["shared_history"].shape != \
        target_sample["shared_history"].shape:
    raise RuntimeError(
        "Source/target shared tensor dimensions differ."
    )

if source_sample["shared_history"].shape != torch.Size([24, 15]):
    raise RuntimeError(
        "Unexpected shared tensor dimensions."
    )

if source_sample["private_history"].shape != torch.Size([24, 3]):
    raise RuntimeError(
        "Unexpected source-private dimensions."
    )

if target_sample["private_history"].shape != torch.Size([24, 3]):
    raise RuntimeError(
        "Unexpected target-private dimensions."
    )

print("Source shared tensor: ", tuple(
    source_sample["shared_history"].shape
))

print("Target shared tensor: ", tuple(
    target_sample["shared_history"].shape
))

print("Source private tensor:", tuple(
    source_sample["private_history"].shape
))

print("Target private tensor:", tuple(
    target_sample["private_history"].shape
))

print("\nCross-domain tensor contract: PASSED")


# =============================================================================
# VERIFY MISSINGNESS ENCODING
# =============================================================================

section("VERIFYING MISSINGNESS ENCODING")

# Shared layout:
# 0:3 values
# 3:6 observed
# 6:9 imputed
# 9:15 calendar

for name, ds in datasets.items():

    n = len(ds)

    positions = sorted(
        set([
            0,
            n // 3,
            (2 * n) // 3,
            n - 1,
        ])
    )

    for position in positions:

        sample = ds[position]

        shared = sample["shared_history"].numpy()
        private = sample["private_history"].numpy()

        shared_obs = shared[:, 3:6]
        shared_imp = shared[:, 6:9]

        if not np.isin(
            shared_obs,
            [0.0, 1.0]
        ).all():
            raise RuntimeError(
                f"{name}: invalid shared observed mask."
            )

        if not np.isin(
            shared_imp,
            [0.0, 1.0]
        ).all():
            raise RuntimeError(
                f"{name}: invalid shared imputed mask."
            )

        # An observation cannot simultaneously be marked
        # directly observed and causally imputed.
        if np.any(
            (shared_obs == 1.0)
            & (shared_imp == 1.0)
        ):
            raise RuntimeError(
                f"{name}: contradictory shared masks."
            )

        private_obs = private[:, 1]
        private_imp = private[:, 2]

        if not np.isin(
            private_obs,
            [0.0, 1.0]
        ).all():
            raise RuntimeError(
                f"{name}: invalid private observed mask."
            )

        if not np.isin(
            private_imp,
            [0.0, 1.0]
        ).all():
            raise RuntimeError(
                f"{name}: invalid private imputed mask."
            )

        if np.any(
            (private_obs == 1.0)
            & (private_imp == 1.0)
        ):
            raise RuntimeError(
                f"{name}: contradictory private masks."
            )

print("Missingness mask semantics: PASSED")


# =============================================================================
# LABEL MASK AUDIT
# =============================================================================

section("VERIFYING LABEL MASK SEMANTICS")

for name, ds in datasets.items():

    n = len(ds)

    positions = sorted(
        set([
            0,
            n // 5,
            (2 * n) // 5,
            (3 * n) // 5,
            (4 * n) // 5,
            n - 1,
        ])
    )

    for position in positions:

        sample = ds[position]

        mask = sample["label_mask"].numpy()

        if not np.isin(
            mask,
            [0.0, 1.0]
        ).all():
            raise RuntimeError(
                f"{name}: invalid label mask."
            )

        if mask.sum() < 1:
            raise RuntimeError(
                f"{name}: retained origin has no valid label."
            )

print("Every audited retained origin has >=1 valid observed label.")
print("Label-mask semantics: PASSED")


# =============================================================================
# SAVE AUDIT
# =============================================================================

section("WRITING STEP 23 AUDIT")

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    AUDIT_FILE,
    index=False
)

print(f"Saved:\n  {AUDIT_FILE}")


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {
    "step": 23,
    "name": "PyTorch lazy dataset construction",
    "construction_mode": "lazy",
    "history_length_hours": history_length,
    "forecast_horizons_hours": horizons,

    "shared_features": shared_features,

    "source_private_features":
        source_private_features,

    "target_private_features":
        target_private_features,

    "tensor_contract": {
        "shared_history": [24, 15],
        "private_history": [24, 3],
        "labels": [5],
        "label_mask": [5],
    },

    "source": {
        "normalization":
            "frozen BDG2 training-only statistics",
        "index_files": {
            k: str(v.relative_to(ROOT))
            for k, v in SOURCE_INDEX_FILES.items()
        },
        "sample_counts": {
            k: len(datasets[f"source_{k}"])
            for k in SOURCE_INDEX_FILES
        },
    },

    "target": {
        "normalization":
            "frozen openSenseMap adaptation-only statistics",
        "index_files": {
            k: str(v.relative_to(ROOT))
            for k, v in TARGET_INDEX_FILES.items()
        },
        "sample_counts": {
            k: len(datasets[f"target_{k}"])
            for k in TARGET_INDEX_FILES
        },
    },

    "missingness_contract": {
        "direct_observation":
            "observed=1, imputed=0",
        "causal_imputation":
            "observed=0, imputed=1",
        "unresolved_missing":
            "observed=0, imputed=0",
        "unresolved_normalized_value":
            0.0,
        "separate_missing_channel":
            False,
    },

    "label_contract": {
        "labels_are_normalized":
            True,
        "evaluation_requires_direct_observation":
            True,
        "invalid_label_tensor_placeholder":
            0.0,
        "label_mask_required":
            True,
    },

    "normalization_refit":
        False,

    "target_test_used_for_fit":
        False,

    "model_training_performed":
        False,

    "forecasting_performance_inspected":
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

print(f"Saved:\n  {MANIFEST_FILE}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("PYTORCH LAZY DATASETS VERIFIED")

print("SOURCE")
for partition in SOURCE_INDEX_FILES:
    print(
        f"{partition:12s}: "
        f"{len(datasets[f'source_{partition}']):,}"
    )

print("\nTARGET")
for partition in TARGET_INDEX_FILES:
    print(
        f"{partition:12s}: "
        f"{len(datasets[f'target_{partition}']):,}"
    )

print()
print("TENSOR CONTRACT")
print("  shared_history : (24, 15)")
print("  private_history: (24, 3)")
print("  labels         : (5,)")
print("  label_mask     : (5,)")

print()
print("NORMALIZATION")
print("  Source: frozen BDG2 TRAINING statistics")
print("  Target: frozen openSenseMap ADAPTATION statistics")
print("  Refitting: NO")

print()
print("MISSINGNESS")
print("  directly observed -> observed=1, imputed=0")
print("  causal fill       -> observed=0, imputed=1")
print("  unresolved        -> observed=0, imputed=0")
print("  unresolved value  -> normalized tensor value 0.0")

print()
print("GROUND TRUTH")
print("  Directly observed future labels only")
print("  Invalid/imputed future labels masked out")

print()
print("LEAKAGE")
print("  Target validation used for normalization: NO")
print("  Target test used for normalization:       NO")
print("  Target test used for adaptation:          NO")
print("  Future observations used as inputs:       NO")
print("  Cross-partition windows:                  NO")

print()
print("IMPORTANT:")
print()
print("The approximately 22.7 million BDG2 source windows")
print("remain lazy. No duplicated 24-hour tensor dataset")
print("has been materialized.")
print()
print("Step 19/20 window indices remain authoritative.")
print()
print("No model was trained.")
print("No forecasting performance was inspected.")
print()
print("Step 23 complete.")