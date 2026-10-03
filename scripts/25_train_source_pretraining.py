from __future__ import annotations
import argparse
from tqdm.auto import tqdm
import gc
import json
import math
import random
import time
import hashlib
import importlib.util
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
METADATA = DATA / "metadata"
PROCESSED = DATA / "processed"
FINAL = DATA / "final"

SOURCE_BUILDING_DIR = FINAL / "bdg2" / "buildings"

TRAIN_INDEX = (
    PROCESSED
    / "bdg2"
    / "source_window_index_train.parquet"
)

VALIDATION_INDEX = (
    PROCESSED
    / "bdg2"
    / "source_window_index_validation.parquet"
)

SOURCE_COHORT = (
    METADATA
    / "bdg2_final_experimental_source_cohort.csv"
)

SOURCE_NORMALIZATION = (
    METADATA
    / "bdg2_training_normalization_statistics.csv"
)

FORECAST_PROTOCOL = (
    METADATA
    / "forecasting_protocol.json"
)

STEP24_PATH = (
    ROOT
    / "scripts"
    / "24_define_transfer_model.py"
)

BASE_OUTPUT_DIR = ROOT / "models" / "source_pretraining"
BASE_AUDIT_DIR = METADATA / "source_pretraining"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 25 - BDG2 source pretraining (multi-seed safe)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Independent-run seed (e.g. 42..61)."
    )
    return parser.parse_args()


ARGS = parse_args()
SEED = int(ARGS.seed)

# Every independent run has an isolated artifact directory.
OUTPUT_DIR = BASE_OUTPUT_DIR / f"seed_{SEED}"
AUDIT_DIR = BASE_AUDIT_DIR / f"seed_{SEED}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# SEED is parsed from --seed above.
# Keep the validation subset fixed across independent runs so that model
# selection is compared on exactly the same source-validation examples.
VALIDATION_SUBSET_SEED = 10_042

# RTX 5050 Laptop / \~8 GB VRAM.
BATCH_SIZE = 512

# DataLoader workers.
#
# Windows + parquet access can behave poorly with too many workers.
# Start conservatively.

NUM_WORKERS = 0
PIN_MEMORY = True
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CLIP_NORM = 1.0

# Adjacent hourly origins are highly overlapping.
#
# Rather than processing all \~15.1M origins each epoch,
# deterministically sample a large subset without replacement.
#
# 1,000,000 windows/epoch gives substantial coverage while
# remaining computationally practical.

TRAIN_SAMPLES_PER_EPOCH = 200_000

# Validation is also very large (\~3.84M origins).
# Freeze a deterministic validation subset ONCE.

VALIDATION_SAMPLES = 50_000

# Report progress periodically.

LOG_EVERY_BATCHES = 250

# Cache a limited number of building parquet files.

BUILDING_CACHE_SIZE = 64

# AMP is enabled only on CUDA.

USE_AMP = True

# Optimizer scheduler.

MIN_LR = 1e-6

# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen input not found:\n{path}"
        )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility-oriented settings.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()

def atomic_torch_save(obj, path):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(obj, tmp)
    tmp.replace(path)

def load_step24_module():
    spec = importlib.util.spec_from_file_location(
        "step24_model",
        STEP24_PATH
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create Step-24 module specification."
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module

def read_normalization(path):
    df = pd.read_csv(path)

    required = {
        "variable",
        "mean",
        "std",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Normalization file missing columns: {missing}"
        )

    result = {}

    for _, row in df.iterrows():
        variable = str(row["variable"])

        mean = float(row["mean"])
        std = float(row["std"])

        if not np.isfinite(mean):
            raise ValueError(
                f"Non-finite mean for {variable}"
            )

        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid std for {variable}: {std}"
            )

        result[variable] = {
            "mean": mean,
            "std": std,
        }

    return result

def normalize_array(values, mean, std):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    result = (
        values - np.float32(mean)
    ) / np.float32(std)

    return result.astype(
        np.float32,
        copy=False
    )

def zero_nonfinite(values):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32,
        copy=False
    )

def bool_float(values):
    s = pd.Series(values)

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(
            np.float32
        ).to_numpy()

    numeric = pd.to_numeric(
        s,
        errors="coerce"
    ).fillna(0.0)

    return (
        numeric.to_numpy() > 0
    ).astype(np.float32)

# =============================================================================
# LOAD FROZEN CONFIGURATION
# =============================================================================

set_seed(SEED)

banner("STEP 25 - BDG2 SOURCE PRETRAINING")

for required_path in [
    TRAIN_INDEX,
    VALIDATION_INDEX,
    SOURCE_COHORT,
    SOURCE_NORMALIZATION,
    FORECAST_PROTOCOL,
    STEP24_PATH,
]:
    require_file(required_path)

print(f"Independent run seed: {SEED}")
print(f"Run output directory: {OUTPUT_DIR}")
print(f"Run audit directory:  {AUDIT_DIR}")
print("All required source-domain frozen inputs found.")

with open(
    FORECAST_PROTOCOL,
    "r",
    encoding="utf-8"
) as f:
    protocol = json.load(f)




HISTORY_LENGTH = int(

    protocol["history_length_hours"]

)



HORIZONS = [

    int(x)

    for x in protocol[

        "forecast_horizons_hours"

    ]

]



SHARED_VARIABLES = list(

    protocol["shared_variables"]

)



SOURCE_PRIVATE_VARIABLE = str(

    protocol["source_private_variable"]

)



CALENDAR_FEATURES = list(

    protocol["calendar_features"]

)





if HISTORY_LENGTH != 24:

    raise ValueError(

        f"Expected frozen history length 24; "

        f"found {HISTORY_LENGTH}."

    )



if HORIZONS != [1, 3, 6, 12, 24]:

    raise ValueError(

        f"Unexpected frozen horizons: {HORIZONS}"

    )



if SHARED_VARIABLES != [

    "temperature",

    "relative_humidity",

    "pressure",

]:

    raise ValueError(

        f"Unexpected shared variables: "

        f"{SHARED_VARIABLES}"

    )



if SOURCE_PRIVATE_VARIABLE != "energy":

    raise ValueError(

        "Source private variable is not energy."

    )





banner("FROZEN FORECASTING CONTRACT")



print(f"History length:    {HISTORY_LENGTH}")

print(f"Horizons:          {HORIZONS}")

print(f"Shared variables:  {SHARED_VARIABLES}")

print(f"Source private:    {SOURCE_PRIVATE_VARIABLE}")

print(f"Calendar features: {CALENDAR_FEATURES}")





# =============================================================================

# STRICT SOURCE-ONLY SAFETY CHECK

# =============================================================================



banner("SOURCE-ONLY LEAKAGE BOUNDARY")



# Step 25 intentionally defines no target-data path.

#

# This script must never read:

#   opensensemap_target_hourly_wide.csv

#   target adaptation index

#   target validation index

#   target test index

#   target normalization statistics

#

# It also intentionally does not define the BDG2 test-index path.



print("BDG2 train index:      ACCESSIBLE")

print("BDG2 validation index: ACCESSIBLE")

print("BDG2 test index:       NOT DEFINED")

print("openSenseMap data:     NOT DEFINED")

print("openSenseMap indices:  NOT DEFINED")

print("Target normalization:  NOT DEFINED")





# =============================================================================

# LOAD NORMALIZATION

# =============================================================================



banner("LOADING FROZEN SOURCE NORMALIZATION")



SOURCE_NORM = read_normalization(

    SOURCE_NORMALIZATION

)



for variable in [

    "temperature",

    "relative_humidity",

    "pressure",

    "energy",

]:

    if variable not in SOURCE_NORM:

        raise ValueError(

            f"Missing normalization for {variable}"

        )



    print(

        f"{variable:<20} "

        f"mean={SOURCE_NORM[variable]['mean']:.6f} "

        f"std={SOURCE_NORM[variable]['std']:.6f}"

    )





# =============================================================================

# VERIFY FROZEN SOURCE COHORT

# =============================================================================



banner("VERIFYING FROZEN SOURCE COHORT")



cohort = pd.read_csv(

    SOURCE_COHORT

)



if "building_id" not in cohort.columns:

    raise ValueError(

        "Frozen source cohort lacks building_id."

    )



frozen_buildings = (

    cohort["building_id"]

    .astype(str)

    .tolist()

)



if len(frozen_buildings) != 1318:

    raise ValueError(

        f"Expected 1,318 buildings; "

        f"found {len(frozen_buildings)}."

    )



if len(set(frozen_buildings)) != len(

    frozen_buildings

):

    raise ValueError(

        "Duplicate building IDs in frozen cohort."

    )



print(

    f"Frozen source buildings: "

    f"{len(frozen_buildings):,}"

)





# =============================================================================

# SOURCE BUILDING FILE MAP

# =============================================================================



banner("INDEXING SOURCE BUILDING FILES")



BUILDING_FILES = {}



for path in SOURCE_BUILDING_DIR.glob(

    "*.parquet"

):

    BUILDING_FILES[path.stem] = path





missing_buildings = [

    building_id

    for building_id in frozen_buildings

    if building_id not in BUILDING_FILES

]



if missing_buildings:

    raise FileNotFoundError(

        "Missing source building parquet files. "

        f"Examples: {missing_buildings[:10]}"

    )



print(

    f"Indexed {len(BUILDING_FILES):,} "

    f"building parquet files."

)





# =============================================================================

# PARQUET INDEX ACCESSOR

# =============================================================================



class ParquetRowGroupAccessor:



    def __init__(self, path):

        self.path = Path(path)



        self.file = pq.ParquetFile(

            self.path

        )



        self.metadata = self.file.metadata



        self.row_group_starts = []



        cumulative = 0



        for rg in range(

            self.metadata.num_row_groups

        ):

            self.row_group_starts.append(

                cumulative

            )



            cumulative += (

                self.metadata

                .row_group(rg)

                .num_rows

            )



        self.total_rows = cumulative



        self.cached_row_group_id = None

        self.cached_row_group = None



    def __len__(self):

        return self.total_rows



    def get(self, idx):



        if idx < 0:

            idx += self.total_rows



        if idx < 0 or idx >= self.total_rows:

            raise IndexError(idx)



        rg = int(

            np.searchsorted(

                self.row_group_starts,

                idx,

                side="right"

            ) - 1

        )



        if rg < 0:

            rg = 0



        local_idx = (

            idx

            - self.row_group_starts[rg]

        )



        if self.cached_row_group_id != rg:



            table = self.file.read_row_group(

                rg

            )



            self.cached_row_group = (

                table.to_pandas()

            )



            self.cached_row_group_id = rg



        return self.cached_row_group.iloc[

            local_idx

        ]





# =============================================================================

# BUILDING CACHE

# =============================================================================



class BuildingCache:



    def __init__(

        self,

        max_size=BUILDING_CACHE_SIZE

    ):

        self.max_size = max_size



        self.cache = OrderedDict()



    def get(self, building_id):



        if building_id in self.cache:

            df = self.cache.pop(

                building_id

            )



            self.cache[

                building_id

            ] = df



            return df



        path = BUILDING_FILES.get(

            building_id

        )



        if path is None:

            raise FileNotFoundError(

                f"No parquet file for "

                f"{building_id}"

            )



        df = pd.read_parquet(path)



        required = {

            "timestamp",

            "temperature",

            "relative_humidity",

            "pressure",

            "energy",

            "temperature_observed",

            "relative_humidity_observed",

            "pressure_observed",

            "energy_observed",

            "temperature_imputed",

            "relative_humidity_imputed",

            "pressure_imputed",

            "energy_imputed",

            *CALENDAR_FEATURES,

        }



        missing = required - set(

            df.columns

        )



        if missing:

            raise ValueError(

                f"{building_id} missing columns: "

                f"{sorted(missing)}"

            )



        timestamps = pd.to_datetime(

            df["timestamp"],

            errors="raise"

        )



        if timestamps.dt.tz is not None:

            timestamps = (

                timestamps

                .dt

                .tz_localize(None)

            )



        df = df.copy()



        df.index = pd.DatetimeIndex(

            timestamps

        )



        if not df.index.is_monotonic_increasing:

            df = df.sort_index()



        if df.index.has_duplicates:

            raise ValueError(

                f"Duplicate timestamps in "

                f"{building_id}"

            )



        self.cache[

            building_id

        ] = df



        while (

            len(self.cache)

            > self.max_size

        ):

            self.cache.popitem(

                last=False

            )



        return df





# =============================================================================

# SOURCE LAZY DATASET

# =============================================================================



class SourceLazyDataset(Dataset):



    def __init__(self, index_path):



        self.index_path = Path(

            index_path

        )



        self.index = (

            ParquetRowGroupAccessor(

                self.index_path

            )

        )



        self.cache = BuildingCache()



    def __len__(self):

        return len(self.index)



    def __getitem__(self, idx):



        row = self.index.get(

            int(idx)

        )



        building_id = str(

            row["building_id"]

        )



        history_start = pd.Timestamp(

            row["history_start_timestamp"]

        )



        origin = pd.Timestamp(

            row["origin_timestamp"]

        )



        if history_start.tzinfo is not None:

            history_start = (

                history_start.tz_localize(

                    None

                )

            )



        if origin.tzinfo is not None:

            origin = origin.tz_localize(

                None

            )



        expected_index = pd.date_range(

            start=history_start,

            end=origin,

            freq="h"

        )



        if len(expected_index) != HISTORY_LENGTH:

            raise RuntimeError(

                f"Expected {HISTORY_LENGTH} "

                f"history hours, got "

                f"{len(expected_index)}."

            )



        building = self.cache.get(

            building_id

        )



        history = building.reindex(

            expected_index

        )



        if len(history) != HISTORY_LENGTH:

            raise RuntimeError(

                "History reindex length mismatch."

            )



        # -------------------------------------------------------------

        # Shared physical values

        # -------------------------------------------------------------



        shared_values = []



        for variable in SHARED_VARIABLES:



            x = normalize_array(

                history[

                    variable

                ].to_numpy(),

                SOURCE_NORM[

                    variable

                ]["mean"],

                SOURCE_NORM[

                    variable

                ]["std"],

            )



            shared_values.append(

                zero_nonfinite(x)

            )



        shared_values = np.stack(

            shared_values,

            axis=1

        )



        # -------------------------------------------------------------

        # Shared observation masks

        # -------------------------------------------------------------



        shared_observed = np.stack(

            [

                bool_float(

                    history[

                        f"{variable}_observed"

                    ].to_numpy()

                )

                for variable

                in SHARED_VARIABLES

            ],

            axis=1

        )



        # -------------------------------------------------------------

        # Shared imputation masks

        # -------------------------------------------------------------



        shared_imputed = np.stack(

            [

                bool_float(

                    history[

                        f"{variable}_imputed"

                    ].to_numpy()

                )

                for variable

                in SHARED_VARIABLES

            ],

            axis=1

        )



        # -------------------------------------------------------------

        # Deterministic calendar features

        # -------------------------------------------------------------



        calendar = np.stack(

            [

                zero_nonfinite(

                    pd.to_numeric(

                        history[

                            feature

                        ],

                        errors="coerce"

                    ).to_numpy(

                        dtype=np.float32

                    )

                )

                for feature

                in CALENDAR_FEATURES

            ],

            axis=1

        )



        shared_history = np.concatenate(

            [

                shared_values,

                shared_observed,

                shared_imputed,

                calendar,

            ],

            axis=1

        ).astype(

            np.float32,

            copy=False

        )



        # -------------------------------------------------------------

        # Source-private energy

        # -------------------------------------------------------------



        energy = normalize_array(

            history[

                "energy"

            ].to_numpy(),

            SOURCE_NORM[

                "energy"

            ]["mean"],

            SOURCE_NORM[

                "energy"

            ]["std"],

        )



        energy = zero_nonfinite(

            energy

        )



        energy_observed = bool_float(

            history[

                "energy_observed"

            ].to_numpy()

        )



        energy_imputed = bool_float(

            history[

                "energy_imputed"

            ].to_numpy()

        )



        private_history = np.stack(

            [

                energy,

                energy_observed,

                energy_imputed,

            ],

            axis=1

        ).astype(

            np.float32,

            copy=False

        )



        # -------------------------------------------------------------
        # Future labels
        # -------------------------------------------------------------

        labels = []
        label_mask = []

        for h in HORIZONS:
            label_column = f"label_t_plus_{h}"
            observed_column = f"observed_t_plus_{h}"
            timestamp_column = f"label_timestamp_t_plus_{h}"

            observed = bool(row[observed_column])
            value = row[label_column]

            # Only usable, directly observed labels are required to match
            # the nominal t+h timestamp. Horizons crossing a frozen partition
            # boundary remain masked and are never pulled from the next split.
            if observed:
                expected_label_timestamp = origin + pd.Timedelta(hours=h)
                actual_timestamp_value = row[timestamp_column]

                if pd.isna(actual_timestamp_value):
                    raise RuntimeError(
                        f"Observed label has no timestamp for h={h}, "
                        f"building={building_id}, origin={origin}."
                    )

                actual_label_timestamp = pd.Timestamp(actual_timestamp_value)

                if actual_label_timestamp.tzinfo is not None:
                    actual_label_timestamp = actual_label_timestamp.tz_localize(None)

                if actual_label_timestamp != expected_label_timestamp:
                    raise RuntimeError(
                        f"Observed label timestamp mismatch for h={h}, "
                        f"building={building_id}, origin={origin}. "
                        f"Expected {expected_label_timestamp}, "
                        f"found {actual_label_timestamp}."
                    )

                if pd.isna(value):
                    raise RuntimeError(
                        f"Observed label is NaN for h={h}, "
                        f"building={building_id}, origin={origin}."
                    )

                normalized_label = (
                    float(value) - SOURCE_NORM["energy"]["mean"]
                ) / SOURCE_NORM["energy"]["std"]

                labels.append(np.float32(normalized_label))
                label_mask.append(np.float32(1.0))

            else:
                # Invalid/unobserved horizons remain masked.
                # No cross-partition label is retrieved or substituted.
                labels.append(np.float32(0.0))
                label_mask.append(np.float32(0.0))

        labels = np.asarray(

            labels,

            dtype=np.float32

        )



        label_mask = np.asarray(

            label_mask,

            dtype=np.float32

        )



        if label_mask.sum() <= 0:

            raise RuntimeError(

                "Retained source origin has "

                "no valid label."

            )



        if shared_history.shape != (

            HISTORY_LENGTH,

            15

        ):

            raise RuntimeError(

                f"Unexpected shared tensor "

                f"shape: "

                f"{shared_history.shape}"

            )



        if private_history.shape != (

            HISTORY_LENGTH,

            3

        ):

            raise RuntimeError(

                f"Unexpected private tensor "

                f"shape: "

                f"{private_history.shape}"

            )



        return {

            "shared_history":

                torch.from_numpy(

                    shared_history

                ),



            "private_history":

                torch.from_numpy(

                    private_history

                ),



            "labels":

                torch.from_numpy(

                    labels

                ),



            "label_mask":

                torch.from_numpy(

                    label_mask

                ),



            "building_id":

                building_id,



            "origin_timestamp":

                str(origin),

        }





# =============================================================================

# DETERMINISTIC SUBSET SAMPLER

# =============================================================================



class DeterministicEpochSampler(Sampler):

    # Cache-friendly deterministic block sampler. This preserves the frozen
    # source partition and changes only which training origins are visited.
    def __init__(self, dataset_size, sample_size, seed, block_size=2048):
        self.dataset_size = int(dataset_size)
        self.sample_size = min(int(sample_size), self.dataset_size)
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.sample_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        n_blocks = math.ceil(self.dataset_size / self.block_size)
        needed = math.ceil(self.sample_size / self.block_size)
        blocks = torch.randperm(n_blocks, generator=generator)[:needed].tolist()
        indices = []
        for block in blocks:
            lo = block * self.block_size
            hi = min(lo + self.block_size, self.dataset_size)
            indices.extend(range(lo, hi))
            if len(indices) >= self.sample_size:
                break
        return iter(indices[:self.sample_size])


class FixedValidationSampler(Sampler):

    # Frozen cache-friendly validation subset.
    def __init__(self, dataset_size, sample_size, seed, block_size=2048):
        self.dataset_size = int(dataset_size)
        self.sample_size = min(int(sample_size), self.dataset_size)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        n_blocks = math.ceil(self.dataset_size / int(block_size))
        needed = math.ceil(self.sample_size / int(block_size))
        blocks = torch.randperm(n_blocks, generator=generator)[:needed].tolist()
        indices = []
        for block in blocks:
            lo = block * int(block_size)
            hi = min(lo + int(block_size), self.dataset_size)
            indices.extend(range(lo, hi))
            if len(indices) >= self.sample_size:
                break
        self.indices = indices[:self.sample_size]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


# =============================================================================

# BUILD DATASETS

# =============================================================================



banner("CONSTRUCTING SOURCE-ONLY LAZY DATASETS")



train_dataset = SourceLazyDataset(

    TRAIN_INDEX

)



validation_dataset = SourceLazyDataset(

    VALIDATION_INDEX

)



print(

    f"Train origins:      "

    f"{len(train_dataset):,}"

)



print(

    f"Validation origins: "

    f"{len(validation_dataset):,}"

)





if len(train_dataset) != 15_089_137:

    raise ValueError(

        "Unexpected source train count."

    )



if len(validation_dataset) != 3_844_682:

    raise ValueError(

        "Unexpected source validation count."

    )





# =============================================================================

# AUDIT SAMPLES BEFORE TRAINING

# =============================================================================



banner("PRETRAINING DATA AUDIT")



audit_positions = [

    0,

    len(train_dataset) // 2,

    len(train_dataset) - 1,

]



for position in audit_positions:



    sample = train_dataset[

        position

    ]



    print(

        f"position={position:,} "

        f"building={sample['building_id']} "

        f"shared="

        f"{tuple(sample['shared_history'].shape)} "

        f"private="

        f"{tuple(sample['private_history'].shape)} "

        f"valid_labels="

        f"{int(sample['label_mask'].sum().item())}"

    )





# =============================================================================

# SAMPLERS AND LOADERS

# =============================================================================



train_sampler = DeterministicEpochSampler(

    dataset_size=len(

        train_dataset

    ),

    sample_size=TRAIN_SAMPLES_PER_EPOCH,

    seed=SEED,

)



validation_sampler = FixedValidationSampler(

    dataset_size=len(

        validation_dataset

    ),

    sample_size=VALIDATION_SAMPLES,

    seed=VALIDATION_SUBSET_SEED,

)





train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=train_sampler,

    num_workers=NUM_WORKERS,

    pin_memory=(

        PIN_MEMORY

        and torch.cuda.is_available()

    ),

    drop_last=False,

)



validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    sampler=validation_sampler,

    num_workers=NUM_WORKERS,

    pin_memory=(

        PIN_MEMORY

        and torch.cuda.is_available()

    ),

    drop_last=False,

)





print()

print(

    f"Training samples/epoch: "

    f"{len(train_sampler):,}"

)



print(

    f"Validation samples:     "

    f"{len(validation_sampler):,}"

)



print(

    f"Training batches/epoch: "

    f"{len(train_loader):,}"

)



print(

    f"Validation batches:     "

    f"{len(validation_loader):,}"

)





# =============================================================================

# MODEL

# =============================================================================



banner("INITIALIZING FROZEN STEP-24 MODEL")



step24 = load_step24_module()



model = (

    step24

    .HeterogeneousTransferForecaster()

)



device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)



model = model.to(

    device

)





print(f"Device: {device}")



if device.type == "cuda":

    print(

        "GPU:    "

        + torch.cuda.get_device_name(0)

    )



    print(

        "VRAM:   "

        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"

    )





# =============================================================================

# FREEZE TARGET-SPECIFIC COMPONENTS

# =============================================================================



banner("FREEZING TARGET-SPECIFIC COMPONENTS")



for parameter in (

    model

    .target_private_encoder

    .parameters()

):

    parameter.requires_grad = False



for parameter in (

    model

    .target_fusion

    .parameters()

):

    parameter.requires_grad = False



for parameter in (

    model

    .target_head

    .parameters()

):

    parameter.requires_grad = False





trainable_components = [

    model.shared_encoder,

    model.source_private_encoder,

    model.source_fusion,

    model.source_head,

]



trainable_parameters = []



for component in trainable_components:

    trainable_parameters.extend(

        list(

            component.parameters()

        )

    )





n_trainable = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad

)



n_frozen = sum(

    p.numel()

    for p in model.parameters()

    if not p.requires_grad

)



print(

    f"Trainable parameters: "

    f"{n_trainable:,}"

)



print(

    f"Frozen target-only parameters: "

    f"{n_frozen:,}"

)



print()

print("TRAINABLE")

print("  shared_encoder")

print("  source_private_encoder")

print("  source_fusion")

print("  source_head")



print()

print("FROZEN / UNUSED")

print("  target_private_encoder")

print("  target_fusion")

print("  target_head")





# =============================================================================

# OPTIMIZER / AMP

# =============================================================================



optimizer = torch.optim.AdamW(

    trainable_parameters,

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    fused=(device.type == "cuda"),

)



scheduler = (

    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=2,

        min_lr=MIN_LR,

    )

)





amp_enabled = (

    USE_AMP

    and device.type == "cuda"

)



# torch.amp API for current PyTorch.

scaler = torch.amp.GradScaler(

    "cuda",

    enabled=amp_enabled

)





print()

print(

    f"Mixed precision: "

    f"{'ENABLED' if amp_enabled else 'DISABLED'}"

)





# =============================================================================

# LOSS ACCUMULATOR

# =============================================================================



def masked_error_statistics(

    prediction,

    target,

    mask

):

    error = (

        prediction - target

    )



    squared = (

        error ** 2

    ) * mask



    absolute = (

        error.abs()

    ) * mask



    return (

        squared.sum().item(),

        absolute.sum().item(),

        mask.sum().item(),

    )





def horizon_statistics(

    prediction,

    target,

    mask

):



    stats = []



    for j, h in enumerate(

        HORIZONS

    ):



        m = mask[:, j]



        n = m.sum().item()



        if n > 0:



            error = (

                prediction[:, j]

                - target[:, j]

            )



            sse = (

                (

                    error ** 2

                )

                * m

            ).sum().item()



            sae = (

                error.abs()

                * m

            ).sum().item()



        else:

            sse = 0.0

            sae = 0.0



        stats.append(

            {

                "horizon": h,

                "sse": sse,

                "sae": sae,

                "n": n,

            }

        )



    return stats





# =============================================================================

# TRAIN ONE EPOCH

# =============================================================================



def train_one_epoch(epoch):



    model.train()



    # Target-only modules should remain frozen.

    model.target_private_encoder.eval()

    model.target_fusion.eval()

    model.target_head.eval()



    train_sampler.set_epoch(

        epoch

    )



    total_sse = 0.0

    total_sae = 0.0

    total_n = 0.0



    horizon_totals = {

        h: {

            "sse": 0.0,

            "sae": 0.0,

            "n": 0.0,

        }

        for h in HORIZONS

    }



    epoch_start = time.time()



    progress = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Epoch {epoch:02d} TRAIN",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_idx, batch in progress:



        shared = batch[

            "shared_history"

        ].to(

            device,

            non_blocking=True

        )



        private = batch[

            "private_history"

        ].to(

            device,

            non_blocking=True

        )



        target = batch[

            "labels"

        ].to(

            device,

            non_blocking=True

        )



        mask = batch[

            "label_mask"

        ].to(

            device,

            non_blocking=True

        )



        optimizer.zero_grad(

            set_to_none=True

        )



        with torch.amp.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=amp_enabled,

        ):



            output = model.forward_source(

                shared,

                private

            )



            prediction = output[

                "prediction"

            ]



            loss = (

                step24.masked_mse_loss(

                    prediction,

                    target,

                    mask

                )

            )



        if not torch.isfinite(loss):

            raise RuntimeError(

                f"Non-finite training loss "

                f"at epoch {epoch}, "

                f"batch {batch_idx}."

            )



        scaler.scale(

            loss

        ).backward()



        scaler.unscale_(

            optimizer

        )



        torch.nn.utils.clip_grad_norm_(

            trainable_parameters,

            GRADIENT_CLIP_NORM

        )



        scaler.step(

            optimizer

        )



        scaler.update()



        with torch.no_grad():



            sse, sae, n = (

                masked_error_statistics(

                    prediction,

                    target,

                    mask

                )

            )



            total_sse += sse

            total_sae += sae

            total_n += n



            hs = horizon_statistics(

                prediction,

                target,

                mask

            )



            for item in hs:

                h = item["horizon"]



                horizon_totals[h][

                    "sse"

                ] += item["sse"]



                horizon_totals[h][

                    "sae"

                ] += item["sae"]



                horizon_totals[h][

                    "n"

                ] += item["n"]



        elapsed = time.time() - epoch_start
        processed = min(
            batch_idx * BATCH_SIZE,
            len(train_sampler)
        )
        rate = processed / elapsed if elapsed > 0 else 0.0
        running_rmse = math.sqrt(
            total_sse / max(total_n, 1.0)
        )
        progress.set_postfix(
            samples=f"{processed:,}",
            rmse_z=f"{running_rmse:.5f}",
            rate=f"{rate:,.1f}/s",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            refresh=False,
        )



    rmse = math.sqrt(

        total_sse

        / total_n

    )



    mae = (

        total_sae

        / total_n

    )



    horizon_metrics = {}



    for h in HORIZONS:



        x = horizon_totals[h]



        horizon_metrics[h] = {

            "rmse": math.sqrt(

                x["sse"]

                / x["n"]

            ),

            "mae": (

                x["sae"]

                / x["n"]

            ),

            "n": int(

                x["n"]

            ),

        }



    return {

        "rmse": rmse,

        "mae": mae,

        "n": int(total_n),

        "horizons":

            horizon_metrics,

        "seconds":

            time.time()

            - epoch_start,

    }





# =============================================================================

# VALIDATION

# =============================================================================



@torch.no_grad()

def validate():



    model.eval()



    total_sse = 0.0

    total_sae = 0.0

    total_n = 0.0



    horizon_totals = {

        h: {

            "sse": 0.0,

            "sae": 0.0,

            "n": 0.0,

        }

        for h in HORIZONS

    }



    start = time.time()



    progress = tqdm(
        enumerate(validation_loader, start=1),
        total=len(validation_loader),
        desc="VALIDATION",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_idx, batch in progress:



        shared = batch[

            "shared_history"

        ].to(

            device,

            non_blocking=True

        )



        private = batch[

            "private_history"

        ].to(

            device,

            non_blocking=True

        )



        target = batch[

            "labels"

        ].to(

            device,

            non_blocking=True

        )



        mask = batch[

            "label_mask"

        ].to(

            device,

            non_blocking=True

        )



        with torch.amp.autocast(

            device_type=device.type,

            dtype=torch.float16,

            enabled=amp_enabled,

        ):



            prediction = (

                model.forward_source(

                    shared,

                    private

                )["prediction"]

            )



        if not torch.isfinite(

            prediction

        ).all():

            raise RuntimeError(

                "Non-finite validation "

                "prediction."

            )



        sse, sae, n = (

            masked_error_statistics(

                prediction,

                target,

                mask

            )

        )



        total_sse += sse

        total_sae += sae

        total_n += n



        hs = horizon_statistics(

            prediction,

            target,

            mask

        )



        for item in hs:



            h = item["horizon"]



            horizon_totals[h][

                "sse"

            ] += item["sse"]



            horizon_totals[h][

                "sae"

            ] += item["sae"]



            horizon_totals[h][

                "n"

            ] += item["n"]



        # Update the validation progress display every batch.
        elapsed = time.time() - start
        processed = min(
            batch_idx * BATCH_SIZE,
            len(validation_sampler)
        )
        rate = processed / elapsed if elapsed > 0 else 0.0
        running_rmse = math.sqrt(
            total_sse / max(total_n, 1.0)
        )
        progress.set_postfix(
            samples=f"{processed:,}",
            rmse_z=f"{running_rmse:.5f}",
            rate=f"{rate:,.1f}/s",
            refresh=False,
        )

    rmse = math.sqrt(

        total_sse

        / total_n

    )



    mae = (

        total_sae

        / total_n

    )



    horizon_metrics = {}



    for h in HORIZONS:



        x = horizon_totals[h]



        horizon_metrics[h] = {

            "rmse": math.sqrt(

                x["sse"]

                / x["n"]

            ),

            "mae": (

                x["sae"]

                / x["n"]

            ),

            "n": int(

                x["n"]

            ),

        }



    return {

        "rmse": rmse,

        "mae": mae,

        "n": int(total_n),

        "horizons":

            horizon_metrics,

        "seconds":

            time.time()

            - start,

    }





# =============================================================================

# CHECKPOINTS

# =============================================================================



BEST_MODEL_PATH = (

    OUTPUT_DIR

    / "bdg2_source_pretraining_best.pt"

)



LAST_MODEL_PATH = (

    OUTPUT_DIR

    / "bdg2_source_pretraining_last.pt"

)



SHARED_ENCODER_PATH = (

    OUTPUT_DIR

    / "bdg2_pretrained_shared_encoder.pt"

)



HISTORY_PATH = (

    AUDIT_DIR

    / "bdg2_source_pretraining_history.csv"

)



MANIFEST_PATH = (

    AUDIT_DIR

    / "bdg2_source_pretraining_manifest.json"

)





def checkpoint_payload(

    epoch,

    best_validation_rmse,

    epochs_without_improvement

):

    return {

        "step": 25,

        "epoch": int(epoch),

        "seed": SEED,



        "model_state_dict":

            model.state_dict(),



        "optimizer_state_dict":

            optimizer.state_dict(),



        "scheduler_state_dict":

            scheduler.state_dict(),



        "scaler_state_dict":

            scaler.state_dict(),



        "best_validation_rmse":

            float(

                best_validation_rmse

            ),



        "epochs_without_improvement":

            int(

                epochs_without_improvement

            ),



        "history_length":

            HISTORY_LENGTH,



        "horizons":

            HORIZONS,



        "source_normalization_file":

            str(

                SOURCE_NORMALIZATION

                .relative_to(ROOT)

            ),



        "train_index_file":

            str(

                TRAIN_INDEX

                .relative_to(ROOT)

            ),



        "validation_index_file":

            str(

                VALIDATION_INDEX

                .relative_to(ROOT)

            ),



        "target_data_accessed":

            False,



        "source_test_accessed":

            False,

    }





# =============================================================================

# TRAINING LOOP

# =============================================================================



banner("SOURCE PRETRAINING")



best_validation_rmse = float(

    "inf"

)



epochs_without_improvement = 0



history_records = []



training_start = time.time()





for epoch in range(

    1,

    MAX_EPOCHS + 1

):



    banner(

        f"EPOCH {epoch}/{MAX_EPOCHS}"

    )



    train_metrics = (

        train_one_epoch(

            epoch

        )

    )



    validation_metrics = (

        validate()

    )



    current_lr = float(

        optimizer.param_groups[

            0

        ]["lr"]

    )



    scheduler.step(

        validation_metrics[

            "rmse"

        ]

    )



    print()

    print("TRAIN")

    print(

        f"  RMSE(z): "

        f"{train_metrics['rmse']:.6f}"

    )

    print(

        f"  MAE(z):  "

        f"{train_metrics['mae']:.6f}"

    )



    print()

    print("VALIDATION")

    print(

        f"  RMSE(z): "

        f"{validation_metrics['rmse']:.6f}"

    )

    print(

        f"  MAE(z):  "

        f"{validation_metrics['mae']:.6f}"

    )



    print()

    print("VALIDATION BY HORIZON")



    for h in HORIZONS:



        hm = (

            validation_metrics[

                "horizons"

            ][h]

        )



        print(

            f"  h={h:>2}h "

            f"RMSE(z)="

            f"{hm['rmse']:.6f} "

            f"MAE(z)="

            f"{hm['mae']:.6f} "

            f"n={hm['n']:,}"

        )



    record = {

        "seed": SEED,

        "epoch": epoch,

        "learning_rate":

            current_lr,



        "train_rmse_z":

            train_metrics[

                "rmse"

            ],



        "train_mae_z":

            train_metrics[

                "mae"

            ],



        "validation_rmse_z":

            validation_metrics[

                "rmse"

            ],



        "validation_mae_z":

            validation_metrics[

                "mae"

            ],



        "train_seconds":

            train_metrics[

                "seconds"

            ],



        "validation_seconds":

            validation_metrics[

                "seconds"

            ],

    }



    for h in HORIZONS:



        record[

            f"validation_rmse_z_h{h}"

        ] = (

            validation_metrics[

                "horizons"

            ][h]["rmse"]

        )



        record[

            f"validation_mae_z_h{h}"

        ] = (

            validation_metrics[

                "horizons"

            ][h]["mae"]

        )



        record[

            f"validation_n_h{h}"

        ] = (

            validation_metrics[

                "horizons"

            ][h]["n"]

        )



    history_records.append(

        record

    )



    pd.DataFrame(

        history_records

    ).to_csv(

        HISTORY_PATH,

        index=False

    )



    improved = (

        validation_metrics[

            "rmse"

        ]

        < best_validation_rmse

    )



    if improved:



        best_validation_rmse = (

            validation_metrics[

                "rmse"

            ]

        )



        epochs_without_improvement = 0



        atomic_torch_save(

            checkpoint_payload(

                epoch,

                best_validation_rmse,

                epochs_without_improvement

            ),

            BEST_MODEL_PATH

        )



        print()

        print(

            "BEST VALIDATION MODEL SAVED"

        )



    else:



        epochs_without_improvement += 1



    atomic_torch_save(

        checkpoint_payload(

            epoch,

            best_validation_rmse,

            epochs_without_improvement

        ),

        LAST_MODEL_PATH

    )



    print()

    print(

        f"Best validation RMSE(z): "

        f"{best_validation_rmse:.6f}"

    )



    print(

        "Epochs without improvement: "

        f"{epochs_without_improvement}"

    )



    if (

        epochs_without_improvement

        >= EARLY_STOPPING_PATIENCE

    ):

        print()

        print(

            "Early stopping triggered."

        )

        break



    gc.collect()



    if device.type == "cuda":

        torch.cuda.empty_cache()





# =============================================================================

# LOAD BEST MODEL

# =============================================================================



banner("RESTORING BEST SOURCE MODEL")



if not BEST_MODEL_PATH.exists():

    raise RuntimeError(

        "Best checkpoint was not created."

    )



best_checkpoint = torch.load(

    BEST_MODEL_PATH,

    map_location=device,

    weights_only=False

)



model.load_state_dict(

    best_checkpoint[

        "model_state_dict"

    ]

)



best_epoch = int(

    best_checkpoint[

        "epoch"

    ]

)



best_validation_rmse = float(

    best_checkpoint[

        "best_validation_rmse"

    ]

)



print(

    f"Best epoch: "

    f"{best_epoch}"

)



print(

    f"Best validation RMSE(z): "

    f"{best_validation_rmse:.6f}"

)





# =============================================================================

# EXPORT TRANSFERABLE SHARED ENCODER

# =============================================================================



banner("EXPORTING TRANSFERABLE SHARED ENCODER")



shared_encoder_payload = {

    "step": 25,

    "seed": SEED,



    "component":

        "shared_encoder",



    "source_dataset":

        "BDG2",



    "source_task":

        "multi-horizon electricity forecasting",



    "history_length":

        HISTORY_LENGTH,



    "horizons":

        HORIZONS,



    "shared_variables":

        SHARED_VARIABLES,



    "shared_input_dimension":

        15,



    "best_source_epoch":

        best_epoch,



    "best_source_validation_rmse_z":

        best_validation_rmse,



    "state_dict":

        model

        .shared_encoder

        .state_dict(),



    "source_test_accessed":

        False,



    "target_domain_accessed":

        False,



    "exported_at_utc":

        datetime.now(

            timezone.utc

        ).isoformat(),

}



atomic_torch_save(

    shared_encoder_payload,

    SHARED_ENCODER_PATH

)



print(

    f"Saved:\n  "

    f"{SHARED_ENCODER_PATH}"

)





# =============================================================================

# FINAL MANIFEST

# =============================================================================



training_seconds = (

    time.time()

    - training_start

)



history_df = pd.DataFrame(

    history_records

)



manifest = {

    "step": 25,



    "stage":

        "BDG2 source-domain pretraining",



    "created_at_utc":

        datetime.now(

            timezone.utc

        ).isoformat(),



    "seed":

        SEED,



    "device":

        str(device),



    "gpu":

        (

            torch.cuda.get_device_name(0)

            if device.type == "cuda"

            else None

        ),



    "torch_version":

        torch.__version__,



    "history_length_hours":

        HISTORY_LENGTH,



    "forecast_horizons_hours":

        HORIZONS,



    "frozen_source_buildings":

        len(frozen_buildings),



    "source_train_origins":

        len(train_dataset),



    "source_validation_origins":

        len(validation_dataset),



    "training_samples_per_epoch":

        len(train_sampler),



    "validation_samples":

        len(validation_sampler),



    "batch_size":

        BATCH_SIZE,



    "max_epochs":

        MAX_EPOCHS,



    "epochs_completed":

        len(history_records),



    "best_epoch":

        best_epoch,



    "best_validation_rmse_z":

        best_validation_rmse,



    "learning_rate":

        LEARNING_RATE,



    "weight_decay":

        WEIGHT_DECAY,



    "gradient_clip_norm":

        GRADIENT_CLIP_NORM,



    "amp_enabled":

        amp_enabled,



    "normalization":

        "frozen BDG2 training-only statistics",



    "selection_metric":

        "masked validation RMSE in normalized energy space",



    "training_sampling":

        {

            "method":

                "deterministic random subset without replacement per epoch",



            "sample_size":

                len(train_sampler),



            "seed":

                SEED,



            "reason":

                "Adjacent hourly 24-hour windows are highly overlapping; full 15M-origin epochs are computationally redundant."

        },



    "validation_sampling":

        {

            "method":

                "single deterministic fixed subset",



            "sample_size":

                len(validation_sampler),



            "seed":

                VALIDATION_SUBSET_SEED,

        },



    "leakage_assertions":

        {

            "bdg2_test_accessed":

                False,



            "target_adaptation_accessed":

                False,



            "target_validation_accessed":

                False,



            "target_test_accessed":

                False,



            "target_normalization_accessed":

                False,



            "future_observations_used_as_inputs":

                False,



            "imputed_labels_used_as_ground_truth":

                False,

        },



    "artifacts":

        {

            "best_source_checkpoint":

                str(

                    BEST_MODEL_PATH

                    .relative_to(ROOT)

                ),



            "last_source_checkpoint":

                str(

                    LAST_MODEL_PATH

                    .relative_to(ROOT)

                ),



            "shared_encoder":

                str(

                    SHARED_ENCODER_PATH

                    .relative_to(ROOT)

                ),



            "history":

                str(

                    HISTORY_PATH

                    .relative_to(ROOT)

                ),

        },



    "input_hashes":

        {

            "forecasting_protocol_sha256":

                sha256_file(

                    FORECAST_PROTOCOL

                ),



            "source_normalization_sha256":

                sha256_file(

                    SOURCE_NORMALIZATION

                ),



            "source_cohort_sha256":

                sha256_file(

                    SOURCE_COHORT

                ),



            "step24_model_sha256":

                sha256_file(

                    STEP24_PATH

                ),

        },



    "training_seconds":

        training_seconds,

}





with open(

    MANIFEST_PATH,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        manifest,

        f,

        indent=2

    )





# =============================================================================

# COMPLETE

# =============================================================================



banner("STEP 25 SOURCE PRETRAINING COMPLETE")

print(f"Seed: {SEED}")



print(

    f"Epochs completed: "

    f"{len(history_records)}"

)



print(

    f"Best epoch: "

    f"{best_epoch}"

)



print(

    f"Best validation RMSE(z): "

    f"{best_validation_rmse:.6f}"

)



print(

    f"Total training time: "

    f"{training_seconds / 60:.2f} minutes"

)



print()

print("Generated:")

print(

    f"  {BEST_MODEL_PATH}"

)

print(

    f"  {LAST_MODEL_PATH}"

)

print(

    f"  {SHARED_ENCODER_PATH}"

)

print(

    f"  {HISTORY_PATH}"

)

print(

    f"  {MANIFEST_PATH}"

)



print()

print("LEAKAGE STATUS")

print()

print(

    "  BDG2 test accessed:              NO"

)

print(

    "  openSenseMap adaptation accessed:NO"

)

print(

    "  openSenseMap validation accessed:NO"

)

print(

    "  openSenseMap test accessed:      NO"

)

print(

    "  Target normalization accessed:   NO"

)



print()

print("TRANSFER STATUS")

print()

print(

    "  Shared environmental encoder: "

    "PRETRAINED"

)

print(

    "  Source-private energy encoder: "

    "PRETRAINED"

)

print(

    "  Source energy fusion/head: "

    "PRETRAINED"

)

print(

    "  Target-private PM2.5 components: "

    "UNTOUCHED"

)



print()

print(

    "The exported shared encoder is the "

    "transferable Step-25 artifact."

)



print()

print(

    "No source-test forecasting result "

    "has been inspected."

)



print(

    "No target-domain forecasting result "

    "has been inspected."

)
