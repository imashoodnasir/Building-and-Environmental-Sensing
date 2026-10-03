from pathlib import Path
import zipfile
import json
import shutil

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "bdg2"
    / "building-data-genome-project-2-v1.0.zip"
)

META_DIR = ROOT / "data" / "metadata"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "final"
    / "bdg2"
    / "buildings"
)

META_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUILDING_AUDIT_FILE = (
    META_DIR
    / "bdg2_source_building_audit.csv"
)

SITE_AUDIT_FILE = (
    META_DIR
    / "bdg2_source_site_weather_audit.csv"
)

COHORT_FILE = (
    META_DIR
    / "bdg2_final_source_cohort.csv"
)

MANIFEST_FILE = (
    META_DIR
    / "bdg2_source_manifest.json"
)

MAX_FFILL_HOURS = 3

MIN_ENERGY_OBSERVED_PERCENT = 70.0

VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
    "energy",
]

# Frozen physical plausibility ranges.
PHYSICAL_RANGES = {
    "temperature": (-60.0, 60.0),       # deg C
    "relative_humidity": (0.0, 100.0),  # %
    "pressure": (800.0, 1100.0),        # hPa
    "energy": (0.0, np.inf),
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def derive_relative_humidity(temp_c, dew_c):
    """
    Magnus approximation.

    RH = 100 * exp(
        a*Td/(b+Td) - a*T/(b+T)
    )

    T  = air temperature in deg C
    Td = dew-point temperature in deg C
    """

    a = 17.625
    b = 243.04

    temp_c = pd.to_numeric(
        temp_c,
        errors="coerce"
    )

    dew_c = pd.to_numeric(
        dew_c,
        errors="coerce"
    )

    rh = 100.0 * np.exp(
        (a * dew_c / (b + dew_c))
        -
        (a * temp_c / (b + temp_c))
    )

    return rh.clip(
        lower=0.0,
        upper=100.0
    )


def physical_filter(series, variable):

    s = pd.to_numeric(
        series,
        errors="coerce"
    )

    low, high = PHYSICAL_RANGES[variable]

    if np.isfinite(high):

        valid = (
            s.ge(low)
            &
            s.le(high)
        )

    else:

        valid = s.ge(low)

    invalid_count = int(
        (
            s.notna()
            &
            ~valid
        ).sum()
    )

    filtered = s.where(
        s.isna() | valid
    )

    return filtered, invalid_count


def causal_fill(series, limit):

    original = series.copy()

    observed = original.notna()

    filled = original.ffill(
        limit=limit
    )

    imputed = (
        original.isna()
        &
        filled.notna()
    )

    missing = filled.isna()

    return (
        filled,
        observed,
        imputed,
        missing
    )


def safe_filename(text):

    text = str(text)

    for char in [
        "/",
        "\\",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
    ]:
        text = text.replace(
            char,
            "_"
        )

    return text


# =============================================================================
# START
# =============================================================================

print("=" * 100)
print("BDG2 MEMORY-EFFICIENT SOURCE DATASET CONSTRUCTION")
print("=" * 100)

if not ARCHIVE.exists():

    raise FileNotFoundError(
        f"\nBDG2 archive not found:\n{ARCHIVE}"
    )

print("\nArchive:")
print(ARCHIVE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# =============================================================================
# LOCATE ARCHIVE FILES
# =============================================================================

with zipfile.ZipFile(ARCHIVE) as z:

    names = z.namelist()

    weather_candidates = [
        x
        for x in names
        if x.endswith(
            "/data/weather/weather.csv"
        )
    ]

    metadata_candidates = [
        x
        for x in names
        if x.endswith(
            "/data/metadata/metadata.csv"
        )
    ]

    electricity_candidates = [
        x
        for x in names
        if x.endswith(
            "/data/meters/cleaned/electricity_cleaned.csv"
        )
    ]

    if len(weather_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely locate weather.csv."
        )

    if len(metadata_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely locate metadata.csv."
        )

    if len(electricity_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely locate electricity_cleaned.csv."
        )

    weather_name = weather_candidates[0]
    metadata_name = metadata_candidates[0]
    electricity_name = electricity_candidates[0]

    print("\nArchive members:")
    print("Weather:    ", weather_name)
    print("Metadata:   ", metadata_name)
    print("Electricity:", electricity_name)

    # These source tables themselves are manageable.
    weather = pd.read_csv(
        z.open(weather_name)
    )

    metadata = pd.read_csv(
        z.open(metadata_name)
    )

    electricity = pd.read_csv(
        z.open(electricity_name)
    )


# =============================================================================
# RAW VALIDATION
# =============================================================================

print("\n" + "=" * 100)
print("RAW INPUT VALIDATION")
print("=" * 100)

print("\nWeather shape:")
print(weather.shape)

print("\nMetadata shape:")
print(metadata.shape)

print("\nElectricity shape:")
print(electricity.shape)

required_weather = {
    "timestamp",
    "site_id",
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
}

missing_weather = (
    required_weather
    - set(weather.columns)
)

if missing_weather:

    raise RuntimeError(
        f"Missing weather variables: "
        f"{sorted(missing_weather)}"
    )

required_metadata = {
    "building_id",
    "site_id",
    "sqm",
    "primaryspaceusage",
}

missing_metadata = (
    required_metadata
    - set(metadata.columns)
)

if missing_metadata:

    raise RuntimeError(
        f"Missing metadata variables: "
        f"{sorted(missing_metadata)}"
    )


# =============================================================================
# TIMESTAMP HANDLING
# =============================================================================

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    errors="coerce"
)

electricity["timestamp"] = pd.to_datetime(
    electricity["timestamp"],
    errors="coerce"
)

weather = weather[
    weather["timestamp"].notna()
].copy()

electricity = electricity[
    electricity["timestamp"].notna()
].copy()

START = max(
    weather["timestamp"].min(),
    electricity["timestamp"].min()
)

END = min(
    weather["timestamp"].max(),
    electricity["timestamp"].max()
)

START = START.floor("h")
END = END.floor("h")

FULL_INDEX = pd.date_range(
    START,
    END,
    freq="h"
)

EXPECTED_HOURS = len(
    FULL_INDEX
)

print("\nCommon period:")
print("Start:", START)
print("End:  ", END)
print("Expected hourly bins:", EXPECTED_HOURS)


# =============================================================================
# HARMONIZE WEATHER
# =============================================================================

print("\n" + "=" * 100)
print("HARMONIZING SITE-LEVEL WEATHER")
print("=" * 100)

weather["temperature"] = pd.to_numeric(
    weather["airTemperature"],
    errors="coerce"
)

weather["dew_temperature"] = pd.to_numeric(
    weather["dewTemperature"],
    errors="coerce"
)

weather["pressure"] = pd.to_numeric(
    weather["seaLvlPressure"],
    errors="coerce"
)

weather["relative_humidity"] = (
    derive_relative_humidity(
        weather["temperature"],
        weather["dew_temperature"]
    )
)

weather_invalid = {}

for variable in [
    "temperature",
    "relative_humidity",
    "pressure",
]:

    (
        weather[variable],
        invalid_count
    ) = physical_filter(
        weather[variable],
        variable
    )

    weather_invalid[variable] = (
        invalid_count
    )

    print(
        f"{variable:20s}"
        f" physical-invalid="
        f"{invalid_count:,}"
    )


# =============================================================================
# BUILD SITE-LEVEL WEATHER CACHE
# =============================================================================

print("\n" + "=" * 100)
print("BUILDING SITE WEATHER CACHE")
print("=" * 100)

site_weather = {}

site_audit_rows = []

sites = sorted(
    weather[
        "site_id"
    ]
    .dropna()
    .astype(str)
    .unique()
)

print("\nSites:", len(sites))

weather["site_id"] = (
    weather["site_id"]
    .astype(str)
)

for site_number, site_id in enumerate(
    sites,
    start=1
):

    print(
        f"[{site_number:02d}/{len(sites):02d}] "
        f"{site_id}"
    )

    temp = weather[
        weather["site_id"] == site_id
    ][
        [
            "timestamp",
            "temperature",
            "relative_humidity",
            "pressure",
        ]
    ].copy()

    # Safety against accidental duplicate site-hour rows.
    temp = (
        temp.groupby(
            "timestamp",
            as_index=True
        )
        [
            [
                "temperature",
                "relative_humidity",
                "pressure",
            ]
        ]
        .mean()
    )

    temp = temp.reindex(
        FULL_INDEX
    )

    result = pd.DataFrame(
        index=FULL_INDEX
    )

    result.index.name = "timestamp"

    for variable in [
        "temperature",
        "relative_humidity",
        "pressure",
    ]:

        (
            filled,
            observed,
            imputed,
            missing
        ) = causal_fill(
            temp[variable],
            MAX_FFILL_HOURS
        )

        result[variable] = (
            filled.astype("float32")
        )

        result[
            f"{variable}_observed"
        ] = observed.astype("int8")

        result[
            f"{variable}_imputed"
        ] = imputed.astype("int8")

        result[
            f"{variable}_missing"
        ] = missing.astype("int8")

        site_audit_rows.append({
            "site_id":
                site_id,

            "canonical_variable":
                variable,

            "total_hours":
                EXPECTED_HOURS,

            "observed_hours":
                int(observed.sum()),

            "observed_percent":
                float(
                    observed.mean()
                    * 100
                ),

            "causally_imputed_hours":
                int(imputed.sum()),

            "causally_imputed_percent":
                float(
                    imputed.mean()
                    * 100
                ),

            "remaining_missing_hours":
                int(missing.sum()),

            "remaining_missing_percent":
                float(
                    missing.mean()
                    * 100
                ),

            "forward_fill_limit_hours":
                MAX_FFILL_HOURS,

            "future_values_used":
                False,
        })

    site_weather[site_id] = result


# =============================================================================
# IDENTIFY ELECTRICITY BUILDINGS
# =============================================================================

print("\n" + "=" * 100)
print("IDENTIFYING BUILDING-LEVEL ENERGY STREAMS")
print("=" * 100)

electricity_columns = [
    str(c)
    for c in electricity.columns
    if c != "timestamp"
]

metadata["building_id"] = (
    metadata["building_id"]
    .astype(str)
)

metadata["site_id"] = (
    metadata["site_id"]
    .astype(str)
)

building_meta = metadata[
    metadata["building_id"].isin(
        electricity_columns
    )
].copy()

if len(building_meta) != len(
    electricity_columns
):

    missing_ids = (
        set(electricity_columns)
        -
        set(
            building_meta[
                "building_id"
            ]
        )
    )

    raise RuntimeError(
        "Some electricity streams do not "
        f"map to metadata: {missing_ids}"
    )

building_meta = (
    building_meta
    .sort_values(
        [
            "site_id",
            "building_id",
        ]
    )
    .reset_index(drop=True)
)

print(
    "\nMapped buildings:",
    len(building_meta)
)

print(
    "Represented sites:",
    building_meta[
        "site_id"
    ].nunique()
)


# =============================================================================
# INDEX ELECTRICITY ONCE
# =============================================================================

electricity = (
    electricity
    .set_index("timestamp")
    .reindex(FULL_INDEX)
)

# Reduce memory footprint.
for column in electricity.columns:

    electricity[column] = pd.to_numeric(
        electricity[column],
        errors="coerce",
        downcast="float"
    )


# =============================================================================
# BUILD EACH BUILDING INDEPENDENTLY
# =============================================================================

print("\n" + "=" * 100)
print("WRITING BUILDING-LEVEL PARQUET FILES")
print("=" * 100)

building_audit_rows = []

for index, row in building_meta.iterrows():

    number = index + 1

    building_id = str(
        row["building_id"]
    )

    site_id = str(
        row["site_id"]
    )

    if site_id not in site_weather:

        raise RuntimeError(
            f"No site weather cache "
            f"for site {site_id}"
        )

    print(
        f"[{number:04d}/"
        f"{len(building_meta):04d}] "
        f"{building_id} | "
        f"{site_id}"
    )

    # -------------------------------------------------------------------------
    # ENERGY
    # -------------------------------------------------------------------------

    raw_energy = pd.to_numeric(
        electricity[building_id],
        errors="coerce"
    )

    (
        clean_energy,
        energy_invalid
    ) = physical_filter(
        raw_energy,
        "energy"
    )

    (
        energy,
        energy_observed,
        energy_imputed,
        energy_missing
    ) = causal_fill(
        clean_energy,
        MAX_FFILL_HOURS
    )

    observed_percent = float(
        energy_observed.mean()
        * 100
    )

    # -------------------------------------------------------------------------
    # BUILD OUTPUT
    # -------------------------------------------------------------------------

    data = site_weather[
        site_id
    ].copy()

    data["energy"] = (
        energy.astype("float32")
    )

    data["energy_observed"] = (
        energy_observed.astype("int8")
    )

    data["energy_imputed"] = (
        energy_imputed.astype("int8")
    )

    data["energy_missing"] = (
        energy_missing.astype("int8")
    )

    data = data.reset_index()

    data.insert(
        0,
        "building_id",
        building_id
    )

    data.insert(
        1,
        "site_id",
        site_id
    )

    # -------------------------------------------------------------------------
    # DETERMINISTIC TIME FEATURES
    # -------------------------------------------------------------------------

    data["hour_of_day"] = (
        data["timestamp"]
        .dt.hour
        .astype("int8")
    )

    data["day_of_week"] = (
        data["timestamp"]
        .dt.dayofweek
        .astype("int8")
    )

    data["day_of_year"] = (
        data["timestamp"]
        .dt.dayofyear
        .astype("int16")
    )

    data["hour_sin"] = (
        np.sin(
            2
            * np.pi
            * data["hour_of_day"]
            / 24.0
        )
        .astype("float32")
    )

    data["hour_cos"] = (
        np.cos(
            2
            * np.pi
            * data["hour_of_day"]
            / 24.0
        )
        .astype("float32")
    )

    data["dow_sin"] = (
        np.sin(
            2
            * np.pi
            * data["day_of_week"]
            / 7.0
        )
        .astype("float32")
    )

    data["dow_cos"] = (
        np.cos(
            2
            * np.pi
            * data["day_of_week"]
            / 7.0
        )
        .astype("float32")
    )

    data["doy_sin"] = (
        np.sin(
            2
            * np.pi
            * data["day_of_year"]
            / 365.25
        )
        .astype("float32")
    )

    data["doy_cos"] = (
        np.cos(
            2
            * np.pi
            * data["day_of_year"]
            / 365.25
        )
        .astype("float32")
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    data["sqm"] = np.float32(
        row["sqm"]
    )

    primary_usage = (
        str(row["primaryspaceusage"])
        if pd.notna(
            row["primaryspaceusage"]
        )
        else "unknown"
    )

    data["primaryspaceusage"] = (
        primary_usage
    )

    # -------------------------------------------------------------------------
    # WRITE ONLY THIS BUILDING
    # -------------------------------------------------------------------------

    filename = (
        safe_filename(building_id)
        + ".parquet"
    )

    output_file = (
        OUTPUT_DIR
        / filename
    )

    data.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    file_size_mb = (
        output_file.stat().st_size
        / 1024
        / 1024
    )

    # -------------------------------------------------------------------------
    # AUDIT
    # -------------------------------------------------------------------------

    building_audit_rows.append({

        "building_id":
            building_id,

        "site_id":
            site_id,

        "sqm":
            float(row["sqm"]),

        "primaryspaceusage":
            primary_usage,

        "total_hours":
            EXPECTED_HOURS,

        "energy_raw_nonmissing_hours":
            int(
                raw_energy.notna().sum()
            ),

        "energy_physical_invalid_records":
            int(energy_invalid),

        "energy_observed_hours":
            int(
                energy_observed.sum()
            ),

        "energy_observed_percent":
            observed_percent,

        "energy_causally_imputed_hours":
            int(
                energy_imputed.sum()
            ),

        "energy_causally_imputed_percent":
            float(
                energy_imputed.mean()
                * 100
            ),

        "energy_remaining_missing_hours":
            int(
                energy_missing.sum()
            ),

        "energy_remaining_missing_percent":
            float(
                energy_missing.mean()
                * 100
            ),

        "passes_70_percent_energy":
            bool(
                observed_percent
                >=
                MIN_ENERGY_OBSERVED_PERCENT
            ),

        "parquet_file":
            str(
                output_file.relative_to(ROOT)
            ),

        "parquet_size_mb":
            file_size_mb,
    })


# =============================================================================
# SAVE AUDITS
# =============================================================================

building_audit = pd.DataFrame(
    building_audit_rows
)

site_audit = pd.DataFrame(
    site_audit_rows
)

building_audit.to_csv(
    BUILDING_AUDIT_FILE,
    index=False
)

site_audit.to_csv(
    SITE_AUDIT_FILE,
    index=False
)


# =============================================================================
# FREEZE SOURCE COHORT
# =============================================================================

source_cohort = building_audit[
    building_audit[
        "passes_70_percent_energy"
    ]
].copy()

source_cohort = (
    source_cohort
    .sort_values(
        [
            "site_id",
            "building_id"
        ]
    )
    .reset_index(drop=True)
)

source_cohort.insert(
    0,
    "source_order",
    np.arange(
        1,
        len(source_cohort) + 1
    )
)

source_cohort.to_csv(
    COHORT_FILE,
    index=False
)


# =============================================================================
# VERIFY PARQUET OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("VERIFYING PARQUET OUTPUT")
print("=" * 100)

parquet_files = list(
    OUTPUT_DIR.glob(
        "*.parquet"
    )
)

if len(parquet_files) != len(
    building_meta
):

    raise RuntimeError(
        "Parquet count mismatch. "
        f"Expected {len(building_meta)}, "
        f"found {len(parquet_files)}."
    )

total_parquet_bytes = sum(
    p.stat().st_size
    for p in parquet_files
)

total_parquet_gb = (
    total_parquet_bytes
    / 1024**3
)

print(
    "\nParquet files:",
    len(parquet_files)
)

print(
    "Total size:",
    round(
        total_parquet_gb,
        3
    ),
    "GB"
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {

    "dataset":
        "Building Data Genome Project 2",

    "source_domain_unit":
        "building",

    "all_electricity_buildings":
        int(len(building_meta)),

    "eligible_source_buildings":
        int(len(source_cohort)),

    "sites":
        int(
            building_meta[
                "site_id"
            ].nunique()
        ),

    "period": {
        "start":
            str(START),

        "end":
            str(END),

        "hours_per_building":
            EXPECTED_HOURS,
    },

    "shared_variables": {

        "temperature": {
            "source":
                "airTemperature",
            "unit":
                "degC",
        },

        "relative_humidity": {
            "source":
                (
                    "derived from airTemperature "
                    "and dewTemperature"
                ),
            "formula":
                "Magnus approximation",
            "unit":
                "percent",
        },

        "pressure": {
            "source":
                "seaLvlPressure",
            "unit":
                "hPa",
        },
    },

    "source_private_variable": {
        "name":
            "energy",
        "source":
            "electricity_cleaned.csv",
        "resolution":
            "building-level",
    },

    "weather_resolution":
        "site-level",

    "causal_missing_policy": {
        "method":
            "past-only forward fill",
        "maximum_consecutive_hours":
            MAX_FFILL_HOURS,
        "future_observations_used":
            False,
        "longer_gaps":
            "remain missing",
    },

    "source_eligibility": {
        "energy_minimum_observed_percent":
            MIN_ENERGY_OBSERVED_PERCENT,
        "selection_uses_forecasting_results":
            False,
    },

    "physical_ranges": {
        "temperature_degC":
            [-60.0, 60.0],
        "relative_humidity_percent":
            [0.0, 100.0],
        "pressure_hPa":
            [800.0, 1100.0],
        "energy":
            ">=0",
    },

    "storage": {
        "format":
            "Parquet",
        "compression":
            "snappy",
        "partition_unit":
            "one file per building",
        "long_form_dataset_created":
            False,
    },

    "normalization_applied":
        False,

    "train_validation_test_split_applied":
        False,

    "model_training_applied":
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
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("BDG2 SOURCE DATASET COMPLETE")
print("=" * 100)

print(
    "\nAll electricity buildings:",
    len(building_meta)
)

print(
    "Sites:",
    building_meta[
        "site_id"
    ].nunique()
)

print(
    "Hours per building:",
    EXPECTED_HOURS
)

print(
    "Total theoretical building-hours:",
    (
        len(building_meta)
        * EXPECTED_HOURS
    )
)

print(
    "\nBuildings with >=70% "
    "observed energy:",
    len(source_cohort)
)

print(
    "Buildings with <70% "
    "observed energy:",
    (
        len(building_meta)
        -
        len(source_cohort)
    )
)

print("\nENERGY COMPLETENESS SUMMARY")

print(
    building_audit[
        [
            "energy_observed_percent",
            "energy_causally_imputed_percent",
            "energy_remaining_missing_percent",
        ]
    ].describe()
)

print("\nSITE WEATHER SUMMARY")

weather_summary = (
    site_audit
    .groupby(
        "canonical_variable"
    )
    .agg(
        sites=(
            "site_id",
            "count"
        ),

        observed_percent_min=(
            "observed_percent",
            "min"
        ),

        observed_percent_mean=(
            "observed_percent",
            "mean"
        ),

        imputed_percent_mean=(
            "causally_imputed_percent",
            "mean"
        ),

        remaining_missing_percent_mean=(
            "remaining_missing_percent",
            "mean"
        ),
    )
)

print(
    weather_summary.to_string()
)

print("\nSTORAGE")

print(
    "Parquet files:",
    len(parquet_files)
)

print(
    "Total Parquet size:",
    round(
        total_parquet_gb,
        3
    ),
    "GB"
)

print("\nGenerated:")

print(
    " data/final/bdg2/buildings/"
    "<building_id>.parquet"
)

print(
    " data/metadata/"
    "bdg2_source_building_audit.csv"
)

print(
    " data/metadata/"
    "bdg2_source_site_weather_audit.csv"
)

print(
    " data/metadata/"
    "bdg2_final_source_cohort.csv"
)

print(
    " data/metadata/"
    "bdg2_source_manifest.json"
)

print("""
IMPORTANT:

One Parquet file represents one BDG2 building.

Site-level temperature, derived relative humidity, and pressure
are paired with building-level electricity.

Relative humidity is derived from contemporaneous air and
dew-point temperature using the frozen Magnus formula.

Physical plausibility filtering occurs before missing-value
handling.

Only past observations are used for short-gap filling.
The maximum forward-fill duration is three hours.

Longer gaps remain missing and observation/imputation masks
are retained.

The >=70% source-energy criterion uses observed values BEFORE
causal filling.

No target-domain data was used.

No normalization, train/validation/test split, model fitting,
hyperparameter selection, or forecasting result was used.

The source cohort is therefore frozen independently of
downstream predictive performance.
""")