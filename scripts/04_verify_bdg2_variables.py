from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import json
import io
import re

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "bdg2"
    / "building-data-genome-project-2-v1.0.zip"
)

OUT_DIR = ROOT / "data" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)


def section(text):
    print("\n" + "-" * 88)
    print(text)
    print("-" * 88)


def locate_file(names, suffix):
    """
    Locate a file in the ZIP using candidate filename fragments.
    """
    suffix = suffix.lower()

    for name in names:
        low = name.lower()

        if low.endswith(suffix):
            return name

    return None


def read_csv_from_zip(zf, member, **kwargs):
    with zf.open(member) as f:
        return pd.read_csv(f, **kwargs)


def normalize_name(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def missing_summary(df, columns):
    rows = []

    for col in columns:
        if col not in df.columns:
            continue

        s = df[col]

        rows.append(
            {
                "variable": col,
                "n_rows": len(s),
                "n_nonmissing": int(s.notna().sum()),
                "n_missing": int(s.isna().sum()),
                "missing_percent": float(s.isna().mean() * 100),
                "n_unique": int(s.nunique(dropna=True)),
                "min": float(s.min()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else None,
                "max": float(s.max()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else None,
                "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else None,
                "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else None,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# VALIDATE ARCHIVE
# ============================================================

banner("BDG2 VARIABLE VERIFICATION")

if not ARCHIVE.exists():
    raise FileNotFoundError(
        f"BDG2 archive not found:\n{ARCHIVE}"
    )

print(f"\nArchive:\n{ARCHIVE}")
print(f"\nSize: {ARCHIVE.stat().st_size / (1024 ** 2):.1f} MB")

# ============================================================
# OPEN ARCHIVE
# ============================================================

with zipfile.ZipFile(ARCHIVE, "r") as zf:

    members = zf.namelist()

    # --------------------------------------------------------
    # LOCATE IMPORTANT FILES
    # --------------------------------------------------------

    weather_file = locate_file(
        members,
        "data/weather/weather.csv"
    )

    metadata_file = locate_file(
        members,
        "data/metadata/metadata.csv"
    )

    meter_candidates = {
        "electricity": "data/meters/cleaned/electricity_cleaned.csv",
        "gas": "data/meters/cleaned/gas_cleaned.csv",
        "hotwater": "data/meters/cleaned/hotwater_cleaned.csv",
        "chilledwater": "data/meters/cleaned/chilledwater_cleaned.csv",
        "steam": "data/meters/cleaned/steam_cleaned.csv",
        "water": "data/meters/cleaned/water_cleaned.csv",
        "solar": "data/meters/cleaned/solar_cleaned.csv",
        "irrigation": "data/meters/cleaned/irrigation_cleaned.csv",
    }

    meter_files = {
        meter: locate_file(members, suffix)
        for meter, suffix in meter_candidates.items()
    }

    section("IMPORTANT FILES")

    print(f"Weather:  {weather_file}")
    print(f"Metadata: {metadata_file}")

    for meter, path in meter_files.items():
        print(f"{meter:<14}: {path}")

    if weather_file is None:
        raise RuntimeError("Could not locate BDG2 weather.csv")

    if metadata_file is None:
        raise RuntimeError("Could not locate BDG2 metadata.csv")

    # ========================================================
    # WEATHER DATA
    # ========================================================

    banner("WEATHER DATA")

    weather = read_csv_from_zip(zf, weather_file)

    print("\nShape:")
    print(weather.shape)

    print("\nColumns:")
    for c in weather.columns:
        print(f"  {c}")

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if "timestamp" not in weather.columns:
        raise RuntimeError("weather.csv does not contain timestamp.")

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="coerce"
    )

    section("WEATHER TEMPORAL COVERAGE")

    print("Earliest timestamp:", weather["timestamp"].min())
    print("Latest timestamp:  ", weather["timestamp"].max())
    print("Invalid timestamps:", weather["timestamp"].isna().sum())

    # --------------------------------------------------------
    # SITE IDS
    # --------------------------------------------------------

    if "site_id" in weather.columns:

        section("WEATHER SITES")

        sites = sorted(
            weather["site_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        print("Number of sites:", len(sites))
        print("Sites:")

        for site in sites:
            print(" ", site)

        site_counts = (
            weather.groupby("site_id", dropna=False)
            .size()
            .reset_index(name="n_rows")
        )

        site_counts.to_csv(
            OUT_DIR / "bdg2_weather_site_counts.csv",
            index=False
        )

    # --------------------------------------------------------
    # EXPECTED WEATHER VARIABLES
    # --------------------------------------------------------

    expected_weather = [
        "airTemperature",
        "cloudCoverage",
        "dewTemperature",
        "precipDepth1HR",
        "precipDepth6HR",
        "seaLvlPressure",
        "windDirection",
        "windSpeed",
    ]

    section("WEATHER VARIABLE AVAILABILITY")

    weather_availability = []

    for var in expected_weather:

        exists = var in weather.columns

        if exists:
            missing_pct = weather[var].isna().mean() * 100
            nonmissing = weather[var].notna().sum()
        else:
            missing_pct = None
            nonmissing = 0

        weather_availability.append(
            {
                "variable": var,
                "exists": exists,
                "nonmissing": int(nonmissing),
                "missing_percent": (
                    float(missing_pct)
                    if missing_pct is not None
                    else None
                ),
            }
        )

        print(
            f"{var:<20} "
            f"exists={str(exists):<5} "
            f"nonmissing={nonmissing:<10} "
            f"missing={missing_pct:.2f}%"
            if exists
            else f"{var:<20} exists=False"
        )

    pd.DataFrame(weather_availability).to_csv(
        OUT_DIR / "bdg2_weather_variable_availability.csv",
        index=False
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    section("WEATHER VARIABLE STATISTICS")

    stats = missing_summary(
        weather,
        expected_weather
    )

    if not stats.empty:
        print(stats.to_string(index=False))

    stats.to_csv(
        OUT_DIR / "bdg2_weather_statistics.csv",
        index=False
    )

    # ========================================================
    # RELATIVE HUMIDITY CHECK
    # ========================================================

    banner("RELATIVE HUMIDITY VERIFICATION")

    normalized_columns = {
        normalize_name(c): c
        for c in weather.columns
    }

    possible_rh_names = [
        "relativehumidity",
        "relativehumid",
        "humidity",
        "rh",
    ]

    direct_rh_columns = []

    for candidate in possible_rh_names:
        if candidate in normalized_columns:
            direct_rh_columns.append(
                normalized_columns[candidate]
            )

    if direct_rh_columns:
        print("\nDirect relative-humidity column(s) found:")
        for col in direct_rh_columns:
            print(" ", col)

        direct_rh = True

    else:
        print("\nNo direct relative-humidity column found.")
        direct_rh = False

    # --------------------------------------------------------
    # CAN RH BE DERIVED?
    # --------------------------------------------------------

    has_temp = "airTemperature" in weather.columns
    has_dew = "dewTemperature" in weather.columns

    print("\nInputs required for RH derivation:")
    print("airTemperature:", has_temp)
    print("dewTemperature:", has_dew)

    can_derive_rh = has_temp and has_dew

    if can_derive_rh:

        print(
            "\nRelative humidity CAN be derived from "
            "air temperature and dew-point temperature."
        )

        T = pd.to_numeric(
            weather["airTemperature"],
            errors="coerce"
        )

        Td = pd.to_numeric(
            weather["dewTemperature"],
            errors="coerce"
        )

        # Magnus approximation
        # Temperatures assumed in degrees Celsius.
        a = 17.625
        b = 243.04

        rh = (
            100.0
            * np.exp((a * Td) / (b + Td))
            / np.exp((a * T) / (b + T))
        )

        # Numerical/physical guard
        rh = rh.clip(lower=0, upper=100)

        weather["derivedRelativeHumidity"] = rh

        valid_rh = weather["derivedRelativeHumidity"].dropna()

        print("\nDerived RH statistics:")

        if len(valid_rh) > 0:
            print("Nonmissing:", len(valid_rh))
            print(
                "Missing %:",
                f"{weather['derivedRelativeHumidity'].isna().mean() * 100:.2f}%"
            )
            print("Minimum:", f"{valid_rh.min():.3f}")
            print("Maximum:", f"{valid_rh.max():.3f}")
            print("Mean:   ", f"{valid_rh.mean():.3f}")
            print("Std:    ", f"{valid_rh.std():.3f}")

    else:

        print(
            "\nRelative humidity cannot be derived because "
            "air temperature and/or dew temperature are unavailable."
        )

    # ========================================================
    # PRESSURE VERIFICATION
    # ========================================================

    banner("PRESSURE VERIFICATION")

    has_pressure = "seaLvlPressure" in weather.columns

    print("seaLvlPressure available:", has_pressure)

    if has_pressure:

        pressure = pd.to_numeric(
            weather["seaLvlPressure"],
            errors="coerce"
        )

        valid = pressure.dropna()

        if len(valid) > 0:
            print("Nonmissing:", len(valid))
            print(
                "Missing %:",
                f"{pressure.isna().mean() * 100:.2f}%"
            )
            print("Minimum:", valid.min())
            print("Maximum:", valid.max())
            print("Mean:   ", valid.mean())

    # ========================================================
    # METADATA
    # ========================================================

    banner("BUILDING METADATA")

    metadata = read_csv_from_zip(
        zf,
        metadata_file
    )

    print("\nShape:")
    print(metadata.shape)

    print("\nColumns:")

    for c in metadata.columns:
        print(" ", c)

    metadata.to_csv(
        OUT_DIR / "bdg2_metadata_snapshot.csv",
        index=False
    )

    # ========================================================
    # ENERGY / METER STREAMS
    # ========================================================

    banner("ENERGY AND METER STREAM VERIFICATION")

    meter_summary = []

    for meter, member in meter_files.items():

        if member is None:

            meter_summary.append(
                {
                    "meter": meter,
                    "available": False,
                    "n_rows": 0,
                    "n_series": 0,
                    "start": None,
                    "end": None,
                    "overall_missing_percent": None,
                }
            )

            print(f"\n{meter}: NOT FOUND")
            continue

        print(f"\nChecking: {meter}")

        # Read header first
        with zf.open(member) as f:
            header = pd.read_csv(
                f,
                nrows=0
            ).columns.tolist()

        n_series = max(0, len(header) - 1)

        # Read timestamp separately to reduce memory
        with zf.open(member) as f:
            timestamps = pd.read_csv(
                f,
                usecols=["timestamp"]
            )

        timestamps["timestamp"] = pd.to_datetime(
            timestamps["timestamp"],
            errors="coerce"
        )

        n_rows = len(timestamps)

        # Missingness is computed in chunks so large electricity
        # files do not have to remain entirely in memory.
        total_cells = 0
        missing_cells = 0

        with zf.open(member) as f:

            for chunk in pd.read_csv(
                f,
                chunksize=5000
            ):

                value_cols = [
                    c
                    for c in chunk.columns
                    if c != "timestamp"
                ]

                values = chunk[value_cols]

                total_cells += values.size
                missing_cells += int(
                    values.isna().sum().sum()
                )

        missing_pct = (
            100 * missing_cells / total_cells
            if total_cells > 0
            else None
        )

        start = timestamps["timestamp"].min()
        end = timestamps["timestamp"].max()

        print("  Rows:       ", n_rows)
        print("  Series:     ", n_series)
        print("  Start:      ", start)
        print("  End:        ", end)

        if missing_pct is not None:
            print(
                "  Missing:    ",
                f"{missing_pct:.2f}%"
            )

        meter_summary.append(
            {
                "meter": meter,
                "available": True,
                "n_rows": int(n_rows),
                "n_series": int(n_series),
                "start": (
                    str(start)
                    if pd.notna(start)
                    else None
                ),
                "end": (
                    str(end)
                    if pd.notna(end)
                    else None
                ),
                "overall_missing_percent": (
                    float(missing_pct)
                    if missing_pct is not None
                    else None
                ),
            }
        )

    meter_summary_df = pd.DataFrame(
        meter_summary
    )

    meter_summary_df.to_csv(
        OUT_DIR / "bdg2_meter_summary.csv",
        index=False
    )

    # ========================================================
    # FINAL EXPERIMENTAL VARIABLE ASSESSMENT
    # ========================================================

    banner("FINAL VARIABLE ASSESSMENT")

    energy_available = any(
        x["available"]
        for x in meter_summary
        if x["meter"] in [
            "electricity",
            "gas",
            "hotwater",
            "chilledwater",
            "steam",
        ]
    )

    assessment = {
        "temperature": {
            "available": bool(has_temp),
            "source": "airTemperature"
            if has_temp else None,
            "type": "direct_measurement",
        },

        "relative_humidity": {
            "available_directly": bool(direct_rh),
            "derivable": bool(can_derive_rh),
            "source": (
                "direct"
                if direct_rh
                else (
                    "derived from airTemperature "
                    "and dewTemperature"
                    if can_derive_rh
                    else None
                )
            ),
            "type": (
                "direct_measurement"
                if direct_rh
                else (
                    "derived_variable"
                    if can_derive_rh
                    else "unavailable"
                )
            ),
        },

        "pressure": {
            "available": bool(has_pressure),
            "source": "seaLvlPressure"
            if has_pressure else None,
            "type": "direct_measurement",
        },

        "energy": {
            "available": bool(energy_available),
            "type": "meter_measurement",
        },
    }

    print("\nTemperature:")
    print(json.dumps(
        assessment["temperature"],
        indent=2
    ))

    print("\nRelative humidity:")
    print(json.dumps(
        assessment["relative_humidity"],
        indent=2
    ))

    print("\nPressure:")
    print(json.dumps(
        assessment["pressure"],
        indent=2
    ))

    print("\nEnergy:")
    print(json.dumps(
        assessment["energy"],
        indent=2
    ))

    with open(
        OUT_DIR / "bdg2_verified_variables.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            assessment,
            f,
            indent=2
        )

# ============================================================
# DONE
# ============================================================

banner("BDG2 VARIABLE VERIFICATION COMPLETE")

print(
    """
Generated:

  data/metadata/bdg2_weather_site_counts.csv
  data/metadata/bdg2_weather_variable_availability.csv
  data/metadata/bdg2_weather_statistics.csv
  data/metadata/bdg2_metadata_snapshot.csv
  data/metadata/bdg2_meter_summary.csv
  data/metadata/bdg2_verified_variables.json

No raw BDG2 files were modified.
"""
)