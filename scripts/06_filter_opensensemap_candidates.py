from pathlib import Path
import pandas as pd
import json
import re


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

INPUT = META / "opensensemap_candidate_sensors.csv"

if not INPUT.exists():
    raise FileNotFoundError(INPUT)


def banner(x):
    print("\n" + "=" * 88)
    print(x)
    print("=" * 88)


def norm(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()

    x = (
        x.replace("μ", "µ")
         .replace("³", "3")
         .replace("^3", "3")
         .replace(" ", "")
    )

    return x


df = pd.read_csv(INPUT)

banner("OPENSENSEMAP SEMANTIC FILTERING")

print("\nCandidate sensor records:", len(df))
print("Candidate boxes:", df["box_id"].nunique())


# ============================================================
# VALID UNIT DEFINITIONS
# ============================================================

def valid_temperature(unit):

    u = norm(unit)

    valid = {
        "°c",
        "ºc",
        "˚c",
        "℃",
        "c",
        "celsius",
        "centigrade",
        "degc",
        "degreec",
        "degreesc",
        "(°c)",
        "*c",
        "c*",
        "c°",
    }

    return u in valid


def valid_humidity(unit):

    u = norm(unit)

    valid = {
        "%",
        "(%)",
        "%rh",
        "rh%",
        "rh",
        "%rf",
        "rf",
        "percent",
        "prozent",
        "rel%",
        "rel.%",
        "relativ",
        "ref%",
    }

    if u in valid:
        return True

    if "%" in u and "°c" not in u and "hpa" not in u:
        return True

    return False


def valid_pressure(unit):

    u = norm(unit)

    valid = {
        "hpa",
        "mbar",
        "millibar",
        "millibars",
        "mb",
        "kpa",
        "pa",
        "(pa)",
        "pascal",
        "bar",
        "inhg",
        "mmhg",
    }

    return u in valid


def valid_pm25(unit):

    u = norm(unit)

    # Canonical mass concentration will later be µg/m3.
    valid_exact = {
        "µg/m3",
        "ug/m3",
        "µg/m^3",
        "ug/m^3",
        "(µg/m3)",
        "microgram/m3",
        "microgramproqubicmeter",
        "microgrammprokubikmeter",
        "mug/m3",
        "mg/m3",
    }

    if u in valid_exact:
        return True

    if (
        ("µg/m3" in u or "ug/m3" in u)
        and "ppm" not in u
    ):
        return True

    return False


validators = {
    "temperature": valid_temperature,
    "relative_humidity": valid_humidity,
    "pressure": valid_pressure,
    "pm25": valid_pm25,
}


# ============================================================
# UNIT VALIDATION
# ============================================================

df["unit_valid"] = False

for variable, validator in validators.items():

    mask = df["canonical_variable"] == variable

    df.loc[mask, "unit_valid"] = (
        df.loc[mask, "sensor_unit"]
        .apply(validator)
    )


banner("VALID UNIT COUNTS")

for variable in validators:

    x = df[
        df["canonical_variable"] == variable
    ]

    valid = x[x["unit_valid"]]

    print(f"\n{variable}")
    print(f"  total sensors: {len(x)}")
    print(f"  valid sensors: {len(valid)}")
    print(
        f"  valid boxes:   "
        f"{valid['box_id'].nunique()}"
    )


# ============================================================
# EXPOSURE AUDIT
# ============================================================

banner("EXPOSURE VALUES")

exposure_counts = (
    df["exposure"]
    .fillna("<missing>")
    .astype(str)
    .value_counts()
)

print(exposure_counts.to_string())


# ============================================================
# OUTDOOR FILTER
# ============================================================

def outdoor_exposure(x):

    x = str(x).strip().lower()

    return x in {
        "outdoor",
        "outside",
    }


df["outdoor_valid"] = (
    df["exposure"]
    .apply(outdoor_exposure)
)


# ============================================================
# SEMANTICALLY VALID SENSORS
# ============================================================

valid = df[
    df["unit_valid"]
    & df["outdoor_valid"]
].copy()


banner("AFTER UNIT + OUTDOOR FILTER")

print("\nSensor records:", len(valid))
print("Boxes:", valid["box_id"].nunique())


# ============================================================
# REQUIRE ALL FOUR VARIABLES PER BOX
# ============================================================

required = {
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
}


box_variables = (
    valid.groupby("box_id")[
        "canonical_variable"
    ]
    .agg(lambda x: set(x))
)


eligible_ids = [
    box_id
    for box_id, variables in box_variables.items()
    if required.issubset(variables)
]


eligible = valid[
    valid["box_id"].isin(eligible_ids)
].copy()


banner("SEMANTICALLY ELIGIBLE BOXES")

print(
    "\nBoxes with valid outdoor "
    "T + RH + P + PM2.5:",
    len(eligible_ids)
)

print(
    "Associated sensor records:",
    len(eligible)
)


# ============================================================
# UNIT DISTRIBUTION AFTER FILTER
# ============================================================

banner("RETAINED UNIT DISTRIBUTIONS")

for variable in sorted(required):

    x = eligible[
        eligible["canonical_variable"] == variable
    ]

    print(f"\n{variable}")

    counts = (
        x["sensor_unit"]
        .fillna("<missing>")
        .value_counts()
        .head(30)
    )

    print(counts.to_string())


# ============================================================
# DUPLICATE SENSOR ANALYSIS
# ============================================================

banner("MULTIPLE SENSORS PER VARIABLE")

counts = (
    eligible.groupby(
        ["box_id", "canonical_variable"]
    )
    .size()
    .reset_index(name="n_sensors")
)

duplicates = counts[
    counts["n_sensors"] > 1
]

print(
    "\nBox-variable combinations with "
    "multiple candidate sensors:",
    len(duplicates)
)

print(
    "Boxes affected:",
    duplicates["box_id"].nunique()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    META / "opensensemap_semantic_filter_audit.csv",
    index=False
)

valid.to_csv(
    META / "opensensemap_valid_outdoor_sensors.csv",
    index=False
)

eligible.to_csv(
    META / "opensensemap_semantically_eligible_sensors.csv",
    index=False
)

counts.to_csv(
    META / "opensensemap_sensor_multiplicity.csv",
    index=False
)


# ============================================================
# BOX-LEVEL MANIFEST
# ============================================================

box_info = (
    eligible[
        [
            "box_id",
            "box_name",
            "latitude",
            "longitude",
            "exposure",
            "model",
            "box_created_at",
            "box_updated_at",
        ]
    ]
    .drop_duplicates("box_id")
)


variable_pivot = (
    eligible.groupby(
        ["box_id", "canonical_variable"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)


box_info = box_info.merge(
    variable_pivot,
    on="box_id",
    how="left"
)


box_info.to_csv(
    META / "opensensemap_eligible_boxes.csv",
    index=False
)


summary = {
    "initial_candidate_boxes":
        int(df["box_id"].nunique()),

    "valid_outdoor_boxes":
        int(valid["box_id"].nunique()),

    "eligible_all_four_boxes":
        int(len(eligible_ids)),

    "required_variables":
        sorted(required),

    "selection_used_forecasting_results":
        False,

    "selection_used_observation_completeness":
        False,

    "selection_stage":
        "semantic_metadata_only",
}


with open(
    META / "opensensemap_semantic_filter_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


banner("SEMANTIC FILTER COMPLETE")

print(
    """
Generated:

  data/metadata/opensensemap_semantic_filter_audit.csv
  data/metadata/opensensemap_valid_outdoor_sensors.csv
  data/metadata/opensensemap_semantically_eligible_sensors.csv
  data/metadata/opensensemap_sensor_multiplicity.csv
  data/metadata/opensensemap_eligible_boxes.csv
  data/metadata/opensensemap_semantic_filter_summary.json

No forecasting results were used.
No completeness threshold was applied.
No station was manually selected.
"""
)