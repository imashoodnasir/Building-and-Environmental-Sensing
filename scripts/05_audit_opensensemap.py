from pathlib import Path
from datetime import datetime, timezone
import json
import urllib.request
import urllib.parse
import time
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "opensensemap"
META_DIR = ROOT / "data" / "metadata"

RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE = "https://api.opensensemap.org"

# First query metadata only.
# Do NOT download observations yet.

PHENOMENA = {
    "temperature": [
        "temperature",
        "temperatur",
    ],
    "relative_humidity": [
        "relative humidity",
        "humidity",
        "luftfeuchte",
    ],
    "pressure": [
        "pressure",
        "luftdruck",
        "barometric pressure",
    ],
    "pm25": [
        "pm2.5",
        "pm25",
        "particulate matter 2.5",
        "feinstaub pm2.5",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def normalize(text):
    if text is None:
        return ""

    return (
        str(text)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def classify_sensor(title):

    title_norm = normalize(title)

    for canonical, aliases in PHENOMENA.items():

        for alias in aliases:

            if normalize(alias) in title_norm:
                return canonical

    return None


def get_json(url, timeout=60):

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Building-Environmental-Sensing-Research/1.0"
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=timeout
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


# ============================================================
# QUERY TIME
# ============================================================

query_time = datetime.now(
    timezone.utc
).isoformat()

banner("OPENSENSEMAP METADATA AUDIT")

print()
print("Query time (UTC):", query_time)


# ============================================================
# FETCH BOX METADATA
# ============================================================

# openSenseMap supports geographic queries.
# For the initial audit we request publicly visible boxes
# through the boxes endpoint and inspect their sensor metadata.
#
# The response can be large, therefore the script first attempts
# a broad metadata query and saves the exact response.

url = f"{API_BASE}/boxes"

print()
print("Endpoint:")
print(url)

print()
print("Requesting box metadata...")


try:

    boxes = get_json(url)

except Exception as e:

    print()
    print("API REQUEST FAILED")
    print(str(e))
    print()
    print(
        "Do not modify the experiment configuration. "
        "Send this error output before continuing."
    )

    raise SystemExit(1)


# ============================================================
# NORMALIZE RESPONSE
# ============================================================

if isinstance(boxes, dict):

    if "data" in boxes and isinstance(
        boxes["data"],
        list
    ):
        boxes_list = boxes["data"]

    elif "boxes" in boxes and isinstance(
        boxes["boxes"],
        list
    ):
        boxes_list = boxes["boxes"]

    else:
        boxes_list = [boxes]

elif isinstance(boxes, list):

    boxes_list = boxes

else:

    raise RuntimeError(
        "Unexpected openSenseMap API response."
    )


print("Boxes returned:", len(boxes_list))


# ============================================================
# SAVE RAW API RESPONSE
# ============================================================

timestamp_tag = (
    datetime.now(timezone.utc)
    .strftime("%Y%m%dT%H%M%SZ")
)

raw_file = (
    RAW_DIR
    / f"boxes_metadata_{timestamp_tag}.json"
)

with open(
    raw_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        boxes,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("Raw response saved:")
print(raw_file)


# ============================================================
# SENSOR INVENTORY
# ============================================================

banner("SENSOR INVENTORY")

records = []


for box in boxes_list:

    if not isinstance(box, dict):
        continue

    box_id = (
        box.get("_id")
        or box.get("id")
    )

    box_name = (
        box.get("name")
        or box.get("title")
    )

    exposure = box.get("exposure")

    model = box.get("model")

    created_at = (
        box.get("createdAt")
        or box.get("created_at")
    )

    updated_at = (
        box.get("updatedAt")
        or box.get("updated_at")
    )

    # ---------------------------------------------
    # LOCATION
    # ---------------------------------------------

    lat = None
    lon = None

    current_location = (
        box.get("currentLocation")
        or box.get("location")
    )

    if isinstance(current_location, dict):

        coordinates = (
            current_location
            .get("coordinates")
        )

        if (
            isinstance(coordinates, list)
            and len(coordinates) >= 2
        ):
            lon = coordinates[0]
            lat = coordinates[1]

    # ---------------------------------------------
    # SENSORS
    # ---------------------------------------------

    sensors = box.get("sensors", [])

    if not isinstance(sensors, list):
        continue

    for sensor in sensors:

        if not isinstance(sensor, dict):
            continue

        sensor_id = (
            sensor.get("_id")
            or sensor.get("id")
        )

        title = sensor.get("title")

        unit = sensor.get("unit")

        sensor_type = (
            sensor.get("sensorType")
            or sensor.get("type")
        )

        canonical = classify_sensor(
            title
        )

        records.append(
            {
                "box_id": box_id,
                "box_name": box_name,
                "exposure": exposure,
                "model": model,
                "latitude": lat,
                "longitude": lon,
                "box_created_at": created_at,
                "box_updated_at": updated_at,
                "sensor_id": sensor_id,
                "sensor_title": title,
                "sensor_unit": unit,
                "sensor_type": sensor_type,
                "canonical_variable": canonical,
            }
        )


inventory = pd.DataFrame(records)


print()
print("Sensors discovered:", len(inventory))


if inventory.empty:

    print()
    print(
        "No sensor metadata were returned. "
        "Stop here and send the terminal output."
    )

    raise SystemExit(1)


# ============================================================
# SAVE COMPLETE INVENTORY
# ============================================================

inventory_file = (
    META_DIR
    / "opensensemap_sensor_inventory.csv"
)

inventory.to_csv(
    inventory_file,
    index=False
)


# ============================================================
# CANDIDATE VARIABLES
# ============================================================

candidate = inventory[
    inventory["canonical_variable"]
    .notna()
].copy()


candidate_file = (
    META_DIR
    / "opensensemap_candidate_sensors.csv"
)

candidate.to_csv(
    candidate_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

banner("TARGET VARIABLE AVAILABILITY")

summary_rows = []


for variable in [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]:

    subset = candidate[
        candidate["canonical_variable"]
        == variable
    ]

    n_sensors = len(subset)

    n_boxes = (
        subset["box_id"]
        .nunique()
    )

    units = sorted(
        subset["sensor_unit"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print()
    print(variable)
    print("  sensors:", n_sensors)
    print("  boxes:  ", n_boxes)
    print("  units:  ", units)

    summary_rows.append(
        {
            "variable": variable,
            "n_sensors": int(n_sensors),
            "n_boxes": int(n_boxes),
            "units": " | ".join(units),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    META_DIR
    / "opensensemap_variable_summary.csv",
    index=False
)


# ============================================================
# MULTIVARIATE BOXES
# ============================================================

banner("MULTIVARIATE BOX ANALYSIS")


box_variables = (
    candidate
    .groupby("box_id")[
        "canonical_variable"
    ]
    .agg(
        lambda x: sorted(set(x))
    )
    .reset_index()
)


box_variables[
    "n_required_variables"
] = box_variables[
    "canonical_variable"
].apply(len)


def contains_required(values):

    required = {
        "temperature",
        "relative_humidity",
        "pressure",
        "pm25",
    }

    return required.issubset(
        set(values)
    )


box_variables[
    "has_all_four"
] = box_variables[
    "canonical_variable"
].apply(
    contains_required
)


all_four = box_variables[
    box_variables["has_all_four"]
]


print()
print(
    "Boxes containing all four "
    "T/RH/P/PM2.5:",
    len(all_four)
)


# Merge metadata for candidate boxes

if len(all_four) > 0:

    selected = (
        candidate[
            candidate["box_id"]
            .isin(all_four["box_id"])
        ]
        .sort_values(
            [
                "box_id",
                "canonical_variable"
            ]
        )
    )

else:

    selected = candidate.iloc[0:0].copy()


selected.to_csv(
    META_DIR
    / "opensensemap_all_four_candidate_boxes.csv",
    index=False
)


# ============================================================
# COMBINATION COUNTS
# ============================================================

box_variables[
    "combination"
] = box_variables[
    "canonical_variable"
].apply(
    lambda x: " + ".join(x)
)


combination_counts = (
    box_variables[
        "combination"
    ]
    .value_counts()
    .reset_index()
)

combination_counts.columns = [
    "variable_combination",
    "n_boxes",
]


combination_counts.to_csv(
    META_DIR
    / "opensensemap_variable_combinations.csv",
    index=False
)


print()
print("Most common relevant combinations:")
print()

print(
    combination_counts
    .head(20)
    .to_string(index=False)
)


# ============================================================
# AUDIT MANIFEST
# ============================================================

manifest = {

    "query_time_utc": query_time,

    "api_base": API_BASE,

    "endpoint": "/boxes",

    "raw_response_file": str(
        raw_file.relative_to(ROOT)
    ),

    "number_of_boxes_returned":
        int(len(boxes_list)),

    "number_of_sensors":
        int(len(inventory)),

    "number_of_candidate_sensors":
        int(len(candidate)),

    "boxes_with_all_four_variables":
        int(len(all_four)),

    "variables": (
        summary_df
        .set_index("variable")
        .to_dict("index")
    ),
}


with open(
    META_DIR
    / "opensensemap_audit_manifest.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

banner("OPENSENSEMAP METADATA AUDIT COMPLETE")

print(
    """
Generated:

  data/raw/opensensemap/boxes_metadata_<timestamp>.json

  data/metadata/opensensemap_sensor_inventory.csv
  data/metadata/opensensemap_candidate_sensors.csv
  data/metadata/opensensemap_variable_summary.csv
  data/metadata/opensensemap_all_four_candidate_boxes.csv
  data/metadata/opensensemap_variable_combinations.csv
  data/metadata/opensensemap_audit_manifest.json

No observations have been downloaded yet.
No station has been selected yet.
No forecasting results have been examined.
"""
)