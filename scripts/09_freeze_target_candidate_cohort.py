from pathlib import Path
import json
import math
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

PRIORITY_FILE = META / "opensensemap_fast_priority_boxes.csv"
BOX_META_FILE = META / "opensensemap_temporally_feasible_boxes.csv"

OUT = META / "opensensemap_frozen_target_candidates.csv"
MANIFEST = META / "opensensemap_frozen_target_candidates_manifest.json"

N_TARGET = 20

REQUIRED_SENSOR_COLUMNS = [
    "temperature_sensor_id",
    "relative_humidity_sensor_id",
    "pressure_sensor_id",
    "pm25_sensor_id",
]


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("FREEZE OPENSENSEMAP TARGET CANDIDATE COHORT")
print("=" * 90)

priority = pd.read_csv(
    PRIORITY_FILE,
    dtype={"box_id": str}
)

metadata = pd.read_csv(
    BOX_META_FILE,
    dtype={"box_id": str}
)

print("\nPriority candidates:", len(priority))

# Identify coordinate columns
print("\nMetadata columns:")
print(", ".join(metadata.columns))


# ============================================================
# MERGE
# ============================================================

# These were the names used in the earlier sensor inventory.
lat_col = None
lon_col = None

for candidate in ["latitude", "lat"]:
    if candidate in metadata.columns:
        lat_col = candidate
        break

for candidate in ["longitude", "lon", "lng"]:
    if candidate in metadata.columns:
        lon_col = candidate
        break

if lat_col is None or lon_col is None:
    raise RuntimeError(
        "Could not identify latitude/longitude columns."
    )

keep_meta = [
    "box_id",
    lat_col,
    lon_col,
]

for c in [
    "box_name",
    "exposure",
    "model",
    "metadata_created_at",
    "metadata_last_measurement_at",
]:
    if c in metadata.columns:
        keep_meta.append(c)

df = priority.merge(
    metadata[keep_meta],
    on="box_id",
    how="left",
)

df[lat_col] = pd.to_numeric(
    df[lat_col],
    errors="coerce"
)

df[lon_col] = pd.to_numeric(
    df[lon_col],
    errors="coerce"
)

# Remove candidates without usable coordinates
before = len(df)

df = df[
    df[lat_col].notna()
    & df[lon_col].notna()
].copy()

print(
    "\nCandidates with valid coordinates:",
    len(df),
    f"(removed {before - len(df)})"
)

if len(df) < N_TARGET:
    raise RuntimeError(
        f"Only {len(df)} candidates have valid coordinates."
    )


# ============================================================
# VERIFY SENSOR IDS
# ============================================================

for col in REQUIRED_SENSOR_COLUMNS:

    if col not in df.columns:
        raise RuntimeError(
            f"Missing required column: {col}"
        )

    missing = df[col].isna().sum()

    print(
        f"{col}: missing={missing}"
    )

df = df.dropna(
    subset=REQUIRED_SENSOR_COLUMNS
).copy()

if len(df) < N_TARGET:
    raise RuntimeError(
        "Insufficient candidates after sensor-ID validation."
    )


# ============================================================
# DETERMINISTIC GEOGRAPHIC SELECTION
# ============================================================

# Stable ordering before selection.
df = df.sort_values(
    "box_id"
).reset_index(drop=True)

# Deterministic starting point:
# candidate nearest the geographic centroid.
centroid_lat = df[lat_col].mean()
centroid_lon = df[lon_col].mean()

df["_centroid_distance"] = df.apply(
    lambda r: haversine(
        r[lat_col],
        r[lon_col],
        centroid_lat,
        centroid_lon,
    ),
    axis=1,
)

first_idx = (
    df.sort_values(
        ["_centroid_distance", "box_id"]
    ).index[0]
)

selected_indices = [first_idx]


# Repeatedly choose candidate maximizing its minimum
# distance to all already selected stations.
while len(selected_indices) < N_TARGET:

    best_idx = None
    best_min_distance = -1.0

    for idx, row in df.iterrows():

        if idx in selected_indices:
            continue

        min_distance = min(
            haversine(
                row[lat_col],
                row[lon_col],
                df.loc[s, lat_col],
                df.loc[s, lon_col],
            )
            for s in selected_indices
        )

        if (
            min_distance > best_min_distance
            or (
                math.isclose(
                    min_distance,
                    best_min_distance
                )
                and best_idx is not None
                and str(row["box_id"])
                < str(df.loc[best_idx, "box_id"])
            )
        ):
            best_idx = idx
            best_min_distance = min_distance

    selected_indices.append(best_idx)


selected = df.loc[
    selected_indices
].copy()

selected["selection_order"] = range(
    1,
    len(selected) + 1
)


# ============================================================
# DISTANCE TO NEAREST SELECTED NEIGHBOR
# ============================================================

nearest = []

for idx, row in selected.iterrows():

    distances = []

    for idx2, row2 in selected.iterrows():

        if idx == idx2:
            continue

        distances.append(
            haversine(
                row[lat_col],
                row[lon_col],
                row2[lat_col],
                row2[lon_col],
            )
        )

    nearest.append(min(distances))

selected[
    "nearest_selected_distance_km"
] = nearest


# ============================================================
# CLEAN + SAVE
# ============================================================

selected = selected.drop(
    columns=["_centroid_distance"],
    errors="ignore",
)

selected = selected.sort_values(
    "selection_order"
)

selected.to_csv(
    OUT,
    index=False
)


manifest = {
    "selection_stage":
        "pre-forecast target candidate freezing",

    "source_population":
        "opensensemap_fast_priority_boxes.csv",

    "priority_population_size":
        int(len(priority)),

    "candidate_pool_after_coordinate_and_sensor_validation":
        int(len(df)),

    "selected_candidate_count":
        int(len(selected)),

    "selection_rule":
        (
            "Deterministic geographic farthest-point sampling "
            "using latitude and longitude only. The initial "
            "candidate is the station nearest the candidate-pool "
            "geographic centroid; subsequent candidates maximize "
            "minimum great-circle distance from already selected "
            "stations, with box_id used only for deterministic "
            "tie-breaking."
        ),

    "forecasting_metrics_used":
        False,

    "target_outcomes_used":
        False,

    "final_eligibility_established":
        False,

    "next_stage":
        (
            "Exact chunked observation retrieval and verification "
            "of >=180-day span and >=70% hourly completeness for "
            "temperature, relative humidity, pressure, and PM2.5."
        ),

    "box_ids":
        selected[
            "box_id"
        ].tolist(),
}

with open(
    MANIFEST,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 90)
print("TARGET CANDIDATE COHORT FROZEN")
print("=" * 90)

print("\nSelected:", len(selected))

display_cols = [
    "selection_order",
    "box_id",
    lat_col,
    lon_col,
    "nearest_selected_distance_km",
]

for c in ["box_name", "model"]:
    if c in selected.columns:
        display_cols.insert(
            2,
            c
        )

print(
    "\n",
    selected[
        display_cols
    ].to_string(
        index=False
    )
)

print(
    "\nMinimum nearest-neighbor distance:",
    round(
        selected[
            "nearest_selected_distance_km"
        ].min(),
        2
    ),
    "km"
)

print(
    "\nMedian nearest-neighbor distance:",
    round(
        selected[
            "nearest_selected_distance_km"
        ].median(),
        2
    ),
    "km"
)

print("\nGenerated:")
print(
    "  data/metadata/"
    "opensensemap_frozen_target_candidates.csv"
)
print(
    "  data/metadata/"
    "opensensemap_frozen_target_candidates_manifest.json"
)

print("""
IMPORTANT:

These 20 boxes are frozen TARGET CANDIDATES, not yet the
final eligible target cohort.

Selection used:
  - semantic eligibility
  - temporal feasibility
  - endpoint coverage screening
  - geographic coordinates

Selection did NOT use:
  - forecasting MAE/RMSE
  - adaptation performance
  - PM2.5 prediction performance
  - uncertainty
  - shift-detection performance

The next stage retrieves observations and independently
verifies exact historical eligibility.
""")