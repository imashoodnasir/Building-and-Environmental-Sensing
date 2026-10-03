from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests
import time
import json


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "metadata"

INPUT = META_DIR / "opensensemap_semantically_eligible_sensors.csv"

CHECKPOINT = META_DIR / "opensensemap_history_screen_checkpoint.csv"
OUTPUT_SENSOR = META_DIR / "opensensemap_history_screen_sensors.csv"
OUTPUT_BOX = META_DIR / "opensensemap_history_screen_boxes.csv"
OUTPUT_ELIGIBLE = META_DIR / "opensensemap_history_eligible_boxes.csv"
OUTPUT_MAP = META_DIR / "opensensemap_selected_sensor_map.csv"
MANIFEST = META_DIR / "opensensemap_history_screen_manifest.json"


# ============================================================
# FROZEN SPECIFICATION
# ============================================================

START = pd.Timestamp("2026-01-01T00:00:00Z")
END = pd.Timestamp("2026-09-30T23:59:59Z")

MIN_SPAN_DAYS = 180
MIN_COMPLETENESS = 0.70

REQUIRED = [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]

EXPECTED_HOURS = len(
    pd.date_range(
        START.floor("h"),
        END.floor("h"),
        freq="h"
    )
)

API_BASE = "https://api.opensensemap.org"

REQUEST_TIMEOUT = 90
REQUEST_DELAY = 0.10
MAX_RETRIES = 5

INITIAL_CHUNK_DAYS = 7

# Observed API ceiling.
RECORD_LIMIT = 10000

# If even a 1-hour interval reaches the ceiling, do not
# silently accept it as complete.
MIN_CHUNK_SECONDS = 3600


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)


def iso_api(ts):
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_request(session, url, params):

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            r = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent":
                    "Building-Environmental-Sensing-Research/1.0"
                }
            )

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                wait = min(60, 5 * attempt)
                print(
                    f"\nRate limited. Waiting {wait}s..."
                )
                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:
                wait = min(30, 2 ** attempt)
                print(
                    f"\nServer error {r.status_code}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            return r

        except requests.RequestException as e:

            if attempt == MAX_RETRIES:
                raise

            wait = min(30, 2 ** attempt)

            print(
                f"\nRequest error: {e}\n"
                f"Retrying in {wait}s..."
            )

            time.sleep(wait)

    return None


def extract_timestamps(payload):

    if isinstance(payload, list):
        records = payload

    elif isinstance(payload, dict):

        if isinstance(payload.get("data"), list):
            records = payload["data"]

        elif isinstance(payload.get("measurements"), list):
            records = payload["measurements"]

        else:
            records = []

    else:
        records = []

    timestamps = []

    for item in records:

        if not isinstance(item, dict):
            continue

        ts = (
            item.get("createdAt")
            or item.get("timestamp")
            or item.get("time")
        )

        if ts is not None:
            timestamps.append(ts)

    if not timestamps:
        return pd.DatetimeIndex([], tz="UTC"), len(records)

    parsed = pd.to_datetime(
        timestamps,
        utc=True,
        errors="coerce"
    )

    parsed = parsed[~pd.isna(parsed)]

    return pd.DatetimeIndex(parsed), len(records)


# ============================================================
# FETCH ONE INTERVAL
# ============================================================

def fetch_interval(
    session,
    box_id,
    sensor_id,
    start,
    end,
    depth=0
):

    url = (
        f"{API_BASE}/boxes/"
        f"{box_id}/data/{sensor_id}"
    )

    params = {
        "from-date": iso_api(start),
        "to-date": iso_api(end),
        "format": "json",
        "download": "false",
    }

    response = safe_request(
        session,
        url,
        params
    )

    if response is None:
        raise RuntimeError("No API response.")

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:250]}"
        )

    try:
        payload = response.json()

    except Exception as e:
        raise RuntimeError(
            f"Invalid JSON: {e}"
        )

    timestamps, n_records = extract_timestamps(
        payload
    )

    # --------------------------------------------------------
    # POSSIBLE API TRUNCATION
    # --------------------------------------------------------

    if n_records >= RECORD_LIMIT:

        duration_seconds = (
            pd.Timestamp(end)
            - pd.Timestamp(start)
        ).total_seconds()

        if duration_seconds <= MIN_CHUNK_SECONDS:

            raise RuntimeError(
                "API record ceiling reached even for "
                f"minimum interval: {start} -> {end}. "
                "Cannot verify completeness safely."
            )

        midpoint = (
            pd.Timestamp(start)
            + (
                pd.Timestamp(end)
                - pd.Timestamp(start)
            ) / 2
        )

        # Prevent overlap around midpoint.
        left_end = midpoint
        right_start = midpoint + pd.Timedelta(
            microseconds=1
        )

        left_ts, left_requests = fetch_interval(
            session,
            box_id,
            sensor_id,
            start,
            left_end,
            depth + 1
        )

        right_ts, right_requests = fetch_interval(
            session,
            box_id,
            sensor_id,
            right_start,
            end,
            depth + 1
        )

        combined = left_ts.append(
            right_ts
        )

        return combined, (
            left_requests
            + right_requests
            + 1
        )

    return timestamps, 1


# ============================================================
# SCREEN ONE SENSOR
# ============================================================

def screen_sensor(
    session,
    box_id,
    sensor_id
):

    all_timestamps = []

    request_count = 0

    current = START

    while current <= END:

        chunk_end = min(
            current
            + pd.Timedelta(
                days=INITIAL_CHUNK_DAYS
            )
            - pd.Timedelta(
                microseconds=1
            ),
            END
        )

        timestamps, n_requests = fetch_interval(
            session,
            box_id,
            sensor_id,
            current,
            chunk_end
        )

        request_count += n_requests

        if len(timestamps) > 0:
            all_timestamps.append(
                timestamps
            )

        current = (
            chunk_end
            + pd.Timedelta(
                microseconds=1
            )
        )

        time.sleep(REQUEST_DELAY)

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if not all_timestamps:

        return {
            "status": "empty",
            "n_observations": 0,
            "n_hourly_bins": 0,
            "first_observation": None,
            "last_observation": None,
            "span_days": 0.0,
            "completeness": 0.0,
            "eligible_sensor": False,
            "api_requests": request_count,
        }

    timestamps = all_timestamps[0]

    for x in all_timestamps[1:]:
        timestamps = timestamps.append(x)

    # Defensive UTC conversion.
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            timestamps,
            utc=True,
            errors="coerce"
        )
    ).dropna()

    # Restrict to frozen interval.
    timestamps = timestamps[
        (timestamps >= START)
        & (timestamps <= END)
    ]

    if len(timestamps) == 0:

        return {
            "status": "empty",
            "n_observations": 0,
            "n_hourly_bins": 0,
            "first_observation": None,
            "last_observation": None,
            "span_days": 0.0,
            "completeness": 0.0,
            "eligible_sensor": False,
            "api_requests": request_count,
        }

    # Remove duplicate timestamp entries for counting.
    timestamps = timestamps.unique()

    timestamps = pd.DatetimeIndex(
        timestamps
    ).sort_values()

    first = timestamps.min()
    last = timestamps.max()

    span_days = (
        (last - first).total_seconds()
        / 86400.0
    )

    hourly_bins = (
        timestamps
        .floor("h")
        .unique()
    )

    n_hourly = len(hourly_bins)

    completeness = min(
        n_hourly / EXPECTED_HOURS,
        1.0
    )

    eligible = (
        span_days >= MIN_SPAN_DAYS
        and completeness
        >= MIN_COMPLETENESS
    )

    return {
        "status": "ok",
        "n_observations":
            int(len(timestamps)),
        "n_hourly_bins":
            int(n_hourly),
        "first_observation":
            first.isoformat(),
        "last_observation":
            last.isoformat(),
        "span_days":
            float(span_days),
        "completeness":
            float(completeness),
        "eligible_sensor":
            bool(eligible),
        "api_requests":
            int(request_count),
    }


# ============================================================
# LOAD INPUT
# ============================================================

banner(
    "OPENSENSEMAP CHUNKED HISTORICAL "
    "AVAILABILITY SCREEN"
)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(
    INPUT,
    dtype={
        "box_id": str,
        "sensor_id": str,
    }
)

df = df[
    df["canonical_variable"]
    .isin(REQUIRED)
].copy()

print("\nFrozen target period:")
print(" Start:", START)
print(" End:  ", END)

print("\nExpected hourly bins:", EXPECTED_HOURS)

print(
    "Minimum temporal span:",
    MIN_SPAN_DAYS,
    "days"
)

print(
    "Minimum completeness:",
    f"{MIN_COMPLETENESS * 100:.1f}%"
)

print(
    "Initial API chunk:",
    INITIAL_CHUNK_DAYS,
    "days"
)

print("\nCandidate boxes:",
      df["box_id"].nunique())

print("Candidate sensors:",
      len(df))


# ============================================================
# LOAD VALID CHECKPOINT
# ============================================================

results = []
completed_keys = set()

if CHECKPOINT.exists():

    previous = pd.read_csv(
        CHECKPOINT,
        dtype={
            "box_id": str,
            "sensor_id": str,
        }
    )

    # Only genuinely completed sensors are skipped.
    valid_status = {
        "ok",
        "empty",
    }

    previous_valid = previous[
        previous["status"].isin(
            valid_status
        )
    ].copy()

    results = (
        previous_valid
        .to_dict("records")
    )

    completed_keys = set(
        zip(
            previous_valid["box_id"],
            previous_valid["sensor_id"],
        )
    )

    print(
        "\nValid checkpoint found:",
        len(previous_valid),
        "completed sensors."
    )

else:

    print(
        "\nNo valid checkpoint found. "
        "Starting new chunked screen."
    )


# ============================================================
# SCREEN
# ============================================================

session = requests.Session()

total = len(df)
processed_this_run = 0


try:

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        box_id = str(row["box_id"])
        sensor_id = str(row["sensor_id"])

        key = (
            box_id,
            sensor_id
        )

        if key in completed_keys:
            continue

        variable = row[
            "canonical_variable"
        ]

        print(
            f"\r[{position}/{total}] "
            f"Box {box_id} | "
            f"{variable:<18}",
            end="",
            flush=True
        )

        try:

            result = screen_sensor(
                session,
                box_id,
                sensor_id
            )

        except Exception as e:

            result = {
                "status":
                    "request_exception",
                "n_observations": 0,
                "n_hourly_bins": 0,
                "first_observation": None,
                "last_observation": None,
                "span_days": 0.0,
                "completeness": 0.0,
                "eligible_sensor": False,
                "api_requests": None,
                "error": str(e),
            }

        record = {
            "box_id": box_id,
            "box_name":
                row.get("box_name"),
            "sensor_id": sensor_id,
            "sensor_title":
                row.get("sensor_title"),
            "sensor_unit":
                row.get("sensor_unit"),
            "canonical_variable":
                variable,
            "latitude":
                row.get("latitude"),
            "longitude":
                row.get("longitude"),
            **result,
        }

        results.append(record)

        # Mark only successful/empty screens as complete.
        if result["status"] in {
            "ok",
            "empty",
        }:
            completed_keys.add(key)

        processed_this_run += 1

        # Save frequently because each sensor may
        # require many API requests.
        if processed_this_run % 5 == 0:

            pd.DataFrame(
                results
            ).to_csv(
                CHECKPOINT,
                index=False
            )

except KeyboardInterrupt:

    print(
        "\n\nInterrupted by user."
    )

    pd.DataFrame(
        results
    ).to_csv(
        CHECKPOINT,
        index=False
    )

    print(
        "Checkpoint saved:",
        CHECKPOINT
    )

    raise SystemExit(0)


# ============================================================
# SAVE SENSOR RESULTS
# ============================================================

sensor_results = pd.DataFrame(
    results
)

sensor_results.to_csv(
    CHECKPOINT,
    index=False
)

sensor_results.to_csv(
    OUTPUT_SENSOR,
    index=False
)


# ============================================================
# SENSOR-LEVEL SUMMARY
# ============================================================

banner("SENSOR-LEVEL RESULTS")

for variable in REQUIRED:

    x = sensor_results[
        sensor_results[
            "canonical_variable"
        ] == variable
    ]

    ok = x[
        x["status"] == "ok"
    ]

    eligible = x[
        x["eligible_sensor"] == True
    ]

    print(f"\n{variable}")
    print("  screened:", len(x))
    print("  API OK:  ", len(ok))
    print("  eligible:", len(eligible))

    if len(ok) > 0:

        print(
            "  median completeness:",
            f"{ok['completeness'].median() * 100:.2f}%"
        )

        print(
            "  median span:",
            f"{ok['span_days'].median():.1f} days"
        )


# ============================================================
# DUPLICATE RESOLUTION
# ============================================================

eligible_sensors = sensor_results[
    sensor_results[
        "eligible_sensor"
    ] == True
].copy()


if not eligible_sensors.empty:

    eligible_sensors[
        "sensor_id_sort"
    ] = (
        eligible_sensors[
            "sensor_id"
        ].astype(str)
    )

    eligible_sensors = (
        eligible_sensors
        .sort_values(
            [
                "box_id",
                "canonical_variable",
                "completeness",
                "span_days",
                "n_observations",
                "sensor_id_sort",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                False,
                True,
            ]
        )
    )

    selected_sensors = (
        eligible_sensors
        .drop_duplicates(
            [
                "box_id",
                "canonical_variable"
            ],
            keep="first"
        )
        .copy()
    )

else:

    selected_sensors = (
        eligible_sensors.copy()
    )


# ============================================================
# REQUIRE ALL FOUR VARIABLES
# ============================================================

if not selected_sensors.empty:

    variable_counts = (
        selected_sensors
        .groupby("box_id")[
            "canonical_variable"
        ]
        .nunique()
    )

    eligible_box_ids = (
        variable_counts[
            variable_counts
            == len(REQUIRED)
        ]
        .index
        .astype(str)
        .tolist()
    )

else:

    eligible_box_ids = []


final_sensors = selected_sensors[
    selected_sensors[
        "box_id"
    ].astype(str).isin(
        eligible_box_ids
    )
].copy()


final_sensors.to_csv(
    OUTPUT_MAP,
    index=False
)


# ============================================================
# BOX-LEVEL SUMMARY
# ============================================================

box_rows = []

for box_id, group in (
    selected_sensors.groupby(
        "box_id"
    )
):

    variables = set(
        group[
            "canonical_variable"
        ]
    )

    box_rows.append(
        {
            "box_id": box_id,

            "box_name":
                group[
                    "box_name"
                ].iloc[0],

            "latitude":
                group[
                    "latitude"
                ].iloc[0],

            "longitude":
                group[
                    "longitude"
                ].iloc[0],

            "n_eligible_variables":
                len(variables),

            "has_temperature":
                "temperature"
                in variables,

            "has_relative_humidity":
                "relative_humidity"
                in variables,

            "has_pressure":
                "pressure"
                in variables,

            "has_pm25":
                "pm25"
                in variables,

            "minimum_completeness":
                group[
                    "completeness"
                ].min(),

            "minimum_span_days":
                group[
                    "span_days"
                ].min(),

            "eligible_all_four":
                set(REQUIRED)
                .issubset(variables),
        }
    )


box_results = pd.DataFrame(
    box_rows
)

box_results.to_csv(
    OUTPUT_BOX,
    index=False
)


if not box_results.empty:

    final_boxes = box_results[
        box_results[
            "eligible_all_four"
        ] == True
    ].copy()

else:

    final_boxes = (
        box_results.copy()
    )


final_boxes.to_csv(
    OUTPUT_ELIGIBLE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

banner("BOX-LEVEL ELIGIBILITY")

print(
    "\nBoxes with all four variables "
    "meeting frozen criteria:",
    len(final_boxes)
)

if len(final_boxes) > 0:

    print(
        "Median minimum completeness:",
        f"{final_boxes['minimum_completeness'].median() * 100:.2f}%"
    )

    print(
        "Median minimum span:",
        f"{final_boxes['minimum_span_days'].median():.1f} days"
    )


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "completed_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "target_period_start":
        START.isoformat(),

    "target_period_end":
        END.isoformat(),

    "expected_hourly_bins":
        EXPECTED_HOURS,

    "minimum_span_days":
        MIN_SPAN_DAYS,

    "minimum_completeness":
        MIN_COMPLETENESS,

    "initial_chunk_days":
        INITIAL_CHUNK_DAYS,

    "api_record_limit_guard":
        RECORD_LIMIT,

    "required_variables":
        REQUIRED,

    "candidate_boxes":
        int(
            df[
                "box_id"
            ].nunique()
        ),

    "candidate_sensors":
        int(len(df)),

    "eligible_boxes_all_four":
        int(len(final_boxes)),

    "selection_used_forecasting_results":
        False,

    "future_data_after_target_period_used":
        False,

    "duplicate_resolution": [
        "eligible sensor only",
        "highest completeness",
        "longest span",
        "largest observation count",
        "sensor_id lexical tie-break",
    ],
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


banner(
    "CHUNKED HISTORICAL SCREEN COMPLETE"
)

print(
    """
Generated:

  data/metadata/opensensemap_history_screen_checkpoint.csv
  data/metadata/opensensemap_history_screen_sensors.csv
  data/metadata/opensensemap_history_screen_boxes.csv
  data/metadata/opensensemap_history_eligible_boxes.csv
  data/metadata/opensensemap_selected_sensor_map.csv
  data/metadata/opensensemap_history_screen_manifest.json

No forecasting performance was used for station selection.
"""
)