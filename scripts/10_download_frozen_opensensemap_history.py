from pathlib import Path
import json
import time
import random
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "metadata"
RAW = ROOT / "data" / "raw" / "opensensemap" / "frozen_history"

COHORT_FILE = META / "opensensemap_frozen_target_candidates.csv"

CHECKPOINT_FILE = (
    META / "opensensemap_frozen_history_download_checkpoint.csv"
)

MANIFEST_FILE = (
    META / "opensensemap_frozen_history_download_manifest.json"
)

RAW.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01T00:00:00Z")
END = pd.Timestamp("2026-09-30T23:59:59Z")

CHUNK_DAYS = 14

TIMEOUT = 180
MAX_RETRIES = 6

API_BASE = "https://api.opensensemap.org"

VARIABLES = {
    "temperature": "temperature_sensor_id",
    "relative_humidity": "relative_humidity_sensor_id",
    "pressure": "pressure_sensor_id",
    "pm25": "pm25_sensor_id",
}

session = requests.Session()

session.headers.update({
    "User-Agent":
        "Building-Environmental-Sensing-Research/1.0"
})


# ============================================================
# CHUNK GENERATION
# ============================================================

def make_chunks(start, end, days=14):

    chunks = []

    current = start

    while current <= end:

        chunk_end = min(
            current + pd.Timedelta(days=days)
            - pd.Timedelta(seconds=1),
            end
        )

        chunks.append(
            (current, chunk_end)
        )

        current = chunk_end + pd.Timedelta(seconds=1)

    return chunks


CHUNKS = make_chunks(
    START,
    END,
    CHUNK_DAYS
)


# ============================================================
# API REQUEST
# ============================================================

def request_chunk(
    box_id,
    sensor_id,
    start,
    end,
):

    url = (
        f"{API_BASE}/boxes/"
        f"{box_id}/data/{sensor_id}"
    )

    params = {
        "from-date":
            start.strftime("%Y-%m-%dT%H:%M:%SZ"),

        "to-date":
            end.strftime("%Y-%m-%dT%H:%M:%SZ"),

        "format": "json",
        "download": "false",
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            t0 = time.time()

            r = session.get(
                url,
                params=params,
                timeout=TIMEOUT,
            )

            elapsed = time.time() - t0

            if r.status_code == 200:

                payload = r.json()

                if isinstance(payload, list):

                    records = payload

                elif isinstance(payload, dict):

                    records = payload.get(
                        "data",
                        payload.get(
                            "measurements",
                            []
                        )
                    )

                else:

                    records = []

                return {
                    "success": True,
                    "records": records,
                    "http_status": 200,
                    "elapsed": elapsed,
                    "attempts": attempt,
                    "error": None,
                }

            if r.status_code == 429:

                wait = min(
                    120,
                    10 * attempt
                )

                print(
                    f"      HTTP 429 -> "
                    f"waiting {wait}s"
                )

                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:

                wait = min(
                    60,
                    2 ** attempt
                )

                print(
                    f"      HTTP {r.status_code} -> "
                    f"retry {attempt}/{MAX_RETRIES}, "
                    f"wait {wait}s"
                )

                time.sleep(wait)
                continue

            return {
                "success": False,
                "records": [],
                "http_status": r.status_code,
                "elapsed": elapsed,
                "attempts": attempt,
                "error": r.text[:500],
            }

        except requests.RequestException as e:

            last_error = str(e)

            if attempt < MAX_RETRIES:

                wait = min(
                    60,
                    2 ** attempt
                )

                print(
                    f"      Request exception -> "
                    f"retry {attempt}/{MAX_RETRIES}, "
                    f"wait {wait}s"
                )

                time.sleep(wait)

    return {
        "success": False,
        "records": [],
        "http_status": None,
        "elapsed": None,
        "attempts": MAX_RETRIES,
        "error": last_error,
    }


# ============================================================
# PARSE RECORDS
# ============================================================

def records_to_dataframe(
    records,
    box_id,
    sensor_id,
    variable,
):

    rows = []

    for item in records:

        if not isinstance(item, dict):
            continue

        timestamp = (
            item.get("createdAt")
            or item.get("timestamp")
            or item.get("time")
        )

        value = (
            item.get("value")
            if "value" in item
            else item.get("measurement")
        )

        if timestamp is None:
            continue

        rows.append({
            "box_id": box_id,
            "sensor_id": sensor_id,
            "canonical_variable": variable,
            "timestamp": timestamp,
            "raw_value": value,
        })

    if not rows:

        return pd.DataFrame(
            columns=[
                "box_id",
                "sensor_id",
                "canonical_variable",
                "timestamp",
                "raw_value",
            ]
        )

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df = df[
        df["timestamp"].notna()
    ].copy()

    df = df[
        (df["timestamp"] >= START)
        &
        (df["timestamp"] <= END)
    ]

    return df


# ============================================================
# LOAD COHORT
# ============================================================

print("=" * 92)
print("EXACT OPENSENSEMAP FROZEN-HISTORY DOWNLOAD")
print("=" * 92)

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={"box_id": str}
)

print("\nFrozen boxes:", len(cohort))
print("Variables per box:", len(VARIABLES))
print("Total sensors:", len(cohort) * len(VARIABLES))
print("Chunks per sensor:", len(CHUNKS))

print(
    "Maximum planned requests:",
    len(cohort)
    * len(VARIABLES)
    * len(CHUNKS)
)

print("\nPeriod:")
print(" Start:", START)
print(" End:  ", END)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint_rows = []

if CHECKPOINT_FILE.exists():

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE,
        dtype={
            "box_id": str,
            "sensor_id": str,
        }
    )

    checkpoint_rows = (
        checkpoint.to_dict("records")
    )

    completed = set(
        zip(
            checkpoint["box_id"],
            checkpoint["sensor_id"],
            checkpoint["canonical_variable"],
        )
    )

    print(
        "\nCheckpoint found."
    )

    print(
        "Completed sensor streams:",
        len(completed)
    )

else:

    completed = set()

    print(
        "\nNo checkpoint found."
    )


# ============================================================
# DOWNLOAD
# ============================================================

try:

    total_streams = (
        len(cohort)
        * len(VARIABLES)
    )

    stream_counter = 0

    for _, box in cohort.sort_values(
        "selection_order"
    ).iterrows():

        box_id = str(
            box["box_id"]
        )

        order = int(
            box["selection_order"]
        )

        print(
            "\n" + "=" * 92
        )

        print(
            f"BOX {order}/"
            f"{len(cohort)}: {box_id}"
        )

        print(
            "=" * 92
        )

        for variable, sensor_col in VARIABLES.items():

            stream_counter += 1

            sensor_id = str(
                box[sensor_col]
            )

            key = (
                box_id,
                sensor_id,
                variable,
            )

            if key in completed:

                print(
                    f"  [{stream_counter}/{total_streams}] "
                    f"{variable}: already complete"
                )

                continue

            print(
                f"\n  [{stream_counter}/{total_streams}] "
                f"{variable}"
            )

            print(
                f"  Sensor: {sensor_id}"
            )

            all_frames = []

            failed = False

            total_api_records = 0
            total_attempts = 0
            total_elapsed = 0.0

            for chunk_number, (
                chunk_start,
                chunk_end,
            ) in enumerate(
                CHUNKS,
                start=1
            ):

                print(
                    f"    Chunk "
                    f"{chunk_number:02d}/"
                    f"{len(CHUNKS)} "
                    f"{chunk_start.date()} -> "
                    f"{chunk_end.date()}",
                    end="",
                    flush=True
                )

                response = request_chunk(
                    box_id,
                    sensor_id,
                    chunk_start,
                    chunk_end,
                )

                total_attempts += (
                    response["attempts"]
                )

                if response["elapsed"]:

                    total_elapsed += (
                        response["elapsed"]
                    )

                if not response["success"]:

                    print(
                        "  FAILED"
                    )

                    print(
                        "      Error:",
                        response["error"]
                    )

                    failed = True
                    break

                records = response[
                    "records"
                ]

                n = len(records)

                total_api_records += n

                print(
                    f"  records={n}"
                )

                # Critical safety condition.
                if n >= 10000:

                    print(
                        "\n      ERROR: "
                        "10,000-record limit reached."
                    )

                    print(
                        "      This stream requires "
                        "smaller adaptive chunks."
                    )

                    failed = True
                    break

                frame = records_to_dataframe(
                    records,
                    box_id,
                    sensor_id,
                    variable,
                )

                if not frame.empty:

                    all_frames.append(
                        frame
                    )

                # Small polite delay.
                time.sleep(
                    0.15
                    + random.uniform(
                        0.0,
                        0.10
                    )
                )

            # ------------------------------------------------
            # SAVE ONLY COMPLETE SENSOR STREAM
            # ------------------------------------------------

            if failed:

                print(
                    "\n  Stream NOT checkpointed."
                )

                print(
                    "  It will be retried on the "
                    "next execution."
                )

                continue

            if all_frames:

                data = pd.concat(
                    all_frames,
                    ignore_index=True
                )

                before = len(data)

                data = (
                    data
                    .sort_values("timestamp")
                    .drop_duplicates(
                        subset=[
                            "timestamp",
                            "raw_value",
                        ]
                    )
                    .reset_index(drop=True)
                )

                duplicates_removed = (
                    before - len(data)
                )

            else:

                data = pd.DataFrame(
                    columns=[
                        "box_id",
                        "sensor_id",
                        "canonical_variable",
                        "timestamp",
                        "raw_value",
                    ]
                )

                duplicates_removed = 0

            box_dir = (
                RAW
                / f"{order:02d}_{box_id}"
            )

            box_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            outfile = (
                box_dir
                / f"{variable}_{sensor_id}.csv"
            )

            data.to_csv(
                outfile,
                index=False
            )

            if len(data):

                first_ts = (
                    data["timestamp"]
                    .min()
                    .isoformat()
                )

                last_ts = (
                    data["timestamp"]
                    .max()
                    .isoformat()
                )

                hourly_bins = (
                    data["timestamp"]
                    .dt.floor("h")
                    .nunique()
                )

            else:

                first_ts = None
                last_ts = None
                hourly_bins = 0

            checkpoint_record = {
                "selection_order": order,
                "box_id": box_id,
                "canonical_variable":
                    variable,
                "sensor_id": sensor_id,
                "status": "complete",
                "api_records":
                    total_api_records,
                "stored_records":
                    len(data),
                "duplicates_removed":
                    duplicates_removed,
                "hourly_bins_raw":
                    hourly_bins,
                "first_timestamp":
                    first_ts,
                "last_timestamp":
                    last_ts,
                "total_request_attempts":
                    total_attempts,
                "total_elapsed_seconds":
                    round(
                        total_elapsed,
                        3
                    ),
                "file":
                    str(
                        outfile.relative_to(
                            ROOT
                        )
                    ),
            }

            checkpoint_rows.append(
                checkpoint_record
            )

            completed.add(key)

            pd.DataFrame(
                checkpoint_rows
            ).to_csv(
                CHECKPOINT_FILE,
                index=False
            )

            print(
                f"\n  Saved: {len(data):,} records"
            )

            print(
                f"  Raw hourly bins: {hourly_bins:,}"
            )

            print(
                f"  Duplicate records removed: "
                f"{duplicates_removed:,}"
            )


except KeyboardInterrupt:

    print(
        "\n\nDownload interrupted by user."
    )

    print(
        "Completed sensor streams have "
        "already been checkpointed."
    )

    raise SystemExit(0)


# ============================================================
# MANIFEST
# ============================================================

checkpoint = pd.DataFrame(
    checkpoint_rows
)

manifest = {
    "dataset": "openSenseMap",
    "cohort":
        "frozen target candidate cohort",
    "number_of_boxes":
        int(len(cohort)),
    "variables":
        list(VARIABLES.keys()),
    "period_start":
        START.isoformat(),
    "period_end":
        END.isoformat(),
    "chunk_days":
        CHUNK_DAYS,
    "api_record_limit_guard":
        10000,
    "download_completed_streams":
        int(len(completed)),
    "expected_streams":
        int(
            len(cohort)
            * len(VARIABLES)
        ),
    "raw_values_modified":
        False,
    "hourly_aggregation_performed":
        False,
    "physical_range_filtering_performed":
        False,
    "unit_conversion_performed":
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


print("\n" + "=" * 92)
print("FROZEN-HISTORY DOWNLOAD COMPLETE")
print("=" * 92)

print(
    "\nCompleted streams:",
    len(completed)
)

print(
    "Expected streams:",
    len(cohort)
    * len(VARIABLES)
)

print(
    "\nCheckpoint:"
)

print(
    " data/metadata/"
    "opensensemap_frozen_history_download_checkpoint.csv"
)

print(
    "\nRaw observation directory:"
)

print(
    " data/raw/opensensemap/frozen_history/"
)

print("""
IMPORTANT:

This stage preserves raw retrieved observations.

It has NOT:
  - converted units
  - filtered physical values
  - interpolated missing observations
  - aggregated observations hourly
  - calculated final completeness
  - selected stations using forecasting performance

The next stage performs auditable harmonization and exact
eligibility verification.
""")