from pathlib import Path
import time
import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

INPUT = META / "opensensemap_temporally_feasible_sensors.csv"

CHECKPOINT = META / "opensensemap_fast_qualification_checkpoint.csv"
SENSOR_OUT = META / "opensensemap_fast_qualification_sensors.csv"
PRIORITY_OUT = META / "opensensemap_fast_priority_boxes.csv"
SECONDARY_OUT = META / "opensensemap_secondary_screen_boxes.csv"

API_BASE = "https://api.opensensemap.org"

REQUIRED = [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]

# Two separated 14-day windows.
PROBES = [
    ("jan", "2026-01-01T00:00:00Z", "2026-01-14T23:59:59Z"),
    ("sep", "2026-09-01T00:00:00Z", "2026-09-14T23:59:59Z"),
]

MIN_COMPLETENESS = 0.70

TIMEOUT = 90
MAX_RETRIES = 3
REQUEST_DELAY = 0.15

session = requests.Session()

session.headers.update({
    "User-Agent":
        "Building-Environmental-Sensing-Research/1.0"
})


# ============================================================
# REQUEST
# ============================================================

def request_data(box_id, sensor_id, start, end):

    url = (
        f"{API_BASE}/boxes/"
        f"{box_id}/data/{sensor_id}"
    )

    params = {
        "from-date": start,
        "to-date": end,
        "format": "json",
        "download": "false",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

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
                        payload.get("measurements", [])
                    )

                else:
                    records = []

                return {
                    "status": "ok",
                    "records": records,
                    "elapsed": elapsed,
                    "error": None,
                }

            if r.status_code == 429:

                wait = 5 * attempt

                print(
                    f"\n    HTTP 429; waiting {wait}s"
                )

                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:

                wait = 2 ** attempt

                print(
                    f"\n    HTTP {r.status_code}; "
                    f"retry {attempt}/{MAX_RETRIES}"
                )

                time.sleep(wait)
                continue

            return {
                "status": f"http_{r.status_code}",
                "records": [],
                "elapsed": elapsed,
                "error": r.text[:200],
            }

        except requests.RequestException as e:

            last_error = str(e)

            if attempt < MAX_RETRIES:

                wait = 2 ** attempt

                print(
                    f"\n    Request exception; "
                    f"retry {attempt}/{MAX_RETRIES}"
                )

                time.sleep(wait)

    return {
        "status": "request_failed",
        "records": [],
        "elapsed": None,
        "error": last_error,
    }


# ============================================================
# HOURLY COMPLETENESS
# ============================================================

def calculate_completeness(records, start, end):

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

    ts = pd.to_datetime(
        timestamps,
        utc=True,
        errors="coerce"
    )

    ts = pd.Series(ts).dropna()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    ts = ts[
        (ts >= start_ts)
        & (ts <= end_ts)
    ]

    hourly = (
        ts.dt.floor("h").nunique()
        if len(ts)
        else 0
    )

    expected = int(
        (
            end_ts.floor("h")
            - start_ts.floor("h")
        ).total_seconds() / 3600
    ) + 1

    completeness = (
        hourly / expected
        if expected
        else 0.0
    )

    return {
        "records": len(ts),
        "hourly_bins": hourly,
        "expected_hours": expected,
        "completeness": completeness,
        "first":
            ts.min().isoformat()
            if len(ts) else None,
        "last":
            ts.max().isoformat()
            if len(ts) else None,
    }


# ============================================================
# TEST SENSOR
# ============================================================

def test_sensor(row):

    box_id = str(row["box_id"])
    sensor_id = str(row["sensor_id"])

    result = {
        "box_id": box_id,
        "sensor_id": sensor_id,
        "canonical_variable":
            row["canonical_variable"],
        "sensor_title":
            row.get("sensor_title"),
        "sensor_unit":
            row.get("sensor_unit"),
    }

    passed = 0
    failures = 0

    for label, start, end in PROBES:

        response = request_data(
            box_id,
            sensor_id,
            start,
            end
        )

        result[f"{label}_status"] = (
            response["status"]
        )

        result[f"{label}_elapsed"] = (
            response["elapsed"]
        )

        if response["status"] != "ok":

            result[f"{label}_records"] = None
            result[f"{label}_hourly_bins"] = None
            result[f"{label}_completeness"] = None

            failures += 1
            continue

        # Safety check
        if len(response["records"]) >= 10000:

            result[f"{label}_records"] = (
                len(response["records"])
            )

            result[f"{label}_hourly_bins"] = None
            result[f"{label}_completeness"] = None

            result[f"{label}_status"] = (
                "possible_truncation"
            )

            failures += 1
            continue

        stats = calculate_completeness(
            response["records"],
            start,
            end
        )

        result[f"{label}_records"] = (
            stats["records"]
        )

        result[f"{label}_hourly_bins"] = (
            stats["hourly_bins"]
        )

        result[f"{label}_completeness"] = (
            stats["completeness"]
        )

        if (
            stats["completeness"]
            >= MIN_COMPLETENESS
        ):
            passed += 1

        time.sleep(REQUEST_DELAY)

    result["passed_endpoint_probes"] = passed
    result["failed_requests"] = failures

    if failures > 0:

        result["fast_status"] = "inconclusive"

    elif passed == 2:

        result["fast_status"] = "endpoint_pass"

    else:

        # This is NOT a rejection.
        result["fast_status"] = "secondary_required"

    return result


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("OPENSENSEMAP FAST STAGED QUALIFICATION")
print("=" * 90)

df = pd.read_csv(
    INPUT,
    dtype={
        "box_id": str,
        "sensor_id": str,
    }
)

print("\nCandidate boxes:", df["box_id"].nunique())
print("Candidate sensors:", len(df))


# ============================================================
# LOAD CHECKPOINT
# ============================================================

box_results = []
sensor_results = []
completed_boxes = set()

if CHECKPOINT.exists():

    old = pd.read_csv(
        CHECKPOINT,
        dtype={"box_id": str}
    )

    box_results = old.to_dict("records")
    completed_boxes = set(old["box_id"])

    print(
        "\nCheckpoint found:",
        len(completed_boxes),
        "boxes"
    )

else:

    print("\nNo checkpoint found.")


if SENSOR_OUT.exists():

    old_sensors = pd.read_csv(
        SENSOR_OUT,
        dtype={
            "box_id": str,
            "sensor_id": str,
        }
    )

    sensor_results = (
        old_sensors.to_dict("records")
    )


# ============================================================
# PROCESS BOXES
# ============================================================

groups = list(
    df.groupby("box_id", sort=True)
)

try:

    for i, (box_id, group) in enumerate(
        groups,
        start=1
    ):

        if box_id in completed_boxes:
            continue

        print(
            f"\n[{i}/{len(groups)}] Box {box_id}"
        )

        selections = {}

        # ----------------------------------------------------
        # Process required variables sequentially.
        # ----------------------------------------------------

        for variable in REQUIRED:

            candidates = group[
                group["canonical_variable"]
                == variable
            ].copy()

            candidates = candidates.sort_values(
                "sensor_id"
            )

            if candidates.empty:

                selections[variable] = {
                    "status": "missing"
                }

                continue

            tested = []

            for _, row in candidates.iterrows():

                print(
                    f"  {variable:<18}"
                    f"{row['sensor_id']}"
                )

                r = test_sensor(row)

                sensor_results.append(r)
                tested.append(r)

                # Endpoint pass is enough for the
                # fast qualification stage.
                if r["fast_status"] == "endpoint_pass":
                    break

            endpoint = [
                x for x in tested
                if x["fast_status"]
                == "endpoint_pass"
            ]

            inconclusive = [
                x for x in tested
                if x["fast_status"]
                == "inconclusive"
            ]

            secondary = [
                x for x in tested
                if x["fast_status"]
                == "secondary_required"
            ]

            if endpoint:

                best = endpoint[0]

                selections[variable] = {
                    "status": "endpoint_pass",
                    "sensor_id":
                        best["sensor_id"],
                }

            elif inconclusive:

                selections[variable] = {
                    "status": "inconclusive"
                }

            elif secondary:

                # Keep deterministic first candidate.
                best = secondary[0]

                selections[variable] = {
                    "status": "secondary_required",
                    "sensor_id":
                        best["sensor_id"],
                }

            else:

                selections[variable] = {
                    "status": "secondary_required"
                }

        # ----------------------------------------------------
        # BOX STATUS
        # ----------------------------------------------------

        statuses = [
            selections[v]["status"]
            for v in REQUIRED
        ]

        if all(
            x == "endpoint_pass"
            for x in statuses
        ):

            box_status = "priority_candidate"

        elif "inconclusive" in statuses:

            box_status = "inconclusive"

        else:

            box_status = "secondary_required"

        record = {
            "box_id": box_id,
            "box_status": box_status,
        }

        for variable in REQUIRED:

            record[
                f"{variable}_status"
            ] = selections[
                variable
            ]["status"]

            record[
                f"{variable}_sensor_id"
            ] = selections[
                variable
            ].get("sensor_id")

        box_results.append(record)
        completed_boxes.add(box_id)

        # ----------------------------------------------------
        # SAVE AFTER EVERY BOX
        # ----------------------------------------------------

        pd.DataFrame(
            box_results
        ).to_csv(
            CHECKPOINT,
            index=False
        )

        pd.DataFrame(
            sensor_results
        ).to_csv(
            SENSOR_OUT,
            index=False
        )


except KeyboardInterrupt:

    print("\n\nInterrupted.")

    print(
        "Completed boxes are already checkpointed."
    )

    raise SystemExit(0)


# ============================================================
# FINAL OUTPUT
# ============================================================

boxes = pd.DataFrame(box_results)

priority = boxes[
    boxes["box_status"]
    == "priority_candidate"
].copy()

secondary = boxes[
    boxes["box_status"]
    != "priority_candidate"
].copy()

priority.to_csv(
    PRIORITY_OUT,
    index=False
)

secondary.to_csv(
    SECONDARY_OUT,
    index=False
)


print("\n" + "=" * 90)
print("FAST QUALIFICATION COMPLETE")
print("=" * 90)

print("\nStatus counts:")

print(
    boxes["box_status"]
    .value_counts(dropna=False)
)

print("\nPriority candidates:")
print(len(priority))

print("\nSecondary/inconclusive:")
print(len(secondary))

print("""
Generated:

  data/metadata/opensensemap_fast_qualification_checkpoint.csv
  data/metadata/opensensemap_fast_qualification_sensors.csv
  data/metadata/opensensemap_fast_priority_boxes.csv
  data/metadata/opensensemap_secondary_screen_boxes.csv

Interpretation:

priority_candidate
    All four required variables achieved >=70% hourly
    coverage in both January and September probe windows.

secondary_required
    Endpoint probes were insufficient. This is NOT rejection.

inconclusive
    At least one API request could not be verified.

Final eligibility will still be determined from exact
historical retrieval.

No forecasting result was used for station selection.
""")