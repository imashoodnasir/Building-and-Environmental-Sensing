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

CHECKPOINT = META / "opensensemap_box_qualification_checkpoint.csv"
SENSOR_RESULTS = META / "opensensemap_box_qualification_sensors.csv"
BOX_RESULTS = META / "opensensemap_box_qualification_boxes.csv"
QUALIFIED = META / "opensensemap_qualified_boxes.csv"

API_BASE = "https://api.opensensemap.org"

REQUIRED = [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]

# Four 14-day windows spread across the frozen target period.
# Passing these is not the final completeness test.
PROBES = [
    ("jan", "2026-01-01T00:00:00Z", "2026-01-14T23:59:59Z"),
    ("apr", "2026-04-01T00:00:00Z", "2026-04-14T23:59:59Z"),
    ("jul", "2026-07-01T00:00:00Z", "2026-07-14T23:59:59Z"),
    ("sep", "2026-09-01T00:00:00Z", "2026-09-14T23:59:59Z"),
]

TIMEOUT = 120
MAX_RETRIES = 4
REQUEST_DELAY = 0.20

# At least three widely separated successful probe windows
# are considered strong enough to prioritize exact retrieval.
MIN_ACTIVE_PROBES = 3

# Within an active 14-day window, require observations in at
# least 70% of its 336 hourly bins.
MIN_PROBE_HOURLY_COMPLETENESS = 0.70


# ============================================================
# REQUEST
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
    "Building-Environmental-Sensing-Research/1.0"
})


def get_records(box_id, sensor_id, start, end):

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
                    "http_status": 200,
                    "error": None,
                }

            if r.status_code == 429:

                wait = min(60, 5 * attempt)

                print(
                    f"\n429 rate limit; waiting {wait}s"
                )

                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:

                wait = min(30, 2 ** attempt)

                print(
                    f"\nHTTP {r.status_code}; "
                    f"retry {attempt}/{MAX_RETRIES}, "
                    f"waiting {wait}s"
                )

                time.sleep(wait)
                continue

            return {
                "status": "http_error",
                "records": [],
                "elapsed": elapsed,
                "http_status": r.status_code,
                "error": r.text[:300],
            }

        except requests.RequestException as e:

            last_error = str(e)

            if attempt < MAX_RETRIES:

                wait = min(30, 2 ** attempt)

                print(
                    f"\nRequest exception; "
                    f"retry {attempt}/{MAX_RETRIES}, "
                    f"waiting {wait}s"
                )

                time.sleep(wait)

    return {
        "status": "request_failed",
        "records": [],
        "elapsed": None,
        "http_status": None,
        "error": last_error,
    }


# ============================================================
# PARSE OBSERVATIONS
# ============================================================

def parse_records(records, start, end):

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

    hourly_bins = (
        ts.dt.floor("h").nunique()
        if len(ts)
        else 0
    )

    expected_hours = int(
        (
            end_ts.floor("h")
            - start_ts.floor("h")
        ).total_seconds() / 3600
    ) + 1

    completeness = (
        hourly_bins / expected_hours
        if expected_hours > 0
        else 0.0
    )

    return {
        "n_records": len(ts),
        "hourly_bins": hourly_bins,
        "expected_hours": expected_hours,
        "hourly_completeness": completeness,
        "first_timestamp":
            ts.min().isoformat()
            if len(ts) else None,
        "last_timestamp":
            ts.max().isoformat()
            if len(ts) else None,
    }


# ============================================================
# TEST ONE SENSOR
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

    active_probes = 0
    successful_requests = 0
    failed_requests = 0

    active_times = []

    for label, start, end in PROBES:

        response = get_records(
            box_id,
            sensor_id,
            start,
            end
        )

        result[f"{label}_status"] = (
            response["status"]
        )

        result[f"{label}_http"] = (
            response["http_status"]
        )

        result[f"{label}_elapsed"] = (
            response["elapsed"]
        )

        if response["status"] != "ok":

            failed_requests += 1

            result[f"{label}_records"] = 0
            result[f"{label}_hourly_bins"] = 0
            result[f"{label}_completeness"] = None

            continue

        successful_requests += 1

        stats = parse_records(
            response["records"],
            start,
            end
        )

        result[f"{label}_records"] = (
            stats["n_records"]
        )

        result[f"{label}_hourly_bins"] = (
            stats["hourly_bins"]
        )

        result[f"{label}_completeness"] = (
            stats["hourly_completeness"]
        )

        # Guard against unnoticed 10,000-record truncation.
        result[f"{label}_limit_hit"] = (
            len(response["records"]) >= 10000
        )

        if (
            stats["hourly_completeness"]
            >= MIN_PROBE_HOURLY_COMPLETENESS
        ):

            active_probes += 1

            active_times.append(
                pd.Timestamp(start)
            )

        time.sleep(REQUEST_DELAY)

    result["successful_requests"] = (
        successful_requests
    )

    result["failed_requests"] = (
        failed_requests
    )

    result["active_probes"] = (
        active_probes
    )

    if len(active_times) >= 2:

        probe_span = (
            max(active_times)
            - min(active_times)
        ).days

    else:
        probe_span = 0

    result["active_probe_span_days"] = (
        probe_span
    )

    # Do not call failed API requests negative evidence.
    if failed_requests > 0:

        result["qualification"] = (
            "inconclusive_request_failure"
        )

    elif active_probes >= MIN_ACTIVE_PROBES:

        result["qualification"] = (
            "priority_candidate"
        )

    elif active_probes > 0:

        result["qualification"] = (
            "weak_candidate"
        )

    else:

        result["qualification"] = (
            "no_probe_evidence"
        )

    return result


# ============================================================
# LOAD INPUT
# ============================================================

print("=" * 90)
print("OPENSENSEMAP BOX-LEVEL QUALIFICATION")
print("=" * 90)

df = pd.read_csv(
    INPUT,
    dtype={
        "box_id": str,
        "sensor_id": str,
    }
)

print("\nBoxes:")
print(df["box_id"].nunique())

print("\nSensors:")
print(len(df))


# ============================================================
# CHECKPOINT
# ============================================================

if CHECKPOINT.exists():

    completed_df = pd.read_csv(
        CHECKPOINT,
        dtype={"box_id": str}
    )

    completed_boxes = set(
        completed_df["box_id"]
    )

    box_results = (
        completed_df.to_dict("records")
    )

    print(
        "\nCheckpoint found."
    )

    print(
        "Completed boxes:",
        len(completed_boxes)
    )

else:

    completed_boxes = set()
    box_results = []

    print(
        "\nNo checkpoint found."
    )


sensor_results = []

if SENSOR_RESULTS.exists():

    old_sensor_results = pd.read_csv(
        SENSOR_RESULTS,
        dtype={
            "box_id": str,
            "sensor_id": str,
        }
    )

    sensor_results = (
        old_sensor_results.to_dict("records")
    )


# ============================================================
# PROCESS BOX-BY-BOX
# ============================================================

groups = list(
    df.groupby(
        "box_id",
        sort=True
    )
)

try:

    for index, (box_id, group) in enumerate(
        groups,
        start=1
    ):

        if box_id in completed_boxes:
            continue

        print(
            f"\n[{index}/{len(groups)}] "
            f"Box {box_id}"
        )

        selected = []
        box_inconclusive = False

        for variable in REQUIRED:

            candidates = group[
                group["canonical_variable"]
                == variable
            ].copy()

            if candidates.empty:

                selected.append({
                    "variable": variable,
                    "status":
                        "missing_sensor"
                })

                continue

            # Deterministic order.
            candidates = candidates.sort_values(
                "sensor_id"
            )

            variable_results = []

            for _, row in candidates.iterrows():

                print(
                    f"  {variable:<18} "
                    f"{row['sensor_id']}"
                )

                result = test_sensor(row)

                sensor_results.append(
                    result
                )

                variable_results.append(
                    result
                )

                # Once a priority candidate is found,
                # alternatives do not need probing.
                if (
                    result["qualification"]
                    == "priority_candidate"
                ):
                    break

            priority = [
                x for x in variable_results
                if x["qualification"]
                == "priority_candidate"
            ]

            inconclusive = [
                x for x in variable_results
                if x["qualification"]
                == "inconclusive_request_failure"
            ]

            weak = [
                x for x in variable_results
                if x["qualification"]
                == "weak_candidate"
            ]

            if priority:

                best = sorted(
                    priority,
                    key=lambda x: (
                        x["active_probes"],
                        x["active_probe_span_days"]
                    ),
                    reverse=True
                )[0]

                selected.append({
                    "variable": variable,
                    "status":
                        "priority_candidate",
                    "sensor_id":
                        best["sensor_id"],
                    "active_probes":
                        best["active_probes"],
                    "probe_span_days":
                        best["active_probe_span_days"],
                })

            elif inconclusive:

                box_inconclusive = True

                selected.append({
                    "variable": variable,
                    "status":
                        "inconclusive_request_failure",
                })

            elif weak:

                best = sorted(
                    weak,
                    key=lambda x: (
                        x["active_probes"],
                        x["active_probe_span_days"]
                    ),
                    reverse=True
                )[0]

                selected.append({
                    "variable": variable,
                    "status":
                        "weak_candidate",
                    "sensor_id":
                        best["sensor_id"],
                    "active_probes":
                        best["active_probes"],
                    "probe_span_days":
                        best["active_probe_span_days"],
                })

            else:

                selected.append({
                    "variable": variable,
                    "status":
                        "no_probe_evidence"
                })

        statuses = {
            x["variable"]: x["status"]
            for x in selected
        }

        all_priority = all(
            statuses.get(v)
            == "priority_candidate"
            for v in REQUIRED
        )

        if box_inconclusive:

            box_status = (
                "inconclusive_request_failure"
            )

        elif all_priority:

            box_status = (
                "priority_candidate"
            )

        else:

            box_status = (
                "not_priority"
            )

        box_record = {
            "box_id": box_id,
            "box_status": box_status,
        }

        for x in selected:

            v = x["variable"]

            box_record[
                f"{v}_status"
            ] = x["status"]

            box_record[
                f"{v}_sensor_id"
            ] = x.get("sensor_id")

            box_record[
                f"{v}_active_probes"
            ] = x.get("active_probes")

            box_record[
                f"{v}_probe_span_days"
            ] = x.get("probe_span_days")

        box_results.append(
            box_record
        )

        completed_boxes.add(
            box_id
        )

        # Save after every completed box.
        pd.DataFrame(
            box_results
        ).to_csv(
            CHECKPOINT,
            index=False
        )

        pd.DataFrame(
            sensor_results
        ).to_csv(
            SENSOR_RESULTS,
            index=False
        )

except KeyboardInterrupt:

    print(
        "\n\nInterrupted by user."
    )

    print(
        "Checkpoint already contains "
        "all completed boxes."
    )

    raise SystemExit(0)


# ============================================================
# FINAL OUTPUT
# ============================================================

box_df = pd.DataFrame(
    box_results
)

box_df.to_csv(
    BOX_RESULTS,
    index=False
)

qualified = box_df[
    box_df["box_status"]
    == "priority_candidate"
].copy()

qualified.to_csv(
    QUALIFIED,
    index=False
)


print("\n" + "=" * 90)
print("QUALIFICATION COMPLETE")
print("=" * 90)

print("\nBox status:")

print(
    box_df[
        "box_status"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nPriority candidate boxes:",
    len(qualified)
)

print(
    "\nGenerated:"
)

print(
    "  data/metadata/"
    "opensensemap_box_qualification_checkpoint.csv"
)

print(
    "  data/metadata/"
    "opensensemap_box_qualification_sensors.csv"
)

print(
    "  data/metadata/"
    "opensensemap_box_qualification_boxes.csv"
)

print(
    "  data/metadata/"
    "opensensemap_qualified_boxes.csv"
)

print(
    """
IMPORTANT:

"priority_candidate" is NOT final dataset eligibility.

The next stage performs exact chunked retrieval over the
frozen target period and computes actual hourly:

  - temporal span
  - completeness
  - missingness
  - valid-value coverage

Only that stage can freeze the final target boxes.

No forecasting performance was used for selection.
"""
)