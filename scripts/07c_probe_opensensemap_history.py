from pathlib import Path
import pandas as pd
import requests
import time


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

INPUT = META / "opensensemap_temporally_feasible_sensors.csv"

CHECKPOINT = META / "opensensemap_monthly_probe_checkpoint.csv"
OUTPUT = META / "opensensemap_monthly_probe_results.csv"
SURVIVORS = META / "opensensemap_monthly_probe_survivors.csv"
BOX_OUTPUT = META / "opensensemap_monthly_probe_boxes.csv"


# ============================================================
# FROZEN TARGET PERIOD
# ============================================================

START = pd.Timestamp("2026-01-01T00:00:00Z")
END = pd.Timestamp("2026-09-30T23:59:59Z")

MIN_SPAN_DAYS = 180

API_BASE = "https://api.opensensemap.org"

REQUEST_TIMEOUT = 60
REQUEST_DELAY = 0.10
MAX_RETRIES = 4

# Probe one 24-hour interval near the middle of each month.
PROBE_DAYS = [
    "2026-01-15",
    "2026-02-15",
    "2026-03-15",
    "2026-04-15",
    "2026-05-15",
    "2026-06-15",
    "2026-07-15",
    "2026-08-15",
    "2026-09-15",
]


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

                wait = min(60, attempt * 5)

                print(
                    f"\nRate limited. Waiting {wait}s..."
                )

                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:

                wait = min(30, 2 ** attempt)

                print(
                    f"\nServer error {r.status_code}. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)
                continue

            return r

        except requests.RequestException:

            if attempt == MAX_RETRIES:
                raise

            time.sleep(
                min(30, 2 ** attempt)
            )

    return None


def extract_timestamps(payload):

    if isinstance(payload, list):
        records = payload

    elif isinstance(payload, dict):

        if isinstance(payload.get("data"), list):
            records = payload["data"]

        elif isinstance(
            payload.get("measurements"),
            list
        ):
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

        return pd.DatetimeIndex(
            [],
            tz="UTC"
        )

    parsed = pd.to_datetime(
        timestamps,
        utc=True,
        errors="coerce"
    )

    parsed = parsed[
        ~pd.isna(parsed)
    ]

    return pd.DatetimeIndex(
        parsed
    )


# ============================================================
# PROBE ONE DAY
# ============================================================

def probe_day(
    session,
    box_id,
    sensor_id,
    day
):

    start = pd.Timestamp(
        f"{day}T00:00:00Z"
    )

    end = pd.Timestamp(
        f"{day}T23:59:59Z"
    )

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

    r = safe_request(
        session,
        url,
        params
    )

    if r is None:
        return "no_response", 0

    if r.status_code != 200:
        return (
            f"http_{r.status_code}",
            0
        )

    try:
        payload = r.json()

    except Exception:
        return "invalid_json", 0

    timestamps = extract_timestamps(
        payload
    )

    timestamps = timestamps[
        (timestamps >= start)
        & (timestamps <= end)
    ]

    return "ok", len(timestamps)


# ============================================================
# PROBE ONE SENSOR
# ============================================================

def probe_sensor(
    session,
    box_id,
    sensor_id
):

    result = {}

    active_dates = []

    request_failures = 0

    for day in PROBE_DAYS:

        status, n = probe_day(
            session,
            box_id,
            sensor_id,
            day
        )

        month = day[5:7]

        result[
            f"probe_{month}_status"
        ] = status

        result[
            f"probe_{month}_n"
        ] = n

        if status == "ok" and n > 0:

            active_dates.append(
                pd.Timestamp(
                    f"{day}T12:00:00Z"
                )
            )

        if status != "ok":
            request_failures += 1

        time.sleep(
            REQUEST_DELAY
        )

    # --------------------------------------------------------
    # MONTHLY PROBE SUMMARY
    # --------------------------------------------------------

    n_active_months = len(
        active_dates
    )

    if active_dates:

        first_probe = min(
            active_dates
        )

        last_probe = max(
            active_dates
        )

        observed_probe_span = (
            last_probe
            - first_probe
        ).days

    else:

        first_probe = None
        last_probe = None
        observed_probe_span = 0

    # IMPORTANT:
    #
    # A monthly probe cannot prove absence between probe days.
    # Therefore we do NOT reject merely because the observed
    # probe span is <180 days.
    #
    # The probe is used to prioritize likely active sensors.
    #
    # Only sensors with zero observations in all nine probes
    # are classified as "no_probe_evidence". They are NOT
    # permanently discarded here.

    if n_active_months == 0:

        probe_class = "no_probe_evidence"

    elif observed_probe_span >= MIN_SPAN_DAYS:

        probe_class = "strong_candidate"

    else:

        probe_class = "partial_candidate"

    return {
        **result,

        "n_active_probe_months":
            n_active_months,

        "first_active_probe":
            (
                first_probe.isoformat()
                if first_probe is not None
                else None
            ),

        "last_active_probe":
            (
                last_probe.isoformat()
                if last_probe is not None
                else None
            ),

        "probe_span_days":
            observed_probe_span,

        "request_failures":
            request_failures,

        "probe_class":
            probe_class,
    }


# ============================================================
# LOAD INPUT
# ============================================================

banner(
    "OPENSENSEMAP MONTHLY HISTORY PROBE"
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

print("\nCandidate boxes:")
print(
    df["box_id"].nunique()
)

print("\nCandidate sensors:")
print(len(df))

print("\nProbe dates:")

for d in PROBE_DAYS:
    print(" ", d)

print(
    "\nRequests per sensor:",
    len(PROBE_DAYS)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

results = []
completed = set()

if CHECKPOINT.exists():

    old = pd.read_csv(
        CHECKPOINT,
        dtype={
            "box_id": str,
            "sensor_id": str,
        }
    )

    results = old.to_dict(
        "records"
    )

    completed = set(
        zip(
            old["box_id"],
            old["sensor_id"]
        )
    )

    print(
        "\nCheckpoint found:",
        len(old),
        "completed sensors."
    )

else:

    print(
        "\nNo checkpoint found."
    )


# ============================================================
# RUN PROBES
# ============================================================

session = requests.Session()

processed = 0

try:

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        box_id = str(
            row["box_id"]
        )

        sensor_id = str(
            row["sensor_id"]
        )

        key = (
            box_id,
            sensor_id
        )

        if key in completed:
            continue

        variable = row[
            "canonical_variable"
        ]

        print(
            f"\r[{i}/{len(df)}] "
            f"{box_id} | "
            f"{variable:<18}",
            end="",
            flush=True
        )

        try:

            probe = probe_sensor(
                session,
                box_id,
                sensor_id
            )

        except Exception as e:

            probe = {
                "probe_class":
                    "request_exception",

                "request_failures":
                    len(PROBE_DAYS),

                "error":
                    str(e),
            }

        record = {
            "box_id":
                box_id,

            "box_name":
                row.get("box_name"),

            "sensor_id":
                sensor_id,

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

            **probe,
        }

        results.append(
            record
        )

        completed.add(
            key
        )

        processed += 1

        if processed % 10 == 0:

            pd.DataFrame(
                results
            ).to_csv(
                CHECKPOINT,
                index=False
            )

except KeyboardInterrupt:

    print(
        "\n\nInterrupted."
    )

    pd.DataFrame(
        results
    ).to_csv(
        CHECKPOINT,
        index=False
    )

    print(
        "Checkpoint saved."
    )

    raise SystemExit(0)


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    CHECKPOINT,
    index=False
)

result_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# PRIORITIZED SURVIVORS
# ============================================================
#
# Strong + partial candidates proceed first.
#
# "no_probe_evidence" is NOT permanently excluded.
# It is retained in the audit output for a later fallback
# check if necessary.
# ============================================================

survivors = result_df[
    result_df["probe_class"].isin(
        [
            "strong_candidate",
            "partial_candidate",
        ]
    )
].copy()

survivors.to_csv(
    SURVIVORS,
    index=False
)


# ============================================================
# BOX SUMMARY
# ============================================================

box_rows = []

for box_id, g in result_df.groupby(
    "box_id"
):

    strong_vars = set(
        g.loc[
            g["probe_class"]
            == "strong_candidate",
            "canonical_variable"
        ]
    )

    any_vars = set(
        g.loc[
            g["probe_class"].isin(
                [
                    "strong_candidate",
                    "partial_candidate",
                ]
            ),
            "canonical_variable"
        ]
    )

    box_rows.append(
        {
            "box_id":
                box_id,

            "n_sensors":
                len(g),

            "n_strong_sensors":
                int(
                    (
                        g["probe_class"]
                        == "strong_candidate"
                    ).sum()
                ),

            "n_partial_sensors":
                int(
                    (
                        g["probe_class"]
                        == "partial_candidate"
                    ).sum()
                ),

            "n_no_probe_sensors":
                int(
                    (
                        g["probe_class"]
                        == "no_probe_evidence"
                    ).sum()
                ),

            "strong_variables":
                len(strong_vars),

            "variables_with_any_probe":
                len(any_vars),

            "all_four_strong":
                set([
                    "temperature",
                    "relative_humidity",
                    "pressure",
                    "pm25"
                ]).issubset(
                    strong_vars
                ),

            "all_four_have_probe_evidence":
                set([
                    "temperature",
                    "relative_humidity",
                    "pressure",
                    "pm25"
                ]).issubset(
                    any_vars
                ),
        }
    )


box_df = pd.DataFrame(
    box_rows
)

box_df.to_csv(
    BOX_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

banner(
    "MONTHLY PROBE RESULTS"
)

print("\nSensor classifications:")

print(
    result_df[
        "probe_class"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nStrong candidate sensors:",
    (
        result_df["probe_class"]
        == "strong_candidate"
    ).sum()
)

print(
    "Partial candidate sensors:",
    (
        result_df["probe_class"]
        == "partial_candidate"
    ).sum()
)

print(
    "No-probe-evidence sensors:",
    (
        result_df["probe_class"]
        == "no_probe_evidence"
    ).sum()
)

print(
    "\nBoxes with all four variables "
    "showing probe evidence:",
    box_df[
        "all_four_have_probe_evidence"
    ].sum()
)

print(
    "Boxes with all four variables "
    "strongly spanning >=180 days:",
    box_df[
        "all_four_strong"
    ].sum()
)

print(
    "\nProbe activity by variable:"
)

print(
    result_df.groupby(
        [
            "canonical_variable",
            "probe_class"
        ]
    ).size()
)


banner(
    "MONTHLY PROBE COMPLETE"
)

print(
    """
Generated:

  data/metadata/opensensemap_monthly_probe_checkpoint.csv
  data/metadata/opensensemap_monthly_probe_results.csv
  data/metadata/opensensemap_monthly_probe_survivors.csv
  data/metadata/opensensemap_monthly_probe_boxes.csv

IMPORTANT:

This is a prioritization screen, not the final
eligibility test.

No sensor is declared complete from these probes.

Final eligibility still requires exact observation
retrieval and:

  span >= 180 days
  hourly completeness >= 70%

No forecasting results were used.
"""
)