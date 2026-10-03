from pathlib import Path
import pandas as pd
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

SENSOR_FILE = META / "opensensemap_temporally_feasible_sensors.csv"

TEST_BOX = "581dab879337ed000f6b308d"

START = pd.Timestamp("2026-01-01T00:00:00Z")
END = pd.Timestamp("2026-09-30T23:59:59Z")

API_BASE = "https://api.opensensemap.org"

TIMEOUT = 90


def iso(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


print("=" * 90)
print("OPENSENSEMAP BULK RETRIEVAL CAPABILITY TEST")
print("=" * 90)

# ------------------------------------------------------------------
# Load sensors belonging to the known strong candidate box
# ------------------------------------------------------------------

df = pd.read_csv(
    SENSOR_FILE,
    dtype={
        "box_id": str,
        "sensor_id": str
    }
)

box = df[
    df["box_id"] == TEST_BOX
].copy()

if box.empty:
    raise RuntimeError(
        f"Test box {TEST_BOX} was not found."
    )

print("\nTest box:")
print(TEST_BOX)

print("\nSensors:")

print(
    box[
        [
            "canonical_variable",
            "sensor_id",
            "sensor_title",
            "sensor_unit"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------------
# We test only ONE sensor first.
# Temperature is sufficient to determine API behaviour.
# ------------------------------------------------------------------

temp = box[
    box["canonical_variable"] == "temperature"
]

if temp.empty:
    raise RuntimeError(
        "Temperature sensor not found in test box."
    )

row = temp.iloc[0]

sensor_id = str(
    row["sensor_id"]
)

print("\n" + "-" * 90)
print("TEST SENSOR")
print("-" * 90)

print("Variable:   temperature")
print("Sensor ID: ", sensor_id)

url = (
    f"{API_BASE}/boxes/"
    f"{TEST_BOX}/data/{sensor_id}"
)

params = {
    "from-date": iso(START),
    "to-date": iso(END),
    "format": "json",
    "download": "false",
}

print("\nRequested interval:")
print("Start:", START)
print("End:  ", END)

print("\nSending ONE full-period request...")

t0 = time.time()

try:

    r = requests.get(
        url,
        params=params,
        timeout=TIMEOUT,
        headers={
            "User-Agent":
            "Building-Environmental-Sensing-Research/1.0"
        }
    )

except Exception as e:

    print("\nREQUEST FAILED")
    print(type(e).__name__)
    print(str(e))
    raise SystemExit(1)

elapsed = time.time() - t0

print("\nHTTP status:")
print(r.status_code)

print("\nElapsed seconds:")
print(round(elapsed, 2))

print("\nResponse size:")
print(
    f"{len(r.content) / 1024 / 1024:.3f} MB"
)

if r.status_code != 200:

    print("\nResponse preview:")
    print(r.text[:2000])

    raise SystemExit(1)

# ------------------------------------------------------------------
# Parse
# ------------------------------------------------------------------

try:
    payload = r.json()

except Exception as e:

    print("\nJSON parsing failed:")
    print(e)

    print("\nResponse preview:")
    print(r.text[:2000])

    raise SystemExit(1)

if isinstance(payload, list):

    records = payload

elif isinstance(payload, dict):

    if isinstance(
        payload.get("data"),
        list
    ):
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

print("\nRecords returned:")
print(len(records))

# ------------------------------------------------------------------
# Extract timestamps
# ------------------------------------------------------------------

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

print("\nValid timestamps:")
print(len(ts))

if len(ts) > 0:

    print("\nEarliest returned:")
    print(ts.min())

    print("\nLatest returned:")
    print(ts.max())

    span = (
        ts.max() - ts.min()
    ).total_seconds() / 86400

    print("\nReturned temporal span:")
    print(
        f"{span:.2f} days"
    )

    print("\nUnique calendar days:")
    print(
        ts.dt.floor("D").nunique()
    )

    print("\nUnique hourly bins:")
    print(
        ts.dt.floor("h").nunique()
    )

    # --------------------------------------------------------------
    # Detect likely API truncation
    # --------------------------------------------------------------

    print("\n" + "-" * 90)
    print("TRUNCATION DIAGNOSTIC")
    print("-" * 90)

    if len(records) == 10000:

        print(
            "WARNING: Exactly 10,000 records were returned."
        )

        print(
            "This strongly suggests a per-request record limit."
        )

    elif len(records) > 10000:

        print(
            "More than 10,000 records were returned."
        )

        print(
            "A 10,000-record limit is therefore not active "
            "for this request."
        )

    else:

        print(
            "Fewer than 10,000 records were returned."
        )

        print(
            "This alone does not establish whether truncation "
            "exists."
        )

    requested_end = END

    last = ts.max()

    gap = (
        requested_end - last
    ).total_seconds() / 86400

    print("\nGap between final returned timestamp and requested end:")
    print(
        f"{gap:.2f} days"
    )

    if gap > 2:

        print(
            "\nThe response does NOT reach the requested end."
        )

        print(
            "Chunked retrieval will probably be required."
        )

    else:

        print(
            "\nThe response reaches approximately the "
            "requested end."
        )

else:

    print(
        "\nNo timestamps were returned."
    )

print("\n" + "=" * 90)
print("CAPABILITY TEST COMPLETE")
print("=" * 90)

print(
    "\nNo files were modified and no station-selection "
    "decision was made."
)