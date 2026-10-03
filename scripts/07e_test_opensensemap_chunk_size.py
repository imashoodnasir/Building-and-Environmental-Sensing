import pandas as pd
import requests
import time

BOX_ID = "581dab879337ed000f6b308d"
SENSOR_ID = "581dab879337ed000f6b3093"

API_BASE = "https://api.opensensemap.org"
TIMEOUT = 90

TESTS = [
    ("1_day",  "2026-06-01T00:00:00Z", "2026-06-01T23:59:59Z"),
    ("3_days", "2026-06-01T00:00:00Z", "2026-06-03T23:59:59Z"),
    ("7_days", "2026-06-01T00:00:00Z", "2026-06-07T23:59:59Z"),
    ("14_days","2026-06-01T00:00:00Z", "2026-06-14T23:59:59Z"),
]

url = f"{API_BASE}/boxes/{BOX_ID}/data/{SENSOR_ID}"

print("=" * 90)
print("OPENSENSEMAP CHUNK-SIZE TEST")
print("=" * 90)

for name, start, end in TESTS:

    print(f"\nTesting: {name}")
    print(f"  {start} -> {end}")

    params = {
        "from-date": start,
        "to-date": end,
        "format": "json",
        "download": "false",
    }

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

        elapsed = time.time() - t0

        print("  HTTP:", r.status_code)
        print("  Time:", round(elapsed, 2), "s")

        if r.status_code != 200:
            print("  ERROR:", r.text[:300])
            continue

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

        print("  Records:", len(records))

        timestamps = []

        for item in records:
            if not isinstance(item, dict):
                continue

            ts = (
                item.get("createdAt")
                or item.get("timestamp")
                or item.get("time")
            )

            if ts:
                timestamps.append(ts)

        ts = pd.to_datetime(
            timestamps,
            utc=True,
            errors="coerce"
        )

        ts = pd.Series(ts).dropna()

        if len(ts):

            print("  First:", ts.min())
            print("  Last: ", ts.max())
            print(
                "  Hourly bins:",
                ts.dt.floor("h").nunique()
            )

        if len(records) == 10000:
            print("  >>> TRUNCATED / LIMIT REACHED")

        elif len(records) < 10000:
            print("  >>> BELOW 10,000 LIMIT")

    except Exception as e:
        print(
            "  REQUEST FAILED:",
            type(e).__name__,
            str(e)
        )

print("\n" + "=" * 90)
print("CHUNK-SIZE TEST COMPLETE")
print("=" * 90)