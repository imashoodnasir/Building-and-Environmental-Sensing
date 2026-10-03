from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "metadata"
RAW = ROOT / "data" / "raw" / "opensensemap" / "frozen_history"
PROCESSED = ROOT / "data" / "processed" / "opensensemap"

PROCESSED.mkdir(parents=True, exist_ok=True)

COHORT_FILE = META / "opensensemap_frozen_target_candidates.csv"
CHECKPOINT_FILE = META / "opensensemap_frozen_history_download_checkpoint.csv"

STREAM_AUDIT_FILE = META / "opensensemap_exact_stream_eligibility.csv"
BOX_AUDIT_FILE = META / "opensensemap_exact_box_eligibility.csv"
FINAL_COHORT_FILE = META / "opensensemap_final_eligible_target_cohort.csv"
FILTER_AUDIT_FILE = META / "opensensemap_quality_filter_audit.csv"
MANIFEST_FILE = META / "opensensemap_harmonization_manifest.json"

START = pd.Timestamp("2026-01-01T00:00:00Z")
END = pd.Timestamp("2026-09-30T23:59:59Z")

EXPECTED_HOURS = len(
    pd.date_range(
        START.floor("h"),
        END.floor("h"),
        freq="h"
    )
)

MIN_SPAN_DAYS = 180.0
MIN_COMPLETENESS = 0.70


# ============================================================
# CANONICAL UNITS
# ============================================================

CANONICAL_UNITS = {
    "temperature": "degC",
    "relative_humidity": "%",
    "pressure": "hPa",
    "pm25": "ug/m3",
}


# ============================================================
# PHYSICAL PLAUSIBILITY RANGES
#
# These are broad plausibility/QC bounds, not climatological
# station-selection criteria.
# ============================================================

VALID_RANGES = {
    "temperature": (-50.0, 60.0),
    "relative_humidity": (0.0, 100.0),
    "pressure": (800.0, 1100.0),
    "pm25": (0.0, 1000.0),
}


# ============================================================
# UNIT CONVERSION
# ============================================================

def convert_to_canonical(values, variable, source_unit):

    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    unit = str(source_unit).strip()

    if variable == "temperature":

        if unit not in {"°C", "C°"}:
            raise ValueError(
                f"Unexpected temperature unit: {unit}"
            )

        return values.astype(float)

    if variable == "relative_humidity":

        if unit != "%":
            raise ValueError(
                f"Unexpected RH unit: {unit}"
            )

        return values.astype(float)

    if variable == "pressure":

        if unit != "Pa":
            raise ValueError(
                f"Unexpected pressure unit: {unit}"
            )

        # Pa -> hPa
        return values.astype(float) / 100.0

    if variable == "pm25":

        if unit != "µg/m³":
            raise ValueError(
                f"Unexpected PM2.5 unit: {unit}"
            )

        return values.astype(float)

    raise ValueError(
        f"Unknown variable: {variable}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 92)
print("OPENSENSEMAP EXACT HARMONIZATION + ELIGIBILITY")
print("=" * 92)

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={"box_id": str}
)

download = pd.read_csv(
    CHECKPOINT_FILE,
    dtype={
        "box_id": str,
        "sensor_id": str,
    }
)

if len(download) != 80:
    raise RuntimeError(
        f"Expected 80 downloaded streams, found {len(download)}."
    )

if not (download["status"] == "complete").all():
    raise RuntimeError(
        "Not all frozen streams are marked complete."
    )

print("\nFrozen boxes:", len(cohort))
print("Downloaded streams:", len(download))
print("Expected hourly bins:", EXPECTED_HOURS)
print("Minimum span:", MIN_SPAN_DAYS, "days")
print(
    "Minimum hourly completeness:",
    f"{MIN_COMPLETENESS * 100:.1f}%"
)


# ============================================================
# RECONSTRUCT SENSOR UNIT INVENTORY
# ============================================================

sensor_meta_file = (
    META /
    "opensensemap_temporally_feasible_sensors.csv"
)

sensor_meta = pd.read_csv(
    sensor_meta_file,
    dtype={
        "box_id": str,
        "sensor_id": str,
    }
)

sensor_meta = sensor_meta[
    [
        "box_id",
        "sensor_id",
        "sensor_title",
        "sensor_unit",
        "sensor_type",
    ]
].drop_duplicates(
    subset=[
        "box_id",
        "sensor_id"
    ]
)

download = download.merge(
    sensor_meta,
    on=[
        "box_id",
        "sensor_id"
    ],
    how="left",
    validate="many_to_one"
)

if download["sensor_unit"].isna().any():

    bad = download[
        download["sensor_unit"].isna()
    ]

    raise RuntimeError(
        "Missing sensor metadata for:\n"
        + bad[
            [
                "box_id",
                "sensor_id"
            ]
        ].to_string(index=False)
    )


# ============================================================
# PROCESS STREAMS
# ============================================================

stream_rows = []
filter_rows = []

for i, row in download.iterrows():

    box_id = str(row["box_id"])
    sensor_id = str(row["sensor_id"])
    variable = row["canonical_variable"]
    source_unit = row["sensor_unit"]

    order = int(row["selection_order"])

    relative_file = Path(row["file"])

    raw_file = ROOT / relative_file

    print(
        f"\n[{i + 1:02d}/{len(download)}] "
        f"Box {order:02d} | {variable}"
    )

    print("  Sensor:", sensor_id)
    print("  Source unit:", source_unit)

    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    df = pd.read_csv(raw_file)

    n_raw = len(df)

    # --------------------------------------------------------
    # TIMESTAMP VALIDATION
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    invalid_timestamp = int(
        df["timestamp"].isna().sum()
    )

    df = df[
        df["timestamp"].notna()
    ].copy()

    outside_period = int(
        (
            (df["timestamp"] < START)
            |
            (df["timestamp"] > END)
        ).sum()
    )

    df = df[
        (df["timestamp"] >= START)
        &
        (df["timestamp"] <= END)
    ].copy()

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    numeric_raw = pd.to_numeric(
        df["raw_value"],
        errors="coerce"
    )

    non_numeric = int(
        numeric_raw.isna().sum()
    )

    # --------------------------------------------------------
    # UNIT HARMONIZATION
    # --------------------------------------------------------

    df["value"] = convert_to_canonical(
        df["raw_value"],
        variable,
        source_unit
    )

    # --------------------------------------------------------
    # PHYSICAL RANGE FILTER
    # --------------------------------------------------------

    low, high = VALID_RANGES[variable]

    range_invalid_mask = (
        df["value"].notna()
        &
        (
            (df["value"] < low)
            |
            (df["value"] > high)
        )
    )

    n_range_invalid = int(
        range_invalid_mask.sum()
    )

    df.loc[
        range_invalid_mask,
        "value"
    ] = np.nan

    n_valid_raw = int(
        df["value"].notna().sum()
    )

    # --------------------------------------------------------
    # OBSERVATION SPAN
    # --------------------------------------------------------

    valid_obs = df[
        df["value"].notna()
    ].copy()

    if len(valid_obs):

        first_valid = valid_obs[
            "timestamp"
        ].min()

        last_valid = valid_obs[
            "timestamp"
        ].max()

        span_days = (
            last_valid - first_valid
        ).total_seconds() / 86400.0

    else:

        first_valid = pd.NaT
        last_valid = pd.NaT
        span_days = 0.0

    # --------------------------------------------------------
    # HOURLY AGGREGATION
    #
    # Mean of valid observations within each UTC hour.
    # No interpolation.
    # --------------------------------------------------------

    valid_obs["hour"] = (
        valid_obs["timestamp"]
        .dt.floor("h")
    )

    hourly = (
        valid_obs
        .groupby(
            "hour",
            as_index=False
        )["value"]
        .mean()
        .rename(
            columns={
                "hour": "timestamp"
            }
        )
    )

    # Restrict explicitly again.
    hourly = hourly[
        (hourly["timestamp"] >= START.floor("h"))
        &
        (hourly["timestamp"] <= END.floor("h"))
    ].copy()

    n_hourly = len(hourly)

    completeness = (
        n_hourly / EXPECTED_HOURS
        if EXPECTED_HOURS
        else 0.0
    )

    # --------------------------------------------------------
    # EXACT ELIGIBILITY
    # --------------------------------------------------------

    span_pass = (
        span_days >= MIN_SPAN_DAYS
    )

    completeness_pass = (
        completeness >= MIN_COMPLETENESS
    )

    eligible = (
        span_pass
        and completeness_pass
    )

    # --------------------------------------------------------
    # SAVE HARMONIZED HOURLY STREAM
    # --------------------------------------------------------

    hourly.insert(
        0,
        "canonical_variable",
        variable
    )

    hourly.insert(
        0,
        "sensor_id",
        sensor_id
    )

    hourly.insert(
        0,
        "box_id",
        box_id
    )

    hourly["unit"] = (
        CANONICAL_UNITS[variable]
    )

    output_dir = (
        PROCESSED /
        f"{order:02d}_{box_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        f"{variable}_hourly.csv"
    )

    hourly.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # AUDIT ROWS
    # --------------------------------------------------------

    stream_rows.append({
        "selection_order": order,
        "box_id": box_id,
        "sensor_id": sensor_id,
        "canonical_variable": variable,
        "sensor_title": row["sensor_title"],
        "sensor_type": row["sensor_type"],
        "source_unit": source_unit,
        "canonical_unit":
            CANONICAL_UNITS[variable],
        "raw_records": n_raw,
        "invalid_timestamp_records":
            invalid_timestamp,
        "outside_period_records":
            outside_period,
        "non_numeric_records":
            non_numeric,
        "physical_range_invalid_records":
            n_range_invalid,
        "valid_raw_records":
            n_valid_raw,
        "first_valid_timestamp":
            (
                first_valid.isoformat()
                if pd.notna(first_valid)
                else None
            ),
        "last_valid_timestamp":
            (
                last_valid.isoformat()
                if pd.notna(last_valid)
                else None
            ),
        "span_days":
            round(span_days, 6),
        "hourly_bins":
            n_hourly,
        "expected_hourly_bins":
            EXPECTED_HOURS,
        "hourly_completeness":
            completeness,
        "hourly_completeness_percent":
            completeness * 100.0,
        "span_pass":
            span_pass,
        "completeness_pass":
            completeness_pass,
        "stream_eligible":
            eligible,
        "hourly_aggregation":
            "arithmetic_mean",
        "interpolation_applied":
            False,
        "processed_file":
            str(
                output_file.relative_to(
                    ROOT
                )
            ),
    })

    filter_rows.append({
        "selection_order": order,
        "box_id": box_id,
        "sensor_id": sensor_id,
        "canonical_variable": variable,
        "raw_records": n_raw,
        "invalid_timestamp":
            invalid_timestamp,
        "outside_period":
            outside_period,
        "non_numeric":
            non_numeric,
        "physical_range_invalid":
            n_range_invalid,
        "valid_after_qc":
            n_valid_raw,
        "retention_percent":
            (
                100.0 * n_valid_raw / n_raw
                if n_raw
                else 0.0
            ),
    })

    print(
        f"  Raw records:       {n_raw:,}"
    )

    print(
        f"  Valid after QC:    {n_valid_raw:,}"
    )

    print(
        f"  Hourly bins:       {n_hourly:,}"
        f"/{EXPECTED_HOURS:,}"
    )

    print(
        f"  Completeness:      "
        f"{completeness * 100:.2f}%"
    )

    print(
        f"  Valid span:        "
        f"{span_days:.2f} days"
    )

    print(
        f"  Exact eligibility: "
        f"{'PASS' if eligible else 'FAIL'}"
    )


# ============================================================
# STREAM-LEVEL AUDIT
# ============================================================

stream_audit = pd.DataFrame(
    stream_rows
).sort_values(
    [
        "selection_order",
        "canonical_variable"
    ]
)

stream_audit.to_csv(
    STREAM_AUDIT_FILE,
    index=False
)

pd.DataFrame(
    filter_rows
).to_csv(
    FILTER_AUDIT_FILE,
    index=False
)


# ============================================================
# BOX-LEVEL ELIGIBILITY
# ============================================================

box_rows = []

required_variables = {
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
}

for order, group in stream_audit.groupby(
    "selection_order"
):

    box_id = str(
        group["box_id"].iloc[0]
    )

    observed_variables = set(
        group["canonical_variable"]
    )

    all_variables_present = (
        observed_variables
        == required_variables
    )

    all_streams_pass = bool(
        group["stream_eligible"].all()
    )

    box_eligible = (
        all_variables_present
        and all_streams_pass
    )

    row = {
        "selection_order":
            int(order),
        "box_id":
            box_id,
        "all_four_variables_present":
            all_variables_present,
        "all_streams_pass":
            all_streams_pass,
        "box_eligible":
            box_eligible,
        "minimum_completeness":
            group[
                "hourly_completeness"
            ].min(),
        "minimum_completeness_percent":
            group[
                "hourly_completeness_percent"
            ].min(),
        "minimum_span_days":
            group["span_days"].min(),
    }

    for variable in sorted(
        required_variables
    ):

        g = group[
            group["canonical_variable"]
            == variable
        ]

        if len(g):

            row[
                f"{variable}_completeness"
            ] = float(
                g[
                    "hourly_completeness"
                ].iloc[0]
            )

            row[
                f"{variable}_span_days"
            ] = float(
                g[
                    "span_days"
                ].iloc[0]
            )

            row[
                f"{variable}_eligible"
            ] = bool(
                g[
                    "stream_eligible"
                ].iloc[0]
            )

    box_rows.append(row)


box_audit = pd.DataFrame(
    box_rows
).sort_values(
    "selection_order"
)

box_audit.to_csv(
    BOX_AUDIT_FILE,
    index=False
)


# ============================================================
# FINAL ELIGIBLE COHORT
# ============================================================

eligible_ids = set(
    box_audit.loc[
        box_audit["box_eligible"],
        "box_id"
    ].astype(str)
)

final_cohort = cohort[
    cohort["box_id"]
    .astype(str)
    .isin(eligible_ids)
].copy()

final_cohort = final_cohort.merge(
    box_audit[
        [
            "box_id",
            "minimum_completeness",
            "minimum_completeness_percent",
            "minimum_span_days",
        ]
    ],
    on="box_id",
    how="left"
)

final_cohort.to_csv(
    FINAL_COHORT_FILE,
    index=False
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "dataset": "openSenseMap",
    "period_start":
        START.isoformat(),
    "period_end":
        END.isoformat(),
    "expected_hourly_bins":
        EXPECTED_HOURS,
    "hourly_aggregation":
        "arithmetic mean of valid observations within each UTC hour",
    "interpolation_before_eligibility":
        False,
    "minimum_observation_span_days":
        MIN_SPAN_DAYS,
    "minimum_hourly_completeness":
        MIN_COMPLETENESS,
    "physical_ranges": {
        k: {
            "minimum": v[0],
            "maximum": v[1],
            "canonical_unit":
                CANONICAL_UNITS[k],
        }
        for k, v in VALID_RANGES.items()
    },
    "unit_conversions": {
        "temperature":
            "°C/C° -> °C, numerical value unchanged",
        "relative_humidity":
            "% -> %, numerical value unchanged",
        "pressure":
            "Pa -> hPa by division by 100",
        "pm25":
            "µg/m³ -> µg/m³, numerical value unchanged",
    },
    "frozen_candidate_boxes":
        int(len(cohort)),
    "eligible_boxes":
        int(len(final_cohort)),
    "selection_uses_forecasting_results":
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


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 92)
print("EXACT ELIGIBILITY SUMMARY")
print("=" * 92)

print(
    "\nFrozen candidate boxes:",
    len(cohort)
)

print(
    "Exact eligible boxes:",
    len(final_cohort)
)

print(
    "Excluded boxes:",
    len(cohort) - len(final_cohort)
)

print("\nStream eligibility:")

print(
    stream_audit[
        "stream_eligible"
    ].value_counts(
        dropna=False
    )
)

print("\nBox eligibility:")

print(
    box_audit[
        "box_eligible"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nCompleteness by variable:"
)

summary = (
    stream_audit
    .groupby(
        "canonical_variable"
    )[
        "hourly_completeness_percent"
    ]
    .agg(
        [
            "count",
            "min",
            "median",
            "mean",
            "max"
        ]
    )
)

print(summary)

print("\nGenerated:")
print(
    " data/metadata/"
    "opensensemap_exact_stream_eligibility.csv"
)
print(
    " data/metadata/"
    "opensensemap_exact_box_eligibility.csv"
)
print(
    " data/metadata/"
    "opensensemap_final_eligible_target_cohort.csv"
)
print(
    " data/metadata/"
    "opensensemap_quality_filter_audit.csv"
)
print(
    " data/metadata/"
    "opensensemap_harmonization_manifest.json"
)

print("""
IMPORTANT:

Eligibility was determined BEFORE interpolation.

A target box passes only when ALL FOUR variables independently
satisfy:

    observation span >= 180 days
    hourly completeness >= 70%

Completeness is based on observed valid hourly bins after unit
harmonization and physical plausibility filtering.

No forecasting, adaptation, uncertainty, or downstream prediction
result was used to determine eligibility.
""")