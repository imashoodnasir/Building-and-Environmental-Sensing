from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed" / "opensensemap"
FINAL_DIR = ROOT / "data" / "final" / "opensensemap"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

COHORT_FILE = META / "opensensemap_final_eligible_target_cohort.csv"
STREAM_FILE = META / "opensensemap_exact_stream_eligibility.csv"

LONG_FILE = FINAL_DIR / "opensensemap_target_hourly_long.csv"
WIDE_FILE = FINAL_DIR / "opensensemap_target_hourly_wide.csv"
AUDIT_FILE = META / "opensensemap_causal_imputation_audit.csv"
MANIFEST_FILE = META / "opensensemap_causal_target_manifest.json"

START = pd.Timestamp("2026-01-01T00:00:00Z")
END = pd.Timestamp("2026-09-30T23:00:00Z")

FULL_INDEX = pd.date_range(
    START,
    END,
    freq="h"
)

MAX_FFILL_HOURS = 3

VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
    "pm25",
]


# ============================================================
# LOAD FINAL COHORT
# ============================================================

print("=" * 92)
print("BUILD CAUSAL OPENSENSEMAP TARGET DATASET")
print("=" * 92)

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={"box_id": str}
)

streams = pd.read_csv(
    STREAM_FILE,
    dtype={
        "box_id": str,
        "sensor_id": str,
    }
)

eligible_ids = set(
    cohort["box_id"].astype(str)
)

streams = streams[
    streams["box_id"]
    .astype(str)
    .isin(eligible_ids)
].copy()

if len(cohort) != 17:
    print(
        f"\nWARNING: expected 17 eligible boxes, "
        f"found {len(cohort)}."
    )

expected_streams = len(cohort) * 4

if len(streams) != expected_streams:
    raise RuntimeError(
        f"Expected {expected_streams} eligible streams, "
        f"found {len(streams)}."
    )

print("\nFinal eligible boxes:", len(cohort))
print("Streams:", len(streams))
print("Hours per box:", len(FULL_INDEX))
print("Causal forward-fill limit:", MAX_FFILL_HOURS, "hours")


# ============================================================
# PROCESS EACH BOX / VARIABLE
# ============================================================

all_long = []
audit_rows = []

for _, box in cohort.sort_values(
    "selection_order"
).iterrows():

    box_id = str(box["box_id"])
    order = int(box["selection_order"])

    print("\n" + "-" * 92)
    print(
        f"Box {order:02d} | {box_id}"
    )
    print("-" * 92)

    for variable in VARIABLES:

        match = streams[
            (streams["box_id"] == box_id)
            &
            (
                streams["canonical_variable"]
                == variable
            )
        ]

        if len(match) != 1:
            raise RuntimeError(
                f"Expected one stream for "
                f"{box_id}/{variable}; found {len(match)}"
            )

        row = match.iloc[0]

        processed_file = (
            ROOT / Path(row["processed_file"])
        )

        if not processed_file.exists():
            raise FileNotFoundError(
                processed_file
            )

        df = pd.read_csv(
            processed_file
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce"
        )

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )

        df = df[
            df["timestamp"].notna()
        ].copy()

        # Safety: exactly one hourly value expected.
        duplicate_hours = int(
            df["timestamp"].duplicated().sum()
        )

        if duplicate_hours:
            raise RuntimeError(
                f"Duplicate hourly timestamps found "
                f"for {box_id}/{variable}"
            )

        # ----------------------------------------------------
        # FULL CAUSAL TIMELINE
        # ----------------------------------------------------

        series = (
            df.set_index("timestamp")["value"]
            .reindex(FULL_INDEX)
        )

        observed_mask = series.notna()

        n_observed = int(
            observed_mask.sum()
        )

        n_missing_before = int(
            series.isna().sum()
        )

        # ----------------------------------------------------
        # CAUSAL SHORT-GAP FILL
        #
        # limit=3 means only past information is propagated
        # forward for at most three consecutive missing hours.
        #
        # No future observation is accessed.
        # ----------------------------------------------------

        filled = series.ffill(
            limit=MAX_FFILL_HOURS
        )

        imputed_mask = (
            series.isna()
            &
            filled.notna()
        )

        remaining_missing_mask = (
            filled.isna()
        )

        n_imputed = int(
            imputed_mask.sum()
        )

        n_remaining_missing = int(
            remaining_missing_mask.sum()
        )

        # ----------------------------------------------------
        # AGE SINCE LAST REAL OBSERVATION
        #
        # This is useful for auditing that every imputed value
        # uses a sufficiently recent past measurement.
        # ----------------------------------------------------

        last_observed_time = pd.Series(
            pd.NaT,
            index=FULL_INDEX,
            dtype="datetime64[ns, UTC]"
        )

        last_observed_time.loc[
            observed_mask
        ] = FULL_INDEX[
            observed_mask
        ]

        last_observed_time = (
            last_observed_time.ffill()
        )

        timestamp_series = pd.Series(
            FULL_INDEX,
            index=FULL_INDEX
        )

        age_hours = (
            (
                timestamp_series
                - last_observed_time
            )
            .dt.total_seconds()
            / 3600.0
        )

        # Only meaningful for imputed entries.
        imputed_ages = age_hours[
            imputed_mask
        ]

        if len(imputed_ages):

            max_imputation_age = float(
                imputed_ages.max()
            )

            if (
                max_imputation_age
                > MAX_FFILL_HOURS
            ):
                raise RuntimeError(
                    "Causal fill exceeded configured limit."
                )

        else:
            max_imputation_age = 0.0

        # ----------------------------------------------------
        # OUTPUT LONG FORM
        # ----------------------------------------------------

        out = pd.DataFrame({
            "timestamp": FULL_INDEX,
            "selection_order": order,
            "box_id": box_id,
            "canonical_variable": variable,
            "value": filled.values,
            "observed":
                observed_mask.astype(int).values,
            "imputed":
                imputed_mask.astype(int).values,
            "missing":
                remaining_missing_mask.astype(int).values,
            "hours_since_last_observation":
                age_hours.values,
        })

        all_long.append(out)

        audit_rows.append({
            "selection_order": order,
            "box_id": box_id,
            "canonical_variable": variable,
            "total_hours": len(FULL_INDEX),
            "observed_hours": n_observed,
            "observed_percent":
                100.0 * n_observed / len(FULL_INDEX),
            "missing_before_fill":
                n_missing_before,
            "causally_imputed_hours":
                n_imputed,
            "causally_imputed_percent":
                100.0 * n_imputed / len(FULL_INDEX),
            "remaining_missing_hours":
                n_remaining_missing,
            "remaining_missing_percent":
                100.0
                * n_remaining_missing
                / len(FULL_INDEX),
            "max_imputation_age_hours":
                max_imputation_age,
            "imputation_method":
                "past-only forward fill",
            "forward_fill_limit_hours":
                MAX_FFILL_HOURS,
            "future_values_used":
                False,
        })

        print(
            f"  {variable:18s} "
            f"observed={n_observed:4d} "
            f"imputed={n_imputed:4d} "
            f"remaining={n_remaining_missing:4d}"
        )


# ============================================================
# COMBINE LONG DATA
# ============================================================

long_df = pd.concat(
    all_long,
    ignore_index=True
)

long_df = long_df.sort_values(
    [
        "selection_order",
        "timestamp",
        "canonical_variable"
    ]
)

long_df.to_csv(
    LONG_FILE,
    index=False
)


# ============================================================
# BUILD WIDE MODELING TABLE
# ============================================================

value_wide = long_df.pivot(
    index=[
        "selection_order",
        "box_id",
        "timestamp"
    ],
    columns="canonical_variable",
    values="value"
).reset_index()

observed_wide = long_df.pivot(
    index=[
        "selection_order",
        "box_id",
        "timestamp"
    ],
    columns="canonical_variable",
    values="observed"
).reset_index()

imputed_wide = long_df.pivot(
    index=[
        "selection_order",
        "box_id",
        "timestamp"
    ],
    columns="canonical_variable",
    values="imputed"
).reset_index()


# Rename masks
observed_wide = observed_wide.rename(
    columns={
        v: f"{v}_observed"
        for v in VARIABLES
    }
)

imputed_wide = imputed_wide.rename(
    columns={
        v: f"{v}_imputed"
        for v in VARIABLES
    }
)

wide = value_wide.merge(
    observed_wide,
    on=[
        "selection_order",
        "box_id",
        "timestamp"
    ],
    how="left"
)

wide = wide.merge(
    imputed_wide,
    on=[
        "selection_order",
        "box_id",
        "timestamp"
    ],
    how="left"
)


# ============================================================
# ADD CALENDAR FEATURES
#
# These are deterministic from current/past timestamp and do
# not leak future observations.
# ============================================================

wide["hour_of_day"] = (
    wide["timestamp"].dt.hour
)

wide["day_of_week"] = (
    wide["timestamp"].dt.dayofweek
)

wide["day_of_year"] = (
    wide["timestamp"].dt.dayofyear
)

wide["hour_sin"] = np.sin(
    2 * np.pi
    * wide["hour_of_day"]
    / 24.0
)

wide["hour_cos"] = np.cos(
    2 * np.pi
    * wide["hour_of_day"]
    / 24.0
)

wide["dow_sin"] = np.sin(
    2 * np.pi
    * wide["day_of_week"]
    / 7.0
)

wide["dow_cos"] = np.cos(
    2 * np.pi
    * wide["day_of_week"]
    / 7.0
)

wide["doy_sin"] = np.sin(
    2 * np.pi
    * wide["day_of_year"]
    / 365.25
)

wide["doy_cos"] = np.cos(
    2 * np.pi
    * wide["day_of_year"]
    / 365.25
)


# ============================================================
# SAVE
# ============================================================

wide = wide.sort_values(
    [
        "selection_order",
        "timestamp"
    ]
)

wide.to_csv(
    WIDE_FILE,
    index=False
)

audit = pd.DataFrame(
    audit_rows
).sort_values(
    [
        "selection_order",
        "canonical_variable"
    ]
)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "dataset": "openSenseMap",
    "final_target_boxes":
        int(len(cohort)),
    "period_start":
        START.isoformat(),
    "period_end":
        END.isoformat(),
    "hours_per_box":
        int(len(FULL_INDEX)),
    "variables":
        VARIABLES,
    "missing_value_policy": {
        "method":
            "past-only forward fill",
        "maximum_consecutive_hours":
            MAX_FFILL_HOURS,
        "future_observations_used":
            False,
        "longer_gaps":
            "retained as missing",
    },
    "observation_masks_preserved":
        True,
    "calendar_features":
        [
            "hour_of_day",
            "day_of_week",
            "day_of_year",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
        ],
    "normalization_applied":
        False,
    "train_validation_test_split_applied":
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
# FINAL AUDIT
# ============================================================

expected_rows = (
    len(cohort)
    * len(FULL_INDEX)
)

if len(wide) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows} wide rows, "
        f"found {len(wide)}."
    )

print("\n" + "=" * 92)
print("CAUSAL TARGET DATASET COMPLETE")
print("=" * 92)

print("\nBoxes:", len(cohort))
print("Hours per box:", len(FULL_INDEX))
print("Wide rows:", len(wide))
print("Long rows:", len(long_df))

print("\nIMPUTATION SUMMARY BY VARIABLE")

summary = (
    audit
    .groupby("canonical_variable")
    .agg(
        streams=("box_id", "count"),
        observed_percent_mean=(
            "observed_percent",
            "mean"
        ),
        imputed_percent_mean=(
            "causally_imputed_percent",
            "mean"
        ),
        remaining_missing_percent_mean=(
            "remaining_missing_percent",
            "mean"
        ),
        maximum_imputation_age=(
            "max_imputation_age_hours",
            "max"
        ),
    )
)

print(summary)

print("\nRemaining missing values in wide dataset:")

print(
    wide[
        VARIABLES
    ].isna().sum()
)

print("\nGenerated:")
print(
    " data/final/opensensemap/"
    "opensensemap_target_hourly_long.csv"
)
print(
    " data/final/opensensemap/"
    "opensensemap_target_hourly_wide.csv"
)
print(
    " data/metadata/"
    "opensensemap_causal_imputation_audit.csv"
)
print(
    " data/metadata/"
    "opensensemap_causal_target_manifest.json"
)

print("""
IMPORTANT:

Only previously observed values were used for short-gap filling.

No future target measurement was used to fill an earlier timestamp.

Missing runs longer than three hours remain explicitly missing.

Observed/imputed masks are retained for downstream sequence
construction and reproducibility.

No normalization or train/validation/test fitting has yet occurred.
""")