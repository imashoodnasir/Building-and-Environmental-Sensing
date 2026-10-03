from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

COHORT_FILE = META / "bdg2_frozen_source_cohort.csv"
AUDIT_FILE = META / "bdg2_temporal_partition_audit.csv"

OUTPUT_FILE = META / "bdg2_final_experimental_source_cohort.csv"
EXCLUSION_FILE = META / "bdg2_partition_energy_exclusions.csv"
MANIFEST_FILE = META / "bdg2_final_experimental_source_cohort_manifest.json"

MIN_ENERGY_OBSERVED_PERCENT = 70.0


print("=" * 100)
print("BDG2 PARTITION-LEVEL ENERGY ELIGIBILITY")
print("=" * 100)

cohort = pd.read_csv(
    COHORT_FILE,
    dtype={"building_id": str, "site_id": str}
)

audit = pd.read_csv(
    AUDIT_FILE,
    dtype={"building_id": str, "site_id": str}
)

required = {"train", "validation", "test"}

if set(audit["partition"].unique()) != required:
    raise RuntimeError("Unexpected temporal partitions.")


# =============================================================================
# CREATE ONE ROW PER BUILDING
# =============================================================================

energy = audit.pivot(
    index=["building_id", "site_id"],
    columns="partition",
    values="energy_observed_percent"
).reset_index()

energy.columns.name = None

energy = energy.rename(columns={
    "train": "train_energy_observed_percent",
    "validation": "validation_energy_observed_percent",
    "test": "test_energy_observed_percent",
})


# =============================================================================
# PASS/FAIL EACH PARTITION
# =============================================================================

energy["train_energy_pass"] = (
    energy["train_energy_observed_percent"]
    >= MIN_ENERGY_OBSERVED_PERCENT
)

energy["validation_energy_pass"] = (
    energy["validation_energy_observed_percent"]
    >= MIN_ENERGY_OBSERVED_PERCENT
)

energy["test_energy_pass"] = (
    energy["test_energy_observed_percent"]
    >= MIN_ENERGY_OBSERVED_PERCENT
)

energy["all_partitions_pass"] = (
    energy[
        [
            "train_energy_pass",
            "validation_energy_pass",
            "test_energy_pass",
        ]
    ].all(axis=1)
)

energy["minimum_partition_energy_observed_percent"] = (
    energy[
        [
            "train_energy_observed_percent",
            "validation_energy_observed_percent",
            "test_energy_observed_percent",
        ]
    ].min(axis=1)
)


# =============================================================================
# EXCLUSION REASON
# =============================================================================

def reason(row):

    failed = []

    if not row["train_energy_pass"]:
        failed.append("train_energy_below_70")

    if not row["validation_energy_pass"]:
        failed.append("validation_energy_below_70")

    if not row["test_energy_pass"]:
        failed.append("test_energy_below_70")

    if not failed:
        return "eligible"

    return ";".join(failed)


energy["partition_eligibility_reason"] = energy.apply(
    reason,
    axis=1
)


# =============================================================================
# MERGE WITH PREVIOUSLY FROZEN COHORT
# =============================================================================

final = cohort.merge(
    energy,
    on=["building_id", "site_id"],
    how="left",
    validate="one_to_one"
)

if final["all_partitions_pass"].isna().any():
    raise RuntimeError(
        "Some frozen buildings are missing partition audit information."
    )


# =============================================================================
# FINAL EXPERIMENTAL COHORT
# =============================================================================

eligible = final[
    final["all_partitions_pass"]
].copy()

excluded = final[
    ~final["all_partitions_pass"]
].copy()

eligible = (
    eligible
    .sort_values(["site_id", "building_id"])
    .reset_index(drop=True)
)

excluded = (
    excluded
    .sort_values(["site_id", "building_id"])
    .reset_index(drop=True)
)

eligible.insert(
    0,
    "experimental_source_order",
    range(1, len(eligible) + 1)
)


# =============================================================================
# CONSISTENCY CHECKS
# =============================================================================

assert (
    eligible["train_energy_observed_percent"]
    >= MIN_ENERGY_OBSERVED_PERCENT
).all()

assert (
    eligible["validation_energy_observed_percent"]
    >= MIN_ENERGY_OBSERVED_PERCENT
).all()

assert (
    eligible["test_energy_observed_percent"]
    >= MIN_ENERGY_OBSERVED_PERCENT
).all()

assert eligible["building_id"].is_unique


# =============================================================================
# SAVE
# =============================================================================

eligible.to_csv(
    OUTPUT_FILE,
    index=False
)

excluded.to_csv(
    EXCLUSION_FILE,
    index=False
)


# =============================================================================
# SUMMARY
# =============================================================================

print("\nOriginal frozen buildings:")
print(len(cohort))

print("\nPASS COUNTS")

for p in ["train", "validation", "test"]:

    c = f"{p}_energy_pass"

    print(
        f"{p:12s}: "
        f"{int(final[c].sum())} / {len(final)}"
    )


print("\nFinal buildings passing ALL three partitions:")
print(len(eligible))

print("\nExcluded:")
print(len(excluded))

print("\nFinal sites:")
print(eligible["site_id"].nunique())

print("\nMINIMUM ENERGY COMPLETENESS IN FINAL COHORT")

print(
    eligible[
        [
            "train_energy_observed_percent",
            "validation_energy_observed_percent",
            "test_energy_observed_percent",
        ]
    ].min().to_string()
)

print("\nEXCLUSION REASONS")

if len(excluded):

    print(
        excluded[
            "partition_eligibility_reason"
        ]
        .value_counts()
        .to_string()
    )

else:
    print("None")


print("\nBUILDINGS BY SITE")

print(
    eligible.groupby("site_id")["building_id"]
    .nunique()
    .sort_values(ascending=False)
    .to_string()
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {

    "status":
        "frozen",

    "input_source_buildings":
        int(len(cohort)),

    "final_experimental_source_buildings":
        int(len(eligible)),

    "excluded_buildings":
        int(len(excluded)),

    "final_sites":
        int(eligible["site_id"].nunique()),

    "partition_energy_requirement_percent":
        MIN_ENERGY_OBSERVED_PERCENT,

    "criterion":
        (
            "Each building must have >=70% observed "
            "energy independently in train, validation, "
            "and test partitions."
        ),

    "criterion_applied_before_model_training":
        True,

    "imputed_values_used_for_eligibility":
        False,

    "forecasting_performance_used":
        False,

    "normalization_statistics_used":
        False,

    "model_selection_used":
        False,

    "target_domain_used":
        False,

    "future_target_domain_observations_used":
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


print("\n" + "=" * 100)
print("FINAL EXPERIMENTAL SOURCE COHORT FROZEN")
print("=" * 100)

print("\nGenerated:")
print(
    " data/metadata/"
    "bdg2_final_experimental_source_cohort.csv"
)

print(
    " data/metadata/"
    "bdg2_partition_energy_exclusions.csv"
)

print(
    " data/metadata/"
    "bdg2_final_experimental_source_cohort_manifest.json"
)

print("""
IMPORTANT:

This quality screen uses only the availability of observed
ground-truth energy measurements.

No forecast errors, predictions, uncertainty values,
hyperparameters, or model outputs were examined.

A retained building has >=70% observed energy independently
within training, validation, and test periods.

Normalization has still NOT been performed.

The next stage computes all source normalization parameters
using TRAINING observations only.
""")