from pathlib import Path
import json
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META_DIR = ROOT / "data" / "metadata"

BUILDING_AUDIT_FILE = (
    META_DIR / "bdg2_source_building_audit.csv"
)

SITE_AUDIT_FILE = (
    META_DIR / "bdg2_source_site_weather_audit.csv"
)

OUTPUT_COHORT_FILE = (
    META_DIR / "bdg2_frozen_source_cohort.csv"
)

EXCLUSION_FILE = (
    META_DIR / "bdg2_source_cohort_exclusions.csv"
)

SITE_QUALITY_FILE = (
    META_DIR / "bdg2_source_site_quality.csv"
)

MANIFEST_FILE = (
    META_DIR / "bdg2_frozen_source_cohort_manifest.json"
)

MIN_OBSERVED_PERCENT = 70.0

SHARED_VARIABLES = [
    "temperature",
    "relative_humidity",
    "pressure",
]


# =============================================================================
# LOAD
# =============================================================================

print("=" * 100)
print("FREEZE FINAL BDG2 SOURCE COHORT")
print("=" * 100)

building = pd.read_csv(BUILDING_AUDIT_FILE)
site = pd.read_csv(SITE_AUDIT_FILE)

building["building_id"] = building["building_id"].astype(str)
building["site_id"] = building["site_id"].astype(str)

site["site_id"] = site["site_id"].astype(str)

print("\nInput buildings:")
print(len(building))

print("\nInput sites:")
print(site["site_id"].nunique())


# =============================================================================
# VERIFY EXPECTED VARIABLES
# =============================================================================

available_variables = set(
    site["canonical_variable"].unique()
)

missing_variables = (
    set(SHARED_VARIABLES)
    - available_variables
)

if missing_variables:
    raise RuntimeError(
        "Missing shared variables from site audit: "
        f"{sorted(missing_variables)}"
    )


# =============================================================================
# SITE-LEVEL SHARED-VARIABLE QUALITY
# =============================================================================

print("\n" + "=" * 100)
print("SITE-LEVEL SHARED-VARIABLE QUALITY")
print("=" * 100)

site_shared = site[
    site["canonical_variable"].isin(
        SHARED_VARIABLES
    )
].copy()

site_shared["variable_pass"] = (
    site_shared["observed_percent"]
    >= MIN_OBSERVED_PERCENT
)

site_quality = (
    site_shared
    .pivot_table(
        index="site_id",
        columns="canonical_variable",
        values="observed_percent",
        aggfunc="first"
    )
    .reset_index()
)

# Make column names explicit.
for variable in SHARED_VARIABLES:
    if variable not in site_quality.columns:
        site_quality[variable] = pd.NA

site_quality = site_quality.rename(
    columns={
        "temperature":
            "temperature_observed_percent",

        "relative_humidity":
            "relative_humidity_observed_percent",

        "pressure":
            "pressure_observed_percent",
    }
)

site_quality["temperature_pass"] = (
    site_quality[
        "temperature_observed_percent"
    ]
    >= MIN_OBSERVED_PERCENT
)

site_quality["relative_humidity_pass"] = (
    site_quality[
        "relative_humidity_observed_percent"
    ]
    >= MIN_OBSERVED_PERCENT
)

site_quality["pressure_pass"] = (
    site_quality[
        "pressure_observed_percent"
    ]
    >= MIN_OBSERVED_PERCENT
)

site_quality["all_shared_variables_pass"] = (
    site_quality[
        [
            "temperature_pass",
            "relative_humidity_pass",
            "pressure_pass",
        ]
    ]
    .all(axis=1)
)

site_quality["minimum_shared_observed_percent"] = (
    site_quality[
        [
            "temperature_observed_percent",
            "relative_humidity_observed_percent",
            "pressure_observed_percent",
        ]
    ]
    .min(axis=1)
)

site_quality.to_csv(
    SITE_QUALITY_FILE,
    index=False
)

print(
    site_quality[
        [
            "site_id",
            "temperature_observed_percent",
            "relative_humidity_observed_percent",
            "pressure_observed_percent",
            "minimum_shared_observed_percent",
            "all_shared_variables_pass",
        ]
    ]
    .sort_values("site_id")
    .to_string(index=False)
)


# =============================================================================
# BUILDING-LEVEL ENERGY QUALITY
# =============================================================================

print("\n" + "=" * 100)
print("BUILDING-LEVEL ENERGY QUALITY")
print("=" * 100)

building["energy_pass"] = (
    building["energy_observed_percent"]
    >= MIN_OBSERVED_PERCENT
)

print(
    "Buildings passing energy criterion:",
    int(building["energy_pass"].sum())
)

print(
    "Buildings failing energy criterion:",
    int((~building["energy_pass"]).sum())
)


# =============================================================================
# MERGE SITE QUALITY INTO BUILDINGS
# =============================================================================

merged = building.merge(
    site_quality,
    on="site_id",
    how="left",
    validate="many_to_one"
)

if merged[
    "all_shared_variables_pass"
].isna().any():

    bad = merged.loc[
        merged[
            "all_shared_variables_pass"
        ].isna(),
        "site_id"
    ].unique()

    raise RuntimeError(
        "Buildings mapped to sites without "
        f"quality records: {bad}"
    )


# =============================================================================
# FINAL ELIGIBILITY
# =============================================================================

merged["source_eligible"] = (
    merged["energy_pass"]
    &
    merged["all_shared_variables_pass"]
)


# =============================================================================
# EXCLUSION REASONS
# =============================================================================

def exclusion_reason(row):

    reasons = []

    if not row["energy_pass"]:
        reasons.append(
            "energy_observed_below_70_percent"
        )

    if not row["temperature_pass"]:
        reasons.append(
            "temperature_observed_below_70_percent"
        )

    if not row["relative_humidity_pass"]:
        reasons.append(
            "relative_humidity_observed_below_70_percent"
        )

    if not row["pressure_pass"]:
        reasons.append(
            "pressure_observed_below_70_percent"
        )

    if not reasons:
        return "eligible"

    return ";".join(reasons)


merged["eligibility_reason"] = merged.apply(
    exclusion_reason,
    axis=1
)


# =============================================================================
# FREEZE FINAL COHORT
# =============================================================================

cohort = merged[
    merged["source_eligible"]
].copy()

cohort = (
    cohort
    .sort_values(
        [
            "site_id",
            "building_id",
        ]
    )
    .reset_index(drop=True)
)

cohort.insert(
    0,
    "source_order",
    range(
        1,
        len(cohort) + 1
    )
)

exclusions = merged[
    ~merged["source_eligible"]
].copy()

exclusions = (
    exclusions
    .sort_values(
        [
            "site_id",
            "building_id",
        ]
    )
    .reset_index(drop=True)
)

cohort.to_csv(
    OUTPUT_COHORT_FILE,
    index=False
)

exclusions.to_csv(
    EXCLUSION_FILE,
    index=False
)


# =============================================================================
# CONSISTENCY CHECKS
# =============================================================================

print("\n" + "=" * 100)
print("CONSISTENCY CHECKS")
print("=" * 100)

# No failing energy buildings may remain.
assert (
    cohort["energy_observed_percent"]
    >= MIN_OBSERVED_PERCENT
).all()

# No failing shared-variable site may remain.
assert (
    cohort["temperature_observed_percent"]
    >= MIN_OBSERVED_PERCENT
).all()

assert (
    cohort["relative_humidity_observed_percent"]
    >= MIN_OBSERVED_PERCENT
).all()

assert (
    cohort["pressure_observed_percent"]
    >= MIN_OBSERVED_PERCENT
).all()

# No duplicates.
assert cohort["building_id"].is_unique

# Ensure Lamb is absent if it failed pressure.
failed_pressure_sites = set(
    site_quality.loc[
        ~site_quality["pressure_pass"],
        "site_id"
    ]
)

retained_sites = set(
    cohort["site_id"]
)

assert not (
    failed_pressure_sites
    &
    retained_sites
), (
    "A site failing pressure quality "
    "remains in the source cohort."
)

print("PASSED")


# =============================================================================
# SUMMARY
# =============================================================================

original_buildings = len(building)

energy_pass_buildings = int(
    building["energy_pass"].sum()
)

final_buildings = len(cohort)

excluded_buildings = len(exclusions)

original_sites = int(
    building["site_id"].nunique()
)

final_sites = int(
    cohort["site_id"].nunique()
)

failed_sites = (
    site_quality.loc[
        ~site_quality[
            "all_shared_variables_pass"
        ],
        "site_id"
    ]
    .tolist()
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {

    "dataset":
        "Building Data Genome Project 2",

    "cohort_status":
        "frozen",

    "selection_unit":
        "building",

    "initial_electricity_buildings":
        original_buildings,

    "energy_quality_pass_buildings":
        energy_pass_buildings,

    "final_source_buildings":
        final_buildings,

    "excluded_buildings":
        excluded_buildings,

    "initial_sites":
        original_sites,

    "final_sites":
        final_sites,

    "excluded_shared_variable_sites":
        failed_sites,

    "minimum_observed_percent":
        MIN_OBSERVED_PERCENT,

    "required_shared_variables":
        SHARED_VARIABLES,

    "source_private_variable":
        "energy",

    "eligibility_definition": {

        "energy":
            (
                ">=70% observed building-level "
                "electricity before causal filling"
            ),

        "temperature":
            (
                ">=70% observed site-level "
                "temperature before causal filling"
            ),

        "relative_humidity":
            (
                ">=70% observed site-level "
                "relative humidity before causal filling"
            ),

        "pressure":
            (
                ">=70% observed site-level "
                "pressure before causal filling"
            ),
    },

    "future_values_used_for_eligibility":
        False,

    "imputed_values_used_for_eligibility":
        False,

    "forecasting_results_used":
        False,

    "target_domain_used":
        False,

    "normalization_used":
        False,

    "model_selection_used":
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


# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL BDG2 SOURCE COHORT")
print("=" * 100)

print("\nInitial electricity buildings:")
print(original_buildings)

print("\nBuildings passing >=70% observed energy:")
print(energy_pass_buildings)

print("\nFinal eligible source buildings:")
print(final_buildings)

print("\nTotal excluded buildings:")
print(excluded_buildings)

print("\nInitial sites:")
print(original_sites)

print("\nFinal retained sites:")
print(final_sites)

print("\nSites excluded for shared-variable quality:")
print(
    failed_sites
    if failed_sites
    else "None"
)

print("\nFINAL MINIMUM OBSERVED COMPLETENESS")

print(
    cohort[
        [
            "energy_observed_percent",
            "temperature_observed_percent",
            "relative_humidity_observed_percent",
            "pressure_observed_percent",
        ]
    ]
    .min()
    .to_string()
)

print("\nEXCLUSION REASONS")

print(
    exclusions[
        "eligibility_reason"
    ]
    .value_counts()
    .to_string()
)

print("\nBUILDINGS BY RETAINED SITE")

print(
    cohort.groupby(
        "site_id"
    )[
        "building_id"
    ]
    .nunique()
    .sort_values(
        ascending=False
    )
    .to_string()
)

print("\nGenerated:")

print(
    " data/metadata/"
    "bdg2_frozen_source_cohort.csv"
)

print(
    " data/metadata/"
    "bdg2_source_cohort_exclusions.csv"
)

print(
    " data/metadata/"
    "bdg2_source_site_quality.csv"
)

print(
    " data/metadata/"
    "bdg2_frozen_source_cohort_manifest.json"
)

print("""
IMPORTANT:

The BDG2 source cohort is now FROZEN.

Eligibility requires BOTH:

  1. >=70% observed building-level electricity
  2. >=70% observed site-level temperature,
     relative humidity, and pressure

All eligibility calculations use observations BEFORE
causal filling.

A building belonging to a site that fails any required
shared variable is excluded regardless of its electricity
quality.

No target-domain data, forecasting result, uncertainty
estimate, adaptation result, normalization statistic,
or model-selection result was used to construct the cohort.

The frozen building IDs must not be changed based on
downstream model performance.
""")