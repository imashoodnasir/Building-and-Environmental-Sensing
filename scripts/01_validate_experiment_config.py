from pathlib import Path
import yaml
import json


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_yaml(
    ROOT / "configs" / "experiment.yaml"
)


errors = []


# ------------------------------------------------------------
# Seeds
# ------------------------------------------------------------

seeds = config["experiment"]["seeds"]

if seeds != list(range(1, 21)):
    errors.append(
        "Seeds must be exactly 1-20."
    )


if config["experiment"]["number_of_runs"] != 20:
    errors.append(
        "number_of_runs must equal 20."
    )


# ------------------------------------------------------------
# Forecast horizons
# ------------------------------------------------------------

expected_horizons = [1, 3, 6, 12, 24]

actual_horizons = (
    config["forecasting"]["horizons_hours"]
)

if actual_horizons != expected_horizons:
    errors.append(
        "Forecast horizons must be "
        "1, 3, 6, 12 and 24 hours."
    )


# ------------------------------------------------------------
# Shared variables
# ------------------------------------------------------------

expected_shared = {
    "temperature",
    "relative_humidity",
    "pressure",
}

actual_shared = set(
    config["variables"]["shared"]
)

if actual_shared != expected_shared:
    errors.append(
        "Shared-variable definition changed."
    )


# ------------------------------------------------------------
# Private variables
# ------------------------------------------------------------

if config["variables"]["source_private"] != ["energy"]:
    errors.append(
        "Source-private target must be energy."
    )


if config["variables"]["target_private"] != ["pm25"]:
    errors.append(
        "Target-private target must be PM2.5."
    )


# ------------------------------------------------------------
# Causality
# ------------------------------------------------------------

causal = config["causality"]

if causal["allow_future_target_observations"]:
    errors.append(
        "Future target observations are enabled."
    )


if not causal["chronological_target_processing"]:
    errors.append(
        "Target processing must be chronological."
    )


if causal["target_normalization"]["future_statistics_allowed"]:
    errors.append(
        "Future target normalization statistics are enabled."
    )


if causal["target_imputation"]["future_observations_allowed"]:
    errors.append(
        "Future observations are enabled for imputation."
    )


if causal["target_graph"]["future_observations_allowed"]:
    errors.append(
        "Future observations are enabled for target graph construction."
    )


if causal["model_selection"]["target_test_allowed"]:
    errors.append(
        "Target test data are enabled for model selection."
    )


if causal["adaptation"]["future_labels_allowed"]:
    errors.append(
        "Future labels are enabled for adaptation."
    )


if causal["calibration"]["future_target_allowed"]:
    errors.append(
        "Future target data are enabled for calibration."
    )


# ------------------------------------------------------------
# Target fractions
# ------------------------------------------------------------

supervision = config["target_supervision"]

if supervision["fraction_selection"] != "chronological_prefix":
    errors.append(
        "Target fractions must use chronological prefixes."
    )


if supervision["random_fraction_sampling"]:
    errors.append(
        "Random target fraction sampling must be disabled."
    )


# ------------------------------------------------------------
# Aggregate metrics
# ------------------------------------------------------------

normalization = config["normalization"]

if normalization["aggregate_metrics"]["MAE"] != "normalized_space":
    errors.append(
        "Aggregate MAE must be calculated in normalized space."
    )


if normalization["aggregate_metrics"]["RMSE"] != "normalized_space":
    errors.append(
        "Aggregate RMSE must be calculated in normalized space."
    )


if normalization["physical_unit_metrics"]["aggregate_across_targets"]:
    errors.append(
        "Physical-unit errors cannot be directly aggregated across targets."
    )


# ------------------------------------------------------------
# Selective forecasting
# ------------------------------------------------------------

selective = (
    config["uncertainty"]
    ["selective_forecasting"]
)

if selective["ranking_space"] != "normalized":
    errors.append(
        "Selective forecasting must rank normalized uncertainty."
    )


if selective["use_raw_physical_sigma"]:
    errors.append(
        "Raw physical-scale sigma cannot be used for cross-target ranking."
    )


# ------------------------------------------------------------
# Shift detection
# ------------------------------------------------------------

shift = config["shift_detection"]

if shift["threshold"]["use_test_perturbations"]:
    errors.append(
        "Test perturbations cannot be used to calibrate the shift threshold."
    )


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

stats = config["statistics"]

if stats["n"] != 20:
    errors.append(
        "Statistical analysis must use 20 matched runs."
    )


if stats["multiple_testing"]["correction"] != "holm":
    errors.append(
        "Holm correction expected."
    )


if stats["bootstrap"]["resamples"] != 10000:
    errors.append(
        "Bootstrap must use 10,000 resamples."
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 72)
print("EXPERIMENT CONFIGURATION VALIDATION")
print("=" * 72)

if errors:

    print("\nFAILED\n")

    for i, error in enumerate(errors, start=1):
        print(f"{i}. {error}")

    raise SystemExit(1)

else:

    print("\nPASSED")
    print()
    print("Seeds:                 1-20")
    print("Runs:                  20")
    print("Source:                BDG2")
    print("Target:                openSenseMap")
    print("Shared:                T, RH, P")
    print("Source-private:        Energy")
    print("Target-private:        PM2.5")
    print("Horizons:              1, 3, 6, 12, 24 h")
    print("Aggregate MAE/RMSE:    normalized space")
    print("Target fractions:      chronological")
    print("Future target access:  disabled")
    print("Bootstrap:             10,000")
    print("Bootstrap seed:        42")
    print()
    print("Experimental specification is valid.")
