from pathlib import Path
import json
import platform
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    raise SystemExit(
        "PyYAML is required.\n"
        "Install it with:\n"
        "pip install pyyaml"
    )


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_NAME = "Building-and-Environmental-Sensing"
ROOT = Path(PROJECT_NAME)

print("=" * 72)
print("STEP 1: FREEZE EXPERIMENTAL SPECIFICATION")
print("=" * 72)
print(f"Project directory: {ROOT.resolve()}")


# ============================================================
# 2. DIRECTORY STRUCTURE
# ============================================================

directories = [
    "configs",
    "data/raw/bdg2",
    "data/raw/opensensemap",
    "data/processed",
    "data/metadata",
    "src",
    "src/data",
    "src/models",
    "src/training",
    "src/adaptation",
    "src/evaluation",
    "src/statistics",
    "experiments",
    "experiments/baselines",
    "experiments/transfer",
    "experiments/ablations",
    "experiments/robustness",
    "experiments/uncertainty",
    "outputs",
    "outputs/runs",
    "outputs/predictions",
    "outputs/metrics",
    "outputs/tables",
    "outputs/figures",
    "scripts",
    "logs",
]

for directory in directories:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

print("Directory structure created.")


# ============================================================
# 3. MASTER EXPERIMENT CONFIGURATION
# ============================================================

experiment_config = {

    "experiment": {
        "name": "historical_to_live_building_environmental_forecasting",
        "version": "1.0",
        "description": (
            "Causal historical-to-live multimodal forecasting "
            "from BDG2 to openSenseMap."
        ),
        "seeds": list(range(1, 21)),
        "number_of_runs": 20,
    },

    "domains": {

        "source": {
            "dataset": "BDG2",
            "role": "historical",
            "temporal_access": "historical_only",
        },

        "target": {
            "dataset": "openSenseMap",
            "role": "live",
            "temporal_access": "causal_only",
        },
    },

    "variables": {

        "shared": [
            "temperature",
            "relative_humidity",
            "pressure",
        ],

        "source_private": [
            "energy",
        ],

        "target_private": [
            "pm25",
        ],

        "abbreviations": {
            "temperature": "T",
            "relative_humidity": "RH",
            "pressure": "P",
            "energy": "E",
            "pm25": "PM2.5",
        },
    },

    "forecasting": {

        "temporal_resolution": "1h",

        "horizons_hours": [
            1,
            3,
            6,
            12,
            24,
        ],

        "multi_horizon": True,
    },

    "evaluation": {

        "repeated_runs": 20,

        "report": {
            "mean": True,
            "standard_deviation": True,
        },

        "aggregate_error_space": "normalized",

        "metrics": [
            "MAE",
            "RMSE",
            "sMAPE",
            "R2",
            "NLL",
            "PICP",
        ],
    },

    "causality": {

        "enabled": True,

        "allow_future_target_observations": False,

        "chronological_target_processing": True,

        "target_label_requires_horizon_maturity": True,

        "target_normalization": {
            "future_statistics_allowed": False,
        },

        "target_imputation": {
            "future_observations_allowed": False,
        },

        "target_graph": {
            "future_observations_allowed": False,
        },

        "model_selection": {
            "target_test_allowed": False,
        },

        "adaptation": {
            "future_labels_allowed": False,
        },

        "calibration": {
            "future_target_allowed": False,
        },
    },

    "entity_correspondence": {

        "source_target_entity_matching": False,

        "source_graph": "independent",

        "target_graph": "independent",
    },

    "normalization": {

        "shared_variables": {
            "method": "zscore",
            "statistics_from": "source_training_only",
        },

        "source_private": {
            "energy": {
                "transform": "log1p",
                "normalization": "zscore",
                "statistics_from": "source_training_only",
            }
        },

        "target_private": {
            "pm25": {
                "transform": "log1p",
                "normalization": "causal_zscore",
                "future_statistics_allowed": False,
            }
        },

        "aggregate_metrics": {
            "MAE": "normalized_space",
            "RMSE": "normalized_space",
        },

        "physical_unit_metrics": {
            "aggregate_across_targets": False,
        },
    },

    "missing_data": {

        "preserve_availability_mask": True,

        "causal_imputation": True,

        "future_values_allowed": False,

        "short_gap_strategy": "interpolation",

        # Exact maximum gap will be determined from actual data.
        "short_gap_max_hours": None,

        "long_gap_strategy": "mask",

        "modality_masking": True,
    },

    "splits": {

        "strategy": "chronological",

        "shuffle_before_split": False,

        "source": {
            "train": "chronological",
            "validation": "chronological",
            "test": "chronological",
        },

        "target": {
            "deployment_stream": "chronological",
            "random_sampling": False,
        },

        "leakage": {
            "future_to_past": False,
            "target_test_for_model_selection": False,
        },
    },

    "target_supervision": {

        "history": [
            "0h",
            "6h",
            "12h",
            "24h",
            "3d",
            "7d",
        ],

        "fractions": [
            0.01,
            0.02,
            0.05,
            0.10,
            0.25,
            0.50,
        ],

        "fraction_selection": "chronological_prefix",

        "random_fraction_sampling": False,

        "require_label_maturity": True,
    },

    "target_private_learning": {

        "variable": "pm25",

        "activation": {
            "causal_only": True,
            "minimum_labels_required": True,
        },

        "stages": [
            "N_min_PM",
            "plus_24h",
            "plus_72h",
            "plus_7d",
            "continual",
        ],

        "initialization_controls": [
            "random_private",
            "source_private",
            "random_shared_alignment",
            "transferred_no_separation",
            "transferred_full_objective",
        ],

        "transfer": {
            "source_private_encoder": True,
            "source_energy_prediction_head": False,
            "target_pm25_prediction_head": "random_initialization",
        },
    },

    "shift_detection": {

        "statistic": "MMD",

        "representation_score": "delta_r_squared",

        "threshold": {
            "calibration_source": "unperturbed_validation_windows",
            "use_test_perturbations": False,
            "rule": "empirical_percentile",
            "percentile": 95,
            "fixed_value": None,
        },

        "perturbations": [
            "missing_modality",
            "timestamp_jitter",
            "sampling_frequency_shift",
            "station_dropout",
            "simultaneous_sensor_failure",
        ],

        "adaptation_requires": {
            "detected_shift": True,
            "matured_target_labels": True,
        },
    },

    "uncertainty": {

        "distribution": "heteroscedastic_gaussian",

        "prediction_interval": {
            "type": "gaussian_model_based",
            "nominal_coverage": 0.95,
            "multiplier": 1.96,
            "conformal": False,
            "distribution_free_guarantee": False,
        },

        "selective_forecasting": {
            "ranking_space": "normalized",
            "use_raw_physical_sigma": False,

            "coverage": [
                1.00,
                0.95,
                0.90,
                0.80,
                0.70,
                0.60,
                0.50,
            ],
        },
    },

    "statistics": {

        "paired_runs": True,

        "n": 20,

        "paired_test": "paired_t_test",

        "alternative": "two_sided",

        "assumption_check": {
            "normality": {
                "test": "shapiro_wilk",
                "applied_to": "paired_differences",
                "alpha": 0.05,
            }
        },

        "multiple_testing": {
            "correction": "holm",
            "scope": "within_hypothesis_family",
        },

        "effect_size": {
            "type": "cohens_dz",
            "formula": "mean_difference_over_sample_sd_difference",
        },

        "bootstrap": {
            "enabled": True,
            "resamples": 10000,
            "method": "percentile",
            "confidence_level": 0.95,
            "seed": 42,
            "resampling_unit": "paired_difference",
        },
    },
}


# ============================================================
# 4. SAVE MASTER CONFIG
# ============================================================

experiment_file = ROOT / "configs" / "experiment.yaml"

with open(experiment_file, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        experiment_config,
        f,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

print(f"Created: {experiment_file}")


# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

# Architecture-specific numerical settings are intentionally left
# unresolved until the actual data dimensionality has been audited.

model_config = {

    "model": {
        "name": "proposed_shared_private_forecaster",
    },

    "components": {

        "modality_specific_encoders": True,

        "temporal_encoder": True,

        "spatial_graph": True,

        "learned_dynamic_graph": True,

        "shared_private_decomposition": True,

        "domain_alignment": True,

        "uncertainty_head": True,
    },

    "outputs": {

        "shared": [
            "temperature",
            "relative_humidity",
            "pressure",
        ],

        "source_private": [
            "energy",
        ],

        "target_private": [
            "pm25",
        ],
    },

    "architecture": {

        "hidden_dimension": None,

        "temporal_layers": None,

        "graph_layers": None,

        "dropout": None,

        "attention_heads": None,
    },

    "losses": {

        "forecasting": True,

        "alignment": True,

        "separation": True,

        "replay": True,

        "uncertainty": True,

        "weights": {
            "forecasting": None,
            "alignment": None,
            "separation": None,
            "replay": None,
            "uncertainty": None,
        },
    },
}


model_file = ROOT / "configs" / "model.yaml"

with open(model_file, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        model_config,
        f,
        sort_keys=False,
        allow_unicode=True,
    )

print(f"Created: {model_file}")


# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================

training_config = {

    "training": {

        "optimizer": None,

        "learning_rate": None,

        "weight_decay": None,

        "batch_size": None,

        "max_epochs": None,

        "gradient_clipping": None,

        "seeded_runs": list(range(1, 21)),
    },

    "validation": {

        "metric": "normalized_MAE",

        "mode": "min",

        "test_set_access": False,
    },

    "early_stopping": {

        "enabled": True,

        "monitor": "validation_normalized_MAE",

        "patience": None,

        "minimum_delta": None,

        "restore_best_weights": True,
    },

    "checkpointing": {

        "enabled": True,

        "save_best_only": True,

        "separate_checkpoint_per_seed": True,
    },

    "reproducibility": {

        "deterministic_algorithms": True,

        "save_seed": True,

        "save_configuration": True,

        "save_predictions": True,

        "save_targets": True,

        "save_metrics": True,
    },
}


training_file = ROOT / "configs" / "training.yaml"

with open(training_file, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        training_config,
        f,
        sort_keys=False,
        allow_unicode=True,
    )

print(f"Created: {training_file}")


# ============================================================
# 7. ADAPTATION CONFIGURATION
# ============================================================

adaptation_config = {

    "adaptation": {

        "mode": "causal_online",

        "future_target_access": False,

        "chronological_processing": True,

        "require_matured_labels": True,

        "update_frequency": None,

        "learning_rate": None,

        "gradient_steps": None,
    },

    "target_history": [
        "0h",
        "6h",
        "12h",
        "24h",
        "3d",
        "7d",
    ],

    "target_data_fractions": [
        0.01,
        0.02,
        0.05,
        0.10,
        0.25,
        0.50,
    ],

    "fraction_strategy": "chronological_prefix",

    "target_only_controls": [
        "24h",
        "72h",
        "7d",
    ],

    "continual_learning": {

        "replay": True,

        "alignment_regularization": True,

        "ewc_control": True,
    },

    "private_target": {

        "variable": "pm25",

        "transfer_encoder_only": True,

        "transfer_energy_head": False,

        "new_prediction_head": True,
    },
}


adaptation_file = ROOT / "configs" / "adaptation.yaml"

with open(adaptation_file, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        adaptation_config,
        f,
        sort_keys=False,
        allow_unicode=True,
    )

print(f"Created: {adaptation_file}")


# ============================================================
# 8. SEED MANIFEST
# ============================================================

seed_manifest = {
    "number_of_runs": 20,
    "seeds": list(range(1, 21)),
    "bootstrap_seed": 42,
}

seed_file = ROOT / "configs" / "seeds.json"

with open(seed_file, "w", encoding="utf-8") as f:
    json.dump(seed_manifest, f, indent=4)

print(f"Created: {seed_file}")


# ============================================================
# 9. EXPERIMENT MANIFEST
# ============================================================

manifest = {

    "project": PROJECT_NAME,

    "created": datetime.now().isoformat(),

    "python": sys.version,

    "platform": platform.platform(),

    "source_dataset": "BDG2",

    "target_dataset": "openSenseMap",

    "runs": 20,

    "seeds": list(range(1, 21)),

    "forecast_horizons_hours": [
        1,
        3,
        6,
        12,
        24,
    ],

    "shared_targets": [
        "temperature",
        "relative_humidity",
        "pressure",
    ],

    "source_private_target": "energy",

    "target_private_target": "pm25",

    "aggregate_error_space": "normalized",

    "causal_target_processing": True,

    "future_target_access": False,

    "status": "EXPERIMENT_SPECIFICATION_FROZEN",
}


manifest_file = ROOT / "configs" / "experiment_manifest.json"

with open(manifest_file, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

print(f"Created: {manifest_file}")


# ============================================================
# 10. CREATE EMPTY PYTHON PACKAGES
# ============================================================

package_dirs = [
    "src",
    "src/data",
    "src/models",
    "src/training",
    "src/adaptation",
    "src/evaluation",
    "src/statistics",
]

for package in package_dirs:

    init_file = ROOT / package / "__init__.py"

    init_file.touch(exist_ok=True)


# ============================================================
# 11. GITKEEP FILES FOR EMPTY DATA/OUTPUT DIRECTORIES
# ============================================================

keep_dirs = [
    "data/raw/bdg2",
    "data/raw/opensensemap",
    "data/processed",
    "data/metadata",
    "outputs/runs",
    "outputs/predictions",
    "outputs/metrics",
    "outputs/tables",
    "outputs/figures",
]

for directory in keep_dirs:
    (ROOT / directory / ".gitkeep").touch(exist_ok=True)


# ============================================================
# 12. GITIGNORE
# ============================================================

gitignore = """
# Python
__pycache__/
*.py[cod]
*.pyd

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# Raw and processed data
data/raw/*
!data/raw/bdg2/
!data/raw/opensensemap/
!data/raw/bdg2/.gitkeep
!data/raw/opensensemap/.gitkeep

data/processed/*
!data/processed/.gitkeep

# Model checkpoints
*.pt
*.pth
*.ckpt

# Temporary files
*.tmp
*.log

# OS
.DS_Store
Thumbs.db
"""

with open(ROOT / ".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore.strip() + "\n")


# ============================================================
# 13. REQUIREMENTS
# ============================================================

requirements = """
numpy
pandas
scipy
statsmodels
scikit-learn
openpyxl
pyyaml
torch
matplotlib
networkx
"""

with open(ROOT / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements.strip() + "\n")


# ============================================================
# 14. CONFIGURATION VALIDATION SCRIPT
# ============================================================

validation_script = r'''
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
'''

validation_file = (
    ROOT
    / "scripts"
    / "validate_experiment_config.py"
)

with open(
    validation_file,
    "w",
    encoding="utf-8"
) as f:
    f.write(validation_script.strip() + "\n")

print(f"Created: {validation_file}")


# ============================================================
# 15. README FOR EXPERIMENTAL PROTOCOL
# ============================================================

readme = """# Building and Environmental Sensing

## Reproducible Experimental Pipeline

This repository implements the historical-to-live forecasting experiments
using BDG2 as the historical source domain and openSenseMap as the target
deployment domain.

### Core experimental principles

- 20 independent seeded runs: seeds 1-20.
- Chronological source and target processing.
- No future target observations are available during live deployment.
- T, RH and P are shared responses.
- Energy is source-private.
- PM2.5 is target-private.
- Forecast horizons are 1, 3, 6, 12 and 24 hours.
- Aggregate MAE and RMSE are computed in normalized target space.
- Physical-unit errors are not directly averaged across heterogeneous targets.
- Target supervision fractions are chronological prefixes.
- Target labels become available only after their forecast horizon matures.
- Shift thresholds are calibrated independently of test perturbations.
- Selective forecasting uses scale-normalized predictive uncertainty.
- Statistical comparisons use 20 seed-matched runs.

### Important

The manuscript tables must be generated from saved run-level experimental
outputs. Numerical values should not be manually inserted into the
experimental pipeline.
"""

with open(ROOT / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)


# ============================================================
# 16. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("STEP 1 COMPLETE")
print("=" * 72)

print(
    """
Created:

  configs/experiment.yaml
  configs/model.yaml
  configs/training.yaml
  configs/adaptation.yaml
  configs/seeds.json
  configs/experiment_manifest.json

  scripts/validate_experiment_config.py

  data/raw/
  data/processed/
  data/metadata/

  src/data/
  src/models/
  src/training/
  src/adaptation/
  src/evaluation/
  src/statistics/

  experiments/baselines/
  experiments/transfer/
  experiments/ablations/
  experiments/robustness/
  experiments/uncertainty/

  outputs/runs/
  outputs/predictions/
  outputs/metrics/
  outputs/tables/
  outputs/figures/

  README.md
  requirements.txt
  .gitignore
"""
)

print("NEXT:")
print()
print(
    f"cd {PROJECT_NAME}"
)

print(
    "python scripts/validate_experiment_config.py"
)