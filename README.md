# Building and Environmental Sensing: Historical-to-Live Transfer Forecasting

This repository contains the complete experimental pipeline for historical-to-live transfer learning in building and environmental sensing. The framework studies how representations learned from large-scale historical building data can be transferred and causally adapted to a live environmental sensing domain with an asymmetric response space.

The implementation covers data acquisition and auditing, source and target cohort construction, causal temporal partitioning, normalization, forecasting-window generation, source pretraining, target adaptation, leakage auditing, final target-test evaluation, baseline and ablation experiments, 20-run independent evaluation, and multi-seed statistical analysis.

## Overview

The experimental setting combines two sensing domains:

- **Historical/source domain:** Building Data Genome Project 2 (BDG2)
- **Live/target domain:** openSenseMap

The response space is asymmetric:

- **Shared variables:** temperature, relative humidity, pressure
- **Source-private variable:** energy
- **Target-private variable:** PM2.5

The forecasting protocol uses:

- **History length:** 24 hours
- **Forecast horizons:** 1, 3, 6, 12, and 24 hours
- **Independent runs:** 20
- **Seeds:** 42–61

The pipeline is designed around strict temporal causality. Future observations are not used as model inputs, normalization statistics are fitted only on permitted partitions, cross-partition forecasting windows are prohibited, and the target test partition is isolated from adaptation and model selection.

## Repository Structure

```text
Building-and-Environmental-Sensing/
│
├── configs/
│   ├── adaptation.yaml
│   ├── experiment.yaml
│   ├── experiment_manifest.json
│   ├── model.yaml
│   ├── seeds.json
│   └── training.yaml
│
├── scripts/
│   ├── 01_validate_experiment_config.py
│   ├── 02_download_bdg2.py
│   ├── ...
│   ├── 25_train_source_pretraining.py
│   ├── 26_train_target_adaptation.py
│   ├── 27_audit_target_adaptation.py
│   ├── 28_evaluate_target_test.py
│   ├── 29_train_baselines_ablations.py
│   ├── 30_evaluate_baselines_ablations_test.py
│   ├── 31_analyze_target_test_errors.py
│   ├── 32_run_20_independent_runs.py
│   ├── 33_aggregate_multiseed_statistics.py
│   └── 34_populate_all_result_tables.py
│
├── 00_setup_experiment.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated datasets, trained models, checkpoints, prediction files, logs, statistical outputs, and manuscript result tables are intentionally excluded from version control.

## Data Sources

### Building Data Genome Project 2

BDG2 provides the historical source-domain building measurements used for source representation learning. The pipeline includes scripts for downloading, inspecting, filtering, harmonizing, temporally partitioning, and normalizing the required source data.

### openSenseMap

openSenseMap provides the target-domain environmental sensing stream. The pipeline performs candidate discovery, historical availability screening, temporal feasibility checks, cohort freezing, data retrieval, harmonization, causal partition construction, and target-domain normalization.

The raw datasets are **not distributed in this repository**. The corresponding scripts retrieve and prepare the required public-source data.

## Installation

Clone the repository:

```bash
git clone https://github.com/imashoodnasir/Building-and-Environmental-Sensing.git
cd Building-and-Environmental-Sensing
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the frozen dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Experimental Setup

Initialize the project structure and experimental configuration with:

```bash
python 00_setup_experiment.py
```

Validate the configuration before running the data pipeline:

```bash
python scripts/01_validate_experiment_config.py
```

The configuration files under `configs/` define the experimental, model, training, adaptation, and seed settings used by the pipeline.

## Data Preparation Pipeline

The source-domain preparation begins with:

```bash
python scripts/02_download_bdg2.py
python scripts/03_inspect_bdg2_archive.py
python scripts/04_verify_bdg2_variables.py
```

The target-domain discovery and qualification stages are implemented in Steps 05–11, including openSenseMap auditing, candidate filtering, historical screening, temporal feasibility analysis, cohort freezing, downloading, and harmonization.

Causal target construction and source preparation continue through Steps 12–15.

The forecasting protocol, target temporal protocol, normalization parameters, and forecasting-window indices are subsequently frozen before any model training is performed.

## Causal Forecasting Protocol

The experiments enforce the following constraints:

```text
History length:                         24 h
Forecast horizons:                     1, 3, 6, 12, 24 h
Future observations in inputs:         prohibited
Future-based interpolation:            prohibited
Target test normalization:             prohibited
Target test adaptation:                prohibited
Target test model selection:           prohibited
Target test early stopping:            prohibited
Target test threshold selection:       prohibited
Cross-partition windows:               prohibited
```

These restrictions ensure that the target evaluation represents historical-to-live forecasting rather than retrospective access to future target information.

## Model-Ready Dataset Construction

After the temporal and normalization protocols have been frozen, run:

```bash
python scripts/18_freeze_forecasting_protocol.py
python scripts/19_build_bdg2_source_window_index.py
python scripts/20_build_opensensemap_target_window_index.py
python scripts/21_freeze_opensensemap_normalization.py
python scripts/22_build_model_ready_datasets.py
python scripts/23_build_pytorch_lazy_datasets.py
```

The transfer architecture is defined in:

```text
scripts/24_define_transfer_model.py
```

## Source Pretraining

Source-domain pretraining is performed independently for each random seed.

For example:

```bash
python scripts/25_train_source_pretraining.py --seed 42
```

During source pretraining, the shared encoder and source-specific components are optimized using BDG2 data. Target-specific components remain unused.

## Target Adaptation

After source pretraining:

```bash
python scripts/26_train_target_adaptation.py --seed 42
```

The target adaptation stage uses only the permitted target adaptation partition. Validation is used for checkpoint selection according to the frozen experimental protocol.

## Transfer Integrity Audit

Before accessing the target test partition, run:

```bash
python scripts/27_audit_target_adaptation.py --seed 42
```

The audit verifies, among other conditions:

- forecasting-contract consistency;
- checkpoint/history consistency;
- transferred-parameter integrity;
- frozen shared-representation integrity;
- temporal ordering of adaptation and validation;
- normalization boundaries;
- provenance of required artifacts;
- absence of target-test access during adaptation.

Target-test evaluation should proceed only after this audit passes.

## Final Target-Test Evaluation

The locked model is evaluated using:

```bash
python scripts/28_evaluate_target_test.py --seed 42
```

This stage performs inference only. It does not perform training, parameter updates, checkpoint selection, or test-based tuning.

## Baselines and Ablations

Validation-stage baseline and ablation training is performed with:

```bash
python scripts/29_train_baselines_ablations.py --seed 42
```

Locked configurations are subsequently evaluated on the target test partition:

```bash
python scripts/30_evaluate_baselines_ablations_test.py --seed 42
```

The implemented experimental variants include the persistence control and learned target/transfer ablations used in the study.

## Diagnostic Analysis

Prediction-level diagnostic analysis is performed using:

```bash
python scripts/31_analyze_target_test_errors.py --seed 42
```

Step 31 operates on saved predictions rather than retraining or retuning models. It produces aggregate, horizon-specific, spatial, and temporal diagnostic statistics.

## 20 Independent Runs

The final experimental protocol uses 20 independent runs corresponding to seeds:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
52, 53, 54, 55, 56, 57, 58, 59, 60, 61
```

The automated multi-run pipeline is available in:

```bash
python scripts/32_run_20_independent_runs.py
```

Each seed is treated as an independent experimental run.

## Multi-Seed Statistical Analysis

After all 20 runs have completed, aggregate the results with:

```bash
python scripts/33_aggregate_multiseed_statistics.py
```

The statistical analysis treats the **independent seed/run as the inferential unit**, rather than treating individual prediction rows as independent experimental replicates.

The analysis includes:

- multi-run descriptive statistics;
- Student-t confidence intervals;
- bootstrap confidence intervals;
- Friedman repeated-measures omnibus testing;
- Kendall's W;
- Shapiro-Wilk diagnostics;
- paired t-tests or Wilcoxon signed-rank tests as appropriate;
- Cohen's \(d_z\);
- small-sample corrected effect sizes;
- rank-biserial effect sizes;
- Holm correction for multiple comparisons;
- win/tie/loss stability;
- horizon-degradation analysis;
- convergence analysis;
- per-location spatial robustness;
- location-by-horizon robustness;
- daily robustness;
- hour-of-day robustness;
- day-of-week robustness.

The bootstrap analysis uses 10,000 repetitions.

## Experimental Outputs

Running the pipeline locally generates directories such as:

```text
data/
models/
logs/
final_result_tables/
```

These directories are intentionally excluded through `.gitignore`.

They may contain:

- harmonized source and target datasets;
- frozen temporal indices;
- normalization statistics;
- model checkpoints;
- training histories;
- target-test predictions;
- baseline and ablation results;
- per-location diagnostics;
- temporal diagnostics;
- multi-seed statistical results;
- generated manuscript tables.

This repository therefore distributes the **experimental code and configuration**, not generated experimental results or trained model artifacts.

## Reproducibility

The final experiments use 20 independent seeds (42–61). For each run, source pretraining, target adaptation, auditing, test evaluation, baseline/ablation evaluation, and diagnostic analysis are performed independently.

The repository preserves the configuration and execution logic required to reconstruct the experimental pipeline from the underlying public data sources.

For strict reproduction, users should retain the generated manifests and audit artifacts produced locally by each stage, as these record the provenance and integrity checks associated with individual runs.

## Important Evaluation Principle

The target test partition is treated as a final evaluation resource. It is not used for:

- normalization fitting;
- adaptation;
- gradient updates;
- early stopping;
- checkpoint selection;
- threshold selection;
- hyperparameter tuning.

This separation is central to the historical-to-live evaluation protocol.

## Requirements

The Python dependencies required by the experimental environment are specified in:

```text
requirements.txt
```

For GPU training, a compatible PyTorch/CUDA installation and NVIDIA GPU are recommended.

## Result-Table Generation

Manuscript-oriented result-table processing is implemented in:

```bash
python scripts/34_populate_all_result_tables.py
```

This stage operates on locally generated experimental outputs. The resulting workbooks are not included in this repository.

## Citation

If you use this code in your research, please cite the associated article.

```bibtex
@article{nasir_building_environmental_sensing,
  title   = {To be updated},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```

The citation information will be updated after publication.

## License

A software license has not yet been specified for this repository. Until a license is added, reuse and redistribution remain subject to the repository owner's copyright.

## Contact

**Inzamam Mashood Nasir**

For questions concerning the implementation, experimental protocol, or reproducibility of the study, please use the GitHub repository's issue tracker.
