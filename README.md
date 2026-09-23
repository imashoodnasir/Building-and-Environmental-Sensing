# Historical-to-Live Forecasting: Complete 12-Stage Repository

This repository implements the paper pipeline: BDG2 historical/source domain, openSenseMap live/target domain, shared T/RH/P responses, source-private energy, target-private PM2.5, modality-aware encoding, within-domain graphs, temporal attention, shared/private representation learning, heteroscedastic prediction, MMD shift detection, continual target adaptation, robustness evaluation, and final metrics.

## Guaranteed smoke-test path
The repository includes a synthetic-data generator so every stage can be executed without external downloads:

```bash
python -m pip install -r requirements.txt
python run_all.py --demo
```

The demo verifies software execution, not the paper's reported numerical results.

## Real data
Put BDG2 CSV exports in `data/raw/bdg2/` and openSenseMap CSV exports in `data/raw/opensensemap/`, then run:

```bash
python run_all.py
```

Stage 1 automatically recognizes common timestamp/entity/T/RH/P/energy/PM2.5/coordinate column names. If your raw export uses different names, add aliases in `src/stage01_prepare.py`.

## Stages
1. Raw-domain harmonization and hourly grids
2. Leakage-safe chronological splits and normalization
3. Causal multi-horizon windows and masks
4. Independent source/target spatial graphs
5. Historical source pretraining
6. Zero-shot target evaluation
7. Heteroscedastic uncertainty calibration
8. Shared-latent MMD calibration
9. Shift-responsive continual target adaptation
10. Robustness perturbations
11. Ablation configuration
12. Final probabilistic/point metrics and efficiency

## Important reproducibility note
The manuscript specifies 1 h resolution, lookback candidates {6,12,24,48,72}, horizons {1,3,6,12,24}, validation-only model selection, causal target-label availability, and seeds 1-20. Those are represented here. Some numerical hyperparameters are not specified in the supplied manuscript text; runnable defaults are therefore declared explicitly in `config.yaml` rather than presented as manuscript-derived values. Replace those defaults with the exact experimental values if different.

For publication experiments, increase `epochs`, run all 20 seeds, and execute each ablation as a separately trained configuration. The default 1 epoch are intentionally small so the repository can be smoke-tested quickly.
