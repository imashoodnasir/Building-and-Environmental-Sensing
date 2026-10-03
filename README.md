# Building and Environmental Sensing

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
