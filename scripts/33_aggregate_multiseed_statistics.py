"""
STEP 33 - DENSE MULTI-SEED STATISTICAL ANALYSIS
================================================

Purpose
-------
Aggregate the completed 20 independent experimental runs (seeds 42-61)
and perform dense run-level statistical analysis.

IMPORTANT:
    * No model training.
    * No model inference.
    * No checkpoint selection.
    * No test-based tuning.
    * No test-set model selection.
    * The independent experimental RUN/SEED is the inferential unit.
    * Prediction rows are NOT treated as independent statistical samples.

Primary models
--------------
M0, M1, M2, M3, M4, M5

Proposed method
---------------
M2

Planned comparisons
-------------------
M2 vs M0
M2 vs M1
M2 vs M3
M2 vs M4
M2 vs M5

Primary metrics
---------------
RMSE
MAE
R2

Secondary descriptive metrics
-----------------------------
Median absolute error
Bias
Error standard deviation
P75 absolute error
P90 absolute error
P95 absolute error
P99 absolute error
Maximum absolute error

Inferential analyses
--------------------
1. Descriptive statistics over 20 independent runs.
2. Student-t 95% CI for means.
3. Bootstrap 95% CI for means.
4. Shapiro-Wilk normality of paired run-level differences.
5. Paired t-test when paired differences are approximately normal.
6. Wilcoxon signed-rank test otherwise.
7. Cohen's dz.
8. Small-sample corrected paired standardized effect.
9. Matched-pairs rank-biserial correlation.
10. Holm multiplicity correction.
11. Friedman repeated-measures omnibus tests.
12. Kendall's W omnibus effect size.
13. Absolute and relative improvement distributions.
14. Win/tie/loss consistency over seeds.
15. Horizon degradation analysis.
16. Source-pretraining convergence stability.
17. Target-adaptation convergence stability.
18. Baseline/ablation convergence stability.
19. Per-box spatial robustness.
20. Per-box x horizon robustness.
21. Daily temporal robustness.
22. Hour-of-day robustness.
23. Day-of-week robustness.

Outputs
-------
data/metadata/multiseed_statistics/
"""

from pathlib import Path
import argparse
import json
import math
import warnings

import numpy as np
import pandas as pd

from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "metadata"

OUTPUT_DIR = META / "multiseed_statistics"

EXPECTED_SEEDS = list(range(42, 62))

VARIANTS = [
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
]

PROPOSED = "M2"

COMPARATORS = [
    "M0",
    "M1",
    "M3",
    "M4",
    "M5",
]

HORIZONS = [
    1,
    3,
    6,
    12,
    24,
]

PRIMARY_METRICS = [
    "rmse_pm25",
    "mae_pm25",
    "r2",
]

SECONDARY_METRICS = [
    "median_ae_pm25",
    "bias_pm25",
    "error_sd_pm25",
    "p75_ae_pm25",
    "p90_ae_pm25",
    "p95_ae_pm25",
    "p99_ae_pm25",
    "max_ae_pm25",
]

ALL_PERFORMANCE_METRICS = PRIMARY_METRICS + SECONDARY_METRICS

ALPHA = 0.05

DEFAULT_BOOTSTRAP_REPS = 10000

BOOTSTRAP_RANDOM_SEED = 330042

EPS = 1e-12


# =============================================================================
# DISPLAY
# =============================================================================

def banner(text):

    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


# =============================================================================
# FILE HELPERS
# =============================================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required Step-33 input does not exist:\n{path}"
        )


def load_seed_csv(
    seed,
    family,
    filename,
):

    path = (
        META
        / family
        / f"seed_{seed}"
        / filename
    )

    require_file(path)

    df = pd.read_csv(path)

    df.insert(
        0,
        "seed_run",
        seed,
    )

    return df


# =============================================================================
# NUMERIC HELPERS
# =============================================================================

def finite_values(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return values[
        np.isfinite(values)
    ]


def student_t_ci(
    values,
    alpha=ALPHA,
):

    x = finite_values(values)

    n = len(x)

    if n < 2:

        return (
            np.nan,
            np.nan,
        )

    mean = np.mean(x)

    se = stats.sem(x)

    critical = stats.t.ppf(
        1.0 - alpha / 2.0,
        n - 1,
    )

    return (
        float(mean - critical * se),
        float(mean + critical * se),
    )


def bootstrap_mean_ci(
    values,
    repetitions,
    random_seed,
    alpha=ALPHA,
):

    x = finite_values(values)

    n = len(x)

    if n < 2:

        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        random_seed
    )

    bootstrap_means = np.empty(
        repetitions,
        dtype=np.float64,
    )

    # Chunking avoids unnecessarily large temporary arrays.
    position = 0

    chunk_size = 2000

    while position < repetitions:

        current = min(
            chunk_size,
            repetitions - position,
        )

        indices = rng.integers(
            low=0,
            high=n,
            size=(
                current,
                n,
            ),
        )

        bootstrap_means[
            position:
            position + current
        ] = x[
            indices
        ].mean(
            axis=1
        )

        position += current

    lower = np.quantile(
        bootstrap_means,
        alpha / 2.0,
    )

    upper = np.quantile(
        bootstrap_means,
        1.0 - alpha / 2.0,
    )

    return (
        float(lower),
        float(upper),
    )


def descriptive_statistics(
    values,
    bootstrap_repetitions,
    bootstrap_seed,
):

    x = finite_values(values)

    n = len(x)

    if n == 0:

        return {}

    mean = float(
        np.mean(x)
    )

    median = float(
        np.median(x)
    )

    minimum = float(
        np.min(x)
    )

    maximum = float(
        np.max(x)
    )

    q1 = float(
        np.quantile(
            x,
            0.25,
        )
    )

    q3 = float(
        np.quantile(
            x,
            0.75,
        )
    )

    if n > 1:

        sd = float(
            np.std(
                x,
                ddof=1,
            )
        )

        sem = float(
            sd / np.sqrt(n)
        )

    else:

        sd = np.nan

        sem = np.nan

    if (
        n > 1
        and abs(mean) > EPS
    ):

        cv_percent = float(
            abs(sd / mean) * 100.0
        )

    else:

        cv_percent = np.nan

    t_low, t_high = student_t_ci(
        x
    )

    boot_low, boot_high = bootstrap_mean_ci(
        x,
        repetitions=bootstrap_repetitions,
        random_seed=bootstrap_seed,
    )

    if n >= 3:

        skewness = float(
            stats.skew(
                x,
                bias=False,
            )
        )

    else:

        skewness = np.nan

    if n >= 4:

        excess_kurtosis = float(
            stats.kurtosis(
                x,
                fisher=True,
                bias=False,
            )
        )

    else:

        excess_kurtosis = np.nan

    return {

        "n_runs":
            int(n),

        "mean":
            mean,

        "sd":
            sd,

        "sem":
            sem,

        "median":
            median,

        "q1":
            q1,

        "q3":
            q3,

        "iqr":
            float(q3 - q1),

        "min":
            minimum,

        "max":
            maximum,

        "range":
            float(
                maximum - minimum
            ),

        "cv_percent":
            cv_percent,

        "mean_ci95_low_t":
            t_low,

        "mean_ci95_high_t":
            t_high,

        "mean_ci95_low_bootstrap":
            boot_low,

        "mean_ci95_high_bootstrap":
            boot_high,

        "skewness":
            skewness,

        "excess_kurtosis":
            excess_kurtosis,
    }


# =============================================================================
# NORMALITY
# =============================================================================

def shapiro_test(values):

    x = finite_values(values)

    if len(x) < 3:

        return (
            np.nan,
            np.nan,
        )

    if np.allclose(
        x,
        x[0],
    ):

        return (
            np.nan,
            np.nan,
        )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        result = stats.shapiro(
            x
        )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


# =============================================================================
# EFFECT SIZE
# =============================================================================

def matched_rank_biserial(
    improvement,
):

    x = finite_values(
        improvement
    )

    x = x[
        np.abs(x) > EPS
    ]

    if len(x) == 0:

        return 0.0

    ranks = stats.rankdata(
        np.abs(x)
    )

    positive = np.sum(
        ranks[
            x > 0
        ]
    )

    negative = np.sum(
        ranks[
            x < 0
        ]
    )

    denominator = (
        positive
        + negative
    )

    if denominator == 0:

        return 0.0

    return float(
        (
            positive
            - negative
        )
        / denominator
    )


# =============================================================================
# HOLM CORRECTION
# =============================================================================

def holm_adjust(
    p_values,
):

    p = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(p)
    )[0]

    if len(valid_indices) == 0:

        return adjusted

    ordered = valid_indices[
        np.argsort(
            p[
                valid_indices
            ]
        )
    ]

    m = len(ordered)

    previous = 0.0

    for rank, original_index in enumerate(
        ordered
    ):

        multiplier = (
            m - rank
        )

        current = min(
            1.0,
            multiplier
            * p[
                original_index
            ],
        )

        current = max(
            previous,
            current,
        )

        adjusted[
            original_index
        ] = current

        previous = current

    return adjusted


# =============================================================================
# PAIRED STATISTICS
# =============================================================================

def paired_analysis(
    m2_values,
    comparator_values,
    metric,
    bootstrap_repetitions,
    bootstrap_seed,
):

    m2 = np.asarray(
        m2_values,
        dtype=float,
    )

    comparator = np.asarray(
        comparator_values,
        dtype=float,
    )

    valid = (
        np.isfinite(m2)
        &
        np.isfinite(comparator)
    )

    m2 = m2[
        valid
    ]

    comparator = comparator[
        valid
    ]

    n = len(m2)

    if metric == "r2":

        # Higher R2 is better.
        improvement = (
            m2
            - comparator
        )

        relative_improvement = np.where(
            np.abs(comparator) > EPS,
            (
                improvement
                / np.abs(comparator)
            )
            * 100.0,
            np.nan,
        )

    else:

        # Lower error is better.
        improvement = (
            comparator
            - m2
        )

        relative_improvement = np.where(
            np.abs(comparator) > EPS,
            (
                improvement
                / np.abs(comparator)
            )
            * 100.0,
            np.nan,
        )

    shapiro_w, shapiro_p = shapiro_test(
        improvement
    )

    normal_difference = bool(
        np.isfinite(shapiro_p)
        and shapiro_p >= ALPHA
    )

    if n < 2:

        test_name = "insufficient"

        statistic = np.nan

        p_value = np.nan

    elif normal_difference:

        result = stats.ttest_rel(
            m2,
            comparator,
        )

        test_name = (
            "paired_t_test"
        )

        statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

    else:

        try:

            result = stats.wilcoxon(
                improvement,
                alternative="two-sided",
                zero_method="wilcox",
            )

            test_name = (
                "wilcoxon_signed_rank"
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        except ValueError:

            # All differences may be zero.
            test_name = (
                "wilcoxon_all_zero"
            )

            statistic = 0.0

            p_value = 1.0

    if n > 1:

        difference_sd = float(
            np.std(
                improvement,
                ddof=1,
            )
        )

    else:

        difference_sd = np.nan

    if (
        n > 1
        and difference_sd > EPS
    ):

        cohens_dz = float(
            np.mean(
                improvement
            )
            / difference_sd
        )

    else:

        cohens_dz = np.nan

    # Small-sample correction for paired standardized difference.
    if (
        n > 1
        and np.isfinite(
            cohens_dz
        )
    ):

        correction = (
            1.0
            - 3.0
            / (
                4.0 * n
                - 5.0
            )
        )

        hedges_gz = float(
            cohens_dz
            * correction
        )

    else:

        hedges_gz = np.nan

    rank_biserial = (
        matched_rank_biserial(
            improvement
        )
    )

    t_low, t_high = student_t_ci(
        improvement
    )

    boot_low, boot_high = (
        bootstrap_mean_ci(
            improvement,
            repetitions=bootstrap_repetitions,
            random_seed=bootstrap_seed,
        )
    )

    wins = int(
        np.sum(
            improvement > EPS
        )
    )

    ties = int(
        np.sum(
            np.abs(
                improvement
            )
            <= EPS
        )
    )

    losses = int(
        np.sum(
            improvement < -EPS
        )
    )

    return {

        "n_runs":
            int(n),

        "m2_mean":
            float(
                np.mean(m2)
            ),

        "comparator_mean":
            float(
                np.mean(
                    comparator
                )
            ),

        "mean_improvement":
            float(
                np.mean(
                    improvement
                )
            ),

        "sd_improvement":
            float(
                np.std(
                    improvement,
                    ddof=1,
                )
            )
            if n > 1
            else np.nan,

        "median_improvement":
            float(
                np.median(
                    improvement
                )
            ),

        "mean_relative_improvement_percent":
            float(
                np.nanmean(
                    relative_improvement
                )
            ),

        "median_relative_improvement_percent":
            float(
                np.nanmedian(
                    relative_improvement
                )
            ),

        "improvement_ci95_low_t":
            t_low,

        "improvement_ci95_high_t":
            t_high,

        "improvement_ci95_low_bootstrap":
            boot_low,

        "improvement_ci95_high_bootstrap":
            boot_high,

        "shapiro_w_difference":
            shapiro_w,

        "shapiro_p_difference":
            shapiro_p,

        "difference_normal_at_0_05":
            normal_difference,

        "selected_test":
            test_name,

        "test_statistic":
            statistic,

        "p_raw":
            p_value,

        "cohens_dz":
            cohens_dz,

        "hedges_gz":
            hedges_gz,

        "rank_biserial":
            rank_biserial,

        "m2_wins":
            wins,

        "ties":
            ties,

        "m2_losses":
            losses,

        "m2_win_fraction":
            float(
                wins / n
            )
            if n
            else np.nan,

        "direction_consistent_80_percent":
            bool(
                (
                    wins / n
                )
                >= 0.80
            )
            if n
            else False,
    }


# =============================================================================
# FRIEDMAN OMNIBUS
# =============================================================================

def friedman_analysis(
    frame,
    metric,
):

    pivot = frame.pivot(
        index="seed_run",
        columns="variant",
        values=metric,
    )

    pivot = pivot.reindex(
        columns=VARIANTS
    )

    pivot = pivot.dropna()

    n = len(pivot)

    k = len(VARIANTS)

    if n < 2:

        return None

    result = stats.friedmanchisquare(
        *[
            pivot[
                variant
            ].values
            for variant
            in VARIANTS
        ]
    )

    kendall_w = float(
        result.statistic
        / (
            n
            * (
                k - 1
            )
        )
    )

    return {

        "metric":
            metric,

        "n_runs":
            int(n),

        "n_variants":
            int(k),

        "friedman_chi_square":
            float(
                result.statistic
            ),

        "df":
            int(
                k - 1
            ),

        "p_raw":
            float(
                result.pvalue
            ),

        "kendalls_w":
            kendall_w,
    }


# =============================================================================
# CORE COMPLETENESS AUDIT
# =============================================================================

def validate_core_results(
    overall,
    horizon,
):

    banner(
        "STEP 33A - COMPLETENESS / DESIGN AUDIT"
    )

    seeds = sorted(
        overall[
            "seed_run"
        ].unique().tolist()
    )

    if seeds != EXPECTED_SEEDS:

        raise RuntimeError(
            "Overall seed set mismatch.\n"
            f"Expected: {EXPECTED_SEEDS}\n"
            f"Found:    {seeds}"
        )

    for seed in EXPECTED_SEEDS:

        current = overall[
            overall[
                "seed_run"
            ]
            == seed
        ]

        found_variants = sorted(
            current[
                "variant"
            ].unique().tolist()
        )

        if found_variants != VARIANTS:

            raise RuntimeError(
                f"Seed {seed}: overall variant completeness failed.\n"
                f"Found: {found_variants}"
            )

        if len(current) != 6:

            raise RuntimeError(
                f"Seed {seed}: expected 6 overall rows, "
                f"found {len(current)}."
            )

        current_horizon = horizon[
            horizon[
                "seed_run"
            ]
            == seed
        ]

        if len(
            current_horizon
        ) != 30:

            raise RuntimeError(
                f"Seed {seed}: expected 30 horizon rows, "
                f"found {len(current_horizon)}."
            )

        for variant in VARIANTS:

            found_horizons = sorted(
                current_horizon[
                    current_horizon[
                        "variant"
                    ]
                    == variant
                ][
                    "horizon_hours"
                ]
                .astype(int)
                .tolist()
            )

            if (
                found_horizons
                != HORIZONS
            ):

                raise RuntimeError(
                    f"Seed {seed}, {variant}: "
                    f"horizon completeness failed.\n"
                    f"Found: {found_horizons}"
                )

    overall_n_check = (
        overall
        .groupby(
            "seed_run"
        )[
            "n"
        ]
        .nunique()
    )

    if not (
        overall_n_check
        == 1
    ).all():

        raise RuntimeError(
            "Common-subset overall n differs "
            "between variants within one or more seeds."
        )

    horizon_n_check = (
        horizon
        .groupby(
            [
                "seed_run",
                "horizon_hours",
            ]
        )[
            "n"
        ]
        .nunique()
    )

    if not (
        horizon_n_check
        == 1
    ).all():

        raise RuntimeError(
            "Common-subset horizon n differs "
            "between variants within one or more seeds."
        )

    print(
        "Independent seeds 42-61:             PASSED"
    )

    print(
        "Exactly 20 runs:                     PASSED"
    )

    print(
        "M0-M5 overall completeness:          PASSED"
    )

    print(
        "M0-M5 x 5 horizons completeness:     PASSED"
    )

    print(
        "Common-subset n consistency:         PASSED"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
    )

    args = parser.parse_args()

    bootstrap_repetitions = (
        args.bootstrap_reps
    )

    if (
        bootstrap_repetitions
        < 1000
    ):

        raise ValueError(
            "--bootstrap-reps must be >= 1000."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner(
        "STEP 33 - DENSE MULTI-SEED STATISTICAL ANALYSIS"
    )

    print(
        "Independent seeds:             42-61"
    )

    print(
        "Number of independent runs:    20"
    )

    print(
        "Inferential unit:              independent seed/run"
    )

    print(
        "Prediction rows as n:          NO"
    )

    print(
        "Training:                      NO"
    )

    print(
        "Inference:                     NO"
    )

    print(
        "Model selection:               NO"
    )

    print(
        "Test-based tuning:             NO"
    )

    print(
        f"Bootstrap repetitions:         {bootstrap_repetitions:,}"
    )

    print(
        "Multiplicity correction:       Holm"
    )


    # =========================================================================
    # LOAD STEP-31 COMMON-SUBSET RESULTS
    # =========================================================================

    banner(
        "LOADING 20-RUN COMMON-SUBSET RESULTS"
    )

    overall_frames = []

    horizon_frames = []

    for seed in EXPECTED_SEEDS:

        overall_frames.append(
            load_seed_csv(
                seed,
                "target_test_analysis",
                "step31_common_subset_overall_metrics.csv",
            )
        )

        horizon_frames.append(
            load_seed_csv(
                seed,
                "target_test_analysis",
                "step31_common_subset_horizon_metrics.csv",
            )
        )

    overall = pd.concat(
        overall_frames,
        ignore_index=True,
    )

    horizon = pd.concat(
        horizon_frames,
        ignore_index=True,
    )

    print(
        f"Overall rows: {len(overall):,}"
    )

    print(
        f"Horizon rows: {len(horizon):,}"
    )


    # =========================================================================
    # AUDIT
    # =========================================================================

    validate_core_results(
        overall,
        horizon,
    )


    # =========================================================================
    # SAVE MASTER RUN-LEVEL DATA
    # =========================================================================

    overall.to_csv(
        OUTPUT_DIR
        / "step33_runlevel_overall_master.csv",
        index=False,
    )

    horizon.to_csv(
        OUTPUT_DIR
        / "step33_runlevel_horizon_master.csv",
        index=False,
    )


    # =========================================================================
    # DESCRIPTIVE STATISTICS
    # =========================================================================

    banner(
        "STEP 33B - DENSE DESCRIPTIVE STATISTICS"
    )

    descriptive_rows = []

    bootstrap_counter = 0

    analysis_sets = [

        (
            "overall",
            overall,
            [
                "variant",
            ],
        ),

        (
            "horizon",
            horizon,
            [
                "variant",
                "horizon_hours",
            ],
        ),
    ]

    for (
        scope,
        dataframe,
        group_columns,
    ) in analysis_sets:

        available_metrics = [

            metric

            for metric
            in ALL_PERFORMANCE_METRICS

            if metric
            in dataframe.columns
        ]

        grouper = (
            group_columns[0]
            if len(group_columns) == 1
            else group_columns
        )

        for keys, group in dataframe.groupby(
            grouper,
            sort=False,
        ):

            if not isinstance(
                keys,
                tuple,
            ):

                keys = (
                    keys,
                )

            base = {
                "scope":
                    scope,
            }

            base.update(
                dict(
                    zip(
                        group_columns,
                        keys,
                    )
                )
            )

            for metric in available_metrics:

                bootstrap_counter += 1

                row = dict(
                    base
                )

                row[
                    "metric"
                ] = metric

                row.update(
                    descriptive_statistics(
                        group[
                            metric
                        ].values,
                        bootstrap_repetitions,
                        BOOTSTRAP_RANDOM_SEED
                        + bootstrap_counter,
                    )
                )

                descriptive_rows.append(
                    row
                )

    descriptive_df = pd.DataFrame(
        descriptive_rows
    )

    descriptive_df.to_csv(
        OUTPUT_DIR
        / "step33_descriptive_statistics.csv",
        index=False,
    )

    print(
        f"Descriptive rows generated: {len(descriptive_df):,}"
    )


    # =========================================================================
    # FRIEDMAN OMNIBUS
    # =========================================================================

    banner(
        "STEP 33C - FRIEDMAN REPEATED-MEASURES OMNIBUS"
    )

    omnibus_rows = []

    for metric in PRIMARY_METRICS:

        result = friedman_analysis(
            overall,
            metric,
        )

        if result is not None:

            result.update(
                {
                    "scope":
                        "overall",

                    "horizon_hours":
                        "all",
                }
            )

            omnibus_rows.append(
                result
            )

        for horizon_value in HORIZONS:

            current = horizon[
                horizon[
                    "horizon_hours"
                ]
                == horizon_value
            ]

            result = friedman_analysis(
                current,
                metric,
            )

            if result is not None:

                result.update(
                    {
                        "scope":
                            "horizon",

                        "horizon_hours":
                            horizon_value,
                    }
                )

                omnibus_rows.append(
                    result
                )

    omnibus_df = pd.DataFrame(
        omnibus_rows
    )

    omnibus_df[
        "p_holm_within_metric"
    ] = np.nan

    for metric, indices in (
        omnibus_df
        .groupby(
            "metric"
        )
        .groups
        .items()
    ):

        adjusted = holm_adjust(
            omnibus_df.loc[
                indices,
                "p_raw",
            ].values
        )

        omnibus_df.loc[
            indices,
            "p_holm_within_metric",
        ] = adjusted

    omnibus_df[
        "significant_holm_0_05"
    ] = (
        omnibus_df[
            "p_holm_within_metric"
        ]
        < ALPHA
    )

    omnibus_df.to_csv(
        OUTPUT_DIR
        / "step33_friedman_omnibus.csv",
        index=False,
    )


    # =========================================================================
    # PLANNED M2 PAIRED CONTRASTS
    # =========================================================================

    banner(
        "STEP 33D - PLANNED M2 PAIRED RUN-LEVEL CONTRASTS"
    )

    paired_rows = []

    comparison_counter = 0

    scopes = [

        (
            "overall",
            "all",
            overall,
        )
    ]

    for horizon_value in HORIZONS:

        scopes.append(
            (
                "horizon",
                horizon_value,
                horizon[
                    horizon[
                        "horizon_hours"
                    ]
                    == horizon_value
                ],
            )
        )

    for (
        scope,
        horizon_value,
        dataframe,
    ) in scopes:

        for metric in PRIMARY_METRICS:

            pivot = dataframe.pivot(
                index="seed_run",
                columns="variant",
                values=metric,
            )

            pivot = pivot.reindex(
                columns=VARIANTS
            )

            if len(
                pivot
            ) != 20:

                raise RuntimeError(
                    f"{scope}/{horizon_value}/{metric}: "
                    f"expected 20 paired runs, "
                    f"found {len(pivot)}."
                )

            for comparator in COMPARATORS:

                comparison_counter += 1

                row = {

                    "scope":
                        scope,

                    "horizon_hours":
                        horizon_value,

                    "metric":
                        metric,

                    "comparison":
                        f"M2_vs_{comparator}",

                    "comparator":
                        comparator,
                }

                row.update(
                    paired_analysis(
                        pivot[
                            PROPOSED
                        ].values,
                        pivot[
                            comparator
                        ].values,
                        metric,
                        bootstrap_repetitions,
                        BOOTSTRAP_RANDOM_SEED
                        + 100000
                        + comparison_counter,
                    )
                )

                paired_rows.append(
                    row
                )

    paired_df = pd.DataFrame(
        paired_rows
    )


    # =========================================================================
    # HOLM CORRECTION - PRIMARY FAMILY
    # =========================================================================

    paired_df[
        "p_holm_family"
    ] = np.nan

    family_groups = (
        paired_df
        .groupby(
            [
                "scope",
                "horizon_hours",
                "metric",
            ],
            dropna=False,
        )
        .groups
    )

    for _, indices in family_groups.items():

        paired_df.loc[
            indices,
            "p_holm_family",
        ] = holm_adjust(
            paired_df.loc[
                indices,
                "p_raw",
            ].values
        )


    # =========================================================================
    # HOLM CORRECTION - STRICT METRIC-WIDE SENSITIVITY
    # =========================================================================

    paired_df[
        "p_holm_metricwide"
    ] = np.nan

    for metric, indices in (
        paired_df
        .groupby(
            "metric"
        )
        .groups
        .items()
    ):

        paired_df.loc[
            indices,
            "p_holm_metricwide",
        ] = holm_adjust(
            paired_df.loc[
                indices,
                "p_raw",
            ].values
        )

    paired_df[
        "significant_family_holm_0_05"
    ] = (
        paired_df[
            "p_holm_family"
        ]
        < ALPHA
    )

    paired_df[
        "significant_metricwide_holm_0_05"
    ] = (
        paired_df[
            "p_holm_metricwide"
        ]
        < ALPHA
    )

    paired_df.to_csv(
        OUTPUT_DIR
        / "step33_m2_planned_paired_contrasts.csv",
        index=False,
    )


    # =========================================================================
    # HORIZON DEGRADATION
    # =========================================================================

    banner(
        "STEP 33E - HORIZON DEGRADATION ANALYSIS"
    )

    horizon_degradation_rows = []

    for seed in EXPECTED_SEEDS:

        for variant in VARIANTS:

            current = horizon[
                (
                    horizon[
                        "seed_run"
                    ]
                    == seed
                )
                &
                (
                    horizon[
                        "variant"
                    ]
                    == variant
                )
            ].sort_values(
                "horizon_hours"
            )

            if len(
                current
            ) != 5:

                raise RuntimeError(
                    f"Seed {seed}, {variant}: "
                    "expected five horizons."
                )

            x = current[
                "horizon_hours"
            ].to_numpy(
                dtype=float
            )

            for metric in PRIMARY_METRICS:

                y = current[
                    metric
                ].to_numpy(
                    dtype=float
                )

                regression = stats.linregress(
                    x,
                    y,
                )

                if metric == "r2":

                    endpoint_degradation = (
                        y[0]
                        - y[-1]
                    )

                else:

                    endpoint_degradation = (
                        y[-1]
                        - y[0]
                    )

                horizon_degradation_rows.append(
                    {

                        "seed":
                            seed,

                        "variant":
                            variant,

                        "metric":
                            metric,

                        "linear_slope_per_hour":
                            float(
                                regression.slope
                            ),

                        "slope_intercept":
                            float(
                                regression.intercept
                            ),

                        "slope_r":
                            float(
                                regression.rvalue
                            ),

                        "slope_p_descriptive":
                            float(
                                regression.pvalue
                            ),

                        "slope_standard_error":
                            float(
                                regression.stderr
                            ),

                        "endpoint_degradation_1h_to_24h":
                            float(
                                endpoint_degradation
                            ),
                    }
                )

    horizon_degradation_df = pd.DataFrame(
        horizon_degradation_rows
    )

    horizon_degradation_df.to_csv(
        OUTPUT_DIR
        / "step33_horizon_degradation_runlevel.csv",
        index=False,
    )


    horizon_summary_rows = []

    bootstrap_counter = 200000

    for (
        variant,
        metric,
    ), group in horizon_degradation_df.groupby(
        [
            "variant",
            "metric",
        ]
    ):

        for quantity in [

            "linear_slope_per_hour",

            "endpoint_degradation_1h_to_24h",
        ]:

            bootstrap_counter += 1

            row = {

                "variant":
                    variant,

                "metric":
                    metric,

                "quantity":
                    quantity,
            }

            row.update(
                descriptive_statistics(
                    group[
                        quantity
                    ].values,
                    bootstrap_repetitions,
                    BOOTSTRAP_RANDOM_SEED
                    + bootstrap_counter,
                )
            )

            horizon_summary_rows.append(
                row
            )

    pd.DataFrame(
        horizon_summary_rows
    ).to_csv(
        OUTPUT_DIR
        / "step33_horizon_degradation_summary.csv",
        index=False,
    )


    # =========================================================================
    # TRAINING / CONVERGENCE STABILITY
    # =========================================================================

    banner(
        "STEP 33F - TRAINING / CONVERGENCE STABILITY"
    )

    source_best_rows = []

    adaptation_best_rows = []

    step29_frames = []

    for seed in EXPECTED_SEEDS:


        # ---------------------------------------------------------------------
        # STEP 25
        # ---------------------------------------------------------------------

        step25 = load_seed_csv(
            seed,
            "source_pretraining",
            "bdg2_source_pretraining_history.csv",
        )

        recorded_seeds = set(
            step25[
                "seed"
            ]
            .astype(int)
            .tolist()
        )

        if recorded_seeds != {
            seed
        }:

            raise RuntimeError(
                f"Step 25 seed mismatch "
                f"for seed_{seed}."
            )

        best_index = (
            step25[
                "validation_rmse_z"
            ]
            .astype(float)
            .idxmin()
        )

        best_row = step25.loc[
            best_index
        ]

        source_best_rows.append(
            {

                "seed":
                    seed,

                "best_epoch":
                    int(
                        best_row[
                            "epoch"
                        ]
                    ),

                "best_validation_rmse_z":
                    float(
                        best_row[
                            "validation_rmse_z"
                        ]
                    ),

                "best_validation_mae_z":
                    float(
                        best_row[
                            "validation_mae_z"
                        ]
                    ),

                "epochs_completed":
                    int(
                        step25[
                            "epoch"
                        ].max()
                    ),

                "total_train_minutes":
                    float(
                        step25[
                            "train_seconds"
                        ].sum()
                        / 60.0
                    ),

                "total_validation_minutes":
                    float(
                        step25[
                            "validation_seconds"
                        ].sum()
                        / 60.0
                    ),
            }
        )


        # ---------------------------------------------------------------------
        # STEP 26
        # ---------------------------------------------------------------------

        step26 = load_seed_csv(
            seed,
            "target_adaptation",
            "opensensemap_target_adaptation_history.csv",
        )

        recorded_seeds = set(
            step26[
                "seed"
            ]
            .astype(int)
            .tolist()
        )

        if recorded_seeds != {
            seed
        }:

            raise RuntimeError(
                f"Step 26 seed mismatch "
                f"for seed_{seed}."
            )

        best_index = (
            step26[
                "validation_rmse_z"
            ]
            .astype(float)
            .idxmin()
        )

        best_row = step26.loc[
            best_index
        ]

        adaptation_best_rows.append(
            {

                "seed":
                    seed,

                "best_epoch":
                    int(
                        best_row[
                            "epoch"
                        ]
                    ),

                "best_validation_rmse_z":
                    float(
                        best_row[
                            "validation_rmse_z"
                        ]
                    ),

                "best_validation_mae_z":
                    float(
                        best_row[
                            "validation_mae_z"
                        ]
                    ),

                "epochs_completed":
                    int(
                        step26[
                            "epoch"
                        ].max()
                    ),

                "total_train_minutes":
                    float(
                        step26[
                            "train_seconds"
                        ].sum()
                        / 60.0
                    ),

                "total_validation_minutes":
                    float(
                        step26[
                            "validation_seconds"
                        ].sum()
                        / 60.0
                    ),
            }
        )


        # ---------------------------------------------------------------------
        # STEP 29
        # ---------------------------------------------------------------------

        step29 = load_seed_csv(
            seed,
            "baselines_ablations",
            "step29_validation_summary.csv",
        )

        if set(
            step29[
                "seed"
            ]
            .astype(int)
            .tolist()
        ) != {
            seed
        }:

            raise RuntimeError(
                f"Step 29 run-seed mismatch "
                f"for seed_{seed}."
            )

        if set(
            step29[
                "source_pretraining_seed"
            ]
            .astype(int)
            .tolist()
        ) != {
            seed
        }:

            raise RuntimeError(
                f"Step 29 source-pretraining seed mismatch "
                f"for seed_{seed}."
            )

        step29_frames.append(
            step29
        )


    source_best_df = pd.DataFrame(
        source_best_rows
    )

    adaptation_best_df = pd.DataFrame(
        adaptation_best_rows
    )

    step29_df = pd.concat(
        step29_frames,
        ignore_index=True,
    )


    source_best_df.to_csv(
        OUTPUT_DIR
        / "step33_step25_best_epochs_by_seed.csv",
        index=False,
    )

    adaptation_best_df.to_csv(
        OUTPUT_DIR
        / "step33_step26_best_epochs_by_seed.csv",
        index=False,
    )

    step29_df.to_csv(
        OUTPUT_DIR
        / "step33_step29_validation_by_seed.csv",
        index=False,
    )


    training_summary_rows = []

    bootstrap_counter = 300000

    for (
        stage,
        dataframe,
    ) in [

        (
            "Step25_source_pretraining",
            source_best_df,
        ),

        (
            "Step26_target_adaptation",
            adaptation_best_df,
        ),
    ]:

        quantities = [

            "best_epoch",

            "best_validation_rmse_z",

            "best_validation_mae_z",

            "epochs_completed",

            "total_train_minutes",

            "total_validation_minutes",
        ]

        for quantity in quantities:

            bootstrap_counter += 1

            row = {

                "stage":
                    stage,

                "variant":
                    "",

                "quantity":
                    quantity,
            }

            row.update(
                descriptive_statistics(
                    dataframe[
                        quantity
                    ].values,
                    bootstrap_repetitions,
                    BOOTSTRAP_RANDOM_SEED
                    + bootstrap_counter,
                )
            )

            training_summary_rows.append(
                row
            )


    for variant, group in step29_df.groupby(
        "variant"
    ):

        for quantity in [

            "best_epoch",

            "best_validation_rmse_z",

            "epochs_completed",

            "training_minutes",
        ]:

            bootstrap_counter += 1

            row = {

                "stage":
                    "Step29_baseline_ablation",

                "variant":
                    variant,

                "quantity":
                    quantity,
            }

            row.update(
                descriptive_statistics(
                    group[
                        quantity
                    ].values,
                    bootstrap_repetitions,
                    BOOTSTRAP_RANDOM_SEED
                    + bootstrap_counter,
                )
            )

            training_summary_rows.append(
                row
            )


    training_summary_df = pd.DataFrame(
        training_summary_rows
    )

    training_summary_df.to_csv(
        OUTPUT_DIR
        / "step33_training_stability_summary.csv",
        index=False,
    )


    # =========================================================================
    # SPATIAL ROBUSTNESS
    # =========================================================================

    banner(
        "STEP 33G - PER-BOX SPATIAL ROBUSTNESS"
    )

    box_frames = []

    box_horizon_frames = []

    for seed in EXPECTED_SEEDS:

        box_frames.append(
            load_seed_csv(
                seed,
                "target_test_analysis",
                "step31_per_box_metrics.csv",
            )
        )

        box_horizon_frames.append(
            load_seed_csv(
                seed,
                "target_test_analysis",
                "step31_per_box_horizon_metrics.csv",
            )
        )

    box_df = pd.concat(
        box_frames,
        ignore_index=True,
    )

    box_horizon_df = pd.concat(
        box_horizon_frames,
        ignore_index=True,
    )

    unique_boxes = sorted(
        box_df[
            "box_id"
        ].astype(str).unique()
    )

    print(
        f"Unique target boxes: {len(unique_boxes)}"
    )

    if len(
        unique_boxes
    ) != 17:

        raise RuntimeError(
            "Expected exactly 17 target boxes."
        )


    box_summary_rows = []

    bootstrap_counter = 400000

    for (
        variant,
        box_id,
    ), group in box_df.groupby(
        [
            "variant",
            "box_id",
        ]
    ):

        for metric in PRIMARY_METRICS:

            bootstrap_counter += 1

            row = {

                "variant":
                    variant,

                "box_id":
                    box_id,

                "metric":
                    metric,
            }

            row.update(
                descriptive_statistics(
                    group[
                        metric
                    ].values,
                    bootstrap_repetitions,
                    BOOTSTRAP_RANDOM_SEED
                    + bootstrap_counter,
                )
            )

            box_summary_rows.append(
                row
            )

    box_summary_df = pd.DataFrame(
        box_summary_rows
    )

    box_summary_df.to_csv(
        OUTPUT_DIR
        / "step33_per_box_multiseed_summary.csv",
        index=False,
    )


    # =========================================================================
    # BOX x HORIZON SUMMARY
    # =========================================================================

    box_horizon_summary_rows = []

    bootstrap_counter = 500000

    for (
        variant,
        box_id,
        horizon_value,
    ), group in box_horizon_df.groupby(
        [
            "variant",
            "box_id",
            "horizon_hours",
        ]
    ):

        for metric in PRIMARY_METRICS:

            bootstrap_counter += 1

            row = {

                "variant":
                    variant,

                "box_id":
                    box_id,

                "horizon_hours":
                    horizon_value,

                "metric":
                    metric,
            }

            row.update(
                descriptive_statistics(
                    group[
                        metric
                    ].values,
                    bootstrap_repetitions,
                    BOOTSTRAP_RANDOM_SEED
                    + bootstrap_counter,
                )
            )

            box_horizon_summary_rows.append(
                row
            )

    box_horizon_summary_df = pd.DataFrame(
        box_horizon_summary_rows
    )

    box_horizon_summary_df.to_csv(
        OUTPUT_DIR
        / "step33_per_box_horizon_multiseed_summary.csv",
        index=False,
    )


    # =========================================================================
    # SPATIAL CONSISTENCY OF M2
    # =========================================================================

    spatial_consistency_rows = []

    for metric in PRIMARY_METRICS:

        pivot = box_df.pivot_table(
            index=[
                "seed_run",
                "box_id",
            ],
            columns="variant",
            values=metric,
            aggfunc="first",
        )

        pivot = pivot.dropna(
            subset=VARIANTS
        )

        for comparator in COMPARATORS:

            if metric == "r2":

                improvement = (
                    pivot[
                        "M2"
                    ]
                    - pivot[
                        comparator
                    ]
                )

            else:

                improvement = (
                    pivot[
                        comparator
                    ]
                    - pivot[
                        "M2"
                    ]
                )

            box_mean_improvement = (
                improvement
                .groupby(
                    level="box_id"
                )
                .mean()
            )

            spatial_consistency_rows.append(
                {

                    "metric":
                        metric,

                    "comparison":
                        f"M2_vs_{comparator}",

                    "n_boxes":
                        int(
                            len(
                                box_mean_improvement
                            )
                        ),

                    "boxes_m2_better":
                        int(
                            (
                                box_mean_improvement
                                > EPS
                            ).sum()
                        ),

                    "boxes_tied":
                        int(
                            (
                                np.abs(
                                    box_mean_improvement
                                )
                                <= EPS
                            ).sum()
                        ),

                    "boxes_m2_worse":
                        int(
                            (
                                box_mean_improvement
                                < -EPS
                            ).sum()
                        ),

                    "box_win_fraction":
                        float(
                            (
                                box_mean_improvement
                                > EPS
                            ).mean()
                        ),

                    "median_box_mean_improvement":
                        float(
                            box_mean_improvement
                            .median()
                        ),

                    "minimum_box_mean_improvement":
                        float(
                            box_mean_improvement
                            .min()
                        ),

                    "maximum_box_mean_improvement":
                        float(
                            box_mean_improvement
                            .max()
                        ),
                }
            )

    spatial_consistency_df = pd.DataFrame(
        spatial_consistency_rows
    )

    spatial_consistency_df.to_csv(
        OUTPUT_DIR
        / "step33_spatial_consistency_m2.csv",
        index=False,
    )


    # =========================================================================
    # TEMPORAL ROBUSTNESS
    # =========================================================================

    banner(
        "STEP 33H - TEMPORAL ROBUSTNESS"
    )

    temporal_specs = [

        (
            "daily",
            "step31_daily_metrics.csv",
            "label_date",
        ),

        (
            "hour_of_day",
            "step31_hour_of_day_metrics.csv",
            "label_hour_utc",
        ),

        (
            "day_of_week",
            "step31_day_of_week_metrics.csv",
            "label_dow",
        ),
    ]


    temporal_summary_rows = []

    bootstrap_counter = 600000

    for (
        temporal_scope,
        filename,
        temporal_key,
    ) in temporal_specs:

        frames = []

        for seed in EXPECTED_SEEDS:

            frames.append(
                load_seed_csv(
                    seed,
                    "target_test_analysis",
                    filename,
                )
            )

        dataframe = pd.concat(
            frames,
            ignore_index=True,
        )

        for (
            variant,
            temporal_unit,
        ), group in dataframe.groupby(
            [
                "variant",
                temporal_key,
            ]
        ):

            for metric in PRIMARY_METRICS:

                bootstrap_counter += 1

                row = {

                    "temporal_scope":
                        temporal_scope,

                    "temporal_unit":
                        temporal_unit,

                    "variant":
                        variant,

                    "metric":
                        metric,
                }

                row.update(
                    descriptive_statistics(
                        group[
                            metric
                        ].values,
                        bootstrap_repetitions,
                        BOOTSTRAP_RANDOM_SEED
                        + bootstrap_counter,
                    )
                )

                temporal_summary_rows.append(
                    row
                )

    temporal_summary_df = pd.DataFrame(
        temporal_summary_rows
    )

    temporal_summary_df.to_csv(
        OUTPUT_DIR
        / "step33_temporal_multiseed_summary.csv",
        index=False,
    )


    # =========================================================================
    # MANUSCRIPT PERFORMANCE TABLE
    # =========================================================================

    banner(
        "STEP 33I - MANUSCRIPT-READY TABLES"
    )

    manuscript_performance = descriptive_df[
        descriptive_df[
            "metric"
        ].isin(
            PRIMARY_METRICS
        )
    ].copy()

    manuscript_performance[
        "mean_plus_minus_sd"
    ] = manuscript_performance.apply(

        lambda row:

        (
            f"{row['mean']:.4f} "
            f"± "
            f"{row['sd']:.4f}"
        ),

        axis=1,
    )

    manuscript_performance[
        "ci95_t"
    ] = manuscript_performance.apply(

        lambda row:

        (
            f"["
            f"{row['mean_ci95_low_t']:.4f}, "
            f"{row['mean_ci95_high_t']:.4f}"
            f"]"
        ),

        axis=1,
    )

    manuscript_performance[
        "ci95_bootstrap"
    ] = manuscript_performance.apply(

        lambda row:

        (
            f"["
            f"{row['mean_ci95_low_bootstrap']:.4f}, "
            f"{row['mean_ci95_high_bootstrap']:.4f}"
            f"]"
        ),

        axis=1,
    )

    performance_columns = [

        "scope",

        "horizon_hours",

        "variant",

        "metric",

        "n_runs",

        "mean_plus_minus_sd",

        "median",

        "iqr",

        "ci95_t",

        "ci95_bootstrap",

        "cv_percent",
    ]

    manuscript_performance[
        performance_columns
    ].to_csv(
        OUTPUT_DIR
        / "step33_manuscript_performance_table.csv",
        index=False,
    )


    # =========================================================================
    # MANUSCRIPT INFERENTIAL TABLE
    # =========================================================================

    inferential_columns = [

        "scope",

        "horizon_hours",

        "metric",

        "comparison",

        "n_runs",

        "m2_mean",

        "comparator_mean",

        "mean_improvement",

        "median_improvement",

        "mean_relative_improvement_percent",

        "improvement_ci95_low_bootstrap",

        "improvement_ci95_high_bootstrap",

        "shapiro_p_difference",

        "selected_test",

        "test_statistic",

        "p_raw",

        "p_holm_family",

        "p_holm_metricwide",

        "cohens_dz",

        "hedges_gz",

        "rank_biserial",

        "m2_wins",

        "ties",

        "m2_losses",

        "m2_win_fraction",

        "significant_family_holm_0_05",

        "significant_metricwide_holm_0_05",
    ]

    paired_df[
        inferential_columns
    ].to_csv(
        OUTPUT_DIR
        / "step33_manuscript_inferential_table.csv",
        index=False,
    )


    # =========================================================================
    # OVERALL-ONLY COMPACT TABLE
    # =========================================================================

    overall_compact = (
        manuscript_performance[
            manuscript_performance[
                "scope"
            ]
            == "overall"
        ]
        .copy()
    )

    overall_compact.to_csv(
        OUTPUT_DIR
        / "step33_manuscript_overall_performance.csv",
        index=False,
    )


    overall_inferential = paired_df[
        paired_df[
            "scope"
        ]
        == "overall"
    ].copy()

    overall_inferential.to_csv(
        OUTPUT_DIR
        / "step33_manuscript_overall_inference.csv",
        index=False,
    )


    # =========================================================================
    # MANIFEST
    # =========================================================================

    manifest = {

        "step":
            33,

        "analysis":
            "dense multi-seed statistical analysis",

        "seeds":
            EXPECTED_SEEDS,

        "n_independent_runs":
            20,

        "variants":
            VARIANTS,

        "proposed_variant":
            PROPOSED,

        "comparators":
            COMPARATORS,

        "horizons_hours":
            HORIZONS,

        "primary_metrics":
            PRIMARY_METRICS,

        "secondary_metrics":
            SECONDARY_METRICS,

        "inferential_unit":
            "independent seed/run",

        "prediction_rows_used_as_inferential_units":
            False,

        "bootstrap_repetitions":
            bootstrap_repetitions,

        "bootstrap_random_seed":
            BOOTSTRAP_RANDOM_SEED,

        "alpha":
            ALPHA,

        "normality_test":
            (
                "Shapiro-Wilk test "
                "on paired run-level differences"
            ),

        "parametric_test":
            "paired t-test",

        "nonparametric_test":
            "Wilcoxon signed-rank test",

        "omnibus_test":
            (
                "Friedman repeated-measures "
                "test across M0-M5"
            ),

        "effect_sizes": [

            "paired Cohen dz",

            "small-sample corrected paired standardized effect",

            "matched-pairs rank-biserial correlation",

            "Kendall W",
        ],

        "multiple_testing": {

            "primary":
                (
                    "Holm correction across the five "
                    "planned M2 comparisons within each "
                    "metric and evaluation scope/horizon"
                ),

            "sensitivity":
                (
                    "Holm correction across all M2 "
                    "comparisons within each metric"
                ),
        },

        "confidence_intervals": [

            "Student-t 95% CI",

            "nonparametric bootstrap 95% CI",
        ],

        "training_performed":
            False,

        "inference_performed":
            False,

        "parameter_updates":
            False,

        "checkpoint_selection":
            False,

        "test_based_model_selection":
            False,

        "test_based_tuning":
            False,
    }


    manifest_path = (
        OUTPUT_DIR
        / "step33_statistics_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    banner(
        "STEP 33 COMPLETE"
    )

    print(
        "20-run completeness audit:          PASSED"
    )

    print(
        "Dense descriptive statistics:       COMPLETE"
    )

    print(
        "Student-t confidence intervals:      COMPLETE"
    )

    print(
        "Bootstrap confidence intervals:      COMPLETE"
    )

    print(
        "Friedman omnibus analysis:           COMPLETE"
    )

    print(
        "Kendall's W:                         COMPLETE"
    )

    print(
        "Shapiro-Wilk paired diagnostics:     COMPLETE"
    )

    print(
        "Paired t / Wilcoxon inference:       COMPLETE"
    )

    print(
        "Cohen's dz:                          COMPLETE"
    )

    print(
        "Small-sample corrected effect size:  COMPLETE"
    )

    print(
        "Rank-biserial effect size:           COMPLETE"
    )

    print(
        "Holm multiple-testing correction:    COMPLETE"
    )

    print(
        "M2 win/tie/loss stability:           COMPLETE"
    )

    print(
        "Horizon degradation analysis:        COMPLETE"
    )

    print(
        "Step-25 convergence stability:       COMPLETE"
    )

    print(
        "Step-26 convergence stability:       COMPLETE"
    )

    print(
        "Step-29 convergence stability:       COMPLETE"
    )

    print(
        "17-box spatial robustness:           COMPLETE"
    )

    print(
        "Box x horizon robustness:            COMPLETE"
    )

    print(
        "Daily temporal robustness:           COMPLETE"
    )

    print(
        "Hour-of-day robustness:              COMPLETE"
    )

    print(
        "Day-of-week robustness:              COMPLETE"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()

    print(
        "No model was trained, tuned, selected, "
        "or re-evaluated during Step 33."
    )


if __name__ == "__main__":

    main()