from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

VARIANTS = ["M0", "M1", "M2", "M3", "M4", "M5"]
HORIZONS = [1, 3, 6, 12, 24]


def banner(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonicalize_predictions(df, default_variant=None):
    """Map Step-28 and Step-30 prediction schemas to one frozen analysis schema."""
    df = df.copy()

    rename = {
        "y_true_pm25": "true_pm25",
        "y_pred_pm25": "pred_pm25",
        "y_true_z": "true_z",
        "y_pred_z": "pred_z",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "variant" not in df.columns:
        if default_variant is None:
            raise RuntimeError("Prediction file has no variant column.")
        df["variant"] = default_variant

    required = [
        "variant", "box_id", "origin_timestamp", "horizon_hours",
        "label_timestamp", "true_pm25", "pred_pm25"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Prediction schema missing columns: {missing}")

    df["variant"] = df["variant"].astype(str)
    df["box_id"] = df["box_id"].astype(str)
    df["horizon_hours"] = pd.to_numeric(df["horizon_hours"], errors="raise").astype(int)
    df["origin_timestamp"] = pd.to_datetime(df["origin_timestamp"], utc=True, errors="raise")
    df["label_timestamp"] = pd.to_datetime(df["label_timestamp"], utc=True, errors="raise")
    df["true_pm25"] = pd.to_numeric(df["true_pm25"], errors="raise")
    df["pred_pm25"] = pd.to_numeric(df["pred_pm25"], errors="raise")

    if "error_pm25" not in df.columns:
        df["error_pm25"] = df["pred_pm25"] - df["true_pm25"]
    else:
        df["error_pm25"] = pd.to_numeric(df["error_pm25"], errors="raise")

    if "absolute_error_pm25" not in df.columns:
        df["absolute_error_pm25"] = df["error_pm25"].abs()
    else:
        df["absolute_error_pm25"] = pd.to_numeric(df["absolute_error_pm25"], errors="raise")

    if not np.isfinite(df[["true_pm25", "pred_pm25", "error_pm25", "absolute_error_pm25"]]).all().all():
        raise RuntimeError("Non-finite prediction values found.")

    return df


def metric_record(g):
    y = g["true_pm25"].to_numpy(dtype=np.float64)
    p = g["pred_pm25"].to_numpy(dtype=np.float64)
    e = p - y
    ae = np.abs(e)
    sse = np.sum(e ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - sse / sst if len(y) > 1 and sst > 0 else np.nan
    return {
        "n": int(len(g)),
        "rmse_pm25": float(np.sqrt(np.mean(e ** 2))),
        "mae_pm25": float(np.mean(ae)),
        "median_ae_pm25": float(np.median(ae)),
        "bias_pm25": float(np.mean(e)),
        "error_sd_pm25": float(np.std(e, ddof=1)) if len(e) > 1 else np.nan,
        "p75_ae_pm25": float(np.quantile(ae, 0.75)),
        "p90_ae_pm25": float(np.quantile(ae, 0.90)),
        "p95_ae_pm25": float(np.quantile(ae, 0.95)),
        "p99_ae_pm25": float(np.quantile(ae, 0.99)),
        "max_ae_pm25": float(np.max(ae)),
        "r2": float(r2),
        "true_mean_pm25": float(np.mean(y)),
        "true_sd_pm25": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
    }


def grouped_metrics(df, group_cols):
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, g in df.groupby(grouper, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(metric_record(g))
        rows.append(row)
    return pd.DataFrame(rows)


def add_time_features(df):
    out = df.copy()
    ts = out["label_timestamp"]
    out["label_date"] = ts.dt.strftime("%Y-%m-%d")
    out["label_hour_utc"] = ts.dt.hour.astype(int)
    out["label_dow"] = ts.dt.dayofweek.astype(int)
    out["label_day_name"] = ts.dt.day_name()
    return out


def key_index(df):
    return pd.MultiIndex.from_frame(
        df[["box_id", "origin_timestamp", "horizon_hours", "label_timestamp"]]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed = int(args.seed)

    step28_dir = META / "target_test" / f"seed_{seed}"
    step30_dir = META / "baseline_ablation_test" / f"seed_{seed}"
    m2_file = step28_dir / "opensensemap_target_test_predictions.csv"
    step28_manifest = step28_dir / "opensensemap_target_test_manifest.json"
    other_file = step30_dir / "step30_test_predictions_baselines_ablations.csv"
    step30_manifest = step30_dir / "step30_test_manifest.json"
    out = META / "target_test_analysis" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    banner("STEP 31 - TARGET TEST DIAGNOSTIC / ERROR ANALYSIS")
    print(f"Seed:                  {seed}")
    print("Seed-chain provenance: ENFORCED")
    print("Training:              NO")
    print("Parameter updates:     NO")
    print("Model selection:       NO")
    print("New test inference:    NO")
    print("Inputs:                saved Step-28/30 predictions only")

    for p in [m2_file, step28_manifest, other_file, step30_manifest]:
        require_file(p)

    step28_info = json.loads(step28_manifest.read_text(encoding="utf-8"))
    step30_info = json.loads(step30_manifest.read_text(encoding="utf-8"))

    if int(step28_info.get("seed", -1)) != seed:
        raise RuntimeError("Step-28 manifest seed mismatch.")
    if int(step28_info.get("source_pretraining_seed", -1)) != seed:
        raise RuntimeError("Step-28 source-pretraining seed mismatch.")
    if step28_info.get("seed_chain_verified") is not True:
        raise RuntimeError("Step-28 seed chain was not verified.")

    if int(step30_info.get("seed", -1)) != seed:
        raise RuntimeError("Step-30 manifest seed mismatch.")
    if int(step30_info.get("source_pretraining_seed", -1)) != seed:
        raise RuntimeError("Step-30 source-pretraining seed mismatch.")
    if step30_info.get("end_to_end_seed_chain_verified") is not True:
        raise RuntimeError("Step-30 end-to-end seed chain was not verified.")

    # Ensure Step 30 imported the exact Step-28 M2 metrics belonging to this run.
    expected_m2_source = step28_dir / "opensensemap_target_test_metrics.csv"
    if Path(step30_info.get("m2_source", "")).resolve() != expected_m2_source.resolve():
        raise RuntimeError("Step-30 M2 source does not belong to this seed's Step-28 run.")

    print("Step-28 provenance:     VERIFIED")
    print("Step-30 provenance:     VERIFIED")
    print("End-to-end seed chain:  VERIFIED")

    m2 = canonicalize_predictions(pd.read_csv(m2_file), "M2")
    others = canonicalize_predictions(pd.read_csv(other_file))

    if set(others["variant"].unique()) != {"M0", "M1", "M3", "M4", "M5"}:
        raise RuntimeError(
            f"Unexpected Step-30 variants: {sorted(others['variant'].unique())}"
        )
    if set(m2["variant"].unique()) != {"M2"}:
        raise RuntimeError("Step-28 prediction file is not exclusively M2.")

    df = pd.concat([others, m2], ignore_index=True)
    df = add_time_features(df)

    if not set(df["horizon_hours"].unique()).issubset(set(HORIZONS)):
        raise RuntimeError("Unexpected forecast horizon in predictions.")

    # Each model must contribute at most one prediction for each frozen key.
    key_cols = ["box_id", "origin_timestamp", "horizon_hours", "label_timestamp"]
    duplicate_counts = df.duplicated(["variant"] + key_cols, keep=False)
    if duplicate_counts.any():
        raise RuntimeError(
            f"Duplicate prediction keys found: {int(duplicate_counts.sum()):,} rows."
        )

    # Check ground-truth consistency across neural models on their common keys.
    neural = df[df["variant"].isin(["M1", "M2", "M3", "M4", "M5"])].copy()
    truth_check = (
        neural.groupby(["box_id", "origin_timestamp", "horizon_hours", "label_timestamp"],
                       observed=True)["true_pm25"]
        .agg(["count", "min", "max"])
        .reset_index()
    )
    if (truth_check["count"] != 5).any():
        raise RuntimeError("Neural-model prediction keys are not identical.")
    if ((truth_check["max"] - truth_check["min"]).abs() > 1e-5).any():
        raise RuntimeError("Ground-truth PM2.5 differs across neural-model files.")

    banner("PREDICTION INVENTORY")
    inventory = (
        df.groupby("variant", observed=True)
        .agg(
            predictions=("true_pm25", "size"),
            boxes=("box_id", "nunique"),
            first_label=("label_timestamp", "min"),
            last_label=("label_timestamp", "max"),
        )
        .reset_index()
    )
    print(inventory.to_string(index=False))

    # ------------------------------------------------------------------
    # 1. Overall / horizon / per-box metrics
    # ------------------------------------------------------------------
    overall = grouped_metrics(df, ["variant"])
    horizon = grouped_metrics(df, ["variant", "horizon_hours"])
    per_box = grouped_metrics(df, ["variant", "box_id"])
    per_box_horizon = grouped_metrics(df, ["variant", "box_id", "horizon_hours"])

    overall.to_csv(out / "step31_overall_error_metrics.csv", index=False)
    horizon.to_csv(out / "step31_horizon_error_metrics.csv", index=False)
    per_box.to_csv(out / "step31_per_box_metrics.csv", index=False)
    per_box_horizon.to_csv(out / "step31_per_box_horizon_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Fair common-subset comparison across ALL M0-M5.
    # ------------------------------------------------------------------
    common = None
    for v in VARIANTS:
        idx = key_index(df[df["variant"] == v])
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()

    common_parts = []
    for v in VARIANTS:
        g = df[df["variant"] == v].copy()
        idx = key_index(g)
        keep = idx.isin(common)
        common_parts.append(g.loc[keep])
    common_df = pd.concat(common_parts, ignore_index=True)

    expected_each = len(common)
    common_counts = common_df.groupby("variant").size()
    if not (common_counts == expected_each).all():
        raise RuntimeError("Common-subset construction failed.")

    common_overall = grouped_metrics(common_df, ["variant"])
    common_horizon = grouped_metrics(common_df, ["variant", "horizon_hours"])
    common_overall.to_csv(out / "step31_common_subset_overall_metrics.csv", index=False)
    common_horizon.to_csv(out / "step31_common_subset_horizon_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 3. M2 paired transfer effects on exactly matched prediction keys.
    # Positive RMSE/MAE reduction means M2 has lower error.
    # ------------------------------------------------------------------
    paired_rows = []
    for comparator in ["M0", "M1", "M3", "M4", "M5"]:
        a = common_df[common_df["variant"] == "M2"][key_cols + ["true_pm25", "pred_pm25"]].copy()
        b = common_df[common_df["variant"] == comparator][key_cols + ["pred_pm25"]].copy()
        b = b.rename(columns={"pred_pm25": "comparator_pred"})
        z = a.merge(b, on=key_cols, how="inner", validate="one_to_one")
        z["m2_ae"] = (z["pred_pm25"] - z["true_pm25"]).abs()
        z["comp_ae"] = (z["comparator_pred"] - z["true_pm25"]).abs()
        z["m2_se"] = (z["pred_pm25"] - z["true_pm25"]) ** 2
        z["comp_se"] = (z["comparator_pred"] - z["true_pm25"]) ** 2

        scopes = [("all", z)] + [(str(h), z[z["horizon_hours"] == h]) for h in HORIZONS]
        for scope, q in scopes:
            m2_rmse = math.sqrt(q["m2_se"].mean())
            c_rmse = math.sqrt(q["comp_se"].mean())
            m2_mae = q["m2_ae"].mean()
            c_mae = q["comp_ae"].mean()
            paired_rows.append({
                "comparator": comparator,
                "horizon_hours": scope,
                "n": len(q),
                "m2_rmse_pm25": m2_rmse,
                "comparator_rmse_pm25": c_rmse,
                "rmse_reduction_pm25": c_rmse - m2_rmse,
                "rmse_reduction_percent": 100.0 * (c_rmse - m2_rmse) / c_rmse,
                "m2_mae_pm25": m2_mae,
                "comparator_mae_pm25": c_mae,
                "mae_reduction_pm25": c_mae - m2_mae,
                "mae_reduction_percent": 100.0 * (c_mae - m2_mae) / c_mae,
                "m2_lower_absolute_error_fraction": float((q["m2_ae"] < q["comp_ae"]).mean()),
                "equal_absolute_error_fraction": float((q["m2_ae"] == q["comp_ae"]).mean()),
            })
    transfer = pd.DataFrame(paired_rows)
    transfer.to_csv(out / "step31_m2_paired_transfer_effects.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Per-box M2 effects versus the most relevant learned comparators.
    # ------------------------------------------------------------------
    box_effects = []
    for comparator in ["M1", "M3", "M4"]:
        for box in sorted(common_df["box_id"].unique()):
            a = common_df[(common_df["variant"] == "M2") & (common_df["box_id"] == box)]
            b = common_df[(common_df["variant"] == comparator) & (common_df["box_id"] == box)]
            ma = metric_record(a)
            mb = metric_record(b)
            box_effects.append({
                "box_id": box,
                "comparator": comparator,
                "n": ma["n"],
                "m2_rmse_pm25": ma["rmse_pm25"],
                "comparator_rmse_pm25": mb["rmse_pm25"],
                "rmse_reduction_pm25": mb["rmse_pm25"] - ma["rmse_pm25"],
                "rmse_reduction_percent": 100.0 * (mb["rmse_pm25"] - ma["rmse_pm25"]) / mb["rmse_pm25"],
                "m2_mae_pm25": ma["mae_pm25"],
                "comparator_mae_pm25": mb["mae_pm25"],
                "mae_reduction_pm25": mb["mae_pm25"] - ma["mae_pm25"],
            })
    pd.DataFrame(box_effects).to_csv(out / "step31_m2_per_box_transfer_effects.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Temporal diagnostics: day and UTC hour. Descriptive only.
    # ------------------------------------------------------------------
    daily = grouped_metrics(df, ["variant", "label_date"])
    hourly = grouped_metrics(df, ["variant", "label_hour_utc"])
    dow = grouped_metrics(df, ["variant", "label_dow", "label_day_name"])
    daily.to_csv(out / "step31_daily_metrics.csv", index=False)
    hourly.to_csv(out / "step31_hour_of_day_metrics.csv", index=False)
    dow.to_csv(out / "step31_day_of_week_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Difficult M2 cases / boxes for inspection, not model selection.
    # ------------------------------------------------------------------
    m2df = df[df["variant"] == "M2"].copy()
    m2_box = grouped_metrics(m2df, ["box_id"]).sort_values("rmse_pm25", ascending=False)
    m2_box.to_csv(out / "step31_m2_boxes_ranked_by_rmse.csv", index=False)

    difficult = (
        m2df.sort_values("absolute_error_pm25", ascending=False)
        [["box_id", "origin_timestamp", "label_timestamp", "horizon_hours",
          "true_pm25", "pred_pm25", "error_pm25", "absolute_error_pm25"]]
        .head(250)
    )
    difficult.to_csv(out / "step31_m2_largest_absolute_errors_top250.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Compact audit/manifest.
    # ------------------------------------------------------------------
    manifest = {
        "step": 31,
        "seed": seed,
        "source_pretraining_seed": seed,
        "end_to_end_seed_chain_verified": True,
        "analysis_only": True,
        "training_performed": False,
        "parameter_updates": False,
        "new_test_inference": False,
        "model_selection_from_analysis": False,
        "variants": VARIANTS,
        "horizons_hours": HORIZONS,
        "total_prediction_rows": int(len(df)),
        "common_subset_predictions_per_variant": int(len(common)),
        "common_subset_total_rows": int(len(common_df)),
        "notes": [
            "M0 has fewer native predictions because persistence requires an observed historical PM2.5 anchor.",
            "All direct M0-M5 comparisons in common-subset outputs use identical prediction keys.",
            "Temporal and difficult-case analyses are descriptive diagnostics only.",
            "No inferential statistics across independent seeds are performed in Step 31."
        ],
        "inputs": {
            "m2_step28_predictions": str(m2_file),
            "step28_manifest": str(step28_manifest),
            "m0_m1_m3_m4_m5_step30_predictions": str(other_file),
            "step30_manifest": str(step30_manifest),
        },
        "input_sha256": {
            "m2_step28_predictions": sha256_file(m2_file),
            "step28_manifest": sha256_file(step28_manifest),
            "m0_m1_m3_m4_m5_step30_predictions": sha256_file(other_file),
            "step30_manifest": sha256_file(step30_manifest),
        },
    }
    (out / "step31_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    banner("STEP 31 - COMMON-SUBSET AGGREGATE COMPARISON")
    print(f"Common prediction keys per model: {len(common):,}")
    print(common_overall[
        ["variant", "n", "rmse_pm25", "mae_pm25", "median_ae_pm25",
         "bias_pm25", "p95_ae_pm25", "r2"]
    ].to_string(index=False))

    banner("STEP 31 - M2 PAIRED EFFECTS")
    print(transfer[transfer["horizon_hours"] == "all"][
        ["comparator", "n", "m2_rmse_pm25", "comparator_rmse_pm25",
         "rmse_reduction_percent", "m2_mae_pm25", "comparator_mae_pm25",
         "mae_reduction_percent", "m2_lower_absolute_error_fraction"]
    ].to_string(index=False))

    banner("STEP 31 COMPLETE")
    print(f"Output directory: {out}")
    print("Training performed:       NO")
    print("Parameter updates:         NO")
    print("New test inference:        NO")
    print("Test-based model selection:NO")
    print(f"Independent run seed:     {seed}")
    print("End-to-end seed chain:    VERIFIED")
    print(f"Seed-{seed} methodology is now complete through diagnostic analysis.")


if __name__ == "__main__":
    main()
