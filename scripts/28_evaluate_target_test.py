from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "data" / "final"

PROTOCOL_FILE = META / "forecasting_protocol.json"
TARGET_NORM_FILE = META / "opensensemap_adaptation_normalization_statistics.csv"
TARGET_DATA_FILE = FINAL / "opensensemap" / "opensensemap_target_hourly_wide.csv"
TEST_INDEX = PROCESSED / "opensensemap" / "target_window_index_test.parquet"
STEP24_PATH = ROOT / "scripts" / "24_define_transfer_model.py"

EXPECTED_HISTORY = 24
EXPECTED_HORIZONS = [1, 3, 6, 12, 24]
EXPECTED_SHARED = ["temperature", "relative_humidity", "pressure"]
EXPECTED_TARGET_PRIVATE = "pm25"

BATCH_SIZE = 2048


def banner(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def require_file(p):
    if not p.exists():
        raise FileNotFoundError(p)


def load_step24():
    spec = importlib.util.spec_from_file_location("step24_model", STEP24_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Step 24 model definition.")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_norm(path):
    df = pd.read_csv(path)
    need = {"variable", "mean", "std"}
    if not need.issubset(df.columns):
        raise ValueError(f"Normalization file missing {need - set(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        mean, std = float(r["mean"]), float(r["std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization row: {r.to_dict()}")
        out[str(r["variable"])] = {"mean": mean, "std": std}
    return out


def numeric_float32(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)


def mask_float32(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(np.float32).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.float32).to_numpy()
    return (
        s.astype(str).str.strip().str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
              "1.0": 1.0, "0.0": 0.0})
        .fillna(0.0).astype(np.float32).to_numpy()
    )


def normalize_np(x, mean, std):
    y = (x.astype(np.float32, copy=False) - np.float32(mean)) / np.float32(std)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def prepare_hourly_arrays(path, shared_vars, calendar, norm):
    banner("PREPARING TARGET HOURLY ARRAYS")
    df = pd.read_csv(path)
    needed = (
        ["box_id", "timestamp"]
        + shared_vars
        + [f"{v}_observed" for v in shared_vars]
        + [f"{v}_imputed" for v in shared_vars]
        + calendar
        + ["pm25", "pm25_observed", "pm25_imputed"]
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Target hourly file missing columns:\n{missing}")

    df["box_id"] = df["box_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = (
        df.sort_values(["box_id", "timestamp"])
        .drop_duplicates(["box_id", "timestamp"], keep="last")
    )

    boxes = {}
    for box, g in df.groupby("box_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ts_ns = g["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)

        cols = []
        for v in shared_vars:
            cols.append(normalize_np(
                numeric_float32(g[v]), norm[v]["mean"], norm[v]["std"]
            ))
        for v in shared_vars:
            cols.append(mask_float32(g[f"{v}_observed"]))
        for v in shared_vars:
            cols.append(mask_float32(g[f"{v}_imputed"]))
        for c in calendar:
            cols.append(np.nan_to_num(
                numeric_float32(g[c]), nan=0.0, posinf=0.0, neginf=0.0
            ))

        shared = np.stack(cols, axis=1).astype(np.float32)
        pm25 = normalize_np(
            numeric_float32(g["pm25"]),
            norm["pm25"]["mean"],
            norm["pm25"]["std"],
        )
        private = np.stack(
            [
                pm25,
                mask_float32(g["pm25_observed"]),
                mask_float32(g["pm25_imputed"]),
            ],
            axis=1,
        ).astype(np.float32)

        if shared.shape[1] != 15 or private.shape[1] != 3:
            raise RuntimeError(
                f"Feature contract failed for box {box}: "
                f"{shared.shape}, {private.shape}"
            )

        boxes[str(box)] = {
            "shared": shared,
            "private": private,
            "time_to_pos": {int(t): i for i, t in enumerate(ts_ns)},
        }

    print(f"Hourly rows: {len(df):,}")
    print(f"Boxes:       {len(boxes):,}")
    return boxes


def utc_ns(series):
    return pd.to_datetime(series, utc=True, errors="raise").to_numpy(dtype="datetime64[ns]").astype(np.int64)


def materialize_test(index_path, boxes, norm, horizons, history_length):
    banner("MATERIALIZING TARGET TEST TENSORS")

    idx = pd.read_parquet(index_path).reset_index(drop=True)
    n = len(idx)

    if "partition" in idx.columns:
        parts = set(idx["partition"].astype(str).unique())
        if parts != {"test"}:
            raise RuntimeError(f"Unexpected test partitions: {parts}")

    shared = np.zeros((n, history_length, 15), dtype=np.float32)
    private = np.zeros((n, history_length, 3), dtype=np.float32)
    labels_z = np.zeros((n, len(horizons)), dtype=np.float32)
    masks = np.zeros((n, len(horizons)), dtype=np.float32)

    box_ids = idx["box_id"].astype(str).to_numpy()
    start_ns = utc_ns(idx["history_start_timestamp"])
    origin_ns = utc_ns(idx["origin_timestamp"])

    label_ts = {h: utc_ns(idx[f"label_timestamp_h{h}"]) for h in horizons}
    label_val = {
        h: pd.to_numeric(idx[f"label_pm25_h{h}"], errors="coerce")
        .to_numpy(dtype=np.float64)
        for h in horizons
    }
    label_valid = {
        h: idx[f"label_valid_h{h}"].fillna(False).astype(bool).to_numpy()
        for h in horizons
    }
    label_obs = {
        h: idx[f"label_observed_h{h}"].fillna(False).astype(bool).to_numpy()
        for h in horizons
    }

    one_hour = np.int64(3_600_000_000_000)
    offsets = np.arange(history_length, dtype=np.int64) * one_hour
    pm_mean = float(norm["pm25"]["mean"])
    pm_std = float(norm["pm25"]["std"])

    missing_history_timestamps = 0

    for box in np.unique(box_ids):
        sel = np.flatnonzero(box_ids == box)
        if box not in boxes:
            raise KeyError(f"Test box missing from hourly data: {box}")

        b = boxes[box]
        expected_matrix = start_ns[sel, None] + offsets[None, :]

        if not np.array_equal(expected_matrix[:, -1], origin_ns[sel]):
            bad_local = np.flatnonzero(
                expected_matrix[:, -1] != origin_ns[sel]
            )[0]
            bad = sel[bad_local]
            raise RuntimeError(
                f"History boundary mismatch for test, box={box}, row={bad}"
            )

        for local_i, row_i in enumerate(sel):
            for t, timestamp_ns in enumerate(expected_matrix[local_i]):
                pos = b["time_to_pos"].get(int(timestamp_ns))
                if pos is None:
                    missing_history_timestamps += 1
                    continue
                shared[row_i, t] = b["shared"][pos]
                private[row_i, t] = b["private"][pos]

        for j, h in enumerate(horizons):
            expected_label_ts = origin_ns[sel] + np.int64(h) * one_hour
            if not np.array_equal(label_ts[h][sel], expected_label_ts):
                bad_local = np.flatnonzero(
                    label_ts[h][sel] != expected_label_ts
                )[0]
                bad = sel[bad_local]
                raise RuntimeError(
                    f"Label timestamp mismatch: test, h={h}, row={bad}"
                )

            good = (
                label_valid[h][sel]
                & label_obs[h][sel]
                & np.isfinite(label_val[h][sel])
            )
            if np.any(good):
                rows = sel[good]
                labels_z[rows, j] = (
                    (label_val[h][rows] - pm_mean) / pm_std
                ).astype(np.float32)
                masks[rows, j] = 1.0

    if not np.isfinite(shared).all():
        raise RuntimeError("Non-finite shared test tensor.")
    if not np.isfinite(private).all():
        raise RuntimeError("Non-finite private test tensor.")
    if not np.isfinite(labels_z).all():
        raise RuntimeError("Non-finite test labels.")
    if np.any(masks.sum(axis=1) < 1):
        raise RuntimeError("Retained test origin with zero valid labels.")

    total_history_positions = n * history_length
    matched_history_positions = total_history_positions - missing_history_timestamps
    match_rate = (
        100.0 * matched_history_positions / total_history_positions
        if total_history_positions else 100.0
    )

    print(f"Test origins:               {n:,}")
    print(f"Shared tensor:              {shared.shape}")
    print(f"Private tensor:             {private.shape}")
    print(f"Valid future labels:        {int(masks.sum()):,}")
    print()
    print("TEST MATERIALIZATION AUDIT")
    print(f"Expected history positions: {total_history_positions:,}")
    print(f"Matched history positions:  {matched_history_positions:,}")
    print(f"Missing history positions:  {missing_history_timestamps:,}")
    print(f"Match rate:                 {match_rate:.4f}%")

    # The frozen test-window index was built from the target hourly table.
    # Never silently evaluate with zero-filled histories when an indexed
    # timestamp cannot be found.
    if missing_history_timestamps != 0:
        raise RuntimeError(
            "Target-test timestamp materialization FAILED: "
            f"{missing_history_timestamps:,}/{total_history_positions:,} "
            "indexed history timestamps were not found. Evaluation is blocked."
        )

    print("Timestamp materialization:  PASSED")

    meta = pd.DataFrame({
        "box_id": box_ids,
        "origin_timestamp": pd.to_datetime(
            idx["origin_timestamp"], utc=True
        ).astype(str),
    })

    for h in horizons:
        meta[f"label_timestamp_h{h}"] = pd.to_datetime(
            idx[f"label_timestamp_h{h}"], utc=True
        ).astype(str)

    return (
        torch.from_numpy(shared),
        torch.from_numpy(private),
        torch.from_numpy(labels_z),
        torch.from_numpy(masks),
        meta,
    )


def masked_r2(y, p):
    if len(y) < 2:
        return float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def metric_row(scope, horizon, y, p, pm_mean, pm_std):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = p - y

    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    r2 = masked_r2(y, p)

    # Values are already in original PM2.5 units here.
    # nRMSE is standardized by the frozen adaptation-set PM2.5 std,
    # making it directly related to RMSE(z).
    nrmse = rmse / pm_std

    return {
        "scope": scope,
        "horizon_hours": horizon,
        "n_valid": int(len(y)),
        "rmse_pm25": rmse,
        "mae_pm25": mae,
        "r2": r2,
        "nrmse_adaptation_std": nrmse,
        "adaptation_pm25_mean": pm_mean,
        "adaptation_pm25_std": pm_std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    # Strict end-to-end seed provenance.
    step25_dir = ROOT / "models" / "source_pretraining" / f"seed_{seed}"
    step25_best = step25_dir / "bdg2_source_pretraining_best.pt"
    step25_shared = step25_dir / "bdg2_pretrained_shared_encoder.pt"

    model_dir = ROOT / "models" / "target_adaptation" / f"seed_{seed}"
    result_dir = META / "target_test" / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)

    best_file = model_dir / "opensensemap_target_adaptation_best.pt"
    audit_file = (
        META / "target_adaptation" / f"seed_{seed}"
        / "opensensemap_target_adaptation_audit.json"
    )

    metrics_file = result_dir / "opensensemap_target_test_metrics.csv"
    predictions_file = result_dir / "opensensemap_target_test_predictions.csv"
    manifest_file = result_dir / "opensensemap_target_test_manifest.json"

    banner("STEP 28 - CORRECTED FINAL TARGET TEST EVALUATION")
    print(f"Seed: {seed}")
    print(f"Matching Step-25 run:         {step25_dir}")
    print(f"Matching Step-26 run:         {model_dir}")
    print("End-to-end seed linkage:      ENFORCED")
    print("Training:                    NO")
    print("Parameter updates:           NO")
    print("Validation inspection:       NO")
    print("Target test access:          YES - FIRST FINAL EVALUATION STAGE")
    print("Model selection from test:   FORBIDDEN")

    for p in [
        PROTOCOL_FILE,
        TARGET_NORM_FILE,
        TARGET_DATA_FILE,
        TEST_INDEX,
        STEP24_PATH,
        step25_best,
        step25_shared,
        best_file,
        audit_file,
    ]:
        require_file(p)

    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    if audit.get("status") != "PASSED":
        raise RuntimeError(
            "Step 27 audit did not pass. Test evaluation is blocked."
        )
    if int(audit.get("seed", -1)) != seed:
        raise RuntimeError("Step 27 audit seed mismatch.")
    if int(audit.get("source_pretraining_seed", -1)) != seed:
        raise RuntimeError("Step 27 source-pretraining seed mismatch.")
    if audit.get("seed_to_seed_transfer_linkage_audited") is not True:
        raise RuntimeError("Step 27 did not verify seed-to-seed transfer linkage.")

    print("Step-27 gate:               PASSED")
    print("Step-27 seed-chain audit:    PASSED")

    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    history = int(
        protocol.get(
            "history_length_hours",
            protocol["forecasting_rules"]["history_length_hours"],
        )
    )
    horizons = list(
        protocol.get(
            "forecast_horizons_hours",
            protocol["forecasting_rules"]["forecast_horizons_hours"],
        )
    )
    shared_vars = list(protocol["shared_variables"])
    calendar = list(protocol["calendar_features"])

    if history != EXPECTED_HISTORY:
        raise RuntimeError("History contract changed.")
    if horizons != EXPECTED_HORIZONS:
        raise RuntimeError("Horizon contract changed.")
    if shared_vars != EXPECTED_SHARED:
        raise RuntimeError("Shared-variable contract changed.")
    if protocol["target_private_variable"] != EXPECTED_TARGET_PRIVATE:
        raise RuntimeError("Target-private contract changed.")

    norm = read_norm(TARGET_NORM_FILE)
    pm_mean = float(norm["pm25"]["mean"])
    pm_std = float(norm["pm25"]["std"])

    boxes = prepare_hourly_arrays(
        TARGET_DATA_FILE, shared_vars, calendar, norm
    )
    test = materialize_test(
        TEST_INDEX, boxes, norm, horizons, history
    )
    del boxes

    shared_t, private_t, labels_t, masks_t, meta = test

    banner("LOADING LOCKED BEST STEP-26 MODEL")
    step24 = load_step24()
    model = step24.HeterogeneousTransferForecaster()

    checkpoint = torch.load(
        best_file, map_location="cpu", weights_only=False
    )

    if int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(
            f"Step-26 checkpoint seed mismatch: "
            f"checkpoint={checkpoint.get('seed')}, requested={seed}"
        )
    if int(checkpoint.get("source_pretraining_seed", -1)) != seed:
        raise RuntimeError(
            f"Step-26 source-pretraining seed mismatch: "
            f"checkpoint={checkpoint.get('source_pretraining_seed')}, requested={seed}"
        )

    # Cross-check the Step-25 artifacts themselves.
    source_best = torch.load(step25_best, map_location="cpu", weights_only=False)
    source_shared = torch.load(step25_shared, map_location="cpu", weights_only=False)
    if int(source_best.get("seed", -1)) != seed:
        raise RuntimeError("Step-25 best checkpoint seed mismatch.")
    if int(source_shared.get("seed", -1)) != seed:
        raise RuntimeError("Step-25 shared-encoder seed mismatch.")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(f"Step-25 source seed:         {source_best.get('seed')}")
    print(f"Step-26 source seed:         {checkpoint.get('source_pretraining_seed')}")
    print("Step-25 -> 26 -> 27 -> 28:   SEED CHAIN VERIFIED")
    print(f"Step-26 best epoch:          {checkpoint.get('epoch')}")
    print(
        f"Step-26 validation RMSE(z): "
        f"{checkpoint.get('best_validation_rmse_z')}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device:                      {device}")
    if device.type == "cuda":
        print(f"GPU:                         {torch.cuda.get_device_name(0)}")

    loader = DataLoader(
        TensorDataset(shared_t, private_t, labels_t, masks_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    banner("RUNNING ONE-SHOT TARGET TEST INFERENCE")
    all_pred = []
    all_y = []
    all_mask = []

    t0 = time.time()
    with torch.inference_mode():
        bar = tqdm(
            loader,
            desc="TEST",
            unit="batch",
            dynamic_ncols=True,
        )
        for shared_b, private_b, y_b, mask_b in bar:
            shared_b = shared_b.to(device, non_blocking=True)
            private_b = private_b.to(device, non_blocking=True)

            # Use the target path explicitly. This mirrors Step 26 and prevents
            # accidental use of the source energy head.
            shared_rep, _ = model.shared_encoder(shared_b)
            private_rep = model.target_private_encoder(private_b)
            fused, _ = model.target_fusion(shared_rep, private_rep)
            pred_b = model.target_head(fused)

            all_pred.append(pred_b.float().cpu())
            all_y.append(y_b.float().cpu())
            all_mask.append(mask_b.float().cpu())

    seconds = time.time() - t0

    pred_z = torch.cat(all_pred, dim=0).numpy()
    y_z = torch.cat(all_y, dim=0).numpy()
    mask = torch.cat(all_mask, dim=0).numpy().astype(bool)

    if pred_z.shape != y_z.shape or pred_z.shape != mask.shape:
        raise RuntimeError(
            f"Prediction shape mismatch: {pred_z.shape}, "
            f"{y_z.shape}, {mask.shape}"
        )
    if pred_z.shape[1] != len(horizons):
        raise RuntimeError("Unexpected number of forecast horizons.")
    if not np.isfinite(pred_z).all():
        raise RuntimeError("Non-finite test predictions.")

    pred_pm25 = pred_z * pm_std + pm_mean
    y_pm25 = y_z * pm_std + pm_mean

    banner("COMPUTING FINAL TEST METRICS")
    rows = []

    pooled_y = y_pm25[mask]
    pooled_p = pred_pm25[mask]
    rows.append(
        metric_row(
            "aggregate",
            "all",
            pooled_y,
            pooled_p,
            pm_mean,
            pm_std,
        )
    )

    for j, h in enumerate(horizons):
        good = mask[:, j]
        rows.append(
            metric_row(
                "horizon",
                h,
                y_pm25[good, j],
                pred_pm25[good, j],
                pm_mean,
                pm_std,
            )
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_file, index=False)

    print(metrics_df.to_string(index=False))

    banner("SAVING PREDICTION-LEVEL RESULTS")
    prediction_rows = []
    for i in range(len(meta)):
        for j, h in enumerate(horizons):
            if not mask[i, j]:
                continue
            prediction_rows.append({
                "seed": seed,
                "box_id": meta.iloc[i]["box_id"],
                "origin_timestamp": meta.iloc[i]["origin_timestamp"],
                "horizon_hours": h,
                "label_timestamp": meta.iloc[i][f"label_timestamp_h{h}"],
                "y_true_z": float(y_z[i, j]),
                "y_pred_z": float(pred_z[i, j]),
                "y_true_pm25": float(y_pm25[i, j]),
                "y_pred_pm25": float(pred_pm25[i, j]),
                "error_pm25": float(pred_pm25[i, j] - y_pm25[i, j]),
                "absolute_error_pm25": float(
                    abs(pred_pm25[i, j] - y_pm25[i, j])
                ),
            })

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(
        predictions_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    manifest = {
        "step": 28,
        "seed": seed,
        "source_pretraining_seed": seed,
        "seed_chain_verified": True,
        "evaluation_type": "locked one-shot target test evaluation",
        "timestamp_materialization": "explicit datetime64[ns], fail-closed zero-missing audit",
        "step27_gate": "PASSED",
        "step27_seed_chain_audit": "PASSED",
        "step25_directory": str(step25_dir),
        "step25_best_checkpoint": str(step25_best),
        "step25_shared_encoder": str(step25_shared),
        "step26_best_checkpoint": str(best_file),
        "step27_audit": str(audit_file),
        "step26_best_epoch": int(checkpoint["epoch"]),
        "history_length_hours": history,
        "forecast_horizons_hours": horizons,
        "test_origins": int(len(shared_t)),
        "valid_test_predictions": int(mask.sum()),
        "batch_size": BATCH_SIZE,
        "inference_seconds": seconds,
        "pm25_denormalization": {
            "mean": pm_mean,
            "std": pm_std,
            "source": "frozen openSenseMap adaptation normalization",
        },
        "metrics": {
            "rmse": "original PM2.5 units",
            "mae": "original PM2.5 units",
            "r2": "computed on original PM2.5 values",
            "nrmse_adaptation_std":
                "RMSE divided by frozen adaptation-set PM2.5 standard deviation",
        },
        "test_usage": {
            "gradient_updates": False,
            "normalization_fit": False,
            "model_selection": False,
            "early_stopping": False,
            "threshold_selection": False,
            "hyperparameter_selection": False,
        },
        "outputs": {
            "metrics_csv": str(metrics_file),
            "predictions_csv": str(predictions_file),
        },
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    banner("STEP 28 COMPLETE")
    print(f"Seed:                  {seed}")
    print(f"Source pretraining:    seed_{seed}")
    print("Seed chain:            VERIFIED")
    print(f"Test origins:          {len(shared_t):,}")
    print(f"Valid predictions:     {int(mask.sum()):,}")
    print(f"Inference time:        {seconds:.2f} s")
    print()
    print("Generated:")
    print(f"  {metrics_file}")
    print(f"  {predictions_file}")
    print(f"  {manifest_file}")
    print()
    print("No training or parameter update was performed.")
    print("No test result was used for model selection.")
    print(f"Seed-{seed} target test evaluation is now complete.")


if __name__ == "__main__":
    main()
