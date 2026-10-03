from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from datetime import datetime, timezone
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
ADAPTATION_INDEX = PROCESSED / "opensensemap" / "target_window_index_adaptation.parquet"
VALIDATION_INDEX = PROCESSED / "opensensemap" / "target_window_index_validation.parquet"
STEP24_PATH = ROOT / "scripts" / "24_define_transfer_model.py"

BATCH_SIZE = 1024
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
USE_AMP = True
DEFAULT_SEED = 42


def banner(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def require_file(p):
    if not p.exists():
        raise FileNotFoundError(p)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_torch_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_step24():
    spec = importlib.util.spec_from_file_location("step24_model", STEP24_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Step 24.")
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
            raise ValueError(f"Invalid normalization: {r.to_dict()}")
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
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})
        .fillna(0.0).astype(np.float32).to_numpy()
    )


def normalize_np(x, mean, std):
    y = (x.astype(np.float32, copy=False) - np.float32(mean)) / np.float32(std)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def prepare_hourly_arrays(path, shared_vars, calendar, norm):
    banner("PREPARING TARGET HOURLY ARRAYS ONCE")
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
    df = df.sort_values(["box_id", "timestamp"]).drop_duplicates(["box_id", "timestamp"], keep="last")

    boxes = {}
    for box, g in df.groupby("box_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ts_ns = g["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        cols = []
        for v in shared_vars:
            cols.append(normalize_np(numeric_float32(g[v]), norm[v]["mean"], norm[v]["std"]))
        for v in shared_vars:
            cols.append(mask_float32(g[f"{v}_observed"]))
        for v in shared_vars:
            cols.append(mask_float32(g[f"{v}_imputed"]))
        for c in calendar:
            cols.append(np.nan_to_num(numeric_float32(g[c]), nan=0.0, posinf=0.0, neginf=0.0))

        shared = np.stack(cols, axis=1).astype(np.float32)
        pm25 = normalize_np(numeric_float32(g["pm25"]), norm["pm25"]["mean"], norm["pm25"]["std"])
        private = np.stack(
            [pm25, mask_float32(g["pm25_observed"]), mask_float32(g["pm25_imputed"])], axis=1
        ).astype(np.float32)

        if shared.shape[1] != 15 or private.shape[1] != 3:
            raise RuntimeError(f"Feature contract failed for box {box}: {shared.shape}, {private.shape}")

        boxes[str(box)] = {
            "timestamps_ns": ts_ns,
            "shared": shared,
            "private": private,
            "time_to_pos": {int(t): i for i, t in enumerate(ts_ns)},
        }

    print(f"Hourly rows: {len(df):,}")
    print(f"Boxes:       {len(boxes):,}")
    return boxes


def utc_ns(series):
    return pd.to_datetime(series, utc=True, errors="raise").to_numpy(dtype="datetime64[ns]").astype(np.int64)


def materialize_partition(index_path, partition, boxes, norm, horizons, history_length):
    banner(f"MATERIALIZING {partition.upper()} TENSORS IN RAM")
    idx = pd.read_parquet(index_path)
    n = len(idx)

    shared = np.zeros((n, history_length, 15), dtype=np.float32)
    private = np.zeros((n, history_length, 3), dtype=np.float32)
    labels = np.zeros((n, len(horizons)), dtype=np.float32)
    masks = np.zeros((n, len(horizons)), dtype=np.float32)

    box_ids = idx["box_id"].astype(str).to_numpy()
    start_ns = utc_ns(idx["history_start_timestamp"])
    origin_ns = utc_ns(idx["origin_timestamp"])

    label_ts = {h: utc_ns(idx[f"label_timestamp_h{h}"]) for h in horizons}
    label_val = {h: pd.to_numeric(idx[f"label_pm25_h{h}"], errors="coerce").to_numpy(dtype=np.float64) for h in horizons}
    label_valid = {h: idx[f"label_valid_h{h}"].fillna(False).astype(bool).to_numpy() for h in horizons}
    label_obs = {h: idx[f"label_observed_h{h}"].fillna(False).astype(bool).to_numpy() for h in horizons}

    one_hour = 3_600_000_000_000
    pm_mean = norm["pm25"]["mean"]
    pm_std = norm["pm25"]["std"]

    missing_history_timestamps = 0
    offsets = np.arange(history_length, dtype=np.int64) * one_hour

    for box in np.unique(box_ids):
        sel = np.flatnonzero(box_ids == box)
        if box not in boxes:
            raise KeyError(f"Target box missing from hourly data: {box}")
        bd = boxes[box]
        expected_matrix = start_ns[sel, None] + offsets[None, :]
        if not np.array_equal(expected_matrix[:, -1], origin_ns[sel]):
            bad_local = np.flatnonzero(expected_matrix[:, -1] != origin_ns[sel])[0]
            raise RuntimeError(f"History boundary mismatch for {partition}, box={box}, row={sel[bad_local]}")
        for local_i, row_i in enumerate(sel):
            for t, timestamp_ns in enumerate(expected_matrix[local_i]):
                pos = bd["time_to_pos"].get(int(timestamp_ns))
                if pos is None:
                    missing_history_timestamps += 1
                    continue
                shared[row_i, t] = bd["shared"][pos]
                private[row_i, t] = bd["private"][pos]

        for j, h in enumerate(horizons):
            expected = origin_ns[sel] + h * one_hour
            if not np.array_equal(label_ts[h][sel], expected):
                bad = sel[np.flatnonzero(label_ts[h][sel] != expected)[0]]
                raise RuntimeError(f"Label timestamp mismatch: {partition}, h={h}, row={bad}")
            good = label_valid[h][sel] & label_obs[h][sel] & np.isfinite(label_val[h][sel])
            if np.any(good):
                rows = sel[good]
                labels[rows, j] = ((label_val[h][rows] - pm_mean) / pm_std).astype(np.float32)
                masks[rows, j] = 1.0

    if not np.isfinite(shared).all() or not np.isfinite(private).all() or not np.isfinite(labels).all():
        raise RuntimeError(f"Non-finite tensor in {partition}")
    if np.any(masks.sum(axis=1) < 1):
        raise RuntimeError(f"{partition} contains retained origins with zero valid labels.")

    total_history_timestamps = n * history_length
    matched_history_timestamps = total_history_timestamps - missing_history_timestamps
    match_rate = (
        100.0 * matched_history_timestamps / total_history_timestamps
        if total_history_timestamps else 100.0
    )

    print(f"Origins:       {n:,}")
    print(f"Shared:        {shared.shape}  {shared.nbytes/1024**2:.1f} MiB")
    print(f"Private:       {private.shape}  {private.nbytes/1024**2:.1f} MiB")
    print(f"Labels/masks:  {labels.shape}")
    print(f"Valid labels:  {int(masks.sum()):,}")
    print()
    print(f"{partition.upper()} MATERIALIZATION AUDIT")
    print(f"Expected history positions: {total_history_timestamps:,}")
    print(f"Matched history positions:  {matched_history_timestamps:,}")
    print(f"Missing history positions:  {missing_history_timestamps:,}")
    print(f"Match rate:                 {match_rate:.4f}%")

    # These frozen window indices were generated from the target hourly table.
    # A missing indexed history timestamp at this stage signals a timestamp
    # representation/materialization error. Never silently train on zero histories.
    if missing_history_timestamps != 0:
        raise RuntimeError(
            f"{partition} timestamp materialization audit FAILED: "
            f"{missing_history_timestamps:,}/{total_history_timestamps:,} "
            "indexed history timestamps were not found. Training is blocked."
        )

    print("Timestamp materialization:  PASSED")

    return (
        torch.from_numpy(shared),
        torch.from_numpy(private),
        torch.from_numpy(labels),
        torch.from_numpy(masks),
    )


def masked_metrics(pred, target, mask, horizons):
    e = pred.float() - target.float()
    sq = e.square() * mask.float()
    ab = e.abs() * mask.float()
    sse, sae, n = sq.sum().item(), ab.sum().item(), mask.sum().item()
    hsse = sq.sum(0).cpu().numpy()
    hsae = ab.sum(0).cpu().numpy()
    hn = mask.sum(0).cpu().numpy()
    if n <= 0:
        raise RuntimeError("No valid labels.")
    return sse, sae, n, hsse, hsae, hn


def finish_metrics(sse, sae, n, hsse, hsae, hn, horizons, seconds):
    hrmse = np.sqrt(np.divide(hsse, hn, out=np.full_like(hsse, np.nan), where=hn > 0))
    hmae = np.divide(hsae, hn, out=np.full_like(hsae, np.nan), where=hn > 0)
    return {
        "rmse": math.sqrt(sse / n),
        "mae": sae / n,
        "valid_labels": int(n),
        "seconds": float(seconds),
        "horizon_rmse": {str(h): float(hrmse[i]) for i, h in enumerate(horizons)},
        "horizon_mae": {str(h): float(hmae[i]) for i, h in enumerate(horizons)},
        "horizon_n": {str(h): int(hn[i]) for i, h in enumerate(horizons)},
    }


def forward_variant(model, shared, private, variant, amp, training):
    # M1: fully target-only, all target-path components learned from random init.
    # M3: pretrained/frozen shared encoder, random target-private encoder.
    # M4: pretrained shared encoder is fine-tuned, source-private -> target-private init.
    # M5: pretrained/frozen shared encoder, private representation ablated to zero.
    if variant in {"M3", "M5"}:
        with torch.no_grad():
            with torch.amp.autocast(device_type=shared.device.type, dtype=torch.float16, enabled=amp):
                shared_rep, _ = model.shared_encoder(shared)
        shared_rep = shared_rep.detach()
    else:
        with torch.amp.autocast(device_type=shared.device.type, dtype=torch.float16, enabled=amp):
            shared_rep, _ = model.shared_encoder(shared)

    with torch.amp.autocast(device_type=shared.device.type, dtype=torch.float16, enabled=amp):
        if variant == "M5":
            # Keep fusion dimensionality identical while removing all information
            # contributed by the PM2.5-private pathway.
            private_rep = model.target_private_encoder(private)
            private_rep = torch.zeros_like(private_rep)
        else:
            private_rep = model.target_private_encoder(private)
        fused, _ = model.target_fusion(shared_rep, private_rep)
        return model.target_head(fused)


def run_epoch(model, loader, device, horizons, variant, optimizer=None, scaler=None, epoch=1):
    training = optimizer is not None
    if training:
        model.train()
        # Frozen modules must remain in eval mode.
        for module in [model.shared_encoder, model.target_private_encoder,
                       model.source_private_encoder, model.source_fusion, model.source_head]:
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        desc = f"{variant} Epoch {epoch:02d} TRAIN"
    else:
        model.eval()
        desc = f"{variant} Epoch {epoch:02d} VALID"

    sse = sae = n = 0.0
    hsse = np.zeros(len(horizons), dtype=np.float64)
    hsae = np.zeros(len(horizons), dtype=np.float64)
    hn = np.zeros(len(horizons), dtype=np.float64)
    t0 = time.time()
    amp = USE_AMP and device.type == "cuda"
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        bar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
        for shared, private, target, mask in bar:
            shared = shared.to(device, non_blocking=True)
            private = private.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred = forward_variant(model, shared, private, variant, amp, training)
            loss = (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{variant}: non-finite loss at epoch {epoch}")

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None],
                    GRADIENT_CLIP_NORM,
                )
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()

            with torch.no_grad():
                a = masked_metrics(pred, target, mask, horizons)
                sse += a[0]; sae += a[1]; n += a[2]
                hsse += a[3]; hsae += a[4]; hn += a[5]
                bar.set_postfix(rmse_z=f"{math.sqrt(sse/max(n,1)):.5f}",
                                valid=f"{int(n):,}", refresh=False)

    return finish_metrics(sse, sae, n, hsse, hsae, hn, horizons, time.time() - t0)


def configure_variant(step24, source_ckpt, shared_payload, variant):
    # Re-seeding occurs immediately before this function, so random target
    # initialization is reproducible for each independent seed.
    model = step24.HeterogeneousTransferForecaster()

    # Freeze everything first. Explicitly enable only the intended pathway.
    for p in model.parameters():
        p.requires_grad = False

    source_state = source_ckpt["model_state_dict"]
    prefix = "source_private_encoder."
    sp = {k[len(prefix):]: v for k, v in source_state.items() if k.startswith(prefix)}
    if not sp:
        raise RuntimeError("Step-25 source_private_encoder not found.")

    if variant == "M1":
        # Target-only: no Step-25 weights enter the target forecasting pathway.
        for module in (model.shared_encoder, model.target_private_encoder,
                       model.target_fusion, model.target_head):
            for p in module.parameters():
                p.requires_grad = True
        description = "Target-only; all target-path modules randomly initialized and trainable"

    elif variant == "M3":
        model.shared_encoder.load_state_dict(shared_payload["state_dict"], strict=True)
        for module in (model.target_private_encoder, model.target_fusion, model.target_head):
            for p in module.parameters():
                p.requires_grad = True
        description = "Shared transfer only; pretrained shared frozen, target-private random"

    elif variant == "M4":
        model.shared_encoder.load_state_dict(shared_payload["state_dict"], strict=True)
        model.target_private_encoder.load_state_dict(sp, strict=True)
        for module in (model.shared_encoder, model.target_private_encoder,
                       model.target_fusion, model.target_head):
            for p in module.parameters():
                p.requires_grad = True
        description = "Proposed initialization with shared encoder fine-tuned"

    elif variant == "M5":
        model.shared_encoder.load_state_dict(shared_payload["state_dict"], strict=True)
        # Initialize private encoder consistently, but it is frozen and its output
        # is explicitly zeroed. It contributes no information and no gradients.
        model.target_private_encoder.load_state_dict(sp, strict=True)
        for module in (model.target_fusion, model.target_head):
            for p in module.parameters():
                p.requires_grad = True
        description = "No target-private information; private representation forced to zero"

    else:
        raise ValueError(variant)

    return model, description


def persistence_validation(private, labels, masks, horizons):
    # Persistence predicts future PM2.5 as the most recent directly observed
    # PM2.5 value in the 24-hour history. Because private channel 1 is the
    # observed mask, imputed PM2.5 values are not selected as the persistence anchor.
    x = private.numpy()
    y = labels.numpy()
    m = masks.numpy().astype(bool)
    preds = np.zeros_like(y, dtype=np.float32)
    usable = np.zeros(len(x), dtype=bool)

    for i in range(len(x)):
        observed_positions = np.flatnonzero(x[i, :, 1] > 0.5)
        if len(observed_positions):
            preds[i, :] = x[i, observed_positions[-1], 0]
            usable[i] = True

    effective = m & usable[:, None]
    if effective.sum() == 0:
        raise RuntimeError("Persistence baseline has no evaluable labels.")
    e = preds - y
    rows = []
    all_e = e[effective]
    rows.append({
        "variant": "M0", "horizon_hours": "all",
        "rmse_z": float(np.sqrt(np.mean(all_e ** 2))),
        "mae_z": float(np.mean(np.abs(all_e))),
        "n_valid": int(effective.sum()),
    })
    for j, h in enumerate(horizons):
        g = effective[:, j]
        rows.append({
            "variant": "M0", "horizon_hours": h,
            "rmse_z": float(np.sqrt(np.mean(e[g, j] ** 2))),
            "mae_z": float(np.mean(np.abs(e[g, j]))),
            "n_valid": int(g.sum()),
        })
    return rows, int(usable.sum())


def r2_from_arrays(pred, target, mask):
    g = mask.astype(bool)
    yt = target[g].astype(np.float64)
    yp = pred[g].astype(np.float64)
    if len(yt) < 2:
        return float("nan")
    sse = np.sum((yt - yp) ** 2)
    sst = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def infer_variant(model, loader, device, variant):
    model.eval()
    preds, targets, masks = [], [], []
    amp = USE_AMP and device.type == "cuda"
    t0 = time.time()
    with torch.inference_mode():
        bar = tqdm(loader, desc=f"{variant} TEST", unit="batch", dynamic_ncols=True)
        for shared, private, target, mask in bar:
            shared = shared.to(device, non_blocking=True)
            private = private.to(device, non_blocking=True)
            pred = forward_variant(model, shared, private, variant, amp, False)
            preds.append(pred.float().cpu())
            targets.append(target.float().cpu())
            masks.append(mask.bool().cpu())
    return (
        torch.cat(preds).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(masks).numpy(),
        time.time() - t0,
    )


def metric_rows(variant, pred_z, true_z, mask, horizons, pm25_mean, pm25_std):
    pred_pm = pred_z * pm25_std + pm25_mean
    true_pm = true_z * pm25_std + pm25_mean
    rows = []

    def one(scope, h, g):
        e = pred_pm[g] - true_pm[g]
        return {
            "variant": variant,
            "scope": scope,
            "horizon_hours": h,
            "n_valid": int(g.sum()),
            "rmse_pm25": float(np.sqrt(np.mean(e ** 2))),
            "mae_pm25": float(np.mean(np.abs(e))),
            "r2": r2_from_arrays(pred_pm, true_pm, g),
            "nrmse_adaptation_std": float(np.sqrt(np.mean(e ** 2)) / pm25_std),
            "adaptation_pm25_mean": float(pm25_mean),
            "adaptation_pm25_std": float(pm25_std),
        }

    rows.append(one("aggregate", "all", mask.astype(bool)))
    for j, h in enumerate(horizons):
        g = np.zeros_like(mask, dtype=bool)
        g[:, j] = mask[:, j]
        rows.append(one("horizon", h, g))
    return rows, pred_pm, true_pm


def persistence_test(private, true_z, mask):
    x = private.numpy()
    pred = np.zeros_like(true_z, dtype=np.float32)
    usable = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        obs = np.flatnonzero(x[i, :, 1] > 0.5)
        if len(obs):
            pred[i, :] = x[i, obs[-1], 0]
            usable[i] = True
    effective = mask & usable[:, None]
    return pred, effective, usable


def prediction_frame(meta, variant, pred_z, true_z, mask, pred_pm, true_pm, horizons):
    records = []
    for j, h in enumerate(horizons):
        g = mask[:, j].astype(bool)
        if not g.any():
            continue
        part = meta.loc[g, ["box_id", "origin_timestamp"]].copy()
        part["variant"] = variant
        part["horizon_hours"] = h
        part["label_timestamp"] = (
            pd.to_datetime(part["origin_timestamp"], utc=True)
            + pd.to_timedelta(h, unit="h")
        )
        part["true_z"] = true_z[g, j]
        part["pred_z"] = pred_z[g, j]
        part["true_pm25"] = true_pm[g, j]
        part["pred_pm25"] = pred_pm[g, j]
        part["error_pm25"] = pred_pm[g, j] - true_pm[g, j]
        part["absolute_error_pm25"] = np.abs(part["error_pm25"])
        records.append(part)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    seed = int(args.seed)

    # Strict end-to-end independent-run provenance.
    step25_dir = ROOT / "models" / "source_pretraining" / f"seed_{seed}"
    step25_best = step25_dir / "bdg2_source_pretraining_best.pt"
    step25_shared = step25_dir / "bdg2_pretrained_shared_encoder.pt"

    # Test path is intentionally introduced only in Step 30.
    test_index = PROCESSED / "opensensemap" / "target_window_index_test.parquet"
    step27_audit = META / "target_adaptation" / f"seed_{seed}" / "opensensemap_target_adaptation_audit.json"
    step26_best = ROOT / "models" / "target_adaptation" / f"seed_{seed}" / "opensensemap_target_adaptation_best.pt"
    step28_dir = META / "target_test" / f"seed_{seed}"
    step28_metrics = step28_dir / "opensensemap_target_test_metrics.csv"
    step28_manifest = step28_dir / "opensensemap_target_test_manifest.json"
    step29_models = ROOT / "models" / "baselines_ablations" / f"seed_{seed}"
    step29_meta = META / "baselines_ablations" / f"seed_{seed}"
    out_dir = META / "baseline_ablation_test" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("STEP 30 - LOCKED BASELINE / ABLATION TARGET-TEST EVALUATION")
    print(f"Seed:                       {seed}")
    print(f"Matching Step-25 run:       {step25_dir}")
    print(f"Matching Step-26 run:       {step26_best.parent}")
    print(f"Matching Step-29 run:       {step29_models}")
    print("End-to-end seed linkage:    ENFORCED")
    print("Training:                   NO")
    print("Parameter updates:          NO")
    print("Checkpoint selection:       validation-locked only")
    print("Target test access:         YES - evaluation only")
    print("Timestamp representation:   explicit datetime64[ns]")

    required = [
        PROTOCOL_FILE, TARGET_NORM_FILE, TARGET_DATA_FILE, test_index,
        STEP24_PATH, step25_best, step25_shared, step27_audit, step26_best,
        step28_metrics, step28_manifest,
        step29_meta / "M0_persistence_validation_metrics.csv",
    ]
    for v in ["M1", "M3", "M4", "M5"]:
        required += [step29_models / f"{v}_best.pt", step29_meta / f"{v}_manifest.json"]
    for p in required:
        require_file(p)

    audit = json.loads(step27_audit.read_text(encoding="utf-8"))
    # Support either explicit status fields or the known Step-27 audit structure.
    audit_text = json.dumps(audit).lower()
    if '"failed"' in audit_text or '"status": "failed"' in audit_text:
        raise RuntimeError("Step-27 audit is not passed.")
    if int(audit.get("seed", -1)) != seed:
        raise RuntimeError("Step-27 audit seed mismatch.")
    if int(audit.get("source_pretraining_seed", -1)) != seed:
        raise RuntimeError("Step-27 source-pretraining seed mismatch.")
    if audit.get("seed_to_seed_transfer_linkage_audited") is not True:
        raise RuntimeError("Step-27 did not audit seed-to-seed transfer linkage.")
    print("Step-27 gate:                PASSED")
    print("Step-27 seed-chain audit:    PASSED")

    with open(PROTOCOL_FILE, encoding="utf-8") as f:
        protocol = json.load(f)
    history = int(protocol.get("history_length_hours",
                               protocol["forecasting_rules"]["history_length_hours"]))
    horizons = list(protocol.get("forecast_horizons_hours",
                                 protocol["forecasting_rules"]["forecast_horizons_hours"]))
    shared_vars = list(protocol["shared_variables"])
    calendar = list(protocol["calendar_features"])
    if history != 24 or horizons != [1, 3, 6, 12, 24]:
        raise RuntimeError("Frozen forecasting contract changed.")

    norm = read_norm(TARGET_NORM_FILE)
    pm25_mean = float(norm["pm25"]["mean"])
    pm25_std = float(norm["pm25"]["std"])

    banner("PREPARING / AUDITING TARGET TEST")
    boxes = prepare_hourly_arrays(TARGET_DATA_FILE, shared_vars, calendar, norm)
    test = materialize_partition(test_index, "test", boxes, norm, horizons, history)
    del boxes
    # Step-26/29 materialize_partition returns exactly four tensors.
    # Test metadata is reconstructed directly from the frozen test index,
    # preserving row order one-to-one with the materialized tensors.
    shared, private, labels, masks = test

    meta = pd.read_parquet(test_index)
    if len(meta) != len(shared):
        raise RuntimeError(
            f"Frozen test-index row count ({len(meta):,}) does not match "
            f"materialized test origins ({len(shared):,})."
        )

    box_col = "box_id" if "box_id" in meta.columns else "box"
    if box_col not in meta.columns:
        raise RuntimeError("Frozen test index has no box_id/box column.")
    if "origin_timestamp" not in meta.columns:
        raise RuntimeError("Frozen test index has no origin_timestamp column.")

    meta = meta[[box_col, "origin_timestamp"]].copy()
    meta = meta.rename(columns={box_col: "box_id"})
    meta["box_id"] = meta["box_id"].astype(str)
    meta["origin_timestamp"] = pd.to_datetime(
        meta["origin_timestamp"], utc=True, errors="raise"
    )
    meta = meta.reset_index(drop=True)

    if shared.shape != (10769, 24, 15) or private.shape != (10769, 24, 3):
        raise RuntimeError(f"Unexpected test tensor shapes: {shared.shape}, {private.shape}")
    expected_valid = 52214
    if int(masks.sum()) != expected_valid:
        raise RuntimeError(f"Unexpected valid test labels: {int(masks.sum())}; expected {expected_valid}")
    print("Locked test tensor contract: PASSED")

    ds = TensorDataset(shared, private, labels, masks)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available(), drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step24 = load_step24()
    source_ckpt = torch.load(step25_best, map_location="cpu", weights_only=False)
    shared_payload = torch.load(step25_shared, map_location="cpu", weights_only=False)
    step26_ckpt = torch.load(step26_best, map_location="cpu", weights_only=False)
    step28_info = json.loads(step28_manifest.read_text(encoding="utf-8"))

    if int(source_ckpt.get("seed", -1)) != seed:
        raise RuntimeError("Step-25 best checkpoint seed mismatch.")
    if int(shared_payload.get("seed", -1)) != seed:
        raise RuntimeError("Step-25 shared-encoder seed mismatch.")
    if int(step26_ckpt.get("seed", -1)) != seed:
        raise RuntimeError("Step-26 checkpoint seed mismatch.")
    if int(step26_ckpt.get("source_pretraining_seed", -1)) != seed:
        raise RuntimeError("Step-26 source-pretraining seed mismatch.")
    if int(step28_info.get("seed", -1)) != seed:
        raise RuntimeError("Step-28 manifest seed mismatch.")
    if int(step28_info.get("source_pretraining_seed", -1)) != seed:
        raise RuntimeError("Step-28 source-pretraining seed mismatch.")
    if step28_info.get("seed_chain_verified") is not True:
        raise RuntimeError("Step-28 manifest does not verify the seed chain.")

    print("Step-25 -> 26 -> 27 -> 28 provenance: VERIFIED")

    all_metrics = []
    pred_frames = []

    banner("M0 - PERSISTENCE TEST EVALUATION")
    p0, m0mask, usable = persistence_test(private, labels.numpy(), masks.numpy().astype(bool))
    rows, p0pm, y0pm = metric_rows("M0", p0, labels.numpy(), m0mask, horizons, pm25_mean, pm25_std)
    all_metrics += rows
    pred_frames.append(prediction_frame(meta, "M0", p0, labels.numpy(), m0mask, p0pm, y0pm, horizons))
    print(f"Usable test origins: {int(usable.sum()):,}/{len(usable):,}")
    print(pd.DataFrame(rows).to_string(index=False))

    # Evaluate neural baselines/ablations from their validation-selected checkpoints.
    for variant in ["M1", "M3", "M4", "M5"]:
        banner(f"{variant} - LOCKED TEST EVALUATION")
        manifest = json.loads((step29_meta / f"{variant}_manifest.json").read_text(encoding="utf-8"))
        ckpt = torch.load(step29_models / f"{variant}_best.pt", map_location="cpu", weights_only=False)

        if int(manifest["seed"]) != seed or int(ckpt["seed"]) != seed:
            raise RuntimeError(f"{variant}: Step-29 run seed mismatch.")
        if int(manifest.get("source_pretraining_seed", -1)) != seed:
            raise RuntimeError(f"{variant}: manifest source-pretraining seed mismatch.")
        if int(ckpt.get("source_pretraining_seed", -1)) != seed:
            raise RuntimeError(f"{variant}: checkpoint source-pretraining seed mismatch.")
        if manifest.get("seed_to_seed_step25_linkage_enforced") is not True:
            raise RuntimeError(f"{variant}: Step-29 seed-to-seed linkage was not enforced.")
        if Path(manifest.get("step25_best_checkpoint", "")).resolve() != step25_best.resolve():
            raise RuntimeError(f"{variant}: Step-25 best-checkpoint provenance mismatch.")
        if Path(manifest.get("step25_shared_encoder", "")).resolve() != step25_shared.resolve():
            raise RuntimeError(f"{variant}: Step-25 shared-encoder provenance mismatch.")
        if int(manifest["best_epoch"]) != int(ckpt["epoch"]):
            raise RuntimeError(f"{variant}: checkpoint is not manifest-selected best epoch.")
        if not math.isclose(float(manifest["best_validation_rmse_z"]),
                            float(ckpt["best_validation_rmse_z"]),
                            rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(f"{variant}: validation metric provenance mismatch.")
        if bool(manifest.get("target_test_accessed", False)):
            raise RuntimeError(f"{variant}: Step-29 manifest indicates prior test access.")

        # Construct the exact Step-29 architecture, then overwrite with the locked state.
        set_seed(seed)
        model, description = configure_variant(step24, source_ckpt, shared_payload, variant)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        pred, true, mask, seconds = infer_variant(model, loader, device, variant)
        rows, ppm, ypm = metric_rows(variant, pred, true, mask, horizons, pm25_mean, pm25_std)
        all_metrics += rows
        pred_frames.append(prediction_frame(meta, variant, pred, true, mask, ppm, ypm, horizons))
        print(f"Best validation epoch: {ckpt['epoch']}")
        print(f"Inference time:        {seconds:.3f} s")
        print(pd.DataFrame(rows).to_string(index=False))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # M2 is NOT re-selected here. Import its already locked corrected Step-28 metrics.
    banner("M2 - IMPORTING ALREADY LOCKED STEP-28 TEST RESULT")
    if not step28_metrics.exists():
        raise FileNotFoundError(
            f"Corrected Step-28 metrics not found: {step28_metrics}. "
            "Do not reconstruct M2 from test data in Step 30."
        )
    # Step-28 manifest was already verified above; therefore these metrics
    # belong to the same independent seed and locked M2 checkpoint chain.
    m2 = pd.read_csv(step28_metrics)
    required_cols = {"scope","horizon_hours","n_valid","rmse_pm25","mae_pm25","r2",
                     "nrmse_adaptation_std","adaptation_pm25_mean","adaptation_pm25_std"}
    if not required_cols.issubset(m2.columns):
        raise RuntimeError("Step-28 metrics schema mismatch.")
    m2.insert(0, "variant", "M2")
    all_metrics += m2[list(["variant"] + list(required_cols))].to_dict("records")
    print(m2.to_string(index=False))

    metrics = pd.DataFrame(all_metrics)
    # Stable publication-friendly ordering.
    order = {"M0":0,"M1":1,"M2":2,"M3":3,"M4":4,"M5":5}
    metrics["_order"] = metrics["variant"].map(order)
    metrics["_h"] = metrics["horizon_hours"].astype(str).map(
        {"all":0,"1":1,"3":3,"6":6,"12":12,"24":24}).fillna(999)
    metrics = metrics.sort_values(["_order","_h"]).drop(columns=["_order","_h"])

    metrics_path = out_dir / "step30_test_metrics_all_models.csv"
    metrics.to_csv(metrics_path, index=False)

    # M0/M1/M3/M4/M5 prediction-level results. M2 remains in the authoritative
    # corrected Step-28 prediction CSV, avoiding an unnecessary second test pass.
    preds = pd.concat(pred_frames, ignore_index=True)
    preds_path = out_dir / "step30_test_predictions_baselines_ablations.csv"
    preds.to_csv(preds_path, index=False)

    aggregate = metrics[metrics["scope"].astype(str) == "aggregate"].copy()
    aggregate_path = out_dir / "step30_aggregate_comparison.csv"
    aggregate.to_csv(aggregate_path, index=False)

    manifest = {
        "step": 30,
        "seed": seed,
        "source_pretraining_seed": seed,
        "end_to_end_seed_chain_verified": True,
        "step25_directory": str(step25_dir),
        "step25_best_checkpoint": str(step25_best),
        "step25_shared_encoder": str(step25_shared),
        "step26_best_checkpoint": str(step26_best),
        "step27_audit": str(step27_audit),
        "step28_manifest": str(step28_manifest),
        "step29_model_directory": str(step29_models),
        "evaluation": "locked one-shot baseline/ablation target-test evaluation",
        "training_performed": False,
        "parameter_updates": False,
        "model_selection_from_test": False,
        "timestamp_materialization": "explicit datetime64[ns], zero-missing fail-closed audit",
        "history_length_hours": history,
        "forecast_horizons_hours": horizons,
        "test_origins": len(ds),
        "valid_test_labels": int(masks.sum()),
        "m2_source": str(step28_metrics),
        "test_access_policy": {
            "M0": "evaluated once in Step 30",
            "M1": "validation-selected Step-29 checkpoint evaluated once in Step 30",
            "M2": "imported from corrected locked Step-28 evaluation; not rerun",
            "M3": "validation-selected Step-29 checkpoint evaluated once in Step 30",
            "M4": "validation-selected Step-29 checkpoint evaluated once in Step 30",
            "M5": "validation-selected Step-29 checkpoint evaluated once in Step 30",
        },
        "input_sha256": {
            "protocol": sha256_file(PROTOCOL_FILE),
            "normalization": sha256_file(TARGET_NORM_FILE),
            "test_index": sha256_file(test_index),
            "step25_best": sha256_file(step25_best),
            "step25_shared": sha256_file(step25_shared),
            "step27_audit": sha256_file(step27_audit),
            "step26_best": sha256_file(step26_best),
            "step28_metrics": sha256_file(step28_metrics),
            "step28_manifest": sha256_file(step28_manifest),
            **{f"{v}_best": sha256_file(step29_models / f"{v}_best.pt")
               for v in ["M1","M3","M4","M5"]},
        },
    }
    (out_dir / "step30_test_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    banner("STEP 30 - AGGREGATE TEST COMPARISON")
    cols = ["variant","n_valid","rmse_pm25","mae_pm25","r2","nrmse_adaptation_std"]
    print(aggregate[cols].to_string(index=False))

    banner("STEP 30 COMPLETE")
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {preds_path}")
    print(f"Aggregate:   {aggregate_path}")
    print(f"Manifest:    {out_dir / 'step30_test_manifest.json'}")
    print()
    print("Training performed:       NO")
    print("Parameter updates:         NO")
    print("Test-based selection:      NO")
    print("M2 test inference rerun:   NO")
    print(f"Independent run seed:      {seed}")
    print("End-to-end seed chain:     VERIFIED")
    print(f"Seed-{seed} Step-30 evaluation complete.")


if __name__ == "__main__":
    main()
