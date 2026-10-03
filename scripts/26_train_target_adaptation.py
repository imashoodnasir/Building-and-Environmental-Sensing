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
# Step-25 artifacts are resolved inside main() from the same independent run seed.

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


def metrics(pred, target, mask, horizons):
    e = pred.float() - target.float()
    sq = (e * e) * mask.float()
    ab = e.abs() * mask.float()
    sse = sq.sum().item()
    sae = ab.sum().item()
    n = mask.sum().item()
    hsse = sq.sum(0).cpu().numpy()
    hsae = ab.sum(0).cpu().numpy()
    hn = mask.sum(0).cpu().numpy()
    return sse, sae, n, hsse, hsae, hn


def finish_metrics(sse, sae, n, hsse, hsae, hn, horizons, seconds):
    if n <= 0:
        raise RuntimeError("No valid labels.")
    hrmse = np.sqrt(np.divide(hsse, hn, out=np.full_like(hsse, np.nan), where=hn > 0))
    hmae = np.divide(hsae, hn, out=np.full_like(hsae, np.nan), where=hn > 0)
    return {
        "rmse": math.sqrt(sse / n),
        "mae": sae / n,
        "valid_labels": int(n),
        "seconds": seconds,
        "horizon_rmse": {str(h): float(hrmse[i]) for i, h in enumerate(horizons)},
        "horizon_mae": {str(h): float(hmae[i]) for i, h in enumerate(horizons)},
        "horizon_n": {str(h): int(hn[i]) for i, h in enumerate(horizons)},
    }


def run_epoch(model, loader, device, horizons, optimizer=None, scaler=None, epoch=1):
    training = optimizer is not None
    if training:
        model.train()
        model.shared_encoder.eval()
        model.source_private_encoder.eval()
        model.source_fusion.eval()
        model.source_head.eval()
        desc = f"Epoch {epoch:02d} TRAIN"
    else:
        model.eval()
        desc = f"Epoch {epoch:02d} VALID"

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

            # Shared encoder is frozen in Step 26.
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    shared_rep, _ = model.shared_encoder(shared)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                private_rep = model.target_private_encoder(private)
                fused, _ = model.target_fusion(shared_rep.detach(), private_rep)
                prediction = model.target_head(fused)
                loss = (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            with torch.no_grad():
                a = metrics(prediction, target, mask, horizons)
                sse += a[0]; sae += a[1]; n += a[2]
                hsse += a[3]; hsae += a[4]; hn += a[5]
                bar.set_postfix(
                    rmse_z=f"{math.sqrt(sse/max(n,1)):.5f}",
                    valid=f"{int(n):,}",
                    refresh=False,
                )

    return finish_metrics(sse, sae, n, hsse, hsae, hn, horizons, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    seed = args.seed
    set_seed(seed)

    # Strict seed-to-seed transfer linkage:
    # Step-26 seed N may only consume Step-25 seed N.
    step25_dir = ROOT / "models" / "source_pretraining" / f"seed_{seed}"
    step25_best = step25_dir / "bdg2_source_pretraining_best.pt"
    step25_shared = step25_dir / "bdg2_pretrained_shared_encoder.pt"

    banner("STEP 26 - MULTI-SEED OPENSENSEMAP TARGET ADAPTATION")
    required = [
        PROTOCOL_FILE, TARGET_NORM_FILE, TARGET_DATA_FILE, ADAPTATION_INDEX,
        VALIDATION_INDEX, STEP24_PATH, step25_best, step25_shared
    ]
    for p in required:
        require_file(p)

    print(f"Independent run seed:                   {seed}")
    print(f"Matching Step-25 run:                   {step25_dir}")
    print("Seed-to-seed Step-25 linkage:           ENFORCED")
    print("Optimization: target windows are materialized once in RAM.")
    print("Timestamp representation: explicit datetime64[ns].")
    print("Per-batch Pandas/timestamp processing: NO")
    print("Target TEST path defined/accessed:     NO")

    with open(PROTOCOL_FILE, encoding="utf-8") as f:
        protocol = json.load(f)
    history = int(protocol.get("history_length_hours", protocol["forecasting_rules"]["history_length_hours"]))
    horizons = list(protocol.get("forecast_horizons_hours", protocol["forecasting_rules"]["forecast_horizons_hours"]))
    shared_vars = list(protocol["shared_variables"])
    calendar = list(protocol["calendar_features"])
    if history != 24 or horizons != [1, 3, 6, 12, 24]:
        raise RuntimeError("Frozen forecasting contract changed.")
    if protocol["target_private_variable"] != "pm25":
        raise RuntimeError("Target-private variable is not pm25.")

    norm = read_norm(TARGET_NORM_FILE)
    for v in shared_vars + ["pm25"]:
        if v not in norm:
            raise RuntimeError(f"Missing normalization for {v}")

    boxes = prepare_hourly_arrays(TARGET_DATA_FILE, shared_vars, calendar, norm)
    tprep = time.time()
    adapt = materialize_partition(ADAPTATION_INDEX, "adaptation", boxes, norm, horizons, history)
    valid = materialize_partition(VALIDATION_INDEX, "validation", boxes, norm, horizons, history)
    prep_seconds = time.time() - tprep
    del boxes

    banner("MATERIALIZED TENSOR AUDIT")
    assert adapt[0].shape == (71660, 24, 15)
    assert adapt[1].shape == (71660, 24, 3)
    assert valid[0].shape == (23936, 24, 15)
    assert valid[1].shape == (23936, 24, 3)
    print("Frozen Step-23 tensor shapes: PASSED")
    print(f"Materialization time: {prep_seconds:.2f} s")

    step24 = load_step24()
    model = step24.HeterogeneousTransferForecaster()

    source_ckpt = torch.load(step25_best, map_location="cpu", weights_only=False)
    shared_payload = torch.load(step25_shared, map_location="cpu", weights_only=False)

    # Fail closed if Step-25 provenance does not match this Step-26 run.
    source_seed = source_ckpt.get("seed")
    shared_seed = shared_payload.get("seed")
    if source_seed is None:
        raise RuntimeError("Step-25 best checkpoint does not record its seed.")
    if shared_seed is None:
        raise RuntimeError("Step-25 shared-encoder artifact does not record its seed.")
    if int(source_seed) != seed:
        raise RuntimeError(
            f"Seed mismatch: Step-26 requested seed {seed}, "
            f"but Step-25 best checkpoint records seed {source_seed}."
        )
    if int(shared_seed) != seed:
        raise RuntimeError(
            f"Seed mismatch: Step-26 requested seed {seed}, "
            f"but Step-25 shared encoder records seed {shared_seed}."
        )
    print(f"Step-25 checkpoint seed:                {source_seed}")
    print(f"Step-25 shared-encoder seed:            {shared_seed}")
    print("Step-25 -> Step-26 seed linkage:        PASSED")
    model.shared_encoder.load_state_dict(shared_payload["state_dict"], strict=True)

    prefix = "source_private_encoder."
    sp = {k[len(prefix):]: v for k, v in source_ckpt["model_state_dict"].items() if k.startswith(prefix)}
    if not sp:
        raise RuntimeError("Source-private encoder not found in Step-25 checkpoint.")
    model.target_private_encoder.load_state_dict(sp, strict=True)

    for p in model.parameters():
        p.requires_grad = False
    for module in (model.target_private_encoder, model.target_fusion, model.target_head):
        for p in module.parameters():
            p.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    banner("MODEL / DEVICE")
    print(f"Seed:                 {seed}")
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU:                  {torch.cuda.get_device_name(0)}")
        print(f"VRAM:                 {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print(f"Batch size:            {BATCH_SIZE}")
    print(f"Trainable parameters:  {sum(p.numel() for p in trainable):,}")
    print("Shared encoder:        PRETRAINED + FROZEN")
    print("Target private:        SOURCE-PRIVATE INITIALIZATION + TRAINABLE")
    print("Target fusion/head:    TRAINABLE")
    print("Energy head transfer:  NO")

    train_ds = TensorDataset(*adapt)
    val_ds = TensorDataset(*valid)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False
    )
    print(f"Training batches:      {len(train_loader)}")
    print(f"Validation batches:    {len(val_loader)}")

    try:
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                      fused=(device.type == "cuda"))
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    amp = USE_AMP and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None
    print(f"Mixed precision:       {'ENABLED' if amp else 'DISABLED'}")

    run_dir = ROOT / "models" / "target_adaptation" / f"seed_{seed}"
    result_dir = META / "target_adaptation" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "opensensemap_target_adaptation_best.pt"
    last_path = run_dir / "opensensemap_target_adaptation_last.pt"
    hist_path = result_dir / "opensensemap_target_adaptation_history.csv"
    manifest_path = result_dir / "opensensemap_target_adaptation_manifest.json"

    banner("TARGET ADAPTATION")
    best_rmse = math.inf
    best_epoch = None
    stale = 0
    rows = []
    train_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = run_epoch(model, train_loader, device, horizons, optimizer, scaler, epoch)
        va = run_epoch(model, val_loader, device, horizons, None, None, epoch)

        print(f"\nEpoch {epoch:02d}: train RMSE(z)={tr['rmse']:.6f} | "
              f"val RMSE(z)={va['rmse']:.6f} | val MAE(z)={va['mae']:.6f}")
        for h in horizons:
            print(f"  h={h:2d}: RMSE(z)={va['horizon_rmse'][str(h)]:.6f} | "
                  f"MAE(z)={va['horizon_mae'][str(h)]:.6f} | n={va['horizon_n'][str(h)]:,}")

        row = {
            "seed": seed, "epoch": epoch,
            "train_rmse_z": tr["rmse"], "train_mae_z": tr["mae"],
            "validation_rmse_z": va["rmse"], "validation_mae_z": va["mae"],
            "train_valid_labels": tr["valid_labels"],
            "validation_valid_labels": va["valid_labels"],
            "train_seconds": tr["seconds"], "validation_seconds": va["seconds"],
        }
        for h in horizons:
            row[f"train_rmse_z_h{h}"] = tr["horizon_rmse"][str(h)]
            row[f"train_mae_z_h{h}"] = tr["horizon_mae"][str(h)]
            row[f"validation_rmse_z_h{h}"] = va["horizon_rmse"][str(h)]
            row[f"validation_mae_z_h{h}"] = va["horizon_mae"][str(h)]
        rows.append(row)
        pd.DataFrame(rows).to_csv(hist_path, index=False)

        if va["rmse"] < best_rmse:
            best_rmse = va["rmse"]
            best_epoch = epoch
            stale = 0
            atomic_torch_save({
                "step": 26, "optimized": True, "seed": seed,
                "source_pretraining_seed": seed, "epoch": epoch,
                "best_validation_rmse_z": best_rmse,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history_length_hours": history,
                "forecast_horizons_hours": horizons,
                "shared_encoder_frozen": True,
                "target_private_initialized_from": "Step-25 source_private_encoder",
                "source_prediction_head_transferred": False,
                "target_test_accessed": False,
            }, best_path)
            print(f"BEST CHECKPOINT UPDATED: epoch={epoch}, val RMSE(z)={best_rmse:.6f}")
        else:
            stale += 1
            print(f"No improvement: {stale}/{EARLY_STOPPING_PATIENCE}")

        atomic_torch_save({
            "step": 26, "optimized": True, "seed": seed,
            "source_pretraining_seed": seed, "epoch": epoch,
            "best_epoch": best_epoch, "best_validation_rmse_z": best_rmse,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "target_test_accessed": False,
        }, last_path)

        if stale >= EARLY_STOPPING_PATIENCE:
            print("EARLY STOPPING TRIGGERED")
            break

    banner("VERIFYING BEST MODEL")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    now = model.shared_encoder.state_dict()
    for k, original in shared_payload["state_dict"].items():
        if k not in now or not torch.equal(original.cpu(), now[k].detach().cpu()):
            raise RuntimeError(f"Frozen shared encoder changed: {k}")
    print("Frozen shared encoder unchanged: PASSED")

    total_seconds = time.time() - train_start
    manifest = {
        "step": 26,
        "optimized_materialized_target_tensors": True,
        "seed": seed,
        "source_pretraining_seed": seed,
        "seed_to_seed_transfer_linkage_enforced": True,
        "planned_independent_seeds": list(range(42, 62)),
        "history_length_hours": history,
        "forecast_horizons_hours": horizons,
        "adaptation_origins": len(train_ds),
        "validation_origins": len(val_ds),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": int(best["epoch"]),
        "best_validation_rmse_z": float(best["best_validation_rmse_z"]),
        "epochs_completed": len(rows),
        "materialization_seconds": prep_seconds,
        "training_seconds": total_seconds,
        "transfer": {
            "shared_encoder": f"Step-25 seed {seed} pretrained, frozen",
            "source_pretraining_seed": seed,
            "source_pretraining_directory": str(step25_dir),
            "step25_best_checkpoint": str(step25_best),
            "step25_shared_encoder": str(step25_shared),
            "target_private_encoder": "initialized from Step-25 source-private encoder, trainable",
            "target_fusion": "trainable",
            "target_head": "trainable",
            "source_energy_head_transferred": False,
        },
        "leakage": {
            "adaptation_gradients": True,
            "validation_gradients": False,
            "validation_checkpoint_selection": True,
            "target_test_defined": False,
            "target_test_accessed": False,
            "normalization_refitted": False,
        },
        "sha256": {
            "protocol": sha256_file(PROTOCOL_FILE),
            "normalization": sha256_file(TARGET_NORM_FILE),
            "adaptation_index": sha256_file(ADAPTATION_INDEX),
            "validation_index": sha256_file(VALIDATION_INDEX),
            "step24": sha256_file(STEP24_PATH),
            "step25_best": sha256_file(step25_best),
            "step25_shared": sha256_file(step25_shared),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    banner("STEP 26 COMPLETE")
    print(f"Seed:                    {seed}")
    print(f"Source pretraining seed: {seed}")
    print("Seed linkage:            VERIFIED")
    print(f"Epochs completed:        {len(rows)}")
    print(f"Best epoch:              {best['epoch']}")
    print(f"Best validation RMSE(z): {best['best_validation_rmse_z']:.6f}")
    print(f"Tensor preparation:      {prep_seconds/60:.2f} min")
    print(f"Training time:           {total_seconds/60:.2f} min")
    print("\nGenerated:")
    print(f"  {best_path}")
    print(f"  {last_path}")
    print(f"  {hist_path}")
    print(f"  {manifest_path}")
    print("\nTarget TEST partition remains untouched.")


if __name__ == "__main__":
    main()
