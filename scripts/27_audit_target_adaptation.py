from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"

PROTOCOL_FILE = META / "forecasting_protocol.json"
TARGET_NORM_FILE = META / "opensensemap_adaptation_normalization_statistics.csv"

# Step-25 artifacts are resolved per independent seed inside main().

ADAPTATION_INDEX = (
    ROOT / "data" / "processed" / "opensensemap"
    / "target_window_index_adaptation.parquet"
)
VALIDATION_INDEX = (
    ROOT / "data" / "processed" / "opensensemap"
    / "target_window_index_validation.parquet"
)

# Deliberately no target-test path is defined in Step 27.

EXPECTED_HORIZONS = [1, 3, 6, 12, 24]
EXPECTED_HISTORY = 24
EXPECTED_SHARED = ["temperature", "relative_humidity", "pressure"]
EXPECTED_TARGET_PRIVATE = "pm25"

TOL = 1e-8


def banner(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def passed(text):
    print(f"[PASS] {text}")


def failed(text):
    print(f"[FAIL] {text}")


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensors_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if a[k].shape != b[k].shape:
            return False
        if a[k].dtype != b[k].dtype:
            return False
        if not torch.equal(a[k].detach().cpu(), b[k].detach().cpu()):
            return False
    return True


def extract_prefix(state, prefix):
    return {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix)
    }


def any_tensor_changed(a, b):
    common = set(a) & set(b)
    if not common:
        return True
    for k in common:
        if a[k].shape != b[k].shape:
            return True
        if not torch.equal(a[k].detach().cpu(), b[k].detach().cpu()):
            return True
    return False


def finite_columns(df, columns):
    bad = []
    for c in columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isna().any():
            bad.append(c)
        elif not x.map(math.isfinite).all():
            bad.append(c)
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    # Strict seed-to-seed provenance:
    # Step-27 seed N audits Step-26 seed N against Step-25 seed N only.
    step25_dir = ROOT / "models" / "source_pretraining" / f"seed_{seed}"
    step25_best = step25_dir / "bdg2_source_pretraining_best.pt"
    step25_shared = step25_dir / "bdg2_pretrained_shared_encoder.pt"

    run_dir = ROOT / "models" / "target_adaptation" / f"seed_{seed}"
    result_dir = META / "target_adaptation" / f"seed_{seed}"

    best_file = run_dir / "opensensemap_target_adaptation_best.pt"
    last_file = run_dir / "opensensemap_target_adaptation_last.pt"
    history_file = result_dir / "opensensemap_target_adaptation_history.csv"
    manifest_file = result_dir / "opensensemap_target_adaptation_manifest.json"
    audit_file = result_dir / "opensensemap_target_adaptation_audit.json"

    banner("STEP 27 - TARGET ADAPTATION INTEGRITY AUDIT")
    print(f"Seed: {seed}")
    print(f"Matching Step-25 run:      {step25_dir}")
    print("Seed-to-seed linkage:      ENFORCED")
    print("Training performed:       NO")
    print("Target test path defined: NO")
    print("Target test accessed:     NO")

    required = [
        PROTOCOL_FILE,
        TARGET_NORM_FILE,
        step25_best,
        step25_shared,
        ADAPTATION_INDEX,
        VALIDATION_INDEX,
        best_file,
        last_file,
        history_file,
        manifest_file,
    ]
    for p in required:
        require(p)
    passed("All required non-test artifacts found.")

    failures = []
    checks = {}

    def check(name, condition, detail=""):
        condition = bool(condition)
        checks[name] = {"passed": condition, "detail": str(detail)}
        if condition:
            passed(name + (f" -- {detail}" if detail else ""))
        else:
            failed(name + (f" -- {detail}" if detail else ""))
            failures.append(name)

    banner("FROZEN FORECASTING CONTRACT")
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    history_len = int(
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
    shared = list(protocol["shared_variables"])
    target_private = protocol["target_private_variable"]

    check("History length is frozen at 24 hours", history_len == EXPECTED_HISTORY, history_len)
    check("Forecast horizons are frozen", horizons == EXPECTED_HORIZONS, horizons)
    check("Shared variables are frozen", shared == EXPECTED_SHARED, shared)
    check("Target-private variable is PM2.5", target_private == EXPECTED_TARGET_PRIVATE, target_private)

    causality = protocol.get("causality", {})
    for key in [
        "future_observations_in_inputs",
        "future_based_interpolation",
        "target_test_for_normalization",
        "target_test_for_adaptation",
        "target_test_for_model_selection",
        "target_test_for_early_stopping",
        "target_test_for_threshold_selection",
        "cross_partition_windows",
    ]:
        check(f"Causality rule {key}=false", causality.get(key) is False, causality.get(key))

    banner("LOADING STEP-26 ARTIFACTS")
    history = pd.read_csv(history_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    best = torch.load(best_file, map_location="cpu", weights_only=False)
    last = torch.load(last_file, map_location="cpu", weights_only=False)
    source_best = torch.load(step25_best, map_location="cpu", weights_only=False)
    source_shared = torch.load(step25_shared, map_location="cpu", weights_only=False)

    print(f"History rows: {len(history):,}")
    print(f"Best epoch checkpoint: {best.get('epoch')}")
    print(f"Manifest best epoch:   {manifest.get('best_epoch')}")
    print(f"Manifest epochs:       {manifest.get('epochs_completed')}")

    banner("HISTORY / CHECKPOINT CONSISTENCY")
    required_history_cols = {
        "seed",
        "epoch",
        "train_rmse_z",
        "train_mae_z",
        "validation_rmse_z",
        "validation_mae_z",
    }
    check(
        "History contains required columns",
        required_history_cols.issubset(history.columns),
        sorted(required_history_cols),
    )

    if required_history_cols.issubset(history.columns):
        metric_cols = [
            c for c in history.columns
            if c.startswith("train_") or c.startswith("validation_")
        ]
        numeric_metric_cols = [
            c for c in metric_cols
            if c not in {"train_valid_labels", "validation_valid_labels"}
        ]
        bad = finite_columns(history, numeric_metric_cols)
        check("All recorded training/validation metrics are finite", len(bad) == 0, bad or "all finite")

        best_row_i = pd.to_numeric(history["validation_rmse_z"], errors="coerce").idxmin()
        csv_best_epoch = int(history.loc[best_row_i, "epoch"])
        csv_best_rmse = float(history.loc[best_row_i, "validation_rmse_z"])

        check(
            "Best checkpoint epoch equals CSV minimum epoch",
            int(best.get("epoch", -1)) == csv_best_epoch,
            f"checkpoint={best.get('epoch')}, csv={csv_best_epoch}",
        )
        check(
            "Manifest best epoch equals CSV minimum epoch",
            int(manifest.get("best_epoch", -1)) == csv_best_epoch,
            f"manifest={manifest.get('best_epoch')}, csv={csv_best_epoch}",
        )
        check(
            "Best checkpoint RMSE equals CSV minimum",
            abs(float(best.get("best_validation_rmse_z", math.inf)) - csv_best_rmse) <= TOL,
            f"checkpoint={best.get('best_validation_rmse_z')}, csv={csv_best_rmse}",
        )
        check(
            "Manifest RMSE equals CSV minimum",
            abs(float(manifest.get("best_validation_rmse_z", math.inf)) - csv_best_rmse) <= TOL,
            f"manifest={manifest.get('best_validation_rmse_z')}, csv={csv_best_rmse}",
        )

    check(
        "Manifest epoch count equals history rows",
        int(manifest.get("epochs_completed", -1)) == len(history),
        f"manifest={manifest.get('epochs_completed')}, csv={len(history)}",
    )
    check(
        "Seed is consistent across Step-26 artifacts",
        int(manifest.get("seed", -1)) == seed
        and int(best.get("seed", -1)) == seed
        and int(last.get("seed", -1)) == seed
        and (history["seed"].astype(int) == seed).all(),
        seed,
    )
    check(
        "Step-25 best checkpoint seed matches audit seed",
        int(source_best.get("seed", -1)) == seed,
        f"step25={source_best.get('seed')}, audit={seed}",
    )
    check(
        "Step-25 shared-encoder seed matches audit seed",
        int(source_shared.get("seed", -1)) == seed,
        f"step25={source_shared.get('seed')}, audit={seed}",
    )
    check(
        "Step-26 manifest records matching source-pretraining seed",
        int(manifest.get("source_pretraining_seed", -1)) == seed,
        f"source_pretraining_seed={manifest.get('source_pretraining_seed')}",
    )
    check(
        "Step-26 BEST checkpoint records matching source-pretraining seed",
        int(best.get("source_pretraining_seed", -1)) == seed,
        f"source_pretraining_seed={best.get('source_pretraining_seed')}",
    )
    check(
        "Step-26 LAST checkpoint records matching source-pretraining seed",
        int(last.get("source_pretraining_seed", -1)) == seed,
        f"source_pretraining_seed={last.get('source_pretraining_seed')}",
    )

    banner("TRANSFER / PARAMETER INTEGRITY")
    best_state = best["model_state_dict"]
    source_state = source_best["model_state_dict"]

    step26_shared = extract_prefix(best_state, "shared_encoder.")
    step25_shared_from_best = extract_prefix(source_state, "shared_encoder.")
    exported_shared = source_shared["state_dict"]

    check(
        "Step-26 shared encoder equals exported Step-25 shared encoder bit-for-bit",
        tensors_equal(step26_shared, exported_shared),
    )
    check(
        "Step-26 shared encoder equals Step-25 best checkpoint bit-for-bit",
        tensors_equal(step26_shared, step25_shared_from_best),
    )

    # IMPORTANT:
    # Step 26 instantiates a fresh Step-24 model, loads ONLY the pretrained
    # shared encoder into model.shared_encoder, and copies the Step-25
    # source_private_encoder into model.target_private_encoder.
    #
    # The unused source_private_encoder/source_fusion/source_head branches in
    # the Step-26 model are therefore NOT expected to equal Step 25. Comparing
    # them with Step 25 is an invalid audit assertion and would create a false
    # failure. Their scientific irrelevance is instead established by the
    # Step-26 transfer contract: they are frozen and absent from the target
    # adaptation forward path.
    check(
        "Step-26 transfer contract identifies shared encoder as frozen",
        manifest.get("transfer", {}).get("shared_encoder")
        == f"Step-25 seed {seed} pretrained, frozen",
        manifest.get("transfer", {}).get("shared_encoder"),
    )
    check(
        "Step-26 transfer contract excludes source energy head transfer",
        manifest.get("transfer", {}).get("source_energy_head_transferred") is False,
        manifest.get("transfer", {}).get("source_energy_head_transferred"),
    )

    source_private = extract_prefix(source_state, "source_private_encoder.")
    target_private_best = extract_prefix(best_state, "target_private_encoder.")
    check(
        "Target-private encoder has structurally compatible source-private initialization",
        set(source_private.keys()) == set(target_private_best.keys())
        and all(source_private[k].shape == target_private_best[k].shape for k in source_private),
    )
    check(
        "Target-private encoder changed during adaptation",
        any_tensor_changed(source_private, target_private_best),
        "expected after target-domain gradient updates",
    )

    # The shared encoder must remain frozen through the LAST checkpoint too.
    # We deliberately do not compare unused source-only branches against
    # Step 25 because Step 26 never loaded Step-25 weights into those branches.
    last_state = last["model_state_dict"]
    check(
        "Last-checkpoint shared encoder is also unchanged",
        tensors_equal(extract_prefix(last_state, "shared_encoder."), exported_shared),
    )

    # Stronger retrospective check available from the existing artifacts:
    # unused source-only branches must be identical between Step-26 BEST and
    # Step-26 LAST. This proves they did not change over the saved adaptation
    # trajectory, without incorrectly requiring equality to Step 25.
    for component in ["source_private_encoder", "source_fusion", "source_head"]:
        check(
            f"Unused {component} identical in Step-26 BEST and LAST",
            tensors_equal(
                extract_prefix(best_state, component + "."),
                extract_prefix(last_state, component + "."),
            ),
        )

    banner("PARTITION / NORMALIZATION AUDIT")
    adapt_idx = pd.read_parquet(
        ADAPTATION_INDEX,
        columns=["partition", "history_start_timestamp", "origin_timestamp"],
    )
    valid_idx = pd.read_parquet(
        VALIDATION_INDEX,
        columns=["partition", "history_start_timestamp", "origin_timestamp"],
    )

    check(
        "Adaptation index contains adaptation partition only",
        set(adapt_idx["partition"].astype(str).unique()) == {"adaptation"},
    )
    check(
        "Validation index contains validation partition only",
        set(valid_idx["partition"].astype(str).unique()) == {"validation"},
    )

    adapt_origin_max = pd.to_datetime(adapt_idx["origin_timestamp"], utc=True).max()
    valid_history_min = pd.to_datetime(valid_idx["history_start_timestamp"], utc=True).min()
    check(
        "Adaptation and validation histories are temporally ordered",
        adapt_origin_max < valid_history_min,
        f"adaptation max origin={adapt_origin_max}; validation min history={valid_history_min}",
    )

    norm = pd.read_csv(TARGET_NORM_FILE)
    check(
        "Target normalization contains required variables",
        set(EXPECTED_SHARED + [EXPECTED_TARGET_PRIVATE]).issubset(set(norm["variable"].astype(str))),
    )

    norm_protocol = protocol.get("normalization_protocol", {}).get("target", {})
    check(
        "Target normalization fit partition is adaptation only",
        norm_protocol.get("fit_partition") == "adaptation only",
        norm_protocol.get("fit_partition"),
    )
    check(
        "Target validation not used for normalization fitting",
        norm_protocol.get("validation_used_for_fit") is False,
    )
    check(
        "Target test not used for normalization fitting",
        norm_protocol.get("test_used_for_fit") is False,
    )

    banner("MANIFEST PROVENANCE")
    hashes = manifest.get("sha256", {})
    hash_targets = {
        "protocol": PROTOCOL_FILE,
        "normalization": TARGET_NORM_FILE,
        "adaptation_index": ADAPTATION_INDEX,
        "validation_index": VALIDATION_INDEX,
        "step25_best": step25_best,
        "step25_shared": step25_shared,
    }
    for key, path in hash_targets.items():
        if key in hashes:
            actual = sha256_file(path)
            check(
                f"SHA-256 provenance matches: {key}",
                hashes[key] == actual,
            )
        else:
            check(f"SHA-256 provenance present: {key}", False, "missing from manifest")

    transfer_manifest = manifest.get("transfer", {})
    check(
        "Manifest Step-25 source directory matches seed-specific path",
        Path(transfer_manifest.get("source_pretraining_directory", "")).resolve()
        == step25_dir.resolve(),
        transfer_manifest.get("source_pretraining_directory"),
    )
    check(
        "Manifest Step-25 best path matches seed-specific artifact",
        Path(transfer_manifest.get("step25_best_checkpoint", "")).resolve()
        == step25_best.resolve(),
        transfer_manifest.get("step25_best_checkpoint"),
    )
    check(
        "Manifest Step-25 shared path matches seed-specific artifact",
        Path(transfer_manifest.get("step25_shared_encoder", "")).resolve()
        == step25_shared.resolve(),
        transfer_manifest.get("step25_shared_encoder"),
    )
    check(
        "Manifest says seed-to-seed transfer linkage was enforced",
        manifest.get("seed_to_seed_transfer_linkage_enforced") is True,
        manifest.get("seed_to_seed_transfer_linkage_enforced"),
    )

    leakage = manifest.get("leakage", {})
    check(
        "Manifest says target test was not defined",
        leakage.get("target_test_defined") is False,
        leakage.get("target_test_defined"),
    )
    check(
        "Manifest says target test was not accessed",
        leakage.get("target_test_accessed") is False,
        leakage.get("target_test_accessed"),
    )
    check(
        "Manifest says normalization was not refitted",
        leakage.get("normalization_refitted") is False,
        leakage.get("normalization_refitted"),
    )
    check(
        "Validation used for checkpoint selection",
        leakage.get("validation_checkpoint_selection") is True,
        leakage.get("validation_checkpoint_selection"),
    )
    check(
        "Validation did not receive gradients",
        leakage.get("validation_gradients") is False,
        leakage.get("validation_gradients"),
    )

    banner("WRITING STEP-27 AUDIT")
    audit = {
        "step": 27,
        "seed": seed,
        "source_pretraining_seed": seed,
        "seed_to_seed_transfer_linkage_audited": True,
        "status": "PASSED" if not failures else "FAILED",
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "target_test_defined_in_step27": False,
        "target_test_accessed_in_step27": False,
        "artifacts": {
            "step25_best": str(step25_best),
            "step25_shared": str(step25_shared),
            "step26_best": str(best_file),
            "step26_last": str(last_file),
            "step26_history": str(history_file),
            "step26_manifest": str(manifest_file),
        },
    }
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"Saved:\n  {audit_file}")

    banner("STEP 27 AUDIT RESULT")
    if failures:
        print(f"STATUS: FAILED ({len(failures)} failed checks)")
        for x in failures:
            print(f"  - {x}")
        print()
        print("Do NOT proceed to target-test evaluation.")
        raise SystemExit(1)

    print("STATUS: PASSED")
    print(f"Independent run seed:                  {seed}")
    print("Step-25 -> Step-26 seed linkage:       VERIFIED")
    print("Step-25 -> Step-26 transfer integrity: VERIFIED")
    print("Frozen shared representation:          VERIFIED")
    print("Unused source-only branches stable:     VERIFIED")
    print("Target adaptation artifacts:           CONSISTENT")
    print("Normalization boundary:                VERIFIED")
    print("Adaptation/validation boundary:        VERIFIED")
    print("Target test accessed:                  NO")
    print()
    print("Step 27 complete.")
    print("The pipeline is cleared for Step 28 target-test evaluation.")


if __name__ == "__main__":
    main()
