from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOG_ROOT = ROOT / "logs" / "20_independent_runs"

SEEDS = list(range(42, 62))

PIPELINE = [
    (25, "25_train_source_pretraining.py"),
    (26, "26_train_target_adaptation.py"),
    (27, "27_audit_target_adaptation.py"),
    (28, "28_evaluate_target_test.py"),
    (29, "29_train_baselines_ablations.py"),
    (30, "30_evaluate_baselines_ablations_test.py"),
    (31, "31_analyze_target_test_errors.py"),
]


def banner(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def completion_artifact(step, seed):
    """
    Final artifact whose existence indicates that a step reached completion.

    This is only used for --resume.
    The first production run should normally be launched WITHOUT --resume
    because seed-42 currently contains smoke-test artifacts.
    """
    paths = {
        25:
            ROOT / "data" / "metadata" /
            "source_pretraining" / f"seed_{seed}" /
            "bdg2_source_pretraining_manifest.json",

        26:
            ROOT / "data" / "metadata" /
            "target_adaptation" / f"seed_{seed}" /
            "opensensemap_target_adaptation_manifest.json",

        27:
            ROOT / "data" / "metadata" /
            "target_adaptation" / f"seed_{seed}" /
            "opensensemap_target_adaptation_audit.json",

        28:
            ROOT / "data" / "metadata" /
            "target_test" / f"seed_{seed}" /
            "opensensemap_target_test_manifest.json",

        29:
            ROOT / "data" / "metadata" /
            "baselines_ablations" / f"seed_{seed}" /
            "step29_validation_summary.csv",

        30:
            ROOT / "data" / "metadata" /
            "baseline_ablation_test" / f"seed_{seed}" /
            "step30_test_manifest.json",

        31:
            ROOT / "data" / "metadata" /
            "target_test_analysis" / f"seed_{seed}" /
            "step31_analysis_manifest.json",
    }

    return paths[step]


def run_step(step, script_name, seed, log_file):
    script = SCRIPTS / script_name

    if not script.exists():
        raise FileNotFoundError(
            f"Missing pipeline script: {script}"
        )

    command = [
        sys.executable,
        str(script),
        "--seed",
        str(seed),
    ]

    banner(
        f"SEED {seed} | STEP {step} | {script_name}"
    )

    print("Command:")
    print(" ".join(command))
    print(f"Log: {log_file}")
    print()

    start = time.time()

    with log_file.open(
        "w",
        encoding="utf-8",
        buffering=1
    ) as log:

        log.write(
            f"Seed: {seed}\n"
            f"Step: {step}\n"
            f"Script: {script_name}\n"
            f"Started: {datetime.now().isoformat()}\n"
            f"Command: {' '.join(command)}\n"
            + "=" * 110
            + "\n\n"
        )

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log.write(line)

        return_code = process.wait()

        elapsed = time.time() - start

        log.write(
            "\n"
            + "=" * 110
            + "\n"
            f"Return code: {return_code}\n"
            f"Elapsed seconds: {elapsed:.2f}\n"
            f"Finished: {datetime.now().isoformat()}\n"
        )

    if return_code != 0:
        raise RuntimeError(
            f"Seed {seed}, Step {step} FAILED "
            f"with return code {return_code}. "
            f"See {log_file}"
        )

    artifact = completion_artifact(step, seed)

    if not artifact.exists():
        raise RuntimeError(
            f"Seed {seed}, Step {step} returned successfully "
            f"but expected completion artifact was not found:\n"
            f"{artifact}"
        )

    return elapsed


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Steps 25-31 for the 20 independent "
            "experimental seeds 42-61."
        )
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--end-seed",
        type=int,
        default=61,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip steps whose completion artifact already exists. "
            "Do NOT use this for the first production run while "
            "smoke-test artifacts are present."
        ),
    )

    args = parser.parse_args()

    if args.start_seed < 42 or args.end_seed > 61:
        raise ValueError(
            "Production seeds must remain within 42-61."
        )

    if args.start_seed > args.end_seed:
        raise ValueError(
            "--start-seed cannot exceed --end-seed."
        )

    selected_seeds = list(
        range(args.start_seed, args.end_seed + 1)
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "STEP 32 - 20 INDEPENDENT END-TO-END RUNS"
    )

    print(f"Python:       {sys.executable}")
    print(f"Project:      {ROOT}")
    print(f"Seeds:        {selected_seeds}")
    print(f"Steps:        {[x[0] for x in PIPELINE]}")
    print(f"Resume mode:  {args.resume}")
    print()
    print(
        "IMPORTANT: Step 25 will use its normal production "
        "MAX_EPOCHS because no --max-epochs override is supplied."
    )

    experiment_start = time.time()

    run_records = []

    for seed in selected_seeds:

        banner(
            f"STARTING INDEPENDENT RUN: SEED {seed}"
        )

        seed_log_dir = LOG_ROOT / f"seed_{seed}"
        seed_log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        seed_start = time.time()

        for step, script_name in PIPELINE:

            artifact = completion_artifact(
                step,
                seed
            )

            if args.resume and artifact.exists():

                print(
                    f"[SKIP] Seed {seed} Step {step}: "
                    f"completion artifact exists."
                )

                run_records.append({
                    "seed": seed,
                    "step": step,
                    "script": script_name,
                    "status": "SKIPPED_EXISTING",
                    "elapsed_seconds": 0.0,
                    "completion_artifact": str(artifact),
                })

                continue

            log_file = (
                seed_log_dir /
                f"step_{step}.log"
            )

            try:

                elapsed = run_step(
                    step,
                    script_name,
                    seed,
                    log_file,
                )

                run_records.append({
                    "seed": seed,
                    "step": step,
                    "script": script_name,
                    "status": "PASSED",
                    "elapsed_seconds": elapsed,
                    "completion_artifact": str(artifact),
                })

            except Exception as exc:

                run_records.append({
                    "seed": seed,
                    "step": step,
                    "script": script_name,
                    "status": "FAILED",
                    "elapsed_seconds": None,
                    "error": str(exc),
                    "completion_artifact": str(artifact),
                })

                status_file = (
                    LOG_ROOT /
                    "step32_run_status.json"
                )

                status_file.write_text(
                    json.dumps(
                        run_records,
                        indent=2
                    ),
                    encoding="utf-8"
                )

                banner(
                    "PIPELINE STOPPED ON FAILURE"
                )

                print(
                    f"Seed: {seed}"
                )
                print(
                    f"Step: {step}"
                )
                print(
                    f"Error: {exc}"
                )
                print()
                print(
                    "No later step or seed was executed."
                )
                print(
                    "Fix the problem and rerun with --resume."
                )

                raise

        seed_elapsed = (
            time.time() - seed_start
        )

        banner(
            f"SEED {seed} COMPLETE"
        )

        print(
            f"Seed runtime: "
            f"{seed_elapsed / 60:.2f} minutes"
        )

        status_file = (
            LOG_ROOT /
            "step32_run_status.json"
        )

        status_file.write_text(
            json.dumps(
                run_records,
                indent=2
            ),
            encoding="utf-8"
        )

    total_elapsed = (
        time.time() - experiment_start
    )

    banner(
        "STEP 32 COMPLETE - ALL REQUESTED RUNS FINISHED"
    )

    print(
        f"Seeds completed: {selected_seeds}"
    )

    print(
        f"Total elapsed time: "
        f"{total_elapsed / 3600:.2f} hours"
    )

    print(
        f"Run status:\n"
        f"{LOG_ROOT / 'step32_run_status.json'}"
    )

    print(
        f"Logs:\n{LOG_ROOT}"
    )

    print()
    print(
        "Next stage: aggregate the 20 independent runs "
        "and perform inferential statistical analysis."
    )


if __name__ == "__main__":
    main()