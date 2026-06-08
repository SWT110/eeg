"""
EEG-Conformer – Batch LOSO Activity Three-Class Training
=========================================================
Runs all LOSO folds (one held-out subject per fold) for the global
activity dataset. Keeps the same interactive/batch-oriented UX used across
this project:

* Auto-discovers subject_ids from ``subject_ids.npy`` in the dataset root
* ``--subject-ids`` to restrict which folds to run
* ``--skip-existing`` to skip folds whose output dir already contains
  ``metrics.json`` or ``best_model.pt``
* Automatic rerun inside the project conda env when CUDA is requested but
  unavailable in the current interpreter

Usage
-----
    python train_activity_loso_batch.py \\
        --subject-ids 1,2,3 \\
        --epochs 200 \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure EEG-Conformer directory is importable (works when loaded via importlib
# as well as when run directly or from a different working directory)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_activity_loso import (  # noqa: E402
    AUTO_RERUN_ENV_VAR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETAS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DEPTH,
    DEFAULT_DEVICE,
    DEFAULT_DROPOUT,
    DEFAULT_EMB_SIZE,
    DEFAULT_ENV_NAME,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_DOMAIN,
    DEFAULT_LR,
    DEFAULT_NUM_HEADS,
    DEFAULT_OUTPUT_DIR,
    cuda_is_usable,
    normalize_device_name,
    parse_class_weights,
    project_env_prefix,
    running_inside_project_env,
    train_loso_fold,
    validate_input_domain,
    validate_device,
)

PROJECT_ROOT = _HERE
EEG_ROOT = PROJECT_ROOT.parent


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class LosoFoldResult(NamedTuple):
    subject_id: int
    status: str          # "trained" | "skipped" | "failed"
    metrics_path: Path | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Subject-ID discovery from the global dataset
# ---------------------------------------------------------------------------

def discover_subject_ids_from_global_dataset(dataset_root: str | Path) -> list[int]:
    """Return sorted unique subject IDs found in ``subject_ids.npy``."""
    root = Path(dataset_root)
    path = root / "subject_ids.npy"
    if not path.exists():
        raise FileNotFoundError(f"subject_ids.npy not found in {root}")
    arr = np.load(path)
    unique_ids = sorted({int(v) for v in arr.tolist()})
    if not unique_ids:
        raise ValueError(f"No subject IDs found in {path}")
    return unique_ids


# ---------------------------------------------------------------------------
# Skip-existing check
# ---------------------------------------------------------------------------

def fold_output_dir(output_dir: str | Path, subject_id: int) -> Path:
    return Path(output_dir) / f"fold_subject_{subject_id}"


def fold_is_complete(
    output_dir: str | Path,
    subject_id: int,
    input_domain: str = DEFAULT_INPUT_DOMAIN,
) -> bool:
    """Return True if the fold directory already contains a result artifact."""
    fold_dir = fold_output_dir(output_dir, subject_id)
    metrics_path = fold_dir / "metrics.json"
    if metrics_path.exists():
        expected_input_domain = validate_input_domain(input_domain)
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        actual_input_domain = validate_input_domain(metrics.get("input_domain"))
        return actual_input_domain == expected_input_domain
    return (fold_dir / "best_model.pt").exists()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_loso_batch(
    subject_ids: list[int],
    dataset_root: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: str | Path,
    skip_existing: bool,
    seed: int = 42,
    input_domain: str = DEFAULT_INPUT_DOMAIN,
    class_weights: list[float] | None = None,
) -> list[LosoFoldResult]:
    resolved_device = validate_device(device)
    resolved_input_domain = validate_input_domain(input_domain)
    results: list[LosoFoldResult] = []

    print(f"Planned LOSO folds: {len(subject_ids)}")
    for subject_id in subject_ids:
        if skip_existing and fold_is_complete(output_dir, subject_id, resolved_input_domain):
            fold_dir = fold_output_dir(output_dir, subject_id)
            print(f"[SKIP] subject={subject_id}  fold_dir={fold_dir}")
            results.append(
                LosoFoldResult(
                    subject_id=subject_id,
                    status="skipped",
                    metrics_path=fold_dir / "metrics.json",
                )
            )
            continue

        print(f"[RUN ] subject={subject_id}")
        try:
            metrics_path = train_loso_fold(
                dataset_root=dataset_root,
                test_subject_id=subject_id,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=resolved_device,
                output_dir=output_dir,
                seed=seed,
                input_domain=resolved_input_domain,
                class_weights=class_weights,
            )
        except Exception as exc:
            print(f"[FAIL] subject={subject_id}  error={exc}")
            results.append(
                LosoFoldResult(
                    subject_id=subject_id,
                    status="failed",
                    metrics_path=None,
                    error=str(exc),
                )
            )
            continue

        print(f"[DONE] subject={subject_id}  metrics={metrics_path}")
        results.append(
            LosoFoldResult(
                subject_id=subject_id,
                status="trained",
                metrics_path=metrics_path,
            )
        )

    trained = sum(r.status == "trained" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = [r for r in results if r.status == "failed"]
    print(f"Batch summary: trained={trained} skipped={skipped} failed={len(failed)}")
    if failed:
        ids = ", ".join(str(r.subject_id) for r in failed)
        raise RuntimeError(f"LOSO batch finished with failed folds: {ids}")
    return results


# ---------------------------------------------------------------------------
# Project-env auto-restart
# ---------------------------------------------------------------------------

def maybe_rerun_in_project_env(argv: list[str], device: str) -> None:
    normalized_device = normalize_device_name(device)
    if not normalized_device.startswith("cuda"):
        return
    if cuda_is_usable():
        return
    if running_inside_project_env():
        return
    if os.environ.get(AUTO_RERUN_ENV_VAR) == "1":
        return

    env_prefix = project_env_prefix()
    if not env_prefix.exists():
        return

    rerun_env = os.environ.copy()
    rerun_env[AUTO_RERUN_ENV_VAR] = "1"
    completed = subprocess.run(
        [str(env_prefix / "bin" / "python"), str(Path(__file__).resolve()), *argv],
        check=False,
        env=rerun_env,
    )
    raise SystemExit(completed.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_subject_id_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values: list[int] = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            raise ValueError("--subject-ids contains an empty item")
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError("--subject-ids must be a comma-separated list of integers") from exc
        if value < 1:
            raise ValueError("--subject-ids values must be >= 1")
        if value not in values:
            values.append(value)
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch LOSO training – EEG-Conformer activity three-class classification"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Global activity dataset directory (X.npy, y.npy, subject_ids.npy, metadata.json)",
    )
    parser.add_argument(
        "--subject-ids",
        type=str,
        default=None,
        help="Optional comma-separated subject IDs to run, e.g. 1,3,5",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--input-domain",
        type=str,
        default=DEFAULT_INPUT_DOMAIN,
        help="Input representation: time, fft, or time_fft dual branch (default: time)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Optional comma-separated class weights for CrossEntropyLoss, e.g. 3,3,1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for per-fold outputs (fold_subject_<id>/)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a fold if its output dir already contains metrics.json or best_model.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    runtime_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(runtime_argv)
    maybe_rerun_in_project_env(runtime_argv, str(args.device))

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    all_subject_ids = discover_subject_ids_from_global_dataset(dataset_root)
    requested = parse_subject_id_list(args.subject_ids)
    class_weights = parse_class_weights(args.class_weights)
    input_domain = validate_input_domain(args.input_domain)
    if requested is not None:
        missing = [sid for sid in requested if sid not in all_subject_ids]
        if missing:
            raise ValueError(
                f"Requested subject IDs not in dataset: {missing}. "
                f"Available: {all_subject_ids}"
            )
        subject_ids = requested
    else:
        subject_ids = all_subject_ids

    run_loso_batch(
        subject_ids=subject_ids,
        dataset_root=dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=str(args.device),
        output_dir=Path(args.output_dir).expanduser(),
        skip_existing=bool(args.skip_existing),
        seed=args.seed,
        input_domain=input_domain,
        class_weights=class_weights,
    )


if __name__ == "__main__":
    main()
