"""
EEG-Conformer – Batch LOSO training for config-generated activity datasets
==========================================================================

Reads window/stride combinations from ``window_stride_configs.json``, maps
them to generated dataset directories, trains each dataset with the existing
``train_activity_loso_batch.py`` logic, and writes per-dataset summaries via
``summarize_loso_results.py``.

Directory convention
--------------------
Datasets:
    eeg-data-processing/data_to_list/global_activity_dataset/<window_...>

Training outputs:
    EEG-Conformer/outputs/activity_loso/<window_...>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent

DEFAULT_CONFIG = EEG_ROOT / "eeg-data-processing" / "data_to_list" / "window_stride_configs.json"
DEFAULT_DATASET_BASE = EEG_ROOT / "eeg-data-processing" / "data_to_list" / "global_activity_dataset"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "outputs" / "activity_loso"
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 72
DEFAULT_LR = 0.0002
DEFAULT_DEVICE = "cuda:0"
DEFAULT_ENV_NAME = "eegconformer310"
AUTO_RERUN_ENV_VAR = "TRAIN_ACTIVITY_LOSO_GENERATED_BATCH_PROJECT_ENV_ACTIVE"

REQUIRED_DATASET_FILES = ("X.npy", "y.npy", "subject_ids.npy", "metadata.json")


class DatasetJob(NamedTuple):
    dataset_name: str
    dataset_root: Path
    output_dir: Path
    window_seconds: float
    stride_seconds: float


class DatasetRunResult(NamedTuple):
    dataset_name: str
    status: str  # "trained" | "skipped" | "failed"
    output_dir: Path
    summary_json: Path | None
    error: str | None = None


@lru_cache(maxsize=1)
def _load_train_batch_module() -> ModuleType:
    path = PROJECT_ROOT / "train_activity_loso_batch.py"
    spec = importlib.util.spec_from_file_location("train_activity_loso_batch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_summary_module() -> ModuleType:
    path = PROJECT_ROOT / "summarize_loso_results.py"
    spec = importlib.util.spec_from_file_location("summarize_loso_results", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subdir_name(window_seconds: float, stride_seconds: float) -> str:
    ws = f"{window_seconds:g}"
    ss = f"{stride_seconds:g}"
    return f"window_{ws}_stride_{ss}"


def project_env_prefix() -> Path:
    return PROJECT_ROOT / ".conda-envs" / DEFAULT_ENV_NAME


def running_inside_project_env() -> bool:
    return Path(sys.executable).resolve() == (project_env_prefix() / "bin" / "python").resolve()


def normalize_device_name(device: str) -> str:
    normalized = device.strip()
    lowered = normalized.lower()
    if lowered in {"cpu", "cuda"}:
        return lowered
    if lowered.startswith("cuda:"):
        index = lowered.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    return normalized


def cuda_is_usable() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return bool(torch.cuda.is_available())


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


def load_config_jobs(
    config_path: str | Path,
    dataset_base: str | Path,
    output_base: str | Path,
) -> list[DatasetJob]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        configs = json.load(fh)

    if not isinstance(configs, list):
        raise ValueError("Config file must contain a JSON array")

    dataset_base_path = Path(dataset_base)
    output_base_path = Path(output_base)
    jobs: list[DatasetJob] = []
    seen_names: set[str] = set()

    for index, cfg in enumerate(configs, start=1):
        if not isinstance(cfg, dict):
            print(f"[SKIP CONFIG] item {index} is not an object: {cfg!r}")
            continue

        window = cfg.get("window_seconds")
        if window is None:
            print(f"[SKIP CONFIG] item {index} missing window_seconds: {cfg!r}")
            continue
        window = float(window)
        if window <= 0:
            print(f"[SKIP CONFIG] item {index} has invalid window_seconds: {cfg!r}")
            continue

        stride = cfg.get("stride_seconds", window)
        stride = float(stride)
        if stride <= 0:
            print(f"[SKIP CONFIG] item {index} has invalid stride_seconds: {cfg!r}")
            continue

        dataset_name = subdir_name(window, stride)
        if dataset_name in seen_names:
            print(f"[SKIP CONFIG] duplicate dataset entry: {dataset_name}")
            continue
        seen_names.add(dataset_name)

        jobs.append(
            DatasetJob(
                dataset_name=dataset_name,
                dataset_root=dataset_base_path / dataset_name,
                output_dir=output_base_path / dataset_name,
                window_seconds=window,
                stride_seconds=stride,
            )
        )

    if not jobs:
        raise ValueError("No valid dataset configs found")
    return jobs


def missing_dataset_files(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    return [name for name in REQUIRED_DATASET_FILES if not (root / name).exists()]


def dataset_output_is_complete(dataset_root: str | Path, output_dir: str | Path) -> bool:
    missing = missing_dataset_files(dataset_root)
    if missing:
        return False

    root = Path(output_dir)
    if not (root / "summary.json").exists() or not (root / "summary.csv").exists():
        return False

    train_batch = _load_train_batch_module()
    subject_ids = train_batch.discover_subject_ids_from_global_dataset(dataset_root)
    return all(train_batch.fold_is_complete(root, subject_id) for subject_id in subject_ids)


def summarize_output_dir(output_dir: str | Path) -> Path:
    summary_module = _load_summary_module()
    summary = summary_module.summarize(output_dir)
    json_path, _ = summary_module.write_summary(summary, output_dir)
    return json_path


def run_generated_dataset_batch(
    jobs: list[DatasetJob],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    skip_existing: bool,
    seed: int = 42,
) -> list[DatasetRunResult]:
    train_batch = _load_train_batch_module()
    results: list[DatasetRunResult] = []

    print(f"Planned datasets: {len(jobs)}")
    for job in jobs:
        try:
            missing = missing_dataset_files(job.dataset_root)
            if missing:
                raise FileNotFoundError(
                    f"Dataset {job.dataset_name} is missing required files in {job.dataset_root}: {missing}"
                )

            if skip_existing and dataset_output_is_complete(job.dataset_root, job.output_dir):
                summary_json = job.output_dir / "summary.json"
                print(f"[SKIP] dataset={job.dataset_name} output_dir={job.output_dir}")
                results.append(
                    DatasetRunResult(
                        dataset_name=job.dataset_name,
                        status="skipped",
                        output_dir=job.output_dir,
                        summary_json=summary_json,
                    )
                )
                continue

            subject_ids = train_batch.discover_subject_ids_from_global_dataset(job.dataset_root)
            print(f"[RUN ] dataset={job.dataset_name} folds={len(subject_ids)} output_dir={job.output_dir}")
            train_batch.run_loso_batch(
                subject_ids=subject_ids,
                dataset_root=job.dataset_root,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                output_dir=job.output_dir,
                skip_existing=skip_existing,
                seed=seed,
            )
            summary_json = summarize_output_dir(job.output_dir)
            print(f"[DONE] dataset={job.dataset_name} summary={summary_json}")
            results.append(
                DatasetRunResult(
                    dataset_name=job.dataset_name,
                    status="trained",
                    output_dir=job.output_dir,
                    summary_json=summary_json,
                )
            )
        except Exception as exc:
            print(f"[FAIL] dataset={job.dataset_name} error={exc}")
            results.append(
                DatasetRunResult(
                    dataset_name=job.dataset_name,
                    status="failed",
                    output_dir=job.output_dir,
                    summary_json=None,
                    error=str(exc),
                )
            )

    trained = sum(result.status == "trained" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = [result for result in results if result.status == "failed"]
    print(f"Batch summary: trained={trained} skipped={skipped} failed={len(failed)}")
    if failed:
        failed_names = ", ".join(result.dataset_name for result in failed)
        raise RuntimeError(f"Dataset batch training finished with failures: {failed_names}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-train all config-generated global activity datasets"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON file containing window/stride configs",
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=DEFAULT_DATASET_BASE,
        help="Directory containing generated dataset subdirectories",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Directory for per-dataset LOSO outputs",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets/folds that already have complete outputs (default: true)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    runtime_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(runtime_argv)
    maybe_rerun_in_project_env(runtime_argv, str(args.device))

    config_path = Path(args.config).expanduser()
    dataset_base = Path(args.dataset_base).expanduser()
    output_base = Path(args.output_base).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not dataset_base.exists():
        raise FileNotFoundError(f"Dataset base directory does not exist: {dataset_base}")
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")

    jobs = load_config_jobs(
        config_path=config_path,
        dataset_base=dataset_base,
        output_base=output_base,
    )
    run_generated_dataset_batch(
        jobs=jobs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=str(args.device),
        skip_existing=bool(args.skip_existing),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
