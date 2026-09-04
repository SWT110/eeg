"""Run reproducible external-model comparisons on the activity EEG dataset.

The default command executes EEGNet, ShallowConvNet, ATCNet, and TCFormer with
the same LOSO folds and optimization protocol used by the current development
experiment.  Each model/seed receives a separate output directory compatible
with ``summarize_loso_results.py``.

This is an exploratory, test-guided protocol: the held-out subject is evaluated
after every epoch and selects the best checkpoint.  It enables a like-for-like
comparison with the archived development result, but it is not a substitute for
the nested participant-wise validation required for a confirmatory estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from comparison_models import (  # noqa: E402
    COMPARISON_MODEL_METADATA,
    IMPLEMENTATION_PROVENANCE,
    VALID_COMPARISON_MODELS,
    build_comparison_model,
    count_trainable_parameters,
    normalize_comparison_model_name,
)
from summarize_loso_results import summarize, write_summary  # noqa: E402
from train_activity_loso import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETAS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    atomic_json_dump,
    cuda_is_usable,
    parse_class_weights,
    train_loso_fold,
)


DEFAULT_COMPARISON_DATASET = DEFAULT_DATASET_ROOT / "window_15_stride_3"
DEFAULT_COMPARISON_OUTPUT = EEG_ROOT / "local_artifacts" / "outputs" / "activity_comparison"
DEFAULT_MODELS = ",".join(VALID_COMPARISON_MODELS)
DEFAULT_SEEDS = "42"
DEFAULT_CLASS_WEIGHTS = "3,3,1"
PROTOCOL_NAME = "exploratory_test_guided_loso_v1"


def parse_model_list(raw: str) -> list[str]:
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("At least one model is required")
    normalized = [normalize_comparison_model_name(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate models after normalization: {normalized}")
    return normalized


def parse_int_list(raw: str | None, parameter_name: str) -> list[int] | None:
    if raw is None:
        return None
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError(f"{parameter_name} cannot be empty")
    parsed = [int(value) for value in values]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{parameter_name} contains duplicate values: {parsed}")
    return parsed


def discover_subject_ids(dataset_root: str | Path) -> list[int]:
    path = Path(dataset_root) / "subject_ids.npy"
    if not path.exists():
        raise FileNotFoundError(f"subject_ids.npy not found: {path}")
    values = np.load(path, mmap_mode="r")
    subjects = sorted({int(value) for value in values.tolist()})
    if not subjects:
        raise ValueError(f"No subject IDs found in {path}")
    return subjects


def inspect_dataset_shape(dataset_root: str | Path) -> tuple[int, int, int]:
    root = Path(dataset_root)
    x_path = root / "X.npy"
    y_path = root / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"X.npy/y.npy not found in {root}")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if x.ndim != 3:
        raise ValueError(f"Expected X shape [N,C,T], got {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"X/y length mismatch: {len(x)} != {len(y)}")
    return int(x.shape[1]), int(x.shape[2]), int(np.max(y)) + 1


def comparison_run_dir(
    output_dir: str | Path, architecture: str, seed: int
) -> Path:
    return Path(output_dir) / architecture / f"seed_{seed}"


def fold_metrics_path(run_dir: str | Path, subject_id: int) -> Path:
    return Path(run_dir) / f"fold_subject_{subject_id}" / "metrics.json"


def completed_fold_matches(
    metrics_path: str | Path,
    *,
    architecture: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    class_weights: list[float] | None,
) -> bool:
    path = Path(metrics_path)
    if not path.exists():
        return False
    required_artifacts = (
        path.parent / "best_model.pt",
        path.parent / "test_predictions.npz",
        path.parent / "epoch_history.csv",
        path.parent / "epoch_history.json",
        path.parent / "train.log",
    )
    if not all(artifact.exists() for artifact in required_artifacts):
        return False
    with open(path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    return (
        metrics.get("architecture") == architecture
        and int(metrics.get("seed", -1)) == int(seed)
        and int(metrics.get("epochs", -1)) == int(epochs)
        and int(metrics.get("batch_size", -1)) == int(batch_size)
        and float(metrics.get("lr", float("nan"))) == float(lr)
        and metrics.get("class_weights") == class_weights
        and metrics.get("input_domain") == "time"
    )


def pooled_accuracy(confusion_matrix: list[list[int]] | None) -> float | None:
    if not confusion_matrix:
        return None
    total = sum(sum(row) for row in confusion_matrix)
    if total == 0:
        return None
    correct = sum(confusion_matrix[index][index] for index in range(len(confusion_matrix)))
    return round(correct / total, 6)


def balanced_accuracy(per_class_metrics: list[dict[str, Any]]) -> float | None:
    if not per_class_metrics:
        return None
    recalls = [float(row["recall"]) for row in per_class_metrics]
    return round(float(np.mean(recalls)), 6)


def protocol_metadata(
    *,
    dataset_root: Path,
    models: list[str],
    subjects: list[int],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    class_weights: list[float] | None,
    device: str,
) -> dict[str, Any]:
    manuscript_eligible = len(subjects) == 11 and epochs == 200
    return {
        "protocol_name": PROTOCOL_NAME,
        "status": "exploratory",
        "run_scope": (
            "full_11_fold_development_comparison"
            if manuscript_eligible
            else "smoke_or_partial_validation"
        ),
        "eligible_for_manuscript_table": manuscript_eligible,
        "selection_warning": (
            "The held-out LOSO subject is evaluated every epoch and selects the best "
            "checkpoint. Use these results only for comparison with the existing "
            "development experiment, not as an unbiased confirmatory estimate."
        ),
        "dataset_root": str(dataset_root.resolve()),
        "input_domain": "time",
        "standardization": "one scalar mean/std from each training fold only",
        "models": models,
        "subjects": subjects,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "lr": lr,
        "optimizer_betas": list(DEFAULT_BETAS),
        "weight_decay": 0.0,
        "class_weights": class_weights,
        "checkpoint_selection": "highest held-out-subject accuracy across epochs",
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "implementation_provenance": IMPLEMENTATION_PROVENANCE,
        "model_references": {
            model: COMPARISON_MODEL_METADATA[model] for model in models
        },
    }


def write_comparison_summary(
    output_dir: str | Path,
    protocol: dict[str, Any],
    run_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "comparison_summary.json"
    csv_path = root / "comparison_summary.csv"
    atomic_json_dump(
        {
            "protocol": protocol,
            "runs": run_records,
        },
        json_path,
        encoding="ascii",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "architecture",
            "display_name",
            "year",
            "doi",
            "seed",
            "n_folds",
            "mean_best_test_acc",
            "std_best_test_acc",
            "pooled_accuracy",
            "macro_f1",
            "balanced_accuracy",
            "trainable_parameters",
            "summary_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in run_records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return json_path, csv_path


def build_dry_run_plan(
    dataset_root: Path,
    models: list[str],
    subjects: list[int],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    class_weights: list[float] | None,
    device: str,
) -> dict[str, Any]:
    n_channels, n_times, n_classes = inspect_dataset_shape(dataset_root)
    model_plans = []
    for architecture in models:
        model = build_comparison_model(
            architecture, n_channels, n_times, n_classes
        )
        model_plans.append(
            {
                "architecture": architecture,
                "display_name": COMPARISON_MODEL_METADATA[architecture]["display_name"],
                "trainable_parameters": count_trainable_parameters(model),
                "architecture_config": model.architecture_config,
            }
        )
    protocol = protocol_metadata(
        dataset_root=dataset_root,
        models=models,
        subjects=subjects,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        class_weights=class_weights,
        device=device,
    )
    return {
        "dataset_shape": [len(np.load(dataset_root / "y.npy", mmap_mode="r")), n_channels, n_times],
        "n_classes": n_classes,
        "planned_folds": len(models) * len(subjects) * len(seeds),
        "protocol": protocol,
        "model_plans": model_plans,
    }


def run_comparisons(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    models: list[str],
    subject_ids: list[int] | None,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    class_weights: list[float] | None,
    device: str,
    skip_existing: bool,
    resume: bool,
) -> tuple[Path, Path]:
    dataset_path = Path(dataset_root).expanduser()
    output_path = Path(output_dir).expanduser()
    available_subjects = discover_subject_ids(dataset_path)
    subjects = available_subjects if subject_ids is None else subject_ids
    unknown_subjects = sorted(set(subjects).difference(available_subjects))
    if unknown_subjects:
        raise ValueError(
            f"Subject IDs absent from dataset: {unknown_subjects}; available={available_subjects}"
        )
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs/batch_size must be >= 1 and lr must be > 0")
    if device.lower().startswith("cuda") and not cuda_is_usable():
        raise RuntimeError(
            f"CUDA device {device!r} was requested, but CUDA is unavailable in "
            f"Python {platform.python_version()} / torch {torch.__version__}"
        )

    protocol = protocol_metadata(
        dataset_root=dataset_path,
        models=models,
        subjects=subjects,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        class_weights=class_weights,
        device=device,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(protocol, output_path / "protocol.json", encoding="ascii")

    run_records: list[dict[str, Any]] = []
    for architecture in models:
        for seed in seeds:
            run_dir = comparison_run_dir(output_path, architecture, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"\n[COMPARISON] architecture={architecture} seed={seed} "
                f"subjects={subjects}"
            )
            for subject_id in subjects:
                metrics_path = fold_metrics_path(run_dir, subject_id)
                if metrics_path.exists():
                    matches = completed_fold_matches(
                        metrics_path,
                        architecture=architecture,
                        seed=seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        class_weights=class_weights,
                    )
                    if skip_existing and matches:
                        print(f"[SKIP] complete matching fold subject={subject_id}")
                        continue
                    raise ValueError(
                        f"Existing completed fold does not match this run: {metrics_path}. "
                        "Choose a new output directory or pass the matching configuration."
                    )

                train_loso_fold(
                    dataset_root=dataset_path,
                    test_subject_id=subject_id,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                    output_dir=run_dir,
                    seed=seed,
                    class_weights=class_weights,
                    architecture=architecture,
                    resume=resume,
                )

            run_summary = summarize(run_dir)
            actual_subjects = {
                int(record["subject_id"])
                for record in run_summary.get("per_fold", [])
            }
            if actual_subjects != set(subjects):
                raise ValueError(
                    f"Run directory {run_dir} contains subjects {sorted(actual_subjects)}, "
                    f"but this invocation expects {sorted(subjects)}"
                )
            summary_json, _summary_csv = write_summary(run_summary, run_dir)
            n_channels, n_times, n_classes = inspect_dataset_shape(dataset_path)
            model = build_comparison_model(
                architecture, n_channels, n_times, n_classes
            )
            metadata = COMPARISON_MODEL_METADATA[architecture]
            run_records.append(
                {
                    "architecture": architecture,
                    "display_name": metadata["display_name"],
                    "year": metadata["year"],
                    "doi": metadata["doi"],
                    "seed": seed,
                    "n_folds": run_summary["n_folds"],
                    "mean_best_test_acc": run_summary["mean_best_test_acc"],
                    "std_best_test_acc": run_summary["std_best_test_acc"],
                    "pooled_accuracy": pooled_accuracy(
                        run_summary.get("overall_confusion_matrix")
                    ),
                    "macro_f1": run_summary.get("macro_f1"),
                    "balanced_accuracy": balanced_accuracy(
                        run_summary.get("per_class_metrics", [])
                    ),
                    "trainable_parameters": count_trainable_parameters(model),
                    "summary_json": str(summary_json),
                }
            )
            write_comparison_summary(output_path, protocol, run_records)

    return write_comparison_summary(output_path, protocol, run_records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external EEG baselines under the common activity LOSO protocol"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_COMPARISON_DATASET
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_COMPARISON_OUTPUT
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=f"Comma-separated subset of: {DEFAULT_MODELS}",
    )
    parser.add_argument(
        "--subject-ids",
        default=None,
        help="Comma-separated held-out subject IDs (default: all in dataset)",
    )
    parser.add_argument(
        "--seeds", default=DEFAULT_SEEDS, help="Comma-separated random seeds"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--class-weights",
        default=DEFAULT_CLASS_WEIGHTS,
        help="Comma-separated CrossEntropyLoss weights; use 'none' to disable",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip completed folds only when their full run configuration matches",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete folds from last_checkpoint.pt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset/configuration and print model parameter counts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    models = parse_model_list(args.models)
    subject_ids = parse_int_list(args.subject_ids, "subject_ids")
    seeds = parse_int_list(args.seeds, "seeds")
    assert seeds is not None
    raw_class_weights = str(args.class_weights).strip().lower()
    class_weights = (
        None
        if raw_class_weights in {"none", "null", "off"}
        else parse_class_weights(args.class_weights)
    )
    dataset_root = Path(args.dataset_root).expanduser()
    subjects = discover_subject_ids(dataset_root) if subject_ids is None else subject_ids

    if args.dry_run:
        plan = build_dry_run_plan(
            dataset_root,
            models,
            subjects,
            seeds,
            args.epochs,
            args.batch_size,
            args.lr,
            class_weights,
            args.device,
        )
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    json_path, csv_path = run_comparisons(
        dataset_root=dataset_root,
        output_dir=Path(args.output_dir).expanduser(),
        models=models,
        subject_ids=subject_ids,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        class_weights=class_weights,
        device=args.device,
        skip_existing=args.skip_existing,
        resume=args.resume,
    )
    print(f"\nComparison summary JSON: {json_path}")
    print(f"Comparison summary CSV:  {csv_path}")


if __name__ == "__main__":
    main()
