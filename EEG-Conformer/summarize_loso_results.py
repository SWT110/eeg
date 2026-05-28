"""
EEG-Conformer – LOSO Result Summarizer
=======================================
Reads per-fold outputs from ``local_artifacts/outputs/activity_loso/fold_subject_<id>/``
and produces:

* ``summary.json``  – full stats including confusion matrix and per-class metrics
* ``summary.csv``   – human-readable table for quick inspection

Usage
-----
    python summarize_loso_results.py
    python summarize_loso_results.py --output-dir /path/to/activity_loso
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import ACTIVITY_LOSO_OUTPUT_DIR

DEFAULT_OUTPUT_DIR = ACTIVITY_LOSO_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Low-level helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def confusion_matrix_from_arrays(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[list[int]]:
    cm: list[list[int]] = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t)][int(p)] += 1
    return cm


def per_class_metrics_from_cm(cm: list[list[int]]) -> list[dict]:
    n = len(cm)
    result = []
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result.append({
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        })
    return result


def macro_f1_from_per_class(per_class: list[dict]) -> float:
    if not per_class:
        return 0.0
    return round(sum(d["f1"] for d in per_class) / len(per_class), 6)


def add_confusion_matrices(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    n = len(a)
    return [[a[i][j] + b[i][j] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def discover_fold_dirs(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("fold_subject_")
    )
    return dirs


def load_fold_result(fold_dir: Path) -> dict:
    """Load metrics.json; attach y_true/y_pred arrays under private keys if available."""
    metrics_path = fold_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {fold_dir}")

    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics: dict = json.load(fh)

    pred_path = fold_dir / "test_predictions.npz"
    if pred_path.exists():
        data = np.load(pred_path)
        metrics["_y_true"] = data["y_true"]
        metrics["_y_pred"] = data["y_pred"]

    return metrics


# ---------------------------------------------------------------------------
# Core summarisation
# ---------------------------------------------------------------------------

def summarize(output_dir: str | Path) -> dict:
    fold_dirs = discover_fold_dirs(output_dir)
    if not fold_dirs:
        raise ValueError(f"No fold_subject_* directories found in {output_dir}")

    fold_records: list[dict] = []
    pooled_y_true: list[np.ndarray] = []
    pooled_y_pred: list[np.ndarray] = []

    for fold_dir in fold_dirs:
        rec = load_fold_result(fold_dir)
        y_true = rec.pop("_y_true", None)
        y_pred = rec.pop("_y_pred", None)
        fold_records.append(rec)
        if y_true is not None and y_pred is not None:
            pooled_y_true.append(y_true)
            pooled_y_pred.append(y_pred)

    best_accs = [r["best_test_acc"] for r in fold_records]
    mean_acc = float(np.mean(best_accs))
    std_acc = float(np.std(best_accs))

    n_classes: int = fold_records[0].get("n_classes", 3) if fold_records else 3

    # Prefer pooling raw predictions; fall back to summing stored confusion matrices
    overall_cm: list[list[int]] | None = None
    per_class: list[dict] = []
    macro_f1: float | None = None

    if len(pooled_y_true) == len(fold_records):
        combined_true = np.concatenate(pooled_y_true)
        combined_pred = np.concatenate(pooled_y_pred)
        overall_cm = confusion_matrix_from_arrays(combined_true, combined_pred, n_classes)
    elif any("confusion_matrix" in r for r in fold_records):
        empty = [[0] * n_classes for _ in range(n_classes)]
        accumulated = empty
        for r in fold_records:
            fcm = r.get("confusion_matrix")
            if fcm:
                accumulated = add_confusion_matrices(accumulated, fcm)
        overall_cm = accumulated

    if overall_cm is not None:
        per_class = per_class_metrics_from_cm(overall_cm)
        macro_f1 = macro_f1_from_per_class(per_class)

    per_fold = [
        {
            "subject_id": r["test_subject_id"],
            "best_test_acc": r["best_test_acc"],
            "average_test_acc": r.get("average_test_acc"),
            "n_train_samples": r.get("n_train_samples"),
            "n_test_samples": r.get("n_test_samples"),
        }
        for r in fold_records
    ]

    return {
        "n_folds": len(fold_records),
        "mean_best_test_acc": round(mean_acc, 6),
        "std_best_test_acc": round(std_acc, 6),
        "per_fold": per_fold,
        "overall_confusion_matrix": overall_cm,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_summary(summary: dict, summary_dir: str | Path) -> tuple[Path, Path]:
    root = Path(summary_dir)
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / "summary.json"
    csv_path = root / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", "best_test_acc", "average_test_acc",
                         "n_train_samples", "n_test_samples"])
        for fold in summary["per_fold"]:
            writer.writerow([
                fold["subject_id"],
                fold["best_test_acc"],
                fold.get("average_test_acc", ""),
                fold.get("n_train_samples", ""),
                fold.get("n_test_samples", ""),
            ])
        writer.writerow([])
        writer.writerow(["mean_best_test_acc", summary["mean_best_test_acc"]])
        writer.writerow(["std_best_test_acc", summary["std_best_test_acc"]])
        if summary.get("macro_f1") is not None:
            writer.writerow(["macro_f1", summary["macro_f1"]])

    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LOSO training results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing fold_subject_* subdirectories",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Where to write summary.json / summary.csv (default: same as --output-dir)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    summary_dir = Path(args.summary_dir).expanduser() if args.summary_dir else output_dir

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    summary = summarize(output_dir)
    json_path, csv_path = write_summary(summary, summary_dir)

    print("Summary written to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"\nMean best_test_acc : {summary['mean_best_test_acc']:.4f} "
          f"± {summary['std_best_test_acc']:.4f}")
    if summary.get("macro_f1") is not None:
        print(f"Macro F1           : {summary['macro_f1']:.4f}")
    if summary.get("overall_confusion_matrix"):
        print("\nOverall Confusion Matrix (row=true, col=pred):")
        for row in summary["overall_confusion_matrix"]:
            print(f"  {row}")


if __name__ == "__main__":
    main()
