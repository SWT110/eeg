"""
Experiment manifest helpers for EEG-Conformer LOSO runs.

Each generated window/stride experiment writes its own manifest into the
corresponding output directory.  Parent output folders are intentionally kept
as loose groupings only; the per-experiment manifest is the authoritative
record for dataset, model, split, runtime, environment, and output metadata.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_JSON_NAME = "experiment_manifest.json"
MANIFEST_MD_NAME = "experiment_manifest.md"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _array_info(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def collect_environment_info(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    status = _git_value(["status", "--short"], root)

    torch_info: dict[str, Any] = {
        "torch_version": None,
        "cuda_available": None,
        "mps_available": None,
    }
    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except Exception:
        pass

    return {
        "git_commit": _git_value(["rev-parse", "HEAD"], root),
        "git_branch": _git_value(["branch", "--show-current"], root),
        "git_dirty": bool(status),
        "git_status_short": status or "",
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        **torch_info,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }


def collect_dataset_info(dataset_root: str | Path, dataset_name: str | None = None) -> dict[str, Any]:
    root = Path(dataset_root)
    metadata = _load_json(root / "metadata.json")

    X_info = _array_info(root / "X.npy")
    y_info = _array_info(root / "y.npy")
    subject_ids_info = _array_info(root / "subject_ids.npy")
    record_ids_info = _array_info(root / "record_ids.npy")
    window_indices_info = _array_info(root / "window_indices.npy")

    y = np.load(root / "y.npy") if (root / "y.npy").exists() else np.array([], dtype=np.int64)
    subject_ids = (
        np.load(root / "subject_ids.npy")
        if (root / "subject_ids.npy").exists()
        else np.array([], dtype=np.int64)
    )
    record_ids = (
        np.load(root / "record_ids.npy")
        if (root / "record_ids.npy").exists()
        else np.array([], dtype=np.int64)
    )

    unique_subject_ids = sorted(int(v) for v in np.unique(subject_ids).tolist()) if len(subject_ids) else []
    per_subject_counts: dict[str, int] = {}
    per_subject_class_counts: dict[str, dict[str, int]] = {}
    for subject_id in unique_subject_ids:
        mask = subject_ids == subject_id
        per_subject_counts[str(subject_id)] = int(mask.sum())
        per_subject_class_counts[str(subject_id)] = _counts(y[mask]) if len(y) else {}

    x_shape = X_info["shape"] if X_info is not None else []
    n_samples = int(x_shape[0]) if len(x_shape) >= 1 else int(len(y))
    n_channels = int(x_shape[1]) if len(x_shape) >= 2 else None
    n_times = int(x_shape[2]) if len(x_shape) >= 3 else None

    return {
        "dataset_root": str(root),
        "dataset_name": dataset_name or root.name,
        "window_seconds": metadata.get("window_seconds"),
        "stride_seconds": metadata.get("stride_seconds"),
        "arrays": {
            "X": X_info,
            "y": y_info,
            "subject_ids": subject_ids_info,
            "record_ids": record_ids_info,
            "window_indices": window_indices_info,
        },
        "X_shape": x_shape,
        "X_dtype": X_info["dtype"] if X_info is not None else None,
        "y_shape": y_info["shape"] if y_info is not None else [],
        "n_samples": n_samples,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_subjects": int(len(unique_subject_ids)),
        "n_records": int(len(np.unique(record_ids))) if len(record_ids) else metadata.get("n_records"),
        "subject_ids": unique_subject_ids,
        "label_map": metadata.get("label_map", {}),
        "class_counts": _counts(y) if len(y) else {},
        "per_subject_counts": per_subject_counts,
        "per_subject_class_counts": per_subject_class_counts,
        "record_id_map": metadata.get("record_id_map", {}),
        "record_counts": _counts(record_ids) if len(record_ids) else {},
        "metadata": metadata,
    }


def collect_loso_split_info(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    if not (root / "subject_ids.npy").exists():
        return {"strategy": "LOSO", "n_folds": 0, "folds": []}

    subject_ids = np.load(root / "subject_ids.npy")
    unique_subject_ids = sorted(int(v) for v in np.unique(subject_ids).tolist())
    folds: list[dict[str, Any]] = []
    for test_subject_id in unique_subject_ids:
        test_mask = subject_ids == test_subject_id
        train_subject_ids = [sid for sid in unique_subject_ids if sid != test_subject_id]
        folds.append(
            {
                "test_subject_id": test_subject_id,
                "train_subject_ids": train_subject_ids,
                "n_train_samples": int((~test_mask).sum()),
                "n_test_samples": int(test_mask.sum()),
            }
        )

    return {
        "strategy": "LOSO",
        "validation_strategy": "none",
        "test_eval_each_epoch": True,
        "best_model_selected_by": "test_acc",
        "n_folds": len(folds),
        "fold_subject_ids": unique_subject_ids,
        "folds": folds,
    }


def collect_output_info(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    fold_dirs = sorted(d for d in root.glob("fold_subject_*") if d.is_dir())
    folds: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        metrics_path = fold_dir / "metrics.json"
        metrics = _load_json(metrics_path)
        folds.append(
            {
                "fold_dir": fold_dir.name,
                "test_subject_id": metrics.get("test_subject_id"),
                "metrics_json": str(metrics_path) if metrics_path.exists() else None,
                "best_model": str(fold_dir / "best_model.pt") if (fold_dir / "best_model.pt").exists() else None,
                "test_predictions": str(fold_dir / "test_predictions.npz") if (fold_dir / "test_predictions.npz").exists() else None,
                "epoch_history_csv": str(fold_dir / "epoch_history.csv") if (fold_dir / "epoch_history.csv").exists() else None,
                "train_log": str(fold_dir / "train.log") if (fold_dir / "train.log").exists() else None,
            }
        )

    return {
        "output_dir": str(root),
        "summary_json": str(root / "summary.json") if (root / "summary.json").exists() else None,
        "summary_csv": str(root / "summary.csv") if (root / "summary.csv").exists() else None,
        "manifest_json": str(root / MANIFEST_JSON_NAME),
        "manifest_md": str(root / MANIFEST_MD_NAME),
        "fold_dirs": [d.name for d in fold_dirs],
        "folds": folds,
    }


def _first_fold_metrics(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    for metrics_path in sorted(root.glob("fold_subject_*/metrics.json")):
        metrics = _load_json(metrics_path)
        if metrics:
            return metrics
    return {}


def build_model_info(
    input_domain: str,
    dataset_info: dict[str, Any],
    training_config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = dict(training_config or {})
    fold_metrics = _first_fold_metrics(output_dir) if output_dir is not None else {}
    raw_n_times = dataset_info.get("n_times")
    fft_n_times = int(raw_n_times // 2 + 1) if isinstance(raw_n_times, int) else None
    n_classes_from_dataset = (
        len(dataset_info.get("label_map", {}))
        or len(dataset_info.get("class_counts", {}))
        or None
    )
    resolved_domain = input_domain
    model_type = "dual_branch" if resolved_domain == "time_fft" else "single_branch"

    info = {
        "model_type": fold_metrics.get("model_type", model_type),
        "input_domain": fold_metrics.get("input_domain", resolved_domain),
        "n_classes": fold_metrics.get("n_classes", cfg.get("n_classes", n_classes_from_dataset)),
        "emb_size": fold_metrics.get("emb_size", cfg.get("emb_size")),
        "depth": fold_metrics.get("depth", cfg.get("depth")),
        "num_heads": fold_metrics.get("num_heads", cfg.get("num_heads")),
        "dropout": fold_metrics.get("dropout", cfg.get("dropout")),
    }

    if info["input_domain"] == "time_fft":
        info.update(
            {
                "time_n_times": fold_metrics.get("time_n_times", raw_n_times),
                "fft_n_times": fold_metrics.get("fft_n_times", fft_n_times),
                "time_branch_model": "ConformerFeatureBranch",
                "fft_branch_model": "ConformerFeatureBranch",
                "fusion_method": "concat",
                "fusion_head": "FusionClassificationHead",
            }
        )
    else:
        info["n_times"] = fold_metrics.get(
            "n_times",
            fft_n_times if info["input_domain"] == "fft" else raw_n_times,
        )

    return info


def build_training_info(
    training_config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = dict(training_config or {})
    fold_metrics = _first_fold_metrics(output_dir) if output_dir is not None else {}
    return {
        "epochs": fold_metrics.get("epochs", cfg.get("epochs")),
        "lr": fold_metrics.get("lr", cfg.get("lr")),
        "seed": fold_metrics.get("seed", cfg.get("seed")),
        "class_weights": fold_metrics.get("class_weights", cfg.get("class_weights")),
        "optimizer": fold_metrics.get("optimizer", cfg.get("optimizer", "Adam")),
        "optimizer_betas": fold_metrics.get("optimizer_betas", cfg.get("optimizer_betas")),
        "loss": fold_metrics.get("loss", cfg.get("loss", "CrossEntropyLoss")),
    }


def build_markdown(manifest: dict[str, Any]) -> str:
    dataset = manifest.get("dataset", {})
    model = manifest.get("model", {})
    training = manifest.get("training", {})
    runtime = manifest.get("runtime", {})
    split = manifest.get("split", {})
    environment = manifest.get("environment", {})
    outputs = manifest.get("outputs", {})

    lines = [
        "# Experiment Manifest",
        "",
        "## Dataset",
        f"- dataset_name: `{dataset.get('dataset_name')}`",
        f"- dataset_root: `{dataset.get('dataset_root')}`",
        f"- window_seconds: `{dataset.get('window_seconds')}`",
        f"- stride_seconds: `{dataset.get('stride_seconds')}`",
        f"- X_shape: `{dataset.get('X_shape')}`",
        f"- X_dtype: `{dataset.get('X_dtype')}`",
        f"- n_samples: `{dataset.get('n_samples')}`",
        f"- n_subjects: `{dataset.get('n_subjects')}`",
        f"- n_records: `{dataset.get('n_records')}`",
        f"- class_counts: `{dataset.get('class_counts')}`",
        "",
        "## Model",
        f"- model_type: `{model.get('model_type')}`",
        f"- input_domain: `{model.get('input_domain')}`",
        f"- n_classes: `{model.get('n_classes')}`",
        f"- emb_size: `{model.get('emb_size')}`",
        f"- depth: `{model.get('depth')}`",
        f"- num_heads: `{model.get('num_heads')}`",
        f"- dropout: `{model.get('dropout')}`",
        f"- time_n_times: `{model.get('time_n_times')}`",
        f"- fft_n_times: `{model.get('fft_n_times')}`",
        f"- fusion_method: `{model.get('fusion_method')}`",
        "",
        "## Training",
        f"- epochs: `{training.get('epochs')}`",
        f"- lr: `{training.get('lr')}`",
        f"- seed: `{training.get('seed')}`",
        f"- class_weights: `{training.get('class_weights')}`",
        f"- optimizer: `{training.get('optimizer')}`",
        f"- optimizer_betas: `{training.get('optimizer_betas')}`",
        f"- loss: `{training.get('loss')}`",
        "",
        "## LOSO Split",
        f"- strategy: `{split.get('strategy')}`",
        f"- n_folds: `{split.get('n_folds')}`",
        f"- fold_subject_ids: `{split.get('fold_subject_ids')}`",
        f"- validation_strategy: `{split.get('validation_strategy')}`",
        f"- best_model_selected_by: `{split.get('best_model_selected_by')}`",
        "",
        "## Runtime",
        f"- device: `{runtime.get('device')}`",
        f"- batch_size: `{runtime.get('batch_size')}`",
        f"- skip_existing: `{runtime.get('skip_existing')}`",
        f"- command_line: `{runtime.get('command_line')}`",
        "",
        "## Environment",
        f"- git_commit: `{environment.get('git_commit')}`",
        f"- git_branch: `{environment.get('git_branch')}`",
        f"- git_dirty: `{environment.get('git_dirty')}`",
        f"- python_version: `{environment.get('python_version')}`",
        f"- torch_version: `{environment.get('torch_version')}`",
        f"- numpy_version: `{environment.get('numpy_version')}`",
        f"- cuda_available: `{environment.get('cuda_available')}`",
        f"- mps_available: `{environment.get('mps_available')}`",
        "",
        "## Outputs",
        f"- output_dir: `{outputs.get('output_dir')}`",
        f"- summary_json: `{outputs.get('summary_json')}`",
        f"- summary_csv: `{outputs.get('summary_csv')}`",
        f"- fold_dirs: `{outputs.get('fold_dirs')}`",
        "",
    ]
    return "\n".join(lines)


def build_loso_experiment_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    dataset_name: str | None = None,
    input_domain: str = "time",
    training_config: dict[str, Any] | None = None,
    runtime_config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    script_path: str | Path | None = None,
    command_line: str | list[str] | None = None,
    run_status: str = "trained",
    run_started_at: str | None = None,
    run_ended_at: str | None = None,
    project_root: str | Path | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    dataset_info = collect_dataset_info(dataset_root, dataset_name=dataset_name)
    runtime = dict(runtime_config or {})
    runtime.setdefault("command_line", " ".join(command_line) if isinstance(command_line, list) else command_line)
    runtime.setdefault("script_path", str(script_path) if script_path is not None else None)
    runtime.setdefault("config_path", str(config_path) if config_path is not None else None)

    manifest = {
        "manifest_version": 1,
        "experiment_type": "LOSO activity three-class classification",
        "run_status": run_status,
        "run_started_at": run_started_at,
        "run_ended_at": run_ended_at,
        "note": note,
        "dataset": dataset_info,
        "model": build_model_info(
            input_domain=input_domain,
            dataset_info=dataset_info,
            training_config=training_config,
            output_dir=output_dir,
        ),
        "training": build_training_info(training_config=training_config, output_dir=output_dir),
        "split": collect_loso_split_info(dataset_root),
        "runtime": runtime,
        "environment": collect_environment_info(project_root=project_root),
        "outputs": collect_output_info(output_dir),
    }
    return _jsonable(manifest)


def write_loso_experiment_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    dataset_name: str | None = None,
    input_domain: str = "time",
    training_config: dict[str, Any] | None = None,
    runtime_config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    script_path: str | Path | None = None,
    command_line: str | list[str] | None = None,
    run_status: str = "trained",
    run_started_at: str | None = None,
    run_ended_at: str | None = None,
    project_root: str | Path | None = None,
    note: str | None = None,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_loso_experiment_manifest(
        dataset_root=dataset_root,
        output_dir=root,
        dataset_name=dataset_name,
        input_domain=input_domain,
        training_config=training_config,
        runtime_config=runtime_config,
        config_path=config_path,
        script_path=script_path,
        command_line=command_line,
        run_status=run_status,
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
        project_root=project_root,
        note=note,
    )

    json_path = root / MANIFEST_JSON_NAME
    md_path = root / MANIFEST_MD_NAME
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    md_path.write_text(build_markdown(manifest), encoding="utf-8")
    return json_path, md_path
