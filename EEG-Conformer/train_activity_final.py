"""
EEG-Conformer – Final Deployment Model Training
================================================
Trains on ALL data (no held-out subject).  Uses a chronological 80/20
train/val split within each record_id so the model sees the full data
diversity while still having a validation signal that is time-leak-free.

Deployment artifacts saved to ``--output-dir``:
  final_model.pt           – best checkpoint (by val accuracy)
  train_config.json        – architecture + hyper-parameters
  normalization_stats.json – global mean/std for inference-time normalisation
  label_map.json           – str_key → int_label map from metadata.json
  val_metrics.json         – best_val_acc, macro_f1, confusion_matrix

Design decision vs. LOSO folds
--------------------------------
LOSO folds measure *generalisation to new subjects*.  The final deployment
model is trained on all subjects; the val split is purely for early-stopping
and to sanity-check training (no subject-level hold-out).

Usage
-----
    python train_activity_final.py \\
        --dataset-root /path/to/global_activity_dataset \\
        --output-dir   /path/to/local_artifacts/outputs/activity_final \
        --epochs 200   \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Paths & top-level constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import ACTIVITY_FINAL_OUTPUT_DIR, GLOBAL_ACTIVITY_DATASET_DIR

DEFAULT_DATASET_ROOT = GLOBAL_ACTIVITY_DATASET_DIR
DEFAULT_OUTPUT_DIR = ACTIVITY_FINAL_OUTPUT_DIR
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 72
DEFAULT_LR = 0.0002
DEFAULT_BETAS = (0.5, 0.999)
DEFAULT_DEVICE = "cuda:0"
DEFAULT_EMB_SIZE = 40
DEFAULT_DEPTH = 6
DEFAULT_NUM_HEADS = 5
DEFAULT_DROPOUT = 0.5
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_ENV_NAME = "eegconformer310"
AUTO_RERUN_ENV_VAR = "TRAIN_ACTIVITY_FINAL_PROJECT_ENV_ACTIVE"


# ---------------------------------------------------------------------------
# Import model and helpers from train_activity_loso.py
# ---------------------------------------------------------------------------

_LOSO_PATH = PROJECT_ROOT / "train_activity_loso.py"


def _load_loso_module():
    spec = importlib.util.spec_from_file_location("_train_activity_loso", _LOSO_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_loso = _load_loso_module()

ActivityConformer = _loso.ActivityConformer
compute_n_patches = _loso.compute_n_patches
evaluate = _loso.evaluate
collect_predictions = _loso.collect_predictions
confusion_matrix_from_arrays = _loso.confusion_matrix_from_arrays
per_class_metrics_from_cm = _loso.per_class_metrics_from_cm
macro_f1_from_per_class = _loso.macro_f1_from_per_class


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

class RuntimeConfig(NamedTuple):
    dataset_root: Path
    epochs: int
    batch_size: int
    lr: float
    device: str
    output_dir: Path
    seed: int
    val_fraction: float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_global_dataset_full(
    dataset_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load X, y, record_ids, window_indices and metadata from dataset_root.

    Returns (X, y, record_ids, window_indices, metadata).
    Raises FileNotFoundError if any required file is missing.
    """
    root = Path(dataset_root)
    required = ("X.npy", "y.npy", "record_ids.npy", "window_indices.npy", "metadata.json")
    for fname in required:
        if not (root / fname).exists():
            raise FileNotFoundError(f"Missing {fname} in {root}")

    X = np.load(root / "X.npy").astype(np.float32, copy=False)
    y = np.load(root / "y.npy").astype(np.int64, copy=False)
    record_ids = np.load(root / "record_ids.npy").astype(np.int64, copy=False)
    window_indices = np.load(root / "window_indices.npy").astype(np.int64, copy=False)

    with open(root / "metadata.json", encoding="utf-8") as fh:
        metadata = json.load(fh)

    return X, y, record_ids, window_indices, metadata


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def final_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
    window_indices: np.ndarray,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 80/20 split within each record_id.

    For each unique record_id the windows are ordered by their window_indices
    (ascending).  The first ``(1 - val_fraction)`` fraction goes to training
    and the remainder to validation.

    Returns (train_X, train_y, val_X, val_y) where X arrays have the
    conv-channel dim prepended: shape (N, 1, C, T).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    train_mask = np.zeros(len(X), dtype=bool)
    val_mask = np.zeros(len(X), dtype=bool)

    for rid in np.unique(record_ids):
        rec_mask = record_ids == rid
        global_indices = np.where(rec_mask)[0]
        win_idx_for_rec = window_indices[rec_mask]
        sorted_order = np.argsort(win_idx_for_rec, kind="stable")
        sorted_global = global_indices[sorted_order]
        n = len(sorted_global)
        n_train = max(1, int(n * (1.0 - val_fraction)))
        train_mask[sorted_global[:n_train]] = True
        if n_train < n:
            val_mask[sorted_global[n_train:]] = True

    train_X = np.expand_dims(X[train_mask], axis=1)
    if val_mask.any():
        val_X = np.expand_dims(X[val_mask], axis=1)
        val_y = y[val_mask]
    else:
        val_X = np.empty((0, 1) + X.shape[1:], dtype=X.dtype)
        val_y = np.empty((0,), dtype=y.dtype)

    return train_X, y[train_mask], val_X, val_y


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def standardize_for_deployment(
    train_X: np.ndarray,
    val_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Z-score using train statistics. Returns (train_norm, val_norm, mean, std)."""
    mean = float(train_X.mean())
    std = float(train_X.std())
    if std == 0.0:
        raise ValueError("Training data std is zero – cannot standardise")
    train_norm = (train_X - mean) / std
    val_norm = (val_X - mean) / std if len(val_X) > 0 else val_X
    return train_norm, val_norm, mean, std


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_final_model(
    dataset_root: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: str | Path,
    seed: int = 42,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    emb_size: int = DEFAULT_EMB_SIZE,
    depth: int = DEFAULT_DEPTH,
    num_heads: int = DEFAULT_NUM_HEADS,
    dropout: float = DEFAULT_DROPOUT,
) -> Path:
    """Train final deployment model and save all artifacts.

    Returns the path to ``val_metrics.json``.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, record_ids, window_indices, metadata = load_global_dataset_full(dataset_root)
    train_X, train_y, val_X, val_y = final_train_val_split(
        X, y, record_ids, window_indices, val_fraction
    )
    train_X, val_X, mean_val, std_val = standardize_for_deployment(train_X, val_X)

    n_channels = train_X.shape[2]
    n_times = train_X.shape[3]
    n_classes = int(y.max()) + 1
    val_has_data = len(val_X) > 0

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_X).float(),
            torch.from_numpy(train_y).long(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader: DataLoader | None = None
    if val_has_data:
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(val_X).float(),
                torch.from_numpy(val_y).long(),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

    model = ActivityConformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        emb_size=emb_size,
        depth=depth,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=DEFAULT_BETAS)
    criterion = nn.CrossEntropyLoss().to(device_obj)

    best_val_acc = -1.0
    best_y_true: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None

    print(
        f"\n[Final model] train={len(train_X)}  val={len(val_X)}  "
        f"shape=(1,{n_channels},{n_times})  classes={n_classes}"
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device_obj)
            batch_y = batch_y.to(device_obj)
            optimizer.zero_grad()
            _, logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_X)
            running_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            running_samples += len(batch_X)

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples

        if val_has_data and val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device_obj)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_y_true, best_y_pred = collect_predictions(
                    model, val_loader, device_obj
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "n_channels": n_channels,
                        "n_times": n_times,
                        "n_classes": n_classes,
                        "emb_size": emb_size,
                        "depth": depth,
                        "num_heads": num_heads,
                    },
                    output_dir / "final_model.pt",
                )
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}"
            )
        else:
            # No val data: save at last epoch
            if epoch == epochs - 1:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "n_channels": n_channels,
                        "n_times": n_times,
                        "n_classes": n_classes,
                        "emb_size": emb_size,
                        "depth": depth,
                        "num_heads": num_heads,
                    },
                    output_dir / "final_model.pt",
                )
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
            )

    # ------------------------------------------------------------------
    # Save deployment artifacts
    # ------------------------------------------------------------------
    with open(output_dir / "normalization_stats.json", "w", encoding="ascii") as fh:
        json.dump({"mean": mean_val, "std": std_val}, fh, indent=2)

    label_map = metadata.get("label_map", {})
    with open(output_dir / "label_map.json", "w", encoding="ascii") as fh:
        json.dump(label_map, fh, indent=2)

    train_config = {
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "emb_size": emb_size,
        "depth": depth,
        "num_heads": num_heads,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "val_fraction": val_fraction,
        "window_seconds": metadata.get("window_seconds"),
        "stride_seconds": metadata.get("stride_seconds"),
    }
    with open(output_dir / "train_config.json", "w", encoding="ascii") as fh:
        json.dump(train_config, fh, indent=2)

    val_metrics: dict = {
        "best_val_acc": best_val_acc,
        "n_train": len(train_X),
        "n_val": len(val_X),
    }
    if best_y_true is not None and best_y_pred is not None:
        cm = confusion_matrix_from_arrays(best_y_true, best_y_pred, n_classes)
        pcm = per_class_metrics_from_cm(cm)
        val_metrics["confusion_matrix"] = cm
        val_metrics["per_class_metrics"] = pcm
        val_metrics["macro_f1"] = macro_f1_from_per_class(pcm)

    metrics_path = output_dir / "val_metrics.json"
    with open(metrics_path, "w", encoding="ascii") as fh:
        json.dump(val_metrics, fh, indent=2)

    print(f"\nFinal model saved to: {output_dir}")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts: final_model.pt  train_config.json  normalization_stats.json  label_map.json  val_metrics.json")
    return metrics_path


# ---------------------------------------------------------------------------
# CLI plumbing  (mirrors train_activity_loso.py style)
# ---------------------------------------------------------------------------

def cuda_is_usable() -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.cuda.is_available()


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


def project_env_prefix() -> Path:
    return PROJECT_ROOT / ".conda-envs" / DEFAULT_ENV_NAME


def running_inside_project_env() -> bool:
    return (
        Path(sys.executable).resolve()
        == (project_env_prefix() / "bin" / "python").resolve()
    )


def build_project_local_train_command() -> str:
    return (
        f"{shlex.quote(str(project_env_prefix() / 'bin' / 'python'))} "
        f"{shlex.quote(str(Path(__file__).resolve()))}"
    )


def cuda_unavailable_message() -> str:
    return (
        "CUDA is not available in the current PyTorch environment. "
        "Choose 'cpu' or install a PyTorch/CUDA build compatible with the NVIDIA driver.\n"
        "If you created the project-local GPU env with setup_env_jupyter.py, rerun with:\n"
        f"{build_project_local_train_command()}"
    )


def validate_device(device: str) -> str:
    normalized = normalize_device_name(device)
    if not normalized:
        raise ValueError("Device must not be empty")
    try:
        parsed_device = torch.device(normalized)
    except RuntimeError as exc:
        raise ValueError(
            "Invalid device string. Use values like 'cpu', 'cuda', or 'cuda:0'."
        ) from exc
    if parsed_device.type == "cuda" and not cuda_is_usable():
        raise ValueError(cuda_unavailable_message())
    return str(parsed_device)


def maybe_rerun_in_project_env(argv: list[str], device: str) -> None:
    """If CUDA is requested but unavailable, try to rerun in the project conda env."""
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
        [
            str(env_prefix / "bin" / "python"),
            str(Path(__file__).resolve()),
            *argv,
        ],
        check=False,
        env=rerun_env,
    )
    raise SystemExit(completed.returncode)


def prompt_path(prompt_text: str, default: Path | None = None, must_exist: bool = False) -> Path:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        candidate = Path(raw).expanduser() if raw else default
        if candidate is None:
            print("Please enter a path.")
            continue
        if must_exist and not candidate.exists():
            print(f"Path does not exist: {candidate}")
            continue
        return candidate


def prompt_int(prompt_text: str, default: int | None = None, minimum: int | None = None) -> int:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_float(prompt_text: str, default: float | None = None, minimum: float | None = None) -> float:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_text(prompt_text_str: str, default: str | None = None) -> str:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text_str}{default_text}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        print("Please enter a value.")


def build_noninteractive_example() -> str:
    return (
        f"{build_project_local_train_command()} "
        f"--dataset-root {shlex.quote(str(DEFAULT_DATASET_ROOT))} "
        f"--epochs {DEFAULT_EPOCHS} "
        f"--batch-size {DEFAULT_BATCH_SIZE} "
        f"--lr {DEFAULT_LR} "
        f"--device {DEFAULT_DEVICE} "
        f"--output-dir {shlex.quote(str(DEFAULT_OUTPUT_DIR))}"
    )


def ensure_interactive_input_available(missing_flags: list[str]) -> None:
    if not missing_flags or sys.stdin.isatty():
        return
    raise ValueError(
        "Missing required arguments for non-interactive execution: "
        + ", ".join(missing_flags)
        + "\nRun with explicit arguments, for example:\n"
        + build_noninteractive_example()
    )


def resolve_runtime_config(
    dataset_root: Path | str | None,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    device: str | None,
    output_dir: Path | str | None,
    seed: int | None,
    val_fraction: float | None,
) -> RuntimeConfig:
    missing_flags: list[str] = []
    if dataset_root is None:
        missing_flags.append("--dataset-root")
    if epochs is None:
        missing_flags.append("--epochs")
    if batch_size is None:
        missing_flags.append("--batch-size")
    if lr is None:
        missing_flags.append("--lr")
    if device is None:
        missing_flags.append("--device")
    if output_dir is None:
        missing_flags.append("--output-dir")
    ensure_interactive_input_available(missing_flags)

    if dataset_root is None:
        resolved_dataset_root = prompt_path(
            "Dataset root (global activity dataset)", default=DEFAULT_DATASET_ROOT, must_exist=True
        )
    else:
        resolved_dataset_root = Path(dataset_root).expanduser()
        if not resolved_dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {resolved_dataset_root}")

    resolved_epochs = (
        prompt_int("Epochs", default=DEFAULT_EPOCHS, minimum=1)
        if epochs is None
        else int(epochs)
    )

    resolved_batch_size = (
        prompt_int("Batch size", default=DEFAULT_BATCH_SIZE, minimum=1)
        if batch_size is None
        else int(batch_size)
    )

    resolved_lr = (
        prompt_float("Learning rate", default=DEFAULT_LR, minimum=0.0)
        if lr is None
        else float(lr)
    )
    if resolved_lr <= 0:
        raise ValueError("lr must be > 0")

    default_device = DEFAULT_DEVICE if cuda_is_usable() else "cpu"
    if device is None:
        while True:
            candidate = prompt_text("Device", default=default_device)
            try:
                resolved_device = validate_device(candidate)
            except ValueError as exc:
                print(exc)
                continue
            break
    else:
        resolved_device = validate_device(str(device))

    resolved_output_dir = (
        prompt_path("Output directory", default=DEFAULT_OUTPUT_DIR)
        if output_dir is None
        else Path(output_dir).expanduser()
    )

    resolved_seed = 42 if seed is None else int(seed)

    resolved_val_fraction = (
        DEFAULT_VAL_FRACTION if val_fraction is None else float(val_fraction)
    )

    return RuntimeConfig(
        dataset_root=resolved_dataset_root,
        epochs=resolved_epochs,
        batch_size=resolved_batch_size,
        lr=resolved_lr,
        device=resolved_device,
        output_dir=resolved_output_dir,
        seed=resolved_seed,
        val_fraction=resolved_val_fraction,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final EEG-Conformer activity model for deployment (all subjects, no LOSO)"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Global activity dataset directory (X.npy, y.npy, record_ids.npy, window_indices.npy, metadata.json)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for final model artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of each record used as validation (default 0.2)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    runtime_argv = list(sys.argv[1:] if argv is None else argv)
    if runtime_argv:
        args = parse_args(runtime_argv)
        maybe_rerun_in_project_env(runtime_argv, str(args.device))
        config = resolve_runtime_config(
            dataset_root=args.dataset_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
    else:
        maybe_rerun_in_project_env([], DEFAULT_DEVICE)
        config = resolve_runtime_config(
            dataset_root=None,
            epochs=None,
            batch_size=None,
            lr=None,
            device=None,
            output_dir=None,
            seed=None,
            val_fraction=None,
        )

    train_final_model(
        dataset_root=config.dataset_root,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        output_dir=config.output_dir,
        seed=config.seed,
        val_fraction=config.val_fraction,
    )


if __name__ == "__main__":
    main()
