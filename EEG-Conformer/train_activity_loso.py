"""
EEG-Conformer – LOSO Activity Three-Class Training Script
==========================================================
Faithful to the original upstream EEG-Conformer architecture, but with:

  * Dynamic n_channels / n_times dimensions (no hard-coded channel count)
  * Global dataset input format  (X / y / subject_ids / metadata.json)
  * LOSO split via --test-subject-id
  * Standardisation computed on train split only
  * CrossEntropyLoss + Adam(lr=0.0002, betas=(0.5, 0.999))  – as in original
  * Per-epoch test evaluation and best_acc tracking
  * JSON metrics output

Usage
-----
    python train_activity_loso.py \\
        --dataset-root  /path/to/global_activity_dataset \\
        --test-subject-id 1 \\
        --epochs 200 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Paths & top-level constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import ACTIVITY_LOSO_OUTPUT_DIR, GLOBAL_ACTIVITY_DATASET_DIR

DEFAULT_DATASET_ROOT = GLOBAL_ACTIVITY_DATASET_DIR
DEFAULT_OUTPUT_DIR = ACTIVITY_LOSO_OUTPUT_DIR
DEFAULT_TEST_SUBJECT_ID = 1
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 72
DEFAULT_LR = 0.0002
DEFAULT_BETAS = (0.5, 0.999)
DEFAULT_DEVICE = "cuda:0"
DEFAULT_EMB_SIZE = 40
DEFAULT_DEPTH = 6
DEFAULT_NUM_HEADS = 5
DEFAULT_DROPOUT = 0.5
DEFAULT_ENV_NAME = "eegconformer310"
AUTO_RERUN_ENV_VAR = "TRAIN_ACTIVITY_LOSO_PROJECT_ENV_ACTIVE"


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

class RuntimeConfig(NamedTuple):
    dataset_root: Path
    test_subject_id: int
    epochs: int
    batch_size: int
    lr: float
    device: str
    output_dir: Path
    seed: int
    class_weights: list[float] | None = None


# ---------------------------------------------------------------------------
# Model – faithful to original EEG-Conformer architecture
# (PatchEmbedding + MultiHeadAttention + ResidualAdd +
#  FeedForwardBlock + TransformerEncoder + flatten-fc head)
# but without einops so it works in the base conda env.
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Shallow-CNN stem that maps (B, 1, C, T) → (B, n_patches, emb_size).

    Architecture mirrors the original EEG-Conformer shallownet branch with the
    spatial conv kernel made dynamic on ``n_channels``.
    """

    def __init__(self, n_channels: int, emb_size: int = 40, dropout: float = 0.5) -> None:
        super().__init__()
        self.shallownet = nn.Sequential(
            # temporal filter
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            # spatial filter – depth-wise across all EEG channels
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # patch pooling (same kernel/stride as original)
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)      # (B, 40, 1, n_patches)
        x = self.projection(x)      # (B, emb_size, 1, n_patches)
        x = x.squeeze(2)            # (B, emb_size, n_patches)
        x = x.transpose(1, 2)       # (B, n_patches, emb_size)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention matching the original EEG-Conformer layout,
    but using view+transpose instead of einops.rearrange."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, N, _ = x.shape
        h = self.num_heads
        d = self.emb_size // h

        def _reshape(t: Tensor) -> Tensor:
            return t.view(B, N, h, d).transpose(1, 2)  # (B, h, N, d)

        queries = _reshape(self.queries(x))
        keys = _reshape(self.keys(x))
        values = _reshape(self.values(x))

        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5  # faithful to original
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum("bhal,bhlv->bhav", att, values)  # (B, h, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, self.emb_size)
        return self.projection(out)


class ResidualAdd(nn.Module):
    """Residual wrapper matching the original EEG-Conformer layout."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    """Position-wise FFN matching the original EEG-Conformer layout."""

    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5) -> None:
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """One Transformer encoder block – identical structure to original."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int = 5,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ) -> None:
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, num_heads: int = 5) -> None:
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])


class ClassificationHead(nn.Module):
    """flatten + fc head with dynamic flat_size.

    Returns (token_features, logits) to match the original interface:
        tok, outputs = model(img)
    """

    def __init__(self, emb_size: int, n_patches: int, n_classes: int) -> None:
        super().__init__()
        flat_size = emb_size * n_patches
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class ActivityConformer(nn.Module):
    """EEG-Conformer for activity three-class LOSO classification.

    Architecture is faithful to the original (shallownet PatchEmbedding →
    stacked TransformerEncoderBlocks → flatten + fc head) but adapts spatial
    conv kernel and fc input size to the actual data dimensions.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 3,
        emb_size: int = 40,
        depth: int = 6,
        num_heads: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        n_patches = compute_n_patches(n_times)
        self.patch_embedding = PatchEmbedding(n_channels, emb_size, dropout)
        self.encoder = TransformerEncoder(depth, emb_size, num_heads)
        self.cls_head = ClassificationHead(emb_size, n_patches, n_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return self.cls_head(x)


def compute_n_patches(n_times: int) -> int:
    """Number of patch tokens produced by PatchEmbedding for ``n_times`` samples.

    Trace:
      temporal conv (1,25) stride (1,1)  → T' = n_times - 24
      AvgPool2d kernel (1,75) stride (1,15) → n_patches = (T' - 75) // 15 + 1
    """
    if n_times < 99:
        raise ValueError(
            f"Input sequence is too short for the original EEG-Conformer stem: "
            f"n_times={n_times}, need at least 99"
        )
    t_after_temporal = n_times - 24
    return (t_after_temporal - 75) // 15 + 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_global_dataset(
    dataset_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, subject_ids from the global activity dataset directory."""
    root = Path(dataset_root)
    X = np.load(root / "X.npy")
    y = np.load(root / "y.npy")
    subject_ids = np.load(root / "subject_ids.npy")

    if len(X) != len(y) or len(X) != len(subject_ids):
        raise ValueError(
            f"Mismatched array lengths in {root}: "
            f"X={len(X)}, y={len(y)}, subject_ids={len(subject_ids)}"
        )
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False), subject_ids.astype(np.int64, copy=False)


def loso_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    test_subject_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split global arrays into train/test by test_subject_id.

    Returns
    -------
    train_X, train_y, test_X, test_y
    """
    test_mask = subject_ids == test_subject_id
    train_mask = ~test_mask

    if not train_mask.any():
        raise ValueError(f"test_subject_id={test_subject_id} leaves no training samples")
    if not test_mask.any():
        raise ValueError(f"test_subject_id={test_subject_id} not found in subject_ids")

    # add conv-channel dim (B, C, T) → (B, 1, C, T)
    train_X = np.expand_dims(X[train_mask], axis=1)
    test_X = np.expand_dims(X[test_mask], axis=1)
    return train_X, y[train_mask], test_X, y[test_mask]


def standardize_by_train(
    train_X: np.ndarray,
    test_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalise using training set statistics only (as in original)."""
    mean = float(train_X.mean())
    std = float(train_X.std())
    if std == 0.0:
        raise ValueError("Training data std is zero – cannot standardise")
    return (train_X - mean) / std, (test_X - mean) / std


def build_dataloaders(
    dataset_root: str | Path,
    test_subject_id: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, int, int, int, int, int]:
    """Load dataset, split LOSO, standardise, return loaders and shape info.

    Returns
    -------
    train_loader, test_loader, n_channels, n_times, n_classes,
    n_train_samples, n_test_samples
    """
    X, y, subject_ids = load_global_dataset(dataset_root)
    train_X, train_y, test_X, test_y = loso_split(X, y, subject_ids, test_subject_id)
    train_X, test_X = standardize_by_train(train_X, test_X)

    n_channels = train_X.shape[2]
    n_times = train_X.shape[3]
    n_classes = int(y.max()) + 1

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_X).float(),
            torch.from_numpy(train_y).long(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(test_X).float(),
            torch.from_numpy(test_y).long(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return (
        train_loader, test_loader,
        n_channels, n_times, n_classes,
        len(train_X), len(test_X),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean_loss, accuracy) over the dataloader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            _, logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_X)
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_samples += len(batch_X)

    return total_loss / total_samples, total_correct / total_samples


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays for the full dataloader (no grad)."""
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            _, logits = model(batch_X)
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(batch_y.tolist())
    return np.array(all_true, dtype=np.int64), np.array(all_pred, dtype=np.int64)


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


# ---------------------------------------------------------------------------
# Training-history persistence
# ---------------------------------------------------------------------------

def write_epoch_history_files(
    fold_dir: str | Path,
    history: list[dict],
    metadata: dict,
) -> tuple[Path, Path]:
    """Persist per-epoch training history in CSV + JSON form."""
    root = Path(fold_dir)
    root.mkdir(parents=True, exist_ok=True)

    csv_path = root / "epoch_history.csv"
    json_path = root / "epoch_history.json"

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "best_test_acc",
        "is_best_epoch",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row.get(name) for name in fieldnames})

    payload = dict(metadata)
    payload["history"] = history
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return csv_path, json_path


def append_training_log(log_path: str | Path, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(message)
        fh.write("\n")


def parse_class_weights(raw: str | list[float] | tuple[float, ...] | None) -> list[float] | None:
    if raw is None:
        return None

    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",")]
    else:
        pieces = [str(item).strip() for item in raw]

    if not pieces or any(piece == "" for piece in pieces):
        raise ValueError("class weights must be a comma-separated list like '3,3,1'")

    weights: list[float] = []
    for piece in pieces:
        try:
            value = float(piece)
        except ValueError as exc:
            raise ValueError("class weights must be numeric, e.g. '3,3,1'") from exc
        if value < 0:
            raise ValueError("class weights must be >= 0")
        weights.append(value)

    if not any(value > 0 for value in weights):
        raise ValueError("class weights must contain at least one value > 0")
    return weights


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_loso_fold(
    dataset_root: str | Path,
    test_subject_id: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: str | Path,
    seed: int = 42,
    emb_size: int = DEFAULT_EMB_SIZE,
    depth: int = DEFAULT_DEPTH,
    num_heads: int = DEFAULT_NUM_HEADS,
    dropout: float = DEFAULT_DROPOUT,
    class_weights: list[float] | None = None,
) -> Path:
    """Train one LOSO fold and return path to metrics.json."""

    # reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device_obj = torch.device(device)

    (
        train_loader, test_loader,
        n_channels, n_times, n_classes,
        n_train_samples, n_test_samples,
    ) = build_dataloaders(dataset_root, test_subject_id, batch_size)

    model = ActivityConformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        emb_size=emb_size,
        depth=depth,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device_obj)

    # Adam + CrossEntropyLoss – identical hyper-parameters to original
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=DEFAULT_BETAS)
    if class_weights is not None and len(class_weights) != n_classes:
        raise ValueError(
            f"class_weights length ({len(class_weights)}) must match n_classes ({n_classes})"
        )
    class_weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32, device=device_obj)
        if class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor).to(device_obj)

    fold_dir = Path(output_dir) / f"fold_subject_{test_subject_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    best_acc = 0.0
    best_epoch: int | None = None
    aver_acc = 0.0
    best_y_true: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None
    epoch_history: list[dict] = []

    history_metadata = {
        "test_subject_id": test_subject_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "class_weights": class_weights,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
    }

    def log(message: str) -> None:
        print(message)
        append_training_log(log_path, message)

    log(
        f"\n[LOSO fold subject={test_subject_id}] "
        f"train={n_train_samples}  test={n_test_samples}  "
        f"shape=(1,{n_channels},{n_times})  classes={n_classes}"
    )

    for epoch in range(epochs):
        # ---- train ----
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

        # ---- evaluate on test subject (every epoch, like original) ----
        test_loss, test_acc = evaluate(model, test_loader, criterion, device_obj)

        aver_acc += test_acc
        is_best_epoch = test_acc > best_acc
        if is_best_epoch:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_y_true, best_y_pred = collect_predictions(model, test_loader, device_obj)
            torch.save(
                {
                    "epoch": epoch,
                    "test_subject_id": test_subject_id,
                    "state_dict": model.state_dict(),
                    "n_channels": n_channels,
                    "n_times": n_times,
                    "n_classes": n_classes,
                    "emb_size": emb_size,
                    "depth": depth,
                    "num_heads": num_heads,
                },
                fold_dir / "best_model.pt",
            )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "best_test_acc": round(best_acc, 6),
            "is_best_epoch": is_best_epoch,
        }
        epoch_history.append(epoch_record)
        history_csv_path, history_json_path = write_epoch_history_files(
            fold_dir=fold_dir,
            history=epoch_history,
            metadata=history_metadata,
        )

        log(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  "
            f"best={best_acc:.4f}"
        )

    aver_acc /= epochs

    metrics: dict = {
        "test_subject_id": test_subject_id,
        "best_test_acc": best_acc,
        "average_test_acc": aver_acc,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "class_weights": class_weights,
        "best_epoch": best_epoch,
        "epoch_history_csv": str(history_csv_path),
        "epoch_history_json": str(history_json_path),
        "train_log": str(log_path),
    }

    if best_y_true is not None and best_y_pred is not None:
        np.savez(fold_dir / "test_predictions.npz", y_true=best_y_true, y_pred=best_y_pred)
        cm = confusion_matrix_from_arrays(best_y_true, best_y_pred, n_classes)
        pcm = per_class_metrics_from_cm(cm)
        metrics["confusion_matrix"] = cm
        metrics["per_class_metrics"] = pcm
        metrics["macro_f1"] = macro_f1_from_per_class(pcm)

    metrics_path = fold_dir / "metrics.json"
    with open(metrics_path, "w", encoding="ascii") as fh:
        json.dump(metrics, fh, indent=2)

    log(f"\nFold subject={test_subject_id}: best_acc={best_acc:.4f}  aver_acc={aver_acc:.4f}")
    log(f"Checkpoint:       {fold_dir / 'best_model.pt'}")
    log(f"Metrics:          {metrics_path}")
    log(f"Epoch history CSV:{history_csv_path}")
    log(f"Epoch history JSON:{history_json_path}")
    log(f"Train log:        {log_path}")

    return metrics_path


# ---------------------------------------------------------------------------
# CLI plumbing
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
    return Path(sys.executable).resolve() == (project_env_prefix() / "bin" / "python").resolve()


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


def prompt_text(prompt_text: str, default: str | None = None) -> str:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        print("Please enter a value.")


def build_noninteractive_example() -> str:
    return (
        f"{build_project_local_train_command()} "
        f"--dataset-root {shlex.quote(str(DEFAULT_DATASET_ROOT))} "
        f"--test-subject-id {DEFAULT_TEST_SUBJECT_ID} "
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
    test_subject_id: int | None,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    device: str | None,
    output_dir: Path | str | None,
    seed: int | None,
    class_weights: str | list[float] | tuple[float, ...] | None = None,
) -> RuntimeConfig:
    missing_flags: list[str] = []
    if dataset_root is None:
        missing_flags.append("--dataset-root")
    if test_subject_id is None:
        missing_flags.append("--test-subject-id")
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

    resolved_test_subject_id = (
        prompt_int("Test subject id (LOSO fold)", default=DEFAULT_TEST_SUBJECT_ID, minimum=1)
        if test_subject_id is None
        else int(test_subject_id)
    )

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
    resolved_class_weights = parse_class_weights(class_weights)

    return RuntimeConfig(
        dataset_root=resolved_dataset_root,
        test_subject_id=resolved_test_subject_id,
        epochs=resolved_epochs,
        batch_size=resolved_batch_size,
        lr=resolved_lr,
        device=resolved_device,
        output_dir=resolved_output_dir,
        seed=resolved_seed,
        class_weights=resolved_class_weights,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEG-Conformer for LOSO activity three-class classification"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Global activity dataset directory (containing X.npy, y.npy, subject_ids.npy, metadata.json)",
    )
    parser.add_argument(
        "--test-subject-id",
        type=int,
        default=DEFAULT_TEST_SUBJECT_ID,
        help="Subject id used as the LOSO test fold (default: 1)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
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
        help="Parent directory for fold checkpoints and metrics",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    runtime_argv = list(sys.argv[1:] if argv is None else argv)
    if runtime_argv:
        args = parse_args(runtime_argv)
        maybe_rerun_in_project_env(runtime_argv, str(args.device))
        config = resolve_runtime_config(
            dataset_root=args.dataset_root,
            test_subject_id=args.test_subject_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
            class_weights=args.class_weights,
        )
    else:
        maybe_rerun_in_project_env([], DEFAULT_DEVICE)
        config = resolve_runtime_config(
            dataset_root=None,
            test_subject_id=None,
            epochs=None,
            batch_size=None,
            lr=None,
            device=None,
            output_dir=None,
            seed=None,
            class_weights=None,
        )

    train_loso_fold(
        dataset_root=config.dataset_root,
        test_subject_id=config.test_subject_id,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        output_dir=config.output_dir,
        seed=config.seed,
        class_weights=config.class_weights,
    )


if __name__ == "__main__":
    main()
