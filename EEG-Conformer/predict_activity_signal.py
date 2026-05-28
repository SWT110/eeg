"""
EEG-Conformer – Activity Signal Predictor
==========================================
Given a new EEG xlsx file and a trained final-model directory, predicts
which activity class (0 / 1 / 2) the signal belongs to.

Preprocessing mirrors the global-dataset builder:
  * The shared 21 EEG channels are extracted in a fixed order when present
  * Sliding windows are cut using the same window_seconds / stride_seconds
    stored in train_config.json
  * Windows are Z-score normalised with normalization_stats.json (mean/std)

Inference:
  * Each window is scored by final_model.pt
  * Class probabilities are averaged across windows (softmax mean)
  * The class with the highest mean probability is the prediction

Output (always printed; optionally saved to JSON):
  predicted_label       – int  (0 / 1 / 2)
  predicted_activity    – str  (e.g. "e_1")
  average_probabilities – list[float]
  n_windows             – int

Usage
-----
    python predict_activity_signal.py \\
        --input /path/to/new_signal.xlsx \\
        --model-dir /path/to/local_artifacts/outputs/activity_final \
        [--output-json /path/to/result.json] \\
        [--window-seconds 5.0] \\
        [--device cpu]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import ACTIVITY_FINAL_OUTPUT_DIR

DEFAULT_MODEL_DIR = ACTIVITY_FINAL_OUTPUT_DIR
DEFAULT_DEVICE = "cpu"

_XLSX_MOD_PATH = (
    EEG_ROOT / "eeg-data-processing" / "data_to_list" / "8.xlsx_to_npy_dataset.py"
)
_LOSO_PATH = PROJECT_ROOT / "train_activity_loso.py"


# ---------------------------------------------------------------------------
# Lazy module imports (xlsx helper + model)
# ---------------------------------------------------------------------------

def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_xlsx_mod = _load_module_from_path("_xlsx_to_npy", _XLSX_MOD_PATH)
_loso_mod = _load_module_from_path("_train_activity_loso", _LOSO_PATH)

load_xlsx_file = _xlsx_mod.load_xlsx_file
extract_windows = _xlsx_mod.extract_windows
ActivityConformer = _loso_mod.ActivityConformer


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_model_artifacts(model_dir: str | Path) -> dict:
    """Load deployment artifacts from model_dir.

    Raises FileNotFoundError if any required file is missing.
    Returns a dict with keys: train_config, norm_stats, label_map, model_dir.
    """
    root = Path(model_dir)
    required = (
        "final_model.pt",
        "train_config.json",
        "normalization_stats.json",
        "label_map.json",
    )
    for fname in required:
        if not (root / fname).exists():
            raise FileNotFoundError(f"Missing {fname} in {root}")

    with open(root / "train_config.json", encoding="utf-8") as fh:
        train_config = json.load(fh)
    with open(root / "normalization_stats.json", encoding="utf-8") as fh:
        norm_stats = json.load(fh)
    with open(root / "label_map.json", encoding="utf-8") as fh:
        label_map = json.load(fh)

    return {
        "train_config": train_config,
        "norm_stats": norm_stats,
        "label_map": label_map,
        "model_dir": root,
    }


def build_model_from_artifacts(
    artifacts: dict,
    device: str,
) -> torch.nn.Module:
    """Instantiate ActivityConformer and load weights from final_model.pt."""
    cfg = artifacts["train_config"]
    model = ActivityConformer(
        n_channels=cfg["n_channels"],
        n_times=cfg["n_times"],
        n_classes=cfg["n_classes"],
        emb_size=cfg.get("emb_size", 40),
        depth=cfg.get("depth", 6),
        num_heads=cfg.get("num_heads", 5),
        dropout=cfg.get("dropout", 0.5),
    )
    ckpt = torch.load(
        str(artifacts["model_dir"] / "final_model.pt"),
        map_location=device,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def extract_windows_from_xlsx(
    xlsx_path: str | Path,
    window_seconds: float,
    stride_seconds: float,
) -> np.ndarray | None:
    """Load xlsx and extract EEG windows as (N, C, T) float32 array.

    Returns None if the file cannot be loaded or produces no windows.
    """
    loaded = load_xlsx_file(Path(xlsx_path))
    if loaded is None:
        return None
    data, times = loaded
    windows = extract_windows(data, times, window_seconds, stride_seconds)
    if not windows:
        return None
    arr = np.stack(windows, axis=0)          # (N, T, C)
    return arr.transpose(0, 2, 1).astype(np.float32)  # (N, C, T)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def softmax_average_probabilities(
    model: torch.nn.Module,
    windows_nct: np.ndarray,
    mean: float,
    std: float,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Run windows through model; return mean softmax probability vector.

    Parameters
    ----------
    model       : trained ActivityConformer (eval mode)
    windows_nct : (N, C, T) float32 array of raw windows
    mean, std   : normalization statistics from normalization_stats.json
    device      : torch device string

    Returns
    -------
    (n_classes,) float64 array of mean probabilities
    """
    X = (windows_nct - mean) / std
    X = np.expand_dims(X, axis=1)          # (N, 1, C, T)

    device_obj = torch.device(device)
    all_probs: list[np.ndarray] = []

    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).float().to(device_obj)
        with torch.no_grad():
            _, logits = model(batch)
            probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    stacked = np.concatenate(all_probs, axis=0)  # (N, n_classes)
    return stacked.mean(axis=0)                   # (n_classes,)


# ---------------------------------------------------------------------------
# Full prediction pipeline
# ---------------------------------------------------------------------------

def predict_signal(
    xlsx_path: str | Path,
    model_dir: str | Path,
    device: str = DEFAULT_DEVICE,
    output_json: str | Path | None = None,
    window_seconds: float | None = None,
    stride_seconds: float | None = None,
) -> dict:
    """End-to-end prediction for a single xlsx EEG file.

    Parameters
    ----------
    xlsx_path       : path to the input xlsx file
    model_dir       : directory containing final model artifacts
    device          : torch device string (default: "cpu")
    output_json     : if given, save result dict to this path
    window_seconds  : override window length; falls back to train_config.json
    stride_seconds  : override stride; falls back to train_config.json

    Returns
    -------
    dict with keys:
        predicted_label       (int)
        predicted_activity    (str)
        average_probabilities (list[float])
        n_windows             (int)
        input_file            (str)
        model_dir             (str)
    """
    artifacts = load_model_artifacts(model_dir)
    cfg = artifacts["train_config"]

    win_sec = window_seconds or cfg.get("window_seconds")
    stride_sec = stride_seconds or cfg.get("stride_seconds") or win_sec

    if win_sec is None:
        raise ValueError(
            "--window-seconds not provided and 'window_seconds' not in train_config.json"
        )
    if stride_sec is None:
        stride_sec = win_sec

    windows_nct = extract_windows_from_xlsx(xlsx_path, win_sec, stride_sec)
    if windows_nct is None or len(windows_nct) == 0:
        raise ValueError(f"No valid EEG windows extracted from: {xlsx_path}")

    model = build_model_from_artifacts(artifacts, device)
    norm = artifacts["norm_stats"]
    avg_probs = softmax_average_probabilities(
        model, windows_nct, norm["mean"], norm["std"], device
    )

    predicted_label = int(np.argmax(avg_probs))

    # Reverse label_map  {"e_1": 0, "e_2": 1, "e_3": 2}  →  {0: "e_1", ...}
    label_map = artifacts["label_map"]
    label_to_activity = {v: k for k, v in label_map.items()}
    predicted_activity = label_to_activity.get(
        predicted_label, f"activity_{predicted_label + 1}"
    )

    result = {
        "predicted_label": predicted_label,
        "predicted_activity": predicted_activity,
        "average_probabilities": avg_probs.tolist(),
        "n_windows": len(windows_nct),
        "input_file": str(xlsx_path),
        "model_dir": str(model_dir),
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="ascii") as fh:
            json.dump(result, fh, indent=2)
        print(f"Result saved to: {output_json}")

    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict activity class for a new EEG xlsx file using the final model"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input EEG xlsx file",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Final model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the prediction result as JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Torch device string (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Window length in seconds (overrides train_config.json)",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=None,
        help="Stride in seconds (overrides train_config.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    predict_signal(
        xlsx_path=args.input,
        model_dir=args.model_dir,
        device=args.device,
        output_json=args.output_json,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )


if __name__ == "__main__":
    main()
