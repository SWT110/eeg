from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import (
    ACTIVITY_API_CACHE_DIR,
    ACTIVITY_LOSO_OUTPUT_DIR,
    LIST_NORMALIZED_DIR,
)

OUTPUTS_ROOT = ACTIVITY_LOSO_OUTPUT_DIR
INPUT_ROOT = LIST_NORMALIZED_DIR
INDEX_HTML_PATH = PROJECT_ROOT / "index.html"
CACHE_DIR = ACTIVITY_API_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DEVICE = os.environ.get("EEG_API_DEVICE", "cpu").strip() or "cpu"
WINDOW_DIR_PATTERN = re.compile(r"^window_(?P<window>.+)_stride_(?P<stride>.+)$")
EEG_FILE_PATTERN = re.compile(r"^(?P<subject>\d+)_e_(?P<label>[123])\.xlsx$")
DEFAULT_LABEL_MAP = {"e_1": 0, "e_2": 1, "e_3": 2}


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_xlsx_mod = _load_module_from_path(
    "_xlsx_to_npy_dataset",
    EEG_ROOT / "eeg-data-processing" / "data_to_list" / "8.xlsx_to_npy_dataset.py",
)
_loso_mod = _load_module_from_path("_train_activity_loso", PROJECT_ROOT / "train_activity_loso.py")

load_xlsx_file = _xlsx_mod.load_xlsx_file
extract_windows = _xlsx_mod.extract_windows
ActivityConformer = _loso_mod.ActivityConformer


class FoldCheckpoint(NamedTuple):
    checkpoint_path: Path
    metrics_path: Path
    test_subject_id: int
    best_test_acc: float
    macro_f1: float
    n_channels: int
    n_times: int
    n_classes: int
    emb_size: int
    depth: int
    num_heads: int
    dropout: float


class SelectedEnsemble(NamedTuple):
    window_dir: Path
    summary_path: Path
    window_seconds: float
    stride_seconds: float
    summary_macro_f1: float
    mean_best_test_acc: float
    std_best_test_acc: float
    n_channels: int
    n_times: int
    n_classes: int
    label_map: dict[str, int]
    checkpoints: list[FoldCheckpoint]


class RuntimeBundle(NamedTuple):
    selected_ensemble: SelectedEnsemble
    models: list[torch.nn.Module]
    device: str
    mean: float
    std: float


_runtime_bundle: RuntimeBundle | None = None


def parse_window_dir_name(name: str) -> tuple[float, float]:
    match = WINDOW_DIR_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Invalid window directory name: {name}")
    return float(match.group("window")), float(match.group("stride"))


def discover_best_loso_ensemble(outputs_root: Path = OUTPUTS_ROOT) -> SelectedEnsemble:
    best_summary: dict | None = None
    best_summary_path: Path | None = None

    for summary_path in sorted(outputs_root.glob("window_*_stride_*/summary.json")):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        if best_summary is None or (
            float(summary.get("macro_f1", -1.0)),
            float(summary.get("mean_best_test_acc", -1.0)),
            -float(summary.get("std_best_test_acc", 1e9)),
        ) > (
            float(best_summary.get("macro_f1", -1.0)),
            float(best_summary.get("mean_best_test_acc", -1.0)),
            -float(best_summary.get("std_best_test_acc", 1e9)),
        ):
            best_summary = summary
            best_summary_path = summary_path

    if best_summary is None or best_summary_path is None:
        raise FileNotFoundError(f"No usable LOSO summary found under {outputs_root}")

    window_dir = best_summary_path.parent
    checkpoints: list[FoldCheckpoint] = []
    for metrics_path in sorted(window_dir.glob("fold_subject_*/metrics.json")):
        checkpoint_path = metrics_path.with_name("best_model.pt")
        if not checkpoint_path.exists():
            continue
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        checkpoints.append(
            FoldCheckpoint(
                checkpoint_path=checkpoint_path,
                metrics_path=metrics_path,
                test_subject_id=int(metrics["test_subject_id"]),
                best_test_acc=float(metrics["best_test_acc"]),
                macro_f1=float(metrics.get("macro_f1", -1.0)),
                n_channels=int(checkpoint.get("n_channels", metrics["n_channels"])),
                n_times=int(checkpoint.get("n_times", metrics["n_times"])),
                n_classes=int(checkpoint.get("n_classes", metrics["n_classes"])),
                emb_size=int(checkpoint.get("emb_size", 40)),
                depth=int(checkpoint.get("depth", 6)),
                num_heads=int(checkpoint.get("num_heads", 5)),
                dropout=0.5,
            )
        )

    if not checkpoints:
        raise FileNotFoundError(f"No usable LOSO checkpoints found under {window_dir}")

    reference = checkpoints[0]
    for item in checkpoints[1:]:
        if (item.n_channels, item.n_times, item.n_classes) != (
            reference.n_channels,
            reference.n_times,
            reference.n_classes,
        ):
            raise ValueError(f"Inconsistent checkpoint shapes under {window_dir}")

    window_seconds, stride_seconds = parse_window_dir_name(window_dir.name)
    return SelectedEnsemble(
        window_dir=window_dir,
        summary_path=best_summary_path,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        summary_macro_f1=float(best_summary.get("macro_f1", -1.0)),
        mean_best_test_acc=float(best_summary.get("mean_best_test_acc", -1.0)),
        std_best_test_acc=float(best_summary.get("std_best_test_acc", -1.0)),
        n_channels=reference.n_channels,
        n_times=reference.n_times,
        n_classes=reference.n_classes,
        label_map=dict(DEFAULT_LABEL_MAP),
        checkpoints=checkpoints,
    )


def build_cache_path(selected_ensemble: SelectedEnsemble) -> Path:
    return CACHE_DIR / f"{selected_ensemble.window_dir.name}_global_normalization_stats.json"


def compute_train_window_stats(
    input_root: Path,
    window_seconds: float,
    stride_seconds: float,
    expected_n_channels: int,
    expected_n_times: int,
) -> tuple[float, float]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    subject_dirs = sorted(
        (candidate for candidate in input_root.iterdir() if candidate.is_dir() and candidate.name.isdigit()),
        key=lambda candidate: int(candidate.name),
    )

    for subject_dir in subject_dirs:
        for path in sorted(subject_dir.glob(f"{subject_dir.name}_e_*.xlsx")):
            if EEG_FILE_PATTERN.match(path.name) is None:
                continue
            loaded = load_xlsx_file(path)
            if loaded is None:
                continue

            data, times = loaded
            try:
                windows = extract_windows(data, times, window_seconds, stride_seconds)
            except ValueError:
                continue

            for window in windows:
                arr = np.asarray(window, dtype=np.float64)
                if arr.shape != (expected_n_times, expected_n_channels):
                    continue
                total_sum += float(arr.sum())
                total_sumsq += float(np.square(arr).sum())
                total_count += int(arr.size)

    if total_count == 0:
        raise ValueError(
            "Could not compute normalization stats: no windows were collected "
            f"from {input_root}"
        )

    mean = total_sum / total_count
    variance = max(total_sumsq / total_count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if std == 0.0:
        raise ValueError("Computed training standard deviation is zero")
    return mean, std


def load_or_compute_normalization_stats(
    selected_ensemble: SelectedEnsemble,
    input_root: Path = INPUT_ROOT,
) -> tuple[float, float]:
    cache_path = build_cache_path(selected_ensemble)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if (
            float(payload.get("window_seconds", -1.0)) == float(selected_ensemble.window_seconds)
            and float(payload.get("stride_seconds", -1.0)) == float(selected_ensemble.stride_seconds)
        ):
            return float(payload["mean"]), float(payload["std"])

    mean, std = compute_train_window_stats(
        input_root=input_root,
        window_seconds=selected_ensemble.window_seconds,
        stride_seconds=selected_ensemble.stride_seconds,
        expected_n_channels=selected_ensemble.n_channels,
        expected_n_times=selected_ensemble.n_times,
    )

    payload = {
        "window_dir": str(selected_ensemble.window_dir),
        "summary_path": str(selected_ensemble.summary_path),
        "window_seconds": selected_ensemble.window_seconds,
        "stride_seconds": selected_ensemble.stride_seconds,
        "mean": mean,
        "std": std,
    }
    with open(cache_path, "w", encoding="ascii") as fh:
        json.dump(payload, fh, indent=2)
    return mean, std


def normalize_device_name(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"cpu", "cuda"}:
        return normalized
    if normalized.startswith("cuda:") and normalized.split(":", maxsplit=1)[1].isdigit():
        return normalized
    return device.strip()


def resolve_device(device: str) -> str:
    resolved = normalize_device_name(device)
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{resolved}' but CUDA is not available in the current PyTorch environment."
        )
    return resolved


def build_runtime_bundle(device: str = DEFAULT_DEVICE) -> RuntimeBundle:
    selected_ensemble = discover_best_loso_ensemble()
    mean, std = load_or_compute_normalization_stats(selected_ensemble)
    models: list[torch.nn.Module] = []
    for checkpoint_info in selected_ensemble.checkpoints:
        ckpt = torch.load(str(checkpoint_info.checkpoint_path), map_location=device)
        model = ActivityConformer(
            n_channels=checkpoint_info.n_channels,
            n_times=checkpoint_info.n_times,
            n_classes=checkpoint_info.n_classes,
            emb_size=checkpoint_info.emb_size,
            depth=checkpoint_info.depth,
            num_heads=checkpoint_info.num_heads,
            dropout=checkpoint_info.dropout,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    return RuntimeBundle(
        selected_ensemble=selected_ensemble,
        models=models,
        device=device,
        mean=mean,
        std=std,
    )


def get_runtime_bundle() -> RuntimeBundle:
    global _runtime_bundle
    if _runtime_bundle is None:
        _runtime_bundle = build_runtime_bundle(device=resolve_device(DEFAULT_DEVICE))
    return _runtime_bundle


def extract_windows_from_xlsx_for_selected_model(
    xlsx_path: Path,
    selected_ensemble: SelectedEnsemble,
) -> np.ndarray:
    loaded = load_xlsx_file(xlsx_path)
    if loaded is None:
        raise ValueError(f"Could not load xlsx data from {xlsx_path}")

    data, times = loaded
    windows = extract_windows(
        data=data,
        times=times,
        window_seconds=selected_ensemble.window_seconds,
        stride_seconds=selected_ensemble.stride_seconds,
    )
    if not windows:
        raise ValueError(f"No valid windows extracted from {xlsx_path}")

    arr = np.stack(windows, axis=0).transpose(0, 2, 1).astype(np.float32)
    if arr.shape[1] != selected_ensemble.n_channels or arr.shape[2] != selected_ensemble.n_times:
        raise ValueError(
            "Uploaded EEG file does not match the selected model input shape: "
            f"expected (C,T)=({selected_ensemble.n_channels},{selected_ensemble.n_times}), "
            f"got ({arr.shape[1]},{arr.shape[2]})"
        )
    return arr


def predict_uploaded_xlsx(xlsx_path: Path) -> dict:
    bundle = get_runtime_bundle()
    windows_nct = extract_windows_from_xlsx_for_selected_model(xlsx_path, bundle.selected_ensemble)
    X = np.expand_dims((windows_nct - bundle.mean) / bundle.std, axis=1)

    with torch.no_grad():
        tensor = torch.from_numpy(X).float().to(torch.device(bundle.device))
        ensemble_probabilities = None
        for model in bundle.models:
            _, logits = model(tensor)
            probs = F.softmax(logits, dim=-1)
            ensemble_probabilities = probs if ensemble_probabilities is None else (ensemble_probabilities + probs)
        probabilities = (ensemble_probabilities / len(bundle.models)).cpu().numpy()

    average_probabilities = probabilities.mean(axis=0)
    predicted_label = int(np.argmax(average_probabilities))
    reverse_label_map = {value: key for key, value in bundle.selected_ensemble.label_map.items()}
    predicted_activity = reverse_label_map.get(predicted_label, f"activity_{predicted_label + 1}")

    return {
        "predicted_label": predicted_label,
        "predicted_activity": predicted_activity,
        "average_probabilities": average_probabilities.tolist(),
        "probabilities_by_activity": {
            reverse_label_map.get(index, f"activity_{index + 1}"): float(probability)
            for index, probability in enumerate(average_probabilities.tolist())
        },
        "n_windows": int(len(windows_nct)),
        "model": {
            "mode": "loso_ensemble",
            "window_dir": str(bundle.selected_ensemble.window_dir),
            "summary_path": str(bundle.selected_ensemble.summary_path),
            "summary_macro_f1": bundle.selected_ensemble.summary_macro_f1,
            "mean_best_test_acc": bundle.selected_ensemble.mean_best_test_acc,
            "std_best_test_acc": bundle.selected_ensemble.std_best_test_acc,
            "window_seconds": bundle.selected_ensemble.window_seconds,
            "stride_seconds": bundle.selected_ensemble.stride_seconds,
            "ensemble_size": len(bundle.models),
            "device": bundle.device,
        },
    }


app = FastAPI(title="EEG Three-Class Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Missing frontend file: {INDEX_HTML_PATH}")
    return FileResponse(INDEX_HTML_PATH)


@app.get("/health")
def health() -> dict:
    bundle = get_runtime_bundle()
    return {
        "ok": True,
        "device": bundle.device,
        "window_dir": str(bundle.selected_ensemble.window_dir),
        "ensemble_size": len(bundle.models),
    }


@app.get("/model-info")
def model_info() -> dict:
    bundle = get_runtime_bundle()
    return {
        "model_name": "best_loso_window_ensemble",
        "summary_macro_f1": bundle.selected_ensemble.summary_macro_f1,
        "mean_best_test_acc": bundle.selected_ensemble.mean_best_test_acc,
        "std_best_test_acc": bundle.selected_ensemble.std_best_test_acc,
        "window_dir": str(bundle.selected_ensemble.window_dir),
        "summary_path": str(bundle.selected_ensemble.summary_path),
        "window_seconds": bundle.selected_ensemble.window_seconds,
        "stride_seconds": bundle.selected_ensemble.stride_seconds,
        "n_channels": bundle.selected_ensemble.n_channels,
        "n_times": bundle.selected_ensemble.n_times,
        "n_classes": bundle.selected_ensemble.n_classes,
        "label_map": bundle.selected_ensemble.label_map,
        "ensemble_size": len(bundle.models),
        "fold_checkpoints": [str(item.checkpoint_path) for item in bundle.selected_ensemble.checkpoints],
        "mean": bundle.mean,
        "std": bundle.std,
        "device": bundle.device,
    }


@app.post("/predict/xlsx")
async def predict_xlsx(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "input.xlsx").suffix.lower()
    if suffix != ".xlsx":
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported")

    tmp_dir = Path(tempfile.mkdtemp(prefix="eeg-predict-"))
    tmp_path = tmp_dir / (file.filename or "upload.xlsx")
    try:
        with tmp_path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        return predict_uploaded_xlsx(tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_activity_api:app", host="0.0.0.0", port=8000, reload=False)
