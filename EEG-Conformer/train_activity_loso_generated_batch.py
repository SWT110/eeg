"""
EEG-Conformer – Batch LOSO training for config-generated activity datasets
==========================================================================

Reads window/stride combinations from ``window_stride_configs.json``, maps
them to generated dataset directories, trains each dataset with the existing
``train_activity_loso_batch.py`` logic, and writes per-dataset summaries plus
experiment manifests via ``summarize_loso_results.py`` and
``experiment_manifest.py``.

Directory convention
--------------------
Datasets:
    local_artifacts/data_to_list/global_activity_dataset/<window_...>

Training outputs:
    local_artifacts/outputs/activity_loso/<window_...>
    local_artifacts/outputs/activity_loso_<domain>_dwconv/<window_...> when --conv-type dwconv and --output-base is omitted
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import (
    ACTIVITY_LOSO_OUTPUT_DIR,
    GLOBAL_ACTIVITY_DATASET_DIR,
    WINDOW_STRIDE_CONFIG,
)
from experiment_manifest import utc_now_iso, write_loso_experiment_manifest

DEFAULT_CONFIG = WINDOW_STRIDE_CONFIG
DEFAULT_DATASET_BASE = GLOBAL_ACTIVITY_DATASET_DIR
DEFAULT_OUTPUT_BASE = ACTIVITY_LOSO_OUTPUT_DIR
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 72
DEFAULT_LR = 0.0002
DEFAULT_DEPTH = 6
DEFAULT_DEVICE = "cuda:0"
DEFAULT_INPUT_DOMAIN = "time"
DEFAULT_CONV_TYPE = "standard"
DWCONV_CONV_TYPE = "dwconv"
DEFAULT_FFT_GLOBAL = "none"
FFT_GLOBAL_MLP = "mlp"
DEFAULT_INPUT_QKV = "none"
INPUT_QKV_CHANNEL = "channel"
INPUT_QKV_TIME = "time"
DEFAULT_INPUT_QKV_DIM = 64
DEFAULT_INPUT_QKV_HEADS = 4
DEFAULT_INPUT_QKV_DROPOUT = 0.1
DEFAULT_INPUT_QKV_RES_SCALE = 0.1
DEFAULT_CUMULATIVE_QUERY_ATTENTION = False
DEFAULT_TRANSFORMER_BRANCHES = 1
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


def validate_input_domain(raw: str | None) -> str:
    train_batch = _load_train_batch_module()
    validator = getattr(train_batch, "validate_input_domain", None)
    if validator is not None:
        return validator(raw)

    value = DEFAULT_INPUT_DOMAIN if raw is None else str(raw).strip().lower()
    if value not in {DEFAULT_INPUT_DOMAIN, "fft", "time_fft"}:
        raise ValueError("input_domain must be one of: time, fft, time_fft")
    return value


def validate_conv_type(raw: str | None) -> str:
    train_batch = _load_train_batch_module()
    validator = getattr(train_batch, "validate_conv_type", None)
    if validator is not None:
        return validator(raw)

    value = DEFAULT_CONV_TYPE if raw is None else str(raw).strip().lower().replace("-", "_")
    if value in {"standard", "conv", "normal"}:
        return DEFAULT_CONV_TYPE
    if value in {"dw", "dwconv", "depthwise", "depthwise_conv"}:
        return DWCONV_CONV_TYPE
    raise ValueError("conv_type must be one of: standard, dwconv")


def validate_fft_global(raw: str | None) -> str:
    train_batch = _load_train_batch_module()
    validator = getattr(train_batch, "validate_fft_global_for_input_domain", None)
    if validator is not None:
        # Validate the raw value with an FFT-capable domain so the delegated
        # helper does not reject mlp merely because the default domain is time.
        return validator("fft", raw)

    value = DEFAULT_FFT_GLOBAL if raw is None else str(raw).strip().lower().replace("-", "_")
    if value in {"none", "off", "false", "no"}:
        return DEFAULT_FFT_GLOBAL
    if value in {"mlp", "frequency_mlp", "freq_mlp"}:
        return FFT_GLOBAL_MLP
    raise ValueError("fft_global must be one of: none, mlp")


def validate_fft_global_for_input_domain(input_domain: str, fft_global: str | None) -> str:
    train_batch = _load_train_batch_module()
    validator = getattr(train_batch, "validate_fft_global_for_input_domain", None)
    if validator is not None:
        return validator(input_domain, fft_global)

    resolved_input_domain = validate_input_domain(input_domain)
    resolved_fft_global = validate_fft_global(fft_global)
    if resolved_input_domain == DEFAULT_INPUT_DOMAIN and resolved_fft_global != DEFAULT_FFT_GLOBAL:
        raise ValueError("--fft-global applies only to fft or time_fft input domains")
    return resolved_fft_global


def validate_input_qkv(raw: str | None) -> str:
    train_batch = _load_train_batch_module()
    validator = getattr(train_batch, "validate_input_qkv", None)
    if validator is not None:
        return validator(raw)

    value = DEFAULT_INPUT_QKV if raw is None else str(raw).strip().lower().replace("-", "_")
    if value in {"none", "off", "false", "no"}:
        return DEFAULT_INPUT_QKV
    if value in {"channel", "channels", "channel_qkv", "qkv"}:
        return INPUT_QKV_CHANNEL
    if value in {"time", "temporal", "time_token", "time_tokens", "time_qkv", "temporal_qkv", "qkv_time"}:
        return INPUT_QKV_TIME
    raise ValueError("input_qkv must be one of: none, channel, time")


def resolve_transformer_branch_depths(
    depth: int = DEFAULT_DEPTH,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
    input_domain: str | None = None,
) -> tuple[int, ...]:
    train_batch = _load_train_batch_module()
    resolver = getattr(train_batch, "resolve_transformer_branch_depths", None)
    if resolver is not None:
        return resolver(
            depth=depth,
            transformer_branches=transformer_branches,
            transformer_depths=transformer_depths,
            input_domain=input_domain,
        )

    resolved_depth = int(depth)
    resolved_branches = int(transformer_branches)
    if resolved_depth < 1:
        raise ValueError("depth must be >= 1")
    if resolved_branches < 1:
        raise ValueError("transformer_branches must be >= 1")
    if transformer_depths is None:
        if resolved_branches != DEFAULT_TRANSFORMER_BRANCHES:
            raise ValueError(
                "--transformer-depths is required when --transformer-branches is greater than 1"
            )
        resolved_depths = (resolved_depth,)
    else:
        resolved_depths = tuple(int(value) for value in transformer_depths)
        if len(resolved_depths) != resolved_branches:
            raise ValueError(
                "--transformer-branches must equal the number of values in --transformer-depths"
            )
        if any(value < 1 for value in resolved_depths):
            raise ValueError("all --transformer-depths values must be >= 1")
        if resolved_branches == DEFAULT_TRANSFORMER_BRANCHES and resolved_depths != (resolved_depth,):
            raise ValueError("use --depth for a single Transformer encoder")
    if len(resolved_depths) > 1 and resolved_depth != DEFAULT_DEPTH:
        raise ValueError("--depth cannot be combined with parallel --transformer-depths")
    if input_domain is not None and len(resolved_depths) > 1:
        if validate_input_domain(input_domain) != "time_fft":
            raise ValueError(
                "parallel Transformer depth branches require --input-domain time_fft"
            )
    return resolved_depths


def output_base_arg_was_provided(argv: list[str]) -> bool:
    return any(arg == "--output-base" or arg.startswith("--output-base=") for arg in argv)


def resolve_output_base(
    output_base: str | Path,
    input_domain: str,
    conv_type: str,
    fft_global: str = DEFAULT_FFT_GLOBAL,
    input_qkv: str = DEFAULT_INPUT_QKV,
    cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
    depth: int = DEFAULT_DEPTH,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
    output_base_explicit: bool = False,
) -> Path:
    base = Path(output_base).expanduser()
    resolved_domain = validate_input_domain(input_domain)
    resolved_conv_type = validate_conv_type(conv_type)
    resolved_fft_global = validate_fft_global_for_input_domain(resolved_domain, fft_global)
    resolved_input_qkv = validate_input_qkv(input_qkv)
    resolved_transformer_depths = resolve_transformer_branch_depths(
        depth=depth,
        transformer_branches=transformer_branches,
        transformer_depths=transformer_depths,
        input_domain=resolved_domain,
    )
    if (
        output_base_explicit
        or (
            resolved_conv_type == DEFAULT_CONV_TYPE
            and resolved_fft_global == DEFAULT_FFT_GLOBAL
            and resolved_input_qkv == DEFAULT_INPUT_QKV
            and not cumulative_query_attention
            and int(depth) == DEFAULT_DEPTH
            and resolved_transformer_depths == (DEFAULT_DEPTH,)
        )
    ):
        return base

    domain_slug = {
        "time": "time",
        "fft": "fft",
        "time_fft": "time_fft",
    }[resolved_domain]
    suffix_parts: list[str] = []
    if resolved_conv_type != DEFAULT_CONV_TYPE:
        suffix_parts.append(resolved_conv_type)
    if resolved_fft_global != DEFAULT_FFT_GLOBAL:
        suffix_parts.append(resolved_fft_global)
    if resolved_input_qkv != DEFAULT_INPUT_QKV:
        suffix_parts.extend(["qkv", resolved_input_qkv])
    if cumulative_query_attention:
        suffix_parts.append("cumulative_q")
    if len(resolved_transformer_depths) > 1:
        suffix_parts.extend(
            ["transformer_depths", *(str(value) for value in resolved_transformer_depths)]
        )
    elif int(depth) != DEFAULT_DEPTH:
        suffix_parts.extend(["depth", str(int(depth))])
    suffix = "_".join(suffix_parts)
    return base.parent / f"activity_loso_{domain_slug}_{suffix}"


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


def fold_is_complete_for_input_domain(
    train_batch: ModuleType | object,
    output_dir: str | Path,
    subject_id: int,
    input_domain: str,
    conv_type: str = DEFAULT_CONV_TYPE,
    fft_global: str = DEFAULT_FFT_GLOBAL,
    input_qkv: str = DEFAULT_INPUT_QKV,
    input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM,
    input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS,
    input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
    input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
    cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
    depth: int = DEFAULT_DEPTH,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
) -> bool:
    checker = getattr(train_batch, "fold_is_complete")
    try:
        return checker(
            output_dir,
            subject_id,
            input_domain,
            conv_type,
            fft_global,
            input_qkv,
            input_qkv_dim,
            input_qkv_heads,
            input_qkv_dropout,
            input_qkv_res_scale,
            cumulative_query_attention,
            depth,
            transformer_branches,
            transformer_depths,
        )
    except TypeError:
        try:
            return checker(output_dir, subject_id, input_domain, conv_type, fft_global, input_qkv)
        except TypeError:
            try:
                return checker(output_dir, subject_id, input_domain, conv_type, fft_global)
            except TypeError:
                try:
                    return checker(output_dir, subject_id, input_domain, conv_type)
                except TypeError:
                    try:
                        return checker(output_dir, subject_id, input_domain)
                    except TypeError:
                        return checker(output_dir, subject_id)


def dataset_output_is_complete(
    dataset_root: str | Path,
    output_dir: str | Path,
    input_domain: str = DEFAULT_INPUT_DOMAIN,
    conv_type: str = DEFAULT_CONV_TYPE,
    fft_global: str = DEFAULT_FFT_GLOBAL,
    input_qkv: str = DEFAULT_INPUT_QKV,
    input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM,
    input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS,
    input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
    input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
    cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
    depth: int = DEFAULT_DEPTH,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
) -> bool:
    missing = missing_dataset_files(dataset_root)
    if missing:
        return False

    root = Path(output_dir)
    if not (root / "summary.json").exists() or not (root / "summary.csv").exists():
        return False

    train_batch = _load_train_batch_module()
    subject_ids = train_batch.discover_subject_ids_from_global_dataset(dataset_root)
    return all(
        fold_is_complete_for_input_domain(
            train_batch,
            root,
            subject_id,
            input_domain,
            conv_type,
            fft_global,
            input_qkv,
            input_qkv_dim,
            input_qkv_heads,
            input_qkv_dropout,
            input_qkv_res_scale,
            cumulative_query_attention,
            depth,
            transformer_branches,
            transformer_depths,
        )
        for subject_id in subject_ids
    )


def summarize_output_dir(output_dir: str | Path) -> Path:
    summary_module = _load_summary_module()
    summary = summary_module.summarize(output_dir)
    json_path, _ = summary_module.write_summary(summary, output_dir)
    return json_path


def build_manifest_training_config(
    train_batch: ModuleType | object,
    epochs: int,
    lr: float,
    seed: int,
    input_domain: str,
    conv_type: str,
    fft_global: str,
    input_qkv: str,
    input_qkv_dim: int,
    input_qkv_heads: int,
    input_qkv_dropout: float,
    input_qkv_res_scale: float,
    cumulative_query_attention: bool,
    depth: int,
    transformer_branches: int,
    transformer_depths: list[int] | tuple[int, ...],
    class_weights: list[float] | None,
) -> dict:
    return {
        "input_domain": input_domain,
        "conv_type": conv_type,
        "fft_global": fft_global,
        "input_qkv": input_qkv,
        "input_qkv_dim": input_qkv_dim,
        "input_qkv_heads": input_qkv_heads,
        "input_qkv_dropout": input_qkv_dropout,
        "input_qkv_res_scale": input_qkv_res_scale,
        "cumulative_query_attention": cumulative_query_attention,
        "transformer_branches": transformer_branches,
        "transformer_branch_depths": list(transformer_depths),
        "transformer_branch_fusion": (
            "softmax_weighted_sum" if transformer_branches > 1 else "single"
        ),
        "transformer_weights_independent_by_domain": transformer_branches > 1,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "class_weights": class_weights,
        "optimizer": "Adam",
        "optimizer_betas": list(getattr(train_batch, "DEFAULT_BETAS", (0.5, 0.999))),
        "loss": "CrossEntropyLoss",
        "emb_size": getattr(train_batch, "DEFAULT_EMB_SIZE", 40),
        "depth": depth,
        "num_heads": getattr(train_batch, "DEFAULT_NUM_HEADS", 5),
        "dropout": getattr(train_batch, "DEFAULT_DROPOUT", 0.5),
    }


def write_dataset_manifest(
    job: DatasetJob,
    train_batch: ModuleType | object,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    skip_existing: bool,
    seed: int,
    input_domain: str,
    conv_type: str,
    fft_global: str,
    input_qkv: str,
    input_qkv_dim: int,
    input_qkv_heads: int,
    input_qkv_dropout: float,
    input_qkv_res_scale: float,
    cumulative_query_attention: bool,
    depth: int,
    transformer_branches: int,
    transformer_depths: list[int] | tuple[int, ...],
    class_weights: list[float] | None,
    config_path: str | Path | None,
    command_line: list[str] | None,
    run_status: str,
    run_started_at: str | None,
    run_ended_at: str | None,
    resume: bool = False,
    note: str | None = None,
) -> tuple[Path, Path]:
    return write_loso_experiment_manifest(
        dataset_root=job.dataset_root,
        output_dir=job.output_dir,
        dataset_name=job.dataset_name,
        input_domain=input_domain,
        training_config=build_manifest_training_config(
            train_batch=train_batch,
            epochs=epochs,
            lr=lr,
            seed=seed,
            input_domain=input_domain,
            conv_type=conv_type,
            fft_global=fft_global,
            input_qkv=input_qkv,
            input_qkv_dim=input_qkv_dim,
            input_qkv_heads=input_qkv_heads,
            input_qkv_dropout=input_qkv_dropout,
            input_qkv_res_scale=input_qkv_res_scale,
            cumulative_query_attention=cumulative_query_attention,
            depth=depth,
            transformer_branches=transformer_branches,
            transformer_depths=transformer_depths,
            class_weights=class_weights,
        ),
        runtime_config={
            "device": device,
            "batch_size": batch_size,
            "skip_existing": skip_existing,
            "resume": resume,
            "output_dir": str(job.output_dir),
            "output_base": str(job.output_dir.parent),
        },
        config_path=config_path,
        script_path=Path(__file__).resolve(),
        command_line=command_line,
        run_status=run_status,
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
        project_root=EEG_ROOT,
        note=note,
    )


def run_generated_dataset_batch(
    jobs: list[DatasetJob],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    skip_existing: bool,
    seed: int = 42,
    input_domain: str = DEFAULT_INPUT_DOMAIN,
    conv_type: str = DEFAULT_CONV_TYPE,
    fft_global: str = DEFAULT_FFT_GLOBAL,
    input_qkv: str = DEFAULT_INPUT_QKV,
    input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM,
    input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS,
    input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
    input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
    cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
    depth: int = DEFAULT_DEPTH,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
    class_weights: list[float] | None = None,
    resume: bool = False,
    config_path: str | Path | None = None,
    command_line: list[str] | None = None,
) -> list[DatasetRunResult]:
    train_batch = _load_train_batch_module()
    resolved_input_domain = validate_input_domain(input_domain)
    resolved_conv_type = validate_conv_type(conv_type)
    resolved_fft_global = validate_fft_global_for_input_domain(resolved_input_domain, fft_global)
    resolved_input_qkv = validate_input_qkv(input_qkv)
    resolved_input_qkv_dim = int(input_qkv_dim)
    resolved_input_qkv_heads = int(input_qkv_heads)
    resolved_input_qkv_dropout = float(input_qkv_dropout)
    resolved_input_qkv_res_scale = float(input_qkv_res_scale)
    resolved_cumulative_query_attention = bool(cumulative_query_attention)
    resolved_depth = int(depth)
    resolved_transformer_depths = resolve_transformer_branch_depths(
        depth=resolved_depth,
        transformer_branches=transformer_branches,
        transformer_depths=transformer_depths,
        input_domain=resolved_input_domain,
    )
    resolved_transformer_branches = len(resolved_transformer_depths)
    results: list[DatasetRunResult] = []

    print(f"Planned datasets: {len(jobs)}")
    for job in jobs:
        try:
            run_started_at = utc_now_iso()
            missing = missing_dataset_files(job.dataset_root)
            if missing:
                raise FileNotFoundError(
                    f"Dataset {job.dataset_name} is missing required files in {job.dataset_root}: {missing}"
                )

            if skip_existing and dataset_output_is_complete(
                job.dataset_root,
                job.output_dir,
                resolved_input_domain,
                resolved_conv_type,
                resolved_fft_global,
                resolved_input_qkv,
                resolved_input_qkv_dim,
                resolved_input_qkv_heads,
                resolved_input_qkv_dropout,
                resolved_input_qkv_res_scale,
                resolved_cumulative_query_attention,
                resolved_depth,
                resolved_transformer_branches,
                resolved_transformer_depths,
            ):
                summary_json = job.output_dir / "summary.json"
                manifest_json = job.output_dir / "experiment_manifest.json"
                if not manifest_json.exists():
                    written_manifest_json, written_manifest_md = write_dataset_manifest(
                        job=job,
                        train_batch=train_batch,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        device=device,
                        skip_existing=skip_existing,
                        seed=seed,
                        input_domain=resolved_input_domain,
                        conv_type=resolved_conv_type,
                        fft_global=resolved_fft_global,
                        input_qkv=resolved_input_qkv,
                        input_qkv_dim=resolved_input_qkv_dim,
                        input_qkv_heads=resolved_input_qkv_heads,
                        input_qkv_dropout=resolved_input_qkv_dropout,
                        input_qkv_res_scale=resolved_input_qkv_res_scale,
                        cumulative_query_attention=resolved_cumulative_query_attention,
                        depth=resolved_depth,
                        transformer_branches=resolved_transformer_branches,
                        transformer_depths=resolved_transformer_depths,
                        class_weights=class_weights,
                        config_path=config_path,
                        command_line=command_line,
                        run_status="skipped_existing",
                        run_started_at=run_started_at,
                        run_ended_at=utc_now_iso(),
                        resume=bool(resume),
                        note=(
                            "Output directory was complete before this invocation; "
                            "fold metrics are used where available, runtime fields reflect the current command."
                        ),
                    )
                    print(f"[MANIFEST] {written_manifest_json}  {written_manifest_md}")
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
                input_domain=resolved_input_domain,
                conv_type=resolved_conv_type,
                fft_global=resolved_fft_global,
                input_qkv=resolved_input_qkv,
                input_qkv_dim=resolved_input_qkv_dim,
                input_qkv_heads=resolved_input_qkv_heads,
                input_qkv_dropout=resolved_input_qkv_dropout,
                input_qkv_res_scale=resolved_input_qkv_res_scale,
                cumulative_query_attention=resolved_cumulative_query_attention,
                depth=resolved_depth,
                transformer_branches=resolved_transformer_branches,
                transformer_depths=resolved_transformer_depths,
                class_weights=class_weights,
                resume=bool(resume),
            )
            summary_json = summarize_output_dir(job.output_dir)
            manifest_json, manifest_md = write_dataset_manifest(
                job=job,
                train_batch=train_batch,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                skip_existing=skip_existing,
                seed=seed,
                input_domain=resolved_input_domain,
                conv_type=resolved_conv_type,
                fft_global=resolved_fft_global,
                input_qkv=resolved_input_qkv,
                input_qkv_dim=resolved_input_qkv_dim,
                input_qkv_heads=resolved_input_qkv_heads,
                input_qkv_dropout=resolved_input_qkv_dropout,
                input_qkv_res_scale=resolved_input_qkv_res_scale,
                cumulative_query_attention=resolved_cumulative_query_attention,
                depth=resolved_depth,
                transformer_branches=resolved_transformer_branches,
                transformer_depths=resolved_transformer_depths,
                class_weights=class_weights,
                config_path=config_path,
                command_line=command_line,
                run_status="trained",
                run_started_at=run_started_at,
                run_ended_at=utc_now_iso(),
                resume=bool(resume),
            )
            print(f"[DONE] dataset={job.dataset_name} summary={summary_json}")
            print(f"[MANIFEST] {manifest_json}  {manifest_md}")
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
            storage_checker = getattr(train_batch, "is_storage_exhaustion_error", None)
            if storage_checker is not None and storage_checker(exc):
                raise RuntimeError(
                    f"Storage exhausted while training dataset {job.dataset_name}; "
                    "aborting remaining datasets"
                ) from exc
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
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="TransformerEncoder block count (default: 6)")
    parser.add_argument(
        "--transformer-branches",
        type=int,
        default=DEFAULT_TRANSFORMER_BRANCHES,
        help="Parallel Transformer encoders per time/FFT branch (default: 1)",
    )
    parser.add_argument(
        "--transformer-depths",
        type=int,
        nargs="+",
        default=None,
        help="Block counts for parallel encoders, e.g. 11 10 8",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--input-domain",
        type=str,
        default=DEFAULT_INPUT_DOMAIN,
        help="Input representation: time, fft, or time_fft dual branch (default: time)",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        default=DEFAULT_CONV_TYPE,
        help="Convolution type in PatchEmbedding spatial conv: standard or dwconv (default: standard)",
    )
    parser.add_argument(
        "--fft-global",
        type=str,
        default=DEFAULT_FFT_GLOBAL,
        help="Optional global FFT preprocessor: none or mlp (default: none)",
    )
    parser.add_argument(
        "--input-qkv",
        type=str,
        default=DEFAULT_INPUT_QKV,
        help="Optional input QKV residual block: none, channel, or time (default: none)",
    )
    parser.add_argument(
        "--input-qkv-dim",
        type=int,
        default=DEFAULT_INPUT_QKV_DIM,
        help="Embedding width for --input-qkv channel/time (default: 64)",
    )
    parser.add_argument(
        "--input-qkv-heads",
        type=int,
        default=DEFAULT_INPUT_QKV_HEADS,
        help="Number of attention heads for --input-qkv channel/time (default: 4)",
    )
    parser.add_argument(
        "--input-qkv-dropout",
        type=float,
        default=DEFAULT_INPUT_QKV_DROPOUT,
        help="Dropout inside the input QKV residual block (default: 0.1)",
    )
    parser.add_argument(
        "--input-qkv-res-scale",
        type=float,
        default=DEFAULT_INPUT_QKV_RES_SCALE,
        help="Initial residual scale gamma for input QKV block (default: 0.1)",
    )
    parser.add_argument(
        "--cumulative-query-attention",
        action="store_true",
        default=DEFAULT_CUMULATIVE_QUERY_ATTENTION,
        help="Accumulate encoder queries across blocks (default: disabled)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Optional comma-separated class weights for CrossEntropyLoss, e.g. 3,3,1",
    )
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--skip-existing",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Skip datasets/folds that already have complete outputs (default: true)",
        )
    else:
        parser.add_argument(
            "--skip-existing",
            dest="skip_existing",
            action="store_true",
            default=True,
            help="Skip datasets/folds that already have complete outputs (default: true)",
        )
        parser.add_argument(
            "--no-skip-existing",
            dest="skip_existing",
            action="store_false",
            help="Do not skip datasets/folds that already have complete outputs",
        )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete folds from their last_checkpoint.pt files",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    runtime_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(runtime_argv)
    maybe_rerun_in_project_env(runtime_argv, str(args.device))

    config_path = Path(args.config).expanduser()
    dataset_base = Path(args.dataset_base).expanduser()
    train_batch = _load_train_batch_module()
    class_weights = train_batch.parse_class_weights(args.class_weights)
    input_domain = validate_input_domain(args.input_domain)
    conv_type = validate_conv_type(args.conv_type)
    fft_global = validate_fft_global_for_input_domain(input_domain, args.fft_global)
    input_qkv = validate_input_qkv(args.input_qkv)
    output_base = resolve_output_base(
        args.output_base,
        input_domain=input_domain,
        conv_type=conv_type,
        fft_global=fft_global,
        input_qkv=input_qkv,
        cumulative_query_attention=args.cumulative_query_attention,
        depth=args.depth,
        transformer_branches=args.transformer_branches,
        transformer_depths=args.transformer_depths,
        output_base_explicit=output_base_arg_was_provided(runtime_argv),
    )

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
    resolved_transformer_depths = resolve_transformer_branch_depths(
        depth=args.depth,
        transformer_branches=args.transformer_branches,
        transformer_depths=args.transformer_depths,
        input_domain=input_domain,
    )

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
        input_domain=input_domain,
        conv_type=conv_type,
        fft_global=fft_global,
        input_qkv=input_qkv,
        input_qkv_dim=args.input_qkv_dim,
        input_qkv_heads=args.input_qkv_heads,
        input_qkv_dropout=args.input_qkv_dropout,
        input_qkv_res_scale=args.input_qkv_res_scale,
        cumulative_query_attention=args.cumulative_query_attention,
        depth=args.depth,
        transformer_branches=len(resolved_transformer_depths),
        transformer_depths=resolved_transformer_depths,
        class_weights=class_weights,
        resume=bool(args.resume),
        config_path=config_path,
        command_line=[sys.executable, str(Path(__file__).resolve()), *runtime_argv],
    )


if __name__ == "__main__":
    main()
