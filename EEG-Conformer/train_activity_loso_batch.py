"""
EEG-Conformer – Batch LOSO Activity Three-Class Training
=========================================================
Runs all LOSO folds (one held-out subject per fold) for the global
activity dataset. Keeps the same interactive/batch-oriented UX used across
this project:

* Auto-discovers subject_ids from ``subject_ids.npy`` in the dataset root
* ``--subject-ids`` to restrict which folds to run
* ``--skip-existing`` to skip folds whose output dir already contains
  ``metrics.json`` or a legacy complete ``best_model.pt``
* ``--resume`` to continue incomplete folds from ``last_checkpoint.pt`` with
  model, Adam optimizer, history, best-metric, and RNG state restored
* Automatic rerun inside the project conda env when CUDA is requested but
  unavailable in the current interpreter

Usage
-----
    python train_activity_loso_batch.py \\
        --subject-ids 1,2,3 \\
        --epochs 200 \\
        --skip-existing \\
        --resume
"""

from __future__ import annotations

import argparse
import errno
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
    DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
    DEFAULT_CONV_TYPE,
    DEFAULT_CUMULATIVE_QUERY_ATTENTION,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DEPTH,
    DEFAULT_DEVICE,
    DEFAULT_DROPOUT,
    DEFAULT_EMB_SIZE,
    DEFAULT_ENV_NAME,
    DEFAULT_EPOCHS,
    DEFAULT_FFT_GLOBAL,
    DEFAULT_INPUT_DOMAIN,
    DEFAULT_INPUT_QKV,
    DEFAULT_INPUT_QKV_DIM,
    DEFAULT_INPUT_QKV_DROPOUT,
    DEFAULT_INPUT_QKV_HEADS,
    DEFAULT_INPUT_QKV_RES_SCALE,
    DEFAULT_LR,
    DEFAULT_NUM_HEADS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSFORMER_BRANCHES,
    DEFAULT_TRANSFORMER_BRANCH_FUSION,
    DEFAULT_TRANSFORMER_BRANCH_QKV,
    DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
    DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    RESUME_CHECKPOINT_FILENAME,
    TRANSFORMER_FUSION_SINGLE,
    TRANSFORMER_FUSION_SOFTMAX,
    cuda_is_usable,
    normalize_device_name,
    parse_class_weights,
    project_env_prefix,
    running_inside_project_env,
    validate_classification_mode,
    resolve_branch_loss_aux_weight,
    resolve_transformer_branch_depths,
    resolve_transformer_branch_fusion,
    resolve_transformer_branch_qkv,
    train_loso_fold,
    validate_conv_type,
    validate_fft_global_for_input_domain,
    validate_input_domain,
    validate_input_qkv,
    validate_device,
    validate_dropout_probability,
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
    transformer_branch_fusion: str = DEFAULT_TRANSFORMER_BRANCH_FUSION,
    branch_loss_aux_weight: float = DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
    transformer_branch_qkv: str = DEFAULT_TRANSFORMER_BRANCH_QKV,
    transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    transformer_branch_qkv_dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
    classification_mode: str = "flat",
) -> bool:
    """Return True if the fold directory already contains a matching result artifact."""
    fold_dir = fold_output_dir(output_dir, subject_id)
    metrics_path = fold_dir / "metrics.json"
    expected_input_domain = validate_input_domain(input_domain)
    expected_conv_type = validate_conv_type(conv_type)
    expected_fft_global = validate_fft_global_for_input_domain(expected_input_domain, fft_global)
    expected_input_qkv = validate_input_qkv(input_qkv)
    expected_input_qkv_dim = int(input_qkv_dim)
    expected_input_qkv_heads = int(input_qkv_heads)
    expected_input_qkv_dropout = float(input_qkv_dropout)
    expected_input_qkv_res_scale = float(input_qkv_res_scale)
    expected_cumulative_query_attention = bool(cumulative_query_attention)
    expected_transformer_encoder_dropout = validate_dropout_probability(
        transformer_encoder_dropout,
        "transformer_encoder_dropout",
    )
    expected_transformer_branch_qkv_dropout = validate_dropout_probability(
        transformer_branch_qkv_dropout,
        "transformer_branch_qkv_dropout",
    )
    expected_depth = int(depth)
    expected_transformer_depths = resolve_transformer_branch_depths(
        depth=expected_depth,
        transformer_branches=transformer_branches,
        transformer_depths=transformer_depths,
        input_domain=expected_input_domain,
    )
    expected_transformer_branches = len(expected_transformer_depths)
    expected_transformer_fusion = resolve_transformer_branch_fusion(
        transformer_branch_fusion,
        transformer_branches=expected_transformer_branches,
        input_domain=expected_input_domain,
    )
    expected_branch_loss_aux_weight = resolve_branch_loss_aux_weight(
        branch_loss_aux_weight,
        expected_transformer_fusion,
    )
    expected_transformer_branch_qkv = resolve_transformer_branch_qkv(
        transformer_branch_qkv,
        transformer_branch_fusion=expected_transformer_fusion,
        transformer_branches=expected_transformer_branches,
        input_domain=expected_input_domain,
    )
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        actual_input_domain = validate_input_domain(metrics.get("input_domain"))
        actual_conv_type = validate_conv_type(metrics.get("conv_type"))
        actual_fft_global = validate_fft_global_for_input_domain(
            actual_input_domain,
            metrics.get("fft_global"),
        )
        actual_input_qkv = validate_input_qkv(metrics.get("input_qkv"))
        actual_input_qkv_dim = int(metrics.get("input_qkv_dim", DEFAULT_INPUT_QKV_DIM))
        actual_input_qkv_heads = int(metrics.get("input_qkv_heads", DEFAULT_INPUT_QKV_HEADS))
        actual_input_qkv_dropout = float(metrics.get("input_qkv_dropout", DEFAULT_INPUT_QKV_DROPOUT))
        actual_input_qkv_res_scale = float(metrics.get("input_qkv_res_scale", DEFAULT_INPUT_QKV_RES_SCALE))
        actual_cumulative_query_attention = bool(
            metrics.get("cumulative_query_attention", DEFAULT_CUMULATIVE_QUERY_ATTENTION)
        )
        actual_depth = int(metrics.get("depth", DEFAULT_DEPTH))
        actual_transformer_encoder_dropout = float(
            metrics.get(
                "transformer_encoder_dropout",
                DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
            )
        )
        actual_transformer_branches = int(
            metrics.get("transformer_branches", DEFAULT_TRANSFORMER_BRANCHES)
        )
        actual_transformer_depths = tuple(
            int(value)
            for value in metrics.get("transformer_branch_depths", [actual_depth])
        )
        actual_transformer_fusion = resolve_transformer_branch_fusion(
            metrics.get(
                "transformer_branch_fusion",
                TRANSFORMER_FUSION_SOFTMAX
                if actual_transformer_branches > 1
                else TRANSFORMER_FUSION_SINGLE,
            ),
            transformer_branches=actual_transformer_branches,
            input_domain=actual_input_domain,
        )
        actual_branch_loss_aux_weight = resolve_branch_loss_aux_weight(
            metrics.get("branch_loss_aux_weight", DEFAULT_BRANCH_LOSS_AUX_WEIGHT),
            actual_transformer_fusion,
        )
        actual_transformer_branch_qkv = resolve_transformer_branch_qkv(
            metrics.get("transformer_branch_qkv", DEFAULT_TRANSFORMER_BRANCH_QKV),
            transformer_branch_fusion=actual_transformer_fusion,
            transformer_branches=actual_transformer_branches,
            input_domain=actual_input_domain,
        )
        actual_transformer_branch_qkv_dropout = float(
            metrics.get(
                "transformer_branch_qkv_dropout",
                DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
            )
        )
        branch_qkv_dropout_matches = True
        if (
            expected_transformer_branch_qkv != DEFAULT_TRANSFORMER_BRANCH_QKV
            or actual_transformer_branch_qkv != DEFAULT_TRANSFORMER_BRANCH_QKV
        ):
            branch_qkv_dropout_matches = (
                actual_transformer_branch_qkv_dropout
                == expected_transformer_branch_qkv_dropout
            )
        qkv_params_match = True
        if expected_input_qkv != DEFAULT_INPUT_QKV or actual_input_qkv != DEFAULT_INPUT_QKV:
            qkv_params_match = (
                actual_input_qkv_dim == expected_input_qkv_dim
                and actual_input_qkv_heads == expected_input_qkv_heads
                and actual_input_qkv_dropout == expected_input_qkv_dropout
                and actual_input_qkv_res_scale == expected_input_qkv_res_scale
            )
        return (
            metrics.get("classification_mode", "flat") == classification_mode
            and actual_input_domain == expected_input_domain
            and actual_conv_type == expected_conv_type
            and actual_fft_global == expected_fft_global
            and actual_input_qkv == expected_input_qkv
            and actual_cumulative_query_attention == expected_cumulative_query_attention
            and actual_depth == expected_depth
            and actual_transformer_encoder_dropout == expected_transformer_encoder_dropout
            and actual_transformer_branches == expected_transformer_branches
            and actual_transformer_depths == expected_transformer_depths
            and actual_transformer_fusion == expected_transformer_fusion
            and actual_branch_loss_aux_weight == expected_branch_loss_aux_weight
            and actual_transformer_branch_qkv == expected_transformer_branch_qkv
            and branch_qkv_dropout_matches
            and qkv_params_match
        )

    # Legacy fallback: a bare best_model.pt has no metadata, so only treat it as
    # complete for the historical default experiment.
    return (
        classification_mode == "flat"
        and (fold_dir / "best_model.pt").exists()
        and expected_input_domain == DEFAULT_INPUT_DOMAIN
        and expected_conv_type == DEFAULT_CONV_TYPE
        and expected_fft_global == DEFAULT_FFT_GLOBAL
        and expected_input_qkv == DEFAULT_INPUT_QKV
        and expected_cumulative_query_attention == DEFAULT_CUMULATIVE_QUERY_ATTENTION
        and expected_depth == DEFAULT_DEPTH
        and expected_transformer_encoder_dropout == DEFAULT_TRANSFORMER_ENCODER_DROPOUT
        and expected_transformer_branches == DEFAULT_TRANSFORMER_BRANCHES
        and expected_transformer_depths == (DEFAULT_DEPTH,)
        and expected_transformer_fusion == TRANSFORMER_FUSION_SINGLE
        and expected_branch_loss_aux_weight == DEFAULT_BRANCH_LOSS_AUX_WEIGHT
        and expected_transformer_branch_qkv == DEFAULT_TRANSFORMER_BRANCH_QKV
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def is_storage_exhaustion_error(exc: BaseException) -> bool:
    """Recognize disk-full/quota errors through wrapped exception chains."""
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, OSError) and current.errno in {
            errno.ENOSPC,
            getattr(errno, "EDQUOT", -1),
        }:
            return True
        message = str(current).lower()
        if "no space left on device" in message or "disk quota exceeded" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


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
    transformer_branch_fusion: str = DEFAULT_TRANSFORMER_BRANCH_FUSION,
    branch_loss_aux_weight: float = DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
    transformer_branch_qkv: str = DEFAULT_TRANSFORMER_BRANCH_QKV,
    resume: bool = False,
    transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    transformer_branch_qkv_dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
    classification_mode: str = "flat",
) -> list[LosoFoldResult]:
    classification_mode = validate_classification_mode(classification_mode)
    if classification_mode == "hierarchical" and input_domain != "time_fft":
        raise ValueError("hierarchical classification requires --input-domain time_fft")
    resolved_device = validate_device(device)
    resolved_input_domain = validate_input_domain(input_domain)
    resolved_conv_type = validate_conv_type(conv_type)
    resolved_fft_global = validate_fft_global_for_input_domain(resolved_input_domain, fft_global)
    resolved_input_qkv = validate_input_qkv(input_qkv)
    resolved_input_qkv_dim = int(input_qkv_dim)
    resolved_input_qkv_heads = int(input_qkv_heads)
    resolved_input_qkv_dropout = float(input_qkv_dropout)
    resolved_input_qkv_res_scale = float(input_qkv_res_scale)
    resolved_cumulative_query_attention = bool(cumulative_query_attention)
    resolved_transformer_encoder_dropout = validate_dropout_probability(
        transformer_encoder_dropout,
        "transformer_encoder_dropout",
    )
    resolved_transformer_branch_qkv_dropout = validate_dropout_probability(
        transformer_branch_qkv_dropout,
        "transformer_branch_qkv_dropout",
    )
    resolved_depth = int(depth)
    resolved_transformer_depths = resolve_transformer_branch_depths(
        depth=resolved_depth,
        transformer_branches=transformer_branches,
        transformer_depths=transformer_depths,
        input_domain=resolved_input_domain,
    )
    resolved_transformer_branches = len(resolved_transformer_depths)
    resolved_transformer_fusion = resolve_transformer_branch_fusion(
        transformer_branch_fusion,
        transformer_branches=resolved_transformer_branches,
        input_domain=resolved_input_domain,
    )
    resolved_branch_loss_aux_weight = resolve_branch_loss_aux_weight(
        branch_loss_aux_weight,
        resolved_transformer_fusion,
    )
    resolved_transformer_branch_qkv = resolve_transformer_branch_qkv(
        transformer_branch_qkv,
        transformer_branch_fusion=resolved_transformer_fusion,
        transformer_branches=resolved_transformer_branches,
        input_domain=resolved_input_domain,
    )
    results: list[LosoFoldResult] = []

    print(f"Planned LOSO folds: {len(subject_ids)}")
    for subject_id in subject_ids:
        fold_dir = fold_output_dir(output_dir, subject_id)
        fold_complete = skip_existing and fold_is_complete(
            output_dir=output_dir,
            subject_id=subject_id,
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
            transformer_branch_fusion=resolved_transformer_fusion,
            classification_mode=classification_mode,
            branch_loss_aux_weight=resolved_branch_loss_aux_weight,
            transformer_branch_qkv=resolved_transformer_branch_qkv,
            transformer_encoder_dropout=resolved_transformer_encoder_dropout,
            transformer_branch_qkv_dropout=resolved_transformer_branch_qkv_dropout,
        )
        # A legacy bare best_model.pt may represent only an interrupted fold.
        # In resume mode, metrics.json is the only completion marker.
        if resume and not (fold_dir / "metrics.json").exists():
            fold_complete = False
        if fold_complete:
            if resume:
                for stale_name in (
                    RESUME_CHECKPOINT_FILENAME,
                    f".{RESUME_CHECKPOINT_FILENAME}.tmp",
                ):
                    try:
                        (fold_dir / stale_name).unlink()
                    except FileNotFoundError:
                        pass
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
                conv_type=resolved_conv_type,
                fft_global=resolved_fft_global,
                input_qkv=resolved_input_qkv,
                input_qkv_dim=resolved_input_qkv_dim,
                input_qkv_heads=resolved_input_qkv_heads,
                input_qkv_dropout=resolved_input_qkv_dropout,
                input_qkv_res_scale=resolved_input_qkv_res_scale,
                cumulative_query_attention=resolved_cumulative_query_attention,
                depth=resolved_depth,
                transformer_encoder_dropout=resolved_transformer_encoder_dropout,
                transformer_branches=resolved_transformer_branches,
                transformer_depths=resolved_transformer_depths,
                class_weights=class_weights,
                transformer_branch_fusion=resolved_transformer_fusion,
                classification_mode=classification_mode,
                branch_loss_aux_weight=resolved_branch_loss_aux_weight,
                transformer_branch_qkv=resolved_transformer_branch_qkv,
                transformer_branch_qkv_dropout=resolved_transformer_branch_qkv_dropout,
                resume=bool(resume),
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
            if is_storage_exhaustion_error(exc):
                raise RuntimeError(
                    f"Storage exhausted while training subject {subject_id}; "
                    "aborting remaining folds so resumable checkpoints are not put at risk"
                ) from exc
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
    parser.add_argument("--classification-mode", choices=["flat", "hierarchical"], default="flat",
                        help="flat: legacy 3-way; hierarchical: e1/e2 vs e3, then e1 vs e2 (time_fft)")
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
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="TransformerEncoder block count (default: 6)")
    parser.add_argument(
        "--transformer-encoder-dropout",
        type=float,
        default=DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
        help=(
            "Shared dropout for attention weights, attention output, FFN internal, "
            "and FFN output in every TransformerEncoderBlock (default: 0.5)"
        ),
    )
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
    parser.add_argument(
        "--transformer-branch-fusion",
        type=str,
        default=DEFAULT_TRANSFORMER_BRANCH_FUSION,
        help=(
            "Parallel-depth fusion: feature_softmax (legacy/default) or "
            "loss_softmax (independent classifier/loss per depth)"
        ),
    )
    parser.add_argument(
        "--branch-loss-aux-weight",
        type=float,
        default=DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
        help="Mean branch-loss coefficient used only with loss_softmax (default: 0.0)",
    )
    parser.add_argument(
        "--transformer-branch-qkv",
        type=str,
        default=DEFAULT_TRANSFORMER_BRANCH_QKV,
        help=(
            "QKV communication between parallel depth outputs: none (default) "
            "or cross_depth"
        ),
    )
    parser.add_argument(
        "--transformer-branch-qkv-dropout",
        type=float,
        default=DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
        help=(
            "Shared attention/output dropout inside CrossDepthQKVResidual "
            "(default: 0.1)"
        ),
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for per-fold outputs (fold_subject_<id>/)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a fold when a matching completed metrics.json exists",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume incomplete folds from last_checkpoint.pt; use with "
            "--skip-existing to skip completed folds"
        ),
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
    conv_type = validate_conv_type(args.conv_type)
    fft_global = validate_fft_global_for_input_domain(input_domain, args.fft_global)
    input_qkv = validate_input_qkv(args.input_qkv)
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
        conv_type=conv_type,
        fft_global=fft_global,
        input_qkv=input_qkv,
        input_qkv_dim=args.input_qkv_dim,
        input_qkv_heads=args.input_qkv_heads,
        input_qkv_dropout=args.input_qkv_dropout,
        input_qkv_res_scale=args.input_qkv_res_scale,
        cumulative_query_attention=args.cumulative_query_attention,
        depth=args.depth,
        transformer_encoder_dropout=args.transformer_encoder_dropout,
        transformer_branches=args.transformer_branches,
        transformer_depths=args.transformer_depths,
        class_weights=class_weights,
        transformer_branch_fusion=args.transformer_branch_fusion,
        classification_mode=args.classification_mode,
        branch_loss_aux_weight=args.branch_loss_aux_weight,
        transformer_branch_qkv=args.transformer_branch_qkv,
        transformer_branch_qkv_dropout=args.transformer_branch_qkv_dropout,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
