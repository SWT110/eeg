"""Subject-group validation, fixed-duration refit, and one outer LOSO evaluation.

This entrypoint deliberately does not use train_loso_fold. Its held-out data
never enter training, checkpoint selection, or inner normalization. Existing
record-normalized X.npy and historical architecture selection remain limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
from train_activity_loso import (
    ActivityConformer, DualBranchActivityConformer, build_comparison_model,
    compute_model_batch_loss, forward_model_batch_with_branches,
    tensor_dataset_from_inputs, transform_windows_to_fft,
)

PRESETS = (
    "time", "time_fft", "mdtf", "no_cross_depth", "uniform_fusion",
    "no_aux", "repeated_depth", "singledepth11", "eegnet",
    "shallowconvnet", "atcnet", "tcformer",
)
EXTERNAL_MODELS = {"eegnet", "shallowconvnet", "atcnet", "tcformer"}
CONTROLLED_MODELS = {"mdtf", "no_cross_depth", "uniform_fusion", "no_aux", "repeated_depth"}
LIMITATIONS = [
    "X.npy is already normalized per complete video record; train-only scalar "
    "normalization does not remove held-out-record distribution adaptation.",
    "Window length, model family, depths, dropout and class weights include "
    "choices developed using these same 11 subjects. This rerun separates current "
    "checkpoint/candidate selection; it does not erase historical model selection "
    "or constitute independent confirmatory validation.",
    "Fixed existing external implementations use their own architectural defaults; "
    "the shared candidate list provides equal optimizer/class-weight search counts, "
    "not architecture-specific optimal tuning.",
]


def write_json(path: Path, payload: object) -> None:
    """Atomically replace only files inside this run's newly created directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Fail instead of silently accepting a nondeterministic training operation.
    torch.use_deterministic_algorithms(True)


def make_subject_splits(subject_ids: np.ndarray, test_subject_id: int, inner_folds: int,
                        split_seed: int) -> dict:
    subjects = np.unique(subject_ids).astype(int)
    if test_subject_id not in subjects:
        raise ValueError(f"Unknown outer subject: {test_subject_id}")
    outer_train = subjects[subjects != test_subject_id]
    if not 2 <= inner_folds <= len(outer_train):
        raise ValueError("inner_folds must be between 2 and the number of outer training subjects")
    shuffled = np.random.default_rng(split_seed).permutation(outer_train)
    groups = []
    for index, validation_ids in enumerate(np.array_split(shuffled, inner_folds)):
        groups.append({
            "inner_fold": index,
            "train_subject_ids": sorted(set(outer_train.tolist()) - set(validation_ids.tolist())),
            "validation_subject_ids": sorted(validation_ids.astype(int).tolist()),
        })
    return {"test_subject_id": int(test_subject_id), "outer_train_subject_ids": outer_train.tolist(),
            "inner_folds": groups, "split_seed": split_seed}


def fit_scalar_normalizer(train: np.ndarray) -> dict:
    mean = float(np.mean(train, dtype=np.float64))
    std = float(np.std(train, dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Training inputs need finite nonzero variance")
    return {"mean": mean, "std": std, "scope": "training elements only", "ddof": 0}


def apply_scalar_normalizer(values: np.ndarray, stats: dict) -> np.ndarray:
    return ((values - stats["mean"]) / stats["std"]).astype(np.float32, copy=False)


def input_domains(raw: np.ndarray, dual: bool) -> tuple[np.ndarray, ...]:
    if not np.isfinite(raw).all():
        raise ValueError("Inputs contain NaN or infinity")
    time = np.asarray(raw, dtype=np.float32)[:, None, :, :]
    return (time, transform_windows_to_fft(time)) if dual else (time,)


def fit_inputs(raw_train: np.ndarray, dual: bool) -> tuple[object, dict]:
    domains = input_domains(raw_train, dual)
    stats = {name: fit_scalar_normalizer(values) for name, values in zip(("time", "fft"), domains)}
    normalized = tuple(apply_scalar_normalizer(values, stats[name])
                       for name, values in zip(("time", "fft"), domains))
    return (normalized if dual else normalized[0]), stats


def transform_inputs(raw: np.ndarray, dual: bool, stats: dict) -> object:
    domains = input_domains(raw, dual)
    normalized = tuple(apply_scalar_normalizer(values, stats[name])
                       for name, values in zip(("time", "fft"), domains))
    return normalized if dual else normalized[0]


def freeze_uniform_fusion(model: nn.Module) -> None:
    if not getattr(model, "uses_branch_loss_fusion", False):
        raise ValueError("Uniform fusion requires the branch-loss fusion model")
    with torch.no_grad():
        model.branch_loss_weight_logits.zero_()
    model.branch_loss_weight_logits.requires_grad_(False)


def build_model(preset: str, shape: tuple[int, int], n_classes: int, candidate: dict) -> nn.Module:
    channels, times = shape
    if preset in EXTERNAL_MODELS:
        return build_comparison_model(preset, channels, times, n_classes)
    common = dict(n_channels=channels, n_classes=n_classes, emb_size=40, num_heads=5,
                  dropout=0.5, transformer_encoder_dropout=0.5, depth=6)
    if preset == "time":
        return ActivityConformer(n_times=times, **common)
    if preset == "singledepth11":
        common.update(depth=11, transformer_encoder_dropout=0.85)
    if preset in CONTROLLED_MODELS:
        common.update(transformer_encoder_dropout=0.85, transformer_branch_qkv_dropout=0.25,
                      transformer_depths=[10, 10, 10] if preset == "repeated_depth" else [11, 10, 8],
                      transformer_branch_fusion="loss_softmax",
                      transformer_branch_qkv="none" if preset == "no_cross_depth" else "cross_depth",
                      branch_loss_aux_weight=0.0 if preset == "no_aux" else 0.2)
    model = DualBranchActivityConformer(time_n_times=times, fft_n_times=times // 2 + 1, **common)
    if preset == "uniform_fusion":
        freeze_uniform_fusion(model)
    return model


def is_dual(preset: str) -> bool:
    return preset != "time" and preset not in EXTERNAL_MODELS


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = cm.diagonal()
    recall = np.divide(tp, support, out=np.zeros(n_classes, float), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes, float), where=predicted > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(n_classes, float), where=(support + predicted) > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "confusion_matrix": cm.tolist(),
            "per_class": [{"class_id": i, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i in range(n_classes)]}


def make_loader(inputs: object, labels: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(tensor_dataset_from_inputs(inputs, labels), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed))


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss, n_samples = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, labels, branch_logits = forward_model_batch_with_branches(model, batch, device)
        loss, _ = compute_model_batch_loss(model, logits, labels, criterion, branch_logits)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        n_samples += len(labels)
    return total_loss / n_samples


@torch.no_grad()
def predict_once(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true, pred = [], []
    for batch in loader:
        logits, labels, _ = forward_model_batch_with_branches(model, batch, device)
        true.append(labels.cpu().numpy())
        pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(true), np.concatenate(pred)


def select_inner_candidate(rows: list[dict], candidate_order: list[str], expected_subject_ids: list[int],
                           max_epochs: int) -> dict:
    """Only inner metrics enter selection; no outer metric is accepted by this API.

    Score each candidate/epoch by equally weighting all validation subjects,
    even when validation groups have unequal sizes. Exact ties prefer the
    earliest epoch, then the candidate appearing first in the frozen config.
    """
    expected = sorted(expected_subject_ids)
    evaluated = []
    for order, candidate_id in enumerate(candidate_order):
        for epoch in range(1, max_epochs + 1):
            matched = [row for row in rows if row["candidate_id"] == candidate_id and row["epoch"] == epoch]
            if sorted(row["validation_subject_id"] for row in matched) != expected:
                raise ValueError("Each candidate/epoch requires exactly one score per inner validation subject")
            score = float(np.mean([row["macro_f1"] for row in matched]))
            if not np.isfinite(score):
                raise ValueError("Non-finite inner validation score")
            evaluated.append({"candidate_id": candidate_id, "epoch": epoch,
                              "mean_subject_macro_f1": score, "candidate_order": order})
    winner = min(evaluated, key=lambda row: (-row["mean_subject_macro_f1"], row["epoch"], row["candidate_order"]))
    return {**winner, "primary_metric": "unweighted mean of per-subject macro-F1",
            "tie_rule": "exact score tie: earliest epoch, then configuration candidate order",
            "candidate_epoch_scores": evaluated}


def training_components(model: nn.Module, candidate: dict, config: dict, device: torch.device):
    weights = candidate.get("class_weights", config["training"]["class_weights"])
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else torch.tensor(weights, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=candidate.get("learning_rate", config["training"]["learning_rate"]),
                                 betas=tuple(config["training"]["adam_betas"]))
    return criterion, optimizer


def run_outer_fold(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, split: dict, preset: str,
                   seed: int, config: dict, output_dir: Path, device: torch.device,
                   model_factory: Callable = build_model) -> dict:
    """Fresh inner models, then a fresh all-training-subject model; outer read last."""
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "split.json", split)
    candidates = config["hyperparameter_candidates"]
    epochs, batch_size = config["training"]["max_epochs"], config["training"]["batch_size"]
    n_classes, dual = config["n_classes"], is_dual(preset)
    rows, normalizers = [], []
    for inner in split["inner_folds"]:
        train_index = np.flatnonzero(np.isin(subject_ids, inner["train_subject_ids"]))
        validation_index = np.flatnonzero(np.isin(subject_ids, inner["validation_subject_ids"]))
        train_inputs, stats = fit_inputs(X[train_index], dual)
        validation_inputs = transform_inputs(X[validation_index], dual, stats)
        normalizers.append({"inner_fold": inner["inner_fold"], "statistics": stats,
                            "fit_subject_ids": inner["train_subject_ids"]})
        validation_loader = make_loader(validation_inputs, y[validation_index], batch_size, False, seed)
        for candidate in candidates:
            # Equal initialization/randomness across candidates for a given fold.
            training_seed = seed + 1009 * (inner["inner_fold"] + 1)
            set_seed(training_seed)
            model = model_factory(preset, tuple(X.shape[1:]), n_classes, candidate).to(device)
            train_loader = make_loader(train_inputs, y[train_index], batch_size, True, training_seed)
            criterion, optimizer = training_components(model, candidate, config, device)
            for epoch in range(1, epochs + 1):
                loss = train_epoch(model, train_loader, optimizer, criterion, device)
                true, predicted = predict_once(model, validation_loader, device)
                for subject_id in inner["validation_subject_ids"]:
                    mask = subject_ids[validation_index] == subject_id
                    metrics = metric_summary(true[mask], predicted[mask], n_classes)
                    rows.append({"candidate_id": candidate["id"], "epoch": epoch,
                                 "inner_fold": inner["inner_fold"], "training_seed": training_seed,
                                 "validation_subject_id": subject_id, "train_loss": loss, **metrics})
                print(f"{preset} outer={split['test_subject_id']} seed={seed} inner={inner['inner_fold']} "
                      f"candidate={candidate['id']} epoch={epoch}/{epochs}", flush=True)
            del model, optimizer, criterion, train_loader
            # Preserve completed inner fits if a later fit is interrupted.
            write_json(output_dir / "inner_scores.json", rows)
        del train_inputs, validation_inputs, validation_loader
    selection = select_inner_candidate(rows, [c["id"] for c in candidates], split["outer_train_subject_ids"], epochs)
    selected = next(c for c in candidates if c["id"] == selection["candidate_id"])
    write_json(output_dir / "selection.json", selection)
    train_index = np.flatnonzero(np.isin(subject_ids, split["outer_train_subject_ids"]))
    train_inputs, stats = fit_inputs(X[train_index], dual)
    normalizers.append({"stage": "outer_refit", "statistics": stats,
                        "fit_subject_ids": split["outer_train_subject_ids"]})
    write_json(output_dir / "normalizers.json", normalizers)
    # Deliberately use a fresh model/optimizer rather than any inner checkpoint.
    refit_seed = seed + 1000003
    set_seed(refit_seed)
    model = model_factory(preset, tuple(X.shape[1:]), n_classes, selected).to(device)
    train_loader = make_loader(train_inputs, y[train_index], batch_size, True, refit_seed)
    criterion, optimizer = training_components(model, selected, config, device)
    history = []
    for epoch in range(1, selection["epoch"] + 1):
        history.append({"epoch": epoch, "train_loss": train_epoch(model, train_loader, optimizer, criterion, device)})
        print(f"{preset} outer={split['test_subject_id']} seed={seed} fixed refit {epoch}/{selection['epoch']}", flush=True)
    write_json(output_dir / "refit_train_history.json", history)
    checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "preset": preset, "shape": list(X.shape[1:]), "n_classes": n_classes,
                  "candidate": selected, "selected_epochs": selection["epoch"], "refit_seed": refit_seed,
                  "normalizers": stats, "uniform_fusion_frozen": preset == "uniform_fusion"}
    torch.save(checkpoint, output_dir / "refit_checkpoint.pt")
    del train_inputs, train_loader, optimizer, criterion
    # First and only outer-subject evaluation, after selection and refit are fixed.
    test_index = np.flatnonzero(subject_ids == split["test_subject_id"])
    test_inputs = transform_inputs(X[test_index], dual, stats)
    test_loader = make_loader(test_inputs, y[test_index], batch_size, False, seed)
    true, predicted = predict_once(model, test_loader, device)
    result = {"preset": preset, "test_subject_id": split["test_subject_id"], "seed": seed,
              "refit_seed": refit_seed, "selected_candidate_id": selected["id"],
              "selected_epochs": selection["epoch"], "outer_evaluation_count": 1,
              "n_train_samples": len(train_index), "n_test_samples": len(test_index),
              "n_parameters": sum(p.numel() for p in model.parameters()),
              "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              **metric_summary(true, predicted, n_classes)}
    write_json(output_dir / "outer_metrics.json", result)
    np.savez_compressed(output_dir / "outer_predictions.npz", y_true=true, y_pred=predicted,
                        sample_indices=test_index, subject_ids=subject_ids[test_index])
    return result


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1 or config.get("primary_metric") != "mean_subject_macro_f1":
        raise ValueError("Expected schema_version=1 and primary_metric=mean_subject_macro_f1")
    if config.get("n_classes") != 3:
        raise ValueError("This first-round protocol expects the three fixed video conditions")
    training = config["training"]
    if int(training["max_epochs"]) < 1 or int(training["batch_size"]) < 2:
        raise ValueError("max_epochs >= 1 and batch_size >= 2 required")
    if not np.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("Positive finite learning rate required")
    if len(training["adam_betas"]) != 2 or any(not 0 <= b < 1 for b in training["adam_betas"]):
        raise ValueError("Adam requires two betas in [0,1)")
    candidates = config["hyperparameter_candidates"]
    if not candidates or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be nonempty and unique")
    for candidate in candidates:
        if set(candidate) - {"id", "learning_rate", "class_weights", "provenance"}:
            raise ValueError("Candidates support id, learning_rate, class_weights and provenance; architectures are locked by preset")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate["id"]):
            raise ValueError("Unsafe candidate ID")
        lr = candidate.get("learning_rate", training["learning_rate"])
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Positive finite candidate learning rate required")
        weights = candidate.get("class_weights", training["class_weights"])
        if weights is not None and (len(weights) != config["n_classes"] or any(not np.isfinite(w) or w <= 0 for w in weights)):
            raise ValueError("Class weights must be null or one positive finite number per class")
    return config


def build_plan(config: dict, dataset_root: Path, models: list[str], subjects: list[int], seeds: list[int],
               subject_ids: np.ndarray, shape: tuple[int, ...], device: str, dry_run: bool) -> dict:
    splits = [make_subject_splits(subject_ids, subject, config["inner_folds"], config["split_seed"]) for subject in subjects]
    n_outer = len(models) * len(subjects) * len(seeds)
    n_inner = n_outer * config["inner_folds"] * len(config["hyperparameter_candidates"])
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run,
            "status": "planned; no training performed" if dry_run else "training_pending",
            "resolved_config": config, "models": models, "seeds": seeds, "device": device,
            "dataset_root": str(dataset_root), "X_shape": list(shape), "splits": splits,
            "budget": {"outer_refits": n_outer, "inner_fits": n_inner,
                       "maximum_total_epochs": (n_outer + n_inner) * config["training"]["max_epochs"]},
            "selection": {"metric": "mean_subject_macro_f1", "class_zero_division": 0,
                          "averaging": "all three class F1s per subject, then equal subject weighting",
                          "tie_rule": "exact score tie: earliest epoch, then configuration candidate order",
                          "refit": "fresh initialization on all outer training subjects for selected epoch count",
                          "inner_seed": "base seed + 1009 * (inner_fold_index + 1)",
                          "refit_seed": "base seed + 1000003", "outer_evaluations_per_refit": 1},
            "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "cuda": torch.version.cuda,
                        "torch_num_threads": torch.get_num_threads(),
                        "deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "limitations": LIMITATIONS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist (also for --dry-run)")
    parser.add_argument("--config", type=Path, default=MODULE_DIR / "configs" / "subject_validation_first_round.json")
    parser.add_argument("--models", nargs="+", choices=PRESETS)
    parser.add_argument("--test-subject-ids", nargs="+", type=int, help="Omit to schedule all subjects")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Hash data/code and emit split/budget plan; construct no models")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    models, seeds = args.models or config["models"], args.seeds or config["seeds"]
    if not models or set(models) - set(PRESETS) or len(set(models)) != len(models):
        raise ValueError("Choose distinct supported model presets")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 - 1000004 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers below 2**32 - 1000004")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    dataset_root, output_dir = args.dataset_root.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    X = np.load(dataset_root / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(dataset_root / "y.npy", allow_pickle=False)
    subject_ids = np.load(dataset_root / "subject_ids.npy", allow_pickle=False)
    if X.ndim != 3 or y.ndim != 1 or subject_ids.ndim != 1 or len(X) != len(y) or len(y) != len(subject_ids):
        raise ValueError("Expected aligned X[N,C,T], y[N], subject_ids[N]")
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(subject_ids.dtype, np.integer):
        raise ValueError("Labels and subject IDs must be integer arrays")
    if not np.array_equal(np.unique(y), np.arange(config["n_classes"])):
        raise ValueError("Expected labels 0,1,2")
    for subject in np.unique(subject_ids):
        if not np.array_equal(np.unique(y[subject_ids == subject]), np.arange(config["n_classes"])):
            raise ValueError(f"Subject {subject} lacks a class; this protocol requires three classes per subject")
    subjects = args.test_subject_ids or np.unique(subject_ids).astype(int).tolist()
    if len(set(subjects)) != len(subjects):
        raise ValueError("Duplicate outer subject IDs")
    plan = build_plan(config, dataset_root, models, subjects, seeds, subject_ids, X.shape, args.device, args.dry_run)
    plan["input_sha256"] = {name: sha256_file(dataset_root / name)
                            for name in ("X.npy", "y.npy", "subject_ids.npy", "metadata.json")}
    plan["source_sha256"] = {name: sha256_file(MODULE_DIR / name) for name in
                             (Path(__file__).name, "train_activity_loso.py", "comparison_models.py")}
    plan["config_path"] = str(args.config.resolve())
    plan["config_sha256"] = sha256_file(args.config)
    if not args.dry_run and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --dry-run for planning or --device cpu for an explicitly small run")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "protocol_plan.json", plan)
    print(json.dumps({"output_dir": str(output_dir), "dry_run": args.dry_run, "budget": plan["budget"]}, indent=2))
    if args.dry_run:
        return
    results = []
    try:
        for preset in models:
            for seed in seeds:
                for split in plan["splits"]:
                    fold_dir = output_dir / preset / f"seed_{seed}" / f"subject_{split['test_subject_id']:02d}"
                    results.append(run_outer_fold(X, y, subject_ids, split, preset, seed, config,
                                                  fold_dir, torch.device(args.device)))
                    write_json(output_dir / "completed_outer_folds.json", results)
        plan["status"] = "complete"
    except BaseException as error:
        plan["status"] = "interrupted_or_failed"
        plan["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output_dir / "protocol_plan.json", plan)


if __name__ == "__main__":
    main()
