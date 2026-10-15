"""Run the new channel-preserving model and four baselines on one cohort.

Default: 15 subjects (S6/S17 excluded), 5 models, 75 LOSO folds, seed 42,
200 epochs, batch 72, unweighted CE and repository-local Adam defaults.
Use --models to assign complete models to separate GPUs; no fold merging needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from channel_preserving_model import ARCHITECTURE
from comparison_models import VALID_COMPARISON_MODELS, build_comparison_model
from evaluate_activity_failurecase import load_best_checkpoint, restore_model as restore_external
from experiment_manifest import collect_environment_info
from run_activity_five_models import sha256_file
from summarize_loso_results import summarize, write_summary
import train_activity_channelpreserve as single
from train_activity_loso import atomic_json_dump, train_loso_fold, validate_device

MODELS = (ARCHITECTURE, *VALID_COMPARISON_MODELS)
PROTOCOL = "channelpreserving_five_models_test_guided_loso_v1"


def code_hashes() -> dict[str, str]:
    return {**single.code_hashes(), Path(__file__).name: sha256_file(Path(__file__))}


def build_plan(args: argparse.Namespace) -> dict:
    models = [s.strip().lower() for s in args.models.split(",")]
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError(f"--models must be distinct members of {MODELS}")
    # Retain all existing cohort, source-record, array and default-profile checks.
    base = single.build_plan(argparse.Namespace(**{k: v for k, v in vars(args).items() if k != "models"}))
    job = ("smoke_job_" if args.smoke_test else "job_") + "_".join(models)
    output = Path(base["output_dir"]).parent / job
    configs, counts, kwargs = {}, {}, {}
    for name in models:
        if name == ARCHITECTURE:
            configs[name] = base["architecture_config"]
            counts[name] = base["trainable_parameters"]
            kwargs[name] = dict(single.MODEL_KWARGS)
        else:
            model = build_comparison_model(name, n_channels=21, n_times=1920, n_classes=3)
            configs[name] = dict(model.architecture_config)
            counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            kwargs[name] = {"architecture": name, "input_domain": "time",
                            "class_weights": None, "train_sampling": "original"}
            del model
    base.pop("model_output_dir")
    return {
        **base, "protocol": PROTOCOL, "models": models,
        "per_model_kwargs": kwargs, "model_architecture_configs": configs,
        "model_trainable_parameters": counts,
        "planned_training_folds": len(models) * len(base["test_subject_ids"]),
        "output_dir": str(output), "code_sha256": code_hashes(),
    }


def model_run_dir(root: Path, name: str, seed: int) -> Path:
    directory = root / name if name == ARCHITECTURE else root / "external" / name
    return directory / f"seed_{seed}"


def completed_record(plan: dict) -> dict:
    return {"status": "completed", "models": plan["models"],
            "smoke_test": plan["smoke_test"], "training_folds": plan["planned_training_folds"]}


def restore_suite_model(checkpoint: dict, metrics: dict, name: str, plan: dict) -> torch.nn.Module:
    config = plan["model_architecture_configs"][name]
    if metrics.get("architecture_config") != config or checkpoint.get("architecture_config") != config:
        raise ValueError(f"Model architecture_config mismatch: {name}")
    if name == ARCHITECTURE:
        return single.restore_checkpoint(checkpoint, metrics, plan)
    return restore_external(checkpoint, metrics, name)


def validate_completed_fold(fold: Path, plan: dict, name: str, subject: int) -> dict:
    if name == ARCHITECTURE:
        metrics = single.validate_completed_fold(fold, plan, subject)
    else:
        for artifact in ("metrics.json", "best_model.pt", "test_predictions.npz", "epoch_history.json"):
            if not (fold / artifact).is_file():
                raise ValueError(f"Missing completed fold artifact: {fold / artifact}")
        metrics = json.loads((fold / "metrics.json").read_text())
        expected = {
            "architecture": name, "architecture_config": plan["model_architecture_configs"][name],
            "test_subject_id": subject, "seed": plan["seed"], "epochs": plan["epochs"],
            "batch_size": plan["batch_size"], "lr": plan["lr"], "class_weights": None,
            "optimizer": "Adam", "optimizer_betas": plan["adam_betas"],
            "input_domain": "time", "classification_mode": "flat", "train_sampling": "original",
            "loss": "CrossEntropyLoss", "n_channels": 21, "n_times": 1920, "n_classes": 3,
            "n_train_samples": plan["dataset_shape"][0] - plan["subject_window_counts"][str(subject)],
            "n_test_samples": plan["subject_window_counts"][str(subject)],
        }
        for key, value in expected.items():
            if key not in metrics or metrics[key] != value:
                raise ValueError(f"Fold metadata mismatch: {name}, S{subject}, {key}")
        epoch = metrics.get("best_epoch")
        if not isinstance(epoch, int) or not 1 <= epoch <= plan["epochs"]:
            raise ValueError(f"Invalid best_epoch: {fold}")
        history = json.loads((fold / "epoch_history.json").read_text())
        if [r.get("epoch") for r in history.get("history", [])] != list(range(1, plan["epochs"] + 1)):
            raise ValueError(f"Incomplete epoch history: {fold}")
        for key in ("architecture", "architecture_config", "epochs", "test_subject_id", "batch_size", "lr"):
            if history.get(key) != metrics[key]:
                raise ValueError(f"History/metrics mismatch: {fold}, {key}")
        with np.load(fold / "test_predictions.npz", allow_pickle=False) as saved:
            true, pred = saved["y_true"], saved["y_pred"]
        if (true.shape != (metrics["n_test_samples"],) or pred.shape != true.shape
                or not np.issubdtype(true.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer)
                or not np.isin(true, [0, 1, 2]).all() or not np.isin(pred, [0, 1, 2]).all()):
            raise ValueError(f"Invalid saved predictions: {fold}")
        data = Path(plan["dataset_root"])
        ids = np.load(data / "subject_ids.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(data / "y.npy", mmap_mode="r", allow_pickle=False)
        if not np.array_equal(true, labels[ids == subject]):
            raise ValueError(f"Saved test labels disagree with formal data: {fold}")
        matrix = np.bincount(true.astype(np.int64) * 3 + pred.astype(np.int64), minlength=9).reshape(3, 3)
        accuracy = metrics.get("best_test_acc", np.nan)
        if (not np.isfinite(accuracy) or not np.isclose(np.mean(true == pred), accuracy, rtol=0, atol=1e-12)
                or matrix.tolist() != metrics.get("confusion_matrix")):
            raise ValueError(f"Saved predictions disagree with metrics: {fold}")
        restore_suite_model(load_best_checkpoint(fold / "best_model.pt"), metrics, name, plan)
    if metrics.get("trainable_parameters") != plan["model_trainable_parameters"][name]:
        raise ValueError(f"Trainable parameter count mismatch: {fold}")
    return metrics


def run_plan(plan: dict, *, resume: bool = False) -> None:
    device = validate_device(plan["device"])
    torch.set_num_threads(plan["cpu_threads"])
    root = Path(plan["output_dir"])
    plan_path = root / "run_plan.json"
    if root.exists():
        if not resume:
            raise FileExistsError(f"Output exists: {root}; use --resume for this exact job")
        if not plan_path.is_file() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Resume configuration/data/code hash mismatch; use a new output base")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(plan, plan_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    environment = f"resume_environment_{stamp}.json" if (root / "environment.json").exists() else "environment.json"
    atomic_json_dump(collect_environment_info(SCRIPT_ROOT.parent), root / environment)
    summaries = {}
    for name in plan["models"]:
        model_dir = model_run_dir(root, name, plan["seed"])
        subject = None
        try:
            for subject in plan["test_subject_ids"]:
                fold = model_dir / f"fold_subject_{subject}"
                if resume and (fold / "metrics.json").exists():
                    validate_completed_fold(fold, plan, name, subject)
                    print(f"[SKIP] {name} S{subject}", flush=True)
                    continue
                atomic_json_dump({"status": "running", "model": name, "subject_id": subject}, root / "status.json")
                print(f"[RUN] {name} S{subject}; S6/S17 excluded from every fold", flush=True)
                train_loso_fold(dataset_root=plan["dataset_root"], test_subject_id=subject,
                    epochs=plan["epochs"], batch_size=plan["batch_size"], lr=plan["lr"],
                    device=device, output_dir=model_dir, seed=plan["seed"], resume=True,
                    **plan["per_model_kwargs"][name])
                validate_completed_fold(fold, plan, name, subject)
            summary = summarize(model_dir)
            if (summary["n_folds"] != len(plan["test_subject_ids"])
                    or sorted(f["subject_id"] for f in summary["per_fold"]) != plan["test_subject_ids"]):
                raise ValueError(f"Completed fold set disagrees with the plan: {name}")
            write_summary(summary, model_dir)
            summaries[name] = summary
            atomic_json_dump({"protocol": PROTOCOL, "included_subject_ids": plan["included_subject_ids"],
                              "excluded_subject_ids": [6, 17], "models": summaries}, root / "five_model_summary.json")
        except Exception as exc:
            atomic_json_dump({"status": "failed", "model": name, "subject_id": subject,
                              "error": str(exc)}, root / "status.json")
            raise
    atomic_json_dump(completed_record(plan), root / "completed.json")
    atomic_json_dump({"status": "completed"}, root / "status.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path,
                        default=SCRIPT_ROOT.parent / "local_artifacts/outputs/activity_channelpreserve_five_models_seed42")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--cohort-subject-ids", default=",".join(map(str, range(1, 18))))
    parser.add_argument("--exclude-subject-ids", default="6,17")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Separate job: first fold per model, 1 epoch, batch 2")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(test_subject_ids=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        run_plan(plan, resume=args.resume)


if __name__ == "__main__":
    main()
