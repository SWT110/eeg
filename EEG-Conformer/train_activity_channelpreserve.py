"""Train the approved 5+5 dual-skip model with repository baseline defaults.

The normal run is 200 epochs, batch 72, Adam 0.0002/(0.5,0.999), D=40,
5 attention heads, dropout .5, ordinary CE, no resampling and no S&R.
Depth=10 is the explicitly approved architecture exception to default depth=6.
S6 and S17 must be absent from every formal training/test array.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from channel_preserving_model import ARCHITECTURE, ChannelPreservingDualConformer, restore_from_config
from experiment_manifest import collect_environment_info
from run_activity_five_models import build_plan as validate_cohort_plan, sha256_file
from summarize_loso_results import summarize, write_summary
from train_activity_loso import (
    DEFAULT_BATCH_SIZE, DEFAULT_BETAS, DEFAULT_DROPOUT, DEFAULT_EMB_SIZE,
    DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_NUM_HEADS, atomic_json_dump,
    train_loso_fold, validate_device,
)

PROTOCOL = "channel_preserving_test_guided_loso_v1"
PROFILE = "repository_local_defaults_v1"
MODEL_KWARGS = {
    "architecture": ARCHITECTURE, "input_domain": "time_fft", "depth": 10,
    "emb_size": DEFAULT_EMB_SIZE, "num_heads": DEFAULT_NUM_HEADS,
    "dropout": DEFAULT_DROPOUT, "transformer_encoder_dropout": DEFAULT_DROPOUT,
    "class_weights": None, "train_sampling": "original",
}
CODE_FILES = (
    "train_activity_channelpreserve.py", "channel_preserving_model.py",
    "train_activity_loso.py", "comparison_models.py", "run_activity_five_models.py",
    "train_activity_loso_batch.py", "train_activity_comparison.py",
    "summarize_loso_results.py", "experiment_manifest.py", "evaluate_activity_failurecase.py",
)


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(SCRIPT_ROOT / name) for name in CODE_FILES}


def build_plan(args: argparse.Namespace) -> dict:
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    epochs = 1 if args.smoke_test else DEFAULT_EPOCHS
    batch_size = 2 if args.smoke_test else DEFAULT_BATCH_SIZE
    # Reuse the existing array/metadata/cohort checks, not its frozen MDTF model,
    # class weighting or training dispatch. Exclusion is checked in the arrays.
    cohort_args = argparse.Namespace(**vars(args), models="mdtf", epochs=epochs,
                                     batch_size=batch_size, lr=DEFAULT_LR)
    checked = validate_cohort_plan(cohort_args)
    if checked["excluded_subject_ids"] != [6, 17]:
        raise ValueError("This experiment requires S6 and S17 as excluded failure cases")
    # Also check record identity/source hashes and the complete 128 Hz metadata
    # now, so an invalid provenance file cannot waste a full training run.
    from evaluate_activity_failurecase import load_dataset
    formal = load_dataset(checked["dataset_root"])
    if formal.sha256 != checked["dataset_sha256"]:
        raise ValueError("Dataset changed during plan validation")
    tests = checked["test_subject_ids"][:1] if args.smoke_test else checked["test_subject_ids"]
    job = "smoke_channelpreserving_depth10_skip10" if args.smoke_test else "channelpreserving_depth10_skip10"
    if args.smoke_test or sorted(tests) != checked["included_subject_ids"]:
        job += "_test_" + "_".join(f"S{sid}" for sid in sorted(tests))
    output = Path(checked["output_dir"]).parent / job
    model = ChannelPreservingDualConformer(n_channels=21, time_n_times=1920, fft_n_times=961,
        n_classes=3, **{k:v for k,v in MODEL_KWARGS.items()
                       if k not in ("architecture","input_domain","class_weights","train_sampling")})
    keys = ("dataset_root", "dataset_shape", "included_subject_ids", "excluded_subject_ids",
            "seed", "dataset_sha256")
    ids = np.load(Path(checked["dataset_root"])/"subject_ids.npy",mmap_mode="r",allow_pickle=False)
    subjects, counts = np.unique(ids,return_counts=True)
    return {
        **{key: checked[key] for key in keys},
        "protocol": PROTOCOL, "hyperparameter_profile": PROFILE,
        "checkpoint_rule": "highest outer test accuracy; first epoch wins ties",
        "models": [ARCHITECTURE], "model_kwargs": dict(MODEL_KWARGS),
        "architecture_config": model.architecture_config,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_subject_ids": tests, "planned_training_folds": len(tests),
        "subject_window_counts": {str(int(s)):int(c) for s,c in zip(subjects,counts)},
        "epochs": epochs, "batch_size": batch_size, "lr": DEFAULT_LR,
        "adam_betas": list(DEFAULT_BETAS), "adam_eps": 1e-8, "weight_decay": 0.0,
        "class_weights": None, "train_sampling": "original", "sr_augmentation": False,
        "smoke_test": bool(args.smoke_test), "cpu_threads": args.cpu_threads,
        "device": args.device, "output_dir": str(output),
        "model_output_dir": str(output / "model"), "code_sha256": code_hashes(),
        "failurecase_used_for_selection": False,
    }


def validate_metrics(metrics: dict, plan: dict, subject: int) -> None:
    expected = {
        "architecture": ARCHITECTURE, "architecture_config": plan["architecture_config"],
        "test_subject_id": subject, "seed": plan["seed"], "epochs": plan["epochs"],
        "batch_size": plan["batch_size"], "lr": plan["lr"], "class_weights": None,
        "input_domain": "time_fft", "classification_mode": "flat", "train_sampling": "original",
        "optimizer": "Adam", "optimizer_betas": plan["adam_betas"],
        "emb_size": MODEL_KWARGS["emb_size"], "num_heads": MODEL_KWARGS["num_heads"],
        "depth": 10, "dropout": DEFAULT_DROPOUT, "transformer_encoder_dropout": DEFAULT_DROPOUT,
        "transformer_branches": 1, "branch_loss_aux_weight": 0.0,
        "transformer_branch_qkv": "none", "n_channels": 21, "n_times": 1920, "n_classes": 3,
        "n_train_samples": plan["dataset_shape"][0]-plan["subject_window_counts"][str(subject)],
        "n_test_samples": plan["subject_window_counts"][str(subject)],
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"New-model fold metadata mismatch: S{subject}, {key}")
    if not isinstance(metrics.get("best_epoch"), int) or not 1 <= metrics["best_epoch"] <= plan["epochs"]:
        raise ValueError("Invalid best_epoch in new-model fold")
    accuracy = metrics.get("best_test_acc")
    if not isinstance(accuracy,(int,float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Invalid best_test_acc in new-model fold")


def restore_checkpoint(checkpoint: dict, metrics: dict, plan: dict) -> torch.nn.Module:
    for key in ("architecture", "architecture_config", "test_subject_id", "depth", "emb_size",
                "num_heads", "dropout", "transformer_encoder_dropout", "n_channels", "n_times",
                "n_classes", "input_domain", "time_n_times", "fft_n_times"):
        if checkpoint.get(key) != metrics.get(key):
            raise ValueError(f"Checkpoint and metrics disagree: {key}")
    if (checkpoint.get("epoch") != metrics["best_epoch"]-1
            or checkpoint.get("architecture_config") != plan["architecture_config"]):
        raise ValueError("Checkpoint best epoch / architecture disagree with this run")
    model = restore_from_config(checkpoint["architecture_config"])
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    return model.eval()


def validate_completed_fold(fold: Path, plan: dict, subject: int) -> dict:
    """Never skip a truncated/corrupted fold solely because metrics.json exists."""
    from evaluate_activity_failurecase import load_best_checkpoint
    for name in ("metrics.json","best_model.pt","test_predictions.npz","epoch_history.json"):
        if not (fold/name).is_file():
            raise ValueError(f"Missing completed fold artifact: {fold/name}")
    metrics = json.loads((fold/"metrics.json").read_text())
    validate_metrics(metrics,plan,subject)
    history = json.loads((fold/"epoch_history.json").read_text())
    rows = history.get("history",[])
    if [r.get("epoch") for r in rows] != list(range(1,plan["epochs"]+1)):
        raise ValueError(f"Incomplete epoch history: {fold}")
    for key in ("architecture","architecture_config","epochs","test_subject_id"):
        if history.get(key) != metrics[key]:
            raise ValueError(f"History/metrics mismatch: {key}")
    with np.load(fold/"test_predictions.npz",allow_pickle=False) as saved:
        true,pred = saved["y_true"],saved["y_pred"]
    if (true.shape!=(metrics["n_test_samples"],) or pred.shape!=true.shape
            or not np.issubdtype(true.dtype,np.integer) or not np.issubdtype(pred.dtype,np.integer)
            or not np.isin(true,[0,1,2]).all() or not np.isin(pred,[0,1,2]).all()):
        raise ValueError(f"Invalid saved predictions: {fold}")
    labels = np.load(Path(plan["dataset_root"])/"y.npy",mmap_mode="r",allow_pickle=False)
    ids = np.load(Path(plan["dataset_root"])/"subject_ids.npy",mmap_mode="r",allow_pickle=False)
    if not np.array_equal(true,labels[ids==subject]):
        raise ValueError(f"Saved test labels disagree with the formal dataset: {fold}")
    matrix = np.bincount(true.astype(np.int64)*3+pred.astype(np.int64),minlength=9).reshape(3,3).tolist()
    if (not math.isclose(float(np.mean(true==pred)),metrics["best_test_acc"],abs_tol=1e-12,rel_tol=0)
            or matrix!=metrics.get("confusion_matrix")):
        raise ValueError(f"Saved predictions disagree with metrics: {fold}")
    restore_checkpoint(load_best_checkpoint(fold/"best_model.pt"),metrics,plan)
    return metrics


def run_plan(plan: dict, *, resume: bool = False) -> None:
    device = validate_device(plan["device"])
    torch.set_num_threads(plan["cpu_threads"])
    root = Path(plan["output_dir"])
    plan_path = root / "run_plan.json"
    if root.exists():
        if not resume:
            raise FileExistsError(f"Output exists: {root}; use --resume for this exact run")
        if not plan_path.is_file() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Resume configuration/data/code hash mismatch; use a new output base")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(plan, plan_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    environment = root / (f"resume_environment_{stamp}.json" if (root/"environment.json").exists() else "environment.json")
    atomic_json_dump(collect_environment_info(SCRIPT_ROOT.parent), environment)
    model_dir = Path(plan["model_output_dir"])
    for subject in plan["test_subject_ids"]:
        fold = model_dir / f"fold_subject_{subject}"
        metrics_path = fold / "metrics.json"
        if resume and metrics_path.exists():
            validate_completed_fold(fold,plan,subject)
            print(f"[SKIP] completed formal fold S{subject}", flush=True)
            continue
        print(f"[RUN] formal fold S{subject}; excluded from all training: S6,S17", flush=True)
        atomic_json_dump({"status":"running","subject_id":subject},root/"status.json")
        try:
            metrics_path = train_loso_fold(dataset_root=plan["dataset_root"], test_subject_id=subject,
                epochs=plan["epochs"], batch_size=plan["batch_size"], lr=plan["lr"], device=device,
                output_dir=model_dir, seed=plan["seed"], resume=True, **plan["model_kwargs"])
            validate_completed_fold(fold,plan,subject)
        except Exception as exc:
            atomic_json_dump({"status":"failed","subject_id":subject,"error":str(exc)},root/"status.json")
            raise
    summary = summarize(model_dir)
    if (summary["n_folds"] != len(plan["test_subject_ids"])
            or sorted(f["subject_id"] for f in summary["per_fold"]) != sorted(plan["test_subject_ids"])):
        raise ValueError("Completed fold set disagrees with the run plan")
    write_summary(summary, model_dir)
    atomic_json_dump({"status":"completed","architecture":ARCHITECTURE,
        "smoke_test":plan["smoke_test"],"training_folds":len(plan["test_subject_ids"])},root/"completed.json")
    atomic_json_dump({"status":"completed"},root/"status.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--output-base",type=Path,default=SCRIPT_ROOT.parent/"local_artifacts/outputs/activity_channelpreserve_loso")
    p.add_argument("--cohort-subject-ids",default=",".join(map(str,range(1,18))))
    p.add_argument("--exclude-subject-ids",default="6,17")
    p.add_argument("--test-subject-ids",help="Optional subset for a separate partial job; default all formal subjects")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--cpu-threads",type=int,default=1)
    p.add_argument("--dry-run",action="store_true",help="Validate data and print immutable plan without training or writing output")
    p.add_argument("--resume",action="store_true",help="Resume identical data/code/configuration, including optimizer and RNG")
    p.add_argument("--smoke-test",action="store_true",help="Separate job: first requested fold only, 1 epoch and batch 2")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return
    run_plan(plan,resume=args.resume)


if __name__ == "__main__":
    main()
