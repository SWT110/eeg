"""Train user-specified class-weight batches with the approved multi-depth D model.

Example: --class-weight-sets '3,2,1|3,3,1|4,3,1' --device cuda:0.
Each candidate completes all requested LOSO folds before the next starts.
Different GPUs may run disjoint batches under the same --output-root.
Results are summarized for manual selection; no winner is chosen automatically.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from channel_preserving_multidepth_model import (
    ARCHITECTURE, AUX_WEIGHT, DEPTHS, MultiDepthChannelPreservingConformer, restore_from_config,
)
from evaluate_activity_failurecase import load_best_checkpoint
from experiment_manifest import collect_environment_info
from run_activity_five_models import sha256_file
import train_activity_channelpreserve as single
from train_activity_loso import (
    DEFAULT_DROPOUT, DEFAULT_EMB_SIZE, DEFAULT_NUM_HEADS,
    DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT, atomic_json_dump, train_loso_fold, validate_device,
)

PROTOCOL = "channel_preserving_multidepth_D_manual_class_weight_sweep_v1"
PROFILE = single.PROFILE
MODEL_KWARGS = {
    "architecture": ARCHITECTURE, "input_domain": "time_fft",
    "emb_size": DEFAULT_EMB_SIZE, "num_heads": DEFAULT_NUM_HEADS,
    "dropout": DEFAULT_DROPOUT, "transformer_encoder_dropout": DEFAULT_DROPOUT,
    "transformer_branches": 3, "transformer_depths": list(DEPTHS),
    "transformer_branch_fusion": "loss_softmax", "branch_loss_aux_weight": AUX_WEIGHT,
    "transformer_branch_qkv": "cross_depth",
    "transformer_branch_qkv_dropout": DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
    "classification_mode": "flat", "train_sampling": "original",
}
CODE_FILES = (*single.CODE_FILES, "channel_preserving_multidepth_model.py", Path(__file__).name)


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(SCRIPT_ROOT / name) for name in CODE_FILES}


def parse_weight_sets(raw: str) -> list[list[float]]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Specify class weights, e.g. --class-weight-sets '3,2,1|3,3,1'")
    result = []
    for group in raw.split("|"):
        try:
            values = [float(piece.strip()) for piece in group.split(",")]
        except ValueError as exc:
            raise ValueError("Class-weight sets must look like '3,2,1|3,3,1'") from exc
        if (len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values)
                or any(v > float(np.finfo(np.float32).max)
                       or v < float(np.nextafter(np.float32(0), np.float32(1))) for v in values)):
            raise ValueError("Each e_1/e_2/e_3 class-weight set needs three positive finite float32 values")
        if values in result:
            raise ValueError(f"Duplicate class-weight set: {values}; specify each set once per batch")
        result.append(values)
    return result


def weight_key(values: list[float]) -> str:
    def token(value):
        raw = str(float(value))
        if raw.endswith(".0"):
            raw = raw[:-2]
        return raw.replace(".", "p").replace("-", "m").replace("+", "")
    return "weights_" + "_".join(token(v) for v in values)


def build_batch_plan(args: argparse.Namespace) -> dict:
    weights = parse_weight_sets(args.class_weight_sets)
    # Retain the existing cohort, provenance and content-hash validation.
    base = single.build_plan(args)
    root = args.output_base.expanduser().resolve()
    if args.smoke_test:
        root = root / "smoke"
    if sorted(base["test_subject_ids"]) != base["included_subject_ids"]:
        root = root / ("partial_test_" + "_".join(f"S{s}" for s in base["test_subject_ids"]))
    model = MultiDepthChannelPreservingConformer(21, 1920, 961)
    config = model.architecture_config
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hashes = code_hashes()
    plans = []
    for values in weights:
        output = root / weight_key(values)
        plans.append({
            **base, "protocol": PROTOCOL, "models": [ARCHITECTURE],
            "architecture_config": config, "trainable_parameters": parameters,
            "class_weights": values, "class_weight_order": ["e_1", "e_2", "e_3"],
            "model_kwargs": {**MODEL_KWARGS, "class_weights": values},
            "weight_selection_rule": "manual comparison after experiment summaries; no automatic winner",
            "output_dir": str(output), "model_output_dir": str(output / "model"),
            "code_sha256": hashes,
        })
    return {"protocol": PROTOCOL, "output_root": str(root), "class_weight_sets": weights,
            "candidate_count": len(plans), "planned_training_folds": sum(p["planned_training_folds"] for p in plans),
            "execution_order": [weight_key(v) for v in weights], "runs": plans}


@contextmanager
def file_lock(path: Path, *, blocking: bool = False):
    """OS locks release after crashes too; a stale filename never blocks resume."""
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {path.parent}; assign disjoint weight batches") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def comparison_signature(plan: dict) -> dict:
    """Device and CPU allocation may differ between independent candidates."""
    keys = ("protocol", "hyperparameter_profile", "checkpoint_rule", "seed", "dataset_sha256",
            "dataset_shape", "included_subject_ids", "excluded_subject_ids", "test_subject_ids",
            "subject_window_counts", "epochs", "batch_size", "lr", "adam_betas", "adam_eps",
            "weight_decay", "train_sampling", "sr_augmentation", "architecture_config", "code_sha256",
            "smoke_test", "failurecase_used_for_selection")
    return {**{k: plan[k] for k in keys},
            "model_kwargs": {k: v for k, v in plan["model_kwargs"].items() if k != "class_weights"}}


def completed_record(plan: dict) -> dict:
    return {"status": "completed", "architecture": ARCHITECTURE, "class_weights": plan["class_weights"],
            "smoke_test": plan["smoke_test"], "training_folds": len(plan["test_subject_ids"])}


def validate_metrics(metrics: dict, plan: dict, subject: int) -> None:
    expected = {
        **plan["model_kwargs"], "architecture_config": plan["architecture_config"],
        "test_subject_id": subject, "seed": plan["seed"], "epochs": plan["epochs"],
        "batch_size": plan["batch_size"], "lr": plan["lr"],
        "optimizer": "Adam", "optimizer_betas": plan["adam_betas"],
        "transformer_branch_depths": list(DEPTHS), "transformer_weights_independent_by_domain": False,
        "n_channels": 21, "n_times": 1920, "n_classes": 3,
        "trainable_parameters": plan["trainable_parameters"],
        "n_train_samples": plan["dataset_shape"][0] - plan["subject_window_counts"][str(subject)],
        "n_test_samples": plan["subject_window_counts"][str(subject)],
        "loss": "LearnableSoftmaxWeightedBranchCrossEntropyLoss",
    }
    # The training API calls this transformer_depths; persisted metadata uses
    # transformer_branch_depths instead.
    expected.pop("transformer_depths")
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"D-model fold metadata mismatch: S{subject}, {key}")
    epoch = metrics.get("best_epoch")
    if type(epoch) is not int or not 1 <= epoch <= plan["epochs"]:
        raise ValueError("Invalid best_epoch")
    accuracy = metrics.get("best_test_acc")
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Invalid best_test_acc")
    branch_weights = np.asarray(metrics.get("transformer_branch_loss_weights", []), dtype=float)
    if (branch_weights.shape != (3,) or not np.isfinite(branch_weights).all()
            or (branch_weights < 0).any() or not np.isclose(branch_weights.sum(), 1, atol=1e-6)):
        raise ValueError("Invalid learned depth-branch weights")


def restore_checkpoint(checkpoint: dict, metrics: dict, plan: dict) -> torch.nn.Module:
    keys = ("architecture", "architecture_config", "test_subject_id", "class_weights", "depth", "emb_size",
            "num_heads", "dropout", "transformer_encoder_dropout", "n_channels", "n_times", "n_classes",
            "input_domain", "time_n_times", "fft_n_times", "transformer_branch_depths",
            "transformer_branch_fusion", "branch_loss_aux_weight", "transformer_branch_qkv",
            "transformer_branch_qkv_dropout", "transformer_branch_loss_weights",
            "time_transformer_branch_qkv_gamma", "fft_transformer_branch_qkv_gamma")
    for key in keys:
        if key not in checkpoint or checkpoint[key] != metrics.get(key):
            raise ValueError(f"Checkpoint and metrics disagree: {key}")
    if (checkpoint.get("epoch") != metrics["best_epoch"] - 1
            or checkpoint["architecture_config"] != plan["architecture_config"]
            or checkpoint["class_weights"] != plan["class_weights"]):
        raise ValueError("Checkpoint epoch / architecture / class weights disagree with the plan")
    model = restore_from_config(checkpoint["architecture_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for key, value in model.transformer_weight_metadata().items():
        # Softmax kernels can round slightly differently on CPU and CUDA.
        if not np.allclose(value, metrics[key], rtol=1e-6, atol=1e-7):
            raise ValueError(f"Learned weights in checkpoint and metrics disagree: {key}")
    return model.eval()


def validate_completed_fold(fold: Path, plan: dict, subject: int, *, restore: bool = True,
                            check_dataset: bool = True) -> dict:
    required = ["metrics.json", "test_predictions.npz", "epoch_history.json"]
    if restore:
        required.append("best_model.pt")
    for name in required:
        if not (fold / name).is_file():
            raise ValueError(f"Missing completed fold artifact: {fold / name}")
    metrics = json.loads((fold / "metrics.json").read_text())
    validate_metrics(metrics, plan, subject)
    history = json.loads((fold / "epoch_history.json").read_text())
    rows = history.get("history", [])
    if [r.get("epoch") for r in rows] != list(range(1, plan["epochs"] + 1)):
        raise ValueError(f"Incomplete epoch history: {fold}")
    for key in ("architecture", "architecture_config", "epochs", "test_subject_id", "class_weights"):
        if history.get(key) != metrics[key]:
            raise ValueError(f"History/metrics mismatch: {key}")
    with np.load(fold / "test_predictions.npz", allow_pickle=False) as saved:
        true, pred = saved["y_true"], saved["y_pred"]
    if (true.shape != (metrics["n_test_samples"],) or pred.shape != true.shape
            or not np.issubdtype(true.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer)
            or not np.isin(true, [0, 1, 2]).all() or not np.isin(pred, [0, 1, 2]).all()):
        raise ValueError(f"Invalid saved predictions: {fold}")
    matrix = np.bincount(true.astype(np.int64) * 3 + pred.astype(np.int64), minlength=9).reshape(3, 3)
    if (not math.isclose(float(np.mean(true == pred)), metrics["best_test_acc"], abs_tol=1e-12, rel_tol=0)
            or matrix.tolist() != metrics.get("confusion_matrix")):
        raise ValueError(f"Saved predictions disagree with metrics: {fold}")
    if check_dataset:
        data = Path(plan["dataset_root"])
        labels = np.load(data / "y.npy", mmap_mode="r", allow_pickle=False)
        ids = np.load(data / "subject_ids.npy", mmap_mode="r", allow_pickle=False)
        if not np.array_equal(true, labels[ids == subject]):
            raise ValueError(f"Saved test labels disagree with dataset: {fold}")
    if restore:
        restore_checkpoint(load_best_checkpoint(fold / "best_model.pt"), metrics, plan)
    return metrics


def metrics_from_matrix(matrix: list[list[int]]) -> dict:
    cm = np.asarray(matrix, dtype=np.float64)
    true, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
    recall = np.divide(tp, true, out=np.zeros(3), where=true > 0)
    f1 = np.divide(2 * tp, true + predicted, out=np.zeros(3), where=true + predicted > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            **{f"e{i+1}_recall": float(recall[i]) for i in range(3)},
            **{f"e{i+1}_f1": float(f1[i]) for i in range(3)},
            "min_e1_e2_recall": float(min(recall[:2]))}


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def summarize_weight_runs(root: Path) -> dict:
    """Recompute equal-subject statistics from saved predictions, without X.npy.

    Supports result archives as well as live training. Partial candidates remain
    explicitly partial; no automatic selection is made from either table.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No experiment output directory: {root}")
    with file_lock(root / ".summary.lock", blocking=True):
        groups, all_subjects, signature = [], [], None
        manifest_path = root / "sweep_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        if manifest is not None:
            signature = manifest["comparison_signature"]
        for directory in sorted(root.glob("weights_*")):
            if not directory.is_dir() or not (directory / "run_plan.json").is_file():
                continue
            plan = json.loads((directory / "run_plan.json").read_text())
            if plan.get("protocol") != PROTOCOL or weight_key(plan["class_weights"]) != directory.name:
                raise ValueError(f"Unexpected experiment in output root: {directory}")
            if parse_weight_sets(",".join(map(str, plan["class_weights"]))) != [plan["class_weights"]]:
                raise ValueError(f"Invalid experiment class weights: {directory}")
            current = comparison_signature(plan)
            if signature is not None and signature != current:
                raise ValueError("Output root mixes incompatible datasets, code, model profiles or fold sets")
            signature = current
            expected_dirs = {f"fold_subject_{s}" for s in plan["test_subject_ids"]}
            actual_dirs = {p.name for p in (directory / "model").glob("fold_subject_*") if p.is_dir()}
            if actual_dirs - expected_dirs:
                raise ValueError(f"Unexpected LOSO fold directories: {directory}")
            subjects, matrices = [], []
            for subject in plan["test_subject_ids"]:
                fold = directory / "model" / f"fold_subject_{subject}"
                if not (fold / "metrics.json").is_file():
                    continue
                metrics = validate_completed_fold(fold, plan, subject, restore=False, check_dataset=False)
                matrices.append(metrics["confusion_matrix"])
                subjects.append({"weight_key": directory.name, "class_weights": plan["class_weights"],
                    "subject_id": subject, "best_epoch": metrics["best_epoch"],
                    **metrics_from_matrix(metrics["confusion_matrix"]),
                    "learned_branch_weights": metrics["transformer_branch_loss_weights"],
                    "confusion_matrix": metrics["confusion_matrix"]})
            done = directory / "completed.json"
            if done.is_file() and (json.loads(done.read_text()) != completed_record(plan)
                                  or len(subjects) != len(plan["test_subject_ids"])):
                raise ValueError(f"Invalid completed experiment marker: {directory}")
            group = {"weight_key": directory.name, "class_weights": plan["class_weights"],
                     "status": "completed" if done.is_file() else "incomplete",
                     "smoke_test": plan["smoke_test"], "completed_folds": len(subjects),
                     "planned_folds": len(plan["test_subject_ids"]), "device": plan["device"],
                     "output_dir": str(directory), "per_subject": subjects}
            if subjects:
                score_keys = list(metrics_from_matrix(matrices[0]))
                group.update({"mean_" + key: float(np.mean([s[key] for s in subjects])) for key in score_keys})
                group["std_accuracy"] = float(np.std([s["accuracy"] for s in subjects]))
                group["e1_zero_recall_subjects"] = [s["subject_id"] for s in subjects if s["e1_recall"] == 0]
                group["e2_zero_recall_subjects"] = [s["subject_id"] for s in subjects if s["e2_recall"] == 0]
                pooled = np.sum(matrices, axis=0).tolist()
                group["pooled_confusion_matrix"] = pooled
                group["pooled_metrics"] = metrics_from_matrix(pooled)
            groups.append(group)
            all_subjects.extend(subjects)
        if manifest is not None:
            started = {g["weight_key"] for g in groups}
            for values in manifest["requested_class_weight_sets"]:
                key = weight_key(values)
                if key not in started:
                    groups.append({"weight_key": key, "class_weights": values, "status": "pending",
                                   "smoke_test": signature["smoke_test"], "completed_folds": 0,
                                   "planned_folds": len(signature["test_subject_ids"]), "device": "",
                                   "output_dir": str(root / key), "per_subject": []})
        groups.sort(key=lambda g: g["class_weights"])
        result = {"protocol": PROTOCOL, "selection_rule": "manual; no automatic winner",
                  "metric_rule": "calculate each subject first, then average subjects equally",
                  "checkpoint_rule": "highest outer test accuracy; first epoch wins ties",
                  "comparison_signature": signature, "candidate_count": len(groups),
                  "completed_candidates": sum(g["status"] == "completed" for g in groups),
                  "groups": groups}
        atomic_json_dump(result, root / "weight_summary.json")
        fields = ["weight_key", "class_weights", "status", "smoke_test", "completed_folds", "planned_folds",
                  "mean_accuracy", "std_accuracy", "mean_macro_f1", "mean_e1_recall", "mean_e2_recall",
                  "mean_e3_recall", "mean_min_e1_e2_recall", "mean_e1_f1", "mean_e2_f1", "mean_e3_f1", "device"]
        atomic_write_text(root / "weight_summary.csv", csv_text(groups, fields))
        subject_fields = ["weight_key", "class_weights", "subject_id", "best_epoch", "accuracy", "macro_f1",
                          "e1_recall", "e2_recall", "e3_recall", "min_e1_e2_recall", "e1_f1", "e2_f1", "e3_f1",
                          "learned_branch_weights", "confusion_matrix"]
        atomic_write_text(root / "subject_results.csv", csv_text(all_subjects, subject_fields))
        lines = ["# D 版类别权重实验汇总", "",
                 "指标先逐受试者计算再等权平均。数值单位为 %；不自动选取最佳权重。", "",
                 "每折仍按最高外层测试准确率选 epoch；结果属于探索性选参。未完成组仅展示进度，其均值不是完整 15 折成绩。", "",
                 "| e₁/e₂/e₃ 权重 | 状态 | 完成折数 | Acc | Macro-F1 | R₁ | R₂ | R₃ | 逐人 min(R₁,R₂) |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for group in groups:
            keys = ("mean_accuracy", "mean_macro_f1", "mean_e1_recall", "mean_e2_recall", "mean_e3_recall", "mean_min_e1_e2_recall")
            scores = [f"{100 * group[k]:.2f}" if k in group else "—" for k in keys]
            label = "/".join(str(v).removesuffix(".0") for v in group["class_weights"])
            status = ("冒烟测试/" if group["smoke_test"] else "") + {
                "completed": "完成", "incomplete": "未完成", "pending": "等待训练",
            }[group["status"]]
            lines.append("| " + " | ".join([label, status, f"{group['completed_folds']}/{group['planned_folds']}", *scores]) + " |")
        atomic_write_text(root / "weight_summary.md", "\n".join(lines) + "\n")
        return result


def run_weight(plan: dict, *, resume: bool = False) -> None:
    root = Path(plan["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".training.lock"):
        plan_path = root / "run_plan.json"
        if plan_path.is_file():
            if not resume:
                raise FileExistsError(f"Weight experiment exists: {root}; use --resume")
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Resume configuration/data/code hash mismatch; use a new output root")
        elif any(p.name != ".training.lock" for p in root.iterdir()):
            raise ValueError(f"Refusing to reuse an untracked experiment directory: {root}")
        atomic_json_dump(plan, plan_path)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        env_path = root / (f"resume_environment_{stamp}.json" if (root / "environment.json").exists() else "environment.json")
        atomic_json_dump(collect_environment_info(SCRIPT_ROOT.parent), env_path)
        try:
            for subject in plan["test_subject_ids"]:
                fold = root / "model" / f"fold_subject_{subject}"
                if resume and (fold / "metrics.json").is_file():
                    validate_completed_fold(fold, plan, subject)
                    print(f"[SKIP] {root.name} S{subject}: completed fold verified", flush=True)
                    continue
                atomic_json_dump({"status": "running", "subject_id": subject}, root / "status.json")
                print(f"[RUN] {root.name} S{subject} on {plan['device']}; S6/S17 excluded", flush=True)
                train_loso_fold(dataset_root=plan["dataset_root"], test_subject_id=subject,
                    epochs=plan["epochs"], batch_size=plan["batch_size"], lr=plan["lr"],
                    device=plan["device"], output_dir=root / "model", seed=plan["seed"],
                    resume=True, **plan["model_kwargs"])
                validate_completed_fold(fold, plan, subject)
                summarize_weight_runs(root.parent)
            atomic_json_dump(completed_record(plan), root / "completed.json")
            atomic_json_dump({"status": "completed"}, root / "status.json")
        except Exception as exc:
            atomic_json_dump({"status": "failed", "error": str(exc)}, root / "status.json")
            raise
    summarize_weight_runs(root.parent)


def run_batch(batch: dict, *, resume: bool = False) -> None:
    plans = batch["runs"]
    validate_device(plans[0]["device"])
    torch.set_num_threads(plans[0]["cpu_threads"])
    root = Path(batch["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".summary.lock", blocking=True):
        signature = comparison_signature(plans[0])
        manifest_path = root / "sweep_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest["comparison_signature"] != signature:
                raise ValueError("Output root already contains incompatible experiments; choose a new output root")
        else:
            manifest = {"protocol": PROTOCOL, "comparison_signature": signature, "requested_class_weight_sets": []}
        for path in root.glob("weights_*/run_plan.json"):
            if comparison_signature(json.loads(path.read_text())) != signature:
                raise ValueError("Output root already contains incompatible experiments; choose a new output root")
        for values in batch["class_weight_sets"]:
            if values not in manifest["requested_class_weight_sets"]:
                manifest["requested_class_weight_sets"].append(values)
        atomic_json_dump(manifest, manifest_path)
    for plan in plans:
        existing = Path(plan["output_dir"]) / "run_plan.json"
        if existing.is_file():
            if not resume:
                raise FileExistsError(f"Experiment exists: {existing.parent}; add --resume")
            if json.loads(existing.read_text()) != plan:
                raise ValueError("Resume configuration/data/code hash mismatch; use the original arguments")
    summarize_weight_runs(root)
    for index, plan in enumerate(plans, 1):
        print(f"[CANDIDATE {index}/{len(plans)}] {plan['class_weights']}", flush=True)
        run_weight(plan, resume=resume)
    print(f"[DONE] Batch completed. Summary: {root / 'weight_summary.md'}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", "--output-base", dest="output_base", type=Path,
                        default=SCRIPT_ROOT.parent / "local_artifacts/outputs/activity_channelpreserve_D_weights_seed42")
    parser.add_argument("--class-weight-sets", "--class-weights", dest="class_weight_sets",
                        help="User-selected triples in e_1/e_2/e_3 order; quote the | separators")
    parser.add_argument("--cohort-subject-ids", default=",".join(map(str, range(1, 18))))
    parser.add_argument("--exclude-subject-ids", default="6,17")
    parser.add_argument("--test-subject-ids", help="Separate partial job; defaults to all 15 formal subjects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Separate first-fold, 1-epoch, batch-2 job")
    parser.add_argument("--summarize-only", action="store_true", help="Read all weight groups in --output-root; no training/data needed")
    args = parser.parse_args(argv)
    if args.summarize_only:
        if args.smoke_test or args.dry_run or args.resume:
            parser.error("--summarize-only cannot be combined with training flags")
    elif args.dataset_root is None or args.class_weight_sets is None:
        parser.error("Training requires --dataset-root and --class-weight-sets")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.summarize_only:
        summary = summarize_weight_runs(args.output_base)
        print(json.dumps({"output_root": str(args.output_base.resolve()),
                          "candidate_count": summary["candidate_count"],
                          "completed_candidates": summary["completed_candidates"]}, ensure_ascii=False, indent=2))
        return
    batch = build_batch_plan(args)
    if args.dry_run:
        print(json.dumps(batch, ensure_ascii=False, indent=2))
        return
    run_batch(batch, resume=args.resume)


if __name__ == "__main__":
    main()
