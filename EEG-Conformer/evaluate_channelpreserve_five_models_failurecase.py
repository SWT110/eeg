"""Evaluate S6/S17 for a completed job from the new five-model runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from channel_preserving_model import ARCHITECTURE
from evaluate_activity_failurecase import (
    Dataset, apply_legacy_normalizers, average_fold_probabilities, classification_metrics,
    fit_legacy_normalizers, load_best_checkpoint, load_dataset, predict_probabilities,
    read_json, verify_saved_predictions,
)
from experiment_manifest import collect_environment_info
import run_activity_channelpreserve_five_models as suite
from run_activity_five_models import sha256_file
from train_activity_loso import atomic_json_dump, atomic_save_npz, validate_device

FORMAL_SUBJECTS = [s for s in range(1, 18) if s not in (6, 17)]


def validate_run(args: argparse.Namespace, formal: Dataset, case: Dataset) -> tuple[dict, list[str], dict]:
    root = args.run_root.expanduser().resolve()
    plan = read_json(root / "run_plan.json")
    if (plan.get("protocol") != suite.PROTOCOL or plan.get("smoke_test") is not False
            or plan.get("included_subject_ids") != FORMAL_SUBJECTS
            or plan.get("test_subject_ids") != FORMAL_SUBJECTS
            or plan.get("excluded_subject_ids") != [6, 17]
            or plan.get("failurecase_used_for_selection") is not False):
        raise ValueError("Failurecase requires complete production folds for all 15 formal subjects")
    if (formal.metadata["included_subject_ids"] != FORMAL_SUBJECTS
            or sorted(formal.metadata.get("excluded_subject_ids", [])) != [6, 17]
            or plan.get("dataset_root") != str(formal.root)
            or plan.get("dataset_sha256") != formal.sha256):
        raise ValueError("Formal dataset/cohort/hash mismatch")
    if (case.metadata["included_subject_ids"] != [args.case_subject_id]
            or args.case_subject_id in case.metadata.get("excluded_subject_ids", [])
            or case.metadata["input_root"] != formal.metadata["input_root"]):
        raise ValueError("Failurecase must contain only the excluded subject and share upstream input_root")
    if args.seed is not None and plan.get("seed") != args.seed:
        raise ValueError("Seed assertion mismatch")
    planned = plan.get("models")
    if (not isinstance(planned, list) or not planned or len(set(planned)) != len(planned)
            or set(planned) - set(suite.MODELS)):
        raise ValueError("Invalid model set in run_plan")
    # Rebuild the immutable production plan: includes all code/data hashes,
    # unweighted local defaults and each external architecture's own config.
    expected = suite.build_plan(suite.parse_args([
        "--dataset-root", str(formal.root), "--output-base", str(root.parents[2]),
        "--models", ",".join(planned), "--seed", str(plan.get("seed")),
        "--device", str(plan.get("device")), "--cpu-threads", str(plan.get("cpu_threads")),
    ]))
    if plan != expected or plan["output_dir"] != str(root):
        raise ValueError("Production configuration/data/code hash mismatch; restore this run's training commit")
    if read_json(root / "completed.json") != suite.completed_record(plan):
        raise ValueError("Missing a completed production job marker")
    models = planned if args.models is None else [s.strip().lower() for s in args.models.split(",")]
    if not models or len(set(models)) != len(models) or set(models) - set(planned):
        raise ValueError("Requested models must be distinct members of this run")
    records = {}
    for name in planned:
        model_dir = suite.model_run_dir(root, name, plan["seed"])
        found = {p.name for p in model_dir.glob("fold_subject_*") if p.is_dir()}
        if found != {f"fold_subject_{s}" for s in FORMAL_SUBJECTS}:
            raise ValueError(f"Incomplete or unexpected fold set: {name}")
        records[name] = {}
        for subject in FORMAL_SUBJECTS:
            fold = model_dir / f"fold_subject_{subject}"
            metrics = suite.validate_completed_fold(fold, plan, name, subject)
            records[name][subject] = {"directory": fold, "metrics": metrics}
    return plan, models, records


def evaluate(args: argparse.Namespace) -> dict:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite failurecase output: {output}")
    if (args.case_subject_id not in (6, 17) or args.batch_size < 1 or args.cpu_threads < 1
            or (args.seed is not None and args.seed < 0)):
        raise ValueError("Case must be S6/S17; positive batch/threads and nonnegative seed required")
    device = torch.device(validate_device(args.device))
    torch.set_num_threads(args.cpu_threads)
    formal, case = load_dataset(args.dataset_root), load_dataset(args.failurecase_root)
    plan, models, records = validate_run(args, formal, case)
    subjects = plan["included_subject_ids"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": "channelpreserving_suite_excluded_subject_probability_ensemble_v1",
        "models": models, "case_subject_id": args.case_subject_id, "seed": plan["seed"],
        "formal_subject_ids": subjects, "excluded_subject_ids": [6, 17],
        "ensemble_rule": "per model: equal mean of per-window softmax probabilities across 15 folds only",
        "fold_weights": [1.0 / len(subjects)] * len(subjects),
        "case_used_for_training_or_selection": False,
        "case_labels_role": "integrity checks and metrics only after all probabilities are fixed",
        "dataset_root": str(formal.root), "failurecase_root": str(case.root),
        "run_root": str(args.run_root.resolve()),
        "run_plan_sha256": sha256_file(args.run_root.resolve() / "run_plan.json"),
        "formal_dataset_sha256": formal.sha256, "failurecase_dataset_sha256": case.sha256,
        "training_code_sha256": plan["code_sha256"],
        "evaluation_code_sha256": sha256_file(Path(__file__)),
        "record_sources": {"formal": formal.metadata["record_sources"],
                           "failurecase": case.metadata["record_sources"]},
        "runtime": collect_environment_info(SCRIPT_ROOT.parent), "device": str(device),
        "case_batch_size": args.batch_size, "folds": {},
        "limitations": [
            "S6/S17 were selected after historical performance inspection; failurecase analysis is separate.",
            "Formal training retains exploratory outer-test epoch selection and upstream whole-record normalization.",
            "Fold models share training participants; each architecture's failurecase ensemble differs from its LOSO mean.",
        ],
    }
    atomic_json_dump(manifest, output / "evaluation_manifest.json")
    formal_X = formal.X.astype(np.float32, copy=False)
    case_X = case.X.astype(np.float32, copy=False)
    probabilities = {name: np.empty((len(subjects), len(case_X), 3), dtype=np.float32) for name in models}
    normalizers = {}
    for index, subject in enumerate(subjects):
        stats = fit_legacy_normalizers(formal_X[formal.subject_ids != subject], dual=ARCHITECTURE in models)
        normalizers[str(subject)] = {"train_subject_ids": [s for s in subjects if s != subject], "statistics": stats}
        atomic_json_dump(normalizers, output / "normalizers.json")
        test_mask = formal.subject_ids == subject
        manifest["folds"][str(subject)] = {}
        for name in models:
            record = records[name][subject]
            checkpoint_path = record["directory"] / "best_model.pt"
            model = suite.restore_suite_model(load_best_checkpoint(checkpoint_path), record["metrics"], name, plan).to(device)
            dual = name == ARCHITECTURE
            held_inputs = apply_legacy_normalizers(formal_X[test_mask], stats, dual)
            _, replay = predict_probabilities(model, held_inputs, device, record["metrics"]["batch_size"])
            verification = verify_saved_predictions(replay, formal.y[test_mask], record)
            del held_inputs, replay
            case_inputs = apply_legacy_normalizers(case_X, stats, dual)
            case_probs, _ = predict_probabilities(model, case_inputs, device, args.batch_size)
            probabilities[name][index] = case_probs
            atomic_save_npz(output / name / f"fold_subject_{subject}" / "probabilities.npz",
                            probabilities=case_probs, sample_indices=np.arange(len(case_X)),
                            subject_ids=case.subject_ids, record_ids=case.record_ids, window_indices=case.window_indices)
            manifest["folds"][str(subject)][name] = {
                "best_epoch": record["metrics"]["best_epoch"], "held_out_replay": verification,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                "metrics_sha256": sha256_file(record["directory"] / "metrics.json"),
                "test_predictions_sha256": sha256_file(record["directory"] / "test_predictions.npz"),
            }
            atomic_json_dump(manifest, output / "evaluation_manifest.json")
            print(f"S{args.case_subject_id}: {name}, formal fold S{subject}; original test replay passed", flush=True)
            del model, case_inputs, case_probs
    results = {}
    for name in models:
        ensemble = average_fold_probabilities(probabilities[name])
        per_fold = []
        for index, subject in enumerate(subjects):
            metrics = {"formal_test_subject_id": subject, **classification_metrics(case.y, probabilities[name][index])}
            atomic_json_dump(metrics, output / name / f"fold_subject_{subject}" / "metrics.json")
            per_fold.append(metrics)
        results[name] = {"n_fold_models": len(subjects), "ensemble": classification_metrics(case.y, ensemble),
                         "per_fold": per_fold}
        atomic_save_npz(output / name / "ensemble_predictions.npz", fold_subject_ids=np.array(subjects),
                        fold_probabilities=probabilities[name], probabilities=ensemble,
                        y_true=case.y, y_pred=ensemble.argmax(axis=-1), sample_indices=np.arange(len(case_X)),
                        subject_ids=case.subject_ids, record_ids=case.record_ids, window_indices=case.window_indices)
        atomic_json_dump(results[name], output / name / "summary.json")
    summary = {"protocol": manifest["protocol"], "case_subject_id": args.case_subject_id, "seed": plan["seed"],
               "formal_subject_ids": subjects, "excluded_subject_ids": [6, 17],
               "case_used_for_training_or_selection": False, "models": results}
    atomic_json_dump(summary, output / "failurecase_summary.json")
    atomic_json_dump({"status": "completed", "case_subject_id": args.case_subject_id, "models": models,
                      "n_verified_checkpoints": len(subjects) * len(models)}, output / "completed.json")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--failurecase-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-subject-id", type=int, required=True, choices=(6, 17))
    parser.add_argument("--models", help="Optional subset of this job's completed models")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "case_subject_id": summary["case_subject_id"],
                      "models": {name: result["ensemble"] for name, result in summary["models"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
