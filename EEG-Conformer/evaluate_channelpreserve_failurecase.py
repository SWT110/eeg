"""Score excluded S6 or S17 using all 15 frozen channel-preserving LOSO models.

Reconstruct each fold's training-only normalization, replay its original held-out
predictions, then average per-window probabilities across the 15 fold models.
Smoke tests, partial jobs and historical cross-depth models are rejected.
"""
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

from channel_preserving_model import ARCHITECTURE, ChannelPreservingDualConformer
from evaluate_activity_failurecase import (
    Dataset, apply_legacy_normalizers, average_fold_probabilities,
    classification_metrics, fit_legacy_normalizers, load_best_checkpoint,
    load_dataset, predict_probabilities, read_json, verify_saved_predictions,
)
from experiment_manifest import collect_environment_info
from run_activity_five_models import sha256_file
from train_activity_channelpreserve import (
    MODEL_KWARGS, PROFILE, PROTOCOL, code_hashes, restore_checkpoint,
    validate_completed_fold,
)
from train_activity_loso import (
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR,
    atomic_json_dump, atomic_save_npz, validate_device,
)

FORMAL_SUBJECTS = [sid for sid in range(1, 18) if sid not in (6, 17)]


def validate_run(args: argparse.Namespace, formal: Dataset, case: Dataset) -> tuple[dict, dict]:
    root = args.run_root.expanduser().resolve()
    plan = read_json(root / "run_plan.json")
    expected = {
        "protocol": PROTOCOL, "hyperparameter_profile": PROFILE,
        "models": [ARCHITECTURE], "model_kwargs": MODEL_KWARGS,
        "included_subject_ids": FORMAL_SUBJECTS, "excluded_subject_ids": [6, 17],
        "smoke_test": False, "planned_training_folds": 15,
        "epochs": DEFAULT_EPOCHS, "batch_size": DEFAULT_BATCH_SIZE, "lr": DEFAULT_LR,
        "adam_betas": [0.5, 0.999], "adam_eps": 1e-8, "weight_decay": 0.0,
        "class_weights": None, "train_sampling": "original", "sr_augmentation": False,
        "failurecase_used_for_selection": False,
        "dataset_root": str(formal.root), "dataset_shape": list(formal.X.shape),
        "dataset_sha256": formal.sha256, "output_dir": str(root),
        "model_output_dir": str(root / "model"),
    }
    for key, value in expected.items():
        if key not in plan or plan[key] != value:
            raise ValueError(f"Failurecase requires a complete 15-fold production run; mismatch: {key}")
    tests = plan.get("test_subject_ids", [])
    if sorted(tests) != FORMAL_SUBJECTS or len(tests) != 15:
        raise ValueError("Failurecase requires every formal LOSO fold; partial jobs are rejected")
    if (not isinstance(plan.get("seed"), int) or plan["seed"] < 0
            or (args.seed is not None and plan["seed"] != args.seed)):
        raise ValueError("Invalid seed or seed assertion mismatch")
    if (formal.metadata["included_subject_ids"] != FORMAL_SUBJECTS
            or sorted(formal.metadata.get("excluded_subject_ids", [])) != [6, 17]):
        raise ValueError("S6 and S17 must be absent from the formal dataset")
    if (case.metadata["included_subject_ids"] != [args.case_subject_id]
            or args.case_subject_id in case.metadata.get("excluded_subject_ids", [])):
        raise ValueError(f"Failurecase dataset must contain only S{args.case_subject_id}")
    if formal.metadata["input_root"] != case.metadata["input_root"]:
        raise ValueError("Formal and failurecase input_root must match")
    subjects, counts = np.unique(formal.subject_ids, return_counts=True)
    if plan.get("subject_window_counts") != {str(int(s)): int(c) for s, c in zip(subjects, counts)}:
        raise ValueError("Formal subject window counts disagree with the run plan")
    if plan.get("code_sha256") != code_hashes():
        raise ValueError("Training code hash mismatch; use the commit that produced this run")
    reference = ChannelPreservingDualConformer(21, 1920, 961)
    if (plan.get("architecture_config") != reference.architecture_config
            or plan.get("trainable_parameters") != sum(p.numel() for p in reference.parameters())):
        raise ValueError("Architecture differs from the approved 5+5 dual-skip model")
    del reference
    done = read_json(root / "completed.json")
    if done != {"status": "completed", "architecture": ARCHITECTURE,
                "smoke_test": False, "training_folds": 15}:
        raise ValueError("Missing a completed production run marker")
    model_dir = root / "model"
    actual = {p.name for p in model_dir.glob("fold_subject_*") if p.is_dir()}
    if actual != {f"fold_subject_{sid}" for sid in FORMAL_SUBJECTS}:
        raise ValueError("Incomplete or unexpected formal fold directories")
    records = {}
    for subject in FORMAL_SUBJECTS:
        fold = model_dir / f"fold_subject_{subject}"
        metrics = validate_completed_fold(fold, plan, subject)
        records[subject] = {"directory": fold, "metrics": metrics}
    return plan, records


def evaluate(args: argparse.Namespace, *, run_validator=None, checkpoint_restorer=None,
             architecture: str = ARCHITECTURE,
             protocol: str = "channelpreserving_excluded_subject_equal_fold_probability_v1",
             evaluation_metadata: dict | None = None) -> dict:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite failurecase output: {output}")
    if (args.case_subject_id not in (6, 17) or args.batch_size < 1 or args.cpu_threads < 1
            or (args.seed is not None and args.seed < 0)):
        raise ValueError("case-subject-id must be 6 or 17; positive batch/threads and nonnegative seed required")
    device = torch.device(validate_device(args.device))
    torch.set_num_threads(args.cpu_threads)
    formal, case = load_dataset(args.dataset_root), load_dataset(args.failurecase_root)
    plan, records = (run_validator or validate_run)(args, formal, case)
    subjects = plan["included_subject_ids"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": protocol,
        "architecture": architecture, "architecture_config": plan["architecture_config"],
        "training_class_weights": plan["class_weights"],
        "case_subject_id": args.case_subject_id, "seed": plan["seed"],
        "formal_subject_ids": subjects, "excluded_subject_ids": [6, 17],
        "ensemble_rule": "mean of per-window softmax probabilities along the fold axis only",
        "fold_weights": [1.0 / len(subjects)] * len(subjects),
        "case_used_for_training_or_selection": False,
        "case_labels_role": "record-integrity checks and metrics after all predictions are fixed",
        "dataset_root": str(formal.root), "failurecase_root": str(case.root),
        "run_root": str(args.run_root.expanduser().resolve()),
        "run_plan_sha256": sha256_file(args.run_root.expanduser().resolve() / "run_plan.json"),
        "formal_dataset_sha256": formal.sha256, "failurecase_dataset_sha256": case.sha256,
        "training_code_sha256": plan["code_sha256"],
        "evaluation_code_sha256": sha256_file(Path(__file__)),
        "record_sources": {"formal": formal.metadata["record_sources"],
                           "failurecase": case.metadata["record_sources"]},
        "runtime": collect_environment_info(SCRIPT_ROOT.parent), "device": str(device),
        "case_batch_size": args.batch_size, "folds": {},
        "limitations": [
            "S6 and S17 were selected after historical performance inspection; this is a separate failurecase analysis.",
            "The formal run retains per-epoch outer-test checkpoint selection and is exploratory.",
            "Upstream windows retain complete-record normalization; fold statistics do not undo that adaptation.",
            "Fold models share training participants. Their ensemble is separate from the formal LOSO mean.",
        ],
    }
    manifest.update(evaluation_metadata or {})
    atomic_json_dump(manifest, output / "evaluation_manifest.json")
    formal_X = formal.X.astype(np.float32, copy=False)
    case_X = case.X.astype(np.float32, copy=False)
    probabilities = np.empty((len(subjects), len(case_X), 3), dtype=np.float32)
    normalizers = {}
    for index, subject in enumerate(subjects):
        record = records[subject]
        stats = fit_legacy_normalizers(formal_X[formal.subject_ids != subject], dual=True)
        normalizers[str(subject)] = {
            "train_subject_ids": [sid for sid in subjects if sid != subject], "statistics": stats,
        }
        atomic_json_dump(normalizers, output / "normalizers.json")
        checkpoint_path = record["directory"] / "best_model.pt"
        model = (checkpoint_restorer or restore_checkpoint)(load_best_checkpoint(checkpoint_path), record["metrics"], plan).to(device)
        test_mask = formal.subject_ids == subject
        held_inputs = apply_legacy_normalizers(formal_X[test_mask], stats, dual=True)
        _, replay = predict_probabilities(model, held_inputs, device, record["metrics"]["batch_size"])
        verification = verify_saved_predictions(replay, formal.y[test_mask], record)
        del held_inputs, replay
        case_inputs = apply_legacy_normalizers(case_X, stats, dual=True)
        case_probs, _ = predict_probabilities(model, case_inputs, device, args.batch_size)
        probabilities[index] = case_probs
        atomic_save_npz(output / f"fold_subject_{subject}" / "probabilities.npz",
                        probabilities=case_probs, sample_indices=np.arange(len(case_X)),
                        subject_ids=case.subject_ids, record_ids=case.record_ids,
                        window_indices=case.window_indices)
        manifest["folds"][str(subject)] = {
            "best_epoch": record["metrics"]["best_epoch"], "held_out_replay": verification,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics_sha256": sha256_file(record["directory"] / "metrics.json"),
            "test_predictions_sha256": sha256_file(record["directory"] / "test_predictions.npz"),
        }
        atomic_json_dump(manifest, output / "evaluation_manifest.json")
        print(f"S{args.case_subject_id}: formal fold S{subject}; original test replay passed", flush=True)
        del model, case_inputs, case_probs
    # Case labels enter metric calculations only after every prediction is fixed.
    ensemble = average_fold_probabilities(probabilities)
    per_fold = []
    for index, subject in enumerate(subjects):
        metrics = {"formal_test_subject_id": subject, **classification_metrics(case.y, probabilities[index])}
        atomic_json_dump(metrics, output / f"fold_subject_{subject}" / "metrics.json")
        per_fold.append(metrics)
    summary = {
        "protocol": manifest["protocol"], "architecture": architecture,
        "training_class_weights": plan["class_weights"],
        "case_subject_id": args.case_subject_id, "seed": plan["seed"],
        "formal_subject_ids": subjects, "excluded_subject_ids": [6, 17],
        "case_used_for_training_or_selection": False, "n_fold_models": len(subjects),
        "ensemble": classification_metrics(case.y, ensemble), "per_fold": per_fold,
    }
    atomic_save_npz(output / "ensemble_predictions.npz", fold_subject_ids=np.array(subjects),
                    fold_probabilities=probabilities, probabilities=ensemble,
                    y_true=case.y, y_pred=ensemble.argmax(axis=-1), sample_indices=np.arange(len(case_X)),
                    subject_ids=case.subject_ids, record_ids=case.record_ids,
                    window_indices=case.window_indices)
    atomic_json_dump(summary, output / "failurecase_summary.json")
    atomic_json_dump({"status": "completed", "case_subject_id": args.case_subject_id,
                      "n_verified_checkpoints": len(subjects)}, output / "completed.json")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--failurecase-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing run_plan.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths refused")
    parser.add_argument("--case-subject-id", type=int, required=True, choices=(6, 17))
    parser.add_argument("--seed", type=int, help="Optional assertion against training seed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "case_subject_id": summary["case_subject_id"], "ensemble": summary["ensemble"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
