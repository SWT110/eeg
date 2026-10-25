"""Evaluate S6/S17 after the user freezes one completed D class-weight experiment.

The 15 fold models must all be complete. Each original held-out prediction is
replayed before scoring the case; training-only normalization and equal-fold
probability averaging reuse the existing evaluator. Cases never select weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import evaluate_channelpreserve_failurecase as engine
import train_activity_multidepth_weight_sweep as training
from channel_preserving_multidepth_model import ARCHITECTURE
from evaluate_activity_failurecase import Dataset, read_json
from run_activity_five_models import sha256_file


def validate_run(args: argparse.Namespace, formal: Dataset, case: Dataset) -> tuple[dict, dict]:
    root = args.run_root.expanduser().resolve()
    plan = read_json(root / "run_plan.json")
    if (formal.metadata["included_subject_ids"] != engine.FORMAL_SUBJECTS
            or sorted(formal.metadata.get("excluded_subject_ids", [])) != [6, 17]):
        raise ValueError("D Failurecase requires the full 15-person formal cohort excluding S6/S17")
    if (case.metadata["included_subject_ids"] != [args.case_subject_id]
            or args.case_subject_id in case.metadata.get("excluded_subject_ids", [])):
        raise ValueError(f"Failurecase dataset must contain only S{args.case_subject_id}")
    if formal.metadata["input_root"] != case.metadata["input_root"]:
        raise ValueError("Formal and failurecase input_root must match")
    if type(plan.get("seed")) is not int or plan["seed"] < 0:
        raise ValueError("Invalid training seed")
    if args.seed is not None and args.seed != plan["seed"]:
        raise ValueError("Seed assertion mismatch")
    weights = training.parse_weight_sets(",".join(map(str, plan.get("class_weights", []))))
    config = plan.get("architecture_config", {})
    expected_args = training.parse_args([
        "--dataset-root", str(formal.root), "--output-root", str(root.parent),
        "--class-weight-sets", ",".join(map(str, weights[0])), "--seed", str(plan["seed"]),
        "--cpu-threads", str(plan.get("cpu_threads")), "--device", str(plan.get("device")),
        "--transformer-depths", ",".join(map(str, config.get("depths", []))),
        "--split-after-blocks", ",".join(map(str, config.get("split_after_blocks", []))),
    ])
    expected = training.build_batch_plan(expected_args)["runs"][0]
    if plan != expected:
        raise ValueError("D Failurecase requires an unchanged, complete production plan/data/code; smoke/partial runs are rejected")
    if read_json(root / "completed.json") != training.completed_record(plan):
        raise ValueError("Missing completed D production run marker")
    model_root = root / "model"
    actual = {p.name for p in model_root.glob("fold_subject_*") if p.is_dir()}
    if actual != {f"fold_subject_{sid}" for sid in engine.FORMAL_SUBJECTS}:
        raise ValueError("Incomplete or unexpected D formal fold directories")
    records = {}
    for subject in engine.FORMAL_SUBJECTS:
        fold = model_root / f"fold_subject_{subject}"
        records[subject] = {"directory": fold, "metrics": training.validate_completed_fold(fold, plan, subject)}
    return plan, records


def evaluate(args: argparse.Namespace) -> dict:
    return engine.evaluate(args, run_validator=validate_run, checkpoint_restorer=training.restore_checkpoint,
        architecture=ARCHITECTURE, protocol="channelpreserving_multidepth_D_excluded_equal_fold_probability_v1",
        evaluation_metadata={"evaluation_adapter_code_sha256": sha256_file(Path(__file__))})


def main(argv: list[str] | None = None) -> None:
    args = engine.parse_args(argv)
    summary = evaluate(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "case_subject_id": summary["case_subject_id"],
                      "training_class_weights": summary["training_class_weights"],
                      "ensemble": summary["ensemble"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
