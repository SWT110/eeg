"""Rerun the selected MDTF and four baselines on an explicitly filtered cohort.

Uses the historical per-epoch test-accuracy checkpoint rule. Subject selection
here verifies the arrays, rather than silently leaving excluded people in other
folds' training sets. Generate the filtered dataset first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from experiment_manifest import collect_environment_info
from summarize_loso_results import summarize, write_summary
from train_activity_comparison import run_comparisons
from train_activity_loso import atomic_json_dump, validate_device
from train_activity_loso_batch import parse_subject_id_list, run_loso_batch

MODELS = ("mdtf", "eegnet", "shallowconvnet", "atcnet", "tcformer")
MDTF_CONFIG = {
    "input_domain": "time_fft", "conv_type": "standard", "fft_global": "none",
    "input_qkv": "none", "cumulative_query_attention": False,
    "depth": 6, "transformer_branches": 3, "transformer_depths": [11, 10, 8],
    "transformer_branch_fusion": "loss_softmax", "branch_loss_aux_weight": 0.2,
    "transformer_branch_qkv": "cross_depth", "transformer_encoder_dropout": 0.85,
    "transformer_branch_qkv_dropout": 0.25, "classification_mode": "flat",
}
DATA_FILES = ("X.npy", "y.npy", "subject_ids.npy", "record_ids.npy", "window_indices.npy", "metadata.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(args: argparse.Namespace) -> dict:
    dataset = args.dataset_root.expanduser().resolve()
    cohort = sorted(parse_subject_id_list(args.cohort_subject_ids) or [])
    excluded = sorted(parse_subject_id_list(args.exclude_subject_ids) or [])
    if set(excluded) - set(cohort):
        raise ValueError("Exclusion list contains subjects outside --cohort-subject-ids")
    expected = sorted(set(cohort) - set(excluded))
    if len(expected) < 2:
        raise ValueError("LOSO needs at least two included subjects")
    models = [name.strip().lower() for name in args.models.split(",")]
    if not models or len(models) != len(set(models)) or set(models) - set(MODELS):
        raise ValueError(f"--models must be distinct names from {MODELS}")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.seed < 0:
        raise ValueError("epochs/batch-size/lr must be positive and seed nonnegative")
    for filename in DATA_FILES:
        if not (dataset / filename).is_file():
            raise FileNotFoundError(f"Missing dataset artifact: {dataset / filename}")
    x = np.load(dataset / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(dataset / "y.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(dataset / "subject_ids.npy", mmap_mode="r", allow_pickle=False)
    if x.ndim != 3 or tuple(x.shape[1:]) != (21, 1920):
        raise ValueError(f"Expected 21-channel 15-second/128Hz windows; got {x.shape}")
    if y.ndim != 1 or ids.ndim != 1 or len(x) != len(y) or len(x) != len(ids):
        raise ValueError("Dataset X/y/subject_ids dimensions disagree")
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Stimulus labels and subject IDs must have integer dtype")
    actual = sorted(int(sid) for sid in np.unique(ids))
    leaked = sorted(set(actual) & set(excluded))
    if leaked:
        raise ValueError(f"Excluded subjects still present in training arrays: {leaked}; regenerate dataset with --exclude-subject-ids")
    if actual != expected:
        raise ValueError(f"Dataset cohort mismatch: expected={expected}, actual={actual}")
    for sid in actual:
        if set(np.unique(y[ids == sid]).tolist()) != {0, 1, 2}:
            raise ValueError(f"Subject {sid} does not have all three stimulus labels")
    for filename in ("record_ids.npy", "window_indices.npy"):
        values = np.load(dataset / filename, mmap_mode="r", allow_pickle=False)
        if values.shape != y.shape:
            raise ValueError(f"Dataset provenance length mismatch: {filename}")
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("window_seconds") != 15 or metadata.get("stride_seconds") != 3:
        raise ValueError("Dataset metadata must specify 15-second windows, 3-second stride")
    if sorted(metadata.get("excluded_subject_ids", [])) != excluded:
        raise ValueError("Dataset exclusion metadata disagrees with requested policy")
    if metadata.get("included_subject_ids") != actual or metadata.get("n_samples") != len(x):
        raise ValueError("Dataset inclusion/count metadata disagrees with arrays")
    tests = parse_subject_id_list(args.test_subject_ids) or actual
    if set(tests) - set(actual):
        raise ValueError("Requested test subjects are absent from formal cohort")
    tag = dataset.parent.name if dataset.name == "window_15_stride_3" else dataset.name
    suffix = "_exclude_" + "_".join(f"S{sid}" for sid in excluded) if excluded else ""
    if suffix and not tag.endswith(suffix):
        tag += suffix
    job_name = "job_" + "_".join(models)
    if sorted(tests) != actual:
        job_name += "_test_" + "_".join(f"S{sid}" for sid in sorted(tests))
    output = args.output_base.expanduser().resolve() / tag / f"seed_{args.seed}" / job_name
    code_names = ("run_activity_five_models.py", "train_activity_loso.py", "train_activity_loso_batch.py",
                  "train_activity_comparison.py", "comparison_models.py", "summarize_loso_results.py")
    return {
        "protocol": "exploratory_test_guided_loso_v1",
        "checkpoint_rule": "highest outer test accuracy; first epoch wins ties",
        "dataset_root": str(dataset), "dataset_shape": list(x.shape),
        "included_subject_ids": actual, "excluded_subject_ids": excluded,
        "test_subject_ids": tests, "models": models, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "class_weights": [3.0, 3.0, 1.0], "adam_betas": [0.5, 0.999],
        "mdtf_config": MDTF_CONFIG, "device": args.device,
        "planned_training_folds": len(tests) * len(models),
        "output_dir": str(output),
        "dataset_sha256": {name: sha256_file(dataset / name) for name in DATA_FILES},
        "code_sha256": {name: sha256_file(SCRIPT_ROOT / name) for name in code_names},
        "failurecase_used_for_selection": False,
    }


def run_plan(plan: dict, *, resume: bool = False) -> None:
    validate_device(plan["device"])
    root = Path(plan["output_dir"])
    existing_plan = root / "run_plan.json"
    if root.exists():
        if not resume:
            raise FileExistsError(f"Run already exists: {root}; choose a new --output-base or use --resume")
        if not existing_plan.is_file():
            raise ValueError(f"Cannot resume an untracked run: {root}")
        previous = json.loads(existing_plan.read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError("Resume configuration/data/code hash mismatch; choose a new output base")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(plan, existing_plan)
    environment_path = root / "environment.json"
    environment = collect_environment_info(SCRIPT_ROOT.parent)
    if not environment_path.exists():
        atomic_json_dump(environment, environment_path)
    else:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_json_dump(environment, root / f"resume_environment_{stamp}.json")
    common = dict(dataset_root=Path(plan["dataset_root"]), epochs=plan["epochs"],
                  batch_size=plan["batch_size"], lr=plan["lr"], device=plan["device"],
                  # Always save optimizer/RNG checkpoints from the first epoch.
                  class_weights=plan["class_weights"], skip_existing=resume, resume=True)
    if "mdtf" in plan["models"]:
        model_root = root / "mdtf" / f"seed_{plan['seed']}"
        run_loso_batch(subject_ids=plan["test_subject_ids"], output_dir=model_root,
                       seed=plan["seed"], **common, **MDTF_CONFIG)
        write_summary(summarize(model_root), model_root)
    external = [name for name in plan["models"] if name != "mdtf"]
    if external:
        run_comparisons(output_dir=root / "external", models=external,
                        subject_ids=plan["test_subject_ids"], seeds=[plan["seed"]], **common)
    model_summaries = {}
    for name in plan["models"]:
        model_dir = (root / "mdtf" if name == "mdtf" else root / "external" / name) / f"seed_{plan['seed']}"
        model_summaries[name] = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    atomic_json_dump({"protocol": plan["protocol"], "included_subject_ids": plan["included_subject_ids"],
                      "excluded_subject_ids": plan["excluded_subject_ids"], "models": model_summaries},
                     root / "five_model_summary.json")
    atomic_json_dump({"status": "completed", "models": plan["models"],
                      "training_folds": plan["planned_training_folds"]}, root / "completed.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-base", type=Path,
                        default=SCRIPT_ROOT.parent / "local_artifacts" / "outputs" / "activity_cohort_rerun")
    parser.add_argument("--cohort-subject-ids", default=",".join(str(sid) for sid in range(1, 18)))
    parser.add_argument("--exclude-subject-ids", default="6",
                        help="Verify exclusion in arrays and output naming; filtering is done by dataset builder")
    parser.add_argument("--test-subject-ids", default=None, help="Optional comma-separated LOSO test folds")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="Resume only with identical plan, data, and code hashes")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print plan without constructing models or training")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.dry_run:
        run_plan(plan, resume=args.resume)


if __name__ == "__main__":
    main()
