"""Evaluate an excluded participant with the frozen best models of a complete LOSO run.

The legacy checkpoints omit normalization statistics. Reconstruct the original
float32, whole-array training statistics for each fold, replay that fold's saved
test predictions, then score the failurecase without labels. Average softmax
probabilities across folds for each window; never average across windows.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from comparison_models import build_comparison_model
from experiment_manifest import collect_environment_info
from run_activity_five_models import DATA_FILES, MDTF_CONFIG, MODELS, sha256_file
from train_activity_loso import (
    DEFAULT_ARCHITECTURE, DualBranchActivityConformer, atomic_json_dump, atomic_save_npz,
    transform_windows_to_fft, validate_device,
)

LABEL_MAP = {"e_1": 0, "e_2": 1, "e_3": 2}
TRAINING_CODE_FILES = {
    "run_activity_five_models.py", "train_activity_loso.py", "train_activity_loso_batch.py",
    "train_activity_comparison.py", "comparison_models.py", "summarize_loso_results.py",
}


@dataclass
class Dataset:
    root: Path
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    record_ids: np.ndarray
    window_indices: np.ndarray
    metadata: dict
    sha256: dict[str, str]


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_dataset(root: Path | str) -> Dataset:
    root = Path(root).expanduser().resolve()
    for name in DATA_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing dataset artifact: {root / name}")
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("X", "y", "subject_ids", "record_ids", "window_indices")}
    X, y = arrays["X"], arrays["y"]
    if X.ndim != 3 or tuple(X.shape[1:]) != (21, 1920) or len(X) == 0:
        raise ValueError(f"Expected nonempty [N,21,1920] windows: {root}; got {X.shape}")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("EEG windows must have floating dtype")
    for name in ("y", "subject_ids", "record_ids", "window_indices"):
        values = arrays[name]
        if values.shape != (len(X),) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Dataset provenance/label array is invalid: {name}")
    subjects = np.unique(arrays["subject_ids"]).astype(int).tolist()
    if any(subject < 1 for subject in subjects):
        raise ValueError("Subject IDs must be positive")
    for subject in subjects:
        if np.unique(y[arrays["subject_ids"] == subject]).tolist() != [0, 1, 2]:
            raise ValueError(f"Subject {subject} does not have all three classes")
    metadata = read_json(root / "metadata.json")
    expected = {
        "window_seconds": 15, "stride_seconds": 3, "n_samples": len(X),
        "n_channels": 21, "n_times": 1920, "n_subjects": len(subjects),
        "included_subject_ids": subjects, "label_map": LABEL_MAP,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Dataset metadata disagrees with arrays/protocol: {root}, {key}")
    if not np.isclose(metadata.get("sampling_rate_hz", 0), 128.0, rtol=1e-6, atol=0):
        raise ValueError(f"Expected 128 Hz metadata: {root}")
    if not isinstance(metadata.get("input_root"), str) or not metadata["input_root"]:
        raise ValueError("Dataset metadata is missing its upstream input_root")
    dataset = Dataset(root=root, metadata=metadata,
                      sha256={name: sha256_file(root / name) for name in DATA_FILES}, **arrays)
    validate_record_provenance(dataset)
    return dataset


def validate_record_provenance(dataset: Dataset) -> None:
    mapping = dataset.metadata.get("record_id_map", {})
    sources = dataset.metadata.get("record_sources", {})
    if not isinstance(mapping, dict) or not isinstance(sources, dict):
        raise ValueError("Dataset is missing record provenance")
    record_ids = np.unique(dataset.record_ids).astype(int).tolist()
    if set(mapping) != {str(value) for value in record_ids} or len(set(mapping.values())) != len(mapping):
        raise ValueError("record_id_map does not identify every record exactly once")
    if set(sources) != set(mapping.values()) or dataset.metadata.get("n_records") != len(record_ids):
        raise ValueError("record_sources/count disagree with record_id_map")
    for record in record_ids:
        name = mapping[str(record)]
        match = re.fullmatch(r"([1-9][0-9]*)_e_([123])", name)
        if match is None:
            raise ValueError(f"Invalid source record name: {name}")
        subject, label = int(match[1]), int(match[2]) - 1
        indices = np.flatnonzero(dataset.record_ids == record)
        if (not np.all(dataset.subject_ids[indices] == subject)
                or not np.all(dataset.y[indices] == label)
                or not np.array_equal(dataset.window_indices[indices], np.arange(len(indices)))):
            raise ValueError(f"Record/window identity disagrees with metadata: {name}")
        source = sources[name]
        if (not isinstance(source, dict) or source.get("relative_path") != f"{subject}/{name}.xlsx"
                or re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) is None
                or source.get("n_windows") != len(indices)
                or not isinstance(source.get("n_rows"), int) or source["n_rows"] < 1):
            raise ValueError(f"Invalid source provenance: {name}")


def model_run_dir(run_root: Path, model_name: str, seed: int) -> Path:
    return (run_root / "mdtf" if model_name == "mdtf" else run_root / "external" / model_name) / f"seed_{seed}"


def validate_run(args: argparse.Namespace, formal: Dataset, case: Dataset) -> tuple[dict, list[str], dict]:
    root = args.run_root.expanduser().resolve()
    plan = read_json(root / "run_plan.json")
    included = formal.metadata["included_subject_ids"]
    excluded = plan.get("excluded_subject_ids")
    if (len(included) < 2 or plan.get("included_subject_ids") != included
            or not isinstance(excluded, list) or args.case_subject_id not in excluded
            or set(included) & set(excluded)):
        raise ValueError("Formal cohort/failurecase exclusion disagree with run_plan")
    if sorted(formal.metadata.get("excluded_subject_ids", [])) != sorted(excluded):
        raise ValueError("Formal exclusion metadata disagree with run_plan")
    if (case.metadata["included_subject_ids"] != [args.case_subject_id]
            or args.case_subject_id in case.metadata.get("excluded_subject_ids", [])):
        raise ValueError(f"Failurecase dataset must contain only S{args.case_subject_id}")
    if formal.metadata["input_root"] != case.metadata["input_root"]:
        raise ValueError("Formal and failurecase datasets must record the same upstream input_root")
    if sorted(plan.get("test_subject_ids", [])) != included or len(plan.get("test_subject_ids", [])) != len(included):
        raise ValueError("Failurecase evaluation requires every formal LOSO fold; partial test jobs are rejected")
    if ((args.seed is not None and plan.get("seed") != args.seed)
            or not isinstance(plan.get("seed"), int) or plan["seed"] < 0
            or plan.get("dataset_shape") != list(formal.X.shape)):
        raise ValueError("Seed/dataset shape disagree with run_plan")
    if plan.get("dataset_sha256") != formal.sha256:
        raise ValueError("Formal dataset content hash mismatch with run_plan")
    planned_models = plan.get("models", [])
    if (not planned_models or len(set(planned_models)) != len(planned_models)
            or set(planned_models) - set(MODELS)):
        raise ValueError("Invalid model set in run_plan")
    models = planned_models if args.models is None else [name.strip().lower() for name in args.models.split(",")]
    if not models or len(set(models)) != len(models) or set(models) - set(planned_models):
        raise ValueError("Requested models must be distinct members of this job's run_plan")
    if (plan.get("protocol") != "exploratory_test_guided_loso_v1"
            or plan.get("failurecase_used_for_selection") is not False
            or plan.get("class_weights") != [3.0, 3.0, 1.0]
            or plan.get("adam_betas") != [0.5, 0.999]
            or plan.get("planned_training_folds") != len(included) * len(planned_models)
            or plan.get("mdtf_config") != MDTF_CONFIG):
        raise ValueError("Run plan does not match the frozen historical protocol")
    hashes = plan.get("code_sha256", {})
    if set(hashes) != TRAINING_CODE_FILES:
        raise ValueError("run_plan is missing its training code hashes")
    for name, expected in hashes.items():
        if sha256_file(SCRIPT_ROOT / name) != expected:
            raise ValueError(f"Training code hash mismatch: {name}; restore the code used by this run")
    fold_records = {}
    for model_name in models:
        model_dir = model_run_dir(root, model_name, plan["seed"])
        fold_dirs = [path for path in model_dir.glob("fold_subject_*") if path.is_dir()]
        names = {path.name for path in fold_dirs}
        if names != {f"fold_subject_{subject}" for subject in included}:
            raise ValueError(f"Incomplete or unexpected model fold set: {model_dir}")
        fold_records[model_name] = {}
        for subject in included:
            fold_dir = model_dir / f"fold_subject_{subject}"
            for name in ("best_model.pt", "metrics.json", "test_predictions.npz"):
                if not (fold_dir / name).is_file():
                    raise FileNotFoundError(f"Incomplete fold artifact: {fold_dir / name}")
            metrics = read_json(fold_dir / "metrics.json")
            validate_metrics(metrics, plan, model_name, subject, formal)
            fold_records[model_name][subject] = {"directory": fold_dir, "metrics": metrics}
    return plan, models, fold_records


def validate_metrics(metrics: dict, plan: dict, model_name: str, subject: int, formal: Dataset) -> None:
    expected = {
        "test_subject_id": subject, "seed": plan["seed"], "epochs": plan["epochs"],
        "batch_size": plan["batch_size"], "lr": plan["lr"], "class_weights": plan["class_weights"],
        "optimizer": "Adam", "optimizer_betas": plan["adam_betas"],
        "n_channels": int(formal.X.shape[1]), "n_times": int(formal.X.shape[2]), "n_classes": 3,
        "n_train_samples": int(np.sum(formal.subject_ids != subject)),
        "n_test_samples": int(np.sum(formal.subject_ids == subject)),
        "classification_mode": "flat", "train_sampling": "original",
        "architecture": DEFAULT_ARCHITECTURE if model_name == "mdtf" else model_name,
        "input_domain": "time_fft" if model_name == "mdtf" else "time",
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"Fold metrics mismatch: model={model_name}, subject={subject}, field={key}")
    epoch = metrics.get("best_epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= plan["epochs"]:
        raise ValueError("Fold metrics have an invalid best_epoch")
    if not np.isfinite(metrics.get("best_test_acc", np.nan)):
        raise ValueError("Fold metrics are missing a finite best_test_acc")


def fit_legacy_normalizers(train_windows: np.ndarray, dual: bool) -> dict:
    """Match core.standardize_by_train exactly; do not substitute chunk sums or float64 reductions."""
    time = np.expand_dims(np.asarray(train_windows, dtype=np.float32), axis=1)
    if time.ndim != 4 or len(time) == 0 or not np.isfinite(time).all():
        raise ValueError("Training windows must be finite, nonempty [N,C,T]")
    stats = {}
    for name in (("time", "fft") if dual else ("time",)):
        values = time if name == "time" else transform_windows_to_fft(time)
        mean, std = float(values.mean()), float(values.std())
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid training-only {name} normalization statistics")
        stats[name] = {"mean": mean, "std": std, "ddof": 0, "reduction_dtype": "float32",
                       "scope": "all elements of this fold's training subjects only"}
    return stats


def apply_legacy_normalizers(windows: np.ndarray, stats: dict, dual: bool):
    time = np.expand_dims(np.asarray(windows, dtype=np.float32), axis=1)
    if time.ndim != 4 or len(time) == 0 or not np.isfinite(time).all():
        raise ValueError("Prediction windows must be finite, nonempty [N,C,T]")
    domains = (time, transform_windows_to_fft(time)) if dual else (time,)
    result = tuple((values - stats[name]["mean"]) / stats[name]["std"]
                   for name, values in zip(("time", "fft"), domains))
    return result if dual else result[0]


def load_best_checkpoint(path: Path) -> dict:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # The recorded Linux environment uses PyTorch 1.12.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or not isinstance(value.get("state_dict"), dict):
        raise ValueError(f"Expected a best_model.pt dictionary: {path}")
    return value


def restore_model(checkpoint: dict, metrics: dict, model_name: str) -> torch.nn.Module:
    """Restore metadata and state strictly; legacy MDTF lacked explicit flat/architecture fields."""
    expected_architecture = DEFAULT_ARCHITECTURE if model_name == "mdtf" else model_name
    if (checkpoint.get("architecture", DEFAULT_ARCHITECTURE) != expected_architecture
            or checkpoint.get("classification_mode", "flat") != "flat"
            or checkpoint.get("train_sampling", "original") != "original"
            or checkpoint.get("test_subject_id") != metrics["test_subject_id"]
            or checkpoint.get("epoch") != metrics["best_epoch"] - 1):
        raise ValueError("Checkpoint architecture/fold/epoch disagree with metrics or frozen protocol")
    shape = {name: metrics[name] for name in ("n_channels", "n_times", "n_classes")}
    for key, value in shape.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint shape mismatch: {key}")
    expected_domain = "time_fft" if model_name == "mdtf" else "time"
    if checkpoint.get("input_domain") != expected_domain or metrics.get("input_domain") != expected_domain:
        raise ValueError("Checkpoint input domain mismatch")
    if model_name == "mdtf":
        settings = dict(MDTF_CONFIG)
        settings.pop("input_domain")
        settings.pop("transformer_branches")
        for key, value in settings.items():
            source_key = "transformer_branch_depths" if key == "transformer_depths" else key
            default = "flat" if key == "classification_mode" else None
            if checkpoint.get(source_key, default) != value or metrics.get(source_key, default) != value:
                raise ValueError(f"MDTF architecture mismatch: {source_key}")
        for key, value in {"emb_size": 40, "num_heads": 5, "dropout": 0.5,
                           "transformer_branches": 3, "transformer_weights_independent_by_domain": False,
                           "time_n_times": shape["n_times"], "fft_n_times": shape["n_times"] // 2 + 1}.items():
            if checkpoint.get(key) != value or metrics.get(key) != value:
                raise ValueError(f"MDTF checkpoint/metrics mismatch: {key}")
        model = DualBranchActivityConformer(
            n_channels=shape["n_channels"], time_n_times=shape["n_times"],
            fft_n_times=shape["n_times"] // 2 + 1, n_classes=shape["n_classes"],
            emb_size=40, num_heads=5, dropout=0.5, **settings)
    else:
        model = build_comparison_model(model_name, **shape)
        if (checkpoint.get("architecture_config") != model.architecture_config
                or metrics.get("architecture_config") != model.architecture_config):
            raise ValueError("External checkpoint architecture_config differs from the current frozen implementation")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def predict_probabilities(model: torch.nn.Module, inputs, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    domains = inputs if isinstance(inputs, tuple) else (inputs,)
    if not domains or len(domains[0]) == 0 or any(len(value) != len(domains[0]) for value in domains):
        raise ValueError("Prediction inputs must be nonempty and aligned")
    model.eval()
    rows, predictions = [], []
    with torch.no_grad():
        for start in range(0, len(domains[0]), batch_size):
            batch = [torch.from_numpy(np.ascontiguousarray(value[start:start + batch_size])).to(device)
                     for value in domains]
            _, logits = model(*batch)
            rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
            # Replay the training code's logit argmax, including near-tie behavior.
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    probabilities = np.concatenate(rows, axis=0)
    validate_probabilities(probabilities)
    return probabilities, np.concatenate(predictions)


def validate_probabilities(probabilities: np.ndarray) -> None:
    if (probabilities.ndim < 2 or probabilities.shape[-1] != 3
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)):
        raise ValueError("Expected finite normalized probabilities for three classes")


def verify_saved_predictions(predictions: np.ndarray, labels: np.ndarray, record: dict) -> dict:
    path = record["directory"] / "test_predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        true, predicted = saved["y_true"], saved["y_pred"]
    replay = predictions
    if not np.array_equal(true, labels) or not np.array_equal(predicted, replay):
        raise ValueError(f"Held-out prediction replay mismatch: {path}; check training environment, code, data, and normalization")
    accuracy = float(np.mean(replay == labels))
    matrix = confusion_matrix(labels, replay)
    metrics = record["metrics"]
    if not np.isclose(accuracy, metrics["best_test_acc"], rtol=0, atol=1e-12) or matrix != metrics.get("confusion_matrix"):
        raise ValueError(f"Replayed accuracy/confusion matrix disagree with metrics: {path}")
    return {"passed": True, "n_test_windows": len(labels), "accuracy": accuracy,
            "batch_size": metrics["batch_size"], "label_order_matches_saved_predictions": True}


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    return np.bincount(labels.astype(np.int64) * 3 + predictions.astype(np.int64), minlength=9).reshape(3, 3).tolist()


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    validate_probabilities(probabilities)
    if probabilities.shape != (len(labels), 3) or len(labels) == 0 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Labels/probabilities are not aligned three-class windows")
    matrix = np.asarray(confusion_matrix(labels, probabilities.argmax(axis=-1)))
    support, predicted, correct = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    recall = np.divide(correct, support, out=np.zeros(3, dtype=float), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(3, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(3, dtype=float), where=support + predicted > 0)
    return {"n_windows": len(labels), "accuracy": float(correct.sum() / matrix.sum()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "confusion_matrix": matrix.tolist(), "class_order": ["e_1", "e_2", "e_3"],
            "per_class_metrics": [{"class_id": index, "support": int(support[index]),
                                   "precision": float(precision[index]), "recall": float(recall[index]),
                                   "f1": float(f1[index])} for index in range(3)]}


def average_fold_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 3 or probabilities.shape[0] < 2 or probabilities.shape[1] == 0:
        raise ValueError("Expected [folds,windows,3] probabilities from a complete cohort")
    validate_probabilities(probabilities)
    # Preserve the window dimension, and give every fold exactly the same weight.
    result = probabilities.mean(axis=0, dtype=np.float64)
    validate_probabilities(result)
    return result


def evaluate(args: argparse.Namespace) -> dict:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite failurecase output: {output}")
    if (args.batch_size < 1 or args.case_subject_id < 1 or args.cpu_threads < 1
            or (args.seed is not None and args.seed < 0)):
        raise ValueError("batch-size, case-subject-id, cpu-threads must be positive; seed must be nonnegative")
    device = torch.device(validate_device(args.device))
    torch.set_num_threads(args.cpu_threads)
    formal, case = load_dataset(args.dataset_root), load_dataset(args.failurecase_root)
    plan, models, fold_records = validate_run(args, formal, case)
    subjects = plan["included_subject_ids"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": "excluded_subject_equal_fold_probability_ensemble_v1",
        "case_subject_id": args.case_subject_id, "seed": plan["seed"], "models": models,
        "formal_subject_ids": subjects, "excluded_subject_ids": plan["excluded_subject_ids"],
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
        "upstream_input_root": formal.metadata["input_root"],
        "record_sources": {"formal": formal.metadata["record_sources"], "failurecase": case.metadata["record_sources"]},
        "runtime": collect_environment_info(SCRIPT_ROOT.parent), "device": str(device),
        "case_batch_size": args.batch_size, "folds": {},
        "limitations": [
            "The excluded participant was seen during historical model development; this is a separate failurecase analysis, not a new untouched confirmatory sample.",
            "Upstream windows retain complete-record normalization; fitting fold statistics does not undo that record-level adaptation.",
            "Legacy saved test predictions contain labels but no row identifiers; replay alignment follows the hash-verified formal arrays and the original nonshuffled test order.",
            "Failurecase ensemble metrics are separate from the formal single-fold LOSO mean. Fold models overlap in their training participants.",
        ],
    }
    atomic_json_dump(manifest, output / "evaluation_manifest.json")
    # Same dtype conversion as core.load_global_dataset, before any fold selection.
    formal_X = formal.X.astype(np.float32, copy=False)
    case_X = case.X.astype(np.float32, copy=False)
    probabilities = {name: np.empty((len(subjects), len(case_X), 3), dtype=np.float32) for name in models}
    normalizers = {}
    for index, subject in enumerate(subjects):
        train_mask = formal.subject_ids != subject
        training_ids = [value for value in subjects if value != subject]
        stats = fit_legacy_normalizers(formal_X[train_mask], dual="mdtf" in models)
        normalizers[str(subject)] = {"train_subject_ids": training_ids, "statistics": stats}
        atomic_json_dump(normalizers, output / "normalizers.json")
        test_mask = formal.subject_ids == subject
        manifest["folds"][str(subject)] = {}
        for name in models:
            record = fold_records[name][subject]
            checkpoint_path = record["directory"] / "best_model.pt"
            checkpoint = load_best_checkpoint(checkpoint_path)
            model = restore_model(checkpoint, record["metrics"], name).to(device)
            dual = name == "mdtf"
            held_inputs = apply_legacy_normalizers(formal_X[test_mask], stats, dual)
            _, replay = predict_probabilities(model, held_inputs, device, record["metrics"]["batch_size"])
            verification = verify_saved_predictions(replay, formal.y[test_mask], record)
            del replay, held_inputs
            case_inputs = apply_legacy_normalizers(case_X, stats, dual)
            case_probabilities, _ = predict_probabilities(model, case_inputs, device, args.batch_size)
            probabilities[name][index] = case_probabilities
            fold_output = output / name / f"fold_subject_{subject}"
            atomic_save_npz(fold_output / "probabilities.npz", probabilities=case_probabilities,
                            sample_indices=np.arange(len(case_X)), subject_ids=case.subject_ids,
                            record_ids=case.record_ids, window_indices=case.window_indices)
            manifest["folds"][str(subject)][name] = {
                "best_epoch": record["metrics"]["best_epoch"], "held_out_replay": verification,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                "metrics_sha256": sha256_file(record["directory"] / "metrics.json"),
                "test_predictions_sha256": sha256_file(record["directory"] / "test_predictions.npz"),
            }
            atomic_json_dump(manifest, output / "evaluation_manifest.json")
            print(f"S{args.case_subject_id}: model={name}, formal fold=S{subject}; original held-out replay passed", flush=True)
            del model, checkpoint, case_inputs, case_probabilities
    # No failurecase metric is computed until all frozen fold probabilities exist.
    results = {}
    for name in models:
        fold_probs = probabilities[name]
        ensemble = average_fold_probabilities(fold_probs)
        per_fold = []
        for index, subject in enumerate(subjects):
            metrics = {"formal_test_subject_id": subject, **classification_metrics(case.y, fold_probs[index])}
            atomic_json_dump(metrics, output / name / f"fold_subject_{subject}" / "metrics.json")
            per_fold.append(metrics)
        results[name] = {"n_fold_models": len(subjects), "ensemble": classification_metrics(case.y, ensemble),
                         "per_fold": per_fold}
        atomic_save_npz(output / name / "ensemble_predictions.npz", fold_subject_ids=np.array(subjects),
                        fold_probabilities=fold_probs, probabilities=ensemble,
                        y_true=case.y, y_pred=ensemble.argmax(axis=-1), sample_indices=np.arange(len(case_X)),
                        subject_ids=case.subject_ids, record_ids=case.record_ids, window_indices=case.window_indices)
        atomic_json_dump(results[name], output / name / "summary.json")
    summary = {"protocol": manifest["protocol"], "case_subject_id": args.case_subject_id,
               "formal_subject_ids": subjects, "excluded_subject_ids": plan["excluded_subject_ids"],
               "seed": plan["seed"], "case_used_for_training_or_selection": False, "models": results}
    atomic_json_dump(summary, output / "failurecase_summary.json")
    atomic_json_dump({"status": "completed", "case_subject_id": args.case_subject_id,
                      "models": models, "n_verified_checkpoints": len(models) * len(subjects)}, output / "completed.json")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Formal filtered window_15_stride_3 dataset")
    parser.add_argument("--failurecase-root", type=Path, required=True, help="Dataset containing only the excluded participant")
    parser.add_argument("--run-root", type=Path, required=True, help="Five-model runner job directory containing run_plan.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--case-subject-id", type=int, default=6)
    parser.add_argument("--models", help="Optional comma-separated subset of this job's models; default is run_plan models")
    parser.add_argument("--seed", type=int, help="Optional seed assertion; default is the frozen run_plan seed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=72, help="Failurecase inference batch; original test replay uses its saved training batch size")
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
