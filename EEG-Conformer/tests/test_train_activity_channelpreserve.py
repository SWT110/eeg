from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_channelpreserve_failurecase as evaluation
import train_activity_channelpreserve as runner
import train_activity_loso as core
from evaluate_activity_failurecase import (
    apply_legacy_normalizers, confusion_matrix, fit_legacy_normalizers,
    load_best_checkpoint, load_dataset, predict_probabilities, verify_saved_predictions,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_dataset(root, subjects=(1, 2), excluded=(6, 17), channels=21, length=1920):
    root.mkdir(parents=True)
    ids = np.repeat(np.array(subjects, dtype=np.int64), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), len(subjects))
    np.save(root / "X.npy", np.random.default_rng(9).normal(size=(len(ids), channels, length)).astype(np.float32))
    np.save(root / "y.npy", labels)
    np.save(root / "subject_ids.npy", ids)
    np.save(root / "record_ids.npy", np.arange(len(ids), dtype=np.int64))
    np.save(root / "window_indices.npy", np.zeros(len(ids), dtype=np.int64))
    names = [f"{sid}_e_{label + 1}" for sid, label in zip(ids, labels)]
    metadata = {
        "window_seconds": 15, "stride_seconds": 3, "sampling_rate_hz": 128,
        "n_channels": channels, "n_times": length, "n_samples": len(ids), "n_subjects": len(subjects),
        "n_records": len(ids), "included_subject_ids": list(subjects), "excluded_subject_ids": list(excluded),
        "input_root": "/synthetic/normalized-records", "label_map": {"e_1": 0, "e_2": 1, "e_3": 2},
        "record_id_map": {str(index): name for index, name in enumerate(names)},
        "record_sources": {name: {"relative_path": f"{sid}/{name}.xlsx", "sha256": "a" * 64,
                                  "n_rows": length, "n_windows": 1}
                           for sid, name in zip(ids, names)},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


def arguments(tmp_path, dataset, *extra, small=True):
    options = ["--cohort-subject-ids", "1,2,6,17"] if small else []
    return runner.parse_args(["--dataset-root", str(dataset), "--output-base", str(tmp_path / "out"),
                              "--device", "cpu", *options, *extra])


def test_production_defaults_cohort_and_separate_debug_outputs(tmp_path):
    dataset = write_dataset(tmp_path / "formal", evaluation.FORMAL_SUBJECTS)
    plan = runner.build_plan(arguments(tmp_path, dataset, small=False))
    assert plan["epochs"] == 200 and plan["batch_size"] == 72 and plan["lr"] == .0002
    assert plan["adam_betas"] == [.5, .999] and plan["class_weights"] is None
    assert plan["train_sampling"] == "original" and plan["sr_augmentation"] is False
    assert plan["model_kwargs"]["num_heads"] == 5 and plan["architecture_config"]["depth"] == 10
    assert plan["planned_training_folds"] == 15 and plan["excluded_subject_ids"] == [6, 17]
    assert not ({6, 17} & set(plan["included_subject_ids"]))
    partial = runner.build_plan(arguments(tmp_path, dataset, "--test-subject-ids", "1", small=False))
    smoke = runner.build_plan(arguments(tmp_path, dataset, "--smoke-test", small=False))
    assert len({plan["output_dir"], partial["output_dir"], smoke["output_dir"]}) == 3
    assert smoke["epochs"] == 1 and smoke["batch_size"] == 2 and smoke["test_subject_ids"] == [1]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("leaked", [6, 17])
def test_excluded_subjects_never_silently_remain_in_training_arrays(tmp_path, leaked):
    dataset = write_dataset(tmp_path / "leak", (1, 2, leaked))
    with pytest.raises(ValueError, match="Excluded subjects still present"):
        runner.build_plan(arguments(tmp_path, dataset))
    assert not (tmp_path / "out").exists()


def test_invalid_provenance_is_rejected_before_training(tmp_path):
    dataset = write_dataset(tmp_path / "formal")
    np.save(dataset / "window_indices.npy", np.ones(6, dtype=np.int64))
    with pytest.raises(ValueError, match="Record/window identity"):
        runner.build_plan(arguments(tmp_path, dataset))


def test_real_original_shape_smoke_training_replay_and_completed_fold_resume(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "formal")
    args = arguments(tmp_path, dataset, "--smoke-test")
    plan = runner.build_plan(args)
    monkeypatch.setattr(runner, "collect_environment_info", lambda *_: {})
    runner.run_plan(plan)
    root = Path(plan["output_dir"])
    fold = Path(plan["model_output_dir"]) / "fold_subject_1"
    metrics = runner.validate_completed_fold(fold, plan, 1)
    assert metrics["class_weights"] is None and metrics["transformer_weights_independent_by_domain"] is True
    checkpoint = load_best_checkpoint(fold / "best_model.pt")
    model = runner.restore_checkpoint(checkpoint, metrics, plan)
    formal = load_dataset(dataset)
    stats = fit_legacy_normalizers(formal.X[formal.subject_ids != 1], dual=True)
    inputs = apply_legacy_normalizers(formal.X[formal.subject_ids == 1], stats, dual=True)
    _, pred = predict_probabilities(model, inputs, torch.device("cpu"), 2)
    verify_saved_predictions(pred, formal.y[formal.subject_ids == 1], {"directory": fold, "metrics": metrics})
    assert json.loads((root / "completed.json").read_text())["smoke_test"] is True
    monkeypatch.setattr(runner, "train_loso_fold", lambda **_: pytest.fail("Completed fold was retrained"))
    runner.run_plan(runner.build_plan(args), resume=True)
    with pytest.raises(FileExistsError):
        runner.run_plan(plan)
    changed = copy.deepcopy(plan)
    changed["code_sha256"]["channel_preserving_model.py"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run_plan(changed, resume=True)
    bad = copy.deepcopy(checkpoint)
    bad["architecture_config"]["late_skip_targets"] = [10]
    with pytest.raises(ValueError, match="disagree"):
        runner.restore_checkpoint(bad, metrics, plan)
    np.savez(fold / "test_predictions.npz", y_true=np.arange(3), y_pred=np.array([99, 1, 2]))
    with pytest.raises(ValueError, match="Invalid saved predictions"):
        runner.run_plan(plan, resume=True)


def test_interrupted_training_restores_optimizer_rng_and_same_best_weights(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "tiny", channels=3, length=256)
    common = dict(dataset_root=dataset, test_subject_id=1, epochs=2, batch_size=2, lr=.0002,
                  device="cpu", seed=42, resume=True, **runner.MODEL_KWARGS)
    original = core.atomic_torch_save
    uninterrupted = core.train_loso_fold(output_dir=tmp_path / "uninterrupted", **common)
    stopped = False

    def interrupt_after_checkpoint(value, path):
        nonlocal stopped
        original(value, path)
        if Path(path).name == "last_checkpoint.pt" and not stopped:
            stopped = True
            raise RuntimeError("synthetic interruption after first epoch checkpoint")

    monkeypatch.setattr(core, "atomic_torch_save", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        core.train_loso_fold(output_dir=tmp_path / "resumed", **common)
    monkeypatch.setattr(core, "atomic_torch_save", original)
    resumed = core.train_loso_fold(output_dir=tmp_path / "resumed", **common)
    before, after = json.loads(uninterrupted.read_text()), json.loads(resumed.read_text())
    assert after["resumed_from_epoch"] == 1
    assert before["best_epoch"] == after["best_epoch"] and before["best_test_acc"] == after["best_test_acc"]
    a = load_best_checkpoint(uninterrupted.parent / "best_model.pt")["state_dict"]
    b = load_best_checkpoint(resumed.parent / "best_model.pt")["state_dict"]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


class ConstantClassifier(nn.Module):
    def __init__(self, winner=0):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([2. if i == winner else 0. for i in range(3)]))

    def forward(self, time, fft):
        return time.flatten(1), self.logits.expand(len(time), -1)


def fake_restore(checkpoint, *_):
    model = ConstantClassifier()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def full_evaluation_fixture(tmp_path, case_id):
    formal_root = write_dataset(tmp_path / "formal", evaluation.FORMAL_SUBJECTS)
    case_root = write_dataset(tmp_path / "case", (case_id,), excluded=())
    plan = runner.build_plan(arguments(tmp_path, formal_root, small=False))
    root = Path(plan["output_dir"])
    root.mkdir(parents=True)
    (root / "run_plan.json").write_text(json.dumps(plan))
    (root / "completed.json").write_text(json.dumps({"status": "completed", "architecture": "channelpreserving",
                                                    "smoke_test": False, "training_folds": 15}))
    for index, subject in enumerate(evaluation.FORMAL_SUBJECTS):
        fold = root / "model" / f"fold_subject_{subject}"
        fold.mkdir(parents=True)
        pred = np.full(3, index % 3, dtype=np.int64)
        metrics = {
            "architecture": "channelpreserving", "architecture_config": plan["architecture_config"],
            "test_subject_id": subject, "seed": 42, "epochs": 200, "batch_size": 72, "lr": .0002,
            "class_weights": None, "input_domain": "time_fft", "classification_mode": "flat",
            "train_sampling": "original", "optimizer": "Adam", "optimizer_betas": [.5, .999],
            "emb_size": 40, "num_heads": 5, "depth": 10, "dropout": .5, "transformer_encoder_dropout": .5,
            "transformer_branches": 1, "branch_loss_aux_weight": 0., "transformer_branch_qkv": "none",
            "n_channels": 21, "n_times": 1920, "n_classes": 3, "n_train_samples": 42, "n_test_samples": 3,
            "best_epoch": 1, "best_test_acc": 1 / 3, "confusion_matrix": confusion_matrix(np.arange(3), pred),
        }
        (fold / "metrics.json").write_text(json.dumps(metrics))
        history = {**metrics, "history": [{"epoch": n} for n in range(1, 201)]}
        (fold / "epoch_history.json").write_text(json.dumps(history))
        np.savez(fold / "test_predictions.npz", y_true=np.arange(3), y_pred=pred)
        torch.save({"state_dict": ConstantClassifier(index % 3).state_dict()}, fold / "best_model.pt")
    args = evaluation.parse_args([
        "--dataset-root", str(formal_root), "--failurecase-root", str(case_root),
        "--run-root", str(root), "--output-dir", str(tmp_path / "evaluation"),
        "--case-subject-id", str(case_id), "--device", "cpu", "--batch-size", "2",
    ])
    return args, plan


@pytest.mark.parametrize("case_id", [6, 17])
def test_both_cases_use_15_frozen_folds_and_average_only_fold_axis(tmp_path, monkeypatch, case_id):
    args, plan = full_evaluation_fixture(tmp_path, case_id)
    monkeypatch.setattr(runner, "restore_checkpoint", fake_restore)
    monkeypatch.setattr(evaluation, "restore_checkpoint", fake_restore)
    monkeypatch.setattr(evaluation, "collect_environment_info", lambda *_: {})
    summary = evaluation.evaluate(args)
    assert summary["n_fold_models"] == 15 and summary["case_used_for_training_or_selection"] is False
    assert summary["ensemble"]["n_windows"] == 3
    with np.load(args.output_dir / "ensemble_predictions.npz") as saved:
        assert saved["fold_probabilities"].shape == (15, 3, 3)
        np.testing.assert_array_equal(saved["probabilities"], saved["fold_probabilities"].mean(axis=0, dtype=np.float64))
    normalizers = json.loads((args.output_dir / "normalizers.json").read_text())
    for sid, row in normalizers.items():
        assert not ({int(sid), 6, 17} & set(row["train_subject_ids"]))
    with pytest.raises(FileExistsError):
        evaluation.evaluate(args)


@pytest.mark.parametrize("field,value", [
    ("smoke_test", True), ("test_subject_ids", [1]), ("class_weights", [3, 3, 1]),
    ("epochs", 1), ("included_subject_ids", [1, 2]), ("code_sha256", {}),
])
def test_failurecase_rejects_debug_old_profile_partial_and_changed_code(tmp_path, monkeypatch, field, value):
    args, plan = full_evaluation_fixture(tmp_path, 17)
    plan[field] = value
    (args.run_root / "run_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(runner, "restore_checkpoint", fake_restore)
    with pytest.raises(ValueError):
        evaluation.evaluate(args)
    assert not args.output_dir.exists()


def test_failurecase_stops_before_scoring_if_original_predictions_do_not_replay(tmp_path, monkeypatch):
    args, _ = full_evaluation_fixture(tmp_path, 17)
    monkeypatch.setattr(runner, "restore_checkpoint", fake_restore)
    monkeypatch.setattr(evaluation, "restore_checkpoint", lambda *_: ConstantClassifier(2))
    monkeypatch.setattr(evaluation, "collect_environment_info", lambda *_: {})
    with pytest.raises(ValueError, match="replay mismatch"):
        evaluation.evaluate(args)
    assert not (args.output_dir / "completed.json").exists()
    assert not (args.output_dir / "failurecase_summary.json").exists()
    assert not list(args.output_dir.glob("fold_subject_*/probabilities.npz"))
