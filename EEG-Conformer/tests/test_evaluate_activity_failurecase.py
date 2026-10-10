"""Synthetic protocol/normalization checks; never infer on real participant data."""
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
import evaluate_activity_failurecase as evaluation
from train_activity_loso import DEFAULT_ARCHITECTURE, forward_model_batch, prepare_split_inputs_for_input_domain


class ConstantClassifier(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, time, *fft):
        return time.flatten(1), self.logits.expand(len(time), -1)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_dataset(root: Path, subjects, excluded=(), source_root="/synthetic/normalized-records"):
    root.mkdir(parents=True)
    ids = np.repeat(np.array(subjects, dtype=np.int64), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), len(subjects))
    X = np.random.default_rng(9).normal(size=(len(ids), 21, 1920)).astype(np.float64)
    np.save(root / "X.npy", X)
    np.save(root / "y.npy", labels)
    np.save(root / "subject_ids.npy", ids)
    np.save(root / "record_ids.npy", np.arange(len(ids), dtype=np.int64))
    np.save(root / "window_indices.npy", np.zeros(len(ids), dtype=np.int64))
    names = [f"{subject}_e_{label + 1}" for subject, label in zip(ids, labels)]
    metadata = {
        "window_seconds": 15, "stride_seconds": 3, "sampling_rate_hz": 128,
        "n_channels": 21, "n_times": 1920, "n_samples": len(ids), "n_subjects": len(subjects),
        "n_records": len(ids), "included_subject_ids": list(subjects), "excluded_subject_ids": list(excluded),
        "input_root": source_root, "label_map": evaluation.LABEL_MAP,
        "record_id_map": {str(index): name for index, name in enumerate(names)},
        "record_sources": {name: {"relative_path": f"{subject}/{name}.xlsx", "sha256": "a" * 64,
                                  "n_rows": 1920, "n_windows": 1}
                           for subject, name in zip(ids, names)},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


def write_run(root: Path, formal_root: Path, *, models=("mdtf", "eegnet"), excluded=(6,)):
    root.mkdir()
    formal = evaluation.load_dataset(formal_root)
    subjects = formal.metadata["included_subject_ids"]
    plan = {
        "protocol": "exploratory_test_guided_loso_v1", "failurecase_used_for_selection": False,
        "included_subject_ids": subjects, "excluded_subject_ids": list(excluded),
        "test_subject_ids": subjects, "models": list(models), "seed": 42,
        "epochs": 200, "batch_size": 72, "lr": 0.0002,
        "class_weights": [3.0, 3.0, 1.0], "adam_betas": [0.5, 0.999],
        "planned_training_folds": len(subjects) * len(models), "mdtf_config": evaluation.MDTF_CONFIG,
        "dataset_shape": list(formal.X.shape), "dataset_sha256": formal.sha256,
        "code_sha256": {name: evaluation.sha256_file(ROOT / name) for name in evaluation.TRAINING_CODE_FILES},
    }
    (root / "run_plan.json").write_text(json.dumps(plan))
    for name in models:
        for index, subject in enumerate(subjects):
            folder = evaluation.model_run_dir(root, name, 42) / f"fold_subject_{subject}"
            folder.mkdir(parents=True)
            logits = [2.0, 0.0, -1.0] if index == 0 else [-1.0, 2.0, 0.0]
            tiny = ConstantClassifier(logits)
            prediction = np.full(3, index % 2, dtype=np.int64)
            labels = np.arange(3, dtype=np.int64)
            metrics = {
                "test_subject_id": subject, "seed": 42, "epochs": 200, "batch_size": 72,
                "lr": 0.0002, "class_weights": [3.0, 3.0, 1.0], "optimizer": "Adam",
                "optimizer_betas": [0.5, 0.999], "n_channels": 21, "n_times": 1920, "n_classes": 3,
                "n_train_samples": (len(subjects) - 1) * 3, "n_test_samples": 3,
                "classification_mode": "flat", "train_sampling": "original",
                "architecture": DEFAULT_ARCHITECTURE if name == "mdtf" else name,
                "input_domain": "time_fft" if name == "mdtf" else "time",
                "best_epoch": 1, "best_test_acc": 1 / 3,
                "confusion_matrix": evaluation.confusion_matrix(labels, prediction),
            }
            (folder / "metrics.json").write_text(json.dumps(metrics))
            np.savez(folder / "test_predictions.npz", y_true=labels, y_pred=prediction)
            torch.save({"state_dict": tiny.state_dict(), "epoch": 0, "test_subject_id": subject}, folder / "best_model.pt")
    return plan


def args_for(tmp_path, *, case_id=6, models=("mdtf", "eegnet"), excluded=(6,)):
    formal = write_dataset(tmp_path / "formal", (1, 2), excluded)
    case = write_dataset(tmp_path / "case", (case_id,))
    run = tmp_path / "run"
    write_run(run, formal, models=models, excluded=excluded)
    return evaluation.parse_args([
        "--dataset-root", str(formal), "--failurecase-root", str(case), "--run-root", str(run),
        "--output-dir", str(tmp_path / "evaluation"), "--case-subject-id", str(case_id),
        "--device", "cpu", "--batch-size", "2",
    ])


def fake_restore(checkpoint, metrics, name):
    model = ConstantClassifier([0, 0, 0])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def test_normalization_is_bitwise_equal_to_legacy_time_fft_without_case_adaptation():
    generator = np.random.default_rng(81)
    train = generator.normal(size=(23, 2, 64)).astype(np.float64)
    test = generator.normal(loc=20, scale=3, size=(5, 2, 64)).astype(np.float64)
    stats = evaluation.fit_legacy_normalizers(train, dual=True)
    expected_train, expected_test = prepare_split_inputs_for_input_domain(
        train.astype(np.float32)[:, None], test.astype(np.float32)[:, None], "time_fft")
    actual_train = evaluation.apply_legacy_normalizers(train, stats, dual=True)
    actual_test = evaluation.apply_legacy_normalizers(test, stats, dual=True)
    for actual, expected in zip(actual_train + actual_test, expected_train + expected_test):
        np.testing.assert_array_equal(actual, expected)
    assert stats["time"]["mean"] == float(train.astype(np.float32).mean())
    assert stats["time"]["std"] == float(train.astype(np.float32).std())
    unchanged = evaluation.fit_legacy_normalizers(train, dual=True)
    evaluation.apply_legacy_normalizers(test * 1000, stats, dual=True)
    assert stats == unchanged
    with pytest.raises(ValueError, match="normalization"):
        evaluation.fit_legacy_normalizers(np.ones((2, 2, 64)), dual=False)


def test_fold_average_preserves_each_window_and_uses_probabilities_instead_of_votes():
    probs = np.array([
        [[.51, .49, 0], [.01, .01, .98]],
        [[.51, .49, 0], [.01, .01, .98]],
        [[.01, .99, 0], [.01, .01, .98]],
    ], dtype=np.float32)
    averaged = evaluation.average_fold_probabilities(probs)
    assert averaged.shape == (2, 3)
    np.testing.assert_array_equal(averaged.argmax(axis=-1), [1, 2])
    np.testing.assert_allclose(averaged, probs.mean(axis=0, dtype=np.float64), rtol=0, atol=0)
    report = evaluation.classification_metrics(np.array([1, 2]), averaged)
    assert report["accuracy"] == 1
    assert report["confusion_matrix"] == [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="probabilities"):
        evaluation.average_fold_probabilities(probs[0])


def test_inference_replays_raw_logit_argmax_even_if_softmax_rounds_a_tie():
    model = ConstantClassifier([0, 1e-8, 0])
    probabilities, predictions = evaluation.predict_probabilities(
        model, np.ones((2, 1, 2, 16), dtype=np.float32), torch.device("cpu"), 1)
    np.testing.assert_array_equal(predictions, [1, 1])
    assert probabilities.shape == (2, 3)
    assert model.training is False


def test_complete_subset_job_freezes_all_probabilities_before_case_metrics_and_reuses_stats(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setattr(evaluation, "restore_model", fake_restore)
    monkeypatch.setattr(evaluation, "collect_environment_info", lambda root: {})
    original_fit, original_metrics = evaluation.fit_legacy_normalizers, evaluation.classification_metrics
    fitted_subjects, predicted_calls = [], []

    def track_stats(values, dual):
        fitted_subjects.append((len(values), dual))
        return original_fit(values, dual)

    original_predict = evaluation.predict_probabilities

    def track_predictions(model, inputs, device, batch_size):
        predicted_calls.append(batch_size)
        return original_predict(model, inputs, device, batch_size)

    def after_probabilities_only(labels, probabilities):
        assert len(predicted_calls) == 8  # held-out + case, for two models and two folds
        assert len(list(args.output_dir.glob("*/fold_subject_*/probabilities.npz"))) == 4
        return original_metrics(labels, probabilities)

    monkeypatch.setattr(evaluation, "fit_legacy_normalizers", track_stats)
    monkeypatch.setattr(evaluation, "predict_probabilities", track_predictions)
    monkeypatch.setattr(evaluation, "classification_metrics", after_probabilities_only)
    summary = evaluation.evaluate(args)
    assert fitted_subjects == [(3, True), (3, True)]
    assert predicted_calls == [72, 2, 72, 2, 72, 2, 72, 2]
    assert summary["case_used_for_training_or_selection"] is False
    assert summary["seed"] == 42 and args.seed is None
    assert set(summary["models"]) == {"mdtf", "eegnet"}
    stats = json.loads((args.output_dir / "normalizers.json").read_text())
    assert stats["1"]["train_subject_ids"] == [2] and stats["2"]["train_subject_ids"] == [1]
    for name in summary["models"]:
        with np.load(args.output_dir / name / "ensemble_predictions.npz", allow_pickle=False) as saved:
            assert saved["fold_probabilities"].shape == (2, 3, 3)
            assert saved["probabilities"].shape == (3, 3)
            np.testing.assert_array_equal(saved["y_pred"], [1, 1, 1])
            np.testing.assert_array_equal(saved["sample_indices"], [0, 1, 2])
        assert summary["models"][name]["ensemble"]["confusion_matrix"] == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    manifest = json.loads((args.output_dir / "evaluation_manifest.json").read_text())
    assert len(manifest["folds"]) == 2
    assert all(row["held_out_replay"]["passed"] for models in manifest["folds"].values() for row in models.values())
    assert json.loads((args.output_dir / "completed.json").read_text())["n_verified_checkpoints"] == 4
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation.evaluate(args)


def test_replay_mismatch_stops_before_case_prediction_and_does_not_mark_complete(tmp_path, monkeypatch):
    args = args_for(tmp_path, models=("eegnet",))
    folder = evaluation.model_run_dir(args.run_root, "eegnet", 42) / "fold_subject_1"
    np.savez(folder / "test_predictions.npz", y_true=np.arange(3), y_pred=np.ones(3, dtype=np.int64))
    monkeypatch.setattr(evaluation, "restore_model", fake_restore)
    monkeypatch.setattr(evaluation, "collect_environment_info", lambda root: {})
    calls = []
    original = evaluation.predict_probabilities

    def track(*parameters):
        calls.append(True)
        return original(*parameters)

    monkeypatch.setattr(evaluation, "predict_probabilities", track)
    with pytest.raises(ValueError, match="replay mismatch"):
        evaluation.evaluate(args)
    assert len(calls) == 1
    assert not (args.output_dir / "completed.json").exists()
    assert not (args.output_dir / "failurecase_summary.json").exists()
    assert not list(args.output_dir.glob("*/fold_subject_*/probabilities.npz"))


@pytest.mark.parametrize("problem", ["partial", "missing_fold", "data_hash", "code_hash", "wrong_case", "upstream"])
def test_invalid_protocol_or_inputs_are_rejected_before_output(tmp_path, problem):
    args = args_for(tmp_path, models=("eegnet",))
    plan_path = args.run_root / "run_plan.json"
    plan = json.loads(plan_path.read_text())
    if problem == "partial":
        plan["test_subject_ids"] = [1]
        plan_path.write_text(json.dumps(plan))
    elif problem == "missing_fold":
        (evaluation.model_run_dir(args.run_root, "eegnet", 42) / "fold_subject_2" / "metrics.json").unlink()
    elif problem == "data_hash":
        X = np.load(args.dataset_root / "X.npy")
        X[0, 0, 0] += 1
        np.save(args.dataset_root / "X.npy", X)
    elif problem == "code_hash":
        plan["code_sha256"]["comparison_models.py"] = "0" * 64
        plan_path.write_text(json.dumps(plan))
    elif problem == "wrong_case":
        args.case_subject_id = 8
    else:
        path = args.failurecase_root / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["input_root"] = "/another/upstream"
        path.write_text(json.dumps(metadata))
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.evaluate(args)
    assert not args.output_dir.exists()


def test_second_excluded_participant_and_single_model_job_are_supported(tmp_path, monkeypatch):
    args = args_for(tmp_path, case_id=8, models=("eegnet",), excluded=(6, 8))
    monkeypatch.setattr(evaluation, "restore_model", fake_restore)
    monkeypatch.setattr(evaluation, "collect_environment_info", lambda root: {})
    summary = evaluation.evaluate(args)
    assert summary["case_subject_id"] == 8 and summary["excluded_subject_ids"] == [6, 8]
    assert set(summary["models"]) == {"eegnet"}


@pytest.mark.parametrize("name", ["eegnet", "shallowconvnet", "atcnet", "tcformer"])
def test_external_architecture_and_weights_restore_strictly_without_forward(name):
    model = evaluation.build_comparison_model(name, n_channels=2, n_times=512, n_classes=3)
    checkpoint = {"architecture": name, "architecture_config": model.architecture_config,
                  "n_channels": 2, "n_times": 512, "n_classes": 3, "input_domain": "time",
                  "classification_mode": "flat", "train_sampling": "original",
                  "test_subject_id": 1, "epoch": 2, "state_dict": model.state_dict()}
    metrics = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    metrics["best_epoch"] = 3
    restored = evaluation.restore_model(checkpoint, metrics, name)
    assert restored.training is False
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    bad = copy.deepcopy(checkpoint)
    bad["epoch"] = 3
    with pytest.raises(ValueError, match="epoch"):
        evaluation.restore_model(bad, metrics, name)
    bad = copy.deepcopy(checkpoint)
    bad["architecture_config"]["dropout"] = -1
    with pytest.raises(ValueError, match="architecture_config"):
        evaluation.restore_model(bad, metrics, name)


@pytest.mark.parametrize("name", ["mdtf", "eegnet", "shallowconvnet", "atcnet", "tcformer"])
def test_synthetic_forward_contract_matches_core_for_every_model(name):
    """Use real model classes, tiny synthetic signals, and no training."""
    if name == "mdtf":
        settings = copy.deepcopy(evaluation.MDTF_CONFIG)
        for key in ("input_domain", "transformer_branches"):
            settings.pop(key)
        model = evaluation.DualBranchActivityConformer(
            n_channels=2, time_n_times=256, fft_n_times=129, n_classes=3,
            emb_size=40, num_heads=5, dropout=.5, **settings)
        inputs = (np.random.default_rng(2).normal(size=(2, 1, 2, 256)).astype(np.float32),
                  np.random.default_rng(3).normal(size=(2, 1, 2, 129)).astype(np.float32))
    else:
        model = evaluation.build_comparison_model(name, n_channels=2, n_times=512, n_classes=3)
        inputs = np.random.default_rng(2).normal(size=(2, 1, 2, 512)).astype(np.float32)
    model.eval()
    domains = inputs if isinstance(inputs, tuple) else (inputs,)
    batch = tuple(torch.from_numpy(value) for value in domains) + (torch.zeros(2, dtype=torch.long),)
    with torch.no_grad():
        core_logits, _ = forward_model_batch(model, batch, torch.device("cpu"))
    probabilities, predictions = evaluation.predict_probabilities(model, inputs, torch.device("cpu"), 2)
    np.testing.assert_allclose(probabilities, core_logits.softmax(dim=-1).numpy(), rtol=0, atol=0)
    np.testing.assert_array_equal(predictions, core_logits.argmax(dim=-1).numpy())


def test_legacy_flat_mdtf_restores_small_shape_without_forward_and_rejects_hierarchical():
    settings = copy.deepcopy(evaluation.MDTF_CONFIG)
    for name in ("input_domain", "transformer_branches"):
        settings.pop(name)
    model = evaluation.DualBranchActivityConformer(
        n_channels=2, time_n_times=256, fft_n_times=129, n_classes=3,
        emb_size=40, num_heads=5, dropout=.5, **settings)
    metadata = copy.deepcopy(evaluation.MDTF_CONFIG)
    metadata["transformer_branch_depths"] = metadata.pop("transformer_depths")
    metadata.update(n_channels=2, n_times=256, n_classes=3, time_n_times=256, fft_n_times=129,
                    emb_size=40, num_heads=5, dropout=.5, transformer_weights_independent_by_domain=False,
                    test_subject_id=1, epoch=2)
    # Historical 72.13% artifacts did not record architecture/classification_mode.
    metadata.pop("classification_mode")
    checkpoint = dict(metadata, state_dict=model.state_dict())
    metrics = dict(metadata, best_epoch=3)
    restored = evaluation.restore_model(checkpoint, metrics, "mdtf")
    assert restored.classification_mode == "flat"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    bad = dict(checkpoint, classification_mode="hierarchical")
    with pytest.raises(ValueError, match="architecture"):
        evaluation.restore_model(bad, metrics, "mdtf")
