from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_activity_loso as core
import train_activity_multidepth_weight_sweep as runner
import evaluate_multidepth_failurecase as evaluation
from evaluate_activity_failurecase import (
    apply_legacy_normalizers, fit_legacy_normalizers, load_best_checkpoint,
    load_dataset, predict_probabilities, verify_saved_predictions,
)
from test_train_activity_channelpreserve import ConstantClassifier, fake_restore, write_dataset


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def arguments(tmp_path, dataset, weights="3,2,1|3,3,1", *extra, small=True):
    cohort = ["--cohort-subject-ids", "1,2,6,17"] if small else []
    return runner.parse_args(["--dataset-root", str(dataset), "--output-root", str(tmp_path / "out"),
                             "--class-weight-sets", weights, "--device", "cpu", "--cpu-threads", "1",
                             *cohort, *extra])


@pytest.mark.parametrize("raw", ["", "1,2", "0,2,1", "-1,2,1", "nan,2,1", "inf,1,1",
                                 "1,2,3|", "3,2,1|3.0,2,1", "1e100,1,1", "1e-100,1,1"])
def test_invalid_weight_sets_rejected(raw):
    with pytest.raises(ValueError):
        runner.parse_weight_sets(raw)


def test_user_weight_values_order_and_unambiguous_directory_keys():
    values = runner.parse_weight_sets(" 3,2,1 | 1.5,4,2 | 3.00000000001,2,1")
    assert values == [[3., 2., 1.], [1.5, 4., 2.], [3.00000000001, 2., 1.]]
    assert runner.weight_key(values[0]) == "weights_3_2_1"
    assert runner.weight_key(values[1]) == "weights_1p5_4_2"
    assert runner.weight_key(values[0]) != runner.weight_key(values[2])


def test_defaults_complete_cohort_manual_weights_and_appendable_per_weight_identity(tmp_path):
    subjects = [s for s in range(1, 18) if s not in (6, 17)]
    dataset = write_dataset(tmp_path / "formal", subjects)
    args = arguments(tmp_path, dataset, small=False)
    batch = runner.build_batch_plan(args)
    assert batch["candidate_count"] == 2 and batch["planned_training_folds"] == 30
    first = batch["runs"][0]
    assert first["architecture_config"]["depths"] == [4, 10, 16]
    assert first["architecture_config"]["split_after_blocks"] == [2, 5, 8]
    assert first["epochs"] == 200 and first["batch_size"] == 72 and first["lr"] == .0002
    assert first["architecture_config"]["num_heads"] == 5
    assert first["architecture_config"]["transformer_branch_qkv_dropout"] == .1
    assert first["architecture_config"]["branch_loss_aux_weight"] == .2
    assert first["train_sampling"] == "original" and first["sr_augmentation"] is False
    assert first["excluded_subject_ids"] == [6, 17] and first["test_subject_ids"] == subjects
    assert first["class_weights"] == [3., 2., 1.]
    assert "manual" in first["weight_selection_rule"]
    more = runner.build_batch_plan(arguments(tmp_path, dataset, "3,2,1|5,4,1", small=False))
    assert more["runs"][0] == first
    assert not (tmp_path / "out").exists()


def test_two_gpu_batches_share_parent_but_never_candidate_directories(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "formal")
    a = runner.build_batch_plan(arguments(tmp_path, dataset, "3,2,1|4,3,1", "--device", "cuda:0"))
    b = runner.build_batch_plan(arguments(tmp_path, dataset, "3,3,1|5,3,1", "--device", "cuda:1", "--cpu-threads", "8"))
    assert a["output_root"] == b["output_root"]
    assert not ({p["output_dir"] for p in a["runs"]} & {p["output_dir"] for p in b["runs"]})
    assert runner.comparison_signature(a["runs"][0]) == runner.comparison_signature(b["runs"][0])
    seen = []
    monkeypatch.setattr(runner, "validate_device", lambda d: d)
    monkeypatch.setattr(runner, "run_weight", lambda p, **_: seen.append((p["class_weights"], p["device"])))
    runner.run_batch(a)
    runner.run_batch(b)
    assert seen == [([3., 2., 1.], "cuda:0"), ([4., 3., 1.], "cuda:0"),
                    ([3., 3., 1.], "cuda:1"), ([5., 3., 1.], "cuda:1")]
    summary = runner.summarize_weight_runs(Path(a["output_root"]))
    assert summary["candidate_count"] == 4 and summary["completed_candidates"] == 0
    assert all(g["status"] == "pending" for g in summary["groups"])


@pytest.mark.parametrize("leaked", [6, 17])
def test_excluded_people_cannot_leak_into_any_weight_run(tmp_path, leaked):
    dataset = write_dataset(tmp_path / "formal", (1, 2, leaked))
    with pytest.raises(ValueError, match="Excluded subjects still present"):
        runner.build_batch_plan(arguments(tmp_path, dataset))


def test_real_two_candidate_training_prediction_replay_resume_and_corruption(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "formal")
    args = arguments(tmp_path, dataset, "3,2,1|3,3,1", "--smoke-test")
    batch = runner.build_batch_plan(args)
    monkeypatch.setattr(runner, "collect_environment_info", lambda *_: {})
    events = []
    original = runner.train_loso_fold

    def observed(**kwargs):
        events.append((kwargs["class_weights"], kwargs["test_subject_id"]))
        return original(**kwargs)

    monkeypatch.setattr(runner, "train_loso_fold", observed)
    runner.run_batch(batch)
    assert events == [([3., 2., 1.], 1), ([3., 3., 1.], 1)]
    root = Path(batch["output_root"])
    assert "smoke" in root.parts
    summary = runner.summarize_weight_runs(root)
    assert summary["completed_candidates"] == 2
    assert all(g["smoke_test"] and g["completed_folds"] == 1 for g in summary["groups"])
    assert "winner" not in summary and "best_weights" not in summary
    plan = batch["runs"][0]
    fold = Path(plan["model_output_dir"]) / "fold_subject_1"
    metrics = runner.validate_completed_fold(fold, plan, 1)
    checkpoint = load_best_checkpoint(fold / "best_model.pt")
    assert checkpoint["class_weights"] == [3., 2., 1.]
    assert metrics["transformer_weights_independent_by_domain"] is False
    assert len(metrics["transformer_branch_loss_weights"]) == 3
    model = runner.restore_checkpoint(checkpoint, metrics, plan)
    formal = load_dataset(dataset)
    stats = fit_legacy_normalizers(formal.X[formal.subject_ids != 1], dual=True)
    inputs = apply_legacy_normalizers(formal.X[formal.subject_ids == 1], stats, dual=True)
    _, pred = predict_probabilities(model, inputs, torch.device("cpu"), 2)
    verify_saved_predictions(pred, formal.y[formal.subject_ids == 1], {"directory": fold, "metrics": metrics})
    monkeypatch.setattr(runner, "train_loso_fold", lambda **_: pytest.fail("Completed fold retrained"))
    reordered = runner.build_batch_plan(arguments(tmp_path, dataset, "3,3,1|3,2,1", "--smoke-test"))
    runner.run_batch(reordered, resume=True)
    with pytest.raises(FileExistsError):
        runner.run_batch(batch)
    changed = copy.deepcopy(batch)
    changed["runs"][0]["code_sha256"]["channel_preserving_multidepth_model.py"] = "0" * 64
    with pytest.raises(ValueError, match="incompatible"):
        runner.run_batch(changed, resume=True)
    bad = copy.deepcopy(checkpoint)
    bad["class_weights"] = [1., 1., 1.]
    with pytest.raises(ValueError, match="class_weights"):
        runner.restore_checkpoint(bad, metrics, plan)
    np.savez(fold / "test_predictions.npz", y_true=np.arange(3), y_pred=np.array([99, 1, 2]))
    with pytest.raises(ValueError, match="Invalid saved predictions"):
        runner.summarize_weight_runs(root)


def test_interrupted_multidepth_training_restores_optimizer_rng_and_all_weights(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "tiny", channels=3, length=256)
    common = dict(dataset_root=dataset, test_subject_id=1, epochs=2, batch_size=2, lr=.0002,
                  device="cpu", seed=42, resume=True, class_weights=[3., 2., 1.], **runner.MODEL_KWARGS)
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
    a, b = json.loads(uninterrupted.read_text()), json.loads(resumed.read_text())
    assert b["resumed_from_epoch"] == 1
    assert a["best_epoch"] == b["best_epoch"] and a["best_test_acc"] == b["best_test_acc"]
    before = load_best_checkpoint(uninterrupted.parent / "best_model.pt")["state_dict"]
    after = load_best_checkpoint(resumed.parent / "best_model.pt")["state_dict"]
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def completed_archive_fold(plan, subject, matrix):
    root = Path(plan["model_output_dir"]) / f"fold_subject_{subject}"
    root.mkdir(parents=True)
    metrics = {**plan["model_kwargs"], "architecture_config": plan["architecture_config"],
        "transformer_branch_depths": plan["model_kwargs"]["transformer_depths"],
        "test_subject_id": subject, "seed": plan["seed"], "epochs": plan["epochs"],
        "batch_size": plan["batch_size"], "lr": plan["lr"], "optimizer": "Adam", "optimizer_betas": [.5, .999],
        "n_channels": 21, "n_times": 1920, "n_classes": 3, "trainable_parameters": plan["trainable_parameters"],
        "n_train_samples": plan["dataset_shape"][0] - plan["subject_window_counts"][str(subject)],
        "n_test_samples": plan["subject_window_counts"][str(subject)], "best_epoch": 1,
        "best_test_acc": runner.metrics_from_matrix(matrix)["accuracy"], "confusion_matrix": matrix,
        "transformer_weights_independent_by_domain": False, "transformer_branch_loss_weights": [1/3]*3,
        "loss": "LearnableSoftmaxWeightedBranchCrossEntropyLoss"}
    pairs = [(i, j) for i in range(3) for j in range(3) for _ in range(matrix[i][j])]
    np.savez(root / "test_predictions.npz", y_true=np.array([x[0] for x in pairs]), y_pred=np.array([x[1] for x in pairs]))
    (root / "metrics.json").write_text(json.dumps(metrics))
    (root / "epoch_history.json").write_text(json.dumps({**metrics, "history": [{"epoch": e} for e in range(1, plan["epochs"] + 1)]}))


def test_equal_subject_metrics_mean_of_minima_and_partial_archive_summary(tmp_path):
    dataset = write_dataset(tmp_path / "formal")
    batch = runner.build_batch_plan(arguments(tmp_path, dataset))
    a, b = batch["runs"]
    for plan in (a, b):
        root = Path(plan["output_dir"])
        root.mkdir(parents=True)
        (root / "run_plan.json").write_text(json.dumps(plan))
    completed_archive_fold(a, 1, [[1, 0, 0], [0, 0, 1], [0, 0, 1]])
    completed_archive_fold(a, 2, [[0, 0, 1], [0, 1, 0], [0, 0, 1]])
    completed_archive_fold(b, 1, [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    (Path(a["output_dir"]) / "completed.json").write_text(json.dumps(runner.completed_record(a)))
    # Aggregation works from result archives without raw training arrays/models.
    (dataset / "X.npy").unlink()
    summary = runner.summarize_weight_runs(Path(batch["output_root"]))
    first, second = summary["groups"]
    assert first["mean_e1_recall"] == first["mean_e2_recall"] == .5
    assert first["mean_min_e1_e2_recall"] == 0
    assert first["status"] == "completed" and second["status"] == "incomplete"
    assert first["mean_macro_f1"] != first["pooled_metrics"]["macro_f1"]
    assert (Path(batch["output_root"]) / "subject_results.csv").is_file()


def test_same_candidate_lock_is_exclusive_and_releases_after_exception(tmp_path):
    path = tmp_path / ".training.lock"
    with runner.file_lock(path):
        with pytest.raises(RuntimeError, match="Another process"):
            with runner.file_lock(path):
                pytest.fail("Lock was not exclusive")
    with runner.file_lock(path):
        pass


def test_two_real_processes_can_train_disjoint_candidates_in_one_output_root(tmp_path):
    dataset = write_dataset(tmp_path / "formal")
    common = [sys.executable, str(Path(runner.__file__)), "--dataset-root", str(dataset),
              "--output-root", str(tmp_path / "out"), "--cohort-subject-ids", "1,2,6,17",
              "--device", "cpu", "--cpu-threads", "1", "--smoke-test", "--resume"]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    processes = [subprocess.Popen([*common, "--class-weight-sets", weights],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
                 for weights in ("3,2,1", "3,3,1")]
    try:
        for process in processes:
            output, _ = process.communicate(timeout=45)
            assert process.returncode == 0, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    batch = runner.build_batch_plan(arguments(tmp_path, dataset, "3,2,1|3,3,1", "--smoke-test"))
    summary = runner.summarize_weight_runs(Path(batch["output_root"]))
    assert summary["candidate_count"] == summary["completed_candidates"] == 2
    assert all(g["completed_folds"] == 1 for g in summary["groups"])


def full_case_fixture(tmp_path, case_id):
    subjects = [s for s in range(1, 18) if s not in (6, 17)]
    formal = write_dataset(tmp_path / "formal", subjects)
    case = write_dataset(tmp_path / "case", (case_id,), excluded=())
    plan = runner.build_batch_plan(arguments(tmp_path, formal, "3,2,1", small=False))["runs"][0]
    root = Path(plan["output_dir"])
    root.mkdir(parents=True)
    (root / "run_plan.json").write_text(json.dumps(plan))
    for index, subject in enumerate(subjects):
        matrix = [[int(j == index % 3) for j in range(3)] for _ in range(3)]
        completed_archive_fold(plan, subject, matrix)
        fold = Path(plan["model_output_dir"]) / f"fold_subject_{subject}"
        torch.save({"state_dict": ConstantClassifier(index % 3).state_dict()}, fold / "best_model.pt")
    (root / "completed.json").write_text(json.dumps(runner.completed_record(plan)))
    args = evaluation.engine.parse_args([
        "--dataset-root", str(formal), "--failurecase-root", str(case), "--run-root", str(root),
        "--output-dir", str(tmp_path / "evaluation"), "--case-subject-id", str(case_id),
        "--device", "cpu", "--batch-size", "2",
    ])
    return args, plan


@pytest.mark.parametrize("case_id", [6, 17])
def test_d_failurecases_use_one_frozen_weight_candidate_and_all_15_folds(tmp_path, monkeypatch, case_id):
    args, plan = full_case_fixture(tmp_path, case_id)
    monkeypatch.setattr(runner, "restore_checkpoint", fake_restore)
    monkeypatch.setattr(evaluation.engine, "collect_environment_info", lambda *_: {})
    result = evaluation.evaluate(args)
    assert result["training_class_weights"] == [3., 2., 1.]
    assert result["n_fold_models"] == 15 and result["case_used_for_training_or_selection"] is False
    with np.load(args.output_dir / "ensemble_predictions.npz") as saved:
        assert saved["fold_probabilities"].shape == (15, 3, 3)
        np.testing.assert_array_equal(saved["probabilities"], saved["fold_probabilities"].mean(axis=0, dtype=np.float64))
    normalizers = json.loads((args.output_dir / "normalizers.json").read_text())
    assert all(not ({int(sid), 6, 17} & set(row["train_subject_ids"])) for sid, row in normalizers.items())


@pytest.mark.parametrize("field,value", [("smoke_test", True), ("class_weights", [1., 1., 1.]),
                                         ("test_subject_ids", [1]), ("code_sha256", {})])
def test_d_failurecase_rejects_smoke_partial_changed_weights_or_code(tmp_path, monkeypatch, field, value):
    args, plan = full_case_fixture(tmp_path, 17)
    plan[field] = value
    (args.run_root / "run_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(runner, "restore_checkpoint", fake_restore)
    with pytest.raises(ValueError, match="production plan"):
        evaluation.evaluate(args)
    assert not args.output_dir.exists()
