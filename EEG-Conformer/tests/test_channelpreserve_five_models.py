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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_activity_channelpreserve_five_models as suite
import evaluate_channelpreserve_five_models_failurecase as evaluation
import train_activity_channelpreserve as single
from evaluate_activity_failurecase import (
    apply_legacy_normalizers, confusion_matrix, fit_legacy_normalizers, load_best_checkpoint,
    load_dataset, predict_probabilities, verify_saved_predictions,
)
from test_train_activity_channelpreserve import write_dataset


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def arguments(tmp_path, dataset, *extra, small=False):
    options = ["--cohort-subject-ids", "1,2,6,17"] if small else []
    return suite.parse_args(["--dataset-root", str(dataset), "--output-base", str(tmp_path / "out"),
                             "--device", "cpu", *options, *extra])


def test_five_model_default_and_two_gpu_model_subsets_share_data_and_unweighted_defaults(tmp_path):
    dataset = write_dataset(tmp_path / "formal", evaluation.FORMAL_SUBJECTS)
    all_models = suite.build_plan(arguments(tmp_path, dataset))
    main = suite.build_plan(arguments(tmp_path, dataset, "--models", "channelpreserving"))
    other = suite.build_plan(arguments(tmp_path, dataset, "--models", ",".join(suite.MODELS[1:])))
    assert all_models["models"] == list(suite.MODELS)
    assert [p["planned_training_folds"] for p in (all_models, main, other)] == [75, 15, 60]
    assert len({p["output_dir"] for p in (all_models, main, other)}) == 3
    assert all_models["dataset_sha256"] == main["dataset_sha256"] == other["dataset_sha256"]
    assert all_models["class_weights"] is None and all_models["epochs"] == 200
    assert all_models["batch_size"] == 72 and all_models["lr"] == .0002
    for name, kwargs in all_models["per_model_kwargs"].items():
        assert kwargs["class_weights"] is None and kwargs["train_sampling"] == "original"
        assert kwargs["input_domain"] == ("time_fft" if name == "channelpreserving" else "time")
    assert all_models["model_architecture_configs"]["channelpreserving"]["late_skip_targets"] == [6,7,8,9,10]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("leaked", [6, 17])
def test_s6_and_s17_rejected_before_any_model_training(tmp_path, leaked):
    dataset = write_dataset(tmp_path / "formal", (1, 2, leaked))
    with pytest.raises(ValueError, match="Excluded subjects still present"):
        suite.build_plan(arguments(tmp_path, dataset, small=True))


@pytest.mark.parametrize("models", ["mdtf", "eegnet,eegnet", "channelpreserving,unknown"])
def test_old_or_duplicate_model_names_rejected(models, tmp_path):
    with pytest.raises(ValueError, match="--models"):
        suite.build_plan(arguments(tmp_path, tmp_path / "absent", "--models", models))


def test_real_smoke_trains_all_five_models_replays_weights_and_resumes_complete_job(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "formal")
    args = arguments(tmp_path, dataset, "--smoke-test", small=True)
    plan = suite.build_plan(args)
    monkeypatch.setattr(suite, "collect_environment_info", lambda *_: {})
    suite.run_plan(plan)
    root = Path(plan["output_dir"])
    assert json.loads((root / "completed.json").read_text()) == suite.completed_record(plan)
    formal = load_dataset(dataset)
    stats = fit_legacy_normalizers(formal.X[formal.subject_ids != 1], dual=True)
    for name in suite.MODELS:
        fold = suite.model_run_dir(root, name, 42) / "fold_subject_1"
        metrics = suite.validate_completed_fold(fold, plan, name, 1)
        assert metrics["class_weights"] is None
        model = suite.restore_suite_model(load_best_checkpoint(fold / "best_model.pt"), metrics, name, plan)
        inputs = apply_legacy_normalizers(formal.X[formal.subject_ids == 1], stats, name == "channelpreserving")
        _, replay = predict_probabilities(model, inputs, torch.device("cpu"), 2)
        verify_saved_predictions(replay, formal.y[formal.subject_ids == 1], {"directory": fold, "metrics": metrics})
    assert len(json.loads((root / "five_model_summary.json").read_text())["models"]) == 5
    monkeypatch.setattr(suite, "train_loso_fold", lambda **_: pytest.fail("A completed fold was retrained"))
    suite.run_plan(suite.build_plan(args), resume=True)
    changed = copy.deepcopy(plan)
    changed["code_sha256"] = {}
    with pytest.raises(ValueError, match="hash mismatch"):
        suite.run_plan(changed, resume=True)
    with pytest.raises(FileExistsError):
        suite.run_plan(plan)
    fold = suite.model_run_dir(root, "eegnet", 42) / "fold_subject_1"
    metrics = json.loads((fold / "metrics.json").read_text())
    metrics["class_weights"] = [3,3,1]
    (fold / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="class_weights"):
        suite.run_plan(plan, resume=True)


class ConstantClassifier(nn.Module):
    def __init__(self, winner=0):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([2. if i == winner else 0. for i in range(3)]))

    def forward(self, time, *fft):
        return time.flatten(1), self.logits.expand(len(time), -1)


def fake_restore(checkpoint, *_):
    model = ConstantClassifier()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def write_completed_fixture(tmp_path, case_id, models=None):
    dataset = write_dataset(tmp_path / "formal", evaluation.FORMAL_SUBJECTS)
    case = write_dataset(tmp_path / "case", (case_id,), excluded=())
    extra = [] if models is None else ["--models", ",".join(models)]
    plan = suite.build_plan(arguments(tmp_path, dataset, *extra))
    root = Path(plan["output_dir"])
    root.mkdir(parents=True)
    (root / "run_plan.json").write_text(json.dumps(plan))
    (root / "completed.json").write_text(json.dumps(suite.completed_record(plan)))
    for name in plan["models"]:
        for index, subject in enumerate(evaluation.FORMAL_SUBJECTS):
            fold = suite.model_run_dir(root, name, 42) / f"fold_subject_{subject}"
            fold.mkdir(parents=True)
            pred = np.full(3, index % 3, dtype=np.int64)
            metrics = {
                "architecture": name, "architecture_config": plan["model_architecture_configs"][name],
                "test_subject_id": subject, "seed": 42, "epochs": 200, "batch_size": 72, "lr": .0002,
                "class_weights": None, "optimizer": "Adam", "optimizer_betas": [.5,.999],
                "input_domain": "time_fft" if name == "channelpreserving" else "time",
                "classification_mode": "flat", "train_sampling": "original", "loss": "CrossEntropyLoss",
                "emb_size": 40, "num_heads": 5, "depth": 10 if name == "channelpreserving" else 6,
                "dropout": .5, "transformer_encoder_dropout": .5, "transformer_branches": 1,
                "branch_loss_aux_weight": 0., "transformer_branch_qkv": "none",
                "n_channels": 21, "n_times": 1920, "n_classes": 3, "n_train_samples": 42, "n_test_samples": 3,
                "best_epoch": 1, "best_test_acc": 1/3,
                "confusion_matrix": confusion_matrix(np.arange(3), pred),
                "trainable_parameters": plan["model_trainable_parameters"][name],
            }
            (fold / "metrics.json").write_text(json.dumps(metrics))
            (fold / "epoch_history.json").write_text(json.dumps({
                **metrics, "history": [{"epoch": n} for n in range(1,201)],
            }))
            np.savez(fold / "test_predictions.npz", y_true=np.arange(3), y_pred=pred)
            torch.save({**metrics, "epoch": 0, "state_dict": ConstantClassifier(index%3).state_dict()},
                       fold / "best_model.pt")
    args = evaluation.parse_args([
        "--dataset-root", str(dataset), "--failurecase-root", str(case), "--run-root", str(root),
        "--output-dir", str(tmp_path / "evaluation"), "--case-subject-id", str(case_id),
        "--device", "cpu", "--batch-size", "2",
    ])
    return args, plan


@pytest.mark.parametrize("case_id,models", [
    (6, None), (17, None), (17, ("channelpreserving",)), (17, suite.MODELS[1:]),
])
def test_complete_single_and_split_model_jobs_evaluate_cases_without_fold_merging(tmp_path, monkeypatch, case_id, models):
    args, plan = write_completed_fixture(tmp_path, case_id, models)
    monkeypatch.setattr(single, "restore_checkpoint", fake_restore)
    monkeypatch.setattr(suite, "restore_suite_model", fake_restore)
    monkeypatch.setattr(evaluation, "collect_environment_info", lambda *_: {})
    summary = evaluation.evaluate(args)
    assert set(summary["models"]) == set(plan["models"])
    assert summary["case_used_for_training_or_selection"] is False
    assert json.loads((args.output_dir / "completed.json").read_text())["n_verified_checkpoints"] == len(plan["models"])*15
    for name in plan["models"]:
        result = summary["models"][name]
        assert result["n_fold_models"] == 15 and result["ensemble"]["n_windows"] == 3
        with np.load(args.output_dir / name / "ensemble_predictions.npz") as saved:
            assert saved["fold_probabilities"].shape == (15,3,3)
            np.testing.assert_array_equal(saved["probabilities"], saved["fold_probabilities"].mean(axis=0,dtype=np.float64))
    normalizers = json.loads((args.output_dir / "normalizers.json").read_text())
    for subject, row in normalizers.items():
        assert not ({6,17,int(subject)} & set(row["train_subject_ids"]))
    with pytest.raises(FileExistsError):
        evaluation.evaluate(args)


@pytest.mark.parametrize("field,value", [
    ("smoke_test",True), ("test_subject_ids",[1]), ("code_sha256",{}),
    ("class_weights",[3,3,1]), ("epochs",1),
])
def test_failurecase_rejects_incomplete_debug_old_weights_and_changed_code(tmp_path, monkeypatch, field, value):
    args, plan = write_completed_fixture(tmp_path,17,("eegnet",))
    plan[field] = value
    (args.run_root / "run_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(suite,"restore_suite_model",fake_restore)
    with pytest.raises(ValueError):
        evaluation.evaluate(args)
    assert not args.output_dir.exists()


def test_failurecase_prediction_replay_blocks_case_scoring(tmp_path, monkeypatch):
    args, _ = write_completed_fixture(tmp_path,17,("eegnet",))
    monkeypatch.setattr(evaluation,"collect_environment_info",lambda *_: {})
    calls = []

    def restore(checkpoint, *_):
        calls.append(True)
        return fake_restore(checkpoint) if len(calls)<=15 else ConstantClassifier(2)

    monkeypatch.setattr(suite,"restore_suite_model",restore)
    with pytest.raises(ValueError,match="replay mismatch"):
        evaluation.evaluate(args)
    assert not (args.output_dir / "failurecase_summary.json").exists()
    assert not list(args.output_dir.glob("*/fold_subject_*/probabilities.npz"))
