"""Protocol checks: isolation, subject weighting, uniform fusion, tiny CPU fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import train_activity_subject_validation as protocol


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_subject_splits_are_disjoint_and_cover_every_subject_once():
    ids = np.repeat(np.arange(1, 12), 3)
    splits = protocol.make_subject_splits(ids, 3, 3, 20260906)
    assert splits == protocol.make_subject_splits(ids[::-1], 3, 3, 20260906)
    all_validation = []
    for inner in splits["inner_folds"]:
        train, validation = set(inner["train_subject_ids"]), set(inner["validation_subject_ids"])
        assert train.isdisjoint(validation)
        assert 3 not in train | validation
        assert train | validation == set(range(1, 12)) - {3}
        all_validation.extend(validation)
    assert sorted(all_validation) == splits["outer_train_subject_ids"]
    with pytest.raises(ValueError):
        protocol.make_subject_splits(ids, 3, 1, 42)


def test_train_only_normalization_separate_time_and_fft():
    rng = np.random.default_rng(7)
    training = rng.normal(size=(6, 2, 32)).astype(np.float32)
    held_out = np.full((3, 2, 32), 100.0, dtype=np.float32)
    normalized, stats = protocol.fit_inputs(training, dual=True)
    transformed = protocol.transform_inputs(held_out, dual=True, stats=stats)
    _, unchanged = protocol.fit_inputs(training, dual=True)
    assert stats == unchanged
    assert abs(normalized[0].mean()) < 1e-6
    assert abs(normalized[1].mean()) < 1e-6
    assert transformed[0].mean() > 90
    assert stats["time"]["mean"] != stats["fft"]["mean"]
    expected = (held_out[:, None] - training.mean(dtype=np.float64)) / training.std(dtype=np.float64)
    np.testing.assert_allclose(transformed[0], expected, rtol=1e-6)
    with pytest.raises(ValueError):
        protocol.fit_scalar_normalizer(np.ones((2, 4)))


def test_selection_weights_subjects_not_groups_and_ignores_outer_fields():
    # A: large validation group is perfect, small group is zero => 3/4.
    # B: all subjects score .6; equal-group averaging would incorrectly choose B.
    rows = []
    for candidate, scores in (("A", [1, 1, 1, 0]), ("B", [.6, .6, .6, .6])):
        for epoch in (1, 2):
            for subject_id, score in enumerate(scores, start=1):
                rows.append({"candidate_id": candidate, "epoch": epoch,
                             "validation_subject_id": subject_id, "macro_f1": score,
                             "inner_fold": 0 if subject_id < 4 else 1,
                             "outer_accuracy": 0 if candidate == "A" else 1})
    result = protocol.select_inner_candidate(rows, ["B", "A"], [1, 2, 3, 4], 2)
    assert result["candidate_id"] == "A"
    assert result["epoch"] == 1
    assert result["mean_subject_macro_f1"] == .75
    for row in rows:
        row["macro_f1"] = .5
    assert protocol.select_inner_candidate(rows, ["B", "A"], [1, 2, 3, 4], 2)["candidate_id"] == "B"
    with pytest.raises(ValueError):
        protocol.select_inner_candidate(rows[:-1], ["B", "A"], [1, 2, 3, 4], 2)


def test_uniform_fusion_stays_exactly_uniform_after_optimizer_step():
    model = protocol.DualBranchActivityConformer(
        n_channels=2, time_n_times=256, fft_n_times=129, n_classes=3,
        emb_size=10, num_heads=5, transformer_depths=[1, 1, 1],
        transformer_branch_fusion="loss_softmax", transformer_branch_qkv="cross_depth",
        branch_loss_aux_weight=.2,
    )
    protocol.freeze_uniform_fusion(model)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
    batch = (torch.randn(3, 1, 2, 256), torch.randn(3, 1, 2, 129), torch.arange(3))
    logits, labels, branches = protocol.forward_model_batch_with_branches(model, batch, torch.device("cpu"))
    loss, _ = protocol.compute_model_batch_loss(model, logits, labels, nn.CrossEntropyLoss(), branches)
    loss.backward()
    optimizer.step()
    assert model.branch_loss_weight_logits.grad is None
    assert not model.branch_loss_weight_logits.requires_grad
    torch.testing.assert_close(model.normalized_branch_loss_weights(), torch.ones(3) / 3, rtol=0, atol=0)


class TinyClassifier(nn.Module):
    def __init__(self, shape, classes):
        super().__init__()
        self.linear = nn.Linear(int(np.prod(shape)), classes)

    def forward(self, inputs):
        features = inputs.flatten(1)
        return features, self.linear(features)


def test_tiny_cpu_nested_round_evaluates_outer_once_after_fixed_refit(tmp_path, monkeypatch):
    config = protocol.load_config(ROOT / "configs" / "subject_validation_first_round.json")
    config["training"].update(max_epochs=1, batch_size=6)
    config["inner_folds"] = 2
    subjects = np.repeat(np.arange(1, 5), 6)
    y = np.tile([0, 1, 2, 0, 1, 2], 4)
    X = np.random.default_rng(2).normal(size=(24, 2, 8)).astype(np.float32)
    split = protocol.make_subject_splits(subjects, 4, 2, 20260906)
    calls = []
    factory_models = []
    output = tmp_path / "tiny"
    original_predict = protocol.predict_once

    def counting_predict(model, loader, device):
        after_refit = (output / "refit_checkpoint.pt").exists()
        calls.append(after_refit)
        if after_refit:
            assert (output / "selection.json").exists()
            assert len(json.loads((output / "refit_train_history.json").read_text())) == 1
        return original_predict(model, loader, device)

    def factory(preset, shape, classes, candidate):
        model = TinyClassifier(shape, classes)
        factory_models.append(model)
        return model

    monkeypatch.setattr(protocol, "predict_once", counting_predict)
    result = protocol.run_outer_fold(X, y, subjects, split, "time", 42, config,
                                     output, torch.device("cpu"), model_factory=factory)
    assert calls == [False, False, True]
    assert len({id(model) for model in factory_models}) == 3
    assert result["outer_evaluation_count"] == 1
    assert result["selected_epochs"] == 1
    assert result["n_test_samples"] == 6
    predicted = np.load(output / "outer_predictions.npz")
    np.testing.assert_array_equal(predicted["sample_indices"], np.arange(18, 24))
    assert set(predicted["subject_ids"]) == {4}
    with pytest.raises(FileExistsError):
        protocol.run_outer_fold(X, y, subjects, split, "time", 42, config, output,
                                torch.device("cpu"), model_factory=factory)


def test_dry_run_never_builds_or_trains_models_and_protects_output(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.save(dataset / "X.npy", np.ones((12, 2, 8), dtype=np.float32))
    np.save(dataset / "y.npy", np.tile([0, 1, 2], 4))
    np.save(dataset / "subject_ids.npy", np.repeat([1, 2, 3, 4], 3))
    (dataset / "metadata.json").write_text("{}")

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not construct or train a model")

    monkeypatch.setattr(protocol, "run_outer_fold", forbidden)
    monkeypatch.setattr(protocol, "build_model", forbidden)
    output = tmp_path / "plan"
    args = ["--dataset-root", str(dataset), "--output-dir", str(output), "--models", "mdtf",
            "--test-subject-ids", "1", "--seeds", "42", "--dry-run"]
    protocol.main(args)
    plan = json.loads((output / "protocol_plan.json").read_text())
    assert plan["budget"] == {"outer_refits": 1, "inner_fits": 3, "maximum_total_epochs": 800}
    assert plan["status"] == "planned; no training performed"
    assert len(plan["input_sha256"]["X.npy"]) == 64
    with pytest.raises(FileExistsError):
        protocol.main(args)
