"""Checks specific to the isolated three-arm resampling entrypoint."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import train_activity_resampling as r


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyDual(nn.Module):
    def __init__(self, shape, config):
        super().__init__()
        self.drop = nn.Dropout(.25)
        self.linear = nn.Linear(int(np.prod(shape)), 3)

    def forward(self, time, fft):
        features = self.drop(time.flatten(1))
        return features, self.linear(features).log_softmax(1)


def tiny_data():
    X = np.random.default_rng(6).normal(size=(24, 2, 8)).astype(np.float32)
    y = np.tile([0, 1, 2, 0, 1, 2], 4)
    subjects = np.repeat(np.arange(1, 5), 6)
    return X, y, subjects


def config(tmp_path, sampling='triple_minority_replacement'):
    args = r.parse_args(['--output-dir', str(tmp_path / 'out'), '--input-domain', 'time_fft',
                         '--classification-mode', 'hierarchical', '--class-weights', '1,1,1',
                         '--train-sampling', sampling, '--epochs', '2', '--batch-size', '6',
                         '--cpu-threads', '1', '--inner-folds', '2', '--device', 'cpu'])
    return r.resolve_config(args)


def test_tripled_sampler_draws_fixed_class_counts_and_all_majority_windows():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    first = r.TripledMinoritySampler(labels, torch.Generator().manual_seed(42))
    second = r.TripledMinoritySampler(labels, torch.Generator().manual_seed(42))
    draws = list(first)
    assert len(first) == 19
    assert np.bincount(labels[draws], minlength=3).tolist() == [9, 6, 4]
    assert sorted(index for index in draws if labels[index] == 2) == [5, 6, 7, 8]
    assert draws == list(second)
    assert list(first) == list(second)


def test_tripled_sampling_rejects_stacked_class_weights(tmp_path):
    args = r.parse_args(['--output-dir', str(tmp_path), '--input-domain', 'time_fft',
                         '--train-sampling', 'triple_minority_replacement', '--class-weights', '3,3,1'])
    with pytest.raises(ValueError, match='requires unweighted loss'):
        r.resolve_config(args)
    args.class_weights = '1,1,1'
    assert r.resolve_config(args)['train_sampling'] == 'triple_minority_replacement'


def test_resampling_exact_epoch_resume(tmp_path, monkeypatch):
    X, y, subjects = tiny_data()
    cfg = config(tmp_path)
    common = (X, y, subjects, np.arange(12), np.arange(12, 18), (2, 8), cfg, 1051, 2)
    r.fit_stage(*common, tmp_path / 'continuous', 'stage', torch.device('cpu'), model_factory=TinyDual)
    original = r.validation.train_epoch
    calls = []
    def stop_at_second_epoch(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('simulated interruption')
        return original(*args)
    monkeypatch.setattr(r.validation, 'train_epoch', stop_at_second_epoch)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        r.fit_stage(*common, tmp_path / 'resumed', 'stage', torch.device('cpu'), model_factory=TinyDual)
    monkeypatch.setattr(r.validation, 'train_epoch', original)
    r.fit_stage(*common, tmp_path / 'resumed', 'stage', torch.device('cpu'), resume=True, model_factory=TinyDual)
    a = r.load_npz(tmp_path / 'continuous/validation_predictions.npz')
    b = r.load_npz(tmp_path / 'resumed/validation_predictions.npz')
    np.testing.assert_array_equal(a['log_probs'], b['log_probs'])
    assert r.read_json(tmp_path / 'continuous/train_history.json') == r.read_json(tmp_path / 'resumed/train_history.json')
    effective = r.read_json(tmp_path / 'resumed/effective_training.json')
    assert effective['original_class_counts'] == [4, 4, 4]
    assert effective['class_draws_per_epoch'] == [12, 12, 4]


def test_outer_subject_is_never_sampled_and_dry_run_identity(tmp_path):
    X, y, subjects = tiny_data()
    cfg = config(tmp_path)
    split = r.validation.make_subject_splits(subjects, 4, 2, 20260906)
    output = tmp_path / 'fold'
    result = r.run_outer_fold(X, y, subjects, split, cfg, output, 'run',
                              torch.device('cpu'), model_factory=TinyDual)
    assert result['outer_evaluation_count'] == 1
    assert result['train_sampling'] == 'triple_minority_replacement'
    for inner in split['inner_folds']:
        effective = r.read_json(output / f"inner_{inner['inner_fold']}" / 'effective_training.json')
        assert 4 not in effective['train_subject_ids']
        assert 4 not in effective['validation_subject_ids']
    outer = r.load_npz(output / 'outer_predictions.npz')
    assert set(outer['subject_ids']) == {4}

    data = tmp_path / 'data'
    data.mkdir()
    for name, value in [('X', X), ('y', y), ('subject_ids', subjects)]:
        np.save(data / f'{name}.npy', value)
    (data / 'metadata.json').write_text('{}')
    argv = ['--dataset-root', str(data), '--output-dir', str(tmp_path / 'plan'),
            '--input-domain', 'time_fft', '--classification-mode', 'hierarchical',
            '--class-weights', '1,1,1', '--train-sampling', 'triple_minority_replacement',
            '--device', 'cpu', '--subject-ids', '1', '--epochs', '2', '--dry-run', '--resume',
            '--cpu-threads', '1']
    r.main(argv)
    plan = r.read_json(tmp_path / 'plan/protocol_plan.json')
    assert plan['full_dataset_class_draws_per_epoch'] == [24, 24, 8]
    with pytest.raises(ValueError, match='mismatch'):
        r.main([*argv, '--train-sampling', 'original'])
