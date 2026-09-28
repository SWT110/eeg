"""Four-way orchestration, sampling isolation, and actual checkpoint/resume coverage."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_activity_weight_sweep as sweep


class TinyDual(torch.nn.Module):
    def __init__(self, n_channels, time_n_times, classification_mode, **kwargs):
        super().__init__()
        self.classification_mode = classification_mode
        self.linear = torch.nn.Linear(n_channels * time_n_times, 3)
        self.drop = torch.nn.Dropout(.25)

    def transformer_weight_metadata(self):
        return {}

    def forward(self, time, fft):
        features = self.drop(time.flatten(1))
        return features, self.linear(features).log_softmax(1)


@pytest.fixture
def data(tmp_path):
    directory = tmp_path / 'data'
    directory.mkdir()
    np.save(directory / 'X.npy', np.random.default_rng(5).normal(size=(12, 2, 32)).astype('float32'))
    np.save(directory / 'y.npy', np.tile([0, 1, 2], 4))
    np.save(directory / 'subject_ids.npy', np.repeat([1, 2], 6))
    (directory / 'metadata.json').write_text('{}')
    return directory


def test_sampling_only_changes_training_draws(data):
    regular = sweep.core.build_dataloaders(data, 2, 4, 'time_fft')
    tripled = sweep.core.build_dataloaders(data, 2, 4, 'time_fft', 'triple_minority_replacement')
    train, test = tripled[:2]
    sampled = list(train.sampler)
    labels = train.dataset.tensors[-1].numpy()
    assert np.bincount(labels[sampled], minlength=3).tolist() == [6, 6, 2]
    assert sorted(i for i in sampled if labels[i] == 2) == [2, 5]
    assert len(test.dataset) == 6
    for original, new in zip(regular[1].dataset.tensors, test.dataset.tensors):
        torch.testing.assert_close(original, new, rtol=0, atol=0)
    for original, new in zip(regular[0].dataset.tensors, train.dataset.tensors):
        torch.testing.assert_close(original, new, rtol=0, atol=0)


def test_four_runs_resume_skip_and_distinct_weights(data, tmp_path, monkeypatch):
    monkeypatch.setattr(sweep.core, 'DualBranchActivityConformer', TinyDual)
    monkeypatch.setitem(sweep.BASELINE, 'epochs', 2)
    monkeypatch.setitem(sweep.BASELINE, 'batch_size', 4)
    args = ['--class-weights', '3,4,1', '--dataset-root', str(data),
            '--output-root', str(tmp_path / 'out'), '--device', 'cpu', '--cpu-threads', '1']
    original_train = sweep.core.train_loso_fold
    calls = []
    def record(**kwargs):
        calls.append((kwargs['classification_mode'], kwargs['train_sampling'], kwargs['test_subject_id']))
        return original_train(**kwargs)
    monkeypatch.setattr(sweep.core, 'train_loso_fold', record)
    sweep.main(args)
    assert len(calls) == 8
    assert [(calls[i][0], calls[i][1]) for i in (0, 2, 4, 6)] == [
        ('flat', 'triple_minority_replacement'), ('flat', 'original'),
        ('hierarchical', 'original'), ('hierarchical', 'triple_minority_replacement')]
    sweep.main(args)
    assert len(calls) == 8
    for directory in (tmp_path / 'out').iterdir():
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['complete'] and len(summary['mean_subject_recall']) == 3
        assert summary['class_weights'] == [3., 4., 1.]
    sweep.main([*args, '--class-weights', '4,4,1'])
    assert len(calls) == 16
    assert len(list((tmp_path / 'out').iterdir())) == 8
    with pytest.raises(ValueError, match='Configuration/data changed'):
        sweep.main([*args, '--cpu-threads', '2'])


@pytest.mark.parametrize('mode', ['flat', 'hierarchical'])
def test_weighted_triple_training_exact_resume(data, tmp_path, monkeypatch, mode):
    torch.set_num_threads(1)
    monkeypatch.setattr(sweep.core, 'DualBranchActivityConformer', TinyDual)
    config = dict(sweep.BASELINE, epochs=2, batch_size=4, class_weights=[3., 4., 1.],
                  classification_mode=mode, train_sampling='triple_minority_replacement',
                  dataset_root=data, test_subject_id=2, device='cpu', resume=True)
    sweep.core.train_loso_fold(**config, output_dir=tmp_path / 'continuous')
    original_save = sweep.core.atomic_torch_save
    def interrupted_save(checkpoint, path):
        original_save(checkpoint, path)
        if Path(path).name == sweep.core.RESUME_CHECKPOINT_FILENAME and checkpoint['completed_epoch'] == 1:
            raise RuntimeError('interruption')
    monkeypatch.setattr(sweep.core, 'atomic_torch_save', interrupted_save)
    with pytest.raises(RuntimeError, match='interruption'):
        sweep.core.train_loso_fold(**config, output_dir=tmp_path / 'resumed')
    monkeypatch.setattr(sweep.core, 'atomic_torch_save', original_save)
    sweep.core.train_loso_fold(**config, output_dir=tmp_path / 'resumed')
    first = json.loads((tmp_path / 'continuous/fold_subject_2/epoch_history.json').read_text())
    second = json.loads((tmp_path / 'resumed/fold_subject_2/epoch_history.json').read_text())
    assert first == second


@pytest.mark.parametrize('value', ['3,4', '3,nan,1', '3,inf,1', '0,4,1'])
def test_invalid_weights_fail_before_output(value, tmp_path):
    with pytest.raises(SystemExit):
        sweep.main(['--class-weights', value, '--output-root', str(tmp_path / 'out'), '--dry-run'])
    assert not (tmp_path / 'out').exists()
