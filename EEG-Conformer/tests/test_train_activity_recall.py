"""Recall protocol: objective, leakage boundaries, weights and exact resume."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import train_activity_recall as r


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(tmp_path, mode='hierarchical'):
    args = r.parse_args(['--output-dir', str(tmp_path / 'out'), '--input-domain', 'time_fft',
                         '--classification-mode', mode, '--class-weights', '3,3,1',
                         '--epochs', '2', '--batch-size', '6', '--cpu-threads', '1',
                         '--inner-folds', '2', '--device', 'cpu',
                         '--gate-thresholds', '0.4,0.5', '--within-thresholds', '0.5'])
    return r.resolve_config(args)


class TinyDual(nn.Module):
    def __init__(self, shape, cfg):
        super().__init__()
        self.mode = cfg['model']['classification_mode']
        self.drop = nn.Dropout(.25)
        self.linear = nn.Linear(int(np.prod(shape)), 3)

    def forward(self, time, fft):
        features = self.drop(time.flatten(1))
        logits = self.linear(features)
        if self.mode == 'hierarchical':
            logits = logits.log_softmax(1)
        return features, logits


def tiny_data():
    X = np.random.default_rng(6).normal(size=(24, 2, 8)).astype(np.float32)
    y = np.tile([0, 1, 2, 0, 1, 2], 4)
    subjects = np.repeat(np.arange(1, 5), 6)
    return X, y, subjects


def test_existing_command_and_loss_weights(tmp_path):
    argv = ['--output-dir', str(tmp_path), '--input-domain', 'time_fft', '--classification-mode', 'hierarchical',
            '--class-weights', '3,3,1', '--transformer-branches', '3', '--transformer-depths', '11', '10', '8',
            '--transformer-encoder-dropout', '.85', '--transformer-branch-fusion', 'loss_softmax',
            '--branch-loss-aux-weight', '.2', '--transformer-branch-qkv', 'cross_depth',
            '--transformer-branch-qkv-dropout', '.25', '--resume', '--skip-existing']
    args = r.parse_args(argv)
    cfg = r.resolve_config(args)
    assert cfg['model']['transformer_depths'] == [11, 10, 8]
    assert cfg['model']['transformer_encoder_dropout'] == .85
    model = TinyDual((2, 8), cfg)
    loss, _ = r.training_components(model, cfg, torch.device('cpu'))
    assert isinstance(loss, nn.NLLLoss)
    torch.testing.assert_close(loss.weight, torch.tensor([3., 3., 1.]))
    lp = torch.tensor([[.1, .1, .8], [.1, .1, .8], [.1, .1, .8]]).log().requires_grad_()
    labels = torch.tensor([0, 1, 2])
    actual = loss(lp, labels)
    expected = -(3 * lp[0, 0] + 3 * lp[1, 1] + lp[2, 2]) / 7
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(lp.grad.diagonal(), -torch.tensor([3., 3., 1.]) / 7)
    cfg['model']['classification_mode'] = 'flat'
    assert isinstance(r.training_components(model, cfg, torch.device('cpu'))[0], nn.CrossEntropyLoss)
    # Legacy parser defaults remain untouched.
    assert r.batch.parse_args([]).classification_mode == 'flat'
    assert r.batch.parse_args([]).class_weights is None


def test_neutral_routing_and_threshold_effect():
    p = np.array([[.31, .29, .40], [.2, .25, .55], [.1, .1, .8]], dtype=np.float32)
    neutral = dict(mode='hierarchical', gate_threshold=.5, within_threshold=.5)
    assert r.apply_decision(np.log(p), neutral).tolist() == [0, 2, 2]
    assert r.apply_decision(np.log(p), {**neutral, 'gate_threshold': .4}).tolist() == [0, 1, 2]
    assert r.apply_decision(np.log(p), dict(mode='flat', biases=[0, 0, 0])).tolist() == [2, 2, 2]
    assert r.apply_decision(np.log(p), dict(mode='flat', biases=[.5, .5, 0])).tolist() == [0, 2, 2]


def test_accuracy_constraint_fallback_and_subject_weighting(tmp_path):
    cfg = config(tmp_path, 'flat')
    cfg['decisions'] = [dict(mode='flat', biases=[0., 0., 0.])]
    # Two subjects: one has ten times more samples; each still contributes 1/2.
    y = np.concatenate([np.tile([0, 1, 2], 10), [0, 1, 2]])
    s = np.repeat([1, 2], [30, 3])
    def scores(pred):
        p = np.full((len(pred), 3), .05)
        p[np.arange(len(pred)), pred] = .9
        return np.log(p)
    first = np.where(s == 1, y, (y + 1) % 3)  # mean accuracy .5, pooled .909
    second = np.where(y == 0, 1, y)  # mean accuracy 2/3
    arr = dict(y_true=y, subject_ids=s, log_probs=np.stack([scores(first), scores(second)]))
    cfg['min_val_accuracy'] = .6
    result = r.select_operating_point([arr], [1, 2], cfg)
    assert result['epoch'] == 2
    assert result['accuracy_floor_met']
    assert result['candidate_scores'][0]['accuracy'] == .5
    cfg['min_val_accuracy'] = .99
    result = r.select_operating_point([arr], [1, 2], cfg)
    assert not result['accuracy_floor_met']
    assert result['fallback'] == 'maximum_mean_subject_macro_f1'
    assert result['epoch'] == 2
    # A missing, duplicate or outer subject cannot silently enter the selection.
    with pytest.raises(ValueError, match='exactly once'):
        r.select_operating_point([arr, arr], [1, 2], cfg)
    with pytest.raises(ValueError, match='exactly once'):
        r.select_operating_point([arr], [1, 2, 4], cfg)


def test_exact_epoch_resume_including_dropout_and_shuffle(tmp_path, monkeypatch):
    X, y, s = tiny_data()
    cfg = config(tmp_path)
    train, val = np.arange(12), np.arange(12, 18)
    common = (X, y, s, train, val, (2, 8), cfg, 1051, 2)
    r.fit_stage(*common, tmp_path / 'continuous', 'stage', torch.device('cpu'), model_factory=TinyDual)
    original = r.validation.train_epoch
    calls = []
    def interrupt(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('simulated interruption')
        return original(*args)
    monkeypatch.setattr(r.validation, 'train_epoch', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        r.fit_stage(*common, tmp_path / 'resumed', 'stage', torch.device('cpu'), model_factory=TinyDual)
    saved = r.core.load_torch_checkpoint(tmp_path / 'resumed/last_checkpoint.pt', torch.device('cpu'))
    assert saved['epoch'] == 1
    monkeypatch.setattr(r.validation, 'train_epoch', original)
    r.fit_stage(*common, tmp_path / 'resumed', 'stage', torch.device('cpu'), resume=True, model_factory=TinyDual)
    a = r.load_npz(tmp_path / 'continuous/validation_predictions.npz')
    b = r.load_npz(tmp_path / 'resumed/validation_predictions.npz')
    np.testing.assert_array_equal(a['log_probs'], b['log_probs'])
    assert r.read_json(tmp_path / 'continuous/train_history.json') == r.read_json(tmp_path / 'resumed/train_history.json')
    with pytest.raises(ValueError, match='mismatch'):
        r.fit_stage(*common, tmp_path / 'resumed', 'different', torch.device('cpu'), resume=True, model_factory=TinyDual)


@pytest.mark.parametrize('mode', ['hierarchical', 'flat'])
def test_nested_run_outer_once_and_outer_labels_do_not_select(tmp_path, monkeypatch, mode):
    X, y, s = tiny_data()
    cfg = config(tmp_path, mode)
    split = r.validation.make_subject_splits(s, 4, 2, 20260906)
    output = tmp_path / 'run'
    original = r.predict_log_probs
    calls = []
    def predict(model, loader, device):
        is_outer = (output / 'refit_checkpoint.pt').exists()
        calls.append(is_outer)
        if is_outer:
            assert (output / 'selection.json').exists()
        return original(model, loader, device)
    monkeypatch.setattr(r, 'predict_log_probs', predict)
    result = r.run_outer_fold(X, y, s, split, cfg, output, 'run', torch.device('cpu'), model_factory=TinyDual)
    assert calls == [False, False, False, False, True]
    assert result['outer_evaluation_count'] == 1
    p = r.load_npz(output / 'outer_predictions.npz')
    np.testing.assert_array_equal(p['sample_indices'], np.arange(18, 24))
    np.testing.assert_allclose(p['probabilities'].sum(1), 1., atol=1e-6)
    assert set(p['subject_ids']) == {4}
    # Complete skip makes no further forward passes.
    r.run_outer_fold(X, y, s, split, cfg, output, 'run', torch.device('cpu'),
                     skip_existing=True, model_factory=TinyDual)
    assert len(calls) == 5
    monkeypatch.setattr(r, 'predict_log_probs', original)
    changed = y.copy()
    changed[s == 4] = (changed[s == 4] + 1) % 3
    r.run_outer_fold(X, changed, s, split, cfg, tmp_path / 'changed', 'changed', torch.device('cpu'), model_factory=TinyDual)
    assert r.read_json(output / 'selection.json') == r.read_json(tmp_path / 'changed/selection.json')
    # Refit weights cannot be affected by held-out labels either.
    a = r.core.load_torch_checkpoint(output / 'refit_checkpoint.pt', torch.device('cpu'))
    b = r.core.load_torch_checkpoint(tmp_path / 'changed/refit_checkpoint.pt', torch.device('cpu'))
    for k in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][k], b['state_dict'][k], rtol=0, atol=0)


def test_dry_run_identity_rejects_changed_weights_and_legacy_directory(tmp_path, monkeypatch):
    X, y, s = tiny_data()
    data = tmp_path / 'data'
    data.mkdir()
    for name, value in [('X', X), ('y', y), ('subject_ids', s)]:
        np.save(data / f'{name}.npy', value)
    (data / 'metadata.json').write_text('{}')
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry-run must not train')
    monkeypatch.setattr(r, 'run_outer_fold', forbidden)
    argv = ['--dataset-root', str(data), '--output-dir', str(tmp_path / 'plan'), '--input-domain', 'time_fft',
            '--classification-mode', 'hierarchical', '--class-weights', '3,3,1', '--device', 'cpu',
            '--subject-ids', '1', '--epochs', '2', '--dry-run', '--resume', '--cpu-threads', '1']
    r.main(argv)
    plan = r.read_json(tmp_path / 'plan/protocol_plan.json')
    assert plan['budget'] == dict(inner_fits=3, refits=1, maximum_total_epochs=8)
    assert plan['config']['class_weights'] == [3, 3, 1]
    changed = argv.copy()
    changed[changed.index('3,3,1')] = '6,6,1'
    with pytest.raises(ValueError, match='mismatch'):
        r.main(changed)
    (tmp_path / 'legacy').mkdir()
    changed = argv.copy()
    changed[changed.index(str(tmp_path / 'plan'))] = str(tmp_path / 'legacy')
    with pytest.raises(FileExistsError, match='NEW'):
        r.main(changed)


def test_real_model_forward_loss_and_optimizer(tmp_path):
    cfg = config(tmp_path)
    cfg['model'].update(transformer_depths=[2, 1], transformer_branch_fusion='loss_softmax',
                        branch_loss_aux_weight=.2, transformer_branch_qkv='cross_depth')
    model = r.build_model((3, 256), cfg)
    criterion, optimizer = r.training_components(model, cfg, torch.device('cpu'))
    inputs = (torch.randn(3, 1, 3, 256), torch.randn(3, 1, 3, 129), torch.arange(3))
    logits, labels, branches = r.core.forward_model_batch_with_branches(model, inputs, torch.device('cpu'))
    torch.testing.assert_close(logits.exp().sum(1), torch.ones(3))
    loss, _ = r.core.compute_model_batch_loss(model, logits, labels, criterion, branches)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(head.group_head.fc[-1].weight.grad is not None for head in model.branch_cls_heads)


def test_prediction_roundtrip_keeps_saved_normalizers_and_decision(tmp_path):
    import predict_activity_recall as prediction
    cfg = config(tmp_path)
    cfg['model'].update(transformer_depths=[1], depth=1)
    shape = (3, 256)
    windows = np.random.default_rng(41).normal(size=(3, *shape)).astype(np.float32)
    inputs, stats = r.validation.fit_inputs(windows, True)
    model = r.build_model(shape, cfg).eval()
    loader = r.validation.make_loader(inputs, np.zeros(3, dtype=np.int64), 3, False, 0)
    _, expected = r.predict_log_probs(model, loader, torch.device('cpu'))
    rule = dict(mode='hierarchical', gate_threshold=.4, within_threshold=.55)
    checkpoint = dict(protocol='subject_validated_recall_v1', shape=list(shape),
                      state_dict=model.state_dict(), model_config=cfg['model'], normalizers=stats,
                      decision=rule, neutral_decision=cfg['decisions'][0])
    actual = prediction.predict_windows(checkpoint, windows, torch.device('cpu'), batch_size=3)
    np.testing.assert_array_equal(actual['log_probs'], expected)
    np.testing.assert_array_equal(actual['y_pred'], r.apply_decision(expected, rule))
    with pytest.raises(ValueError, match='Expected a refit_checkpoint'):
        prediction.predict_windows({}, windows, torch.device('cpu'))


def test_refit_interruption_resumes_without_retraining_inner_models(tmp_path, monkeypatch):
    X, y, s = tiny_data()
    cfg = config(tmp_path)
    split = r.validation.make_subject_splits(s, 4, 2, 20260906)
    continuous = tmp_path / 'continuous'
    interrupted = tmp_path / 'interrupted'
    r.run_outer_fold(X, y, s, split, cfg, continuous, 'run', torch.device('cpu'), model_factory=TinyDual)
    original = r.core.atomic_torch_save
    stopped = []
    def save_then_stop(payload, path):
        result = original(payload, path)
        if Path(path).parent.name == 'refit' and Path(path).name == 'last_checkpoint.pt' and not stopped:
            stopped.append(True)
            raise RuntimeError('refit power loss')
        return result
    monkeypatch.setattr(r.core, 'atomic_torch_save', save_then_stop)
    with pytest.raises(RuntimeError, match='refit power loss'):
        r.run_outer_fold(X, y, s, split, cfg, interrupted, 'run', torch.device('cpu'), model_factory=TinyDual)
    assert not (interrupted / 'outer_predictions.npz').exists()
    assert (interrupted / 'inner_0/complete.json').exists()
    monkeypatch.setattr(r.core, 'atomic_torch_save', original)
    r.run_outer_fold(X, y, s, split, cfg, interrupted, 'run', torch.device('cpu'), resume=True, model_factory=TinyDual)
    a = r.load_npz(continuous / 'outer_predictions.npz')
    b = r.load_npz(interrupted / 'outer_predictions.npz')
    np.testing.assert_array_equal(a['log_probs'], b['log_probs'])
    assert r.read_json(continuous / 'outer_metrics.json') == r.read_json(interrupted / 'outer_metrics.json')
