"""Hierarchical path loss, hard routing, and LOSO compatibility checks."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from test_train_activity_loso import load_module, _make_fake_dataset

m = load_module()


def test_path_loss_matches_two_binary_tasks_and_masks_e3():
    torch.manual_seed(42)
    head = m.HierarchicalClassificationHead(8, 3).eval()
    x = torch.randn(6, 8)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    weights = torch.tensor([3., 3., 1.])
    joint = head(x)[1]
    torch.testing.assert_close(joint.exp().sum(1), torch.ones(6))
    group = head.group_head(x)[1]
    within = head.within_group_head(x)[1]
    expected = F.cross_entropy(group, (y == 2).long(), reduction='none')
    mask = y != 2
    expected[mask] += F.cross_entropy(within[mask], y[mask], reduction='none')
    torch.testing.assert_close(F.nll_loss(joint, y, weight=weights),
                               (expected * weights[y]).sum() / weights[y].sum())
    F.nll_loss(head(x)[1], torch.full((6,), 2)).backward()
    assert any(p.grad.abs().sum() > 0 for p in head.group_head.parameters())
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in head.within_group_head.parameters())


def test_hard_gate_differs_from_flat_argmax():
    scores = torch.tensor([[.31, .29, .40], [.25, .35, .40], [.2, .2, .6]]).log()
    assert m.predict_class_labels(SimpleNamespace(classification_mode='hierarchical'), scores).tolist() == [0, 1, 2]
    assert m.predict_class_labels(SimpleNamespace(), scores).tolist() == [2, 2, 2]


def test_cross_depth_hierarchical_forward_backward():
    model = m.DualBranchActivityConformer(
        n_channels=3, time_n_times=256, fft_n_times=129, emb_size=10,
        num_heads=2, transformer_depths=(11, 10, 8),
        transformer_branch_fusion='loss_softmax', branch_loss_aux_weight=.2,
        transformer_branch_qkv='cross_depth', transformer_encoder_dropout=.85,
        transformer_branch_qkv_dropout=.25, classification_mode='hierarchical',
    )
    _, logits, branches = model.forward_with_branch_logits(torch.randn(3, 1, 3, 256), torch.randn(3, 1, 3, 129))
    torch.testing.assert_close(logits.exp().sum(1), torch.ones(3))
    criterion = torch.nn.NLLLoss(weight=torch.tensor([3., 3., 1.]))
    loss, branch_losses = m.compute_model_batch_loss(model, logits, torch.tensor([0, 1, 2]), criterion, branches)
    expected = (model.normalized_branch_loss_weights() * torch.stack(branch_losses)).sum() + .2 * torch.stack(branch_losses).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(model.branch_loss_weight_logits.grad).all()
    for head in model.branch_cls_heads:
        assert head.group_head.fc[-1].weight.grad.abs().sum() > 0
        assert head.within_group_head.fc[-1].weight.grad.abs().sum() > 0


def test_cli_and_legacy_resume_defaults():
    import train_activity_loso_batch as batch
    for module in (m, batch):
        assert module.parse_args([]).classification_mode == 'flat'
        assert module.parse_args(['--classification-mode', 'hierarchical']).classification_mode == 'hierarchical'
    checkpoint = dict(resume_checkpoint_version=m.RESUME_CHECKPOINT_VERSION,
                      model_state_dict={}, optimizer_state_dict={}, rng_state={},
                      epoch_history=[{'epoch': 1}], training_config={},
                      completed_epoch=1, average_test_acc_sum=0.)
    assert m.validate_resume_checkpoint(checkpoint, {'classification_mode': 'flat'}, 2) == 1
    with pytest.raises(ValueError, match='classification_mode'):
        m.validate_resume_checkpoint(checkpoint, {'classification_mode': 'hierarchical'}, 2)


def test_real_fold_resume_and_skip_modes(tmp_path):
    import train_activity_loso_batch as batch
    data = tmp_path / 'dataset'
    _make_fake_dataset(data, n_subjects=2, n_channels=3, n_times=256)
    common = dict(dataset_root=data, test_subject_id=1, epochs=2, batch_size=3,
                  lr=.0002, device='cpu', output_dir=tmp_path / 'out', resume=True,
                  emb_size=10, num_heads=2, input_domain='time_fft',
                  transformer_branches=3, transformer_depths=(3, 2, 1),
                  transformer_branch_fusion='loss_softmax', branch_loss_aux_weight=.2,
                  transformer_branch_qkv='cross_depth', classification_mode='hierarchical',
                  class_weights=[3., 3., 1.])
    with patch.object(m, 'write_epoch_history_files', side_effect=OSError('interrupted')):
        with pytest.raises(OSError, match='interrupted'):
            m.train_loso_fold(**common)
    with pytest.raises(ValueError, match='classification_mode'):
        m.train_loso_fold(**{**common, 'classification_mode': 'flat'})
    metrics_path = m.train_loso_fold(**common)
    metrics = json.loads(metrics_path.read_text())
    assert metrics['classification_mode'] == 'hierarchical'
    assert metrics['resumed_from_epoch'] == 1
    best = m.load_torch_checkpoint(metrics_path.parent / 'best_model.pt', torch.device('cpu'))
    assert best['classification_mode'] == 'hierarchical'
    match = dict(output_dir=common['output_dir'], subject_id=1, input_domain='time_fft',
                 transformer_branches=3, transformer_depths=(3, 2, 1),
                 transformer_branch_fusion='loss_softmax', branch_loss_aux_weight=.2,
                 transformer_branch_qkv='cross_depth')
    assert batch.fold_is_complete(**match, classification_mode='hierarchical')
    assert not batch.fold_is_complete(**match)


def test_batch_forwards_mode(tmp_path):
    import train_activity_loso_batch as batch
    with patch.object(batch, 'train_loso_fold', return_value=tmp_path / 'metrics.json') as train:
        batch.run_loso_batch(
            subject_ids=[1], dataset_root=tmp_path, epochs=2, batch_size=3,
            lr=.0002, device='cpu', output_dir=tmp_path, skip_existing=False,
            input_domain='time_fft', classification_mode='hierarchical',
        )
    assert train.call_args.kwargs['classification_mode'] == 'hierarchical'
