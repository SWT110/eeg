from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from channel_preserving_model import TwoStageElectrodeEncoder
from channel_preserving_multidepth_model import (
    DEPTHS, SPLITS, MultiDepthChannelPreservingConformer, restore_from_config,
)
from train_activity_loso import compute_model_batch_loss


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("depth,split", [(2, 1), (5, 2), (8, 4), (4, 2), (10, 5), (16, 8), (5, 3)])
def test_every_stage_output_gets_its_own_fixed_skip_once(depth, split):
    encoder = TwoStageElectrodeEncoder(3, 10, 5, 0, False, depth=depth, split_after_block=split)
    assert len(encoder.early_blocks) == split and len(encoder.late_blocks) == depth - split
    encoder.early_blocks = nn.ModuleList([nn.Identity() for _ in range(split)])
    encoder.late_blocks = nn.ModuleList([nn.Identity() for _ in range(depth - split)])
    x = torch.arange(2 * 3 * 4 * 10).reshape(2, 3, 4, 10).float()
    early_inputs, late_inputs, convolutions = [], [], []
    for block in encoder.early_blocks:
        block.register_forward_pre_hook(lambda _m, args: early_inputs.append(args[0].clone()))
    for block in encoder.late_blocks:
        block.register_forward_pre_hook(lambda _m, args: late_inputs.append(args[0].clone()))
    encoder.early_channel_conv.register_forward_hook(lambda *_: convolutions.append("early"))
    encoder.fuse_channels.register_forward_hook(lambda *_: convolutions.append("fuse"))
    with torch.no_grad():
        encoder.early_channel_conv.weight.copy_(torch.tensor([
            [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
        ])[:, :, None, None])
        encoder.early_channel_conv.bias.zero_()
        encoder.fuse_channels.weight.copy_(torch.tensor([1., 2., 4.])[None, :, None, None])
        encoder.fuse_channels.bias.zero_()
        early = encoder.early_channel_conv(x)
        fused = encoder.fuse_channels(x + split * early).squeeze(1)
        convolutions.clear()
        actual = encoder(x)
    torch.testing.assert_close(actual, (depth - split + 1) * fused)
    assert convolutions == ["early", "fuse"]
    for index, value in enumerate(early_inputs):
        assert value.shape == (6, 4, 10)
        torch.testing.assert_close(value.reshape_as(x), x + index * early)
    for index, value in enumerate(late_inputs):
        torch.testing.assert_close(value, (index + 1) * fused)


@pytest.mark.parametrize("depths,splits", [(DEPTHS, SPLITS), ((4, 10, 16), (2, 5, 8)), ((2, 5, 8), (1, 3, 6))])
def test_original_shape_single_stem_per_domain_independent_encoders_heads_and_exact_fusion(depths, splits):
    model = MultiDepthChannelPreservingConformer(21, 1920, 961,
        transformer_depths=depths, split_after_blocks=splits).eval()
    shapes, head_inputs = [], []
    parameter_sets = []
    for branch in (model.time_branch, model.fft_branch):
        branch.patch_embedding.register_forward_hook(lambda _m, _a, out: shapes.append(tuple(out.shape)))
        assert [e.depth for e in branch.encoders] == list(depths)
        assert [e.split_after_block for e in branch.encoders] == list(splits)
        for encoder in branch.encoders:
            parameter_sets.append({p.data_ptr() for p in encoder.parameters()})
    for head in model.branch_cls_heads:
        head.register_forward_pre_hook(lambda _m, args: head_inputs.append(tuple(args[0].shape)))
        parameter_sets.append({p.data_ptr() for p in head.parameters()})
    for i, group in enumerate(parameter_sets):
        assert not any(group & other for other in parameter_sets[i + 1:])
    assert not ({p.data_ptr() for p in model.time_branch.parameters()}
                & {p.data_ptr() for p in model.fft_branch.parameters()})
    with torch.no_grad():
        model.branch_loss_weight_logits.copy_(torch.tensor([1., 0., -1.]))
        features, logits, branches = model.forward_with_branch_logits(
            torch.randn(1, 1, 21, 1920), torch.randn(1, 1, 21, 961))
    assert shapes == [(1, 21, 122, 40), (1, 21, 58, 40)]
    assert head_inputs == [(1, 7200)] * 3
    assert features.shape == (1, 7200) and logits.shape == (1, 3)
    weights = torch.tensor([1., 0., -1.]).softmax(0)
    torch.testing.assert_close(logits, sum(w * z for w, z in zip(weights, branches)))
    metadata = model.transformer_weight_metadata()
    assert metadata["transformer_branch_loss_weights"] == weights.tolist()
    assert model.architecture_config["early_skip_targets"] == [list(range(1, s + 1)) for s in splits]
    assert model.architecture_config["late_skip_targets"] == [list(range(s + 1, d + 1)) for d, s in zip(depths, splits)]


def test_class_weighted_branch_loss_matches_legacy_formula_and_all_parameters_receive_gradients():
    model = MultiDepthChannelPreservingConformer(3, 256, 129, emb_size=10).train()
    _, logits, branches = model.forward_with_branch_logits(torch.randn(2, 1, 3, 256), torch.randn(2, 1, 3, 129))
    labels = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([3., 2., 1.]))
    loss, branch_losses = compute_model_batch_loss(model, logits, labels, criterion, branches)
    expected_losses = torch.stack([criterion(z, labels) for z in branches])
    expected = (model.normalized_branch_loss_weights() * expected_losses).sum() + .2 * expected_losses.mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(branch_losses), expected_losses, rtol=0, atol=0)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.branch_loss_weight_logits.grad.abs().sum() > 0


def test_cross_depth_attention_connects_one_classifier_to_all_depth_encoders():
    model = MultiDepthChannelPreservingConformer(3, 256, 129, emb_size=10).eval()
    _, _, branches = model.forward_with_branch_logits(torch.randn(2, 1, 3, 256), torch.randn(2, 1, 3, 129))
    nn.functional.cross_entropy(branches[0], torch.tensor([0, 1])).backward()
    for domain in (model.time_branch, model.fft_branch):
        for encoder in domain.encoders:
            gradients = [p.grad for p in encoder.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(g.abs().sum() for g in gradients) > 0


def test_dropout_checkpointing_preserves_logits_gradients_and_strict_reload():
    torch.manual_seed(8)
    direct = MultiDepthChannelPreservingConformer(3, 256, 129, emb_size=10, activation_checkpointing=False).train()
    recomputed = MultiDepthChannelPreservingConformer(3, 256, 129, emb_size=10, activation_checkpointing=True).train()
    recomputed.load_state_dict(direct.state_dict())
    time, fft = torch.randn(2, 1, 3, 256), torch.randn(2, 1, 3, 129)
    outputs = []
    for model in (direct, recomputed):
        torch.manual_seed(73)
        _, logits, branches = model.forward_with_branch_logits(time, fft)
        loss, _ = compute_model_batch_loss(model, logits, torch.tensor([0, 2]), nn.CrossEntropyLoss(), branches)
        loss.backward()
        outputs.append(logits.detach())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for p, q in zip(direct.parameters(), recomputed.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-6)
    restored = restore_from_config(direct.architecture_config)
    restored.load_state_dict(direct.state_dict(), strict=True)
    direct.eval(), restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(direct(time, fft)[1], restored(time, fft)[1], rtol=0, atol=0)
    bad = copy.deepcopy(direct.architecture_config)
    bad["late_skip_targets"] = [[2], [5], [8]]
    with pytest.raises(ValueError, match="metadata differs"):
        restore_from_config(bad)


def test_old_depths_config_restores_without_using_new_defaults():
    original = MultiDepthChannelPreservingConformer(3, 256, 129,
        transformer_depths=(4, 10, 16), split_after_blocks=(2, 5, 8)).eval()
    restored = restore_from_config(copy.deepcopy(original.architecture_config))
    restored.load_state_dict(original.state_dict(), strict=True)
    restored.eval()
    assert restored.transformer_branch_depths == (4, 10, 16) and restored.split_after_blocks == (2, 5, 8)
    time, fft = torch.randn(1, 1, 3, 256), torch.randn(1, 1, 3, 129)
    with torch.no_grad():
        torch.testing.assert_close(original(time, fft)[1], restored(time, fft)[1], rtol=0, atol=0)


@pytest.mark.parametrize("changes", [
    {"transformer_depths": (1, 5, 8)}, {"transformer_depths": (2, 5)},
    {"transformer_depths": (2, 5., 8)}, {"transformer_depths": (True, 5, 8)},
    {"split_after_blocks": (1, 2)}, {"split_after_blocks": (0, 2, 4)},
    {"split_after_blocks": (1, 5, 4)}, {"split_after_blocks": (1, 2.5, 4)},
])
def test_invalid_branch_or_stage_config_rejected(changes):
    with pytest.raises(ValueError):
        MultiDepthChannelPreservingConformer(3, 256, 129, **changes)
