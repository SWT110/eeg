from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from channel_preserving_model import (
    ChannelPreservingDualConformer, TwoStageElectrodeEncoder, restore_from_config,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_both_skips_reuse_one_fixed_feature_and_target_every_block():
    encoder = TwoStageElectrodeEncoder(3, 10, 5, 0, False)
    encoder.early_blocks = nn.ModuleList([nn.Identity() for _ in range(5)])
    encoder.late_blocks = nn.ModuleList([nn.Identity() for _ in range(5)])
    x = torch.arange(2 * 3 * 4 * 10).reshape(2, 3, 4, 10).float()
    early_inputs, late_inputs, convolutions = [], [], []
    for block in encoder.early_blocks:
        block.register_forward_pre_hook(lambda _m, args: early_inputs.append(args[0].clone()))
    for block in encoder.late_blocks:
        block.register_forward_pre_hook(lambda _m, args: late_inputs.append(args[0].clone()))
    encoder.early_channel_conv.register_forward_hook(lambda *_: convolutions.append("early"))
    encoder.fuse_channels.register_forward_hook(lambda *_: convolutions.append("fuse"))
    with torch.no_grad():
        # Each output electrode reads all electrodes with distinct weights.
        encoder.early_channel_conv.weight.copy_(torch.tensor([
            [1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
        ])[:, :, None, None])
        encoder.early_channel_conv.bias.zero_()
        encoder.fuse_channels.weight.copy_(torch.tensor([1., 2., 4.])[None, :, None, None])
        encoder.fuse_channels.bias.zero_()
        skip = torch.einsum("oc,bcnf->bonf", encoder.early_channel_conv.weight[:, :, 0, 0], x)
        fused = torch.einsum("c,bcnf->bnf", torch.tensor([1., 2., 4.]), x + 5 * skip)
        actual = encoder(x)
    torch.testing.assert_close(actual, 6 * fused)
    assert convolutions == ["early", "fuse"]
    for i, value in enumerate(early_inputs):
        assert value.shape == (2 * 3, 4, 10)
        torch.testing.assert_close(value.reshape_as(x), x + i * skip)
    for i, value in enumerate(late_inputs):
        assert value.shape == (2, 4, 10)
        torch.testing.assert_close(value, (i + 1) * fused)


def test_original_window_dimensions_independent_domains_and_strict_reload():
    torch.manual_seed(42)
    model = ChannelPreservingDualConformer(21, 1920, 961).eval()
    time, fft = torch.randn(1, 1, 21, 1920), torch.randn(1, 1, 21, 961)
    shapes = []
    for branch in (model.time_branch, model.fft_branch):
        branch.patch_embedding.register_forward_hook(lambda _m, _a, out: shapes.append(tuple(out.shape)))
    with torch.no_grad():
        features, logits = model(time, fft)
    assert shapes == [(1, 21, 122, 40), (1, 21, 58, 40)]
    assert features.shape == (1, 7200) and logits.shape == (1, 3)
    assert not ({p.data_ptr() for p in model.time_branch.parameters()}
                & {p.data_ptr() for p in model.fft_branch.parameters()})
    restored = restore_from_config(model.architecture_config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(time, fft)[1], logits, rtol=0, atol=0)
    bad = copy.deepcopy(model.architecture_config)
    bad["late_skip_targets"] = [10]
    with pytest.raises(ValueError, match="metadata differs"):
        restore_from_config(bad)


def test_checkpointing_preserves_dropout_outputs_and_gradients():
    torch.manual_seed(4)
    direct = ChannelPreservingDualConformer(3, 256, 129, activation_checkpointing=False).train()
    recomputed = ChannelPreservingDualConformer(3, 256, 129, activation_checkpointing=True).train()
    recomputed.load_state_dict(direct.state_dict())
    time, fft = torch.randn(2, 1, 3, 256), torch.randn(2, 1, 3, 129)
    outputs = []
    for model in (direct, recomputed):
        torch.manual_seed(73)
        logits = model(time, fft)[1]
        nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        outputs.append(logits.detach())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for p, q in zip(direct.parameters(), recomputed.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("changes", [{"depth": 6}, {"num_heads": 6}, {"fft_n_times": 130}])
def test_invalid_architecture_settings_rejected(changes):
    config = dict(n_channels=3, time_n_times=256, fft_n_times=129)
    config.update(changes)
    with pytest.raises(ValueError):
        ChannelPreservingDualConformer(**config)
