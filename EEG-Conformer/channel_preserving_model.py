"""Dual-domain Conformer with electrode-preserving early blocks and two skips.

C is the EEG electrode axis, N is the patch axis, and D is the learned feature
axis. Attention runs over N independently for each electrode in blocks 1--5.
The same early spatial feature is added after every early block. After a single
C->1 convolution, the same fused feature is added after every block 6--10.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from train_activity_loso import (
    DEFAULT_DROPOUT, DEFAULT_EMB_SIZE, DEFAULT_NUM_HEADS,
    FusionClassificationHead, TransformerEncoderBlock, compute_n_patches,
)

ARCHITECTURE = "channelpreserving"
ARCHITECTURE_VERSION = 1
DEPTH = 10
SPLIT_AFTER_BLOCK = 5


class ElectrodePreservingPatchEmbedding(nn.Module):
    """(B,1,C,L) -> (B,C,N,D), retaining every electrode in the stem."""

    def __init__(self, n_channels: int, emb_size: int, dropout: float) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=1),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        # Mix learned features at each electrode/patch, not EEG electrodes.
        self.projection = nn.Conv2d(40, emb_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_channels:
            raise ValueError(f"Expected (B,1,{self.n_channels},L), got {tuple(x.shape)}")
        return self.projection(self.shallownet(x)).permute(0, 2, 3, 1).contiguous()


class TwoStageElectrodeEncoder(nn.Module):
    """Two stages of ordinary Blocks, each reusing one fixed skip feature."""

    def __init__(self, n_channels: int, emb_size: int, num_heads: int,
                 dropout: float, activation_checkpointing: bool = True, *,
                 depth: int = DEPTH, split_after_block: int = SPLIT_AFTER_BLOCK) -> None:
        super().__init__()
        if n_channels < 1 or emb_size < 1 or num_heads < 1 or emb_size % num_heads:
            raise ValueError("Positive channel/feature/head counts and emb_size divisible by heads required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if (type(depth) is not int or type(split_after_block) is not int
                or not 0 < split_after_block < depth):
            raise ValueError("depth/split_after_block must be integers with 0 < split < depth")
        self.n_channels = n_channels
        self.emb_size = emb_size
        self.depth = depth
        self.split_after_block = split_after_block
        self.activation_checkpointing = bool(activation_checkpointing)
        # In this layout Conv2d's channel dimension IS the EEG electrode axis.
        # Each (patch,feature) position mixes C electrodes with the same weights.
        self.early_channel_conv = nn.Conv2d(n_channels, n_channels, 1)
        self.fuse_channels = nn.Conv2d(n_channels, 1, 1)
        self.early_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, drop_p=dropout, forward_drop_p=dropout)
            for _ in range(split_after_block)
        ])
        self.late_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, drop_p=dropout, forward_drop_p=dropout)
            for _ in range(depth - split_after_block)
        ])

    def _block(self, block: nn.Module, x: Tensor) -> Tensor:
        # Recompute attention activations on backward to retain the requested
        # batch size despite the 21 independent early sequences per window.
        # RNG preservation keeps dropout identical to an uncheckpointed pass.
        if self.activation_checkpointing and self.training and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(block, x, use_reentrant=True, preserve_rng_state=True)
        return block(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.n_channels or x.shape[-1] != self.emb_size:
            raise ValueError(f"Expected (B,{self.n_channels},N,{self.emb_size}), got {tuple(x.shape)}")
        batch, channels, patches, features = x.shape
        early_skip = self.early_channel_conv(x)
        h = x
        for block in self.early_blocks:
            # A single set of parameters is shared across the C electrodes.
            h = self._block(block, h.reshape(batch * channels, patches, features))
            h = h.reshape(batch, channels, patches, features) + early_skip
        fused_skip = self.fuse_channels(h).squeeze(1)  # (B,N,D), computed ONCE.
        h = fused_skip
        for block in self.late_blocks:
            h = self._block(block, h) + fused_skip
        return h


class ElectrodePreservingBranch(nn.Module):
    def __init__(self, n_channels: int, n_times: int, emb_size: int, num_heads: int,
                 stem_dropout: float, encoder_dropout: float, activation_checkpointing: bool) -> None:
        super().__init__()
        self.n_times = n_times
        self.n_patches = compute_n_patches(n_times)
        self.flat_size = self.n_patches * emb_size
        self.patch_embedding = ElectrodePreservingPatchEmbedding(n_channels, emb_size, stem_dropout)
        self.encoder = TwoStageElectrodeEncoder(n_channels, emb_size, num_heads, encoder_dropout,
                                               activation_checkpointing)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n_times:
            raise ValueError(f"Expected {self.n_times} input points, got {x.shape[-1]}")
        return self.encoder(self.patch_embedding(x)).flatten(1)


class ChannelPreservingDualConformer(nn.Module):
    """Independent Time/FFT branches, concatenated into one baseline MLP head."""

    def __init__(self, n_channels: int, time_n_times: int, fft_n_times: int,
                 n_classes: int = 3, emb_size: int = DEFAULT_EMB_SIZE, depth: int = DEPTH,
                 num_heads: int = DEFAULT_NUM_HEADS, dropout: float = DEFAULT_DROPOUT,
                 transformer_encoder_dropout: float = DEFAULT_DROPOUT,
                 activation_checkpointing: bool = True) -> None:
        super().__init__()
        if depth != DEPTH:
            raise ValueError("The approved two-stage architecture requires depth=10 (5+5)")
        if fft_n_times != time_n_times // 2 + 1:
            raise ValueError("FFT length must match the real FFT of the time-domain window")
        if not 0 <= dropout < 1:
            raise ValueError("stem dropout must be in [0,1)")
        self.classification_mode = "flat"
        self.uses_branch_loss_fusion = False
        common = dict(n_channels=n_channels, emb_size=emb_size, num_heads=num_heads,
                      stem_dropout=dropout, encoder_dropout=transformer_encoder_dropout,
                      activation_checkpointing=activation_checkpointing)
        self.time_branch = ElectrodePreservingBranch(n_times=time_n_times, **common)
        self.fft_branch = ElectrodePreservingBranch(n_times=fft_n_times, **common)
        self.cls_head = FusionClassificationHead(
            self.time_branch.flat_size + self.fft_branch.flat_size, n_classes)
        self.architecture_config = {
            "name": ARCHITECTURE, "version": ARCHITECTURE_VERSION,
            "n_channels": n_channels, "time_n_times": time_n_times, "fft_n_times": fft_n_times,
            "n_classes": n_classes, "emb_size": emb_size, "depth": depth,
            "num_heads": num_heads, "dropout": dropout,
            "transformer_encoder_dropout": transformer_encoder_dropout,
            "activation_checkpointing": bool(activation_checkpointing),
            "split_after_block": SPLIT_AFTER_BLOCK,
            "electrode_weights_shared": True, "domains_independent": True,
            "channel_conv_kernel": [1, 1], "channel_conv_groups": 1,
            "early_skip_targets": [1, 2, 3, 4, 5], "late_skip_targets": [6, 7, 8, 9, 10],
            "skip_operation": "add_fixed_stage_feature", "class_head": [256, 32, n_classes],
            "stem_temporal_kernel": 25, "pool_kernel": 75, "pool_stride": 15,
            "ffn_expansion": 4, "head_dropout": [0.5, 0.3],
        }

    def forward(self, x_time: Tensor, x_fft: Tensor) -> tuple[Tensor, Tensor]:
        if x_time.shape[0] != x_fft.shape[0]:
            raise ValueError("Time and FFT batches must be aligned")
        features = torch.cat((self.time_branch(x_time), self.fft_branch(x_fft)), dim=1)
        return self.cls_head(features)


def restore_from_config(config: dict) -> ChannelPreservingDualConformer:
    """Require full architecture metadata, including both fixed skip sources."""
    fields = ("n_channels", "time_n_times", "fft_n_times", "n_classes", "emb_size", "depth",
              "num_heads", "dropout", "transformer_encoder_dropout", "activation_checkpointing")
    if any(name not in config for name in fields):
        raise ValueError("Incomplete channel-preserving architecture_config")
    model = ChannelPreservingDualConformer(**{name: config[name] for name in fields})
    if model.architecture_config != config:
        raise ValueError("Channel-preserving architecture metadata differs from this implementation")
    return model
