"""D model: three configurable depth encoders per domain, with two-stage skips.

Time and FFT each embed their input once, then run three independent encoders.
Cross-depth attention exchanges the three completed outputs at each patch.
Same-depth time/FFT features feed three heads, using the legacy Branch Loss.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from channel_preserving_model import ElectrodePreservingPatchEmbedding, TwoStageElectrodeEncoder
from train_activity_loso import (
    DEFAULT_DROPOUT, DEFAULT_EMB_SIZE, DEFAULT_NUM_HEADS,
    DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT, DEFAULT_TRANSFORMER_BRANCH_QKV_RES_SCALE,
    CrossDepthQKVResidual, FusionClassificationHead, compute_n_patches,
)

ARCHITECTURE = "channelpreservingmultidepth"
DEPTHS = (2, 5, 8)
SPLITS = tuple(depth // 2 for depth in DEPTHS)
AUX_WEIGHT = 0.2


def resolve_depth_config(transformer_depths=DEPTHS, split_after_blocks=None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Keep three branches; default odd depths to floor(depth/2) early blocks."""
    if not isinstance(transformer_depths, (list, tuple)) or len(transformer_depths) != 3:
        raise ValueError("transformer_depths must contain exactly three integer depths")
    depths = tuple(transformer_depths)
    if any(type(depth) is not int or depth < 2 for depth in depths):
        raise ValueError("Each depth must be an integer >= 2 for two nonempty stages")
    if split_after_blocks is None:
        splits = tuple(depth // 2 for depth in depths)
    elif isinstance(split_after_blocks, (list, tuple)) and len(split_after_blocks) == 3:
        splits = tuple(split_after_blocks)
    else:
        raise ValueError("split_after_blocks must contain exactly three integer positions")
    if any(type(split) is not int or not 0 < split < depth for depth, split in zip(depths, splits)):
        raise ValueError("Each split_after_blocks position must be an integer with 0 < split < depth")
    return depths, splits


class MultiDepthElectrodeBranch(nn.Module):
    """One domain-specific stem, three independently parameterized encoders."""

    def __init__(self, n_channels: int, n_times: int, emb_size: int, num_heads: int,
                 dropout: float, encoder_dropout: float, activation_checkpointing: bool,
                 transformer_depths=DEPTHS, split_after_blocks=None) -> None:
        super().__init__()
        depths, splits = resolve_depth_config(transformer_depths, split_after_blocks)
        self.n_times = n_times
        self.n_patches = compute_n_patches(n_times)
        self.flat_size = self.n_patches * emb_size
        self.patch_embedding = ElectrodePreservingPatchEmbedding(n_channels, emb_size, dropout)
        self.encoders = nn.ModuleList([
            TwoStageElectrodeEncoder(n_channels, emb_size, num_heads, encoder_dropout,
                                    activation_checkpointing, depth=depth, split_after_block=split)
            for depth, split in zip(depths, splits)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if x.shape[-1] != self.n_times:
            raise ValueError(f"Expected {self.n_times} input points, got {x.shape[-1]}")
        embedded = self.patch_embedding(x)
        return tuple(encoder(embedded) for encoder in self.encoders)


class MultiDepthChannelPreservingConformer(nn.Module):
    def __init__(self, n_channels: int, time_n_times: int, fft_n_times: int,
                 n_classes: int = 3, emb_size: int = DEFAULT_EMB_SIZE,
                 num_heads: int = DEFAULT_NUM_HEADS, dropout: float = DEFAULT_DROPOUT,
                 transformer_encoder_dropout: float = DEFAULT_DROPOUT,
                 transformer_branch_qkv_dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
                 activation_checkpointing: bool = True,
                 transformer_depths=DEPTHS, split_after_blocks=None) -> None:
        super().__init__()
        depths, splits = resolve_depth_config(transformer_depths, split_after_blocks)
        if fft_n_times != time_n_times // 2 + 1:
            raise ValueError("FFT length must match the real FFT of the time-domain window")
        if not 0 <= dropout < 1:
            raise ValueError("stem dropout must be in [0,1)")
        self.classification_mode = "flat"
        self.uses_branch_loss_fusion = True
        self.uses_cross_depth_qkv = True
        self.transformer_branch_depths = depths
        self.split_after_blocks = splits
        self.transformer_branches = len(depths)
        self.branch_loss_aux_weight = AUX_WEIGHT
        common = dict(n_channels=n_channels, emb_size=emb_size, num_heads=num_heads,
                      dropout=dropout, encoder_dropout=transformer_encoder_dropout,
                      activation_checkpointing=activation_checkpointing,
                      transformer_depths=depths, split_after_blocks=splits)
        self.time_branch = MultiDepthElectrodeBranch(n_times=time_n_times, **common)
        self.fft_branch = MultiDepthElectrodeBranch(n_times=fft_n_times, **common)
        attention = dict(n_branches=3, emb_size=emb_size, num_heads=num_heads,
                         dropout=transformer_branch_qkv_dropout)
        self.time_cross_depth_qkv = CrossDepthQKVResidual(**attention)
        self.fft_cross_depth_qkv = CrossDepthQKVResidual(**attention)
        fused_size = self.time_branch.flat_size + self.fft_branch.flat_size
        self.branch_cls_heads = nn.ModuleList([
            FusionClassificationHead(fused_size, n_classes) for _ in depths
        ])
        self.branch_loss_weight_logits = nn.Parameter(torch.zeros(3))
        self.architecture_config = {
            "name": ARCHITECTURE, "version": 1, "variant": "D",
            "n_channels": n_channels, "time_n_times": time_n_times, "fft_n_times": fft_n_times,
            "n_classes": n_classes, "emb_size": emb_size, "num_heads": num_heads,
            "dropout": dropout, "transformer_encoder_dropout": transformer_encoder_dropout,
            "activation_checkpointing": bool(activation_checkpointing),
            "depths": list(depths), "split_after_blocks": list(splits),
            "early_skip_targets": [list(range(1, split + 1)) for split in splits],
            "late_skip_targets": [list(range(split + 1, depth + 1)) for depth, split in zip(depths, splits)],
            "skip_operation": "add_fixed_stage_feature", "channel_conv_kernel": [1, 1],
            "channel_conv_groups": 1, "electrode_weights_shared": True,
            "domains_independent": True, "depth_encoder_parameters_independent": True,
            "channel_convolutions_independent_by_depth": True, "shared_stem_within_domain": True,
            "transformer_branch_qkv": "cross_depth", "transformer_branch_qkv_dropout": transformer_branch_qkv_dropout,
            "transformer_branch_qkv_res_scale": DEFAULT_TRANSFORMER_BRANCH_QKV_RES_SCALE,
            "transformer_branch_fusion": "loss_softmax", "branch_loss_aux_weight": AUX_WEIGHT,
            "loss_rule": "sum(softmax(alpha)*branch_CE) + 0.2*mean(branch_CE)",
            "prediction_rule": "sum(softmax(alpha)*branch_logits)",
            "class_head": [256, 32, n_classes], "classification_heads": 3,
            "stem_temporal_kernel": 25, "pool_kernel": 75, "pool_stride": 15,
            "ffn_expansion": 4, "head_dropout": [0.5, 0.3],
        }

    def normalized_branch_loss_weights(self) -> Tensor:
        return self.branch_loss_weight_logits.softmax(dim=0)

    def transformer_weight_metadata(self) -> dict:
        return {
            "transformer_branch_loss_weights": self.normalized_branch_loss_weights().detach().cpu().tolist(),
            "time_transformer_branch_qkv_gamma": float(self.time_cross_depth_qkv.gamma.detach().cpu()),
            "fft_transformer_branch_qkv_gamma": float(self.fft_cross_depth_qkv.gamma.detach().cpu()),
        }

    def paired_transformer_features(self, x_time: Tensor, x_fft: Tensor) -> tuple[Tensor, ...]:
        if x_time.shape[0] != x_fft.shape[0]:
            raise ValueError("Time and FFT batches must be aligned")
        time = self.time_cross_depth_qkv(self.time_branch(x_time))
        fft = self.fft_cross_depth_qkv(self.fft_branch(x_fft))
        return tuple(torch.cat((t.flatten(1), f.flatten(1)), dim=1) for t, f in zip(time, fft))

    def forward_with_branch_logits(self, x_time: Tensor, x_fft: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        features = self.paired_transformer_features(x_time, x_fft)
        logits = tuple(head(value)[1] for head, value in zip(self.branch_cls_heads, features))
        weights = self.normalized_branch_loss_weights().view(-1, 1, 1)
        return ((weights * torch.stack(features)).sum(dim=0),
                (weights * torch.stack(logits)).sum(dim=0), logits)

    def forward(self, x_time: Tensor, x_fft: Tensor) -> tuple[Tensor, Tensor]:
        features, logits, _ = self.forward_with_branch_logits(x_time, x_fft)
        return features, logits


def restore_from_config(config: dict) -> MultiDepthChannelPreservingConformer:
    fields = ("n_channels", "time_n_times", "fft_n_times", "n_classes", "emb_size", "num_heads",
              "dropout", "transformer_encoder_dropout", "transformer_branch_qkv_dropout", "activation_checkpointing")
    if any(name not in config for name in (*fields, "depths", "split_after_blocks")):
        raise ValueError("Incomplete multi-depth architecture_config")
    model = MultiDepthChannelPreservingConformer(**{name: config[name] for name in fields},
        transformer_depths=config["depths"], split_after_blocks=config["split_after_blocks"])
    if model.architecture_config != config:
        raise ValueError("Multi-depth architecture metadata differs from this implementation")
    return model
