"""Reproducible external EEG baselines for the activity LOSO experiment.

The implementations in this file are plain-PyTorch adaptations of the official
TCFormer repository (https://github.com/Altaheri/TCFormer, commit
701a361640708ef1cc07cf22660d47edc6d9bf3c).  That repository provides unified
implementations of EEGNet, ShallowNet, ATCNet, and TCFormer.  The adaptation
removes PyTorch Lightning and einops dependencies, accepts this project's
``[batch, 1, channels, time]`` tensors, and returns ``(features, logits)`` as
expected by ``train_activity_loso.py``.

Copyright (c) 2025 Hamdi Altaheri.  Used under the MIT License; the full notice
is retained in THIRD_PARTY_NOTICES.md.  Scientific citations and architecture
defaults are exposed through ``COMPARISON_MODEL_METADATA``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


VALID_COMPARISON_MODELS = (
    "eegnet",
    "shallowconvnet",
    "atcnet",
    "tcformer",
)

_MODEL_ALIASES = {
    "eegnet": "eegnet",
    "shallow": "shallowconvnet",
    "shallownet": "shallowconvnet",
    "shallowconvnet": "shallowconvnet",
    "atc": "atcnet",
    "atcnet": "atcnet",
    "tcformer": "tcformer",
}


COMPARISON_MODEL_METADATA: dict[str, dict[str, Any]] = {
    "eegnet": {
        "display_name": "EEGNet",
        "year": 2018,
        "doi": "10.1088/1741-2552/aace8c",
        "paper": "EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces",
        "role": "classic compact CNN baseline",
    },
    "shallowconvnet": {
        "display_name": "ShallowConvNet",
        "year": 2017,
        "doi": "10.1002/hbm.23730",
        "paper": "Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization",
        "role": "classic filter-bank-inspired CNN baseline",
    },
    "atcnet": {
        "display_name": "ATCNet",
        "year": 2023,
        "doi": "10.1109/TII.2022.3197419",
        "paper": "Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification",
        "role": "attention plus temporal-convolution baseline",
    },
    "tcformer": {
        "display_name": "TCFormer",
        "year": 2025,
        "doi": "10.1038/s41598-025-16219-7",
        "paper": "Temporal convolutional transformer for EEG based motor imagery decoding",
        "role": "recent multi-kernel Transformer-TCN baseline",
    },
}

IMPLEMENTATION_PROVENANCE = {
    "repository": "https://github.com/Altaheri/TCFormer",
    "commit": "701a361640708ef1cc07cf22660d47edc6d9bf3c",
    "license": "MIT",
    "adaptation": "plain PyTorch; project input/output adapter; no Lightning/einops",
}


def normalize_comparison_model_name(raw: str) -> str:
    value = str(raw).strip().lower().replace("-", "").replace("_", "")
    resolved = _MODEL_ALIASES.get(value)
    if resolved is None:
        allowed = ", ".join(VALID_COMPARISON_MODELS)
        raise ValueError(f"Unknown comparison model {raw!r}; expected one of: {allowed}")
    return resolved


def _glorot_weight_zero_bias(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args: Any, max_norm: float | None = None, **kwargs: Any) -> None:
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(
                    torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
                )
        return super().forward(x)


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args: Any, max_norm: float | None = None, **kwargs: Any) -> None:
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(
                    torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
                )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args: Any, max_norm: float | None = None, **kwargs: Any) -> None:
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(
                    torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
                )
        return super().forward(x)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


def _squeeze_project_input(x: Tensor) -> Tensor:
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim != 3:
        raise ValueError(
            "Comparison models expect [batch, 1, channels, time] or "
            f"[batch, channels, time], got {tuple(x.shape)}"
        )
    return x


class ComparisonInputAdapter(nn.Module):
    """Expose a uniform project-facing input/output contract."""

    def __init__(
        self,
        core: nn.Module,
        architecture_name: str,
        architecture_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.core = core
        self.architecture_name = architecture_name
        self.architecture_config = architecture_config

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.core(_squeeze_project_input(x))


class EEGNetCore(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        input_window_samples: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 32,
        dropout: float = 0.25,
        pool_time_length: int = 4,
        pool_time_stride: int = 4,
        separable_kernel_length: int = 16,
    ) -> None:
        super().__init__()
        if f2 != f1 * depth_multiplier:
            raise ValueError("This faithful EEGNet configuration requires f2 == f1 * D")

        self.temporal = nn.Conv2d(
            1,
            f1,
            (1, kernel_length),
            bias=False,
            padding=(0, kernel_length // 2),
        )
        self.temporal_bn = nn.BatchNorm2d(f1, momentum=0.01, eps=1e-3)
        self.spatial = Conv2dWithConstraint(
            f1,
            f1 * depth_multiplier,
            (n_channels, 1),
            bias=False,
            groups=f1,
            max_norm=1.0,
        )
        self.spatial_bn = nn.BatchNorm2d(f1 * depth_multiplier, momentum=0.01, eps=1e-3)
        self.pool1 = nn.AvgPool2d(
            (1, pool_time_length), stride=(1, pool_time_stride)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.separable = nn.Sequential(
            nn.Conv2d(
                f2,
                f2,
                (1, separable_kernel_length),
                bias=False,
                groups=f2,
                padding=(0, separable_kernel_length // 2),
            ),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        temporal_out = input_window_samples + 2 * (kernel_length // 2) - kernel_length + 1
        pool1_out = (temporal_out - pool_time_length) // pool_time_stride + 1
        separable_out = (
            pool1_out + 2 * (separable_kernel_length // 2) - separable_kernel_length + 1
        )
        classifier_width = (separable_out - 8) // 8 + 1
        if classifier_width < 1:
            raise ValueError("input_window_samples is too short for EEGNet")
        self.classifier = nn.Conv2d(f2, n_classes, (1, classifier_width))
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.unsqueeze(1)
        x = self.temporal_bn(self.temporal(x))
        x = F.elu(self.spatial_bn(self.spatial(x)))
        x = self.dropout1(self.pool1(x))
        x = self.separable(x)
        features = x.flatten(1)
        logits = self.classifier(x).flatten(1)
        return features, logits


class ShallowConvNetCore(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        input_window_samples: int,
        n_temporal_filters: int = 40,
        temporal_kernel_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            1, n_temporal_filters, (temporal_kernel_length, 1), bias=True
        )
        self.spatial = nn.Conv2d(
            n_temporal_filters,
            n_temporal_filters,
            (1, n_channels),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(n_temporal_filters)
        self.pool = nn.AvgPool2d(
            (pool_time_length, 1), (pool_time_stride, 1)
        )
        self.dropout = nn.Dropout(dropout)

        temporal_out = input_window_samples - temporal_kernel_length + 1
        classifier_height = (temporal_out - pool_time_length) // pool_time_stride + 1
        if classifier_height < 1:
            raise ValueError("input_window_samples is too short for ShallowConvNet")
        self.classifier = nn.Conv2d(
            n_temporal_filters, n_classes, (classifier_height, 1)
        )
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.bn(self.spatial(self.temporal(x)))
        x = torch.square(x)
        x = torch.log(torch.clamp(self.pool(x), min=1e-6))
        x = self.dropout(x)
        features = x.flatten(1)
        logits = self.classifier(x).flatten(1)
        return features, logits


class ATCConvBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        f1: int = 16,
        temporal_kernel_length: int = 64,
        pool_length: int = 8,
        depth_multiplier: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        d_model = f1 * depth_multiplier
        self.temporal = nn.Conv2d(
            1,
            f1,
            (1, temporal_kernel_length),
            padding=(0, temporal_kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = Conv2dWithConstraint(
            f1,
            d_model,
            (n_channels, 1),
            bias=False,
            groups=f1,
            max_norm=1.0,
        )
        self.bn2 = nn.BatchNorm2d(d_model)
        self.pool1 = nn.AvgPool2d((1, pool_length))
        self.dropout1 = nn.Dropout(dropout)
        self.temporal2 = nn.Conv2d(
            d_model, d_model, (1, 16), padding=(0, 8), bias=False
        )
        self.bn3 = nn.BatchNorm2d(d_model)
        self.pool2 = nn.AvgPool2d((1, 7))
        self.dropout2 = nn.Dropout(dropout)
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.temporal(x.unsqueeze(1)))
        x = F.elu(self.bn2(self.spatial(x)))
        x = self.dropout1(self.pool1(x))
        x = F.elu(self.bn3(self.temporal2(x)))
        return self.dropout2(self.pool2(x))


class ATCAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        key_dim: int = 8,
        n_heads: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        projection_size = n_heads * key_dim
        self.query = nn.Linear(d_model, projection_size)
        self.key = nn.Linear(d_model, projection_size)
        self.value = nn.Linear(d_model, projection_size)
        self.output = nn.Linear(projection_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        batch, length, _ = x.shape
        q = self.query(x).view(batch, length, self.n_heads, -1).permute(2, 0, 1, 3)
        k = self.key(x).view(batch, length, self.n_heads, -1).permute(2, 0, 1, 3)
        v = self.value(x).view(batch, length, self.n_heads, -1).permute(2, 0, 1, 3)
        attention = torch.einsum("hblk,hbtk->hblt", q, k) / math.sqrt(q.shape[-1])
        attention = attention.softmax(dim=-1)
        output = torch.einsum("hblt,hbtv->hblv", attention, v)
        output = output.permute(1, 2, 0, 3).contiguous().flatten(2)
        return residual + self.dropout(self.output(output))


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        n_filters: int,
        kernel_length: int,
        dilation: int,
        dropout: float,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            n_filters,
            n_filters,
            kernel_length,
            dilation=dilation,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(
            n_filters,
            n_filters,
            kernel_length,
            dilation=dilation,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.dropout2 = nn.Dropout(dropout)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dropout1(F.elu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.elu(self.bn2(self.conv2(x))))
        return F.elu(residual + x)


class TCN(nn.Module):
    def __init__(
        self,
        depth: int,
        kernel_length: int,
        n_filters: int,
        dropout: float,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TCNResidualBlock(
                n_filters,
                kernel_length,
                dilation=2**index,
                dropout=dropout,
                groups=groups,
            )
            for index in range(depth)
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ATCWindowHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_classes: int,
        key_dim: int,
        n_heads: int,
        attention_dropout: float,
        tcn_depth: int,
        tcn_kernel_length: int,
        tcn_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = ATCAttentionBlock(
            d_model, key_dim, n_heads, attention_dropout
        )
        self.tcn = TCN(
            tcn_depth, tcn_kernel_length, d_model, tcn_dropout
        )
        self.classifier = LinearWithConstraint(
            d_model, n_classes, max_norm=0.25
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x).transpose(1, 2)
        x = self.tcn(x)
        return self.classifier(x[:, :, -1])


class ATCNetCore(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        f1: int = 16,
        temporal_kernel_length: int = 64,
        pool_length: int = 8,
        depth_multiplier: int = 2,
        conv_dropout: float = 0.3,
        d_model: int = 32,
        key_dim: int = 8,
        n_heads: int = 2,
        attention_dropout: float = 0.5,
        tcn_depth: int = 2,
        tcn_kernel_length: int = 4,
        tcn_dropout: float = 0.3,
        n_windows: int = 5,
    ) -> None:
        super().__init__()
        if d_model != f1 * depth_multiplier:
            raise ValueError("ATCNet requires d_model == f1 * D")
        self.conv = ATCConvBlock(
            n_channels,
            f1,
            temporal_kernel_length,
            pool_length,
            depth_multiplier,
            conv_dropout,
        )
        self.windows = nn.ModuleList(
            ATCWindowHead(
                d_model,
                n_classes,
                key_dim,
                n_heads,
                attention_dropout,
                tcn_depth,
                tcn_kernel_length,
                tcn_dropout,
            )
            for _ in range(n_windows)
        )
        self.n_windows = n_windows

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.conv(x).squeeze(2).transpose(1, 2)
        sequence_length = features.shape[1]
        window_length = sequence_length - self.n_windows + 1
        if window_length < 1:
            raise ValueError(
                f"ATCNet feature sequence ({sequence_length}) is shorter than "
                f"n_windows ({self.n_windows})"
            )
        logits = torch.stack(
            [
                head(features[:, index : index + window_length, :])
                for index, head in enumerate(self.windows)
            ],
            dim=0,
        ).mean(dim=0)
        return features.flatten(1), logits


class ChannelGroupAttention(nn.Module):
    def __init__(
        self, in_channels: int, n_groups: int, reduction: int = 4
    ) -> None:
        super().__init__()
        if in_channels % n_groups != 0:
            raise ValueError("in_channels must be divisible by n_groups")
        reduced_channels = in_channels // reduction
        if reduced_channels % n_groups != 0:
            raise ValueError("in_channels/reduction must be divisible by n_groups")
        self.in_channels = in_channels
        self.n_groups = n_groups
        self.group_size = in_channels // n_groups
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            groups=n_groups,
            bias=False,
        )
        self.expand = nn.Conv2d(
            reduced_channels,
            n_groups,
            kernel_size=1,
            groups=n_groups,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channels}"
            )
        weights = torch.sigmoid(self.expand(F.relu(self.reduce(self.pool(x)))))
        grouped = x.reshape(
            batch, self.n_groups, self.group_size, height, width
        )
        return (grouped * weights.reshape(batch, self.n_groups, 1, 1, 1)).reshape_as(x)


class MultiKernelConvBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        temporal_kernel_lengths: tuple[int, ...] = (20, 32, 64),
        f1: int = 32,
        depth_multiplier: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        dropout: float = 0.4,
        d_group: int = 16,
        use_group_attention: bool = True,
    ) -> None:
        super().__init__()
        self.n_groups = len(temporal_kernel_lengths)
        self.d_model = d_group * self.n_groups
        self.temporal_convs = nn.ModuleList(
            nn.Sequential(
                nn.ConstantPad2d(
                    (
                        kernel // 2 - 1,
                        kernel // 2,
                        0,
                        0,
                    )
                    if kernel % 2 == 0
                    else (kernel // 2, kernel // 2, 0, 0),
                    0,
                ),
                nn.Conv2d(1, f1, (1, kernel), bias=False),
                nn.BatchNorm2d(f1),
            )
            for kernel in temporal_kernel_lengths
        )
        temporal_channels = f1 * self.n_groups
        spatial_channels = temporal_channels * depth_multiplier
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_channels,
                spatial_channels,
                (n_channels, 1),
                bias=False,
                groups=temporal_channels,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.dropout1 = nn.Dropout(dropout)
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(
                spatial_channels,
                self.d_model,
                (1, 1),
                bias=False,
                groups=self.n_groups,
            ),
            nn.BatchNorm2d(self.d_model),
        )
        self.temporal2 = nn.Sequential(
            nn.Conv2d(
                self.d_model,
                self.d_model,
                (1, 16),
                padding="same",
                bias=False,
                groups=self.n_groups,
            ),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
        )
        self.group_attention = (
            ChannelGroupAttention(self.d_model, self.n_groups)
            if self.n_groups > 1 and use_group_attention
            else None
        )
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.dropout2 = nn.Dropout(dropout)
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = torch.cat([conv(x) for conv in self.temporal_convs], dim=1)
        x = self.dropout1(self.pool1(self.spatial(x)))
        x = self.temporal2(self.channel_reduction(x))
        if self.group_attention is not None:
            x = x + self.group_attention(x)
        return self.dropout2(self.pool2(x)).squeeze(2)


def _build_rotary_cache(
    head_dim: int, sequence_length: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    theta = 1.0 / (
        10000
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    positions = torch.arange(
        sequence_length, device=device, dtype=torch.float32
    )
    frequencies = torch.outer(positions, theta)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos(), embedding.sin()


def _apply_rope(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor
) -> tuple[Tensor, Tensor]:
    def rotate(x: Tensor) -> Tensor:
        first, second = x[..., ::2], x[..., 1::2]
        return torch.stack((-second, first), dim=-1).flatten(-2)

    return (
        query * cosine + rotate(query) * sine,
        key * cosine + rotate(key) * sine,
    )


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        query_heads: int,
        key_value_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % query_heads != 0:
            raise ValueError("d_model must be divisible by query_heads")
        if query_heads % key_value_heads != 0:
            raise ValueError("query_heads must be a multiple of key_value_heads")
        self.query_heads = query_heads
        self.key_value_heads = key_value_heads
        self.head_dim = d_model // query_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key_value = nn.Linear(
            d_model, 2 * key_value_heads * self.head_dim, bias=False
        )
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        batch, length, channels = x.shape
        query = self.query(x).reshape(
            batch, length, self.query_heads, self.head_dim
        ).transpose(1, 2)
        key_value = self.key_value(x).reshape(
            batch, length, self.key_value_heads, 2, self.head_dim
        )
        key = key_value[..., 0, :].transpose(1, 2)
        value = key_value[..., 1, :].transpose(1, 2)
        repeat = self.query_heads // self.key_value_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        query, key = _apply_rope(
            query, key, cosine[:length], sine[:length]
        )
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.dropout(attention.softmax(dim=-1))
        output = attention @ value
        output = output.transpose(1, 2).contiguous().reshape(
            batch, length, channels
        )
        return self.output(output)


class DropPath(nn.Module):
    def __init__(self, drop_probability: float) -> None:
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.drop_probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        return x.div(keep_probability) * random_tensor.floor()


class TCFormerTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        query_heads: int,
        key_value_heads: int,
        dropout: float,
        drop_path_probability: float,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = GroupedQueryAttention(
            d_model, query_heads, key_value_heads, dropout
        )
        self.drop_path = DropPath(drop_path_probability)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        x = x + self.drop_path(
            self.attention(self.norm1(x), cosine, sine)
        )
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TCFormerHead(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        n_groups: int,
        n_classes: int,
        tcn_depth: int,
        tcn_kernel_length: int,
        tcn_dropout: float,
    ) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.tcn = TCN(
            tcn_depth,
            tcn_kernel_length,
            feature_channels,
            tcn_dropout,
            groups=n_groups,
        )
        self.classifier = Conv1dWithConstraint(
            feature_channels,
            n_classes * n_groups,
            kernel_size=1,
            groups=n_groups,
            max_norm=0.25,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.tcn(x)[:, :, -1:]
        x = self.classifier(x).squeeze(-1)
        return x.reshape(x.shape[0], self.n_groups, self.n_classes).mean(dim=1)


class TCFormerCore(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        f1: int = 32,
        temporal_kernel_lengths: tuple[int, ...] = (20, 32, 64),
        d_group: int = 16,
        depth_multiplier: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        conv_dropout: float = 0.4,
        use_group_attention: bool = True,
        query_heads: int = 4,
        key_value_heads: int = 2,
        transformer_depth: int = 5,
        transformer_dropout: float = 0.4,
        max_drop_path: float = 0.25,
        tcn_depth: int = 2,
        tcn_kernel_length: int = 4,
        tcn_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_groups = len(temporal_kernel_lengths)
        self.d_model = d_group * self.n_groups
        self.conv = MultiKernelConvBlock(
            n_channels,
            temporal_kernel_lengths,
            f1,
            depth_multiplier,
            pool_length_1,
            pool_length_2,
            conv_dropout,
            d_group,
            use_group_attention,
        )
        self.mix = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, 1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )
        drop_rates = (
            torch.linspace(0, 1, transformer_depth) ** 2 * max_drop_path
        )
        self.transformer = nn.ModuleList(
            TCFormerTransformerBlock(
                self.d_model,
                query_heads,
                key_value_heads,
                transformer_dropout,
                float(drop_rates[index]),
            )
            for index in range(transformer_depth)
        )
        self.reduce = nn.Sequential(
            nn.Conv1d(self.d_model, d_group, 1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )
        feature_channels = d_group * (self.n_groups + 1)
        self.head = TCFormerHead(
            feature_channels,
            self.n_groups + 1,
            n_classes,
            tcn_depth,
            tcn_kernel_length,
            tcn_dropout,
        )
        self.register_buffer("_cosine", None, persistent=False)
        self.register_buffer("_sine", None, persistent=False)

    def _rotary_cache(
        self, sequence_length: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        head_dim = self.transformer[0].attention.head_dim
        if self._cosine is None or self._cosine.shape[0] < sequence_length:
            cosine, sine = _build_rotary_cache(
                head_dim, sequence_length, device
            )
            self._cosine = cosine
            self._sine = sine
        assert self._sine is not None
        return self._cosine, self._sine

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        conv_features = self.conv(x)
        tokens = self.mix(conv_features).transpose(1, 2)
        cosine, sine = self._rotary_cache(tokens.shape[1], tokens.device)
        for block in self.transformer:
            tokens = block(tokens, cosine, sine)
        transformer_features = self.reduce(tokens.transpose(1, 2))
        features = torch.cat((conv_features, transformer_features), dim=1)
        return features.flatten(1), self.head(features)


def build_comparison_model(
    architecture: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
) -> ComparisonInputAdapter:
    name = normalize_comparison_model_name(architecture)
    if n_channels < 1 or n_times < 1 or n_classes < 2:
        raise ValueError("n_channels/n_times must be positive and n_classes >= 2")

    if name == "eegnet":
        config: dict[str, Any] = {
            "F1": 8,
            "D": 2,
            "F2": 16,
            "kernel_length": 32,
            "dropout": 0.25,
            "pool_time_length": 4,
            "pool_time_stride": 4,
            "separable_kernel_length": 16,
        }
        core = EEGNetCore(
            n_channels,
            n_classes,
            n_times,
            f1=config["F1"],
            depth_multiplier=config["D"],
            f2=config["F2"],
            kernel_length=config["kernel_length"],
            dropout=config["dropout"],
            pool_time_length=config["pool_time_length"],
            pool_time_stride=config["pool_time_stride"],
            separable_kernel_length=config["separable_kernel_length"],
        )
    elif name == "shallowconvnet":
        config = {
            "n_temporal_filters": 40,
            "temporal_kernel_length": 25,
            "pool_time_length": 75,
            "pool_time_stride": 15,
            "dropout": 0.5,
        }
        core = ShallowConvNetCore(
            n_channels,
            n_classes,
            n_times,
            **config,
        )
    elif name == "atcnet":
        config = {
            "f1": 16,
            "temporal_kernel_length": 64,
            "pool_length": 8,
            "depth_multiplier": 2,
            "conv_dropout": 0.3,
            "d_model": 32,
            "key_dim": 8,
            "n_heads": 2,
            "attention_dropout": 0.5,
            "tcn_depth": 2,
            "tcn_kernel_length": 4,
            "tcn_dropout": 0.3,
            "n_windows": 5,
        }
        core = ATCNetCore(n_channels, n_classes, **config)
    else:
        config = {
            "f1": 32,
            "temporal_kernel_lengths": [20, 32, 64],
            "d_group": 16,
            "depth_multiplier": 2,
            "pool_length_1": 8,
            "pool_length_2": 7,
            "conv_dropout": 0.4,
            "use_group_attention": True,
            "query_heads": 4,
            "key_value_heads": 2,
            "transformer_depth": 5,
            "transformer_dropout": 0.4,
            "max_drop_path": 0.25,
            "tcn_depth": 2,
            "tcn_kernel_length": 4,
            "tcn_dropout": 0.3,
        }
        core = TCFormerCore(
            n_channels,
            n_classes,
            f1=config["f1"],
            temporal_kernel_lengths=tuple(config["temporal_kernel_lengths"]),
            d_group=config["d_group"],
            depth_multiplier=config["depth_multiplier"],
            pool_length_1=config["pool_length_1"],
            pool_length_2=config["pool_length_2"],
            conv_dropout=config["conv_dropout"],
            use_group_attention=config["use_group_attention"],
            query_heads=config["query_heads"],
            key_value_heads=config["key_value_heads"],
            transformer_depth=config["transformer_depth"],
            transformer_dropout=config["transformer_dropout"],
            max_drop_path=config["max_drop_path"],
            tcn_depth=config["tcn_depth"],
            tcn_kernel_length=config["tcn_kernel_length"],
            tcn_dropout=config["tcn_dropout"],
        )

    config = {
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "n_classes": int(n_classes),
        **config,
    }
    return ComparisonInputAdapter(core, name, config)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
