"""
EEG-Conformer – LOSO Activity Three-Class Training Script
==========================================================
Faithful to the original upstream EEG-Conformer architecture, but with:

  * Dynamic n_channels / n_times dimensions (no hard-coded channel count)
  * Global dataset input format  (X / y / subject_ids / metadata.json)
  * LOSO split via --test-subject-id
  * Standardisation computed on train split only
  * CrossEntropyLoss + Adam(lr=0.0002, betas=(0.5, 0.999))  – as in original
  * Per-epoch test evaluation and best_acc tracking
  * JSON metrics output

Usage
-----
    python train_activity_loso.py \\
        --dataset-root  /path/to/global_activity_dataset \\
        --test-subject-id 1 \\
        --epochs 200 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from comparison_models import (
    build_comparison_model,
    count_trainable_parameters,
    normalize_comparison_model_name,
)


# ---------------------------------------------------------------------------
# Paths & top-level constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
EEG_ROOT = PROJECT_ROOT.parent
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import ACTIVITY_LOSO_OUTPUT_DIR, GLOBAL_ACTIVITY_DATASET_DIR

DEFAULT_DATASET_ROOT = GLOBAL_ACTIVITY_DATASET_DIR
DEFAULT_OUTPUT_DIR = ACTIVITY_LOSO_OUTPUT_DIR
DEFAULT_TEST_SUBJECT_ID = 1
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 72
DEFAULT_LR = 0.0002
DEFAULT_BETAS = (0.5, 0.999)
DEFAULT_DEVICE = "cuda:0"
DEFAULT_EMB_SIZE = 40
DEFAULT_DEPTH = 6
DEFAULT_NUM_HEADS = 5
DEFAULT_DROPOUT = 0.5
DEFAULT_TRANSFORMER_ENCODER_DROPOUT = 0.5
DEFAULT_ENV_NAME = "eegconformer310"
DEFAULT_ARCHITECTURE = "eegconformer"
DEFAULT_INPUT_DOMAIN = "time"
FFT_INPUT_DOMAIN = "fft"
DUAL_INPUT_DOMAIN = "time_fft"
VALID_INPUT_DOMAINS = (DEFAULT_INPUT_DOMAIN, FFT_INPUT_DOMAIN, DUAL_INPUT_DOMAIN)
DEFAULT_CONV_TYPE = "standard"
DWCONV_CONV_TYPE = "dwconv"
VALID_CONV_TYPES = (DEFAULT_CONV_TYPE, DWCONV_CONV_TYPE)
CONV_TYPE_ALIASES = {
    "standard": DEFAULT_CONV_TYPE,
    "conv": DEFAULT_CONV_TYPE,
    "normal": DEFAULT_CONV_TYPE,
    "dw": DWCONV_CONV_TYPE,
    "dwconv": DWCONV_CONV_TYPE,
    "depthwise": DWCONV_CONV_TYPE,
    "depthwise_conv": DWCONV_CONV_TYPE,
}
DEFAULT_FFT_GLOBAL = "none"
FFT_GLOBAL_MLP = "mlp"
VALID_FFT_GLOBALS = (DEFAULT_FFT_GLOBAL, FFT_GLOBAL_MLP)
FFT_GLOBAL_ALIASES = {
    "none": DEFAULT_FFT_GLOBAL,
    "off": DEFAULT_FFT_GLOBAL,
    "false": DEFAULT_FFT_GLOBAL,
    "no": DEFAULT_FFT_GLOBAL,
    "mlp": FFT_GLOBAL_MLP,
    "frequency_mlp": FFT_GLOBAL_MLP,
    "freq_mlp": FFT_GLOBAL_MLP,
}
DEFAULT_INPUT_QKV = "none"
INPUT_QKV_CHANNEL = "channel"
INPUT_QKV_TIME = "time"
VALID_INPUT_QKVS = (DEFAULT_INPUT_QKV, INPUT_QKV_CHANNEL, INPUT_QKV_TIME)
INPUT_QKV_ALIASES = {
    "none": DEFAULT_INPUT_QKV,
    "off": DEFAULT_INPUT_QKV,
    "false": DEFAULT_INPUT_QKV,
    "no": DEFAULT_INPUT_QKV,
    "channel": INPUT_QKV_CHANNEL,
    "channels": INPUT_QKV_CHANNEL,
    "channel_qkv": INPUT_QKV_CHANNEL,
    "qkv": INPUT_QKV_CHANNEL,
    "time": INPUT_QKV_TIME,
    "temporal": INPUT_QKV_TIME,
    "time_token": INPUT_QKV_TIME,
    "time_tokens": INPUT_QKV_TIME,
    "time_qkv": INPUT_QKV_TIME,
    "temporal_qkv": INPUT_QKV_TIME,
    "qkv_time": INPUT_QKV_TIME,
}
DEFAULT_INPUT_QKV_DIM = 64
DEFAULT_INPUT_QKV_HEADS = 4
DEFAULT_INPUT_QKV_DROPOUT = 0.1
DEFAULT_INPUT_QKV_RES_SCALE = 0.1
DEFAULT_CUMULATIVE_QUERY_ATTENTION = False
DEFAULT_TRANSFORMER_BRANCHES = 1
TRANSFORMER_FUSION_SINGLE = "single"
TRANSFORMER_FUSION_SOFTMAX = "softmax_weighted_sum"
TRANSFORMER_FUSION_LOSS_SOFTMAX = "loss_softmax"
DEFAULT_TRANSFORMER_BRANCH_FUSION = TRANSFORMER_FUSION_SOFTMAX
DEFAULT_BRANCH_LOSS_AUX_WEIGHT = 0.0
DEFAULT_TRANSFORMER_BRANCH_QKV = "none"
TRANSFORMER_BRANCH_QKV_CROSS_DEPTH = "cross_depth"
DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT = 0.1
DEFAULT_TRANSFORMER_BRANCH_QKV_RES_SCALE = 0.1
TRANSFORMER_BRANCH_QKV_ALIASES = {
    "none": DEFAULT_TRANSFORMER_BRANCH_QKV,
    "off": DEFAULT_TRANSFORMER_BRANCH_QKV,
    "false": DEFAULT_TRANSFORMER_BRANCH_QKV,
    "no": DEFAULT_TRANSFORMER_BRANCH_QKV,
    "cross_depth": TRANSFORMER_BRANCH_QKV_CROSS_DEPTH,
    "crossdepth": TRANSFORMER_BRANCH_QKV_CROSS_DEPTH,
    "depth": TRANSFORMER_BRANCH_QKV_CROSS_DEPTH,
}
TRANSFORMER_BRANCH_FUSION_ALIASES = {
    "single": TRANSFORMER_FUSION_SINGLE,
    "feature": TRANSFORMER_FUSION_SOFTMAX,
    "feature_softmax": TRANSFORMER_FUSION_SOFTMAX,
    "softmax": TRANSFORMER_FUSION_SOFTMAX,
    "softmax_weighted_sum": TRANSFORMER_FUSION_SOFTMAX,
    "loss": TRANSFORMER_FUSION_LOSS_SOFTMAX,
    "loss_softmax": TRANSFORMER_FUSION_LOSS_SOFTMAX,
}
DEFAULT_INPUT_QKV_TIME_TOKEN_LEN = 32
RESUME_CHECKPOINT_FILENAME = "last_checkpoint.pt"
RESUME_CHECKPOINT_VERSION = 1
AUTO_RERUN_ENV_VAR = "TRAIN_ACTIVITY_LOSO_PROJECT_ENV_ACTIVE"


def validate_conv_type(raw: str | None) -> str:
    value = DEFAULT_CONV_TYPE if raw is None else str(raw).strip().lower().replace("-", "_")
    resolved = CONV_TYPE_ALIASES.get(value)
    if resolved is None:
        allowed = ", ".join(VALID_CONV_TYPES)
        raise ValueError(f"conv_type must be one of: {allowed}")
    return resolved


def validate_fft_global(raw: str | None) -> str:
    value = DEFAULT_FFT_GLOBAL if raw is None else str(raw).strip().lower().replace("-", "_")
    resolved = FFT_GLOBAL_ALIASES.get(value)
    if resolved is None:
        allowed = ", ".join(VALID_FFT_GLOBALS)
        raise ValueError(f"fft_global must be one of: {allowed}")
    return resolved


def validate_input_qkv(raw: str | None) -> str:
    value = DEFAULT_INPUT_QKV if raw is None else str(raw).strip().lower().replace("-", "_")
    resolved = INPUT_QKV_ALIASES.get(value)
    if resolved is None:
        allowed = ", ".join(VALID_INPUT_QKVS)
        raise ValueError(f"input_qkv must be one of: {allowed}")
    return resolved


def validate_fft_global_for_input_domain(input_domain: str | None, fft_global: str | None) -> str:
    resolved_input_domain = validate_input_domain(input_domain)
    resolved_fft_global = validate_fft_global(fft_global)
    if resolved_input_domain == DEFAULT_INPUT_DOMAIN and resolved_fft_global != DEFAULT_FFT_GLOBAL:
        raise ValueError("--fft-global applies only to fft or time_fft input domains")
    return resolved_fft_global


def resolve_transformer_branch_depths(
    depth: int = DEFAULT_DEPTH,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
    input_domain: str | None = None,
) -> tuple[int, ...]:
    """Validate the optional parallel Transformer depth configuration.

    The historical single-encoder path remains ``--depth``.  Parallel depth
    branches are deliberately limited to the ``time_fft`` dual-input model,
    where time and FFT each receive an independent set of encoders and
    independent learnable fusion logits.
    """
    resolved_depth = int(depth)
    if resolved_depth < 1:
        raise ValueError("depth must be >= 1")

    resolved_branches = int(transformer_branches)
    if resolved_branches < 1:
        raise ValueError("transformer_branches must be >= 1")

    if transformer_depths is None:
        if resolved_branches != DEFAULT_TRANSFORMER_BRANCHES:
            raise ValueError(
                "--transformer-depths is required when --transformer-branches is greater than 1"
            )
        resolved_depths = (resolved_depth,)
    else:
        resolved_depths = tuple(int(value) for value in transformer_depths)
        if len(resolved_depths) != resolved_branches:
            raise ValueError(
                "--transformer-branches must equal the number of values in --transformer-depths"
            )
        if any(value < 1 for value in resolved_depths):
            raise ValueError("all --transformer-depths values must be >= 1")
        if resolved_branches == DEFAULT_TRANSFORMER_BRANCHES and resolved_depths != (resolved_depth,):
            raise ValueError("use --depth for a single Transformer encoder")

    if len(resolved_depths) > 1 and resolved_depth != DEFAULT_DEPTH:
        raise ValueError("--depth cannot be combined with parallel --transformer-depths")

    if input_domain is not None and len(resolved_depths) > 1:
        if validate_input_domain(input_domain) != DUAL_INPUT_DOMAIN:
            raise ValueError(
                "parallel Transformer depth branches require --input-domain time_fft"
            )
    return resolved_depths


def validate_transformer_branch_fusion(raw: str | None) -> str:
    """Normalize the opt-in parallel-depth fusion strategy."""
    value = (
        DEFAULT_TRANSFORMER_BRANCH_FUSION
        if raw is None
        else str(raw).strip().lower().replace("-", "_")
    )
    resolved = TRANSFORMER_BRANCH_FUSION_ALIASES.get(value)
    if resolved is None:
        raise ValueError(
            "transformer_branch_fusion must be one of: feature_softmax, loss_softmax"
        )
    return resolved


def resolve_transformer_branch_fusion(
    raw: str | None,
    transformer_branches: int,
    input_domain: str | None = None,
) -> str:
    """Return the effective fusion mode and reject incompatible combinations."""
    resolved = validate_transformer_branch_fusion(raw)
    branches = int(transformer_branches)
    if branches < 1:
        raise ValueError("transformer_branches must be >= 1")
    if branches == 1:
        if resolved == TRANSFORMER_FUSION_LOSS_SOFTMAX:
            raise ValueError("loss_softmax requires at least two Transformer branches")
        return TRANSFORMER_FUSION_SINGLE
    if resolved == TRANSFORMER_FUSION_SINGLE:
        raise ValueError("single fusion cannot be used with parallel Transformer branches")
    if (
        resolved == TRANSFORMER_FUSION_LOSS_SOFTMAX
        and input_domain is not None
        and validate_input_domain(input_domain) != DUAL_INPUT_DOMAIN
    ):
        raise ValueError("loss_softmax requires --input-domain time_fft")
    return resolved


def resolve_branch_loss_aux_weight(raw: float, transformer_branch_fusion: str) -> float:
    """Validate the additive mean-branch-loss coefficient."""
    value = float(raw)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("branch_loss_aux_weight must be a finite value >= 0")
    if transformer_branch_fusion != TRANSFORMER_FUSION_LOSS_SOFTMAX and value != 0.0:
        raise ValueError(
            "--branch-loss-aux-weight applies only when "
            "--transformer-branch-fusion loss_softmax is enabled"
        )
    return value


def validate_transformer_branch_qkv(raw: str | None) -> str:
    """Normalize the optional QKV communication mode between depth branches."""
    value = (
        DEFAULT_TRANSFORMER_BRANCH_QKV
        if raw is None
        else str(raw).strip().lower().replace("-", "_")
    )
    resolved = TRANSFORMER_BRANCH_QKV_ALIASES.get(value)
    if resolved is None:
        raise ValueError("transformer_branch_qkv must be one of: none, cross_depth")
    return resolved


def resolve_transformer_branch_qkv(
    raw: str | None,
    transformer_branch_fusion: str,
    transformer_branches: int,
    input_domain: str | None = None,
) -> str:
    """Validate Cross-Depth QKV against the surrounding branch architecture."""
    resolved = validate_transformer_branch_qkv(raw)
    if resolved == DEFAULT_TRANSFORMER_BRANCH_QKV:
        return resolved
    if int(transformer_branches) < 2:
        raise ValueError("cross_depth QKV requires at least two Transformer branches")
    if transformer_branch_fusion != TRANSFORMER_FUSION_LOSS_SOFTMAX:
        raise ValueError(
            "cross_depth QKV requires --transformer-branch-fusion loss_softmax"
        )
    if input_domain is not None and validate_input_domain(input_domain) != DUAL_INPUT_DOMAIN:
        raise ValueError("cross_depth QKV requires --input-domain time_fft")
    return resolved


def validate_dropout_probability(raw: float, parameter_name: str) -> float:
    """Return a finite dropout probability in the half-open interval [0, 1)."""
    value = float(raw)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{parameter_name} must be a finite value in [0, 1)")
    return value


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

class RuntimeConfig(NamedTuple):
    dataset_root: Path
    test_subject_id: int
    epochs: int
    batch_size: int
    lr: float
    device: str
    output_dir: Path
    seed: int
    input_domain: str = DEFAULT_INPUT_DOMAIN
    conv_type: str = DEFAULT_CONV_TYPE
    fft_global: str = DEFAULT_FFT_GLOBAL
    class_weights: list[float] | None = None
    input_qkv: str = DEFAULT_INPUT_QKV
    input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM
    input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS
    input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT
    input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE
    cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION
    depth: int = DEFAULT_DEPTH
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES
    transformer_depths: tuple[int, ...] = (DEFAULT_DEPTH,)
    transformer_branch_fusion: str = TRANSFORMER_FUSION_SINGLE
    branch_loss_aux_weight: float = DEFAULT_BRANCH_LOSS_AUX_WEIGHT
    transformer_branch_qkv: str = DEFAULT_TRANSFORMER_BRANCH_QKV
    resume: bool = False
    transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT
    transformer_branch_qkv_dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT
    classification_mode: str = "flat"


# ---------------------------------------------------------------------------
# Model – faithful to original EEG-Conformer architecture
# (PatchEmbedding + MultiHeadAttention + ResidualAdd +
#  FeedForwardBlock + TransformerEncoder + flatten-fc head)
# but without einops so it works in the base conda env.
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Shallow-CNN stem that maps (B, 1, C, T) → (B, n_patches, emb_size).

    Architecture mirrors the original EEG-Conformer shallownet branch with the
    spatial conv kernel made dynamic on ``n_channels``.
    """

    def __init__(
        self,
        n_channels: int,
        emb_size: int = 40,
        dropout: float = 0.5,
        conv_type: str = DEFAULT_CONV_TYPE,
    ) -> None:
        super().__init__()
        self.conv_type = validate_conv_type(conv_type)
        spatial_groups = 40 if self.conv_type == DWCONV_CONV_TYPE else 1
        self.shallownet = nn.Sequential(
            # temporal filter
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            # spatial filter across all EEG channels; optionally depthwise over the 40 temporal-filter maps
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1), groups=spatial_groups),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # patch pooling (same kernel/stride as original)
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)      # (B, 40, 1, n_patches)
        x = self.projection(x)      # (B, emb_size, 1, n_patches)
        x = x.squeeze(2)            # (B, emb_size, n_patches)
        x = x.transpose(1, 2)       # (B, n_patches, emb_size)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention matching the original EEG-Conformer layout,
    but using view+transpose instead of einops.rearrange."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        cumulative_queries: Tensor | None = None,
        cumulative_query_attention: bool = False,
        return_cumulative_queries: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        B, N, _ = x.shape
        h = self.num_heads
        d = self.emb_size // h

        def _reshape(t: Tensor) -> Tensor:
            return t.view(B, N, h, d).transpose(1, 2)  # (B, h, N, d)

        queries = _reshape(self.queries(x))
        keys = _reshape(self.keys(x))
        values = _reshape(self.values(x))

        updated_cumulative_queries: Tensor | None = None
        energy_queries = queries
        if cumulative_query_attention:
            updated_cumulative_queries = (
                queries if cumulative_queries is None else cumulative_queries + queries
            )
            energy_queries = updated_cumulative_queries

        energy = torch.einsum("bhqd,bhkd->bhqk", energy_queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5  # faithful to original
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum("bhal,bhlv->bhav", att, values)  # (B, h, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, self.emb_size)
        projected = self.projection(out)
        if return_cumulative_queries:
            return projected, updated_cumulative_queries
        return projected


class ResidualAdd(nn.Module):
    """Residual wrapper matching the original EEG-Conformer layout."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    """Position-wise FFN matching the original EEG-Conformer layout."""

    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5) -> None:
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """One Transformer encoder block with optional cross-block query accumulation."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int = 5,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ) -> None:
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )

    def forward(
        self,
        x: Tensor,
        cumulative_queries: Tensor | None = None,
        cumulative_query_attention: bool = False,
        return_cumulative_queries: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        # Keep the original Sequential child layout (and state_dict keys), while
        # explicitly exposing the attention queries to the enclosing encoder.
        attention_stack = self[0].fn
        normalized = attention_stack[0](x)
        attention_out, cumulative_queries = attention_stack[1](
            normalized,
            cumulative_queries=cumulative_queries,
            cumulative_query_attention=cumulative_query_attention,
            return_cumulative_queries=True,
        )
        x = x + attention_stack[2](attention_out)
        x = self[1](x)
        if return_cumulative_queries:
            return x, cumulative_queries
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        depth: int,
        emb_size: int,
        num_heads: int = 5,
        cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
        dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    ) -> None:
        dropout_p = validate_dropout_probability(dropout, "transformer_encoder_dropout")
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size,
                    num_heads,
                    drop_p=dropout_p,
                    forward_drop_p=dropout_p,
                )
                for _ in range(depth)
            ]
        )
        self.dropout_p = dropout_p
        self.cumulative_query_attention = bool(cumulative_query_attention)

    def forward(self, x: Tensor) -> Tensor:
        cumulative_queries: Tensor | None = None
        for block in self:
            x, cumulative_queries = block(
                x,
                cumulative_queries=cumulative_queries,
                cumulative_query_attention=self.cumulative_query_attention,
                return_cumulative_queries=True,
            )
        return x


class ClassificationHead(nn.Module):
    """flatten + fc head with dynamic flat_size.

    Returns (token_features, logits) to match the original interface:
        tok, outputs = model(img)
    """

    def __init__(self, emb_size: int, n_patches: int, n_classes: int) -> None:
        super().__init__()
        flat_size = emb_size * n_patches
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


def compute_fft_global_hidden_size(n_times: int) -> int:
    """Small hidden width for the optional FFT frequency MLP."""
    return max(16, int(n_times) // 4)


class FFTGlobalMLP(nn.Module):
    """Fast global frequency mixer for FFT inputs.

    Shape is preserved: (B, 1, C, F) -> (B, 1, C, F).  The same small MLP is
    shared across EEG channels and operates along the full frequency axis.
    """

    def __init__(self, n_times: int, hidden_size: int | None = None) -> None:
        super().__init__()
        self.n_times = int(n_times)
        self.hidden_size = compute_fft_global_hidden_size(self.n_times) if hidden_size is None else int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.n_times, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.n_times),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"FFTGlobalMLP expects a 4D tensor (B,1,C,F), got shape={tuple(x.shape)}")
        B, one, C, Freq = x.shape
        if one != 1:
            raise ValueError(f"FFTGlobalMLP expects channel dimension 1, got {one}")
        if Freq != self.n_times:
            raise ValueError(f"FFTGlobalMLP was built for F={self.n_times}, got F={Freq}")
        y = x.reshape(B * C, Freq)
        y = self.net(y)
        return y.reshape(B, one, C, Freq)


class InputQKVResidual(nn.Module):
    """Channel-token pre-network QKV attention with an input skip connection.

    The input shape is preserved: (B, 1, C, L) -> (B, 1, C, L).  EEG channels
    are treated as tokens and each token uses the full time/frequency axis as
    its feature vector.  This keeps the attention cost small while allowing
    channel-wise mixing before the original EEG-Conformer stem.
    """

    def __init__(
        self,
        n_times: int,
        d_model: int = DEFAULT_INPUT_QKV_DIM,
        num_heads: int = DEFAULT_INPUT_QKV_HEADS,
        dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
        res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
    ) -> None:
        super().__init__()
        self.n_times = int(n_times)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout_p = float(dropout)
        if self.n_times < 1:
            raise ValueError("InputQKVResidual n_times must be >= 1")
        if self.d_model < 1:
            raise ValueError("input_qkv_dim must be >= 1")
        if self.num_heads < 1:
            raise ValueError("input_qkv_heads must be >= 1")
        if self.d_model % self.num_heads != 0:
            raise ValueError("input_qkv_dim must be divisible by input_qkv_heads")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("input_qkv_dropout must be in [0, 1)")

        self.norm = nn.LayerNorm(self.n_times)
        self.in_proj = nn.Linear(self.n_times, self.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.out_proj = nn.Linear(self.d_model, self.n_times)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gamma = nn.Parameter(torch.tensor(float(res_scale), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"InputQKVResidual expects a 4D tensor (B,1,C,L), got shape={tuple(x.shape)}")
        B, one, C, L = x.shape
        if one != 1:
            raise ValueError(f"InputQKVResidual expects channel dimension 1, got {one}")
        if L != self.n_times:
            raise ValueError(f"InputQKVResidual was built for L={self.n_times}, got L={L}")

        residual = x
        z = x.squeeze(1)          # (B, C, L)
        z = self.norm(z)
        z = self.in_proj(z)       # (B, C, d_model)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = self.out_proj(z)      # (B, C, L)
        z = self.dropout(z).unsqueeze(1)
        return residual + self.gamma * z


def compute_temporal_qkv_token_shape(
    n_times: int,
    preferred_token_len: int = DEFAULT_INPUT_QKV_TIME_TOKEN_LEN,
) -> tuple[int, int, int, int]:
    """Return (n_tokens, token_len, padded_n_times, pad_len) for time-token QKV.

    The default keeps chunks close to 32 samples.  For example:
      * L=1920 -> 60 tokens x 32 samples
      * L=961  -> 31 tokens x 31 samples
    If the length is not exactly divisible, the signal is zero-padded only
    inside the QKV block and cropped back before the residual add.
    """
    n_times = int(n_times)
    preferred_token_len = int(preferred_token_len)
    if n_times < 1:
        raise ValueError("n_times must be >= 1 for temporal QKV")
    if preferred_token_len < 1:
        raise ValueError("preferred_token_len must be >= 1 for temporal QKV")
    n_tokens = max(1, (n_times + preferred_token_len - 1) // preferred_token_len)
    token_len = max(1, (n_times + n_tokens - 1) // n_tokens)
    padded_n_times = n_tokens * token_len
    pad_len = padded_n_times - n_times
    return n_tokens, token_len, padded_n_times, pad_len


class InputTemporalQKVResidual(nn.Module):
    """Time-token pre-network QKV attention with an input skip connection.

    The input shape is preserved: (B, 1, C, L) -> (B, 1, C, L).  The time or
    frequency axis is split into short consecutive chunks.  Each chunk is one
    token whose feature vector contains all EEG channels within that chunk.
    Therefore attention is over temporal/frequency chunks (e.g. 60x60 for a
    1920-sample time window), not over the 21 electrodes.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        preferred_token_len: int = DEFAULT_INPUT_QKV_TIME_TOKEN_LEN,
        d_model: int = DEFAULT_INPUT_QKV_DIM,
        num_heads: int = DEFAULT_INPUT_QKV_HEADS,
        dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
        res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.preferred_token_len = int(preferred_token_len)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout_p = float(dropout)
        if self.n_channels < 1:
            raise ValueError("InputTemporalQKVResidual n_channels must be >= 1")
        if self.n_times < 1:
            raise ValueError("InputTemporalQKVResidual n_times must be >= 1")
        if self.d_model < 1:
            raise ValueError("input_qkv_dim must be >= 1")
        if self.num_heads < 1:
            raise ValueError("input_qkv_heads must be >= 1")
        if self.d_model % self.num_heads != 0:
            raise ValueError("input_qkv_dim must be divisible by input_qkv_heads")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("input_qkv_dropout must be in [0, 1)")

        (
            self.n_tokens,
            self.token_len,
            self.padded_n_times,
            self.pad_len,
        ) = compute_temporal_qkv_token_shape(self.n_times, self.preferred_token_len)
        self.feature_dim = self.n_channels * self.token_len

        self.norm = nn.LayerNorm(self.feature_dim)
        self.in_proj = nn.Linear(self.feature_dim, self.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.out_proj = nn.Linear(self.d_model, self.feature_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gamma = nn.Parameter(torch.tensor(float(res_scale), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"InputTemporalQKVResidual expects a 4D tensor (B,1,C,L), got shape={tuple(x.shape)}")
        B, one, C, L = x.shape
        if one != 1:
            raise ValueError(f"InputTemporalQKVResidual expects channel dimension 1, got {one}")
        if C != self.n_channels:
            raise ValueError(f"InputTemporalQKVResidual was built for C={self.n_channels}, got C={C}")
        if L != self.n_times:
            raise ValueError(f"InputTemporalQKVResidual was built for L={self.n_times}, got L={L}")

        residual = x
        z = x.squeeze(1)  # (B, C, L)
        if self.pad_len:
            z = F.pad(z, (0, self.pad_len))
        z = z.reshape(B, C, self.n_tokens, self.token_len)
        z = z.permute(0, 2, 1, 3).contiguous().view(B, self.n_tokens, self.feature_dim)
        z = self.norm(z)
        z = self.in_proj(z)  # (B, n_tokens, d_model)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = self.out_proj(z)  # (B, n_tokens, C * token_len)
        z = self.dropout(z)
        z = z.view(B, self.n_tokens, C, self.token_len)
        z = z.permute(0, 2, 1, 3).contiguous().view(B, C, self.padded_n_times)
        if self.pad_len:
            z = z[..., :L]
        return residual + self.gamma * z.unsqueeze(1)


class ConformerFeatureBranch(nn.Module):
    """EEG-Conformer feature extractor used by the dual-branch model.

    It mirrors the single-branch stem (PatchEmbedding + TransformerEncoder),
    but returns the flattened token features instead of applying a classifier.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        emb_size: int = 40,
        depth: int = 6,
        num_heads: int = 5,
        dropout: float = 0.5,
        conv_type: str = DEFAULT_CONV_TYPE,
        input_qkv: str = DEFAULT_INPUT_QKV,
        input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM,
        input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS,
        input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
        input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
        cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
        transformer_depths: list[int] | tuple[int, ...] | None = None,
        enable_feature_fusion: bool = True,
        transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    ) -> None:
        super().__init__()
        self.conv_type = validate_conv_type(conv_type)
        self.input_qkv = validate_input_qkv(input_qkv)
        self.cumulative_query_attention = bool(cumulative_query_attention)
        self.enable_feature_fusion = bool(enable_feature_fusion)
        self.transformer_branch_depths = (
            (int(depth),)
            if transformer_depths is None
            else tuple(int(value) for value in transformer_depths)
        )
        if not self.transformer_branch_depths or any(
            value < 1 for value in self.transformer_branch_depths
        ):
            raise ValueError("transformer_depths must contain positive integers")
        self.transformer_branches = len(self.transformer_branch_depths)
        self.transformer_fusion = (
            TRANSFORMER_FUSION_SOFTMAX
            if self.transformer_branches > 1
            else TRANSFORMER_FUSION_SINGLE
        )
        n_patches = compute_n_patches(n_times)
        self.flat_size = emb_size * n_patches
        if self.input_qkv == INPUT_QKV_CHANNEL:
            self.input_qkv_layer = InputQKVResidual(
                n_times=n_times,
                d_model=input_qkv_dim,
                num_heads=input_qkv_heads,
                dropout=input_qkv_dropout,
                res_scale=input_qkv_res_scale,
            )
        elif self.input_qkv == INPUT_QKV_TIME:
            self.input_qkv_layer = InputTemporalQKVResidual(
                n_channels=n_channels,
                n_times=n_times,
                d_model=input_qkv_dim,
                num_heads=input_qkv_heads,
                dropout=input_qkv_dropout,
                res_scale=input_qkv_res_scale,
            )
        else:
            self.input_qkv_layer = nn.Identity()
        self.patch_embedding = PatchEmbedding(n_channels, emb_size, dropout, conv_type=self.conv_type)
        if self.transformer_branches > 1:
            self.depth_encoders = nn.ModuleList(
                [
                    TransformerEncoder(
                        branch_depth,
                        emb_size,
                        num_heads,
                        dropout=transformer_encoder_dropout,
                        cumulative_query_attention=self.cumulative_query_attention,
                    )
                    for branch_depth in self.transformer_branch_depths
                ]
            )
            if self.enable_feature_fusion:
                # Equal zero logits become equal 1/n weights after softmax.  The
                # time and FFT ConformerFeatureBranch instances own separate logits.
                self.depth_weight_logits = nn.Parameter(torch.zeros(self.transformer_branches))
        else:
            # Keep the historical attribute/state_dict layout unchanged when
            # the new feature is disabled, so old checkpoints still load.
            self.encoder = TransformerEncoder(
                depth,
                emb_size,
                num_heads,
                dropout=transformer_encoder_dropout,
                cumulative_query_attention=self.cumulative_query_attention,
            )

    def normalized_transformer_weights(self) -> Tensor:
        if self.transformer_branches == 1:
            parameter = next(self.encoder.parameters())
            return parameter.new_ones(1)
        if not self.enable_feature_fusion:
            raise RuntimeError("feature-fusion weights are disabled for this branch")
        return F.softmax(self.depth_weight_logits, dim=0)

    def transformer_weight_values(self) -> list[float]:
        weights = self.normalized_transformer_weights().detach().cpu().tolist()
        return [float(value) for value in weights]

    def forward_transformer_branch_tokens(self, x: Tensor) -> tuple[Tensor, ...]:
        """Return unflattened ``(B, patches, emb)`` tokens for every depth."""
        x = self.input_qkv_layer(x)
        tokens = self.patch_embedding(x)
        if self.transformer_branches > 1:
            return tuple(encoder(tokens) for encoder in self.depth_encoders)
        return (self.encoder(tokens),)

    @staticmethod
    def flatten_transformer_branch_tokens(
        branch_tokens: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        return tuple(
            value.contiguous().view(value.size(0), -1)
            for value in branch_tokens
        )

    def forward_transformer_branches(self, x: Tensor) -> tuple[Tensor, ...]:
        """Return one flattened feature tensor for each Transformer depth."""
        return self.flatten_transformer_branch_tokens(
            self.forward_transformer_branch_tokens(x)
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_features = self.forward_transformer_branches(x)
        if self.transformer_branches > 1:
            if not self.enable_feature_fusion:
                raise RuntimeError(
                    "Use forward_transformer_branches when feature fusion is disabled"
                )
            stacked = torch.stack(branch_features, dim=0)
            weights = self.normalized_transformer_weights().view(-1, 1, 1)
            return (weights * stacked).sum(dim=0)
        return branch_features[0]


class CrossDepthQKVResidual(nn.Module):
    """Exchange information across parallel depths at each patch position.

    Inputs are one ``(B, patches, emb_size)`` tensor per depth.  The depth axis
    becomes a short attention sequence, so every branch query can attend to the
    keys/values of all depth branches while patch positions remain aligned.
    """

    def __init__(
        self,
        n_branches: int,
        emb_size: int,
        num_heads: int,
        dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
        res_scale: float = DEFAULT_TRANSFORMER_BRANCH_QKV_RES_SCALE,
    ) -> None:
        super().__init__()
        self.n_branches = int(n_branches)
        self.emb_size = int(emb_size)
        self.dropout_p = validate_dropout_probability(
            dropout,
            "transformer_branch_qkv_dropout",
        )
        if self.n_branches < 2:
            raise ValueError("CrossDepthQKVResidual requires at least two branches")
        self.depth_embeddings = nn.Parameter(
            torch.empty(self.n_branches, self.emb_size)
        )
        nn.init.normal_(self.depth_embeddings, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(self.emb_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.emb_size,
            num_heads=int(num_heads),
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gamma = nn.Parameter(torch.tensor(float(res_scale), dtype=torch.float32))

    def forward(self, branch_tokens: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if len(branch_tokens) != self.n_branches:
            raise ValueError(
                f"Expected {self.n_branches} depth tensors, got {len(branch_tokens)}"
            )
        # (B, patches, depth, emb); torch.stack also verifies equal token shapes.
        residual = torch.stack(branch_tokens, dim=2)
        if residual.ndim != 4 or residual.shape[-1] != self.emb_size:
            raise ValueError(
                "Cross-depth tokens must have shape (B, patches, emb_size)"
            )
        batch_size, n_patches, _, _ = residual.shape
        depth_positions = self.depth_embeddings.view(1, 1, self.n_branches, self.emb_size)
        attended_input = self.norm(residual + depth_positions)
        attended_input = attended_input.reshape(
            batch_size * n_patches,
            self.n_branches,
            self.emb_size,
        )
        attended, _ = self.attention(
            attended_input,
            attended_input,
            attended_input,
            need_weights=False,
        )
        attended = attended.reshape(
            batch_size,
            n_patches,
            self.n_branches,
            self.emb_size,
        )
        updated = residual + self.gamma * self.dropout(attended)
        return tuple(updated[:, :, index, :] for index in range(self.n_branches))


class FusionClassificationHead(nn.Module):
    """MLP classifier for already-flattened single or fused features."""

    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x, self.fc(x)


def validate_classification_mode(value: str) -> str:
    if value not in ("flat", "hierarchical"):
        raise ValueError("classification_mode must be flat or hierarchical")
    return value


class HierarchicalClassificationHead(nn.Module):
    """P(e1)=P(group12)P(e1|group12), likewise e2; P(e3)=P(group3).

    CE on these normalized log probabilities is the sum of gate CE and
    conditional CE, with no conditional loss for true e3 samples.
    """

    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        if n_classes != 3:
            raise ValueError("hierarchical classification requires exactly 3 classes")
        self.group_head = FusionClassificationHead(in_features, 2)
        self.within_group_head = FusionClassificationHead(in_features, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        group = F.log_softmax(self.group_head(x)[1], dim=1)
        within = F.log_softmax(self.within_group_head(x)[1], dim=1)
        return x, torch.cat((group[:, :1] + within, group[:, 1:]), dim=1)


def predict_class_labels(model: nn.Module, logits: Tensor) -> Tensor:
    """Hard gate first, then distinguish e1/e2 (labels 0/1/2)."""
    if getattr(model, "classification_mode", "flat") == "hierarchical":
        group12 = torch.logsumexp(logits[:, :2], dim=1)
        return torch.where(group12 >= logits[:, 2], logits[:, :2].argmax(dim=1), 2)
    return logits.argmax(dim=1)


class DualBranchActivityConformer(nn.Module):
    """Dual-domain EEG-Conformer with opt-in feature- or loss-level fusion.

    The legacy/default path independently softmax-fuses the parallel depth
    features in the time and FFT domains and then uses one classifier.  The
    ``loss_softmax`` path pairs equal-depth time/FFT features, gives every pair
    its own classifier, and learns one shared softmax weight per depth.
    """

    def __init__(
        self,
        n_channels: int,
        time_n_times: int,
        fft_n_times: int,
        n_classes: int = 3,
        emb_size: int = 40,
        depth: int = 6,
        num_heads: int = 5,
        dropout: float = 0.5,
        conv_type: str = DEFAULT_CONV_TYPE,
        fft_global: str = DEFAULT_FFT_GLOBAL,
        input_qkv: str = DEFAULT_INPUT_QKV,
        input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM,
        input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS,
        input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
        input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
        cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
        transformer_depths: list[int] | tuple[int, ...] | None = None,
        transformer_branch_fusion: str = DEFAULT_TRANSFORMER_BRANCH_FUSION,
        branch_loss_aux_weight: float = DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
        transformer_branch_qkv: str = DEFAULT_TRANSFORMER_BRANCH_QKV,
        transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
        transformer_branch_qkv_dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
        classification_mode: str = "flat",
    ) -> None:
        super().__init__()
        self.classification_mode = validate_classification_mode(classification_mode)
        head_type = (
            HierarchicalClassificationHead
            if classification_mode == "hierarchical"
            else FusionClassificationHead
        )
        self.conv_type = validate_conv_type(conv_type)
        self.fft_global = validate_fft_global(fft_global)
        self.input_qkv = validate_input_qkv(input_qkv)
        self.cumulative_query_attention = bool(cumulative_query_attention)
        self.transformer_branch_depths = (
            (int(depth),)
            if transformer_depths is None
            else tuple(int(value) for value in transformer_depths)
        )
        if not self.transformer_branch_depths or any(
            value < 1 for value in self.transformer_branch_depths
        ):
            raise ValueError("transformer_depths must contain positive integers")
        self.transformer_branches = len(self.transformer_branch_depths)
        self.transformer_fusion = resolve_transformer_branch_fusion(
            transformer_branch_fusion,
            transformer_branches=self.transformer_branches,
            input_domain=DUAL_INPUT_DOMAIN,
        )
        self.branch_loss_aux_weight = resolve_branch_loss_aux_weight(
            branch_loss_aux_weight,
            self.transformer_fusion,
        )
        self.uses_branch_loss_fusion = (
            self.transformer_fusion == TRANSFORMER_FUSION_LOSS_SOFTMAX
        )
        self.transformer_branch_qkv = resolve_transformer_branch_qkv(
            transformer_branch_qkv,
            transformer_branch_fusion=self.transformer_fusion,
            transformer_branches=self.transformer_branches,
            input_domain=DUAL_INPUT_DOMAIN,
        )
        self.uses_cross_depth_qkv = (
            self.transformer_branch_qkv == TRANSFORMER_BRANCH_QKV_CROSS_DEPTH
        )
        self.transformer_encoder_dropout = validate_dropout_probability(
            transformer_encoder_dropout,
            "transformer_encoder_dropout",
        )
        self.transformer_branch_qkv_dropout = validate_dropout_probability(
            transformer_branch_qkv_dropout,
            "transformer_branch_qkv_dropout",
        )
        self.fft_global_layer = (
            FFTGlobalMLP(fft_n_times) if self.fft_global == FFT_GLOBAL_MLP else nn.Identity()
        )
        self.time_branch = ConformerFeatureBranch(
            n_channels=n_channels,
            n_times=time_n_times,
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            transformer_encoder_dropout=self.transformer_encoder_dropout,
            conv_type=self.conv_type,
            input_qkv=self.input_qkv,
            input_qkv_dim=input_qkv_dim,
            input_qkv_heads=input_qkv_heads,
            input_qkv_dropout=input_qkv_dropout,
            input_qkv_res_scale=input_qkv_res_scale,
            cumulative_query_attention=self.cumulative_query_attention,
            transformer_depths=self.transformer_branch_depths,
            enable_feature_fusion=not self.uses_branch_loss_fusion,
        )
        self.fft_branch = ConformerFeatureBranch(
            n_channels=n_channels,
            n_times=fft_n_times,
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            transformer_encoder_dropout=self.transformer_encoder_dropout,
            conv_type=self.conv_type,
            input_qkv=self.input_qkv,
            input_qkv_dim=input_qkv_dim,
            input_qkv_heads=input_qkv_heads,
            input_qkv_dropout=input_qkv_dropout,
            input_qkv_res_scale=input_qkv_res_scale,
            cumulative_query_attention=self.cumulative_query_attention,
            transformer_depths=self.transformer_branch_depths,
            enable_feature_fusion=not self.uses_branch_loss_fusion,
        )
        fused_size = self.time_branch.flat_size + self.fft_branch.flat_size
        if self.uses_cross_depth_qkv:
            # Time and FFT keep independent QKV parameters because their token
            # distributions and patch counts differ.
            self.time_cross_depth_qkv = CrossDepthQKVResidual(
                n_branches=self.transformer_branches,
                emb_size=emb_size,
                num_heads=num_heads,
                dropout=self.transformer_branch_qkv_dropout,
            )
            self.fft_cross_depth_qkv = CrossDepthQKVResidual(
                n_branches=self.transformer_branches,
                emb_size=emb_size,
                num_heads=num_heads,
                dropout=self.transformer_branch_qkv_dropout,
            )
        if self.uses_branch_loss_fusion:
            self.branch_cls_heads = nn.ModuleList(
                [head_type(fused_size, n_classes) for _ in self.transformer_branch_depths]
            )
            # Zero logits initialize all depth-loss weights to 1 / n.
            self.branch_loss_weight_logits = nn.Parameter(torch.zeros(self.transformer_branches))
        else:
            # Preserve the legacy module/state_dict layout when the new mode is disabled.
            self.cls_head = head_type(fused_size, n_classes)

    def normalized_branch_loss_weights(self) -> Tensor:
        if not self.uses_branch_loss_fusion:
            raise RuntimeError("branch-loss weights are available only in loss_softmax mode")
        return F.softmax(self.branch_loss_weight_logits, dim=0)

    def transformer_weight_metadata(self) -> dict[str, list[float] | float]:
        if self.transformer_branches == 1:
            return {}
        if self.uses_branch_loss_fusion:
            values = self.normalized_branch_loss_weights().detach().cpu().tolist()
            metadata: dict[str, list[float] | float] = {
                "transformer_branch_loss_weights": [float(value) for value in values],
            }
            if self.uses_cross_depth_qkv:
                metadata.update(
                    {
                        "time_transformer_branch_qkv_gamma": float(
                            self.time_cross_depth_qkv.gamma.detach().cpu().item()
                        ),
                        "fft_transformer_branch_qkv_gamma": float(
                            self.fft_cross_depth_qkv.gamma.detach().cpu().item()
                        ),
                    }
                )
            return metadata
        return {
            "time_transformer_branch_weights": self.time_branch.transformer_weight_values(),
            "fft_transformer_branch_weights": self.fft_branch.transformer_weight_values(),
        }

    def paired_transformer_features(
        self,
        x_time: Tensor,
        x_fft: Tensor,
    ) -> tuple[Tensor, ...]:
        """Concatenate time/FFT features for each corresponding depth."""
        time_tokens = self.time_branch.forward_transformer_branch_tokens(x_time)
        x_fft = self.fft_global_layer(x_fft)
        fft_tokens = self.fft_branch.forward_transformer_branch_tokens(x_fft)
        if self.uses_cross_depth_qkv:
            time_tokens = self.time_cross_depth_qkv(time_tokens)
            fft_tokens = self.fft_cross_depth_qkv(fft_tokens)
        time_features = self.time_branch.flatten_transformer_branch_tokens(time_tokens)
        fft_features = self.fft_branch.flatten_transformer_branch_tokens(fft_tokens)
        return tuple(
            torch.cat([time_value, fft_value], dim=1)
            for time_value, fft_value in zip(time_features, fft_features)
        )

    def forward_with_branch_logits(
        self,
        x_time: Tensor,
        x_fft: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        """Return weighted features/logits plus all independently supervised logits."""
        if not self.uses_branch_loss_fusion:
            raise RuntimeError("forward_with_branch_logits requires loss_softmax mode")
        paired_features = self.paired_transformer_features(x_time, x_fft)
        branch_logits = tuple(
            head(features)[1]
            for head, features in zip(self.branch_cls_heads, paired_features)
        )
        weights = self.normalized_branch_loss_weights()
        fused_features = (
            weights.view(-1, 1, 1) * torch.stack(paired_features, dim=0)
        ).sum(dim=0)
        fused_logits = (
            weights.view(-1, 1, 1) * torch.stack(branch_logits, dim=0)
        ).sum(dim=0)
        if self.classification_mode == "hierarchical":
            # Mix normalized joint probabilities, keeping the group probabilities coherent.
            fused_logits = torch.logsumexp(
                F.log_softmax(self.branch_loss_weight_logits, dim=0).view(-1, 1, 1)
                + torch.stack(branch_logits, dim=0), dim=0,
            )
        return fused_features, fused_logits, branch_logits

    def forward(self, x_time: Tensor, x_fft: Tensor) -> tuple[Tensor, Tensor]:
        if self.uses_branch_loss_fusion:
            features, logits, _ = self.forward_with_branch_logits(x_time, x_fft)
            return features, logits
        time_features = self.time_branch(x_time)
        x_fft = self.fft_global_layer(x_fft)
        fft_features = self.fft_branch(x_fft)
        fused_features = torch.cat([time_features, fft_features], dim=1)
        return self.cls_head(fused_features)


class ActivityConformer(nn.Module):
    """EEG-Conformer for activity three-class LOSO classification.

    Architecture is faithful to the original (shallownet PatchEmbedding →
    stacked TransformerEncoderBlocks → flatten + fc head) but adapts spatial
    conv kernel and fc input size to the actual data dimensions.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 3,
        emb_size: int = 40,
        depth: int = 6,
        num_heads: int = 5,
        dropout: float = 0.5,
        conv_type: str = DEFAULT_CONV_TYPE,
        fft_global: str = DEFAULT_FFT_GLOBAL,
        input_qkv: str = DEFAULT_INPUT_QKV,
        input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM,
        input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS,
        input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
        input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
        cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
        transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    ) -> None:
        super().__init__()
        self.conv_type = validate_conv_type(conv_type)
        self.fft_global = validate_fft_global(fft_global)
        self.input_qkv = validate_input_qkv(input_qkv)
        self.cumulative_query_attention = bool(cumulative_query_attention)
        self.fft_global_layer = (
            FFTGlobalMLP(n_times) if self.fft_global == FFT_GLOBAL_MLP else nn.Identity()
        )
        if self.input_qkv == INPUT_QKV_CHANNEL:
            self.input_qkv_layer = InputQKVResidual(
                n_times=n_times,
                d_model=input_qkv_dim,
                num_heads=input_qkv_heads,
                dropout=input_qkv_dropout,
                res_scale=input_qkv_res_scale,
            )
        elif self.input_qkv == INPUT_QKV_TIME:
            self.input_qkv_layer = InputTemporalQKVResidual(
                n_channels=n_channels,
                n_times=n_times,
                d_model=input_qkv_dim,
                num_heads=input_qkv_heads,
                dropout=input_qkv_dropout,
                res_scale=input_qkv_res_scale,
            )
        else:
            self.input_qkv_layer = nn.Identity()
        n_patches = compute_n_patches(n_times)
        self.patch_embedding = PatchEmbedding(n_channels, emb_size, dropout, conv_type=self.conv_type)
        self.encoder = TransformerEncoder(
            depth,
            emb_size,
            num_heads,
            dropout=transformer_encoder_dropout,
            cumulative_query_attention=self.cumulative_query_attention,
        )
        self.cls_head = ClassificationHead(emb_size, n_patches, n_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.fft_global_layer(x)
        x = self.input_qkv_layer(x)
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return self.cls_head(x)


def compute_n_patches(n_times: int) -> int:
    """Number of patch tokens produced by PatchEmbedding for ``n_times`` samples.

    Trace:
      temporal conv (1,25) stride (1,1)  → T' = n_times - 24
      AvgPool2d kernel (1,75) stride (1,15) → n_patches = (T' - 75) // 15 + 1
    """
    if n_times < 99:
        raise ValueError(
            f"Input sequence is too short for the original EEG-Conformer stem: "
            f"n_times={n_times}, need at least 99"
        )
    t_after_temporal = n_times - 24
    return (t_after_temporal - 75) // 15 + 1


def temporal_qkv_shape_metadata(prefix: str, n_channels: int, n_times: int) -> dict[str, int]:
    n_tokens, token_len, padded_n_times, pad_len = compute_temporal_qkv_token_shape(n_times)
    return {
        f"{prefix}input_qkv_time_tokens": int(n_tokens),
        f"{prefix}input_qkv_time_token_len": int(token_len),
        f"{prefix}input_qkv_time_padded_n_times": int(padded_n_times),
        f"{prefix}input_qkv_time_pad_len": int(pad_len),
        f"{prefix}input_qkv_time_feature_dim": int(n_channels) * int(token_len),
    }


def validate_input_domain(raw: str | None) -> str:
    value = DEFAULT_INPUT_DOMAIN if raw is None else str(raw).strip().lower()
    if value not in VALID_INPUT_DOMAINS:
        allowed = ", ".join(VALID_INPUT_DOMAINS)
        raise ValueError(f"input_domain must be one of: {allowed}")
    return value


def is_dual_input_domain(input_domain: str | None) -> bool:
    return validate_input_domain(input_domain) == DUAL_INPUT_DOMAIN


def transform_windows_to_fft(windows: np.ndarray) -> np.ndarray:
    """Convert windows to log-power rFFT representation along the time axis."""
    array = np.asarray(windows, dtype=np.float32)
    spectrum = np.fft.rfft(array, axis=-1)
    power = np.abs(spectrum) ** 2
    return np.log1p(power).astype(np.float32, copy=False)


def transform_windows_for_input_domain(
    windows: np.ndarray,
    input_domain: str,
) -> np.ndarray:
    """Return a single-domain representation for ``time`` or ``fft`` modes.

    The dual-domain ``time_fft`` mode intentionally uses
    ``prepare_split_inputs_for_input_domain`` because it needs two separately
    standardised tensors instead of one array.
    """
    resolved_input_domain = validate_input_domain(input_domain)
    array = np.asarray(windows, dtype=np.float32)
    if resolved_input_domain == DEFAULT_INPUT_DOMAIN:
        return array.astype(np.float32, copy=False)
    if resolved_input_domain == FFT_INPUT_DOMAIN:
        return transform_windows_to_fft(array)
    raise ValueError(
        "time_fft is a dual-input mode; use prepare_split_inputs_for_input_domain"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_global_dataset(
    dataset_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, subject_ids from the global activity dataset directory."""
    root = Path(dataset_root)
    X = np.load(root / "X.npy")
    y = np.load(root / "y.npy")
    subject_ids = np.load(root / "subject_ids.npy")

    if len(X) != len(y) or len(X) != len(subject_ids):
        raise ValueError(
            f"Mismatched array lengths in {root}: "
            f"X={len(X)}, y={len(y)}, subject_ids={len(subject_ids)}"
        )
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False), subject_ids.astype(np.int64, copy=False)


def loso_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    test_subject_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split global arrays into train/test by test_subject_id.

    Returns
    -------
    train_X, train_y, test_X, test_y
    """
    test_mask = subject_ids == test_subject_id
    train_mask = ~test_mask

    if not train_mask.any():
        raise ValueError(f"test_subject_id={test_subject_id} leaves no training samples")
    if not test_mask.any():
        raise ValueError(f"test_subject_id={test_subject_id} not found in subject_ids")

    # add conv-channel dim (B, C, T) → (B, 1, C, T)
    train_X = np.expand_dims(X[train_mask], axis=1)
    test_X = np.expand_dims(X[test_mask], axis=1)
    return train_X, y[train_mask], test_X, y[test_mask]


def standardize_by_train(
    train_X: np.ndarray,
    test_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalise using training set statistics only (as in original)."""
    mean = float(train_X.mean())
    std = float(train_X.std())
    if std == 0.0:
        raise ValueError("Training data std is zero – cannot standardise")
    return (train_X - mean) / std, (test_X - mean) / std


def prepare_split_inputs_for_input_domain(
    train_X: np.ndarray,
    test_X: np.ndarray,
    input_domain: str = DEFAULT_INPUT_DOMAIN,
) -> tuple[np.ndarray | tuple[np.ndarray, np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """Transform and standardise train/test arrays for the requested domain.

    ``time`` and ``fft`` return one train/test array each.  ``time_fft`` returns
    ``(time_array, fft_array)`` for each split and standardises the two domains
    independently with statistics computed from the training split only.
    """
    resolved_input_domain = validate_input_domain(input_domain)

    if resolved_input_domain == DUAL_INPUT_DOMAIN:
        train_time = transform_windows_for_input_domain(train_X, DEFAULT_INPUT_DOMAIN)
        test_time = transform_windows_for_input_domain(test_X, DEFAULT_INPUT_DOMAIN)
        train_fft = transform_windows_for_input_domain(train_X, FFT_INPUT_DOMAIN)
        test_fft = transform_windows_for_input_domain(test_X, FFT_INPUT_DOMAIN)

        train_time, test_time = standardize_by_train(train_time, test_time)
        train_fft, test_fft = standardize_by_train(train_fft, test_fft)
        return (train_time, train_fft), (test_time, test_fft)

    train_single = transform_windows_for_input_domain(train_X, resolved_input_domain)
    test_single = transform_windows_for_input_domain(test_X, resolved_input_domain)
    return standardize_by_train(train_single, test_single)


def tensor_dataset_from_inputs(
    inputs: np.ndarray | tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
) -> TensorDataset:
    label_tensor = torch.from_numpy(labels).long()
    if isinstance(inputs, tuple):
        time_X, fft_X = inputs
        return TensorDataset(
            torch.from_numpy(time_X).float(),
            torch.from_numpy(fft_X).float(),
            label_tensor,
        )
    return TensorDataset(torch.from_numpy(inputs).float(), label_tensor)


def primary_input_array(
    inputs: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    return inputs[0] if isinstance(inputs, tuple) else inputs


def build_dataloaders(
    dataset_root: str | Path,
    test_subject_id: int,
    batch_size: int,
    input_domain: str = DEFAULT_INPUT_DOMAIN,
) -> tuple[DataLoader, DataLoader, int, int, int, int, int]:
    """Load dataset, split LOSO, standardise, return loaders and shape info.

    Returns
    -------
    train_loader, test_loader, n_channels, n_times, n_classes,
    n_train_samples, n_test_samples
    """
    X, y, subject_ids = load_global_dataset(dataset_root)
    train_X, train_y, test_X, test_y = loso_split(X, y, subject_ids, test_subject_id)
    train_inputs, test_inputs = prepare_split_inputs_for_input_domain(
        train_X,
        test_X,
        input_domain,
    )

    primary_train_X = primary_input_array(train_inputs)
    primary_test_X = primary_input_array(test_inputs)
    n_channels = primary_train_X.shape[2]
    n_times = primary_train_X.shape[3]
    n_classes = int(y.max()) + 1

    train_loader = DataLoader(
        tensor_dataset_from_inputs(train_inputs, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        tensor_dataset_from_inputs(test_inputs, test_y),
        batch_size=batch_size,
        shuffle=False,
    )
    return (
        train_loader, test_loader,
        n_channels, n_times, n_classes,
        len(primary_train_X), len(primary_test_X),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def forward_model_batch_with_branches(
    model: nn.Module,
    batch: list[Tensor] | tuple[Tensor, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    """Move a batch to device and return fused logits, labels, and branch logits."""
    if len(batch) == 3:
        batch_X_time, batch_X_fft, batch_y = batch
        batch_X_time = batch_X_time.to(device)
        batch_X_fft = batch_X_fft.to(device)
        batch_y = batch_y.to(device)
        if bool(getattr(model, "uses_branch_loss_fusion", False)):
            _, logits, branch_logits = model.forward_with_branch_logits(
                batch_X_time,
                batch_X_fft,
            )
            return logits, batch_y, branch_logits
        _, logits = model(batch_X_time, batch_X_fft)
        return logits, batch_y, ()
    if len(batch) == 2:
        batch_X, batch_y = batch
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        _, logits = model(batch_X)
        return logits, batch_y, ()
    raise ValueError(f"Expected batch with 2 or 3 tensors, got {len(batch)}")


def forward_model_batch(
    model: nn.Module,
    batch: list[Tensor] | tuple[Tensor, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Backward-compatible helper returning only fused logits and labels."""
    logits, labels, _ = forward_model_batch_with_branches(model, batch, device)
    return logits, labels


def compute_model_batch_loss(
    model: nn.Module,
    logits: Tensor,
    labels: Tensor,
    criterion: nn.Module,
    branch_logits: tuple[Tensor, ...] = (),
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Compute legacy CE or learnable weighted per-depth CE losses."""
    if not branch_logits:
        return criterion(logits, labels), ()
    if not bool(getattr(model, "uses_branch_loss_fusion", False)):
        raise ValueError("branch logits were returned by a model outside loss_softmax mode")

    branch_losses = tuple(criterion(value, labels) for value in branch_logits)
    weights = model.normalized_branch_loss_weights()
    if len(branch_losses) != int(weights.numel()):
        raise ValueError("branch loss count does not match the learnable weight count")
    stacked_losses = torch.stack(branch_losses)
    weighted_loss = (weights * stacked_losses).sum()
    auxiliary_loss = float(model.branch_loss_aux_weight) * stacked_losses.mean()
    return weighted_loss + auxiliary_loss, branch_losses


def evaluate_with_branch_losses(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Return mean total loss, fused-logit accuracy, and mean branch losses."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    branch_loss_totals: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            logits, batch_y, branch_logits = forward_model_batch_with_branches(
                model,
                batch,
                device,
            )
            loss, branch_losses = compute_model_batch_loss(
                model,
                logits,
                batch_y,
                criterion,
                branch_logits,
            )
            batch_size = len(batch_y)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predict_class_labels(model, logits) == batch_y).sum().item())
            total_samples += batch_size
            if branch_losses and not branch_loss_totals:
                branch_loss_totals = [0.0] * len(branch_losses)
            for index, branch_loss in enumerate(branch_losses):
                branch_loss_totals[index] += float(branch_loss.item()) * batch_size

    mean_branch_losses = [value / total_samples for value in branch_loss_totals]
    return total_loss / total_samples, total_correct / total_samples, mean_branch_losses


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Backward-compatible evaluation returning (mean_loss, accuracy)."""
    loss, accuracy, _ = evaluate_with_branch_losses(model, dataloader, criterion, device)
    return loss, accuracy


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays for the full dataloader (no grad)."""
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            logits, batch_y = forward_model_batch(model, batch, device)
            all_pred.extend(predict_class_labels(model, logits).cpu().tolist())
            all_true.extend(batch_y.cpu().tolist())
    return np.array(all_true, dtype=np.int64), np.array(all_pred, dtype=np.int64)


def confusion_matrix_from_arrays(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[list[int]]:
    cm: list[list[int]] = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t)][int(p)] += 1
    return cm


def per_class_metrics_from_cm(cm: list[list[int]]) -> list[dict]:
    n = len(cm)
    result = []
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result.append({
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        })
    return result


def macro_f1_from_per_class(per_class: list[dict]) -> float:
    if not per_class:
        return 0.0
    return round(sum(d["f1"] for d in per_class) / len(per_class), 6)


# ---------------------------------------------------------------------------
# Training-history and resumable-checkpoint persistence
# ---------------------------------------------------------------------------

def temporary_artifact_path(path: str | Path) -> Path:
    """Return the same-directory temporary path used for atomic replacement."""
    target = Path(path)
    return target.with_name(f".{target.name}.tmp")


def _remove_temporary_artifact(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        # Cleanup must never hide the original checkpoint/write failure.
        pass


def atomic_torch_save(payload: dict, path: str | Path) -> Path:
    """Write a torch artifact without replacing the previous valid file on failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_artifact_path(target)
    _remove_temporary_artifact(temporary)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except BaseException:
        _remove_temporary_artifact(temporary)
        raise
    return target


def atomic_json_dump(
    payload: dict,
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace a JSON artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_artifact_path(target)
    _remove_temporary_artifact(temporary)
    try:
        with open(temporary, "w", encoding=encoding) as fh:
            json.dump(payload, fh, indent=2)
        os.replace(temporary, target)
    except BaseException:
        _remove_temporary_artifact(temporary)
        raise
    return target


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """Atomically replace an ``np.savez`` artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_artifact_path(target)
    _remove_temporary_artifact(temporary)
    try:
        with open(temporary, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(temporary, target)
    except BaseException:
        _remove_temporary_artifact(temporary)
        raise
    return target


def capture_training_rng_state(device: torch.device) -> dict:
    """Capture RNG streams needed to continue the next epoch."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_training_rng_state(state: dict, device: torch.device) -> None:
    """Restore RNG streams saved after the last completed epoch."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move optimizer tensors after loading a checkpoint with a different map location."""
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if torch.is_tensor(value):
                parameter_state[key] = value.to(device)


def load_torch_checkpoint(path: str | Path, map_location: torch.device) -> dict:
    """Load a trusted local checkpoint across old and new PyTorch defaults."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch 1.12 does not expose the weights_only argument.
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Resume checkpoint must contain a dict: {path}")
    return checkpoint


def validate_resume_checkpoint(
    checkpoint: dict,
    expected_training_config: dict,
    target_epochs: int,
) -> int:
    """Validate checkpoint format/config and return its completed epoch count."""
    version = int(checkpoint.get("resume_checkpoint_version", -1))
    if version != RESUME_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported resume checkpoint version {version}; "
            f"expected {RESUME_CHECKPOINT_VERSION}"
        )

    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "epoch_history",
        "training_config",
        "completed_epoch",
        "average_test_acc_sum",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Resume checkpoint is missing fields: {missing}")

    saved_config = checkpoint["training_config"]
    if not isinstance(saved_config, dict):
        raise ValueError("Resume checkpoint training_config must be a dict")
    mismatches: list[str] = []
    legacy_config_defaults = {
        "classification_mode": "flat",
        # Checkpoints created before encoder dropout was parameterized used 0.5.
        "transformer_encoder_dropout": DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    }
    for key, expected_value in expected_training_config.items():
        # Increasing --epochs is allowed; all data/model/optimizer settings must match.
        if key == "epochs":
            continue
        if key not in saved_config:
            if legacy_config_defaults.get(key) == expected_value:
                continue
            mismatches.append(f"{key}=<missing> (expected {expected_value!r})")
        elif saved_config[key] != expected_value:
            mismatches.append(
                f"{key}={saved_config[key]!r} (expected {expected_value!r})"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Resume checkpoint configuration mismatch: {details}")

    completed_epoch = int(checkpoint["completed_epoch"])
    if completed_epoch < 1:
        raise ValueError("Resume checkpoint completed_epoch must be >= 1")
    if completed_epoch > int(target_epochs):
        raise ValueError(
            f"Resume checkpoint already completed epoch {completed_epoch}, "
            f"which exceeds requested --epochs {target_epochs}"
        )
    history = checkpoint["epoch_history"]
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise ValueError(
            "Resume checkpoint epoch_history length must equal completed_epoch"
        )
    if int(history[-1].get("epoch", -1)) != completed_epoch:
        raise ValueError("Resume checkpoint history does not end at completed_epoch")
    best_epoch = checkpoint.get("best_epoch")
    if best_epoch is not None and not (1 <= int(best_epoch) <= completed_epoch):
        raise ValueError("Resume checkpoint best_epoch is outside the completed range")
    return completed_epoch


def write_epoch_history_files(
    fold_dir: str | Path,
    history: list[dict],
    metadata: dict,
) -> tuple[Path, Path]:
    """Persist per-epoch training history in CSV + JSON form."""
    root = Path(fold_dir)
    root.mkdir(parents=True, exist_ok=True)

    csv_path = root / "epoch_history.csv"
    json_path = root / "epoch_history.json"
    temporary_csv = temporary_artifact_path(csv_path)
    _remove_temporary_artifact(temporary_csv)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "best_test_acc",
        "is_best_epoch",
    ]
    try:
        with open(temporary_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow({name: row.get(name) for name in fieldnames})
        os.replace(temporary_csv, csv_path)
    except BaseException:
        _remove_temporary_artifact(temporary_csv)
        raise

    payload = dict(metadata)
    payload["history"] = history
    atomic_json_dump(payload, json_path)

    return csv_path, json_path


def append_training_log(log_path: str | Path, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(message)
        fh.write("\n")


def parse_class_weights(raw: str | list[float] | tuple[float, ...] | None) -> list[float] | None:
    if raw is None:
        return None

    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",")]
    else:
        pieces = [str(item).strip() for item in raw]

    if not pieces or any(piece == "" for piece in pieces):
        raise ValueError("class weights must be a comma-separated list like '3,3,1'")

    weights: list[float] = []
    for piece in pieces:
        try:
            value = float(piece)
        except ValueError as exc:
            raise ValueError("class weights must be numeric, e.g. '3,3,1'") from exc
        if value < 0:
            raise ValueError("class weights must be >= 0")
        weights.append(value)

    if not any(value > 0 for value in weights):
        raise ValueError("class weights must contain at least one value > 0")
    return weights


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_loso_fold(
    dataset_root: str | Path,
    test_subject_id: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: str | Path,
    seed: int = 42,
    emb_size: int = DEFAULT_EMB_SIZE,
    depth: int = DEFAULT_DEPTH,
    num_heads: int = DEFAULT_NUM_HEADS,
    dropout: float = DEFAULT_DROPOUT,
    input_domain: str = DEFAULT_INPUT_DOMAIN,
    conv_type: str = DEFAULT_CONV_TYPE,
    fft_global: str = DEFAULT_FFT_GLOBAL,
    input_qkv: str = DEFAULT_INPUT_QKV,
    input_qkv_dim: int = DEFAULT_INPUT_QKV_DIM,
    input_qkv_heads: int = DEFAULT_INPUT_QKV_HEADS,
    input_qkv_dropout: float = DEFAULT_INPUT_QKV_DROPOUT,
    input_qkv_res_scale: float = DEFAULT_INPUT_QKV_RES_SCALE,
    cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
    class_weights: list[float] | None = None,
    transformer_branch_fusion: str = DEFAULT_TRANSFORMER_BRANCH_FUSION,
    branch_loss_aux_weight: float = DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
    transformer_branch_qkv: str = DEFAULT_TRANSFORMER_BRANCH_QKV,
    resume: bool = False,
    transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    transformer_branch_qkv_dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
    architecture: str = DEFAULT_ARCHITECTURE,
    classification_mode: str = "flat",
) -> Path:
    """Train one LOSO fold and return path to metrics.json."""

    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device_obj = torch.device(device)
    architecture_value = str(architecture).strip().lower().replace("-", "").replace("_", "")
    resolved_architecture = (
        DEFAULT_ARCHITECTURE
        if architecture_value == DEFAULT_ARCHITECTURE
        else normalize_comparison_model_name(architecture)
    )
    resolved_input_domain = validate_input_domain(input_domain)
    classification_mode = validate_classification_mode(classification_mode)
    if classification_mode == "hierarchical" and (
        resolved_input_domain != DUAL_INPUT_DOMAIN or resolved_architecture != DEFAULT_ARCHITECTURE
    ):
        raise ValueError("hierarchical classification requires Conformer with --input-domain time_fft")
    resolved_conv_type = validate_conv_type(conv_type)
    resolved_fft_global = validate_fft_global_for_input_domain(resolved_input_domain, fft_global)
    resolved_input_qkv = validate_input_qkv(input_qkv)
    resolved_input_qkv_dim = int(input_qkv_dim)
    resolved_input_qkv_heads = int(input_qkv_heads)
    resolved_input_qkv_dropout = float(input_qkv_dropout)
    resolved_input_qkv_res_scale = float(input_qkv_res_scale)
    resolved_cumulative_query_attention = bool(cumulative_query_attention)
    resolved_transformer_encoder_dropout = validate_dropout_probability(
        transformer_encoder_dropout,
        "transformer_encoder_dropout",
    )
    resolved_transformer_branch_qkv_dropout = validate_dropout_probability(
        transformer_branch_qkv_dropout,
        "transformer_branch_qkv_dropout",
    )
    resolved_transformer_depths = resolve_transformer_branch_depths(
        depth=depth,
        transformer_branches=transformer_branches,
        transformer_depths=transformer_depths,
        input_domain=resolved_input_domain,
    )
    resolved_transformer_branches = len(resolved_transformer_depths)
    resolved_transformer_fusion = resolve_transformer_branch_fusion(
        transformer_branch_fusion,
        transformer_branches=resolved_transformer_branches,
        input_domain=resolved_input_domain,
    )
    resolved_branch_loss_aux_weight = resolve_branch_loss_aux_weight(
        branch_loss_aux_weight,
        resolved_transformer_fusion,
    )
    resolved_transformer_branch_qkv = resolve_transformer_branch_qkv(
        transformer_branch_qkv,
        transformer_branch_fusion=resolved_transformer_fusion,
        transformer_branches=resolved_transformer_branches,
        input_domain=resolved_input_domain,
    )
    branch_qkv_config_metadata: dict[str, str | float] = {
        "transformer_branch_qkv": resolved_transformer_branch_qkv,
    }
    if resolved_transformer_branch_qkv == TRANSFORMER_BRANCH_QKV_CROSS_DEPTH:
        branch_qkv_config_metadata.update(
            {
                "transformer_branch_qkv_dropout": resolved_transformer_branch_qkv_dropout,
                "transformer_branch_qkv_res_scale": DEFAULT_TRANSFORMER_BRANCH_QKV_RES_SCALE,
            }
        )
    transformer_weights_independent_by_domain = (
        resolved_transformer_fusion == TRANSFORMER_FUSION_SOFTMAX
    )
    loss_name = (
        "LearnableSoftmaxWeightedBranchCrossEntropyLoss"
        if resolved_transformer_fusion == TRANSFORMER_FUSION_LOSS_SOFTMAX
        else "CrossEntropyLoss"
    )

    if classification_mode == "hierarchical":
        loss_name = loss_name.replace("CrossEntropyLoss", "HierarchicalPathNLLLoss")

    (
        train_loader, test_loader,
        n_channels, n_times, n_classes,
        n_train_samples, n_test_samples,
    ) = build_dataloaders(
        dataset_root,
        test_subject_id,
        batch_size,
        input_domain=resolved_input_domain,
    )

    is_comparison_model = resolved_architecture != DEFAULT_ARCHITECTURE
    if is_comparison_model:
        incompatible_options: list[str] = []
        if resolved_input_domain != DEFAULT_INPUT_DOMAIN:
            incompatible_options.append("input_domain")
        if resolved_conv_type != DEFAULT_CONV_TYPE:
            incompatible_options.append("conv_type")
        if resolved_fft_global != DEFAULT_FFT_GLOBAL:
            incompatible_options.append("fft_global")
        if resolved_input_qkv != DEFAULT_INPUT_QKV:
            incompatible_options.append("input_qkv")
        if resolved_cumulative_query_attention:
            incompatible_options.append("cumulative_query_attention")
        if resolved_transformer_branches != DEFAULT_TRANSFORMER_BRANCHES:
            incompatible_options.append("transformer_branches")
        if incompatible_options:
            joined = ", ".join(incompatible_options)
            raise ValueError(
                f"External comparison architecture {resolved_architecture!r} uses "
                f"time-domain input only; incompatible options: {joined}"
            )

    model_type = (
        "external_comparison"
        if is_comparison_model
        else (
            "dual_branch"
            if resolved_input_domain == DUAL_INPUT_DOMAIN
            else "single_branch"
        )
    )
    branch_shape_metadata: dict[str, int] = {"n_times": int(n_times)}
    architecture_config: dict = {}
    if is_comparison_model:
        model = build_comparison_model(
            resolved_architecture,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
        ).to(device_obj)
        architecture_config = dict(model.architecture_config)
    elif resolved_input_domain == DUAL_INPUT_DOMAIN:
        dataset_tensors = train_loader.dataset.tensors  # type: ignore[attr-defined]
        time_n_times = int(dataset_tensors[0].shape[3])
        fft_n_times = int(dataset_tensors[1].shape[3])
        branch_shape_metadata.update(
            {
                "time_n_times": time_n_times,
                "fft_n_times": fft_n_times,
            }
        )
        if resolved_input_qkv == INPUT_QKV_TIME:
            branch_shape_metadata.update({"input_qkv_time_preferred_token_len": DEFAULT_INPUT_QKV_TIME_TOKEN_LEN})
            branch_shape_metadata.update(temporal_qkv_shape_metadata("time_", n_channels, time_n_times))
            branch_shape_metadata.update(temporal_qkv_shape_metadata("fft_", n_channels, fft_n_times))
        model = DualBranchActivityConformer(
            n_channels=n_channels,
            time_n_times=time_n_times,
            fft_n_times=fft_n_times,
            n_classes=n_classes,
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            transformer_encoder_dropout=resolved_transformer_encoder_dropout,
            conv_type=resolved_conv_type,
            fft_global=resolved_fft_global,
            input_qkv=resolved_input_qkv,
            input_qkv_dim=resolved_input_qkv_dim,
            input_qkv_heads=resolved_input_qkv_heads,
            input_qkv_dropout=resolved_input_qkv_dropout,
            input_qkv_res_scale=resolved_input_qkv_res_scale,
            cumulative_query_attention=resolved_cumulative_query_attention,
            transformer_depths=(
                resolved_transformer_depths
                if resolved_transformer_branches > 1
                else None
            ),
            transformer_branch_fusion=resolved_transformer_fusion,
            classification_mode=classification_mode,
            branch_loss_aux_weight=resolved_branch_loss_aux_weight,
            transformer_branch_qkv=resolved_transformer_branch_qkv,
            transformer_branch_qkv_dropout=resolved_transformer_branch_qkv_dropout,
        ).to(device_obj)
    else:
        if resolved_input_qkv == INPUT_QKV_TIME:
            branch_shape_metadata.update({"input_qkv_time_preferred_token_len": DEFAULT_INPUT_QKV_TIME_TOKEN_LEN})
            branch_shape_metadata.update(temporal_qkv_shape_metadata("", n_channels, n_times))
        model = ActivityConformer(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            transformer_encoder_dropout=resolved_transformer_encoder_dropout,
            conv_type=resolved_conv_type,
            fft_global=resolved_fft_global,
            input_qkv=resolved_input_qkv,
            input_qkv_dim=resolved_input_qkv_dim,
            input_qkv_heads=resolved_input_qkv_heads,
            input_qkv_dropout=resolved_input_qkv_dropout,
            input_qkv_res_scale=resolved_input_qkv_res_scale,
            cumulative_query_attention=resolved_cumulative_query_attention,
        ).to(device_obj)

    # Hierarchical NLL = group CE + conditional CE (only for true e1/e2).
    # Original class weights apply to each sample's full path loss.
    # Adam hyper-parameters remain identical to original.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=DEFAULT_BETAS)
    if class_weights is not None and len(class_weights) != n_classes:
        raise ValueError(
            f"class_weights length ({len(class_weights)}) must match n_classes ({n_classes})"
        )
    class_weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32, device=device_obj)
        if class_weights is not None
        else None
    )
    criterion_type = nn.NLLLoss if classification_mode == "hierarchical" else nn.CrossEntropyLoss
    criterion = criterion_type(weight=class_weight_tensor).to(device_obj)

    fold_dir = Path(output_dir) / f"fold_subject_{test_subject_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "train.log"
    history_csv_path = fold_dir / "epoch_history.csv"
    history_json_path = fold_dir / "epoch_history.json"
    resume_checkpoint_path = fold_dir / RESUME_CHECKPOINT_FILENAME
    resume_checkpoint_temporary = temporary_artifact_path(resume_checkpoint_path)
    _remove_temporary_artifact(resume_checkpoint_temporary)
    legacy_partial_artifacts = (
        bool(resume)
        and not resume_checkpoint_path.exists()
        and any(
            path.exists()
            for path in (fold_dir / "best_model.pt", history_json_path, history_csv_path)
        )
    )
    if resume and resume_checkpoint_path.exists():
        log_path.touch(exist_ok=True)
    else:
        log_path.write_text("", encoding="utf-8")
        if not resume:
            try:
                resume_checkpoint_path.unlink()
            except FileNotFoundError:
                pass

    # -1 guarantees epoch 1 becomes a valid best checkpoint even if accuracy is 0.
    best_acc = -1.0
    best_epoch: int | None = None
    aver_acc = 0.0
    best_y_true: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None
    best_transformer_weight_metadata: dict[str, list[float] | float] = {}
    best_test_branch_losses: list[float] = []
    epoch_history: list[dict] = []

    history_metadata = {
        "test_subject_id": test_subject_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "transformer_encoder_dropout": resolved_transformer_encoder_dropout,
        "input_domain": resolved_input_domain,
        "model_type": model_type,
        "conv_type": resolved_conv_type,
        "fft_global": resolved_fft_global,
        "input_qkv": resolved_input_qkv,
        "input_qkv_dim": resolved_input_qkv_dim,
        "input_qkv_heads": resolved_input_qkv_heads,
        "input_qkv_dropout": resolved_input_qkv_dropout,
        "input_qkv_res_scale": resolved_input_qkv_res_scale,
        "cumulative_query_attention": resolved_cumulative_query_attention,
        "transformer_branches": resolved_transformer_branches,
        "transformer_branch_depths": list(resolved_transformer_depths),
        "transformer_branch_fusion": resolved_transformer_fusion,
        "transformer_weights_independent_by_domain": transformer_weights_independent_by_domain,
        "classification_mode": classification_mode,
        "branch_loss_aux_weight": resolved_branch_loss_aux_weight,
        **branch_qkv_config_metadata,
        "class_weights": class_weights,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "emb_size": emb_size,
        "depth": depth,
        "num_heads": num_heads,
        "dropout": dropout,
        "optimizer": "Adam",
        "optimizer_betas": list(DEFAULT_BETAS),
        "loss": loss_name,
        **branch_shape_metadata,
    }
    if is_comparison_model:
        history_metadata.update(
            {
                "architecture": resolved_architecture,
                "architecture_config": architecture_config,
                "trainable_parameters": count_trainable_parameters(model),
            }
        )

    def log(message: str) -> None:
        print(message)
        append_training_log(log_path, message)

    def build_best_model_checkpoint(
        completed_best_epoch: int,
        transformer_weight_metadata: dict[str, list[float] | float],
    ) -> dict:
        return {
            # Keep the historical zero-based epoch field used by prediction code.
            "epoch": completed_best_epoch - 1,
            "test_subject_id": test_subject_id,
            "state_dict": model.state_dict(),
            "transformer_encoder_dropout": resolved_transformer_encoder_dropout,
            "n_channels": n_channels,
            "n_times": n_times,
            "n_classes": n_classes,
            "emb_size": emb_size,
            "depth": depth,
            "num_heads": num_heads,
            "dropout": dropout,
            "optimizer": "Adam",
            "optimizer_betas": list(DEFAULT_BETAS),
            "loss": loss_name,
            "architecture": resolved_architecture,
            "architecture_config": architecture_config,
            "trainable_parameters": count_trainable_parameters(model),
            "input_domain": resolved_input_domain,
            "model_type": model_type,
            "conv_type": resolved_conv_type,
            "fft_global": resolved_fft_global,
            "input_qkv": resolved_input_qkv,
            "input_qkv_dim": resolved_input_qkv_dim,
            "input_qkv_heads": resolved_input_qkv_heads,
            "input_qkv_dropout": resolved_input_qkv_dropout,
            "input_qkv_res_scale": resolved_input_qkv_res_scale,
            "cumulative_query_attention": resolved_cumulative_query_attention,
            "transformer_branches": resolved_transformer_branches,
            "transformer_branch_depths": list(resolved_transformer_depths),
            "transformer_branch_fusion": resolved_transformer_fusion,
            "transformer_weights_independent_by_domain": transformer_weights_independent_by_domain,
            "classification_mode": classification_mode,
            "branch_loss_aux_weight": resolved_branch_loss_aux_weight,
            **branch_qkv_config_metadata,
            **transformer_weight_metadata,
            **branch_shape_metadata,
        }

    start_epoch = 0
    resumed_from_epoch: int | None = None
    resume_rng_state: dict | None = None
    resume_message: str | None = None
    if resume and resume_checkpoint_path.exists():
        checkpoint = load_torch_checkpoint(resume_checkpoint_path, device_obj)
        start_epoch = validate_resume_checkpoint(
            checkpoint,
            expected_training_config=history_metadata,
            target_epochs=epochs,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device_obj)
        best_acc = float(checkpoint.get("best_acc", 0.0))
        best_epoch_value = checkpoint.get("best_epoch")
        best_epoch = None if best_epoch_value is None else int(best_epoch_value)
        aver_acc = float(checkpoint["average_test_acc_sum"])
        raw_best_y_true = checkpoint.get("best_y_true")
        raw_best_y_pred = checkpoint.get("best_y_pred")
        best_y_true = (
            None
            if raw_best_y_true is None
            else np.asarray(raw_best_y_true, dtype=np.int64)
        )
        best_y_pred = (
            None
            if raw_best_y_pred is None
            else np.asarray(raw_best_y_pred, dtype=np.int64)
        )
        best_transformer_weight_metadata = dict(
            checkpoint.get("best_transformer_weight_metadata", {})
        )
        best_test_branch_losses = [
            float(value)
            for value in checkpoint.get("best_test_branch_losses", [])
        ]
        epoch_history = list(checkpoint["epoch_history"])
        resume_rng_state = checkpoint["rng_state"]
        resumed_from_epoch = start_epoch
        resume_message = (
            f"[RESUME] restored model + Adam optimizer from "
            f"{resume_checkpoint_path}; completed={start_epoch}/{epochs}, "
            f"next_epoch={start_epoch + 1 if start_epoch < epochs else 'finalize'}"
        )
        # The latest state is also the best state when best_epoch equals the
        # completed epoch. Recreate best_model.pt in case interruption happened
        # after the resumable checkpoint but before the smaller best checkpoint.
        if best_epoch == start_epoch:
            atomic_torch_save(
                build_best_model_checkpoint(
                    start_epoch,
                    best_transformer_weight_metadata,
                ),
                fold_dir / "best_model.pt",
            )
        # The checkpoint is authoritative if a previous history write was interrupted.
        write_epoch_history_files(
            fold_dir=fold_dir,
            history=epoch_history,
            metadata=history_metadata,
        )
    elif resume and legacy_partial_artifacts:
        resume_message = (
            "[RESUME] legacy partial artifacts found, but no last_checkpoint.pt "
            "with optimizer state exists; restarting this fold from epoch 1"
        )
    elif resume:
        resume_message = (
            "[RESUME] no resumable checkpoint found; starting this fold from epoch 1"
        )

    if resolved_input_domain == DUAL_INPUT_DOMAIN:
        shape_text = (
            f"time_shape=(1,{n_channels},{branch_shape_metadata['time_n_times']})  "
            f"fft_shape=(1,{n_channels},{branch_shape_metadata['fft_n_times']})"
        )
    else:
        shape_text = f"shape=(1,{n_channels},{n_times})"

    qkv_shape_text = ""
    if resolved_input_qkv == INPUT_QKV_TIME:
        if resolved_input_domain == DUAL_INPUT_DOMAIN:
            qkv_shape_text = (
                f"  time_qkv_tokens={branch_shape_metadata['time_input_qkv_time_tokens']}x"
                f"{branch_shape_metadata['time_input_qkv_time_token_len']}"
                f"  fft_qkv_tokens={branch_shape_metadata['fft_input_qkv_time_tokens']}x"
                f"{branch_shape_metadata['fft_input_qkv_time_token_len']}"
            )
        else:
            qkv_shape_text = (
                f"  qkv_tokens={branch_shape_metadata['input_qkv_time_tokens']}x"
                f"{branch_shape_metadata['input_qkv_time_token_len']}"
            )

    log(
        f"\n[LOSO fold subject={test_subject_id}] "
        f"train={n_train_samples}  test={n_test_samples}  "
        f"{shape_text}  classes={n_classes}  model={model_type}  "
        f"architecture={resolved_architecture}  "
        f"conv_type={resolved_conv_type}  fft_global={resolved_fft_global}  "
        f"input_qkv={resolved_input_qkv}  "
        f"cumulative_query_attention={resolved_cumulative_query_attention}  "
        f"transformer_depths={list(resolved_transformer_depths)}  "
        f"transformer_encoder_dropout={resolved_transformer_encoder_dropout}  "
        f"classification_mode={classification_mode}  "
        f"transformer_fusion={resolved_transformer_fusion}  "
        f"branch_loss_aux_weight={resolved_branch_loss_aux_weight}  "
        f"transformer_branch_qkv={resolved_transformer_branch_qkv}  "
        f"transformer_branch_qkv_dropout={resolved_transformer_branch_qkv_dropout}"
        f"{qkv_shape_text}"
    )
    if resume_message is not None:
        log(resume_message)
    if resume_rng_state is not None:
        # Restore after model/dataloader construction and all checkpoint loading.
        restore_training_rng_state(resume_rng_state, device_obj)

    for epoch in range(start_epoch, epochs):
        # ---- train ----
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        running_branch_loss_totals: list[float] = []

        for batch in train_loader:
            optimizer.zero_grad()
            logits, batch_y, branch_logits = forward_model_batch_with_branches(
                model,
                batch,
                device_obj,
            )
            loss, branch_losses = compute_model_batch_loss(
                model,
                logits,
                batch_y,
                criterion,
                branch_logits,
            )
            loss.backward()
            optimizer.step()

            batch_size_actual = len(batch_y)
            running_loss += float(loss.item()) * batch_size_actual
            running_correct += int((predict_class_labels(model, logits) == batch_y).sum().item())
            running_samples += batch_size_actual
            if branch_losses and not running_branch_loss_totals:
                running_branch_loss_totals = [0.0] * len(branch_losses)
            for index, branch_loss in enumerate(branch_losses):
                running_branch_loss_totals[index] += (
                    float(branch_loss.item()) * batch_size_actual
                )

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples
        train_branch_losses = [
            value / running_samples for value in running_branch_loss_totals
        ]

        # ---- evaluate on test subject (every epoch, like original) ----
        test_loss, test_acc, test_branch_losses = evaluate_with_branch_losses(
            model,
            test_loader,
            criterion,
            device_obj,
        )

        aver_acc += test_acc
        is_best_epoch = test_acc > best_acc
        current_transformer_weight_metadata = (
            model.transformer_weight_metadata()
            if isinstance(model, DualBranchActivityConformer)
            else {}
        )
        if is_best_epoch:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_y_true, best_y_pred = collect_predictions(model, test_loader, device_obj)
            best_transformer_weight_metadata = current_transformer_weight_metadata
            best_test_branch_losses = list(test_branch_losses)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "best_test_acc": round(best_acc, 6),
            "is_best_epoch": is_best_epoch,
            **current_transformer_weight_metadata,
        }
        if train_branch_losses:
            epoch_record["train_transformer_branch_losses"] = [
                round(value, 6) for value in train_branch_losses
            ]
            epoch_record["test_transformer_branch_losses"] = [
                round(value, 6) for value in test_branch_losses
            ]
        epoch_history.append(epoch_record)
        if resume:
            atomic_torch_save(
                {
                    "resume_checkpoint_version": RESUME_CHECKPOINT_VERSION,
                    "completed_epoch": epoch + 1,
                    "test_subject_id": test_subject_id,
                    "training_config": history_metadata,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "average_test_acc_sum": aver_acc,
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "best_y_true": best_y_true,
                    "best_y_pred": best_y_pred,
                    "best_transformer_weight_metadata": best_transformer_weight_metadata,
                    "best_test_branch_losses": best_test_branch_losses,
                    "epoch_history": epoch_history,
                    "rng_state": capture_training_rng_state(device_obj),
                },
                resume_checkpoint_path,
            )
        if is_best_epoch:
            # Save this only after the optimizer checkpoint. If disk space runs
            # out, the previous resume point and best checkpoint remain valid.
            atomic_torch_save(
                build_best_model_checkpoint(
                    epoch + 1,
                    current_transformer_weight_metadata,
                ),
                fold_dir / "best_model.pt",
            )
        history_csv_path, history_json_path = write_epoch_history_files(
            fold_dir=fold_dir,
            history=epoch_history,
            metadata=history_metadata,
        )

        branch_log_text = ""
        if train_branch_losses:
            weights = current_transformer_weight_metadata.get(
                "transformer_branch_loss_weights",
                [],
            )
            branch_log_text = (
                f"  branch_train={[round(value, 4) for value in train_branch_losses]}"
                f"  branch_test={[round(value, 4) for value in test_branch_losses]}"
                f"  branch_weights={[round(value, 4) for value in weights]}"
            )
        log(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  "
            f"best={best_acc:.4f}"
            f"{branch_log_text}"
        )

    aver_acc /= epochs

    metrics: dict = {
        "test_subject_id": test_subject_id,
        "best_test_acc": best_acc,
        "average_test_acc": aver_acc,
        "transformer_encoder_dropout": resolved_transformer_encoder_dropout,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "architecture": resolved_architecture,
        "architecture_config": architecture_config,
        "trainable_parameters": count_trainable_parameters(model),
        "input_domain": resolved_input_domain,
        "model_type": model_type,
        "conv_type": resolved_conv_type,
        "fft_global": resolved_fft_global,
        "input_qkv": resolved_input_qkv,
        "input_qkv_dim": resolved_input_qkv_dim,
        "input_qkv_heads": resolved_input_qkv_heads,
        "input_qkv_dropout": resolved_input_qkv_dropout,
        "input_qkv_res_scale": resolved_input_qkv_res_scale,
        "cumulative_query_attention": resolved_cumulative_query_attention,
        "transformer_branches": resolved_transformer_branches,
        "transformer_branch_depths": list(resolved_transformer_depths),
        "transformer_branch_fusion": resolved_transformer_fusion,
        "transformer_weights_independent_by_domain": transformer_weights_independent_by_domain,
        "classification_mode": classification_mode,
        "branch_loss_aux_weight": resolved_branch_loss_aux_weight,
        **branch_qkv_config_metadata,
        **best_transformer_weight_metadata,
        "class_weights": class_weights,
        "emb_size": emb_size,
        "depth": depth,
        "num_heads": num_heads,
        "dropout": dropout,
        "optimizer": "Adam",
        "optimizer_betas": list(DEFAULT_BETAS),
        "loss": loss_name,
        "best_epoch": best_epoch,
        "optimizer_resume_enabled": bool(resume),
        "resumed_from_epoch": resumed_from_epoch,
        **branch_shape_metadata,
        "epoch_history_csv": str(history_csv_path),
        "epoch_history_json": str(history_json_path),
        "train_log": str(log_path),
    }
    if best_test_branch_losses:
        metrics["best_test_transformer_branch_losses"] = [
            round(value, 6) for value in best_test_branch_losses
        ]

    if best_y_true is not None and best_y_pred is not None:
        atomic_save_npz(
            fold_dir / "test_predictions.npz",
            y_true=best_y_true,
            y_pred=best_y_pred,
        )
        cm = confusion_matrix_from_arrays(best_y_true, best_y_pred, n_classes)
        pcm = per_class_metrics_from_cm(cm)
        metrics["confusion_matrix"] = cm
        metrics["per_class_metrics"] = pcm
        metrics["macro_f1"] = macro_f1_from_per_class(pcm)

    metrics_path = fold_dir / "metrics.json"
    atomic_json_dump(metrics, metrics_path, encoding="ascii")

    log(f"\nFold subject={test_subject_id}: best_acc={best_acc:.4f}  aver_acc={aver_acc:.4f}")
    log(f"Checkpoint:       {fold_dir / 'best_model.pt'}")
    log(f"Metrics:          {metrics_path}")
    log(f"Epoch history CSV:{history_csv_path}")
    log(f"Epoch history JSON:{history_json_path}")
    log(f"Train log:        {log_path}")

    # metrics.json marks a complete fold; the larger optimizer checkpoint is
    # needed only while the fold is incomplete.
    if resume:
        try:
            resume_checkpoint_path.unlink()
        except FileNotFoundError:
            pass
        _remove_temporary_artifact(resume_checkpoint_temporary)

    return metrics_path


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def cuda_is_usable() -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.cuda.is_available()


def normalize_device_name(device: str) -> str:
    normalized = device.strip()
    lowered = normalized.lower()
    if lowered in {"cpu", "cuda"}:
        return lowered
    if lowered.startswith("cuda:"):
        index = lowered.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    return normalized


def project_env_prefix() -> Path:
    return PROJECT_ROOT / ".conda-envs" / DEFAULT_ENV_NAME


def running_inside_project_env() -> bool:
    return Path(sys.executable).resolve() == (project_env_prefix() / "bin" / "python").resolve()


def build_project_local_train_command() -> str:
    return (
        f"{shlex.quote(str(project_env_prefix() / 'bin' / 'python'))} "
        f"{shlex.quote(str(Path(__file__).resolve()))}"
    )


def cuda_unavailable_message() -> str:
    return (
        "CUDA is not available in the current PyTorch environment. "
        "Choose 'cpu' or install a PyTorch/CUDA build compatible with the NVIDIA driver.\n"
        "If you created the project-local GPU env with setup_env_jupyter.py, rerun with:\n"
        f"{build_project_local_train_command()}"
    )


def validate_device(device: str) -> str:
    normalized = normalize_device_name(device)
    if not normalized:
        raise ValueError("Device must not be empty")
    try:
        parsed_device = torch.device(normalized)
    except RuntimeError as exc:
        raise ValueError(
            "Invalid device string. Use values like 'cpu', 'cuda', or 'cuda:0'."
        ) from exc
    if parsed_device.type == "cuda" and not cuda_is_usable():
        raise ValueError(cuda_unavailable_message())
    return str(parsed_device)


def maybe_rerun_in_project_env(argv: list[str], device: str) -> None:
    """If CUDA is requested but unavailable, try to rerun in the project conda env."""
    normalized_device = normalize_device_name(device)
    if not normalized_device.startswith("cuda"):
        return
    if cuda_is_usable():
        return
    if running_inside_project_env():
        return
    if os.environ.get(AUTO_RERUN_ENV_VAR) == "1":
        return

    env_prefix = project_env_prefix()
    if not env_prefix.exists():
        return

    rerun_env = os.environ.copy()
    rerun_env[AUTO_RERUN_ENV_VAR] = "1"
    completed = subprocess.run(
        [
            str(env_prefix / "bin" / "python"),
            str(Path(__file__).resolve()),
            *argv,
        ],
        check=False,
        env=rerun_env,
    )
    raise SystemExit(completed.returncode)


def prompt_path(prompt_text: str, default: Path | None = None, must_exist: bool = False) -> Path:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        candidate = Path(raw).expanduser() if raw else default
        if candidate is None:
            print("Please enter a path.")
            continue
        if must_exist and not candidate.exists():
            print(f"Path does not exist: {candidate}")
            continue
        return candidate


def prompt_int(prompt_text: str, default: int | None = None, minimum: int | None = None) -> int:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_float(prompt_text: str, default: float | None = None, minimum: float | None = None) -> float:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_text(prompt_text: str, default: str | None = None) -> str:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{default_text}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        print("Please enter a value.")


def build_noninteractive_example() -> str:
    return (
        f"{build_project_local_train_command()} "
        f"--dataset-root {shlex.quote(str(DEFAULT_DATASET_ROOT))} "
        f"--test-subject-id {DEFAULT_TEST_SUBJECT_ID} "
        f"--epochs {DEFAULT_EPOCHS} "
        f"--batch-size {DEFAULT_BATCH_SIZE} "
        f"--lr {DEFAULT_LR} "
        f"--device {DEFAULT_DEVICE} "
        f"--conv-type {DEFAULT_CONV_TYPE} "
        f"--fft-global {DEFAULT_FFT_GLOBAL} "
        f"--input-qkv {DEFAULT_INPUT_QKV} "
        f"--output-dir {shlex.quote(str(DEFAULT_OUTPUT_DIR))}"
    )


def ensure_interactive_input_available(missing_flags: list[str]) -> None:
    if not missing_flags or sys.stdin.isatty():
        return
    raise ValueError(
        "Missing required arguments for non-interactive execution: "
        + ", ".join(missing_flags)
        + "\nRun with explicit arguments, for example:\n"
        + build_noninteractive_example()
    )


def resolve_runtime_config(
    dataset_root: Path | str | None,
    test_subject_id: int | None,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    device: str | None,
    output_dir: Path | str | None,
    seed: int | None,
    input_domain: str | None = None,
    conv_type: str | None = None,
    fft_global: str | None = None,
    input_qkv: str | None = None,
    input_qkv_dim: int | None = None,
    input_qkv_heads: int | None = None,
    input_qkv_dropout: float | None = None,
    input_qkv_res_scale: float | None = None,
    cumulative_query_attention: bool = DEFAULT_CUMULATIVE_QUERY_ATTENTION,
    depth: int = DEFAULT_DEPTH,
    transformer_branches: int = DEFAULT_TRANSFORMER_BRANCHES,
    transformer_depths: list[int] | tuple[int, ...] | None = None,
    class_weights: str | list[float] | tuple[float, ...] | None = None,
    transformer_branch_fusion: str = DEFAULT_TRANSFORMER_BRANCH_FUSION,
    branch_loss_aux_weight: float = DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
    transformer_branch_qkv: str = DEFAULT_TRANSFORMER_BRANCH_QKV,
    resume: bool = False,
    transformer_encoder_dropout: float = DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
    transformer_branch_qkv_dropout: float = DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
    classification_mode: str = "flat",
) -> RuntimeConfig:
    missing_flags: list[str] = []
    if dataset_root is None:
        missing_flags.append("--dataset-root")
    if test_subject_id is None:
        missing_flags.append("--test-subject-id")
    if epochs is None:
        missing_flags.append("--epochs")
    if batch_size is None:
        missing_flags.append("--batch-size")
    if lr is None:
        missing_flags.append("--lr")
    if device is None:
        missing_flags.append("--device")
    if output_dir is None:
        missing_flags.append("--output-dir")
    ensure_interactive_input_available(missing_flags)

    if dataset_root is None:
        resolved_dataset_root = prompt_path(
            "Dataset root (global activity dataset)", default=DEFAULT_DATASET_ROOT, must_exist=True
        )
    else:
        resolved_dataset_root = Path(dataset_root).expanduser()
        if not resolved_dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {resolved_dataset_root}")

    resolved_test_subject_id = (
        prompt_int("Test subject id (LOSO fold)", default=DEFAULT_TEST_SUBJECT_ID, minimum=1)
        if test_subject_id is None
        else int(test_subject_id)
    )

    resolved_epochs = (
        prompt_int("Epochs", default=DEFAULT_EPOCHS, minimum=1)
        if epochs is None
        else int(epochs)
    )

    resolved_batch_size = (
        prompt_int("Batch size", default=DEFAULT_BATCH_SIZE, minimum=1)
        if batch_size is None
        else int(batch_size)
    )

    resolved_lr = (
        prompt_float("Learning rate", default=DEFAULT_LR, minimum=0.0)
        if lr is None
        else float(lr)
    )
    if resolved_lr <= 0:
        raise ValueError("lr must be > 0")

    default_device = DEFAULT_DEVICE if cuda_is_usable() else "cpu"
    if device is None:
        while True:
            candidate = prompt_text("Device", default=default_device)
            try:
                resolved_device = validate_device(candidate)
            except ValueError as exc:
                print(exc)
                continue
            break
    else:
        resolved_device = validate_device(str(device))

    resolved_output_dir = (
        prompt_path("Output directory", default=DEFAULT_OUTPUT_DIR)
        if output_dir is None
        else Path(output_dir).expanduser()
    )

    resolved_seed = 42 if seed is None else int(seed)
    resolved_input_domain = validate_input_domain(input_domain)
    resolved_conv_type = validate_conv_type(conv_type)
    resolved_fft_global = validate_fft_global_for_input_domain(resolved_input_domain, fft_global)
    resolved_input_qkv = validate_input_qkv(input_qkv)
    resolved_input_qkv_dim = DEFAULT_INPUT_QKV_DIM if input_qkv_dim is None else int(input_qkv_dim)
    resolved_input_qkv_heads = DEFAULT_INPUT_QKV_HEADS if input_qkv_heads is None else int(input_qkv_heads)
    resolved_input_qkv_dropout = DEFAULT_INPUT_QKV_DROPOUT if input_qkv_dropout is None else float(input_qkv_dropout)
    resolved_input_qkv_res_scale = (
        DEFAULT_INPUT_QKV_RES_SCALE if input_qkv_res_scale is None else float(input_qkv_res_scale)
    )
    resolved_transformer_encoder_dropout = validate_dropout_probability(
        transformer_encoder_dropout,
        "transformer_encoder_dropout",
    )
    resolved_transformer_branch_qkv_dropout = validate_dropout_probability(
        transformer_branch_qkv_dropout,
        "transformer_branch_qkv_dropout",
    )
    resolved_depth = int(depth)
    resolved_transformer_depths = resolve_transformer_branch_depths(
        depth=resolved_depth,
        transformer_branches=transformer_branches,
        transformer_depths=transformer_depths,
        input_domain=resolved_input_domain,
    )
    resolved_transformer_fusion = resolve_transformer_branch_fusion(
        transformer_branch_fusion,
        transformer_branches=len(resolved_transformer_depths),
        input_domain=resolved_input_domain,
    )
    resolved_branch_loss_aux_weight = resolve_branch_loss_aux_weight(
        branch_loss_aux_weight,
        resolved_transformer_fusion,
    )
    resolved_transformer_branch_qkv = resolve_transformer_branch_qkv(
        transformer_branch_qkv,
        transformer_branch_fusion=resolved_transformer_fusion,
        transformer_branches=len(resolved_transformer_depths),
        input_domain=resolved_input_domain,
    )
    resolved_class_weights = parse_class_weights(class_weights)

    return RuntimeConfig(
        dataset_root=resolved_dataset_root,
        test_subject_id=resolved_test_subject_id,
        epochs=resolved_epochs,
        batch_size=resolved_batch_size,
        lr=resolved_lr,
        device=resolved_device,
        output_dir=resolved_output_dir,
        seed=resolved_seed,
        input_domain=resolved_input_domain,
        conv_type=resolved_conv_type,
        fft_global=resolved_fft_global,
        class_weights=resolved_class_weights,
        input_qkv=resolved_input_qkv,
        input_qkv_dim=resolved_input_qkv_dim,
        input_qkv_heads=resolved_input_qkv_heads,
        input_qkv_dropout=resolved_input_qkv_dropout,
        input_qkv_res_scale=resolved_input_qkv_res_scale,
        cumulative_query_attention=bool(cumulative_query_attention),
        depth=resolved_depth,
        transformer_encoder_dropout=resolved_transformer_encoder_dropout,
        transformer_branches=len(resolved_transformer_depths),
        transformer_depths=resolved_transformer_depths,
        transformer_branch_fusion=resolved_transformer_fusion,
        branch_loss_aux_weight=resolved_branch_loss_aux_weight,
        transformer_branch_qkv=resolved_transformer_branch_qkv,
        transformer_branch_qkv_dropout=resolved_transformer_branch_qkv_dropout,
        classification_mode=validate_classification_mode(classification_mode),
        resume=bool(resume),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEG-Conformer for LOSO activity three-class classification"
    )
    parser.add_argument("--classification-mode", choices=["flat", "hierarchical"], default="flat",
                        help="flat: legacy 3-way; hierarchical: e1/e2 vs e3, then e1 vs e2 (time_fft)")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Global activity dataset directory (containing X.npy, y.npy, subject_ids.npy, metadata.json)",
    )
    parser.add_argument(
        "--test-subject-id",
        type=int,
        default=DEFAULT_TEST_SUBJECT_ID,
        help="Subject id used as the LOSO test fold (default: 1)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="TransformerEncoder block count (default: 6)")
    parser.add_argument(
        "--transformer-encoder-dropout",
        type=float,
        default=DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
        help=(
            "Shared dropout for attention weights, attention output, FFN internal, "
            "and FFN output in every TransformerEncoderBlock (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--transformer-branches",
        type=int,
        default=DEFAULT_TRANSFORMER_BRANCHES,
        help="Parallel Transformer encoders per time/FFT branch (default: 1)",
    )
    parser.add_argument(
        "--transformer-depths",
        type=int,
        nargs="+",
        default=None,
        help="Block counts for parallel encoders, e.g. 11 10 8",
    )
    parser.add_argument(
        "--transformer-branch-fusion",
        type=str,
        default=DEFAULT_TRANSFORMER_BRANCH_FUSION,
        help=(
            "Parallel-depth fusion: feature_softmax (legacy/default) or "
            "loss_softmax (independent classifier/loss per depth)"
        ),
    )
    parser.add_argument(
        "--branch-loss-aux-weight",
        type=float,
        default=DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
        help="Mean branch-loss coefficient used only with loss_softmax (default: 0.0)",
    )
    parser.add_argument(
        "--transformer-branch-qkv",
        type=str,
        default=DEFAULT_TRANSFORMER_BRANCH_QKV,
        help=(
            "QKV communication between parallel depth outputs: none (default) "
            "or cross_depth"
        ),
    )
    parser.add_argument(
        "--transformer-branch-qkv-dropout",
        type=float,
        default=DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
        help=(
            "Shared attention/output dropout inside CrossDepthQKVResidual "
            "(default: 0.1)"
        ),
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--input-domain",
        type=str,
        default=DEFAULT_INPUT_DOMAIN,
        help="Input representation: time, fft, or time_fft dual branch (default: time)",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        default=DEFAULT_CONV_TYPE,
        help="Convolution type in PatchEmbedding spatial conv: standard or dwconv (default: standard)",
    )
    parser.add_argument(
        "--fft-global",
        type=str,
        default=DEFAULT_FFT_GLOBAL,
        help="Optional global FFT preprocessor: none or mlp (default: none)",
    )
    parser.add_argument(
        "--input-qkv",
        type=str,
        default=DEFAULT_INPUT_QKV,
        help="Optional input QKV residual block: none, channel, or time (default: none)",
    )
    parser.add_argument(
        "--input-qkv-dim",
        type=int,
        default=DEFAULT_INPUT_QKV_DIM,
        help="Embedding width for --input-qkv channel/time (default: 64)",
    )
    parser.add_argument(
        "--input-qkv-heads",
        type=int,
        default=DEFAULT_INPUT_QKV_HEADS,
        help="Number of attention heads for --input-qkv channel/time (default: 4)",
    )
    parser.add_argument(
        "--input-qkv-dropout",
        type=float,
        default=DEFAULT_INPUT_QKV_DROPOUT,
        help="Dropout inside the input QKV residual block (default: 0.1)",
    )
    parser.add_argument(
        "--input-qkv-res-scale",
        type=float,
        default=DEFAULT_INPUT_QKV_RES_SCALE,
        help="Initial residual scale gamma for input QKV block (default: 0.1)",
    )
    parser.add_argument(
        "--cumulative-query-attention",
        action="store_true",
        default=DEFAULT_CUMULATIVE_QUERY_ATTENTION,
        help="Use (Q1 + ... + Qi) @ Ki^T in encoder block i (default: disabled)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Optional comma-separated class weights for CrossEntropyLoss, e.g. 3,3,1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for fold checkpoints and metrics",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an incomplete fold from last_checkpoint.pt, including Adam "
            "state, epoch history, best metrics, and RNG state"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    runtime_argv = list(sys.argv[1:] if argv is None else argv)
    if runtime_argv:
        args = parse_args(runtime_argv)
        maybe_rerun_in_project_env(runtime_argv, str(args.device))
        config = resolve_runtime_config(
            dataset_root=args.dataset_root,
            test_subject_id=args.test_subject_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
            input_domain=args.input_domain,
            conv_type=getattr(args, "conv_type", DEFAULT_CONV_TYPE),
            fft_global=getattr(args, "fft_global", DEFAULT_FFT_GLOBAL),
            input_qkv=getattr(args, "input_qkv", DEFAULT_INPUT_QKV),
            input_qkv_dim=getattr(args, "input_qkv_dim", DEFAULT_INPUT_QKV_DIM),
            input_qkv_heads=getattr(args, "input_qkv_heads", DEFAULT_INPUT_QKV_HEADS),
            input_qkv_dropout=getattr(args, "input_qkv_dropout", DEFAULT_INPUT_QKV_DROPOUT),
            input_qkv_res_scale=getattr(args, "input_qkv_res_scale", DEFAULT_INPUT_QKV_RES_SCALE),
            cumulative_query_attention=getattr(
                args, "cumulative_query_attention", DEFAULT_CUMULATIVE_QUERY_ATTENTION
            ),
            depth=getattr(args, "depth", DEFAULT_DEPTH),
            transformer_encoder_dropout=getattr(
                args,
                "transformer_encoder_dropout",
                DEFAULT_TRANSFORMER_ENCODER_DROPOUT,
            ),
            transformer_branches=getattr(
                args, "transformer_branches", DEFAULT_TRANSFORMER_BRANCHES
            ),
            transformer_depths=getattr(args, "transformer_depths", None),
            class_weights=getattr(args, "class_weights", None),
            classification_mode=getattr(args, "classification_mode", "flat"),
            transformer_branch_fusion=getattr(
                args,
                "transformer_branch_fusion",
                DEFAULT_TRANSFORMER_BRANCH_FUSION,
            ),
            branch_loss_aux_weight=getattr(
                args,
                "branch_loss_aux_weight",
                DEFAULT_BRANCH_LOSS_AUX_WEIGHT,
            ),
            transformer_branch_qkv=getattr(
                args,
                "transformer_branch_qkv",
                DEFAULT_TRANSFORMER_BRANCH_QKV,
            ),
            transformer_branch_qkv_dropout=getattr(
                args,
                "transformer_branch_qkv_dropout",
                DEFAULT_TRANSFORMER_BRANCH_QKV_DROPOUT,
            ),
            resume=bool(getattr(args, "resume", False)),
        )
    else:
        maybe_rerun_in_project_env([], DEFAULT_DEVICE)
        config = resolve_runtime_config(
            dataset_root=None,
            test_subject_id=None,
            epochs=None,
            batch_size=None,
            lr=None,
            device=None,
            output_dir=None,
            seed=None,
            input_domain=None,
            class_weights=None,
            resume=False,
        )

    train_loso_fold(
        dataset_root=config.dataset_root,
        test_subject_id=config.test_subject_id,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        output_dir=config.output_dir,
        seed=config.seed,
        input_domain=config.input_domain,
        conv_type=config.conv_type,
        fft_global=config.fft_global,
        input_qkv=config.input_qkv,
        input_qkv_dim=config.input_qkv_dim,
        input_qkv_heads=config.input_qkv_heads,
        input_qkv_dropout=config.input_qkv_dropout,
        input_qkv_res_scale=config.input_qkv_res_scale,
        cumulative_query_attention=config.cumulative_query_attention,
        depth=config.depth,
        transformer_encoder_dropout=config.transformer_encoder_dropout,
        transformer_branches=config.transformer_branches,
        transformer_depths=config.transformer_depths,
        class_weights=config.class_weights,
        transformer_branch_fusion=config.transformer_branch_fusion,
        classification_mode=config.classification_mode,
        branch_loss_aux_weight=config.branch_loss_aux_weight,
        transformer_branch_qkv=config.transformer_branch_qkv,
        transformer_branch_qkv_dropout=config.transformer_branch_qkv_dropout,
        resume=config.resume,
    )


if __name__ == "__main__":
    main()
