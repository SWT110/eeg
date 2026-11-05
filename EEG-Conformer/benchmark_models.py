"""Task adapters for nine additional published EEG architecture families.

These are documented PyTorch/128-Hz task adaptations, not claims of exact
reproduction of the authors' complete training pipelines. Pinned primary
references and all nontrivial adaptations live in benchmark_vendor/sources.json.

EEG-ITNet and Deep4 reference: Braindecode (BSD-3-Clause).
EEG-ITNet reference authors: Ghaith Bouallegue, Sarthak Tayal.
See benchmark_vendor/LICENSE-braindecode.txt for the retained BSD notice.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class NormConv(nn.Conv2d):
    def __init__(self, *args, max_norm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, 2, 0, self.max_norm))
        return super().forward(x)


class NormLinear(nn.Linear):
    def __init__(self, *args, max_norm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, 2, 0, self.max_norm))
        return super().forward(x)


class SameConv(NormConv):
    """TensorFlow-style asymmetric SAME padding along time, stride=1."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, padding=0, **kwargs)

    def forward(self, x):
        total = (self.kernel_size[1] - 1) * self.dilation[1]
        return super().forward(F.pad(x, (total // 2, total - total // 2, 0, 0)))


def keras_bn(channels, dimension=2):
    constructor = nn.BatchNorm2d if dimension == 2 else nn.BatchNorm1d
    return constructor(channels, eps=1e-3, momentum=0.01)


def glorot(module):
    for layer in module.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def keras_depthwise_glorot(layer):
    """Keras fan-in/out use [H,W,input_channels,depth_multiplier]."""
    area = math.prod(layer.kernel_size)
    multiplier = layer.out_channels // layer.in_channels
    bound = math.sqrt(6 / (area * (layer.in_channels + multiplier)))
    nn.init.uniform_(layer.weight, -bound, bound)


class DeepConvNet(nn.Module):
    def __init__(self, channels, times, classes):
        super().__init__()
        layers = [nn.Conv2d(1, 25, (1, 10)), nn.Conv2d(25, 25, (channels, 1), bias=False),
                  nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3))]
        length = (times - 9) // 3
        width = 25
        for output in (50, 100, 200):
            layers += [nn.Dropout(0.5), nn.Conv2d(width, output, (1, 10), bias=False),
                       nn.BatchNorm2d(output), nn.ELU(), nn.MaxPool2d((1, 3))]
            width, length = output, (length - 9) // 3
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.classifier = nn.Linear(200 * length, classes)
        glorot(self)
        # Preserve the original final Conv2d Xavier fan-out, not Linear fan-out.
        nn.init.xavier_uniform_(self.classifier.weight.view(classes, 200, length, 1))

    def forward_features(self, x):
        return self.features(x)


class EEGNeX(nn.Module):
    def __init__(self, channels, times, classes):
        super().__init__()
        self.features = nn.Sequential(
            SameConv(1, 8, (1, 64), bias=False), keras_bn(8), nn.ELU(),
            SameConv(8, 32, (1, 64), bias=False), keras_bn(32),
            NormConv(32, 64, (channels, 1), groups=32, bias=False, max_norm=1),
            keras_bn(64), nn.ELU(), nn.AvgPool2d((1, 4), ceil_mode=True), nn.Dropout(0.5),
            SameConv(64, 32, (1, 16), dilation=(1, 2), bias=False), keras_bn(32),
            SameConv(32, 8, (1, 16), dilation=(1, 4), bias=False), keras_bn(8), nn.ELU(),
            nn.AvgPool2d((1, 4), ceil_mode=True), nn.Dropout(0.5), nn.Flatten(),
        )
        self.classifier = NormLinear(8 * math.ceil(math.ceil(times / 4) / 4), classes, max_norm=0.25)
        glorot(self)
        keras_depthwise_glorot(self.features[5])

    def forward_features(self, x):
        return self.features(x)


class CausalBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation, dropout, *, depthwise=False, keras=False):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.layers = nn.ModuleList()
        for incoming in (inputs, outputs):
            groups = incoming if depthwise else 1
            conv = nn.Conv1d(incoming, outputs, kernel, dilation=dilation, groups=groups, bias=not depthwise)
            if not depthwise:
                nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
                nn.init.zeros_(conv.bias)
            norm = keras_bn(outputs, 1) if keras else nn.BatchNorm1d(outputs)
            self.layers.append(nn.Sequential(conv, norm, nn.ELU(), nn.Dropout(dropout)))
        self.skip = nn.Identity() if inputs == outputs else nn.Conv1d(inputs, outputs, 1)
        self.activation = nn.Identity() if depthwise else nn.ELU()

    def forward(self, x):
        residual = self.skip(x)
        for layer in self.layers:
            x = layer(F.pad(x, (self.pad, 0)))
        return self.activation(x + residual)


class EEGTCNet(nn.Module):
    def __init__(self, channels, times, classes):
        super().__init__()
        self.front = nn.Sequential(
            SameConv(1, 8, (1, 32), bias=False), keras_bn(8),
            NormConv(8, 16, (channels, 1), groups=8, bias=False, max_norm=1),
            keras_bn(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.2),
            SameConv(16, 16, (1, 16), groups=16, bias=False), nn.Conv2d(16, 16, 1, bias=False),
            keras_bn(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.2),
        )
        self.tcn = nn.Sequential(CausalBlock(16, 12, 4, 1, 0.3, keras=True),
                                 CausalBlock(12, 12, 4, 2, 0.3, keras=True))
        self.classifier = NormLinear(12, classes, max_norm=0.25)
        glorot(self.front)
        glorot(self.classifier)
        glorot(self.tcn[0].skip)
        keras_depthwise_glorot(self.front[2])

    def forward_features(self, x):
        return self.tcn(self.front(x).squeeze(2))[:, :, -1]


class EEGITNet(nn.Module):
    """BSD Braindecode interpretation; temporal depthwise convs, width 14."""
    def __init__(self, channels, times, classes):
        super().__init__()
        branches = []
        for width, kernel in ((2, 16), (4, 32), (8, 64)):
            branches.append(nn.Sequential(
                SameConv(1, width, (1, kernel), bias=False), nn.BatchNorm2d(width),
                NormConv(width, width, (channels, 1), groups=width, bias=False, max_norm=1),
                nn.BatchNorm2d(width), nn.ELU()))
        self.branches = nn.ModuleList(branches)
        self.pool = nn.Sequential(nn.AvgPool1d(4), nn.Dropout(0.2))
        self.tcn = nn.Sequential(*[CausalBlock(14, 14, 4, d, 0.2, depthwise=True) for d in (1, 2, 4, 8)])
        self.reduce = nn.Sequential(nn.Conv1d(14, 14, 1), nn.BatchNorm1d(14), nn.ELU(),
                                    nn.AvgPool1d(4), nn.Dropout(0.2), nn.Flatten())
        self.classifier = NormLinear(14 * (times // 4 // 4), classes, max_norm=0.25)

    def forward_features(self, x):
        features = torch.cat([branch(x).squeeze(2) for branch in self.branches], dim=1)
        return self.reduce(self.tcn(self.pool(features)))


class EEGSimpleConv(nn.Module):
    """Author cross-subject example W109/K4/k8, with identity 128Hz resampling."""
    def __init__(self, channels, times, classes):
        super().__init__()
        kernel, width = 8, 109
        self.stem = nn.Sequential(nn.Conv1d(channels, width, kernel, padding=kernel // 2, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU())
        blocks = []
        incoming = width
        for index in range(4):
            if index:
                width = int(1.414 * width)
            blocks.append(nn.Sequential(
                nn.Conv1d(incoming, width, kernel, padding=kernel // 2, bias=False), nn.BatchNorm1d(width),
                nn.MaxPool1d(2), nn.ReLU(),
                nn.Conv1d(width, width, kernel, padding=kernel // 2, bias=False), nn.BatchNorm1d(width), nn.ReLU()))
            incoming = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(width, classes)

    def forward_features(self, x):
        return self.blocks(self.stem(x.squeeze(1))).mean(dim=-1)


class CTAttention(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value = [nn.Linear(dim, dim) for _ in range(3)]
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)  # Preserve author's embedding-size scaling.

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = [layer(x).reshape(batch, tokens, self.heads, dim // self.heads).transpose(1, 2)
                   for layer in (self.query, self.key, self.value)]
        attention = self.drop(torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1))
        return self.out((attention @ v).transpose(1, 2).reshape(batch, tokens, dim))


class CTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = CTAttention()
        self.ff = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Dropout(0.5), nn.Linear(64, 16))
        self.norm1, self.norm2 = nn.LayerNorm(16), nn.LayerNorm(16)
        self.drop1, self.drop2 = nn.Dropout(0.5), nn.Dropout(0.5)

    def forward(self, x):
        x = self.norm1(x + self.drop1(self.attention(x)))
        return self.norm2(x + self.drop2(self.ff(x)))


class CTNet(nn.Module):
    def __init__(self, channels, times, classes):
        super().__init__()
        self.front = nn.Sequential(
            SameConv(1, 8, (1, 64), bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
            SameConv(16, 16, (1, 16), bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.25))
        self.position = nn.Parameter(torch.randn(1, max(100, times // 64), 16))
        self.position_drop = nn.Dropout(0.1)
        self.transformer = nn.Sequential(*[CTBlock() for _ in range(6)])
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear((times // 8 // 8) * 16, classes))

    def forward_features(self, x):
        cnn = self.front(x).squeeze(2).transpose(1, 2) * 4
        cnn = self.position_drop(cnn + self.position[:, :cnn.shape[1]])
        return (cnn + self.transformer(cnn)).flatten(1)


class TSception(nn.Module):
    """All 21 channels globally; paired 9-left/9-right hemispheric branches."""
    def __init__(self, channels, times, classes):
        super().__init__()
        if channels != 21:
            raise ValueError("TSception adapter requires this project's explicit 21-channel layout")
        def block(inputs, outputs, kernel, stride=1, pool=8):
            return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, stride=stride),
                                 nn.LeakyReLU(), nn.AvgPool2d((1, pool)))
        self.temporal = nn.ModuleList([block(1, 15, (1, kernel)) for kernel in (64, 32, 16)])
        self.temporal_bn = nn.BatchNorm2d(15)
        self.global_spatial = block(15, 15, (21, 1), pool=2)
        self.hemisphere_spatial = block(15, 15, (9, 1), stride=(9, 1), pool=2)
        self.register_buffer("hemisphere_order", torch.tensor(list(range(0, 18, 2)) + list(range(1, 18, 2))))
        self.spatial_bn = nn.BatchNorm2d(15)
        self.fusion = block(15, 15, (3, 1), pool=4)
        self.fusion_bn = nn.BatchNorm2d(15)
        self.classifier = nn.Sequential(nn.Linear(15, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, classes))

    def forward_features(self, x):
        temporal = self.temporal_bn(torch.cat([branch(x) for branch in self.temporal], dim=-1))
        spatial = torch.cat((self.global_spatial(temporal),
                             self.hemisphere_spatial(temporal.index_select(2, self.hemisphere_order))), dim=2)
        fused = self.fusion_bn(self.fusion(self.spatial_bn(spatial)))
        return fused.mean(dim=-1).squeeze(-1)


class DeformAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * 16 * 16, bias=False)
        self.output = nn.Sequential(nn.Linear(16 * 16, dim), nn.Dropout(0.25))

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = [v.reshape(batch, tokens, 16, 16).transpose(1, 2) for v in self.qkv(x).chunk(3, dim=-1)]
        attention = torch.softmax(q @ k.transpose(-2, -1) / 4, dim=-1)
        return self.output((attention @ v).transpose(1, 2).reshape(batch, tokens, 256))


class EEGDeformer(nn.Module):
    def __init__(self, channels, times, classes):
        super().__init__()
        self.stem = nn.Sequential(
            SameConv(1, 64, (1, 51), max_norm=2), NormConv(64, 64, (channels, 1), max_norm=2),
            nn.BatchNorm2d(64), nn.ELU(), nn.MaxPool2d((1, 2)))
        self.position = nn.Parameter(torch.randn(1, 64, times // 2))
        self.layers = nn.ModuleList()
        dim = times // 2
        for _ in range(4):
            dim //= 2
            attention = DeformAttention(dim)
            ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 16), nn.GELU(), nn.Dropout(0.25),
                               nn.Linear(16, dim), nn.Dropout(0.25))
            fine = nn.Sequential(nn.Dropout(0.25), nn.Conv1d(64, 64, 51, padding=25),
                                 nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2))
            self.layers.append(nn.ModuleList((attention, ff, fine)))
        self.classifier = nn.Linear(64 * (dim + 4), classes)

    def forward_features(self, x):
        x = self.stem(x).squeeze(2) + self.position
        dense = []
        for attention, ff, fine in self.layers:
            coarse = F.max_pool1d(x, 2)
            coarse = coarse + attention(coarse)
            fine_features = fine(x)
            dense.append(torch.log(fine_features.square().mean(dim=-1).clamp_min(1e-7)))
            x = ff(coarse) + fine_features
        return torch.cat((x.flatten(1), *dense), dim=-1)


class FixedFilterBank(nn.Module):
    """Causal Chebyshev-II lfilter via exactly equivalent zero-padded convolution.

    For a length-T input with zero initial state, only the first T impulse
    samples contribute. FFT length 2T prevents circular wraparound. No labels or
    training/test distribution statistics enter these fixed coefficients.
    """
    def __init__(self, times, sfreq=128):
        super().__init__()
        import numpy as np
        from scipy import signal
        impulse = np.zeros(times, dtype=np.float64)
        impulse[0] = 1
        kernels = []
        self.coefficients = []
        for low in range(4, 40, 4):
            passband = np.array([low, low + 4]) / (sfreq / 2)
            stopband = np.array([low - 2, low + 6]) / (sfreq / 2)
            order, _ = signal.cheb2ord(passband, stopband, 3, 30)
            b, a = signal.cheby2(order, 30, stopband, btype="bandpass")
            kernels.append(signal.lfilter(b, a, impulse))
            self.coefficients.append({"b": b.tolist(), "a": a.tolist()})
        spectrum = torch.fft.rfft(torch.from_numpy(np.stack(kernels)), n=2 * times).to(torch.complex64)
        self.register_buffer("spectrum", spectrum)
        self.times = times

    def forward(self, x):
        x = x.squeeze(1)
        spectrum = torch.fft.rfft(x, n=2 * self.times)
        bands = spectrum[:, None] * self.spectrum[None, :, None]
        return torch.fft.irfft(bands, n=2 * self.times)[..., :self.times]


class FBCNet(nn.Module):
    def __init__(self, channels, times, classes):
        super().__init__()
        if times % 4:
            raise ValueError("FBCNet uses four equal temporal segments")
        self.bank = FixedFilterBank(times)
        self.spatial = nn.Sequential(NormConv(9, 9 * 32, (channels, 1), groups=9, max_norm=2),
                                      nn.BatchNorm2d(288), nn.SiLU())
        self.classifier = NormLinear(288 * 4, classes, max_norm=0.5)

    def forward_features(self, x):
        spatial = self.spatial(self.bank(x)).squeeze(2)
        segments = spatial.reshape(spatial.shape[0], 288, 4, -1)
        return torch.log(segments.var(dim=-1).clamp(1e-6, 1e6)).flatten(1)


CLASSES = {
    "deepconvnet": DeepConvNet, "fbcnet": FBCNet, "eegtcnet": EEGTCNet, "ctnet": CTNet,
    "eegnex": EEGNeX, "eegsimpleconv": EEGSimpleConv, "eegitnet": EEGITNet,
    "tsception": TSception, "eegdeformer": EEGDeformer,
}


class BenchmarkAdapter(nn.Module):
    def __init__(self, core, name, channels, times, classes):
        super().__init__()
        self.core = core
        self.classification_mode = "flat"
        sources = json.loads((Path(__file__).parent / "benchmark_vendor" / "sources.json").read_text())
        self.architecture_config = {"name": name, "version": 1, "n_channels": channels,
                                    "n_times": times, "n_classes": classes, "sampling_rate": 128,
                                    "reference": sources[name]}

    def forward(self, x):
        features = self.core.forward_features(x)
        return features, self.core.classifier(features)


def build_benchmark_model(architecture: str, n_channels: int, n_times: int, n_classes: int):
    name = architecture.lower().replace("-", "").replace("_", "")
    if name not in CLASSES:
        raise ValueError(f"Unknown benchmark architecture: {architecture}")
    core = CLASSES[name](n_channels, n_times, n_classes)
    return BenchmarkAdapter(core, name, n_channels, n_times, n_classes)
