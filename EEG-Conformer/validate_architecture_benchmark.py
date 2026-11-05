"""Preflight at the approved real batch/time/channel sizes, on one free GPU."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from benchmark_models import build_benchmark_model
from run_architecture_benchmark import MODELS, NEW_MODELS
from train_activity_loso import (
    ActivityConformer, DualBranchActivityConformer, build_comparison_model,
    compute_model_batch_loss,
)
from channel_preserving_model import ChannelPreservingDualConformer


def build(name):
    if name in NEW_MODELS:
        return build_benchmark_model(name, 21, 1920, 3)
    if name == "mdtf":
        return DualBranchActivityConformer(
            21, 1920, 961, n_classes=3, emb_size=40, depth=6, num_heads=5,
            dropout=0.5, transformer_encoder_dropout=0.85, transformer_depths=[11, 10, 8],
            transformer_branch_fusion="loss_softmax", branch_loss_aux_weight=0.2,
            transformer_branch_qkv="cross_depth", transformer_branch_qkv_dropout=0.25)
    if name == "channelpreserving":
        return ChannelPreservingDualConformer(21, 1920, 961)
    if name == "eegconformer":
        return ActivityConformer(21, 1920)
    return build_comparison_model(name, 21, 1920, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight required")
    rows = []
    for name in args.models:
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build(name).cuda()
        x = torch.randn(72, 1, 21, 1920, device="cuda")
        fft = torch.randn(72, 1, 21, 961, device="cuda") if name in {"mdtf", "channelpreserving"} else None
        labels = torch.arange(72, device="cuda") % 3
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            branches = ()
            if name == "mdtf":
                _, logits, branches = model.forward_with_branch_logits(x, fft)
            else:
                _, logits = model(x, fft) if fft is not None else model(x)
            if tuple(logits.shape) != (72, 3):
                raise AssertionError((name, logits.shape))
            loss, _ = compute_model_batch_loss(model, logits, labels, torch.nn.CrossEntropyLoss(), branches)
            if not torch.isfinite(loss):
                raise ArithmeticError(f"Nonfinite loss: {name}")
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ArithmeticError(f"Nonfinite gradient: {name}")
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            _, prediction = model(x[:2], fft[:2]) if fft is not None else model(x[:2])
            if not torch.isfinite(prediction).all():
                raise ArithmeticError(f"Nonfinite evaluation: {name}")
            for _warmup in range(3):
                model(x, fft) if fft is not None else model(x)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _repeat in range(10):
                model(x, fft) if fft is not None else model(x)
            end.record()
            torch.cuda.synchronize()
            inference_batch_ms = begin.elapsed_time(end) / 10
        row = {"model": name, "passed": True, "batch_size": 72, "shape": [21, 1920],
               "training_steps": 2, "seconds": elapsed,
               "parameters": sum(p.numel() for p in model.parameters()),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "inference_batch_ms": inference_batch_ms,
               "inference_ms_per_window": inference_batch_ms / 72,
               "inference_timing": "GPU forward only; batch72; 3 warmups; 10 repeats; excludes IO",
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
        del model, optimizer, x, fft, labels, loss, logits, branches, prediction
        gc.collect()
    print("ALL_PREFLIGHTS_PASSED", flush=True)


if __name__ == "__main__":
    main()
