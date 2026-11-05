"""Frozen, resumable single-GPU task queue for the approved 16-model survey.

Each task uses the existing historical LOSO loop. Test-guided epoch selection
is intentional and every result is labelled exploratory. No automatic winner.
Run workers from an immutable source snapshot, not the live project directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE))
from train_activity_loso import atomic_json_dump, train_loso_fold
from train_activity_subject_validation import metric_summary

MODELS = (
    "eegnet", "shallowconvnet", "deepconvnet", "eegconformer", "fbcnet",
    "eegtcnet", "atcnet", "tcformer", "ctnet", "eegnex", "eegsimpleconv",
    "eegitnet", "tsception", "eegdeformer", "mdtf", "channelpreserving",
)
NEW_MODELS = set(MODELS) - {
    "eegnet", "shallowconvnet", "atcnet", "tcformer", "eegconformer", "mdtf", "channelpreserving"
}
SUBJECTS = [i for i in range(1, 17) if i != 6]
DATA_FILES = ("X.npy", "y.npy", "subject_ids.npy", "record_ids.npy", "window_indices.npy", "metadata.json")
MDTF = dict(input_domain="time_fft", transformer_branches=3, transformer_depths=[11, 10, 8],
            transformer_branch_fusion="loss_softmax", branch_loss_aux_weight=0.2,
            transformer_branch_qkv="cross_depth", transformer_encoder_dropout=0.85,
            transformer_branch_qkv_dropout=0.25)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes() -> dict:
    files = [MODULE / name for name in (
        "run_architecture_benchmark.py", "benchmark_models.py", "validate_architecture_benchmark.py", "train_activity_loso.py",
        "train_activity_subject_validation.py", "comparison_models.py", "channel_preserving_model.py",
    )]
    files += sorted((MODULE / "benchmark_vendor").rglob("*.py"))
    files += sorted((MODULE / "benchmark_vendor").rglob("*.json"))
    result = {str(p.relative_to(MODULE)): digest(p) for p in files}
    result["../eeg_project_paths.py"] = digest(MODULE.parent / "eeg_project_paths.py")
    return result


def initialize(args):
    root, dataset = args.run_root.resolve(), args.dataset_root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to replace an existing run: {root}")
    x = np.load(dataset / "X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(dataset / "y.npy", allow_pickle=False)
    ids = np.load(dataset / "subject_ids.npy", allow_pickle=False)
    metadata = read_json(dataset / "metadata.json")
    if tuple(x.shape) != (7869, 21, 1920) or y.shape != (len(x),) or ids.shape != y.shape:
        raise ValueError("Expected approved 15-person [7869,21,1920] dataset")
    if sorted(np.unique(ids).tolist()) != SUBJECTS:
        raise ValueError("Subject list differs from approved cohort")
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Integer stimulus and subject IDs required")
    if metadata.get("window_seconds") != 15 or metadata.get("stride_seconds") != 3:
        raise ValueError("Unexpected window metadata")
    if sorted(metadata.get("excluded_subject_ids", [])) != [6, 17]:
        raise ValueError("Exclusion metadata differs from approved S6/S17 policy")
    if len(set(args.gpu_uuid)) != 4:
        raise ValueError("Four distinct GPU UUIDs required")
    for start in range(0, len(x), 256):
        if not np.isfinite(x[start:start + 256]).all():
            raise ValueError("Nonfinite signal values in the frozen dataset")
    for sid in SUBJECTS:
        if np.unique(y[ids == sid]).tolist() != [0, 1, 2]:
            raise ValueError(f"Subject {sid} missing a class")
    for filename in ("record_ids.npy", "window_indices.npy"):
        if np.load(dataset / filename, mmap_mode="r", allow_pickle=False).shape != y.shape:
            raise ValueError(f"Unaligned {filename}")
    tasks = [{"id": f"{model}__S{sid:02d}", "model": model, "subject": sid}
             for sid in SUBJECTS for model in MODELS]
    plan = {
        "protocol": "exploratory_test_guided_loso_16models_v1",
        "selection": "highest outer test accuracy across 200 epochs; first epoch wins ties",
        "winner_selection": "deferred to user after inspecting all metrics",
        "dataset_root": str(dataset), "shape": list(x.shape), "subjects": SUBJECTS,
        "excluded_subjects": [6, 17], "seed": 42, "epochs": 200, "batch_size": 72,
        "learning_rate": 0.0002, "adam_betas": [0.5, 0.999], "class_weights": None,
        "train_sampling": "original", "augmentation": False, "models": list(MODELS),
        "gpu_uuids": args.gpu_uuid, "tasks": tasks,
        "dataset_sha256": {name: digest(dataset / name) for name in DATA_FILES},
        "source_sha256": source_hashes(), "created_unix": time.time(),
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "cuda_runtime": torch.version.cuda, "platform": platform.platform(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "limitations": ["outer test labels select epoch", "complete-record standardization before windowing",
                        "historically reused subjects and post-hoc S6/S17 exclusion",
                        "one seed and one shared optimizer recipe, not architecture-specific optimal tuning",
                        "MDTF/ChannelPreserving use time+FFT; other input transforms are documented per model"],
    }
    stable = {k: v for k, v in plan.items() if k != "created_unix"}
    plan["fingerprint"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True)
    write_json(root / "plan.json", plan)
    for task in tasks:
        write_json(root / "tasks" / task["id"] / "task.json", task)
    print(json.dumps({"run_root": str(root), "training_folds": len(tasks), "fingerprint": plan["fingerprint"]}))


def verify_source(plan):
    if source_hashes() != plan["source_sha256"]:
        raise RuntimeError("Frozen benchmark source changed; refusing to mix implementations")


def fold_kwargs(model: str) -> dict:
    if model == "mdtf":
        return dict(architecture="eegconformer", **MDTF)
    if model == "channelpreserving":
        return dict(architecture="channelpreserving", input_domain="time_fft", depth=10)
    if model in NEW_MODELS:
        from benchmark_models import build_benchmark_model
        return dict(architecture=model, external_model_factory=build_benchmark_model)
    return dict(architecture=model)


def run_fold(args):
    plan = read_json(args.run_root / "plan.json")
    verify_source(plan)
    task = next(t for t in plan["tasks"] if t["id"] == args.task)
    torch.set_num_threads(args.cpu_threads)
    out = args.run_root / "results" / task["model"] / "seed_42"
    task_dir = args.run_root / "tasks" / task["id"]
    if (task_dir / "completed.json").exists():
        return
    device = "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; real benchmark never falls back silently to CPU")
    if args.gpu_uuid not in plan["gpu_uuids"]:
        raise ValueError("GPU is not included in the frozen allocation")
    x = np.load(Path(plan["dataset_root"]) / "X.npy", mmap_mode="r", allow_pickle=False)
    if list(x.shape) != plan["shape"]:
        raise ValueError("Input shape changed after freeze")
    # Check all small provenance/labels on each task; X hash was verified on init
    # and once again at worker startup before any real fold is claimed.
    for name in DATA_FILES[1:]:
        if digest(Path(plan["dataset_root"]) / name) != plan["dataset_sha256"][name]:
            raise RuntimeError(f"Input changed after freeze: {name}")
    torch.cuda.reset_peak_memory_stats()
    started, started_monotonic = time.time(), time.perf_counter()
    write_json(task_dir / "attempt.json", {"started_unix": started, "host": socket.gethostname(),
               "pid": os.getpid(), "gpu_uuid": args.gpu_uuid, "plan_fingerprint": plan["fingerprint"]})
    metrics_path = out / f"fold_subject_{task['subject']}" / "metrics.json"
    postprocess_only = metrics_path.exists()
    if not postprocess_only:
        metrics_path = train_loso_fold(
            dataset_root=plan["dataset_root"], test_subject_id=task["subject"],
            epochs=plan["epochs"], batch_size=plan["batch_size"], lr=plan["learning_rate"],
            seed=plan["seed"], device=device, output_dir=out, class_weights=None, resume=True,
            fail_on_nonfinite_loss=True,
            **fold_kwargs(task["model"]),
        )
    fold_dir = metrics_path.parent
    metrics = read_json(metrics_path)
    with np.load(fold_dir / "test_predictions.npz", allow_pickle=False) as saved:
        true, pred = saved["y_true"], saved["y_pred"]
    labels = np.load(Path(plan["dataset_root"]) / "y.npy", allow_pickle=False)
    ids = np.load(Path(plan["dataset_root"]) / "subject_ids.npy", allow_pickle=False)
    if not np.array_equal(true, labels[ids == task["subject"]]):
        raise RuntimeError("Saved predictions do not match the original held-out subject labels")
    recomputed = metric_summary(true, pred, 3)
    if abs(recomputed["accuracy"] - metrics["best_test_acc"]) > 1e-10:
        raise RuntimeError("Saved accuracy and predictions disagree")
    if (metrics["epochs"] != 200 or metrics["class_weights"] is not None
            or metrics["seed"] != 42 or metrics["batch_size"] != 72 or metrics["lr"] != 0.0002):
        raise RuntimeError("Fold training recipe differs from approved plan")
    history = read_json(fold_dir / "epoch_history.json")["history"]
    if len(history) != 200 or max(row["epoch"] for row in history) != 200:
        raise RuntimeError("Fold incomplete")
    best_row = max(history, key=lambda row: row["test_acc"])
    if metrics["best_epoch"] != best_row["epoch"]:
        # Recorded history is rounded; use the original 'is_best_epoch' flag
        # to resolve near-equal accuracies instead of inventing a new tie rule.
        if metrics["best_epoch"] != [r["epoch"] for r in history if r["is_best_epoch"]][-1]:
            raise RuntimeError("Checkpoint/history epoch mismatch")
    elapsed = time.perf_counter() - started_monotonic
    summary = {"task": task, "metrics_path": str(metrics_path), "metrics": recomputed,
               "best_epoch": metrics["best_epoch"], "parameters": metrics["trainable_parameters"],
               "last_session_seconds": elapsed, "resumed_from_epoch": metrics["resumed_from_epoch"],
               "postprocessing_only_retry": postprocess_only,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gpu_uuid": args.gpu_uuid,
               "finished_unix": time.time(), "plan_fingerprint": plan["fingerprint"],
               "protocol": plan["protocol"], "architecture_config": metrics["architecture_config"]}
    write_json(task_dir / "completed.json", summary)
    print(json.dumps({"completed": task["id"], "seconds": elapsed, **recomputed}))


def claim(task_dir: Path, owner: dict, stale_seconds=None) -> Path | None:
    lease = task_dir / "lease"
    try:
        lease.mkdir()
    except FileExistsError:
        # Never recycle a lease based only on wall-clock age: containers may
        # have different clocks and an orphaned training child may still live.
        # Recover only after explicitly verifying that the old process exited.
        if stale_seconds is None:
            return None
        heartbeat = lease / "heartbeat.json"
        stamp = heartbeat.stat().st_mtime if heartbeat.exists() else lease.stat().st_mtime
        if time.time() - stamp < stale_seconds:
            return None
        try:
            lease.rename(task_dir / ("stale_lease_" + uuid.uuid4().hex))
            lease.mkdir()
        except (FileNotFoundError, FileExistsError):
            return None
    write_json(lease / "heartbeat.json", {**owner, "unix": time.time()})
    return lease


def worker(args):
    plan = read_json(args.run_root / "plan.json")
    verify_source(plan)
    if args.gpu_uuid not in plan["gpu_uuids"]:
        raise ValueError("Unauthorized GPU allocation")
    for name, expected in plan["dataset_sha256"].items():
        if digest(Path(plan["dataset_root"]) / name) != expected:
            raise RuntimeError(f"Dataset changed: {name}")
    owner = {"hostname": socket.gethostname(), "pid": os.getpid(), "gpu_uuid": args.gpu_uuid}
    worker_dir = args.run_root / "workers"
    worker_dir.mkdir(exist_ok=True)
    worker_id = args.gpu_uuid.replace("GPU-", "")
    while not (args.run_root / "STOP").exists():
        available = False
        for task in plan["tasks"]:
            directory = args.run_root / "tasks" / task["id"]
            if (directory / "completed.json").exists():
                continue
            if (directory / "failed.json").exists() and not args.retry_failed:
                continue
            available = True
            lease = claim(directory, owner)
            if lease is None:
                continue
            stop = threading.Event()
            def heartbeat():
                while not stop.wait(15):
                    write_json(lease / "heartbeat.json", {**owner, "unix": time.time()})
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            write_json(worker_dir / f"{worker_id}.json", {**owner, "task": task, "started_unix": time.time()})
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "fold", "--run-root", str(args.run_root),
                       "--task", task["id"], "--gpu-uuid", args.gpu_uuid, "--cpu-threads", str(args.cpu_threads)]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                               OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
            try:
                with (directory / "process.log").open("a", buffering=1) as log:
                    result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode or not (directory / "completed.json").exists():
                    write_json(directory / "failed.json", {"returncode": result.returncode,
                               "unix": time.time(), "owner": owner, "log": str(directory / "process.log")})
                else:
                    (directory / "failed.json").unlink(missing_ok=True)
            finally:
                stop.set()
                thread.join(timeout=20)
                (lease / "heartbeat.json").unlink(missing_ok=True)
                lease.rmdir()
            aggregate(args.run_root)
            break
        else:
            aggregate(args.run_root)
            if not available:
                break
            time.sleep(15)
    write_json(worker_dir / f"{worker_id}.json", {**owner, "state": "stopped", "unix": time.time()})


def aggregate(root: Path):
    plan = read_json(root / "plan.json")
    complete, failed, running = [], [], []
    for task in plan["tasks"]:
        directory = root / "tasks" / task["id"]
        if (directory / "completed.json").exists():
            complete.append(read_json(directory / "completed.json"))
        elif (directory / "failed.json").exists():
            failed.append(task)
        elif (directory / "lease").exists():
            running.append(task)
    rows = []
    for model in MODELS:
        values = [r for r in complete if r["task"]["model"] == model]
        if not values:
            continue
        row = {"model": model, "completed_folds": len(values), "expected_folds": len(SUBJECTS)}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            scores = [v["metrics"][metric] for v in values]
            row["mean_subject_" + metric] = float(np.mean(scores))
            row["sd_subject_" + metric] = float(np.std(scores))
        for cls in range(3):
            row[f"mean_subject_recall_e{cls+1}"] = float(np.mean([v["metrics"]["per_class"][cls]["recall"] for v in values]))
        row["mean_subject_min_recall_e1_e2"] = float(np.mean([
            min(v["metrics"]["per_class"][0]["recall"], v["metrics"]["per_class"][1]["recall"]) for v in values]))
        row["parameters"] = values[0]["parameters"]
        row["max_peak_allocated_bytes"] = max(v["peak_allocated_bytes"] for v in values)
        rows.append(row)
    report = {"protocol": plan["protocol"], "winner": None, "planned": len(plan["tasks"]),
              "completed": len(complete), "failed": failed, "running": running,
              "all_complete": len(complete) == len(plan["tasks"]), "models": rows, "folds": complete,
              "updated_unix": time.time(), "limitations": plan["limitations"]}
    # Atomic, unique temporary files make independent workers safe writers.
    write_json(root / "progress.json", report)
    if rows:
        target = root / "model_summary.csv"
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--dataset-root", type=Path, required=True)
    init.add_argument("--gpu-uuid", nargs=4, required=True)
    for name in ("worker", "fold"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run-root", type=Path, required=True)
        cmd.add_argument("--gpu-uuid", required=True)
        cmd.add_argument("--cpu-threads", type=int, default=4)
        if name == "worker":
            cmd.add_argument("--retry-failed", action="store_true")
        else:
            cmd.add_argument("--task", required=True)
    status = sub.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "fold":
        run_fold(args)
    else:
        report = aggregate(args.run_root)
        print(json.dumps({k: report[k] for k in ("planned", "completed", "failed", "running", "all_complete", "models")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
