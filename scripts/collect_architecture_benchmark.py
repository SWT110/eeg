"""Collect small immutable fold artifacts, audit them, and report without choosing a winner.

Run with local_artifacts/.venv-benchmark-reporting/bin/python. --partial produces
only a progress report and audited fold CSVs until all 240 tasks are complete.
No checkpoints or raw EEG are downloaded; no training process is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/public/home/gd_110413/SWT/work/eeg"
RUN = REMOTE + "/local_artifacts/outputs/architecture16_seed42_20261102"
FINGERPRINT = "f9bcdc14fd3fd137696d74820ca4dd9a97c6e1d0cf6b3971563e9b2ebb5d959a"
MODELS = ["eegnet", "shallowconvnet", "deepconvnet", "eegconformer", "fbcnet", "eegtcnet",
          "atcnet", "tcformer", "ctnet", "eegnex", "eegsimpleconv", "eegitnet", "tsception",
          "eegdeformer", "mdtf", "channelpreserving"]
NAMES = ["EEGNet", "ShallowConvNet", "DeepConvNet", "EEG Conformer", "FBCNet", "EEG-TCNet",
         "ATCNet", "TCFormer", "CTNet", "EEGNeX", "EEGSimpleConv", "EEG-ITNet", "TSception",
         "EEG-Deformer", "MDTF", "ChannelPreserving"]
DISPLAY = dict(zip(MODELS, NAMES))
SUBJECTS = [i for i in range(1, 17) if i != 6]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def download_archive(ssh_host, remote_path, archive):
    """Stream over an ordinary SSH exec channel; do not require SFTP support."""
    temporary = archive.with_name(archive.name + ".part")
    last_error = ""
    for attempt in range(3):
        with temporary.open("wb") as stream:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                 "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", ssh_host,
                 "cat -- " + shlex.quote(remote_path)], stdout=stream, stderr=subprocess.PIPE)
        if result.returncode == 0 and tarfile.is_tarfile(temporary):
            os.replace(temporary, archive)
            return
        last_error = result.stderr.decode(errors="replace")
        print(f"SSH archive transfer attempt {attempt+1}/3 failed", flush=True)
        temporary.unlink(missing_ok=True)
    raise RuntimeError("Unable to download complete result archive: " + last_error)


def collect(output, ssh_host="school-eeg"):
    if ssh_host not in {"school-eeg", "school-eeg-2"}:
        raise ValueError("Use one of the two known shared-storage SSH entries")
    # Completed folds are immutable within this fingerprint. Reuse validated
    # local artifacts so a fragile reverse tunnel need not transfer them again.
    known = []
    local = output / "received"
    for marker in (local / "tasks").glob("*/completed.json"):
        value = read(marker)
        task = value.get("task", {})
        if value.get("plan_fingerprint") != FINGERPRINT:
            continue
        if task.get("model") not in MODELS or task.get("subject") not in SUBJECTS:
            continue
        fold = local / "results" / task["model"] / "seed_42" / f"fold_subject_{task['subject']}"
        if all((fold / name).is_file() for name in ("metrics.json", "test_predictions.npz", "epoch_history.json", "epoch_history.csv")):
            known.append(task["id"])
    script = '''import json,pathlib,tarfile,uuid
run=pathlib.Path(%r)
plan=json.loads((run/"plan.json").read_text())
assert plan["fingerprint"]==%r
known=set(json.loads(%r))
directory=run/"report_exports"
directory.mkdir(exist_ok=True)
target=directory/("small_artifacts_"+uuid.uuid4().hex+".tar.gz")
count=0
exported=0
with tarfile.open(target,"w:gz") as archive:
 for name in ("plan.json","progress.json","model_summary.csv"):
  if (run/name).is_file(): archive.add(run/name,arcname=name)
 preflight=run.parents[1]/"benchmark_preflight_20261102_v1.json"
 if preflight.exists(): archive.add(preflight,arcname="preflight.json")
 for task in plan["tasks"]:
  marker=run/"tasks"/task["id"]/"completed.json"
  if not marker.exists(): continue
  count+=1
  if task["id"] in known: continue
  exported+=1
  archive.add(marker,arcname=str(marker.relative_to(run)))
  fold=run/"results"/task["model"]/"seed_42"/("fold_subject_"+str(task["subject"]))
  for name in ("metrics.json","test_predictions.npz","epoch_history.json","epoch_history.csv"):
   path=fold/name
   if not path.exists(): raise RuntimeError("Complete task missing artifact: "+str(path))
   archive.add(path,arcname=str(path.relative_to(run)))
print(json.dumps({"archive":str(target),"completed":count,"exported":exported,"bytes":target.stat().st_size}))
''' % (RUN, FINGERPRINT, json.dumps(known))
    python = REMOTE + "/.conda-envs/eegconformer310/bin/python"
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ssh_host,
                             python + " -c " + shlex.quote(script)], check=True, text=True, capture_output=True)
    response = json.loads(result.stdout)
    if not response["archive"].startswith(RUN + "/report_exports/small_artifacts_"):
        raise ValueError("Unexpected remote export path")
    transport = output / "transport"
    transport.mkdir(parents=True, exist_ok=True)
    archive = transport / Path(response["archive"]).name
    download_archive(ssh_host, response["archive"], archive)
    destination = output / "received"
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            target.relative_to(destination.resolve())
            if member.issym() or member.islnk() or member.size > 50_000_000:
                raise ValueError("Unexpected export member")
        bundle.extractall(destination)
    print(json.dumps({"received_complete_folds": response["completed"], "local": str(destination)}))
    return destination


def scores(matrix):
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    true = matrix.diagonal()
    recall = np.divide(true, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(true, predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(2 * true, support + predicted, out=np.zeros(3), where=(support + predicted) > 0)
    return {"accuracy": float(true.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "recall": recall, "precision": precision, "f1": f1}


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(data):
    plan = read(data / "plan.json")
    if plan["fingerprint"] != FINGERPRINT or plan["models"] != MODELS or plan["subjects"] != SUBJECTS:
        raise ValueError("Export is not the approved frozen experiment")
    rows, matrices, labels = [], {}, {}
    for task in plan["tasks"]:
        marker = data / "tasks" / task["id"] / "completed.json"
        if not marker.exists():
            continue
        completed = read(marker)
        if completed["plan_fingerprint"] != FINGERPRINT or completed["task"] != task:
            raise ValueError("Task provenance mismatch")
        fold = data / "results" / task["model"] / "seed_42" / f"fold_subject_{task['subject']}"
        metrics = read(fold / "metrics.json")
        history = read(fold / "epoch_history.json")["history"]
        if [h["epoch"] for h in history] != list(range(1, 201)) or metrics["epochs"] != 200:
            raise ValueError(f"Incomplete epoch history: {task['id']}")
        if metrics["best_epoch"] != [h["epoch"] for h in history if h["is_best_epoch"]][-1]:
            raise ValueError("Best checkpoint epoch does not match history")
        if (metrics["seed"] != 42 or metrics["batch_size"] != 72 or metrics["class_weights"] is not None
                or metrics["lr"] != 0.0002 or metrics["optimizer"] != "Adam"
                or metrics["optimizer_betas"] != [0.5, 0.999]
                or metrics["classification_mode"] != "flat" or metrics["train_sampling"] != "original"):
            raise ValueError("Training recipe mismatch")
        if (metrics["test_subject_id"] != task["subject"] or metrics["n_channels"] != 21
                or metrics["n_times"] != 1920 or metrics["n_classes"] != 3
                or metrics["n_train_samples"] + metrics["n_test_samples"] != 7869):
            raise ValueError("Fold data dimensions or subject ID differ from the frozen protocol")
        expected_domain = "time_fft" if task["model"] in {"mdtf", "channelpreserving"} else "time"
        if metrics["input_domain"] != expected_domain:
            raise ValueError("Input domain differs from the approved architecture adapter")
        with np.load(fold / "test_predictions.npz", allow_pickle=False) as artifact:
            true, pred = artifact["y_true"], artifact["y_pred"]
        sid = task["subject"]
        if true.ndim != 1 or pred.shape != true.shape or not np.isin(pred, [0, 1, 2]).all():
            raise ValueError("Invalid prediction shape or class IDs")
        expected = [96, 102, 307 if sid == 3 else 328]
        if np.bincount(true, minlength=3).tolist() != expected:
            raise ValueError(f"Held-out stimulus support differs: S{sid}")
        if sid in labels and not np.array_equal(labels[sid], true):
            raise ValueError("Models were evaluated on different labels/order")
        labels[sid] = true.copy()
        matrix = np.zeros((3, 3), dtype=np.int64)
        np.add.at(matrix, (true, pred), 1)
        value = scores(matrix)
        # Legacy JSON rounds each class F1 and then its macro average to 6
        # decimals. The new completed marker retains full-precision NPZ scores.
        if (abs(value["accuracy"] - metrics["best_test_acc"]) > 1e-12
                or abs(value["macro_f1"] - metrics["macro_f1"]) > 1.01e-6
                or abs(value["macro_f1"] - completed["metrics"]["macro_f1"]) > 1e-12):
            raise ValueError("Stored metrics differ from raw prediction recomputation")
        for field in ("train_loss", "test_loss", "train_acc", "test_acc", "best_test_acc"):
            if not np.isfinite([h[field] for h in history]).all():
                raise ValueError("Nonfinite value in epoch history: " + field)
        if abs(metrics["best_test_acc"] - max(h["test_acc"] for h in history)) > 0.51e-6:
            raise ValueError("Best reported accuracy differs from the epoch history")
        matrices[(task["model"], sid)] = matrix
        row = {"model": task["model"], "subject": sid, "seed": 42, "epochs": 200,
               "best_epoch": metrics["best_epoch"], "accuracy": value["accuracy"],
               "macro_f1": value["macro_f1"], "balanced_accuracy": value["balanced_accuracy"]}
        for index in range(3):
            for metric in ("precision", "recall", "f1"):
                row[f"e{index+1}_{metric}"] = float(value[metric][index])
        row.update(parameters=metrics["trainable_parameters"],
                   peak_allocated_GiB=completed["peak_allocated_bytes"] / 1024 ** 3,
                   last_session_seconds=completed["last_session_seconds"],
                   resumed_from_epoch=completed["resumed_from_epoch"],
                   postprocessing_only_retry=completed.get("postprocessing_only_retry", False))
        rows.append(row)
    return plan, rows, matrices


def report(output, data, allow_partial):
    plan, folds, matrices = audit(data)
    csv_file(output / "audited_fold_metrics.csv", folds)
    complete = len(folds) == 240
    status = {"planned": 240, "audited": len(folds), "all_complete": complete,
              "fingerprint": FINGERPRINT, "automatic_winner": None}
    (output / "audit_status.json").write_text(json.dumps(status, indent=2))
    if not complete:
        text = [f"# 实验进度：已核验 {len(folds)}/240 折", "", "16模型 × 15受试者 × 200epoch，seed42。完整实验尚未完成，当前不作最终模型排名。", "",
                "| 模型 | 已完成并核验的折数 |", "|---|---:|"]
        text += [f"| {DISPLAY[m]} | {sum(row['model'] == m for row in folds)}/15 |" for m in MODELS]
        text += ["", "逐折指标见audited_fold_metrics.csv。所有结果沿用用户确认的最高外层测试accuracy选轮口径，属于探索性开发比较。"]
        (output / "progress_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        if not allow_partial:
            raise RuntimeError(f"Only {len(folds)}/240 folds complete; final report withheld")
        return status
    summaries = []
    preflight = read(data / "preflight.json") if (data / "preflight.json").exists() else read(output / "preflight.json")
    timing = {row["model"]: row for row in preflight}
    for model in MODELS:
        rows = [row for row in folds if row["model"] == model]
        if sorted(row["subject"] for row in rows) != SUBJECTS:
            raise ValueError("Incomplete or duplicate model/subject cells")
        if len(set(row["parameters"] for row in rows)) != 1:
            raise ValueError("Architecture capacity changed between folds")
        pooled = sum((matrices[model, sid] for sid in SUBJECTS), np.zeros((3, 3), dtype=np.int64))
        combined = scores(pooled)
        row = {"model": model, "folds": 15}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            row["mean_subject_" + metric] = float(np.mean([r[metric] for r in rows]))
            row["sd_subject_" + metric] = float(np.std([r[metric] for r in rows], ddof=0))
            row["pooled_" + metric] = combined[metric]
        for index in range(3):
            row[f"mean_subject_recall_e{index+1}"] = float(np.mean([r[f"e{index+1}_recall"] for r in rows]))
            row[f"pooled_recall_e{index+1}"] = float(combined["recall"][index])
            row[f"zero_recall_subjects_e{index+1}"] = sum(r[f"e{index+1}_recall"] == 0 for r in rows)
        row["mean_subject_min_recall_e1_e2"] = float(np.mean([min(r["e1_recall"], r["e2_recall"]) for r in rows]))
        row["parameters"] = rows[0]["parameters"]
        row["max_peak_allocated_GiB"] = max(r["peak_allocated_GiB"] for r in rows)
        row["inference_batch72_ms"] = timing[model]["inference_batch_ms"]
        row["inference_batch72_ms_per_window"] = timing[model]["inference_ms_per_window"]
        row["fully_timed_folds"] = sum(not r["resumed_from_epoch"] and not r["postprocessing_only_retry"] for r in rows)
        row["mean_complete_fold_seconds"] = float(np.mean([
            r["last_session_seconds"] for r in rows if not r["resumed_from_epoch"] and not r["postprocessing_only_retry"]])) if row["fully_timed_folds"] else None
        summaries.append(row)
    csv_file(output / "model_summary_audited.csv", summaries)
    make_figures(output, summaries, folds, matrices)
    lines = ["# EEG 16模型探索性比较结果", "", "完整完成并复核：16模型 × 15受试者 × 200epoch，共240折，seed42。", "",
             "以下按预定模型顺序列出，最终骨干由你结合数据选择。Acc、Macro-F1、BA先逐受试者计算，再对15人等权平均；±为受试者间总体标准差，不是多个seed的波动或置信区间。", "",
             "| 模型 | Acc ± SD | Macro-F1 ± SD | BA | e1召回 | e2召回 | e3召回 | 参数量 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {DISPLAY[row['model']]} | {100*row['mean_subject_accuracy']:.2f} ± {100*row['sd_subject_accuracy']:.2f}% | {100*row['mean_subject_macro_f1']:.2f} ± {100*row['sd_subject_macro_f1']:.2f}% | {100*row['mean_subject_balanced_accuracy']:.2f}% | {100*row['mean_subject_recall_e1']:.2f}% | {100*row['mean_subject_recall_e2']:.2f}% | {100*row['mean_subject_recall_e3']:.2f}% | {row['parameters']:,} |")
    majority = [scores(np.array([[0, 0, 96], [0, 0, 102], [0, 0, 307 if sid == 3 else 328]])) for sid in SUBJECTS]
    lines += ["", f"多数类参照：始终预测e3时，受试者等权Acc为{100*np.mean([v['accuracy'] for v in majority]):.2f}%，Macro-F1为{100*np.mean([v['macro_f1'] for v in majority]):.2f}%，BA为33.33%。", "",
              "![三项指标](figures/model_scores.png)", "", "![逐人Macro-F1](figures/subject_macro_f1.png)", "", "![混淆矩阵](figures/confusion_matrices.png)", "",
              "所有模型使用共同训练配方，尚未进行架构特定调参或多seed确认。最高外层测试accuracy用于选epoch、整条视频记录参与自身标准化、固定视频顺序以及历史队列事后排除均限制结论，结果属于探索性开发比较。S6/S17不在正式15人评估中，不将其历史使用解释为独立未触碰测试。", "",
              "时域/FFT双分支、FBCNet滤波器组、TSception电极布局、SimpleConv采样率以及EEGNeX的BN适配见冻结sources.json；它们属于明确记录的方法差异。", "",
              "完整逐折指标见audited_fold_metrics.csv，等权/pooled指标、各类召回、零召回人数、参数/显存/计时见model_summary_audited.csv。推理计时为同一A10的batch72 GPU前向测量，不含I/O；均摊每窗口时间不等于batch1实时延迟。训练耗时包含数据准备、评估和断点写盘，受并发共享存储影响；中断恢复折只记录最后会话耗时，不冒充完整200轮耗时。", "",
              "原始指标、预测与200轮历史位于received/；最佳权重保留在服务器，不需要为分析下载全部大权重。"]
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    completion = ("# 实验已完成：240/240折\n\n"
                  "16个模型均完成15折，每折200轮，seed42；全部原始预测、标签、训练历史与配置已复核。\n\n"
                  "完整结果见[比较报告](final_report.md)、[模型指标表](model_summary_audited.csv)和[逐折指标](audited_fold_metrics.csv)。\n\n"
                  "最终骨干由用户结合完整结果选择，所有指标沿用已确认的探索性测试选轮口径。\n")
    (output / "progress_report.md").write_text(completion, encoding="utf-8")
    return status


def make_figures(output, summaries, folds, matrices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1", "balanced_accuracy"), ("Accuracy", "Macro-F1", "Balanced accuracy")):
        axis.bar(x, [100*r["mean_subject_"+metric] for r in summaries],
                 yerr=[100*r["sd_subject_"+metric] for r in summaries], color="#4778ab", capsize=2)
        axis.set_xticks(x, NAMES, rotation=90, fontsize=8)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Subject-equal score (%)")
        axis.set_title(title + " (subject SD)")
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Exploratory LOSO: 15 subjects / seed42 / test-guided epoch selection")
    fig.savefig(directory / "model_scores.png", dpi=180)
    plt.close(fig)
    grid = np.array([[next(r["macro_f1"] for r in folds if r["model"] == m and r["subject"] == s) for s in SUBJECTS] for m in MODELS])
    fig, axis = plt.subplots(figsize=(13, 8), layout="constrained")
    plot = axis.imshow(100*grid, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    axis.set_xticks(range(15), [f"S{s}" for s in SUBJECTS])
    axis.set_yticks(range(16), NAMES)
    for i in range(16):
        for j in range(15):
            axis.text(j, i, f"{grid[i,j]*100:.0f}", ha="center", va="center", fontsize=7,
                      color="white" if grid[i,j] > .6 else "black")
    axis.set_title("Held-out subject Macro-F1 (%) / exploratory")
    fig.colorbar(plot, ax=axis)
    fig.savefig(directory / "subject_macro_f1.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(15, 14), layout="constrained")
    for model, axis in zip(MODELS, axes.flat):
        matrix = sum((matrices[model, s] for s in SUBJECTS), np.zeros((3, 3), dtype=np.int64))
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{matrix[i,j]}\n{100*normalized[i,j]:.1f}%", ha="center", va="center", fontsize=8,
                          color="white" if normalized[i,j] > .6 else "black")
        axis.set_xticks(range(3), ["e1", "e2", "e3"])
        axis.set_yticks(range(3), ["e1", "e2", "e3"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(DISPLAY[model])
    fig.suptitle("Pooled confusion matrices: counts and row recall / exploratory")
    fig.colorbar(image, ax=axes, shrink=.65)
    fig.savefig(directory / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/architecture16_seed42_20261102")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--ssh-host", choices=["school-eeg", "school-eeg-2"], default="school-eeg",
                        help="Read-only shared-storage entry; never launches GPU work")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = collect(args.output, args.ssh_host) if args.fetch else args.output / "received"
    print(json.dumps(report(args.output, data, args.partial)))


if __name__ == "__main__":
    main()
