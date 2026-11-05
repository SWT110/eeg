"""Interpret the audited 16-model experiment using only non-weight artifacts."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from collect_architecture_benchmark import MODELS, DISPLAY, SUBJECTS, scores

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/architecture16_seed42_20261102"


def load_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def main():
    out = OUTPUT
    audit = json.loads((out / "audit_status.json").read_text())
    if not audit["all_complete"] or audit["audited"] != 240:
        raise RuntimeError("Interpretation requires all 240 audited folds")
    summaries = load_csv(out / "model_summary_audited.csv")
    for row in summaries:
        for key in row:
            if key != "model":
                row[key] = float(row[key]) if row[key] else None
    summary = {r["model"]: r for r in summaries}
    ordered = sorted(summaries, key=lambda r: r["mean_subject_macro_f1"], reverse=True)
    fold_rows = load_csv(out / "audited_fold_metrics.csv")
    by = {(r["model"], int(r["subject"])): r for r in fold_rows}
    paired = {}
    for name in MODELS:
        if name == "eegsimpleconv":
            continue
        delta = np.array([float(by["eegsimpleconv", s]["macro_f1"]) - float(by[name, s]["macro_f1"]) for s in SUBJECTS])
        paired[name] = {
            "mean_difference_pp": float(delta.mean() * 100),
            "median_difference_pp": float(np.median(delta) * 100),
            "wins": int((delta > 1e-12).sum()), "ties": int((abs(delta) <= 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
            "lower_subjects": [{"subject": s, "difference_pp": float(d * 100)}
                               for s, d in zip(SUBJECTS, delta) if d < -1e-12],
        }
    winner_counts = Counter()
    subject_winners = []
    for sid in SUBJECTS:
        values = {m: float(by[m, sid]["macro_f1"]) for m in MODELS}
        high = max(values.values())
        names = [m for m in MODELS if abs(values[m] - high) <= 1e-12]
        winner_counts.update(names)
        subject_winners.append({"subject": sid, "models": names, "macro_f1": high})
    all_e3, matrices, histories = {}, {}, {}
    for model in MODELS:
        all_e3[model] = []
        matrix = np.zeros((3, 3), dtype=np.int64)
        histories[model] = []
        for sid in SUBJECTS:
            fold = out / "received/results" / model / "seed_42" / f"fold_subject_{sid}"
            with np.load(fold / "test_predictions.npz", allow_pickle=False) as arrays:
                true, pred = arrays["y_true"], arrays["y_pred"]
                np.add.at(matrix, (true, pred), 1)
                if np.all(pred == 2):
                    all_e3[model].append(sid)
            histories[model].append(json.loads((fold / "epoch_history.json").read_text())["history"])
        matrices[model] = matrix
    majority = [scores(np.array([[0, 0, 96], [0, 0, 102], [0, 0, 307 if s == 3 else 328]])) for s in SUBJECTS]
    baseline = {k: float(np.mean([r[k] for r in majority])) for k in ("accuracy", "macro_f1", "balanced_accuracy")}
    curve_summary = {}
    for model in MODELS:
        selected_epochs = [int(by[model, sid]["best_epoch"]) for sid in SUBJECTS]
        curves = histories[model]
        curve_summary[model] = {
            "median_selected_epoch": float(np.median(selected_epochs)),
            "selected_le10": sum(v <= 10 for v in selected_epochs),
            "selected_le20": sum(v <= 20 for v in selected_epochs),
            "mean_last_test_accuracy": float(np.mean([h[-1]["test_acc"] for h in curves])),
            "mean_early20_test_accuracy": float(np.mean([[e["test_acc"] for e in h[:20]] for h in curves])),
            "mean_late20_test_accuracy": float(np.mean([[e["test_acc"] for e in h[-20:]] for h in curves])),
            "mean_late20_train_accuracy": float(np.mean([[e["train_acc"] for e in h[-20:]] for h in curves])),
            "max_uniform_epoch_mean_test_accuracy": float(np.array([[e["test_acc"] for e in h] for h in curves]).mean(axis=0).max()),
            "epoch_of_max_uniform_mean_test_accuracy": int(np.array([[e["test_acc"] for e in h] for h in curves]).mean(axis=0).argmax() + 1),
        }
    facts = {"simpleconv_paired_macro_f1": paired, "subject_winners": subject_winners,
             "subject_winner_counts": dict(winner_counts), "all_e3_subjects": all_e3,
             "majority_baseline": baseline, "curve_summary": curve_summary,
             "pooled_confusion_matrices": {m: cm.tolist() for m, cm in matrices.items()},
             "local_pt_files": len(list(out.rglob("*.pt"))), "audited_folds": 240}
    (out / "analysis_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
    make_figures(out, ordered, summary, histories)
    write_analysis(out, ordered, summary, by, paired, winner_counts, all_e3, baseline, curve_summary)
    print(json.dumps({"analysis_report": str(out / "analysis_report.md"), "audited_folds": 240,
                      "local_pt_files": facts["local_pt_files"], "simpleconv_vs_conformer": paired["eegconformer"]}, ensure_ascii=False))


def make_figures(out, ordered, summary, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = out / "figures"
    labels = [DISPLAY[r["model"]] for r in ordered]
    y = np.arange(len(ordered))
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 9), sharey=True, layout="constrained")
    means = np.array([r["mean_subject_macro_f1"] for r in ordered]) * 100
    sd = np.array([r["sd_subject_macro_f1"] for r in ordered]) * 100
    a.barh(y, means, xerr=sd, color="#3c719e", error_kw={"ecolor": "#454545", "capsize": 2})
    a.set_yticks(y, labels)
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.set_xlabel("Subject-equal Macro-F1 (%)")
    a.set_title("Mean score; whiskers = subject SD")
    for index, (value, spread) in enumerate(zip(means, sd)):
        a.text(value + spread + 1, index, f"{value:.2f}", va="center", fontsize=8)
    for cls, color, marker in [(1, "#d77e2c", "o"), (2, "#a54477", "s"), (3, "#2c8b65", "^")]:
        b.scatter([100*r[f"mean_subject_recall_e{cls}"] for r in ordered], y,
                  label=f"e{cls}", color=color, marker=marker, s=45)
    b.set_xlim(0, 105)
    b.set_xlabel("Subject-equal recall (%)")
    b.set_title("Per-class recall")
    b.legend(loc="upper left", ncol=3, frameon=False)
    for axis in (a, b):
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("16 EEG models / 15-subject LOSO / seed42\nExploratory: checkpoints selected by held-out accuracy")
    fig.savefig(directory / "f1_and_class_recall.png", dpi=180)
    fig.savefig(directory / "f1_and_class_recall.svg")
    plt.close(fig)

    selected = ["eegsimpleconv", "eegconformer", "eegtcnet", "tcformer", "mdtf", "channelpreserving"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, layout="constrained")
    for model, axis in zip(selected, axes.flat):
        train = np.array([[r["train_acc"] for r in h] for h in histories[model]])
        test = np.array([[r["test_acc"] for r in h] for h in histories[model]])
        axis.plot(np.arange(1, 201), train.mean(axis=0)*100, label="Mean train accuracy", color="#3478a4")
        axis.plot(np.arange(1, 201), test.mean(axis=0)*100, label="Mean held-out accuracy", color="#c36d3b")
        axis.set_title(DISPLAY[model])
        axis.set_ylim(0, 105)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Training trajectories: mean across 15 held-out subjects\nTrain values recorded in train mode; held-out values in eval mode")
    fig.savefig(directory / "learning_trajectories.png", dpi=180)
    plt.close(fig)


def write_analysis(out, ordered, s, by, paired, winner_counts, all_e3, baseline, curves):
    simple, conformer, tiny = s["eegsimpleconv"], s["eegconformer"], s["eegtcnet"]
    zero_e1_subjects = [sid for sid in SUBJECTS if float(by["eegsimpleconv", sid]["e1_recall"]) == 0]
    zero_e1_names = "、".join(f"S{sid}" for sid in zero_e1_subjects)
    lines = [
        "# 16种EEG模型实验分析", "", "核验与分析日期：2026-11-05。16模型 × 15名受试者 × 200轮，共240折、48,000条epoch记录均已齐全。只下载了指标、预测、历史与配置，未下载.pt权重。", "",
        "## 当前结果支持的候选", "",
        f"本轮共同训练配方下，EEGSimpleConv的受试者等权Accuracy（{simple['mean_subject_accuracy']*100:.2f}%）、Macro-F1（{simple['mean_subject_macro_f1']*100:.2f}%）、Balanced Accuracy（{simple['mean_subject_balanced_accuracy']*100:.2f}%）均为16种方法的最高观测均值。它最值得作为类别均衡目标下的下一轮复验候选。", "",
        f"EEG Conformer的Macro-F1为{conformer['mean_subject_macro_f1']*100:.2f}%，可作为CNN–Transformer结构优化参照；EEG-TCNet以{int(tiny['parameters']):,}参数取得{tiny['mean_subject_accuracy']*100:.2f}% Accuracy、{tiny['mean_subject_macro_f1']*100:.2f}% Macro-F1，是资源受限时值得保留的轻量候选。最终骨干由你按目标选择。", "",
        "## 全16模型结果", "",
        "按受试者等权Macro-F1降序呈现。每名受试者先计算三类Macro-F1，再对15人等权平均；±为受试者间标准差，单seed结果不含种子间不确定性。所有F1均对应按最高外层测试Accuracy选中的checkpoint。", "",
        "| 模型 | Accuracy % | Macro-F1 ± SD % | BA % | e1召回 % | e2召回 % | e3召回 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ordered:
        lines.append(f"| {DISPLAY[r['model']]} | {100*r['mean_subject_accuracy']:.2f} | {100*r['mean_subject_macro_f1']:.2f} ± {100*r['sd_subject_macro_f1']:.2f} | {100*r['mean_subject_balanced_accuracy']:.2f} | {100*r['mean_subject_recall_e1']:.2f} | {100*r['mean_subject_recall_e2']:.2f} | {100*r['mean_subject_recall_e3']:.2f} |")
    lines += ["", "![F1和各类召回](figures/f1_and_class_recall.png)", "",
              "## 最主要短板是同一个人的e1/e2覆盖", "",
              f"该队列包含e1/e2/e3窗口1,440/1,530/4,899个。始终预测e3的受试者等权Accuracy已有{baseline['accuracy']*100:.2f}%，Macro-F1为{baseline['macro_f1']*100:.2f}%，BA为33.33%。多数类背景必须与模型成绩一起看。", "",
              f"EEGSimpleConv的e1/e2平均召回为{simple['mean_subject_recall_e1']*100:.2f}%/{simple['mean_subject_recall_e2']*100:.2f}%，e3为{simple['mean_subject_recall_e3']*100:.2f}%。这两类弱类均为本轮最高均值，但仍有明显漏识别。其e1零召回为2/15人、e2零召回为0/15人；{zero_e1_names}的e1召回均为0，需要保留为错误定位对象。", "",
              "分别平均两类召回可能掩盖“某人只识别e1、另一人只识别e2”。逐人先取min(R1,R2)再平均，更贴近同一人两种条件都能识别的要求：", "",
              "| 模型 | 平均R1 % | 平均R2 % | 逐人min(R1,R2)均值 % | e1/e2零召回人数 |",
              "|---|---:|---:|---:|---:|"]
    for model in ["eegsimpleconv", "eegconformer", "tcformer", "eegnex", "eegtcnet", "mdtf", "channelpreserving"]:
        r = s[model]
        lines.append(f"| {DISPLAY[model]} | {r['mean_subject_recall_e1']*100:.2f} | {r['mean_subject_recall_e2']*100:.2f} | {r['mean_subject_min_recall_e1_e2']*100:.2f} | {int(r['zero_recall_subjects_e1'])}/{int(r['zero_recall_subjects_e2'])} |")
    lines += ["", f"所保存的最高Accuracy checkpoint中，EEGNet有{len(all_e3['eegnet'])}/15折把全部窗口预测成e3；EEGSimpleConv和EEG Conformer均为0/15。这是实际预测分布的复核结果，不能只凭Accuracy高低判断三类是否都在工作。", "",
              "## 逐受试者优势与例外", "",
              f"EEGSimpleConv对EEG Conformer的平均Macro-F1差值为+{paired['eegconformer']['mean_difference_pp']:.2f}个百分点，中位差+{paired['eegconformer']['median_difference_pp']:.2f}个百分点，在{paired['eegconformer']['wins']}/15名受试者上更高。对TCFormer在{paired['tcformer']['wins']}/15人上更高，平均差+{paired['tcformer']['mean_difference_pp']:.2f}个百分点。它在全16方法逐人比较中获得{winner_counts['eegsimpleconv']}/15人的最高F1。", "",
              "与EEG Conformer相比的四个例外如下；它们说明后续优化还需要保留逐人检查：", "",
              "| 受试者 | SimpleConv F1 % | Conformer F1 % | 差值 pp |", "|---|---:|---:|---:|"]
    for r in paired["eegconformer"]["lower_subjects"]:
        sid = r["subject"]
        lines.append(f"| S{sid} | {float(by['eegsimpleconv',sid]['macro_f1'])*100:.2f} | {float(by['eegconformer',sid]['macro_f1'])*100:.2f} | {r['difference_pp']:.2f} |")
    lines += ["", "这些是配对描述性差异，没有作多模型事后筛选下的显著性或独立泛化优势声明。完整逐人图见[15人F1热图](figures/subject_macro_f1.png)。", "",
              "## 计算成本：均衡表现和轻量化各有明确候选", "",
              "同一A10、batch72的GPU前向计时；训练耗时为完整200轮fold的均值。", "",
              "| 模型 | 参数量 | GPU前向 ms/batch72 | 正式训练峰值 GiB | 完整fold均时 min |",
              "|---|---:|---:|---:|---:|"]
    for model in ["eegsimpleconv", "eegconformer", "eegtcnet", "eegnex", "tcformer", "mdtf", "channelpreserving"]:
        r = s[model]
        lines.append(f"| {DISPLAY[model]} | {int(r['parameters']):,} | {r['inference_batch72_ms']:.3f} | {r['max_peak_allocated_GiB']:.3f} | {r['mean_complete_fold_seconds']/60:.2f} |")
    lines += ["",
              f"EEG-TCNet相对SimpleConv的Accuracy低{(simple['mean_subject_accuracy']-tiny['mean_subject_accuracy'])*100:.2f}个百分点，Macro-F1低{(simple['mean_subject_macro_f1']-tiny['mean_subject_macro_f1'])*100:.2f}个百分点，体现了轻量化与类别均衡的取舍。其e1/e2在同一人上的共同覆盖尤其需要改进。", "",
              f"当前MDTF与ChannelPreserving的Macro-F1分别为{s['mdtf']['mean_subject_macro_f1']*100:.2f}%/{s['channelpreserving']['mean_subject_macro_f1']*100:.2f}%，低于基础EEG Conformer的{conformer['mean_subject_macro_f1']*100:.2f}%；单fold均时则约{s['mdtf']['mean_complete_fold_seconds']/60:.0f}/{s['channelpreserving']['mean_complete_fold_seconds']/60:.0f}分钟。本轮共同配方下没有观察到复杂分支带来相应收益。训练配方、输入变换与结构适配仍需在后续消融中区分。", "",
              "F1–参数量的非支配候选为EEG-TCNet、EEGNeX、TCFormer、EEG Conformer和EEGSimpleConv；F1–GPU前向耗时的非支配候选为EEG-TCNet与EEGSimpleConv。这里的集合只描述本次测得的点，不代表充分调参后的性能上限。", "",
              "计时边界：batch72均摊时间不是batch1延迟；输入驻留GPU，不含I/O；MDTF/ChannelPreserving的外部FFT变换不含在前向计时中，FBCNet内置滤波计入。显存为框架allocated峰值，含工作空间。训练耗时含评估和断点I/O，受并发影响；MDTF S1从epoch24续训，均时只用其余14个完整任务。全16成本表见[成本分析](efficiency_findings.md)。", "",
              "## 训练轮次与跨受试者稳定性", "",
              "240折中80折（33.3%）的最高Accuracy点落在前10轮，109折（45.4%）在前20轮；仍有37折在150轮以后。各模型需要的选轮策略存在差异。", "",
              f"EEGSimpleConv被选轮次中位数为{curves['eegsimpleconv']['median_selected_epoch']:.0f}，最佳Accuracy均值{simple['mean_subject_accuracy']*100:.2f}%，第200轮均值{curves['eegsimpleconv']['mean_last_test_accuracy']*100:.2f}%，相差{(simple['mean_subject_accuracy']-curves['eegsimpleconv']['mean_last_test_accuracy'])*100:.2f}个百分点。它的前20/后20轮平均test accuracy为{curves['eegsimpleconv']['mean_early20_test_accuracy']*100:.2f}%/{curves['eegsimpleconv']['mean_late20_test_accuracy']*100:.2f}%，所以不能将峰值与末轮的差距一概解释为单调的后期下降。", "",
              "224/240折后20轮train loss低于前20轮、同时test loss更高。该现象与训练拟合、跨受试者迁移不足和置信度变化相容；单靠曲线不能确定具体原因。最佳点与末轮差值也不能作为测试选轮偏差的独立估计。", "",
              "### 为什么表格分数高于下面的平均曲线", "",
              "成绩表使用`mean_subject(max_epoch(test_acc))`：每个留出受试者各自取最高轮次，再平均。曲线使用`mean_subject(test_acc_at_epoch)`：同一epoch上对15人平均。各折峰值轮次不一致，所以这两个汇总的数值会不同。", "",
              f"例如SimpleConv的逐折峰值均值为{simple['mean_subject_accuracy']*100:.2f}%，但统一epoch平均曲线的最大值是{curves['eegsimpleconv']['max_uniform_epoch_mean_test_accuracy']*100:.2f}%（第{curves['eegsimpleconv']['epoch_of_max_uniform_mean_test_accuracy']}轮）。Conformer相应为{conformer['mean_subject_accuracy']*100:.2f}%和{curves['eegconformer']['max_uniform_epoch_mean_test_accuracy']*100:.2f}%。这不是文件或作图不一致，而是评估对逐折选轮敏感的直接表现。", "",
              "统一epoch曲线最大值也来自同一批测试记录，以上仅作汇总口径诊断，不当作新的正式分数，更不能将两者差值直接认定为独立估计的选择偏差。仅凭这些日志，尚不能确定验证集选轮后能保留多少优势；下一阶段需要用受试者验证集确认。", "",
              "![平均训练轨迹](figures/learning_trajectories.png)", "",
              "后续应通过受试者验证集确定训练轮次与调参规则，并检查多seed稳定性；仅增加训练轮数缺少这批轨迹的支持。详细曲线复核见[curve_findings.md](curve_findings.md)。", "",
              "## 下一轮可收敛的研究方向", "",
              f"1. **三类均衡识别优先：EEGSimpleConv作为主候选。** 对{zero_e1_names}的e1零召回定位，同时将类别损失权重/采样策略与架构改动分开验证。", "",
              "2. **低参数、低显存优先：EEG-TCNet作为轻量候选。** 重点观察e1/e2在同一受试者上的识别短板，而不仅是整体Accuracy。", "",
              "3. **保留Transformer路线：基础EEG Conformer作为结构参照。** 在固定输入、损失和选轮规则后，再验证多分支、跨层交互等模块的独立收益。", "",
              "这些是供你选择下一步的建议，本次没有启动新的调参或结构优化训练。", "",
              "## 本次结论的适用范围与产物", "",
              "本轮为用户确认的单seed42、15人LOSO、共同Adam配方、按最高外层测试Accuracy选epoch的探索性比较。窗口与架构已有同队列开发历史，S6/S17是事后单列；完整记录标准化使用了测试记录自身的统计量。结论限定为这套离线视频刺激条件识别流程，不将其解释为独立认知状态或新视频泛化的验证。", "",
              "适配差异已记录：SimpleConv保留128Hz输入；EEGNeX用特征通道BN；TSception保留21通道全局支路并按左右9/9处理；FBCNet有固定滤波器组；两个项目模型使用Time+FFT。各方法尚未经过架构特定的充分超参数搜索。", "",
              "复核范围：240份标签/预测、15人每类支持、48,000条epoch记录、训练配置和选中轮次；9个训练源文件及6个数据文件的最终SHA256与冻结计划一致；240份最佳权重在服务器存在且非空。未下载权重，未进行checkpoint逐份推理重放。", "",
              "- [全16模型复算表](model_summary_audited.csv)\n- [240折原始预测复算指标](audited_fold_metrics.csv)\n- [机器生成明细报告与混淆矩阵](final_report.md)\n- [数据/源码/权重存在性核查](final_integrity_audit.json)\n- [分析数字与逐人差值](analysis_facts.json)",
    ]
    (out / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
