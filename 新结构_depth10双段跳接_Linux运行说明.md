# 新结构：保留电极通道、10 层双段跳接

本次入口为 `EEG-Conformer/train_activity_channelpreserve.py`。Time / FFT 各一条独立的 10 层链：前 5 层保留 21 个电极，入口的 21→21 卷积特征分别加到这 5 层的输出；第 5 层后用一次 21→1 卷积融合，所得固定特征既进入第 6 层，也分别加到第 6–10 层的输出。每个完整 Block 内部的注意力与 FFN 残差保留。

用户确认采用仓库本地默认参数。正式队列为 S1–S17 去掉 S6、S17 后的 **15 人、15 折 LOSO**，每折 14 人训练、1 人测试。S6 与 S17 各自单独构建数据集、单独评估。本文沿用 [此前 Linux 文档](新增16人实验_Linux运行说明.md) 的服务器目录与环境；服务器上的拉取、预处理和训练由你执行。

## 1. 固定配置与张量含义

| 项目 | 本轮设置 |
| --- | --- |
| 输入 | 15 秒窗、3 秒步长、128 Hz；Time `[B,1,21,1920]`，FFT `[B,1,21,961]` |
| FFT | 现有 `log1p(abs(rFFT(x))²)` 对数功率谱，Time / FFT 各按本折训练数据的均值与标准差归一化 |
| Stem | `(1,25)` 时间/频率卷积、BN、ELU、`(1,75)` 池化、步长 `(1,15)`、Dropout、1×1 特征投影；保留电极轴 |
| 电极轴 C / 特征轴 D | C=21 是实际 EEG 电极；D=40 是学习到的嵌入特征，二者不同 |
| Token 数 | Time N=122，FFT N=58 |
| 前 5 层 | `[B,21,N,40]` 整理成 `[B×21,N,40]`；每层参数在 21 个电极间共享，注意力沿 N 计算 |
| 跨电极卷积 | 在 `[B,C,N,D]` 上分别用 `Conv2d(21,21,1)`、`Conv2d(21,1,1)`，groups=1，含 bias；每个 `(N,D)` 位置混合电极 |
| 外部跳接 | `S=Conv21→21(X0)`；`Hi=Block_i(H_(i−1))+S`，i=1…5；`F5=squeeze(Conv21→1(H5))`；`Z5=F5`，`Zi=Block_i(Z_(i−1))+F5`，i=6…10 |
| Encoder | depth=10（本轮结构要求；本地原默认为 6）、emb=40、heads=5、FFN expansion=4 |
| Dropout | Stem=.5，Encoder 各处=.5，分类头=.5/.3 |
| 分类 | Time 4880 + FFT 2320 → 7200→256→32→3，ELU；普通无权重 CE |
| 优化 | Adam，lr=.0002，betas=(.5,.999)，eps=1e-8，weight_decay=0 |
| 训练日程 | 200 epochs、batch=72、seed=42；原始采样，不启用 S&R |
| 最佳权重 | 沿用历史每轮留出测试集最高准确率选轮，平分取第一次；属于探索性开发评估 |

两个域的 Stem、卷积、Encoder 参数独立。移除旧 11/10/8 并行链、Cross-Depth QKV、深度权重、三组分类头和辅助分支损失；也取消旧实验 `[3,3,1]` 类别权重。训练默认对 Block 启用激活重算，减少保留 21 电极带来的显存占用，保留 Dropout 随机状态，不改变 batch 或优化参数。

最新可编辑框图：[`activity_loso_time_fft_channelpreserve_depth10_skip10_seed42_结构框架图.json`](html_json/eeg_json/activity_loso_time_fft_channelpreserve_depth10_skip10_seed42_结构框架图.json)。可导入原 HTML 编辑器查看。

## 2. 在服务器拉取 GitHub

以下命令在服务器的同一个 Bash 会话中依次执行：

```bash
cd /public/home/gd_110413/SWT/work/eeg
git status --short
git branch --show-current
```

确认当前分支为 `main`，并先处理 `git status` 中的本地改动，然后拉取：

```bash
git pull --ff-only origin main
git log -1 --oneline

EEG_TRAIN_PY="$PWD/.conda-envs/eegconformer310/bin/python"
EEG_PREP_PY="$PWD/.venv-preprocessing/bin/python"
EEG_DATA_BASE="$PWD/local_artifacts/data_to_list"
EEG_NORMALIZED="$EEG_DATA_BASE/list_normalization_fixed_duration"
EEG_FORMAL="$EEG_DATA_BASE/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/window_15_stride_3"
EEG_RUN_BASE="$PWD/local_artifacts/outputs/activity_channelpreserve_loso"
EEG_JOB_ROOT="$EEG_RUN_BASE/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/seed_42/channelpreserving_depth10_skip10"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$EEG_TRAIN_PY" -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
nvidia-smi
```

使用已准备好的 CUDA 环境。若预处理环境或输入包尚未准备，按此前文档第 1–3 节准备；第一次完整预处理可直接运行：

```bash
bash scripts/prepare_expanded_activity_dataset.sh \
  --python "$EEG_PREP_PY" --exclude-subject-ids 6,17
```

该命令会处理新增 12–17 的 EDF。S17 仍只取前三段刺激条件记录（300/318/998 秒），旧 1–11 的归一化记录保持原版本。若之前 16 人实验已完成预处理，直接执行下一节，从已有归一化 Excel 构建本轮 NPY 即可。

## 3. 从已有归一化 Excel 构建正式与失败案例数据

正式数据使用新的 `_exclude_S6_S17` 目录；原 16 人数据继续保留。构建器会在 `--output-root` 的上级自动添加排除名单，所以参数中先填未加排除后缀的目录。

```bash
if [ ! -f "$EEG_FORMAL/metadata.json" ]; then
  "$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
    --input-root "$EEG_NORMALIZED" \
    --output-root "$EEG_DATA_BASE/global_activity_dataset_1to17_S17_first3/window_15_stride_3" \
    --include-subject-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
    --exclude-subject-ids 6,17 --require-complete-records \
    --window-seconds 15 --stride-seconds 3
fi

for EEG_CASE_ID in 6 17; do
  EEG_CASE_ROOT="$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3"
  if [ ! -f "$EEG_CASE_ROOT/metadata.json" ]; then
    "$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
      --input-root "$EEG_NORMALIZED" --output-root "$EEG_CASE_ROOT" \
      --include-subject-ids "$EEG_CASE_ID" --require-complete-records \
      --window-seconds 15 --stride-seconds 3
  fi
done
```

已有有效数据集会复用。若目录里只有中断产生的部分文件，构建器会拒绝覆盖；请检查后使用新输出目录，或按构建器帮助显式重建衍生数据，保留原始文件。

核验三个数据集及来源记录：

```bash
"$EEG_TRAIN_PY" - "$EEG_FORMAL" \
  "$EEG_DATA_BASE/global_activity_failurecase_S6/window_15_stride_3" \
  "$EEG_DATA_BASE/global_activity_failurecase_S17/window_15_stride_3" <<'PY'
import sys
import numpy as np
sys.path.insert(0, "EEG-Conformer")
from evaluate_activity_failurecase import load_dataset
formal, s6, s17 = [load_dataset(path) for path in sys.argv[1:]]
expected = [i for i in range(1,18) if i not in (6,17)]
assert formal.metadata["included_subject_ids"] == expected
assert sorted(formal.metadata["excluded_subject_ids"]) == [6,17]
assert s6.metadata["included_subject_ids"] == [6]
assert s17.metadata["included_subject_ids"] == [17]
assert len({d.metadata["input_root"] for d in (formal,s6,s17)}) == 1
for d in (formal,s6,s17):
    print(d.root, d.X.shape, "records=",d.metadata["n_records"],
          "classes=",np.bincount(d.y,minlength=3).tolist())
PY
```

沿用当前预处理记录时，预期正式数据 **7869 窗口、45 条记录**，三类为 `[1440,1530,4899]`；S6 与 S17 各 **526 窗口、3 条记录**，三类各为 `[96,102,328]`。这些是输入版本对应的预期，不会把窗口数量写死在训练器里；若不同，先核实数据来源与时间裁切。

## 4. 检查计划并做一次独立试跑

```bash
"$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 --dry-run
```

应看到 `planned_training_folds=15`、排除 `[6,17]`、depth=10、heads=5、200 epochs、batch=72、`class_weights=null`。启动前会核对数组中的真实编号、类别、记录来源、元数据与数据/代码哈希；`--dry-run` 不训练、不写运行目录。

试跑用第一名正式受试者，1 epoch、batch=2，输出与正式实验隔离：

```bash
"$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 --smoke-test
```

试跑目录为同一 `seed_42/` 下的 `smoke_channelpreserving_depth10_skip10_test_S1/`。它仍使用全部正式训练者的数据，只缩短训练日程；不能当作正式结果或用于 Failure case 汇总。若中断，原命令加 `--resume`；若已完成，同样可用 `--resume` 验证并跳过。

## 5. 后台启动正式 15 折训练

在一张空闲 GPU 上启动一份作业：

```bash
mkdir -p EEG-Conformer/logs
EEG_RUN_TAG="channelpreserve_depth10_skip10_seed42_$(date +%Y%m%d_%H%M%S)"
EEG_LOG="$PWD/EEG-Conformer/logs/${EEG_RUN_TAG}.log"
EEG_PID_FILE="$PWD/EEG-Conformer/logs/${EEG_RUN_TAG}.pid"

nohup "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 \
  > "$EEG_LOG" 2>&1 < /dev/null &
EEG_PID=$!
printf '%s\n' "$EEG_PID" > "$EEG_PID_FILE"
printf 'PID=%s\nLOG=%s\nJOB=%s\n' "$EEG_PID" "$EEG_LOG" "$EEG_JOB_ROOT"
tail -f "$EEG_LOG"
```

`Ctrl+C` 只退出 `tail`，后台训练继续。可用 `ps -p "$EEG_PID" -o pid,etime,args`、`nvidia-smi` 和 `cat "$EEG_JOB_ROOT/status.json"` 查看状态。若你重新登录，先按第 2 节恢复变量，再从 `EEG-Conformer/logs/` 找对应日志和 PID。

输出目录已经存在时，普通启动会拒绝覆盖；不要同时启动两份写同一目录的作业。GPU 显存是否足够仍需服务器实测，试跑 batch=2 只能检查流程；如正式 batch=72 显存不足，保留日志和断点，先确认 GPU 及占用，不要把改小 batch 的结果混入默认参数实验。

## 6. 中断后的续训

确认旧进程已经停止，再执行相同命令加 `--resume`。后台运行方式如下：

```bash
EEG_RESUME_LOG="$PWD/EEG-Conformer/logs/channelpreserve_resume_$(date +%Y%m%d_%H%M%S).log"
nohup "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 --resume \
  > "$EEG_RESUME_LOG" 2>&1 < /dev/null &
EEG_PID=$!
printf '%s\n' "$EEG_PID" > "${EEG_RESUME_LOG%.log}.pid"
tail -f "$EEG_RESUME_LOG"
```

从第一轮起保存模型、Adam 状态、随机状态及历史。完整折会先核验权重、历史和预测再跳过；未完成折从上一完整 epoch 恢复。数据、代码或参数变化会拒绝续训，应使用原训练 commit，或另设 `--output-base` 新开实验。失败会记录 `status.json` 并停止，不把缺失折包装成完整结果。

## 7. 全部完成后分别评估 S6、S17

先确认：

```bash
cat "$EEG_JOB_ROOT/completed.json"
cat "$EEG_JOB_ROOT/model/summary.json"
```

应为 `smoke_test=false`、`training_folds=15`。新的 Failure case 入口与此前五模型评估入口分开：

```bash
for EEG_CASE_ID in 6 17; do
  "$EEG_TRAIN_PY" -u EEG-Conformer/evaluate_channelpreserve_failurecase.py \
    --dataset-root "$EEG_FORMAL" \
    --failurecase-root "$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3" \
    --run-root "$EEG_JOB_ROOT" \
    --output-dir "$PWD/local_artifacts/outputs/activity_channelpreserve_failurecase_S${EEG_CASE_ID}_seed42" \
    --case-subject-id "$EEG_CASE_ID" --seed 42 \
    --device cuda:0 --batch-size 72 --cpu-threads 1 || break
done
```

每折使用该折 14 名正式训练者的数据恢复 Time / FFT 归一化；先重播原留出测试预测并核对一致，再预测失败案例。所有概率固定后才计算失败案例指标。15 折概率只沿模型维度逐窗口等权平均，526 个窗口仍是 526 个窗口；两个失败案例各有独立的综合混淆矩阵。

入口拒绝 smoke、部分折、旧架构、旧类别权重、数据/代码哈希不匹配或无法重播原测试预测的作业。已有评估目录不覆盖，重跑时选择新输出目录。若重播失败，优先核对训练 commit、环境、输入内容和 batch，保留日志定位。

## 8. 输出位置与结果解释

正式作业在 `$EEG_JOB_ROOT/` 保存 `run_plan.json`、`environment.json`、`status.json` 与全部完成标记。`model/fold_subject_<id>/` 包含 `best_model.pt`、`metrics.json`、`test_predictions.npz`、`epoch_history.csv/json`、`train.log`；未完成折还保留 `last_checkpoint.pt`。正式汇总为 `model/summary.json`、`summary.csv`，含逐折准确率和总体混淆矩阵。

每个 Failure case 目录保存 `failurecase_summary.json`、`ensemble_predictions.npz`、`normalizers.json`、`evaluation_manifest.json` 和 15 份单折概率/指标。综合矩阵行是真实 e_1/e_2/e_3，列是预测类别；该集成指标与正式 LOSO 的逐折均值分别报告。

本次本地通过 332 项模型与训练相关测试（含 21 项新增测试）及 25 项数据集构建测试，覆盖真实输入形状的前向、反向、合成数据训练、最佳权重恢复、原测试预测重播及中断续训；11 段 Bash 命令通过语法检查。FastAPI 接口测试因本机缺少该依赖未运行。真实 15 折 CUDA 训练和 S6/S17 推理由你在服务器执行，本次交付不包含新的真实准确率。
