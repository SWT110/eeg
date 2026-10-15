# 新结构：保留电极通道、10 层双段跳接

本次入口为 `EEG-Conformer/train_activity_channelpreserve.py`。Time / FFT 各一条独立的 10 层链：前 5 层保留 21 个电极，入口的 21→21 卷积特征分别加到这 5 层的输出；第 5 层后用一次 21→1 卷积融合，所得固定特征既进入第 6 层，也分别加到第 6–10 层的输出。每个完整 Block 内部的注意力与 FFN 残差保留。

用户确认采用仓库本地默认参数。正式队列为 S1–S17 去掉 S6、S17 后的 **15 人、15 折 LOSO**，每折 14 人训练、1 人测试。S6 与 S17 各自单独构建数据集、单独评估。本文沿用 [此前 Linux 文档](新增16人实验_Linux运行说明.md) 的服务器目录与环境；服务器上的拉取、预处理和训练由你执行。

先执行第 2–4 节，再选择第 5 节单 GPU 或第 6 节多 GPU。所有重计算命令均绑定 CPU：默认单个作业最多使用 2 个逻辑 CPU，PyTorch 与数学库各设为 1 线程；双 GPU 两个作业合计最多使用 4 个逻辑 CPU。多 GPU 按独立 LOSO 折并行，每折仍为 batch=72、200 epochs。

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

EEG_GPU0=0
EEG_CPUS_PER_JOB=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_WAIT_POLICY=PASSIVE

command -v taskset
EEG_CPUSET0="$("$EEG_TRAIN_PY" - "$EEG_CPUS_PER_JOB" <<'PY'
import os
import sys
allowed = sorted(os.sched_getaffinity(0))
print(",".join(str(cpu) for cpu in allowed[:int(sys.argv[1])]))
PY
)"
printf 'Single-job CPUs: %s\n' "$EEG_CPUSET0"
env CUDA_VISIBLE_DEVICES="$EEG_GPU0" taskset -c "$EEG_CPUSET0" \
  "$EEG_TRAIN_PY" -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
nvidia-smi
```

`EEG_GPU0=0` 是示例 GPU 编号，按 `nvidia-smi` 改成空闲卡。`CUDA_VISIBLE_DEVICES` 选择服务器 GPU；后文统一用 `--device cuda:0`，表示该进程内可见的第一张卡。多 GPU 的每个进程也只暴露它负责的一张卡。

线程数限制控制计算库的并行度；`taskset` 将整个进程及其后续线程、子进程限制在指定 CPU 上。上面的 CPU 编号自动从当前会话允许的集合中选取，兼容编号不从 0 开始的分配；单作业最多取 `EEG_CPUS_PER_JOB=2` 个，若只分到 1 个就只使用 1 个。Linux 的 `top` 按单逻辑 CPU 为 100% 计数，2 个逻辑 CPU 对应约 200% 的计算上限，并非进程只能创建两个线程。

`--cpu-threads 1` 还会设置 PyTorch 计算线程数，数据加载器默认 `num_workers=0`。不要在启动后把 OMP/MKL 等变量重新调大。若 `command -v taskset` 没有输出，先在服务器准备 `util-linux` 中的 `taskset` 再继续。

对已经运行的本人作业，先把 `EEG_PID` 设置为日志 `.pid` 文件中的训练 PID，再核对进程、收紧所有现有线程的 CPU 绑定，无需重启：

```bash
ps -p "$EEG_PID" -o pid,user,etime,args
taskset -apc "$EEG_CPUSET0" "$EEG_PID"
taskset -apc "$EEG_PID"
```

已经运行的进程不会读取后来修改的 OMP/MKL 环境变量。若还需要更改线程参数，按第 7 节中断续训，使新启动命令生效。

使用已准备好的 CUDA 环境。若预处理环境或输入包尚未准备，按此前文档第 1–3 节准备；第一次完整预处理可直接运行：

```bash
taskset -c "$EEG_CPUSET0" bash scripts/prepare_expanded_activity_dataset.sh \
  --python "$EEG_PREP_PY" --exclude-subject-ids 6,17
```

该命令会处理新增 12–17 的 EDF。S17 仍只取前三段刺激条件记录（300/318/998 秒），旧 1–11 的归一化记录保持原版本。若之前 16 人实验已完成预处理，直接执行下一节，从已有归一化 Excel 构建本轮 NPY 即可。

## 3. 从已有归一化 Excel 构建正式与失败案例数据

正式数据使用新的 `_exclude_S6_S17` 目录；原 16 人数据继续保留。构建器会在 `--output-root` 的上级自动添加排除名单，所以参数中先填未加排除后缀的目录。

```bash
if [ ! -f "$EEG_FORMAL/metadata.json" ]; then
  taskset -c "$EEG_CPUSET0" "$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
    --input-root "$EEG_NORMALIZED" \
    --output-root "$EEG_DATA_BASE/global_activity_dataset_1to17_S17_first3/window_15_stride_3" \
    --include-subject-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
    --exclude-subject-ids 6,17 --require-complete-records \
    --window-seconds 15 --stride-seconds 3
fi

for EEG_CASE_ID in 6 17; do
  EEG_CASE_ROOT="$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3"
  if [ ! -f "$EEG_CASE_ROOT/metadata.json" ]; then
    taskset -c "$EEG_CPUSET0" "$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
      --input-root "$EEG_NORMALIZED" --output-root "$EEG_CASE_ROOT" \
      --include-subject-ids "$EEG_CASE_ID" --require-complete-records \
      --window-seconds 15 --stride-seconds 3
  fi
done
```

已有有效数据集会复用。若目录里只有中断产生的部分文件，构建器会拒绝覆盖；请检查后使用新输出目录，或按构建器帮助显式重建衍生数据，保留原始文件。

核验三个数据集及来源记录：

```bash
taskset -c "$EEG_CPUSET0" "$EEG_TRAIN_PY" - "$EEG_FORMAL" \
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
env CUDA_VISIBLE_DEVICES="$EEG_GPU0" taskset -c "$EEG_CPUSET0" \
  "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 --dry-run
```

应看到 `planned_training_folds=15`、排除 `[6,17]`、depth=10、heads=5、200 epochs、batch=72、`class_weights=null`。启动前会核对数组中的真实编号、类别、记录来源、元数据与数据/代码哈希；`--dry-run` 不训练、不写运行目录。

试跑用第一名正式受试者，1 epoch、batch=2，输出与正式实验隔离：

```bash
env CUDA_VISIBLE_DEVICES="$EEG_GPU0" taskset -c "$EEG_CPUSET0" \
  "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 --smoke-test
```

试跑目录为同一 `seed_42/` 下的 `smoke_channelpreserving_depth10_skip10_test_S1/`。它仍使用全部正式训练者的数据，只缩短训练日程；不能当作正式结果或用于 Failure case 汇总。若中断，原命令加 `--resume`；若已完成，同样可用 `--resume` 验证并跳过。

## 5. 单 GPU：后台顺序训练 15 折

在一张空闲 GPU 上启动一份作业：

```bash
mkdir -p EEG-Conformer/logs
EEG_RUN_TAG="channelpreserve_depth10_skip10_seed42_$(date +%Y%m%d_%H%M%S)"
EEG_LOG="$PWD/EEG-Conformer/logs/${EEG_RUN_TAG}.log"
EEG_PID_FILE="$PWD/EEG-Conformer/logs/${EEG_RUN_TAG}.pid"

nohup env CUDA_VISIBLE_DEVICES="$EEG_GPU0" taskset -c "$EEG_CPUSET0" \
  "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 \
  > "$EEG_LOG" 2>&1 < /dev/null &
EEG_PID=$!
printf '%s\n' "$EEG_PID" > "$EEG_PID_FILE"
printf 'PID=%s\nLOG=%s\nJOB=%s\n' "$EEG_PID" "$EEG_LOG" "$EEG_JOB_ROOT"
tail -f "$EEG_LOG"
```

`Ctrl+C` 只退出 `tail`，后台训练继续。若你重新登录，先按第 2 节恢复变量，再从 `EEG-Conformer/logs/` 找对应日志和 PID。下面几条命令分别查看进程、CPU 绑定、线程、GPU 和训练状态：

```bash
ps -p "$EEG_PID" -o pid,etime,pcpu,nlwp,args
taskset -apc "$EEG_PID"
top -H -p "$EEG_PID"
nvidia-smi
cat "$EEG_JOB_ROOT/status.json"
```

输出目录已经存在时，普通启动会拒绝覆盖；不要同时启动两份写同一目录的作业。GPU 显存是否足够仍需服务器实测，试跑 batch=2 只能检查流程；如正式 batch=72 显存不足，保留日志和断点，先确认 GPU 及占用，不要把改小 batch 的结果混入默认参数实验。

## 6. 多 GPU：并行训练独立 LOSO 折，再合并

### 6.1 分组启动与续训

下面用两张卡完成 15 折：第一张训练 S1、S2、S3、S4、S5、S7、S8、S9 的 8 个测试折，第二张训练 S10–S16 的 7 个测试折。每折仍使用完整正式队列中的其余 14 人训练，`--test-subject-ids` 只分配测试折，不会把训练队列分成两半。

先按 `nvidia-smi` 选择两张空闲卡。每进程独立加载并归一化数据，双进程也会增加主存占用。以下命令会验证分组恰好覆盖 15 人、无重复，且当前允许的 CPU 足够；双卡默认绑定互不重叠的 2+2 个逻辑 CPU，不足时不启动。

```bash
EEG_GPUS=(0 1)
EEG_FOLD_GROUPS=("1,2,3,4,5,7,8,9" "10,11,12,13,14,15,16")
EEG_CPUS_PER_JOB=2
```

如果选用四张卡，将上面的配置替换为以下配置，默认需要 8 个允许使用的逻辑 CPU；随后执行同一个启动块与合并步骤：

```bash
EEG_GPUS=(0 1 2 3)
EEG_FOLD_GROUPS=("1,2,3,4" "5,7,8,9" "10,11,12,13" "14,15,16")
EEG_CPUS_PER_JOB=2
```

选择一套配置后执行启动块。若 CPU 分配较少，可以在启动前将 `EEG_CPUS_PER_JOB=1`：

```bash
EEG_MULTI_PIDS=()

if EEG_CPUSETS_TEXT="$("$EEG_TRAIN_PY" - "$EEG_CPUS_PER_JOB" \
  "${EEG_GPUS[*]}" "${EEG_FOLD_GROUPS[@]}" <<'PY'
import os
import sys
per_job = int(sys.argv[1])
gpus = sys.argv[2].split()
groups = [[int(s) for s in arg.split(",")] for arg in sys.argv[3:]]
expected = [s for s in range(1,18) if s not in (6,17)]
if (per_job < 1 or len(gpus) < 2 or len(gpus) != len(set(gpus))
        or len(gpus) != len(groups) or any(not g for g in groups)
        or sorted(s for group in groups for s in group) != expected):
    raise SystemExit("GPU/分组无效：每卡一组，15 名正式受试者须各出现一次")
allowed = sorted(os.sched_getaffinity(0))
needed = len(gpus) * per_job
if len(allowed) < needed:
    raise SystemExit(f"需要 {needed} 个逻辑 CPU，当前仅允许 {len(allowed)} 个；可将每作业限制改为 1 后重试")
for index in range(len(gpus)):
    print(",".join(map(str, allowed[index*per_job:(index+1)*per_job])))
PY
)"; then
  mapfile -t EEG_CPUSETS <<< "$EEG_CPUSETS_TEXT"
  EEG_GPU0="${EEG_GPUS[0]}"
  EEG_CPUSET0="${EEG_CPUSETS[0]}"
  mkdir -p EEG-Conformer/logs
  EEG_MULTI_TAG="channelpreserve_multi_seed42_$(date +%Y%m%d_%H%M%S)"
  for EEG_IDX in "${!EEG_GPUS[@]}"; do
    EEG_SHARD_LOG="$PWD/EEG-Conformer/logs/${EEG_MULTI_TAG}_part${EEG_IDX}.log"
    nohup env CUDA_VISIBLE_DEVICES="${EEG_GPUS[$EEG_IDX]}" \
      taskset -c "${EEG_CPUSETS[$EEG_IDX]}" \
      "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
      --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
      --test-subject-ids "${EEG_FOLD_GROUPS[$EEG_IDX]}" \
      --device cuda:0 --cpu-threads 1 --resume \
      > "$EEG_SHARD_LOG" 2>&1 < /dev/null &
    EEG_SHARD_PID=$!
    EEG_MULTI_PIDS+=("$EEG_SHARD_PID")
    printf '%s\n' "$EEG_SHARD_PID" > "${EEG_SHARD_LOG%.log}.pid"
    printf 'part=%s GPU=%s CPUs=%s PID=%s LOG=%s\n' \
      "$EEG_IDX" "${EEG_GPUS[$EEG_IDX]}" "${EEG_CPUSETS[$EEG_IDX]}" \
      "$EEG_SHARD_PID" "$EEG_SHARD_LOG"
  done
else
  printf 'CPU/GPU 分组检查失败，未启动训练。\n' >&2
fi
```

命令中的 `--resume` 同时支持首次创建和中断续训，已完成折会核验后跳过。重复执行整个启动块前先确认所有旧分折进程都已停止；如果只需恢复失败的一组，可以仅重新执行该组的原启动命令，保留 `--resume`，无需等待其他组完成。各组均使用 `--device cuda:0`，由各自的 `CUDA_VISIBLE_DEVICES` 将其映射到不同卡。

查看各组日志和 CPU 绑定：

```bash
for EEG_PID in "${EEG_MULTI_PIDS[@]}"; do
  ps -p "$EEG_PID" -o pid,etime,pcpu,nlwp,args
  taskset -apc "$EEG_PID"
done
nvidia-smi
tail -f EEG-Conformer/logs/"${EEG_MULTI_TAG}"_part*.log
```

`Ctrl+C` 退出日志查看。双卡示例的两个目录位于同一 `seed_42/` 下，分别为：

```text
channelpreserving_depth10_skip10_test_S1_S2_S3_S4_S5_S7_S8_S9/
channelpreserving_depth10_skip10_test_S10_S11_S12_S13_S14_S15_S16/
```

每组有自己的 `run_plan.json`、日志、断点和完成标记，互不写入对方目录。该组的 `completed.json` 只表示它负责的测试折完成，双卡为 8/7 折，四卡为 4/4/4/3 折。不要同时再启动第 5 节覆盖全部 15 折的作业。

### 6.2 全部分折完成后，合并为完整 15 折作业

Failure case 评估要求完整 15 折。下面的命令核对每组的参数、数据与代码哈希、完成标记、权重、历史和预测，确认 15 折恰好各一份后，复制结果到标准的 `$EEG_JOB_ROOT`，并保留分组清单。源作业保留，目标存在时拒绝覆盖。

在同一会话中保留第 6.1 节实际使用的 `EEG_FOLD_GROUPS`；重新登录后按原分组重新定义数组。合并前所有分折作业须结束，标准 `$EEG_JOB_ROOT` 须尚未创建。若此前已有单 GPU 完整作业，请使用独立的 `EEG_RUN_BASE` 组织多 GPU 实验。

```bash
if taskset -c "$EEG_CPUSET0" "$EEG_TRAIN_PY" - \
  "$EEG_FORMAL" "$EEG_RUN_BASE" "${EEG_FOLD_GROUPS[@]}" <<'PY'
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "EEG-Conformer")
from run_activity_five_models import sha256_file
from train_activity_channelpreserve import build_plan, parse_args, validate_completed_fold
from train_activity_loso import atomic_json_dump

common = ["--dataset-root", sys.argv[1], "--output-base", sys.argv[2],
          "--device", "cuda:0", "--cpu-threads", "1"]
full = build_plan(parse_args(common))
target = Path(full["output_dir"])
if target.exists():
    raise FileExistsError(f"目标已存在，拒绝覆盖：{target}")
if len(sys.argv[3:]) < 2:
    raise ValueError("合并至少需要两个分折作业；请恢复 EEG_FOLD_GROUPS 数组")

sources, folds, seen = [], [], set()
for group in sys.argv[3:]:
    expected = build_plan(parse_args(common + ["--test-subject-ids", group]))
    root = Path(expected["output_dir"])
    actual = json.loads((root / "run_plan.json").read_text())
    if actual != expected:
        raise ValueError(f"分折作业参数、数据或代码与当前配置不一致：{root}")
    done = json.loads((root / "completed.json").read_text())
    ids = expected["test_subject_ids"]
    marker = {"status": "completed", "architecture": "channelpreserving",
              "smoke_test": False, "training_folds": len(ids)}
    if done != marker or seen.intersection(ids):
        raise ValueError(f"分折未完成或有重复受试者：{root}")
    model_root = root / "model"
    found = {p.name for p in model_root.glob("fold_subject_*") if p.is_dir()}
    if found != {f"fold_subject_{sid}" for sid in ids}:
        raise ValueError(f"分折目录集合不一致：{root}")
    for sid in ids:
        fold = model_root / f"fold_subject_{sid}"
        validate_completed_fold(fold, full, sid)
        folds.append(fold)
    seen.update(ids)
    sources.append({"root": str(root), "test_subject_ids": ids,
                    "run_plan_sha256": sha256_file(root / "run_plan.json")})
if sorted(seen) != full["test_subject_ids"]:
    raise ValueError("合并结果没有恰好覆盖全部 15 名正式受试者")

target.parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=".channelpreserve_merge_", dir=target.parent))
try:
    for fold in folds:
        shutil.copytree(fold, staging / "model" / fold.name)
    for index, source in enumerate(sources):
        saved = staging / "shard_manifests" / f"part_{index}"
        saved.mkdir(parents=True)
        for name in ("run_plan.json", "environment.json", "completed.json"):
            shutil.copy2(Path(source["root"]) / name, saved / name)
    atomic_json_dump(full, staging / "run_plan.json")
    atomic_json_dump({"sources": sources, "fold_subject_ids": sorted(seen),
                      "operation": "copy completed folds; no training"}, staging / "merge_manifest.json")
    if target.exists():
        raise FileExistsError(f"合并期间目标被创建，拒绝覆盖：{target}")
    staging.rename(target)
except Exception:
    shutil.rmtree(staging)
    raise
print(f"已合并 {len(folds)} 折到 {target}")
PY
then
  env CUDA_VISIBLE_DEVICES="$EEG_GPU0" taskset -c "$EEG_CPUSET0" \
    "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
    --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
    --device cuda:0 --cpu-threads 1 --resume
fi
```

最后的 `--resume` 会核验并跳过已复制的全部 15 折，生成统一 `model/summary.json`、`summary.csv` 和 `completed.json`，随后可直接使用第 8 节评估命令。若复制成功但汇总被中断，按第 7 节对完整作业续训即可，不必再次合并。

## 7. 中断后的续训

确认旧进程已经停止，再执行相同命令加 `--resume`。后台运行方式如下：

```bash
EEG_RESUME_LOG="$PWD/EEG-Conformer/logs/channelpreserve_resume_$(date +%Y%m%d_%H%M%S).log"
nohup env CUDA_VISIBLE_DEVICES="$EEG_GPU0" taskset -c "$EEG_CPUSET0" \
  "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_channelpreserve.py \
  --dataset-root "$EEG_FORMAL" --output-base "$EEG_RUN_BASE" \
  --device cuda:0 --cpu-threads 1 --resume \
  > "$EEG_RESUME_LOG" 2>&1 < /dev/null &
EEG_PID=$!
printf '%s\n' "$EEG_PID" > "${EEG_RESUME_LOG%.log}.pid"
tail -f "$EEG_RESUME_LOG"
```

从第一轮起保存模型、Adam 状态、随机状态及历史。完整折会先核验权重、历史和预测再跳过；未完成折从上一完整 epoch 恢复。数据、代码或参数变化会拒绝续训，应使用原训练 commit，或另设 `--output-base` 新开实验。失败会记录 `status.json` 并停止，不把缺失折包装成完整结果。

多 GPU 作业用第 6.1 节原分组命令续训，其中已包含 `--resume`。若要换一张空闲 GPU，调整 `CUDA_VISIBLE_DEVICES` 对应的卡号，保留 `--device cuda:0`、`--cpu-threads 1`、测试受试者分组和输出目录；单 GPU 完整作业与多 GPU 分折作业各自续训。

## 8. 全部完成后分别评估 S6、S17

先确认：

```bash
cat "$EEG_JOB_ROOT/completed.json"
cat "$EEG_JOB_ROOT/model/summary.json"
```

应为 `smoke_test=false`、`training_folds=15`。新的 Failure case 入口与此前五模型评估入口分开：

```bash
for EEG_CASE_ID in 6 17; do
  env CUDA_VISIBLE_DEVICES="$EEG_GPU0" taskset -c "$EEG_CPUSET0" \
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

## 9. 输出位置与结果解释

正式作业在 `$EEG_JOB_ROOT/` 保存 `run_plan.json`、`environment.json`、`status.json` 与全部完成标记。`model/fold_subject_<id>/` 包含 `best_model.pt`、`metrics.json`、`test_predictions.npz`、`epoch_history.csv/json`、`train.log`；未完成折还保留 `last_checkpoint.pt`。正式汇总为 `model/summary.json`、`summary.csv`，含逐折准确率和总体混淆矩阵。

每个 Failure case 目录保存 `failurecase_summary.json`、`ensemble_predictions.npz`、`normalizers.json`、`evaluation_manifest.json` 和 15 份单折概率/指标。综合矩阵行是真实 e_1/e_2/e_3，列是预测类别；该集成指标与正式 LOSO 的逐折均值分别报告。

训练脚本此前本地通过 332 项模型与训练相关测试（含 21 项新增测试）及 25 项数据集构建测试，覆盖真实输入形状的前向、反向、合成数据训练、最佳权重恢复、原测试预测重播及中断续训。FastAPI 接口测试因本机缺少该依赖未运行。本文新增命令在本地核验 Bash 语法、CPU 分配与分折合并逻辑；真实多 GPU、Linux CPU 绑定、15 折 CUDA 训练和 S6/S17 推理由你在服务器执行，本次交付不包含新的真实准确率。
