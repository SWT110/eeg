# 新结构 D 版：可配置层数与批量类别权重实验

本轮只训练新结构 D 版：Time、FFT 默认各有 **2／5／8 个 Block** 的三个独立分支，分别在第 **1／2／4** 个 Block 后完成 21→1 通道融合，保留前后两段逐层固定跳接；启用 **Cross-Depth QKV＋Branch Loss**。

正式队列仍是 **S1–S17 排除 S6、S17 后的 15 人**。你自行填写 e₁／e₂／e₃ 的类别损失权重，每组完整训练 15 折，再开始下一组。两张 GPU 可以各跑一批，结果放在同一个 outputs 母目录中。不自动选冠军，跑完汇总后再选。

除类别损失权重及已确认的 D 版结构外，沿用仓库本地默认：seed=42、200 epochs、batch=72、Adam lr=.0002、betas=(.5,.999)、emb=40、heads=5、stem/encoder dropout=.5、分类头 dropout=.5/.3、Cross-Depth dropout=.1、gamma 初值=.1、Branch Loss 辅助系数=.2。原始采样，无 S&R、无三倍重采样、无分层分类。

以下命令按顺序在服务器**同一个 Bash 终端**执行。第 6 节的单卡与双卡方式选一种。

## 1. 拉取最新代码

```bash
cd /public/home/gd_110413/SWT/work/eeg
git status --short
git pull --ff-only origin main
git log -1 --oneline
```

有服务器本地改动时先处理再拉取。此次入口是 `EEG-Conformer/train_activity_multidepth_weight_sweep.py`，旧 depth10 和旧四组合入口保留。

## 2. 准备环境并限制 CPU

```bash
EEG_TRAIN_PY="$PWD/.conda-envs/eegconformer310/bin/python"

# 想用 8 个逻辑 CPU，只需在首次启动前把这里的 6 改为 8。
export EEG_CPU_THREADS=6
export OMP_NUM_THREADS="$EEG_CPU_THREADS"
export MKL_NUM_THREADS="$EEG_CPU_THREADS"
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE

# 两个作业各选一组当前允许使用的逻辑 CPU。
EEG_CPU_A="$("$EEG_TRAIN_PY" -c 'import os; c=sorted(os.sched_getaffinity(0)); k=int(os.environ["EEG_CPU_THREADS"]); print(",".join(map(str,c[:k])))')"
EEG_CPU_B="$("$EEG_TRAIN_PY" -c 'import os; c=sorted(os.sched_getaffinity(0)); k=int(os.environ["EEG_CPU_THREADS"]); print(",".join(map(str,c[k:2*k] or c[:k])))')"
printf '线程数=%s; CPU A=%s; CPU B=%s\n' "$EEG_CPU_THREADS" "$EEG_CPU_A" "$EEG_CPU_B"
command -v taskset
nvidia-smi
```

`taskset` 限制 CPU 范围，后面的 `--cpu-threads` 限制 PyTorch 线程数，两者都由上面同一个变量控制。允许的 CPU 不足时两组会共享。CPU 数与 GPU 是各自独立的设置，CPU=6 不代表 GPU=6。

## 3. 设置数据与共同结果目录

如果上一轮 15 人五模型实验已经跑过，直接复用数据，不必重新预处理。

```bash
EEG_DATA_BASE="$PWD/local_artifacts/data_to_list"
EEG_NEW_DATA="$EEG_DATA_BASE/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/window_15_stride_3"
EEG_RUN_BASE="$PWD/local_artifacts/outputs/activity_channelpreserve_D_2_5_8_weights_seed42"
```

正式数据预期为 15 人、7869 个窗口、21 通道、每窗 1920 点，三类窗口数 `[1440,1530,4899]`。脚本会核对真实数组、来源记录、排除名单和数据哈希，S6 或 S17 留在正式数组中会报错。尚未准备数据时，参照[上一轮文档第 3、4 节](新结构_depth10双段跳接_Linux运行说明.md)。

共同母目录下按权重分别保存：

```text
activity_channelpreserve_D_2_5_8_weights_seed42/
├── sweep_manifest.json
├── weights_3_2_1/
│   ├── run_plan.json
│   ├── status.json
│   ├── completed.json                 # 本组所有折完成后才生成
│   └── model/
│       ├── fold_subject_1/
│       ├── fold_subject_2/
│       └── ...
├── weights_3_3_1/
├── ...
├── weight_summary.md
├── weight_summary.csv
├── weight_summary.json
└── subject_results.csv
```

## 4. 设置层数与类别权重

每组内部按 **e₁,e₂,e₃** 的顺序用逗号分隔，多组之间用 `|` 分隔。整段必须加引号。支持正有限小数，e₃ 不必固定为 1；同一批不要重复填写相同组合。

下面只是示例，可全部换成你想比较的权重：

```bash
# 分支总层数，以及每个分支在第几个 Block 后融合通道。
EEG_DEPTHS='2,5,8'
EEG_SPLITS='1,2,4'

# 单 GPU 使用的完整列表：依次完成 3/2/1、3/3/1、4/3/1、3/4/1。
EEG_WEIGHTS_ALL='3,2,1|3,3,1|4,3,1|3,4,1'

# 双 GPU 使用的两个列表：两张卡分配的组合互不重复。
EEG_WEIGHTS_GPU0='3,2,1|3,3,1'
EEG_WEIGHTS_GPU1='4,3,1|3,4,1'
```

当前分段为 **1+1、2+3、4+4**；5 层默认前 2 后 3。以后更改结构时修改 `EEG_DEPTHS` 和 `EEG_SPLITS`，并把第 3 节的 `EEG_RUN_BASE` 改为新的目录。

命令支持 `--transformer-depths 2,5,8 --split-after-blocks 1,2,4`；省略 `--split-after-blocks` 时，自动按总层数的一半向下取整。若想让 5 层前 3 后 2，可设 `1,3,4`。原结构可用 `--transformer-depths 4,10,16 --split-after-blocks 2,5,8` 指定。每个总层数至少为 2，每个分段位置必须大于 0 且小于对应总层数，仍保留三个分支。

每组 15 折，示例四组共 60 折。只搜索类别损失权重；所选三个深度分支的融合权重依旧由模型学习，两者不会混在一起。

## 5. 先检查训练计划

```bash
taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_multidepth_weight_sweep.py \
  --dataset-root "$EEG_NEW_DATA" --output-root "$EEG_RUN_BASE" \
  --transformer-depths "$EEG_DEPTHS" --split-after-blocks "$EEG_SPLITS" \
  --class-weight-sets "$EEG_WEIGHTS_ALL" \
  --device cuda:0 --cpu-threads "$EEG_CPU_THREADS" --dry-run
```

示例预期 `candidate_count=4`、`planned_training_folds=60`。每组计划内应有 `depths=[2,5,8]`、`split_after_blocks=[1,2,4]`、`transformer_branch_qkv=cross_depth`、`transformer_branch_fusion=loss_softmax`、`branch_loss_aux_weight=0.2`、`class_weights` 为你填写的三值。

`--dry-run` 不训练、不创建结果目录。如需先检查一组能正常前向、反向和保存，执行：

```bash
taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_multidepth_weight_sweep.py \
  --dataset-root "$EEG_NEW_DATA" --output-root "$EEG_RUN_BASE" \
  --transformer-depths "$EEG_DEPTHS" --split-after-blocks "$EEG_SPLITS" \
  --class-weight-sets '3,2,1' \
  --device cuda:0 --cpu-threads "$EEG_CPU_THREADS" --smoke-test --resume
```

冒烟测试只跑第一折、1 epoch、batch=2，结果在 `smoke/` 子目录中，不进入正式权重汇总。

## 6. 开始批量训练

### 单 GPU：所有权重依次训练

```bash
mkdir -p EEG-Conformer/logs
nohup taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_multidepth_weight_sweep.py \
  --dataset-root "$EEG_NEW_DATA" --output-root "$EEG_RUN_BASE" \
  --transformer-depths "$EEG_DEPTHS" --split-after-blocks "$EEG_SPLITS" \
  --class-weight-sets "$EEG_WEIGHTS_ALL" \
  --device cuda:0 --cpu-threads "$EEG_CPU_THREADS" --resume \
  > EEG-Conformer/logs/multidepth_D_2_5_8_weights_all.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/multidepth_D_2_5_8_weights_all.pid
```

### 双 GPU：每张卡依次完成自己的一批权重

在同一个终端依次执行这两条后台命令即可。根据 `nvidia-smi` 选择实际空闲的 GPU 编号。

```bash
mkdir -p EEG-Conformer/logs

# GPU 0：依次训练 EEG_WEIGHTS_GPU0 中的组合。
nohup taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_multidepth_weight_sweep.py \
  --dataset-root "$EEG_NEW_DATA" --output-root "$EEG_RUN_BASE" \
  --transformer-depths "$EEG_DEPTHS" --split-after-blocks "$EEG_SPLITS" \
  --class-weight-sets "$EEG_WEIGHTS_GPU0" \
  --device cuda:0 --cpu-threads "$EEG_CPU_THREADS" --resume \
  > EEG-Conformer/logs/multidepth_D_2_5_8_weights_gpu0.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/multidepth_D_2_5_8_weights_gpu0.pid

# GPU 1：依次训练 EEG_WEIGHTS_GPU1 中的组合。
nohup taskset -c "$EEG_CPU_B" "$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_multidepth_weight_sweep.py \
  --dataset-root "$EEG_NEW_DATA" --output-root "$EEG_RUN_BASE" \
  --transformer-depths "$EEG_DEPTHS" --split-after-blocks "$EEG_SPLITS" \
  --class-weight-sets "$EEG_WEIGHTS_GPU1" \
  --device cuda:1 --cpu-threads "$EEG_CPU_THREADS" --resume \
  > EEG-Conformer/logs/multidepth_D_2_5_8_weights_gpu1.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/multidepth_D_2_5_8_weights_gpu1.pid
```

两张卡使用**相同的 `EEG_RUN_BASE`**。每张卡一次只训练一组，完整完成该组 15 折后进入自己的下一组。这里是两个独立训练进程，不是两卡联合训练同一个模型。脚本会阻止两个进程同时写同一组权重，也会拒绝把不同数据、代码、结构或训练超参数混入同一母目录。

查看对应日志：

```bash
tail -n 50 EEG-Conformer/logs/multidepth_D_2_5_8_weights_gpu0.log
tail -n 50 EEG-Conformer/logs/multidepth_D_2_5_8_weights_gpu1.log
```

单卡方式查看 `multidepth_D_2_5_8_weights_all.log`。日志中 `[CANDIDATE 1/2]` 表示本进程当前训练第几组，`[RUN] weights_3_2_1 S1` 表示该组正在训练的受试者折；`branch_weights` 表示模型学习的三个深度融合权重。

## 7. 中断续训与追加权重

更换层数或分段位置属于新实验，必须另用一个 outputs 母目录；不能通过 `--resume` 把旧结构继续训练成新结构。旧代码生成的断点仍使用原代码版本续训。

第 6 节命令已经包含 `--resume`，首次启动也可使用。中断后确认原进程已停止，重新执行对应命令即可：已完成折核验后跳过，未完成折恢复上一完整 epoch 的模型、Adam 和随机状态。

追加新权重时，只改对应列表，再执行第 6 节对应的启动命令。例如 GPU 0 当前任务结束后：

```bash
EEG_WEIGHTS_GPU0='5,3,1|3,5,1|4,4,1'
```

这三组会继续写入原来的 `EEG_RUN_BASE`，旧结果保留。也可以在列表中包含已完成组合，带 `--resume` 会核验并跳过。续训同一组时保持原来的数据、代码、seed、GPU 编号和 CPU 线程数；CPU=6/8 在首次启动该组前确定。

## 8. 汇总实验并选择一组权重

每完成一折会更新母目录汇总；两个 GPU 的更新会加锁，避免互相覆盖。也可以单独重新汇总：

```bash
"$EEG_TRAIN_PY" -u EEG-Conformer/train_activity_multidepth_weight_sweep.py \
  --output-root "$EEG_RUN_BASE" --summarize-only
cat "$EEG_RUN_BASE/weight_summary.md"
```

主要看这些文件：

- `weight_summary.md`：每组的状态、完成折数、准确率、Macro-F1、e₁/e₂/e₃ 召回率，以及逐人较低召回率，数值显示为百分数。
- `weight_summary.csv`：便于导入 Excel 的相同汇总，指标以 0–1 小数保存。
- `subject_results.csv`：每组、每名受试者的指标、最佳 epoch、混淆矩阵与学到的深度融合权重。
- `weight_summary.json`：完整机器可读记录，额外区分合并混淆矩阵后的 pooled 指标。

汇总指标先逐受试者计算，再等权平均；逐人较低召回率为先逐人取 `min(R₁,R₂)` 再平均。尚未开始或未完成的组合会明确标注，不能把部分折均值当作完整 15 折结果。表格按类别权重数值排列，不自动宣布最佳；全部候选完成后再结合准确率、Macro-F1 和两类识别表现选择。

每折仍沿用此前确认的“最高外层测试准确率保存模型”协议，再用这些模型比较类别权重，因此属于探索性选参结果。S6、S17 不参与本轮权重选择。

### 选定一组后，再评估 S6、S17

仅在选定组合且该组 15 折全部完成后执行。下面示例选择 `3,2,1`，把 `EEG_SELECTED_KEY` 改成实际选定的目录名：

```bash
EEG_SELECTED_KEY='weights_3_2_1'
for EEG_CASE_ID in 6 17; do
  taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/evaluate_multidepth_failurecase.py \
    --dataset-root "$EEG_NEW_DATA" \
    --failurecase-root "$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3" \
    --run-root "$EEG_RUN_BASE/$EEG_SELECTED_KEY" \
    --output-dir "$EEG_RUN_BASE/failurecase_${EEG_SELECTED_KEY}_S${EEG_CASE_ID}" \
    --case-subject-id "$EEG_CASE_ID" --seed 42 --device cuda:0 \
    --batch-size 72 --cpu-threads "$EEG_CPU_THREADS" || break
done
```

评估从所选组合的 `run_plan.json` 读取实际层数与分段位置，先重放并核验每折原测试预测，再对失败案例逐窗口等权平均 15 折概率。评估输出要求使用新目录；正式 LOSO 汇总与 Failure case 集成结果分别报告。
