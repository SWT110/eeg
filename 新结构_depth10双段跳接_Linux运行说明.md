# 新结构 + 四个对照模型：Linux 手动运行

本轮训练五个模型：**新结构（channelpreserving）、EEGNet、ShallowConvNet、ATCNet、TCFormer**。正式队列为 S1–S17 中排除 S6、S17 后的 **15 人**；每个模型 15 折 LOSO，共 **75 折**。S6、S17 的原始及预处理记录保留，单独做 Failure case 评估。

新结构为 Time + FFT、每域 10 层、前后各 5 层固定特征跳接。其余超参数使用已确认的仓库本地默认：emb=40、heads=5、dropout=.5、200 epochs、batch=72、lr=.0002、Adam betas=(.5,.999)。四个对照模型沿用仓库已有结构实现；五个模型统一使用上述训练日程、seed=42、**无类别加权 CE、原始采样、无 S&R**。选轮仍沿用每轮留出测试准确率最高时保存最佳权重的历史开发协议。

本轮五模型入口是 `run_activity_channelpreserve_five_models.py`。按下面顺序执行，第 6 节在单 GPU 与双 GPU 中选一种即可。

## 1. 拉取代码与还原输入

```bash
cd /public/home/gd_110413/SWT/work/eeg
git status --short
git pull --ff-only origin main
git log -1 --oneline
```

若有本地改动，先处理后再拉取。GitHub 不包含 `local_artifacts/` 数据；如果之前的 16 人实验已准备好输入，继续使用即可。首次准备时，参照 [原文档第 1 节](新增16人实验_Linux运行说明.md)，将输入包与校验文件放到项目根目录后执行：

```bash
sha256sum -c expanded_cohort_inputs_20261009.tar.gz.sha256
tar -xzf expanded_cohort_inputs_20261009.tar.gz
```

输入包括新增 12–17 的 EDF、时间标注和旧 1–11 的归一化 Excel。S17 仍只保留前三段刺激条件用于失败案例分析，不进入正式训练。

## 2. 准备环境并限制 CPU

以下命令在服务器同一个 Bash 会话中执行：

```bash
EEG_TRAIN_PY="$PWD/.conda-envs/eegconformer310/bin/python"
EEG_PREP_PY="$PWD/.venv-preprocessing/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE

# 自动从当前允许使用的 CPU 中选取，每个作业最多使用 2 个逻辑 CPU。
EEG_CPU_A="$("$EEG_TRAIN_PY" -c 'import os; c=sorted(os.sched_getaffinity(0)); print(",".join(map(str,c[:2])))')"
EEG_CPU_B="$("$EEG_TRAIN_PY" -c 'import os; c=sorted(os.sched_getaffinity(0)); print(",".join(map(str,c[2:4] or c[:2])))')"
printf 'CPU A=%s; CPU B=%s\n' "$EEG_CPU_A" "$EEG_CPU_B"
command -v taskset
nvidia-smi
```

后文用 `taskset` 限制整个作业的 CPU 范围，并用 `--cpu-threads 1` 限制 PyTorch 计算线程。双卡默认分配两组 CPU；允许的 CPU 较少时会共享已有 CPU。`taskset` 由 Linux 的 `util-linux` 提供。

预处理环境若已存在，可直接继续。尚未创建时执行：

```bash
"$EEG_TRAIN_PY" -m venv .venv-preprocessing
"$EEG_PREP_PY" -m pip install -r requirements-preprocessing.txt
```

## 3. 一条命令完成本轮预处理

```bash
taskset -c "$EEG_CPU_A" bash scripts/prepare_expanded_activity_dataset.sh \
  --python "$EEG_PREP_PY" --exclude-subject-ids 6,17
```

该脚本完成新增 12–17 的 EDF 转换、前三段裁切和归一化，再构建正式数据及 S6、S17 的独立数据。旧 1–11 的预处理版本保留。

正式数据路径为：

```text
local_artifacts/data_to_list/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/window_15_stride_3/
```

S6、S17 的独立数据分别为：

```text
local_artifacts/data_to_list/global_activity_failurecase_S6/window_15_stride_3/
local_artifacts/data_to_list/global_activity_failurecase_S17/window_15_stride_3/
```

沿用现有输入版本时，正式数据预期 15 人、45 条记录、7869 个窗口，三类数量为 `[1440,1530,4899]`；S6、S17 各 3 条记录、526 个窗口。每窗 21 通道、1920 点。以数组和 metadata 核验为准。

如果之前 16 人实验已经完成归一化，可以跳过本节，只按下一节从已有 Excel 生成本轮 NPY。

## 4. 已有归一化数据时，单独生成本轮 NPY

```bash
EEG_DATA_BASE="$PWD/local_artifacts/data_to_list"
EEG_NORMALIZED="$EEG_DATA_BASE/list_normalization_fixed_duration"
EEG_NEW_DATA="$EEG_DATA_BASE/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/window_15_stride_3"

if [ ! -f "$EEG_NEW_DATA/metadata.json" ]; then
  taskset -c "$EEG_CPU_A" "$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
    --input-root "$EEG_NORMALIZED" \
    --output-root "$EEG_DATA_BASE/global_activity_dataset_1to17_S17_first3/window_15_stride_3" \
    --include-subject-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
    --exclude-subject-ids 6,17 --require-complete-records \
    --window-seconds 15 --stride-seconds 3
fi

for EEG_CASE_ID in 6 17; do
  EEG_CASE_DATA="$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3"
  if [ ! -f "$EEG_CASE_DATA/metadata.json" ]; then
    taskset -c "$EEG_CPU_A" "$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
      --input-root "$EEG_NORMALIZED" --output-root "$EEG_CASE_DATA" \
      --include-subject-ids "$EEG_CASE_ID" --require-complete-records \
      --window-seconds 15 --stride-seconds 3
  fi
done
```

正式数据的排除后缀由构建器自动添加，`--output-root` 按上面填未带后缀的目录。已有有效数据会复用；中断留下不完整 NPY 时，构建器会拒绝覆盖，应检查后换新目录或显式重建衍生数据。

## 5. 先检查 75 折计划

```bash
EEG_DATA_BASE="$PWD/local_artifacts/data_to_list"
EEG_NEW_DATA="$EEG_DATA_BASE/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/window_15_stride_3"
EEG_RUN_BASE="$PWD/local_artifacts/outputs/activity_channelpreserve_five_models_seed42"
EEG_JOB_BASE="$EEG_RUN_BASE/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/seed_42"

taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_channelpreserve_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --exclude-subject-ids 6,17 --device cuda:0 --cpu-threads 1 --dry-run
```

期望计划为 15 名正式受试者、5 个模型、`planned_training_folds=75`，排除 `[6,17]`，200 epochs、batch=72、lr=.0002、`class_weights=null`。程序核对真实数组、元数据、来源记录以及数据和代码哈希；S6 或 S17 仍在数组里会直接报错。`--dry-run` 不训练、不创建结果目录。

如需先检查五个模型都能运行，可用相同命令将 `--dry-run` 换成 `--smoke-test`。它为每模型仅训练第一折 1 epoch、batch=2，输出到独立的 `smoke_job_...` 目录，不能作为正式结果。

## 6. 开始完整训练

### 单 GPU：一张卡依次完成五个模型

```bash
mkdir -p EEG-Conformer/logs
nohup taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_channelpreserve_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --exclude-subject-ids 6,17 --device cuda:0 --cpu-threads 1 \
  > EEG-Conformer/logs/channelpreserve_five_seed42_all.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/channelpreserve_five_seed42_all.pid
```

单卡结果根目录：

```text
<EEG_JOB_BASE>/job_channelpreserving_eegnet_shallowconvnet_atcnet_tcformer/
```

### 双 GPU：GPU 0 跑新结构，GPU 1 跑四个对照模型

在同一个终端依次执行下面两条后台命令，就会同时运行两个作业。按 `nvidia-smi` 将 `cuda:0`、`cuda:1` 改成实际要用的两张空闲卡。

```bash
mkdir -p EEG-Conformer/logs

# GPU 0：新结构，15 折。
nohup taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_channelpreserve_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --models channelpreserving --exclude-subject-ids 6,17 --device cuda:0 --cpu-threads 1 \
  > EEG-Conformer/logs/channelpreserve_five_seed42_main.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/channelpreserve_five_seed42_main.pid

# GPU 1：EEGNet、ShallowConvNet、ATCNet、TCFormer，依次训练，共 60 折。
nohup taskset -c "$EEG_CPU_B" "$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_channelpreserve_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --models eegnet,shallowconvnet,atcnet,tcformer \
  --exclude-subject-ids 6,17 --device cuda:1 --cpu-threads 1 \
  > EEG-Conformer/logs/channelpreserve_five_seed42_external.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/channelpreserve_five_seed42_external.pid
```

两个作业共享同一正式数据，每个模型均做完整 15 折。它们分别保存在：

```text
<EEG_JOB_BASE>/job_channelpreserving/
<EEG_JOB_BASE>/job_eegnet_shallowconvnet_atcnet_tcformer/
```

每个目录独立保存计划、断点和汇总，评估时分别指定对应目录即可。每个模型在每折都重新初始化。若有更多 GPU，可按 `--models` 分配完整模型；同一个模型只分配给一个作业。

查看日志：

```bash
tail -f EEG-Conformer/logs/channelpreserve_five_seed42_main.log
# 或查看四基线日志：
tail -f EEG-Conformer/logs/channelpreserve_five_seed42_external.log
```

`Ctrl+C` 只退出日志查看，后台训练继续。单卡作业查看 `channelpreserve_five_seed42_all.log`。运行中不要重复启动同一作业。

### 中断后的续训

程序从首轮保存模型、Adam 和随机状态。确认原进程已停止后，在你实际使用的启动命令中加 `--resume`，其他参数保持一致。下面是双卡方案的续训命令：

```bash
nohup taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_channelpreserve_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --models channelpreserving --exclude-subject-ids 6,17 --device cuda:0 --cpu-threads 1 --resume \
  > EEG-Conformer/logs/channelpreserve_five_seed42_main_resume.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/channelpreserve_five_seed42_main_resume.pid

nohup taskset -c "$EEG_CPU_B" "$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_channelpreserve_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --models eegnet,shallowconvnet,atcnet,tcformer \
  --exclude-subject-ids 6,17 --device cuda:1 --cpu-threads 1 --resume \
  > EEG-Conformer/logs/channelpreserve_five_seed42_external_resume.log 2>&1 < /dev/null &
echo $! > EEG-Conformer/logs/channelpreserve_five_seed42_external_resume.pid
```

单卡方案在原单卡命令中加 `--resume` 即可。完成的折会核验后跳过，未完成折恢复上一完整 epoch。数据、代码或参数变化会拒绝续训，避免混合不同实验。

## 7. 完成后单独评估 S6、S17

双卡方案先确认两个完成标记：

```bash
cat "$EEG_JOB_BASE/job_channelpreserving/completed.json"
cat "$EEG_JOB_BASE/job_eegnet_shallowconvnet_atcnet_tcformer/completed.json"
```

应分别是 15 折、60 折完成，且 `smoke_test=false`。然后执行：

```bash
for EEG_CASE_ID in 6 17; do
  # 新结构的 Failure case。
  taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/evaluate_channelpreserve_five_models_failurecase.py \
    --dataset-root "$EEG_NEW_DATA" \
    --failurecase-root "$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3" \
    --run-root "$EEG_JOB_BASE/job_channelpreserving" \
    --output-dir "$PWD/local_artifacts/outputs/activity_channelpreserve_failurecase_S${EEG_CASE_ID}_seed42_main" \
    --case-subject-id "$EEG_CASE_ID" --seed 42 --device cuda:0 --batch-size 72 --cpu-threads 1 || break

  # 四个对照模型的 Failure case。
  taskset -c "$EEG_CPU_B" "$EEG_TRAIN_PY" -u EEG-Conformer/evaluate_channelpreserve_five_models_failurecase.py \
    --dataset-root "$EEG_NEW_DATA" \
    --failurecase-root "$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3" \
    --run-root "$EEG_JOB_BASE/job_eegnet_shallowconvnet_atcnet_tcformer" \
    --output-dir "$PWD/local_artifacts/outputs/activity_channelpreserve_failurecase_S${EEG_CASE_ID}_seed42_external" \
    --case-subject-id "$EEG_CASE_ID" --seed 42 --device cuda:1 --batch-size 72 --cpu-threads 1 || break
done
```

若使用单 GPU 跑完五个模型，则改用下面这组评估命令：

```bash
for EEG_CASE_ID in 6 17; do
  taskset -c "$EEG_CPU_A" "$EEG_TRAIN_PY" -u EEG-Conformer/evaluate_channelpreserve_five_models_failurecase.py \
    --dataset-root "$EEG_NEW_DATA" \
    --failurecase-root "$EEG_DATA_BASE/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3" \
    --run-root "$EEG_JOB_BASE/job_channelpreserving_eegnet_shallowconvnet_atcnet_tcformer" \
    --output-dir "$PWD/local_artifacts/outputs/activity_channelpreserve_failurecase_S${EEG_CASE_ID}_seed42_all" \
    --case-subject-id "$EEG_CASE_ID" --seed 42 --device cuda:0 --batch-size 72 --cpu-threads 1 || break
done
```

每个模型使用自己的 15 份最佳权重，先重播各折原留出测试预测，再预测 S6 或 S17。所有概率固定后才计算失败案例指标；逐窗口只沿 15 折模型维度等权平均。每种架构单独汇总，不把五种架构混成一个分类器。失败案例不参与训练、选轮或调参；已有评估目录拒绝覆盖，重跑需换新目录。

## 8. 结果位置与验证边界

每个作业根目录有 `run_plan.json`、`environment.json`、`status.json`、`five_model_summary.json`，全部完成后有 `completed.json`。新结构结果在 `channelpreserving/seed_42/fold_subject_<id>/`；四个对照模型在 `external/<模型名>/seed_42/fold_subject_<id>/`。

各折保存 `best_model.pt`、`metrics.json`、`test_predictions.npz`、训练历史和日志。各模型的 `summary.json`、`summary.csv` 保存正式 LOSO 汇总。Failure case 目录中的 `failurecase_summary.json` 保存该作业所含模型的综合指标，每个模型子目录另有 15 份单折结果与综合预测。

正式 LOSO 与 S6/S17 的失败案例集成指标分别报告。本地验证使用合成数据检查五模型训练、权重恢复、原测试预测重播、续训和独立评估；真实 75 折 CUDA 训练与失败案例推理由你在服务器执行。
