# 新增 16 人实验：Linux 手动运行

本轮正式队列为 S1–S17 中除 S6 外的 16 人。S17 只取前三段，依次对应傅里叶变换、通信原理、心灵捕手，时长 300/318/998 秒。S6 从全局窗口数据集生成阶段剔除，原始及预处理记录保留，五个模型均不会训练它。

五个模型是历史准确率 72.13% 的 MDTF-EEGConformer，以及 EEGNet、ShallowConvNet、ATCNet、TCFormer。本轮沿用每轮评估留出受试者、保存最高测试 accuracy 的旧开发协议。数据队列已经改变，新的结果与原 11 人结果分别保存。

## 1. 拉取代码与还原输入

以下沿用历史 Linux 项目路径；如果你的实际项目位置不同，修改第一条 `cd`。

```bash
cd /public/home/gd_110413/SWT/work/eeg
git pull --ff-only
```

GitHub 不包含 `local_artifacts/` 的大文件。请先将本次输入包和校验文件放到服务器的项目根目录：

- `expanded_cohort_inputs_20261009.tar.gz`，约 900 MB。
- `expanded_cohort_inputs_20261009.tar.gz.sha256`。

输入包内含新增 12–17 的六个原始 EDF、完整 `time_data.txt`、旧 1–11 的 33 份归一化 Excel，以及源文件 SHA-256 清单。旧 S6 的三条记录也保留，供单独 Failure case 评估；S17 的原始六条时间仍完整保留。

```bash
sha256sum -c expanded_cohort_inputs_20261009.tar.gz.sha256
tar -xzf expanded_cohort_inputs_20261009.tar.gz
```

解包后应有：

```text
local_artifacts/data_to_list/
├── data/
│   ├── 12_e.edf ... 17_e.edf
│   └── time_data.txt
├── list_normalization_fixed_duration/
│   ├── 1/1_e_1.xlsx ... 1_e_3.xlsx
│   ├── ...
│   └── 11/11_e_1.xlsx ... 11_e_3.xlsx
└── expanded_cohort_input_manifest_20261009.json
```

## 2. 创建独立预处理环境

用现有 CUDA 环境中的 Python 3.10 创建一个独立环境；该步骤不会替换原训练环境的 PyTorch。

```bash
.conda-envs/eegconformer310/bin/python -m venv .venv-preprocessing
.venv-preprocessing/bin/python -m pip install -r requirements-preprocessing.txt
```

这些依赖覆盖 EDF 读取、Excel 转换、标准化和 NPY 构建。Mac 实测 SciPy 1.15.3 在当前系统中加载失败，独立预处理环境采用通过实际导入与 EDF 转换验证的 1.14.1；同类加载错误也有 [SciPy 项目问题记录](https://github.com/scipy/scipy/issues/25635)。

## 3. 一条命令完成本轮预处理

```bash
bash scripts/prepare_expanded_activity_dataset.sh \
  --python "$PWD/.venv-preprocessing/bin/python" \
  --exclude-subject-ids 6
```

该脚本执行：新增 12–17 EDF 转完整 Excel → 只裁切前三段 → 新六人逐段标准化 → 与保留的旧 1–11 记录合并切窗 → 主数据集排除 S6 → 单独生成 S6 数据集。

旧 1–11 不重新裁切或标准化，保留原实验的输入。尤其旧 S3 第三段实际较短，不能为了本次更新而把它悄悄改成另一个预处理版本。

正式数据路径：

```text
local_artifacts/data_to_list/global_activity_dataset_1to17_S17_first3_exclude_S6/window_15_stride_3/
```

S6 独立数据路径：

```text
local_artifacts/data_to_list/global_activity_failurecase_S6/window_15_stride_3/
```

正式数据应为 16 人、48 条记录、21 通道，每窗口 1920 点；根据保留的旧数据与完整新增时段，预期 8395 个窗口。S6 为 1 人、3 条记录、526 个窗口。以实际数组和 metadata 核验为准。

需要排除多人时，参数写成 `--exclude-subject-ids 6,8`，主输出上级目录自动变为 `..._exclude_S6_S8/`，并给每位排除者分别生成独立数据集。已经存在的 NPY 数据集默认拒绝覆盖；确认重建衍生产物时，加 `--rebuild-datasets`，或换一个 `--output-base`。

## 4. 理解各步的独立命令

当需要定位某一步的问题时，可以逐步执行：

```bash
EEG_PREP_PY="$PWD/.venv-preprocessing/bin/python"

"$EEG_PREP_PY" -u eeg-data-processing/data_to_list/2.edf_to_list.py \
  --subject-ids 12 13 14 15 16 17

"$EEG_PREP_PY" -u eeg-data-processing/data_to_list/3.time_edit_fixed_duration.py \
  --subject-ids 12 13 14 15 16 17

"$EEG_PREP_PY" -u eeg-data-processing/data_to_list/6.edf_list_normalization_fixed_duration.py \
  --subject-ids 12 13 14 15 16 17

"$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
  --input-root local_artifacts/data_to_list/list_normalization_fixed_duration \
  --output-root local_artifacts/data_to_list/global_activity_dataset_1to17_S17_first3/window_15_stride_3 \
  --include-subject-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
  --exclude-subject-ids 6 --require-complete-records \
  --window-seconds 15 --stride-seconds 3

"$EEG_PREP_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
  --input-root local_artifacts/data_to_list/list_normalization_fixed_duration \
  --output-root local_artifacts/data_to_list/global_activity_failurecase_S6/window_15_stride_3 \
  --include-subject-ids 6 --require-complete-records \
  --window-seconds 15 --stride-seconds 3
```

EDF/裁切/标准化使用空格分隔的编号；数据集生成的纳入与排除名单使用逗号分隔。全局生成命令的 `--output-root` 填未加排除后缀的路径，脚本会自动添加。

第 2、3、6 步通常复用已有有效缓存。若同名 EDF 或时间标注改变，应先使用新中间输出目录重新转换/裁切；裁切脚本的 `--overwrite` 可显式重切。本轮提供的源文件和选择策略已固定，无需对旧记录做这些操作。

## 5. 先检查 80 折计划

继续使用现有的 CUDA 训练环境：

```bash
EEG_TRAIN_PY="$PWD/.conda-envs/eegconformer310/bin/python"
EEG_NEW_DATA="$PWD/local_artifacts/data_to_list/global_activity_dataset_1to17_S17_first3_exclude_S6/window_15_stride_3"
EEG_RUN_BASE="$PWD/local_artifacts/outputs/activity_expanded_cohort_seed42"

"$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --exclude-subject-ids 6 --device cuda:0 --dry-run
```

期望计划包含：16 名正式受试者、没有 S6、5 个模型、`planned_training_folds=80`、seed 42、200 epochs、batch 72、学习率 0.0002、类别权重 3/3/1。

启动前会核对数组中的真实编号、三类标签、形状、元数据，并计算数据和代码哈希。S6 若仍存在，会直接报错。`--dry-run` 不构造模型、不训练、不创建运行目录。

## 6. 开始完整训练

```bash
"$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --exclude-subject-ids 6 --device cuda:0
```

主模型固定使用 time+FFT、11/10/8 深度、cross-depth QKV、loss-softmax 融合、0.85/0.25 dropout、额外平均分支损失系数 0.2、flat 分类。每折从头初始化。四个外部模型使用现有实现；五个模型共享同一正式队列和旧选轮规则。

程序从首轮就保存优化器和随机状态断点。发生中断后，用相同参数追加 `--resume`：

```bash
"$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --exclude-subject-ids 6 --device cuda:0 --resume
```

数据、代码或参数哈希改变时会拒绝续训，避免把不同实验拼接。已有完整折跳过，未完成折恢复 Adam、随机状态与历史记录。

本次完整单 GPU 作业的根目录为：

```bash
EEG_JOB_ROOT="$EEG_RUN_BASE/global_activity_dataset_1to17_S17_first3_exclude_S6/seed_42/job_mdtf_eegnet_shallowconvnet_atcnet_tcformer"
```

其中主模型在 `mdtf/seed_42/fold_subject_<id>/`，四基线在 `external/<模型名>/seed_42/fold_subject_<id>/`。各折保存权重、预测、训练历史和指标；全部完成后有 `completed.json` 和 `five_model_summary.json`。

若需要使用两张 GPU，可以用两个独立的模型子集命令，输出目录会按模型列表分开：

```bash
# GPU 0：主模型，16 折
"$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" --models mdtf --device cuda:0

# 在另一终端 GPU 1：四基线，64 折
"$EEG_TRAIN_PY" -u EEG-Conformer/run_activity_five_models.py \
  --dataset-root "$EEG_NEW_DATA" --output-base "$EEG_RUN_BASE" \
  --models eegnet,shallowconvnet,atcnet,tcformer --device cuda:1
```

这种拆分对应两个 job 根目录，S6 评估时分别传入各自的 `--run-root`。只选择部分测试受试者的调试作业不能生成完整 Failure case 汇总。

## 7. 完成后单独评估 S6

```bash
"$EEG_TRAIN_PY" -u EEG-Conformer/evaluate_activity_failurecase.py \
  --dataset-root "$EEG_NEW_DATA" \
  --failurecase-root "$PWD/local_artifacts/data_to_list/global_activity_failurecase_S6/window_15_stride_3" \
  --run-root "$EEG_JOB_ROOT" \
  --output-dir "$PWD/local_artifacts/outputs/activity_expanded_failurecase_S6_seed42" \
  --case-subject-id 6 --device cuda:0 --batch-size 72
```

每个模型的 16 份最佳权重先重播各自原留出测试预测，核对架构与训练归一化恢复正确，然后各预测 S6 一次。所有概率固定后才使用 S6 标签计算指标；S6 不参与训练、选 epoch、调参或选择融合权重。

逐窗口概率仅沿 16 折模型维度等权平均，得到每种架构一张综合 3×3 矩阵；同时保留 16 张单折矩阵、原始概率、准确率、Macro-F1 与各类召回。行是真实 e_1/e_2/e_3，列是预测类别。16 次预测仍对应 S6 的 526 个窗口，不能当成 8416 个独立样本。

主要输出为 `failurecase_summary.json`、每个模型的 `summary.json`、`normalizers.json` 和 `evaluation_manifest.json`。Failure case 综合矩阵是 16 个模型的概率平均结果，与正式 LOSO 的单折均值分开报告。已有评估输出目录拒绝覆盖，补跑需选新目录。

## 8. 本次交付的验证边界

175 项回归测试及 2 项子测试通过，覆盖时间解析（包括 S15 控制字符与 S17 六条只取前三条）、多人排除、目录标记、旧记录保留、缓存损坏与写入中断保护、训练调度参数、数据/代码哈希续训保护，以及 S6 评估。五种真实模型类还各用两折合成数据完成训练、最佳权重读取、原测试预测重播与失败案例概率平均，共 10 个合成训练折。本机实际完成了 12、13 的 EDF 转 Excel，均为 128 Hz。

完整 16 人数据集构建、80 折真实 CUDA 训练及新模型 S6 推理由你在 Linux 执行。本轮没有产生新的真实训练准确率。S6 的持续分类失败支持单独复核，采集异常仍是待验证的解释。
