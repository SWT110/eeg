# EEG-Conformer LOSO 三分类完整流程规划

## 1. 目标定义

本项目的最终目标不是区分受试者，而是训练一个**活动三分类模型**：

- 输入：一段 EEG 信号（最终会切成固定长度窗口）
- 输出：该信号属于活动 `1 / 2 / 3` 中的哪一类

当前明确采用的评估方式是 **LOSO（Leave-One-Subject-Out，留一受试者法）**：

- 共 `11` 个受试者
- 每轮拿 `1` 个受试者做测试集
- 其余 `10` 个受试者做训练集
- 模型只预测**活动类别**，不预测受试者编号

这更接近原版 EEG-Conformer 论文代码里的“按实验协议做完整训练与评估”，而不是只做一次 smoke run。

---

## 2. 当前数据理解

当前原始数据位于：

```text
eeg-data-processing/data_to_list/list_normalization_fixed_duration/
  1/
    1_e_1.xlsx
    1_e_2.xlsx
    1_e_3.xlsx
  2/
    2_e_1.xlsx
    2_e_2.xlsx
    2_e_3.xlsx
  ...
  11/
    11_e_1.xlsx
    11_e_2.xlsx
    11_e_3.xlsx
```

含义是：

- `1..11`：受试者编号
- `e_1 / e_2 / e_3`：三类活动
- 每个 xlsx 是一段连续 EEG 信号

这意味着训练样本不是整份 xlsx，而是：

```text
连续 EEG
-> 固定时间窗口切分
-> 很多窗口样本
-> 每个窗口继承所属活动标签
```

---

## 3. 为什么当前 `holdout_group` 方案不适合作为最终流程

早期已经跑通过一条单 subject smoke run 链路（现已从当前仓库移除），本质上是：

- 以单个受试者为单位训练
- 用 `groups.npy` 做 holdout 划分
- 保存 `subject_x_holdout_y.pt`

这条链路的价值是：

- 验证 CUDA 环境可用
- 验证数据能进入模型
- 验证训练代码能正常跑完

但它**不是最终的三分类实验方案**，原因有 3 点：

| 问题 | 当前方案 | 最终需求 |
|---|---|---|
| 预测目标 | 单 subject 内部训练 | 跨 subject 的活动分类 |
| 划分方式 | `groups` 留出 | LOSO：留一整个 subject 做测试 |
| 结果产物 | `33` 个中间 checkpoint | `11` 个 LOSO fold 的评估结果 + 可选最终模型 |

所以，这条旧链路属于“训练链路打通阶段”，不是“论文式完整实验阶段”。

---

## 4. 目标流程（推荐的完整版本）

完整流程建议按下面 6 个阶段推进：

| 阶段 | 目标 | 核心输出 |
|---|---|---|
| 阶段 A | 从 xlsx 构建统一窗口级数据集 | 可直接读取的窗口数据与元信息 |
| 阶段 B | 构建 LOSO fold 规则 | `11` 个 fold 的 train / val / test 定义 |
| 阶段 C | 实现全局三分类训练入口 | 单个脚本可跑任一 LOSO fold |
| 阶段 D | 实现完整评估与结果汇总 | 每个 fold 的 acc、macro-F1、混淆矩阵 |
| 阶段 E | 训练所有 `11` 个 LOSO fold | 完整实验结果 |
| 阶段 F | 训练可部署的最终模型与预测脚本 | 用于实际识别新信号的模型与推理入口 |

---

## 5. 阶段 A：数据集重构

### 5.1 目标

把当前按 subject 分散保存的窗口数据，升级成适合做 **全局活动分类** 的数据读取方式。

### 5.2 推荐策略

保留现有 `xlsx -> npy` 转换脚本，但在训练阶段按**全局样本池**读取：

- `X`：窗口信号
- `y`：活动标签（`0 / 1 / 2`）
- `subject_id`：窗口来自哪个 subject
- `source_file` 或 `record_id`：窗口来自哪个原始活动文件
- `window_index`：窗口在原信号中的顺序位置

### 5.3 为什么需要新增元信息

LOSO 的关键不是 `groups`，而是：

- 哪些样本来自测试 subject
- 哪些样本来自训练 subject
- 验证集如何从训练 subject 内部切出来

因此最终训练数据至少要支持以下过滤：

```text
按 subject_id 过滤
按 activity_label 过滤
按原始记录和时间位置过滤
```

### 5.4 推荐产物

推荐新增一个“全局索引文件”或“metadata 文件”，例如：

```text
eeg-data-processing/data_to_list/npy_dataset_5s_global/
  X.npy
  y.npy
  subject_ids.npy
  record_ids.npy
  window_starts.npy
  metadata.json
```

这样后续 LOSO 训练脚本就不需要反复自己扫描目录。

---

## 6. 阶段 B：LOSO 划分设计

### 6.1 测试集

每个 fold 固定留出 `1` 个 subject 做测试：

- Fold 1：`subject_1` 测试，其余训练
- Fold 2：`subject_2` 测试，其余训练
- ...
- Fold 11：`subject_11` 测试，其余训练

### 6.2 验证集

验证集**不能**再用“留出一个完整活动文件”的方式做，因为那会让某些类别在训练中缺失。

更合理的做法是：

- 只在训练 subject 内部生成验证集
- 对每个 subject 的每个活动文件，按时间顺序切出一部分窗口作为验证

推荐方式：

- 前 `80%` 窗口：训练
- 后 `20%` 窗口：验证

如果窗口重叠较多，必须用**按时间连续切分**，不要完全随机打乱，否则会造成信息泄漏。

### 6.3 每个 fold 的数据语义

每个 LOSO fold 应满足：

- `train subjects`：`10` 个
- `val subjects`：仍来自这 `10` 个 subject 内部，但只取每类活动文件的后部窗口
- `test subjects`：整套留出的 `1` 个 subject

这样模型最终评估的就是：

**模型是否能在“从未见过的 subject”上正确区分 3 类活动。**

---

## 7. 阶段 C：训练入口设计

### 7.1 新训练脚本职责

建议新增一个新的全局训练入口，例如：

```text
EEG-Conformer/train_activity_loso.py
```

它和早期 smoke run 链路的区别是：

| 脚本 | 定位 |
|---|---|
| 早期 smoke run 链路（已移除） | 单 subject 内部的 `groups` holdout 验证 |
| `train_activity_loso.py` | 全局活动三分类的正式 LOSO 训练入口 |


### 7.2 训练脚本需要做的事

1. 读取全局窗口数据和元信息  
2. 根据 `--test-subject-id` 构造当前 LOSO fold  
3. 在训练 subject 内部再切 train / val  
4. 建立 EEG-Conformer 模型  
5. 训练并保存最佳 checkpoint  
6. 在测试 subject 上做最终评估  
7. 保存本 fold 的指标结果

### 7.3 推荐参数

```text
--dataset-root
--test-subject-id
--window-seconds
--batch-size
--epochs
--lr
--device
--output-dir
--seed
```

---

## 8. 阶段 D：结果与评估

这一步是“像原版 EEG-Conformer 一样的完整流程”最关键的部分。

最终不能只输出 `.pt`，还要输出实验结果。

### 8.1 每个 fold 至少输出

- `test_accuracy`
- `macro_f1`
- `per_class_precision`
- `per_class_recall`
- `confusion_matrix`
- 训练过程中的 `train_loss / val_loss / val_acc`

### 8.2 全部 `11` 个 fold 汇总

需要一个汇总脚本，例如：

```text
EEG-Conformer/summarize_loso_results.py
```

输出：

- 11 个 fold 的结果表
- 平均准确率
- 平均 macro-F1
- 各类别平均召回率
- 总体混淆矩阵

推荐保存成：

```text
outputs/activity_loso/
  fold_1/
  fold_2/
  ...
  fold_11/
  summary.csv
  summary.json
  confusion_matrix.png
```

---

## 9. 阶段 E：最终用于实际预测的模型

### 9.1 论文式实验结果

如果你的目标是得到“像原版 EEG-Conformer 一样的完整实验结果”，那么：

- 主要产物是 `11` 个 LOSO fold 的评估结果
- `.pt` 是每个 fold 的模型快照

### 9.2 实际部署 / 实际识别

如果你的目标还包括“后面输入一段新 EEG，让模型告诉我是活动几”，那么在完成 LOSO 实验后，还需要再训练一个**最终部署模型**。

推荐方式：

- 用全部 `11` 个 subject 的训练数据
- 按每类活动内部切一部分做验证
- 训练一个最终三分类模型

最终产物类似：

```text
outputs/activity_final/
  final_model.pt
  train_config.json
  normalization_stats.json
  label_map.json
```

这个 `final_model.pt` 才是后续实际预测最直接使用的模型。

---

## 10. 阶段 F：预测流程

还需要新增一个推理脚本，例如：

```text
EEG-Conformer/predict_activity_signal.py
```

### 它要做的事

1. 输入一段新的 EEG 信号  
2. 做和训练时一致的预处理  
3. 切成同样长度的窗口  
4. 用 `final_model.pt` 对每个窗口做预测  
5. 汇总窗口结果，输出整段信号属于活动 `1 / 2 / 3`

### 推荐汇总方式

- 窗口 softmax 概率取平均
- 取平均后概率最大的类别作为整段预测结果

这样比简单多数投票更稳定。

---

## 11. 建议新增或修改的文件

| 文件 | 作用 |
|---|---|
| `eeg-data-processing/data_to_list/8.xlsx_to_npy_dataset.py` | 继续负责基础窗口切分，必要时补充全局元信息导出 |
| `eeg-data-processing/data_to_list/build_activity_global_index.py` | 构建全局窗口索引与 metadata |
| `EEG-Conformer/train_activity_loso.py` | 正式的 LOSO 三分类训练脚本 |
| `EEG-Conformer/train_activity_loso_batch.py` | 一次性跑完 11 个 LOSO fold |
| `EEG-Conformer/summarize_loso_results.py` | 汇总 11 个 fold 的结果 |
| `EEG-Conformer/predict_activity_signal.py` | 对新 EEG 信号做活动预测 |
| `EEG-Conformer/tests/...` | 对数据索引、fold 划分、训练入口、结果汇总、预测流程做测试 |

---

## 12. 推荐执行顺序

| 顺序 | 任务 | 说明 |
|---|---|---|
| 1 | 重构数据索引 | 让训练脚本能按全局样本池工作 |
| 2 | 实现 LOSO fold 划分 | 先把 train / val / test 规则固定下来 |
| 3 | 实现单 fold 训练脚本 | 先跑通 `subject_1` 作为测试集 |
| 4 | 实现批量 LOSO 训练 | 跑完全部 11 个 fold |
| 5 | 实现结果汇总 | 生成完整实验表和图 |
| 6 | 实现最终模型训练 | 训练用于真实预测的最终模型 |
| 7 | 实现预测脚本 | 输入新信号，输出活动类别 |

---

## 13. 这次规划完成后，你真正会得到什么

完成这个完整流程后，你最终会有两套产物：

### A. 论文式实验产物

- `11` 个 LOSO fold 的结果
- 平均准确率、macro-F1、混淆矩阵
- 每个 fold 的 checkpoint

### B. 实际可用产物

- `1` 个最终三分类模型
- `1` 个预测脚本
- 输入一段新 EEG 信号，输出属于活动 `1 / 2 / 3`

---

## 14. 当前最关键的结论

你现在需要的不是继续扩展 `holdout_group` 方案，而是切换到一条新的正式实验线：

```text
xlsx
-> 全局窗口数据集
-> LOSO fold 划分
-> 单 fold 训练
-> 11 fold 批量训练
-> 指标汇总
-> 最终模型训练
-> 新信号预测
```

这才是和原版 EEG-Conformer 更一致的“完整流程”。
