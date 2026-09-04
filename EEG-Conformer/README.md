# EEG-Conformer

这个目录现在不是上游 `eeyhsong/EEG-Conformer` 的原样镜像，而是面向当前项目整理后的活动三分类训练子项目。

保留内容只覆盖你现在实际在用的流程：

1. 基于全局窗口数据集做 LOSO 三分类训练
2. 批量跑完整套 LOSO 实验并汇总结果
3. 用全部数据训练最终部署模型
4. 对单个 xlsx EEG 文件做预测
5. 可选地通过 FastAPI + 简单网页做演示

当前模型结构仍然来自原版 EEG-Conformer，但已经内嵌到 `train_activity_loso.py` 中；因此原仓库里针对公开数据集的旧训练脚本、MATLAB 预处理脚本和可视化代码已经不再作为当前工作流的一部分。

## 目录定位

更完整的中文说明请看仓库根目录下这两份文档：

- `../EEG_服务器改动与使用说明.md`
- `../EEG-Conformer_LOSO_三分类完整流程规划.md`

如果只看本目录，可以把它理解成：

```text
eeg-data-processing/data_to_list/build_activity_global_index.py
-> EEG-Conformer/train_activity_loso.py
-> EEG-Conformer/train_activity_loso_batch.py
-> EEG-Conformer/summarize_loso_results.py
-> EEG-Conformer/train_activity_final.py
-> EEG-Conformer/predict_activity_signal.py
```

## 保留的主要脚本

| 文件 | 作用 |
| --- | --- |
| `train_activity_loso.py` | 正式的 LOSO 单 fold 三分类训练入口 |
| `train_activity_loso_batch.py` | 批量跑完单套数据集的全部 LOSO fold |
| `train_activity_loso_generated_batch.py` | 批量遍历多套 window/stride 数据集并自动汇总 |
| `train_activity_comparison.py` | 在相同 LOSO 协议下运行 EEGNet、ShallowConvNet、ATCNet 和 TCFormer |
| `comparison_models.py` | 四个外部基线的 plain-PyTorch 适配与可追溯架构配置 |
| `summarize_loso_results.py` | 汇总准确率、混淆矩阵、Macro-F1 |
| `train_activity_final.py` | 用全部数据训练最终部署模型 |
| `predict_activity_signal.py` | 对单个 EEG xlsx 做整段活动预测 |
| `app_activity_api.py` | FastAPI 服务，封装模型选择与上传预测 |
| `index.html` | API 对应的简单前端页面 |
| `setup_env_jupyter.py` | 在服务器上创建项目内 conda 环境和 Jupyter kernel |

## 数据输入约定

正式 LOSO 和最终模型流程都依赖由外部脚本构建好的全局数据集目录，例如：

```text
../local_artifacts/data_to_list/global_activity_dataset/
├── X.npy
├── y.npy
├── subject_ids.npy
├── record_ids.npy
├── window_indices.npy
└── metadata.json
```

其中：

- `X.npy` 形状为 `(N, C, T)`
- `y.npy` 为类别标签
- `subject_ids.npy` 用于 LOSO 划分
- `metadata.json` 保存 `label_map`、窗口长度、步长等信息

## 常用命令

单套数据集跑完整 LOSO：

```bash
python train_activity_loso_batch.py \
  --dataset-root ../local_artifacts/data_to_list/global_activity_dataset \
  --device cuda:0
```

三个并行深度分支分别分类并使用可学习 loss 权重：

```bash
python train_activity_loso_batch.py \
  --dataset-root ../local_artifacts/data_to_list/global_activity_dataset/window_15_stride_3 \
  --input-domain time_fft \
  --transformer-branches 3 \
  --transformer-depths 11 10 8 \
  --transformer-branch-fusion loss_softmax \
  --branch-loss-aux-weight 0.2 \
  --device cuda:1
```

`loss_softmax` 为每个对应深度的 time/FFT 特征配置独立分类头，并学习三个
softmax loss 权重；辅助系数用于保证低权重分支仍能得到监督。不传这两个参数时，
仍使用原有的特征级 softmax 融合。

在三个深度编码器输出与分类头之间启用 Cross-Depth QKV：

```bash
--transformer-branch-qkv cross_depth
```

该选项要求同时使用 `time_fft`、并行深度分支和 `loss_softmax`。Time 与 FFT
分别拥有独立的 QKV 模块；默认值为 `none`，因此旧命令和旧 checkpoint 结构不变。

### 中断后恢复训练（含 Adam optimizer）

单 fold 和批量 LOSO 均支持 `--resume`。批量训练建议同时使用：

```bash
python train_activity_loso_batch.py \
  ...原实验参数... \
  --skip-existing \
  --resume
```

启用后，每个未完成 fold 会在每个完整 epoch 后原子更新
`fold_subject_<id>/last_checkpoint.pt`，其中包含：

- 模型 `state_dict`
- Adam `optimizer_state_dict`（包括一阶、二阶动量）
- 已完成 epoch、累计测试准确率和完整 epoch history
- 当前最佳指标、预测及分支权重元数据
- Python、NumPy、PyTorch CPU/CUDA RNG 状态

重启时会校验数据形状、模型结构、batch size、学习率、seed、融合方式等配置；
配置不一致会拒绝恢复。可以增大 `--epochs`，但不能将其设为小于已完成 epoch。
fold 成功生成 `metrics.json` 后会删除较大的 `last_checkpoint.pt`，只保留
`best_model.pt` 和结果文件。原子写入保证保存失败时不会覆盖上一个有效的恢复点；
批量脚本检测到磁盘写满或 quota 超限后会立即终止剩余 folds，不再逐个制造失败目录。

旧实验产生的 `best_model.pt` 不含 optimizer 状态，不能真正续接到原 epoch；若没有
新版 `last_checkpoint.pt`，`--resume` 会明确提示并从该 fold 的 epoch 1 重跑。

批量比较多套 window/stride：

```bash
python train_activity_loso_generated_batch.py \
  --device cuda:0
```

运行外部模型公平比较（完整文献依据、服务器命令和验收规则见
`../对比实验方案与运行命令.md`）：

```bash
python train_activity_comparison.py \
  --dataset-root ../local_artifacts/data_to_list/global_activity_dataset/window_15_stride_3 \
  --models eegnet,shallowconvnet,atcnet,tcformer \
  --seeds 42 \
  --epochs 200 \
  --batch-size 72 \
  --lr 0.0002 \
  --class-weights 3,3,1 \
  --device cuda:0 \
  --resume \
  --skip-existing
```

训练最终部署模型：

```bash
python train_activity_final.py \
  --dataset-root ../local_artifacts/data_to_list/global_activity_dataset \
  --device cuda:0
```

预测单个新文件：

```bash
python predict_activity_signal.py \
  --input /path/to/sample.xlsx \
  --model-dir ../local_artifacts/outputs/activity_final \
  --device cpu
```

默认情况下，数据集、训练输出和 API 缓存都会解析到项目根目录的 `local_artifacts/`。如需在服务器使用其他 artifact 根目录，可设置 `EEG_LOCAL_ARTIFACTS_ROOT`。

启动本地 API：

```bash
python app_activity_api.py
```

## 环境准备

如果服务器驱动较旧，或者你想把运行环境完全收在项目目录内，可以使用：

```bash
python setup_env_jupyter.py --dry-run
python setup_env_jupyter.py
```

脚本会在当前目录内创建：

- `.conda-envs/eegconformer310`
- `.conda-pkgs`

并注册 `Python (eegconformer310)` 内核。

## 与上游项目的关系

- 上游仓库：`https://github.com/eeyhsong/EEG-Conformer`
- 当前目录保留了原始模型思路，但训练入口、数据组织方式和部署流程都已经针对当前 EEG 项目改写
- 如果你使用了这套模型结构，请同时参考原论文与上游仓库说明

## Citation

```text
@article{song2023eeg,
  title = {EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization},
  author = {Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  year = {2023},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume = {31},
  pages = {710--719},
  doi = {10.1109/TNSRE.2022.3230250}
}
```
