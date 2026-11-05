# EEG 模型比较：服务器与数据只读审计

审计日期：2026-11-01（会话的 America/New_York 日期）。显存与利用率为审计时的瞬时快照。没有启动训练、占用新 GPU、杀进程或修改远程文件。

2026-11-02讨论更新：用户已确认15人队列、现有整段标准化/15秒窗口/3秒步长输入，并明确选择旧最高测试准确率选轮，作为探索性实验。最终骨干由用户根据完整结果分析选取，不预设自动裁定胜者的主指标。下文资源读数仍是原审计时快照；待决定清单中的历史条目请结合本更新与experiment_plan.md阅读。

## 已确认的服务器事实

两个 SSH 别名均可用，登录用户均为 `gd_110413`，连接时启用 `BatchMode=yes`。

| SSH 别名 | 容器 hostname | 可见 GPU 数 | 仓库 HEAD |
| --- | --- | ---: | --- |
| `school-eeg` | `a3e679f3cfa9` | 2 | `8af2010f9af5be159ac42d559bdaf121831e5489` |
| `school-eeg-2` | `6de29035c2ec` | 4 | `8af2010f9af5be159ac42d559bdaf121831e5489` |

所有可见卡均为 NVIDIA A10，`nvidia-smi` 报告每卡总显存 22,731 MiB。以下六个 UUID 全部不同，两个容器之间没有重复列出同一 GPU。

| 别名 | 容器内编号 | GPU UUID | 已用显存 MiB | 利用率 |
| --- | ---: | --- | ---: | ---: |
| `school-eeg` | 0 | `GPU-f2e48d32-e06f-0d8e-1daf-1b308ca64783` | 6509 | 100% |
| `school-eeg` | 1 | `GPU-6c454bdd-a90d-89fd-11cd-d3ebd9cc1fd3` | 2 | 0% |
| `school-eeg-2` | 0 | `GPU-100cf18b-1173-1982-d644-96da8c1b954e` | 0 | 0% |
| `school-eeg-2` | 1 | `GPU-4dfadde7-eb76-8d4c-126c-25f402950e54` | 0 | 0% |
| `school-eeg-2` | 2 | `GPU-5819f585-4ffe-ec02-ed47-5e36567ccc4c` | 14371 | 0% |
| `school-eeg-2` | 3 | `GPU-ce91c9e3-cca3-668e-f2b8-1133cf03debb` | 14271 | 100% |

审计时六张可见卡中三张空闲。用户授权的是可使用四张卡；具体四张的分配仍需确认，不能将六张可见卡自动解释为六张均获授权，也不能抢占已有任务。

`school-eeg` 的进程命名空间中能看到一个本项目 `train_activity_multidepth_weight_sweep.py` 进程（容器内 PID 30542），使用排除 S6/S17 的 15 人数据与 `cuda:0`，分支深度 `2,5,8`、分割位置 `1,2,4`，输出到 `activity_channelpreserve_D_2_5_8_weights_seed42`。本次未触碰它。`school-eeg-2` 的命名空间中未看到本项目训练脚本。`nvidia-smi` 列出的占用进程显示 `[Not Found]`，对应 PID 在容器中无法直接查询，因此无法仅凭这些 PID 确认所有占用者或关联训练任务。

### 共享存储

仓库均为 `/public/home/gd_110413/SWT/work/eeg`。两个容器中 `local_artifacts` 的 `stat` 结果完全相同：设备号及 inode 为 `42:1297051139`。两端 `df -T` 均显示 `A12_Stor_public` / `parastor`，挂载点为 `/public/home/gd_110413`。这是共享目录，而非两个独立副本。

后续调度建议：

- 代码和新实验配置部署一次即可，两端会读同一份内容；运行中应冻结代码版本。
- 每个模型、seed、外层受试者及候选超参数任务使用唯一输出目录与互斥领取机制，防止两台容器重复执行或覆盖文件。
- `cuda:0` 是容器局部编号，两个容器的 `cuda:0` 指向不同 UUID。调度记录必须同时保存 SSH 别名和 GPU UUID。
- 先采用单 GPU 独立实验任务队列；是否需要单模型跨卡训练，应由后续性能测量决定。

这些为执行建议，尚未部署调度器或启动任务。

## 已确认的环境和代码

当前训练 Python：

`/public/home/gd_110413/SWT/work/eeg/.conda-envs/eegconformer310/bin/python`

| 项目 | 实测结果 |
| --- | --- |
| Python | 3.10.20 |
| NumPy | 1.26.4 |
| PyTorch | 1.12.1+cu113 |
| PyTorch CUDA runtime | 11.3 |
| NVIDIA driver（`school-eeg-2`） | 465.19.01 |
| `torch.cuda.is_available()` | 两端均为 True |
| PyTorch 可见卡数 | 分别为 2、4 |

`EEG-Conformer/.conda-envs/` 不是实际环境路径。仓库根目录的 `.venv-preprocessing/bin/python` 链接到上述 conda Python。

新模型如果依赖 PyTorch 2.x、更新 CUDA 或额外算子，需要先核验驱动与依赖兼容性；目前没有升级共享环境。没有进行模型速度、显存峰值或单折时间测量，因此尚不能据此承诺全套实验完成时间。

另外抽查扩展 16 人实验中 MDTF、EEGNet、TCFormer 的 `fold_subject_1`：`metrics.json`、`epoch_history.json` 和无时间戳的 `train.log` 未提供可用的始末时间或训练耗时记录。对应训练实现没有记录 epoch duration，不能可靠给出历史每折时间；不使用文件修改时间估算。正式预算仍需在已确认协议下进行后续短跑实测。

服务器具备以下入口，`train_activity_subject_validation` 的只读导入已通过（禁用了 Python 字节码缓存写入）：

- `EEG-Conformer/train_activity_subject_validation.py`
- `EEG-Conformer/configs/subject_validation_first_round.json`
- `EEG-Conformer/run_activity_channelpreserve_five_models.py`
- `EEG-Conformer/train_activity_multidepth_weight_sweep.py`

本地与服务器以下四份文件的 SHA-256 完全相同：

| 文件 | SHA-256 |
| --- | --- |
| `train_activity_subject_validation.py` | `f11382312a42a0cd9ad4cb1db13dd46d2bccb20026ea2e0a1d5e6f28f7351ef9` |
| `configs/subject_validation_first_round.json` | `a387fb96987fc6d09f8e4a2cf5b16fb92a46787baf9933a1f6d7f0b5efc2e375` |
| `train_activity_loso.py` | `cb62062ed15fb16c57c7c3879e60208b50f3e7732ec4cb0159cf4b703e917d8d` |
| `comparison_models.py` | `7717652a2bc508c8695dc2c3949882ab989ea618ec76d4a411a3378df75d5c34` |

现有外部模型实现为 EEGNet、ShallowConvNet、ATCNet、TCFormer，来源记录在 `comparison_models.py`。受试者验证入口还支持基础时域 Conformer、时域+FFT、MDTF 和若干受控消融。其 12 个 preset 不是 12 种独立论文架构。

当前受试者验证入口只接受学习率和类别损失权重候选；架构由 preset 固定。拓展到新的论文模型、架构特定搜索和跨容器队列需要额外实现。该入口要求全新输出目录，没有现成的通用多 GPU 队列或整轮断点续训接口。

## 已确认的数据版本

以下人数和类别数通过读取小型 `subject_ids.npy` / `y.npy` 与 metadata 核对；`X.npy` 只读取 NPY 头部形状，没有加载大型信号数组。

根路径：`/public/home/gd_110413/SWT/work/eeg/local_artifacts/data_to_list/`。

| 数据目录（均取 `window_15_stride_3`） | 真实受试者 | 记录数 | 窗口数 | 类别 0/1/2 窗口数 |
| --- | --- | ---: | ---: | --- |
| `global_activity_dataset/` | S1–S11，11 人 | 33 | 5765 | 1056 / 1122 / 3587 |
| `global_activity_dataset_1to17_S17_first3_exclude_S6/` | S1–S17 除 S6，16 人 | 48 | 8395 | 1536 / 1632 / 5227 |
| `global_activity_dataset_1to17_S17_first3_exclude_S6_S17/` | S1–S16 除 S6，15 人 | 45 | 7869 | 1440 / 1530 / 4899 |
| `global_activity_failurecase_S6/` | S6，1 人 | 3 | 526 | 96 / 102 / 328 |
| `global_activity_failurecase_S17/` | S17，1 人 | 3 | 526 | 96 / 102 / 328 |

主数据形状均为 `[窗口数, 21, 1920]`，float64；采样率 128 Hz，15 秒窗口、3 秒步长，相邻窗口重叠 80%。旧 11 人数据还保留六种其他窗口/步长配置；扩展队列现存目录只核验到上述 15 秒/3 秒版本。

实际标签为三个指定视频刺激条件：`e_1→0, e_2→1, e_3→2`。固定视频次序和不等时长不能自动解释为经独立测量的认知状态标签。

### 原始数据和预处理

服务器保存全部 17 份 `data/<id>_e.edf` 和 `data/time_data.txt`。`list_cut_fixed_duration/` 与 `list_normalization_fixed_duration/` 均有 S1–S17 的目录。原始、中间与归一化数据可作为重建新实验版本的基础，当前未重建。

已确认的历史预处理流程：EDF 转 Excel → 按固定起点和时长裁切前三段 → 每条视频记录各自逐通道 `StandardScaler.fit_transform` → 切窗。当前 EDF 主线没有执行带通、陷波、伪迹清理或重新参考。来源是现有已确认决策文档与 `6.edf_list_normalization_fixed_duration.py:70` 的实际实现；并非本次新增设定。

所有窗口的输入通道顺序固定为 Fp1/Fp2/F3/F4/C3/C4/P3/P4/O1/O2/F7/F8/T3/T4/T5/T6/M1/M2/Fz/Cz/Pz，额外 A1/A2 不进入窗口。已有决策文档记录 S5/S6 的 EEG 通道名称后缀为 `-AV`，其他人为 `-REF`；不能只去掉名称后缀就宣称参考方式已统一，也不能据此单独判定 S6 异常。

整条测试视频记录参与自身的均值/方差估计，属于记录层的分布适配。后续在训练折内重新计算一个 scalar normalizer 不能撤销此前适配。是否保留这种离线整段可用的输入假设，或从原始/未归一化裁切信号重建训练侧归一化版本，是待决定的实验协议。

### 历史使用与独立性

- S12–S17 已在扩展 16 人五模型实验中完成全部正式折，不能重新称为未触碰的新外部测试人群。
- 16 人实验五模型共 80 折；服务器两个 job 的 `completed.json` 均为 `completed`，每模型汇总为 16 折。
- 排除 S6/S17 的 15 人实验也已完成：通道保留模型 15 折、四个外部模型各 15 折，共 75 折，两份完成标记均为 `completed`。
- 上述历史比较沿用每轮查看测试受试者、选择最高测试 accuracy 的开发协议，不能当作验证集选轮后的独立泛化估计。
- S6 的事后失败案例排除已有明确历史依据与用户决定；S17 在较晚版本也被单列。新的主实验使用 15 人、16 人还是包含全部人的敏感性分析，需要显式冻结，不能把这些队列的结果混为一组。

在 `local_artifacts/outputs` 的顶层目录名及深度不超过 5 的 `protocol_plan.json` 检查中，没有找到受试者嵌套验证入口对应的新完成结果。发现的五份计划属于旧 recall/resampling 开发实验。该结论只是“在此次检查范围内未发现”，不能证明其他自定义路径从未执行过新入口。

## 待决定事项和执行前缺口

1. 明确本轮主队列：15 人、16 人，还是另建完整队列与敏感性分析。已有全部受试者均有开发使用历史。
2. 明确严格训练侧归一化，或保留离线整记录适配协议；两种结果应分开命名。
3. 明确四张授权 GPU 的具体容器/UUID，以及已有 `school-eeg` GPU0 任务的处理安排。当前仅三张可见空闲卡。
4. 明确主指标、内外层划分、模型与超参数搜索预算，以及是否另设预训练模型赛道。
5. 新架构实现后先做小规模兼容性与耗时测量，再给出完整运行时间和可恢复的任务计划。

当前配置的预算量级：若 12 个 preset、15 个外层受试者、3 个 seed、3 个内层折、1 个候选，则是 540 次外层评估，包含 1620 次内层训练和 540 次重训，共 2160 次模型拟合；这不是时间估算。增加调参候选会增加内层训练数。现成 preset 中包含消融，不能将这个数量直接作为已覆盖 12 种论文模型的证据。

## 证据入口

- 本地 `新增数据实验_流程回顾与已确认决策.md`：独立受试者关系、EDF 主线、S6 排除及历史协议。
- 本地 `新增16人实验结果分析_seed42.md`：16 人 80 折结果、S6/S17 历史使用及实验局限。
- `EEG-Conformer/train_activity_subject_validation.py`：内层按受试者 Macro-F1 选轮、从头重训、最后一次外层预测。
- `EEG-Conformer/configs/subject_validation_first_round.json`：既有模型 preset、3 个 seed、3 个内层折及历史设定来源。
- 远程数据数组/metadata 与两组已完成 job 的 JSON：本次只读实测。
