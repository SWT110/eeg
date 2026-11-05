# 16模型正式探索性实验：执行记录

> 2026-11-05最新状态：16模型、240折、每折200轮全部完成并复核，当前阶段没有剩余训练。分析见 `analysis/architecture16_seed42_20261102/analysis_report.md`。本地未下载.pt，权重保留服务器。下文启动、四卡及两卡调整为历史执行记录。

日期：2026-11-02（America/New_York）。用户已回答全部方案问题并授权执行；最终最佳骨干由用户查看完整结果后选择。

## 已执行

- 核实PID30542为school-eeg上D版深度2/5/8的权重搜索后，按用户明确指令发送SIGTERM。已确认该进程退出、两张卡释放；既有结果与断点保留，没有删除产物。
- 九个新架构已实现并按原始论文/作者代码或明确的BSD参考逐项核对；非平凡适配记录于`EEG-Conformer/benchmark_vendor/sources.json`。
- 16模型在真实服务器、PyTorch1.12.1+cu113、NVIDIA A10上均以batch72、21通道、1920点通过两步训练、有限损失/梯度及eval检查。另记录同卡batch72的GPU前向计时，含3次warmup、10次重复，不含I/O；每窗口时间是batch均摊值，不是单窗口在线延迟。
- 新增7项测试在服务器全部通过，包括因果滤波SciPy数值一致性、TCN因果性、TSception中线/半球路径、真实尺寸梯度、单任务互斥及中断后相同训练轨迹恢复。本地原训练相关149项与原比较模型4项测试通过。
- 独立只读源快照已部署，正式运行不修改在线仓库或现有Python环境。原始数据不重建、不覆盖。
- 四个独立GPU worker已启动，已观察到四个首折均进入真实数据训练并保存多轮历史；完整240折仍在运行。

## 冻结条件

16模型：EEGNet、ShallowConvNet、DeepConvNet、EEG Conformer、FBCNet、EEG-TCNet、ATCNet、TCFormer、CTNet、EEGNeX、EEGSimpleConv、EEG-ITNet、TSception V2、EEG-Deformer、MDTF、ChannelPreserving。

15名受试者：S1–S16排除S6；S6/S17单列。正式数据7869窗口、21通道、128Hz、15秒窗/3秒步长，沿用完整视频记录逐通道标准化。

每模型15折LOSO、seed42、200epochs、batch72、Adam lr0.0002、betas0.5/0.999，无类别损失加权、无重采样、无增强。MDTF自身的分支融合/辅助损失仍是其架构的一部分；“无类别权重”不表示删除架构自带的分支机制。

按用户选择，保存外层测试accuracy最高的epoch，首次最高值赢得平局；测试集参与选轮，结果统一标记探索性。所有架构使用共同训练配方，此轮不是每个模型充分独立调参后的上限比较。

## 远程路径与指纹

```text
项目根目录：/public/home/gd_110413/SWT/work/eeg
Python：.conda-envs/eegconformer310/bin/python
输入：local_artifacts/data_to_list/global_activity_dataset_1to17_S17_first3_exclude_S6_S17/window_15_stride_3
代码快照：local_artifacts/benchmark_sources/architecture16_20261102_v1
运行目录：local_artifacts/outputs/architecture16_seed42_20261102
计划指纹：f9bcdc14fd3fd137696d74820ca4dd9a97c6e1d0cf6b3971563e9b2ebb5d959a
```

计划保存六个数据文件与全部训练源代码的SHA256。每个worker启动时核对数据；每个训练子进程核对冻结源码和小型索引。两容器共享存储，通过原子任务锁防重复执行。每个fold独立子进程，保存Adam状态和RNG状态，支持断点恢复。

| SSH | 局部GPU | GPU UUID | 初始worker PID |
|---|---:|---|---:|
| school-eeg | 0 | GPU-f2e48d32-e06f-0d8e-1daf-1b308ca64783 | 13352 |
| school-eeg | 1 | GPU-6c454bdd-a90d-89fd-11cd-d3ebd9cc1fd3 | 13357 |
| school-eeg-2 | 0 | GPU-100cf18b-1173-1982-d644-96da8c1b954e | 35456 |
| school-eeg-2 | 1 | GPU-4dfadde7-eb76-8d4c-126c-25f402950e54 | 35468 |

school-eeg-2的GPU2/3现有任务未干预。worker PID可能在恢复时变化，以运行目录`workers/`记录为准。

## 状态与产物

在任一SSH入口执行：

```bash
cd /public/home/gd_110413/SWT/work/eeg
CUDA_VISIBLE_DEVICES='' .conda-envs/eegconformer310/bin/python \
  local_artifacts/benchmark_sources/architecture16_20261102_v1/EEG-Conformer/run_architecture_benchmark.py \
  status --run-root local_artifacts/outputs/architecture16_seed42_20261102
```

- `plan.json`：固定任务、数据和代码指纹、模型配置与局限。
- `progress.json` / `model_summary.csv`：自动更新的完整性、逐模型指标；未完成15折的行只是部分结果，不用于先行排名。
- `tasks/<model>__S<id>/`：独立进程日志、任务锁/heartbeat、成功或失败标记、逐折复算指标。
- `results/<model>/seed_42/fold_subject_<id>/`：最佳权重、逐epoch历史、训练日志、真实标签与预测。只有训练完200epoch并通过标签/指标核对后才有任务完成标记。
- `workers/`：GPU分配、进程和启动日志。

任务锁不按超时自动回收：两个容器和客户端的时钟存在差异，孤儿训练子进程也可能仍在运行。恢复时先在对应SSH确认旧worker和子进程均退出，再处理孤立锁；不通过直接删除活锁让两份训练同时写同一fold。

本地镜像入口：`analysis/architecture16_seed42_20261102/`，包含execution.json、plan.json、preflight.json；结果报告完成时在该目录补充CSV、逐折数据与图表。

小型产物收集/报告命令已用首批4个真实完成折运行验证：

```bash
local_artifacts/.venv-benchmark-reporting/bin/python \
  scripts/collect_architecture_benchmark.py --fetch --partial
```

未满240折时只生成`progress_report.md`与已核验逐折CSV；全部完成时生成`final_report.md`、`model_summary_audited.csv`和三张比较图。旧训练JSON把逐类F1和Macro-F1舍入到六位，收集脚本对此使用合理舍入容差，同时与完成标记的全精度预测重算结果严格比对。已完成4折的标签、200轮历史与预测复算通过。

## 后续自动推进

本聊天已建立每30分钟的检查（automation id `eeg-16`）。正常运行保持安静，发生实质失败、需要用户处理或完整完成时再报告。240折全部完成后核对完整性、拉回小型指标/预测/历史产物，整理中文比较报告；不自动选择最佳骨干。没有将GPU训练已启动表述为全部实验完成。

## 最新资源调整：school-eeg-2暂停

2026-11-02用户明确要求将school-eeg-2的GPU0/1让给老师。已停止该主机两个benchmark worker及其当前训练子进程，确认GPU0/1显存均为0MiB、利用率0%；GPU2/3其他任务未干预。school-eeg的GPU0/1继续训练。已有结果及可用断点保留，中断任务在确认旧进程退出后回收到其余两卡队列。资源策略保存于远程run的resource_policy.json及workers/pause_school_eeg_2.json；监控指令已更新，未经用户明确要求不会自动恢复school-eeg-2。冻结模型、数据、训练参数及计划指纹不变。

## 分析交付完成（2026-11-05）

已通过共享存储只读入口拉齐非权重产物，240折、48,000条epoch与125,904次方法预测均复核通过。本地received共1204个文件、18,637,949字节，.pt文件数为0；240份最佳权重保留服务器。完整分析见`analysis/architecture16_seed42_20261102/analysis_report.md`，包括类别短板、逐人配对差异、参数/显存/耗时和训练轨迹。未启动新的调参、架构优化或GPU任务。完成监控eeg-16已删除。
