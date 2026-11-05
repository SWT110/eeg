# EEG模型扩展比较：文献、方案与环境索引

研究始于2026-11-01，2026-11-02方案确认并启动训练；2026-11-05已核验240折全部完成，分析报告已生成（America/New_York）。最终骨干由用户选择。

完整结果保存在本机 `analysis/architecture16_seed42_20261102/analysis_report.md`。用户选择让 `analysis/` 留在本地，该目录不通过 GitHub 同步；仅拉取指标、预测、逐epoch历史与配置，未下载.pt。代码三端同步见 [SYNC.md](../../SYNC.md)。

## 已整理的材料

| 文档 | 内容 |
|---|---|
| [实验方案与决策树](experiment_plan.md) | 用户已经确认与尚未确认的选择、探索性评估口径、预算计算及执行边界 |
| [监督模型论文清单](papers_supervised.md) | 14个架构、13篇论文，精确题名/发表信息、作者代码、输入适配和实现状态 |
| [预训练与图模型](papers_transfer_and_graph.md) | 8项补充候选、checkpoint声明、输入条件、方法差异和未完成核验 |
| [服务器只读审计](server_audit.md) | 两容器共享存储、GPU UUID与瞬时占用、环境、数据版本、历史结果和运行缺口 |
| [执行记录](execution_status.md) | 已启动的16模型240折四卡队列、冻结指纹、检查结果、状态命令和后续检查 |

## 候选覆盖

| 路线 | 候选 |
|---|---|
| 紧凑与深层CNN | EEGNet、ShallowConvNet、DeepConvNet、EEGNeX、EEGSimpleConv |
| 频带与时间卷积 | FBCNet、EEG-TCNet、ATCNet、EEG-ITNet |
| CNN与Transformer | EEG Conformer、CTNet、TCFormer、EEG-Deformer |
| 时间尺度与半球空间结构 | TSception V2 |
| 外部预训练/自监督 | BENDR、BIOT、LaBraM、EEGPT、CBraMod、EEG2Rep |
| 图与视频识别 | RGNN、ICASSP2018视频识别图卷积方法（研究文档简称Video-GCNN） |

总计14种监督架构与8项补充模型/方法。某些自监督方法的主要差异在预训练目标，不能仅按名称数当作完全不同的独立骨干。Shallow/DeepConvNet共享一篇论文。项目已有MDTF、ChannelPreserving属于额外项目参照，不计入外部论文数量。

## 已确认的本轮条件

- 现有三类视频刺激条件，跨受试者预测。
- S1–S16排除S6，共15人；S6/S17单列。
- 128Hz、21通道、整段逐通道标准化、15秒窗口/3秒步长；7869个正式窗口。
- 每个模型按旧外层测试accuracy最高轮保存权重，结果统一标记为探索性开发比较。
- 用户在实验后查看完整指标，自行选择最佳骨干；本轮不自动裁定胜出者。

## 已确认并实施

- Q6：14种监督架构+2种项目参照，共16种，从头训练。
- Q7：seed42、统一历史训练配方，每模型全部15折，共240折。
- Q8：按用户指令停止school-eeg旧权重搜索，使用两容器各GPU0/1四卡。

模型接入、兼容性/小批训练验证已完成，正式队列已启动。原训练环境和在线仓库未改动，使用独立只读代码快照与新输出目录。详见执行记录。

文献初筛的核验范围为出版元数据、摘要/部分方法段及官方实现说明；正式接入的九个新架构另完成官方源码/独立BSD参考的逐项结构核对与数值/训练验证。详见两份implementation_audit文档及benchmark_vendor/sources.json。适配实现不声称完整复现作者所有数据处理/训练配方，也不使用不同公开数据集上的论文自报准确率代替本项目结果。
