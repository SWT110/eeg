# EEG 监督学习模型架构候选：论文与复现来源

检索日期：2026-11-01（America/New_York），2026-11-02继续讨论。状态：文献初筛，正式实验名单和运行规模待确认；尚未新增模型实现或运行训练。本轮已由用户明确选择15人队列、现有15秒/3秒输入和旧最高测试准确率选轮，结果作为探索性实验；下文关于独立验证的建议不代表本轮已切换协议。

本表覆盖 **14 个架构、13 篇论文**，包括仓库已有外部基线。ShallowConvNet 与 DeepConvNet 出自同一篇论文。选择依据是结构代表性、公开实现和本项目的适配可行性，不以不同公开数据集的论文准确率排序。

## 检索与证据范围

- 已实际调用 Paper Search MCP：`search_crossref` 检索 EEGNet、Deep/ShallowConvNet、FBCNet、EEG-TCNet；这些自由文本查询有偏题结果，不能直接采纳为模型证据。
- 已通过 Paper Search MCP 的 `get_crossref_paper_by_doi` 核验 EEGNet、Deep/ShallowConvNet、EEG Conformer、ATCNet、TCFormer、CTNet、EEGNeX、EEGSimpleConv、EEG-ITNet、TSception V2、EEG-Deformer 的精确题名、DOI 和卷期信息。
- `search_arxiv` 曾对 `ti:EEGNet`、`ti:FBCNet`、`ti:EEGTCNet`、`ti:EEGDeformer` 发起检索；本轮等待未得到可用返回后终止该批调用。不能把这部分记为成功检索证据。
- 用作者官方仓库、作者机构的论文页面、出版方论文及 arXiv 原文页面补充核验。下文来源链接均指向这些一手来源。
- 此次是摘要、出版元数据和实现说明层面的架构初筛；尚未逐篇完成全文审计或逐层复现核对，也未验证新模型在本项目上的性能。

## 候选架构

以下“适配价值”和“风险/工作量”是依据结构与本项目输入条件作出的工程判断，**不是在本项目上验证的实验结果**。输入条件暂按项目现有的 21 通道时域窗口、视频刺激条件三分类理解；窗口、采样率和队列由最终实验协议固定。

| 架构 | 发表年份/状态 | 核心结构 | 本项目适配价值 | 适配边界或工作量 | 论文与作者代码 |
| --- | --- | --- | --- | --- | --- |
| EEGNet | 2018，JNE；预印本 2016 | 时间卷积、深度可分离空间卷积、可分离卷积 | 小容量基线，判断复杂模型增加的收益 | 需核对版本、滤波器数和时间核的实际毫秒尺度；作者原实现为 Keras，仓库已存在 PyTorch 适配版 | [P01](https://doi.org/10.1088/1741-2552/aace8c)，[官方实现](https://github.com/vlawhern/arl-eegmodels) |
| ShallowConvNet | 2017，Human Brain Mapping | 时间/空间卷积、平方、池化与对数，提取功率相关模式 | 对照频谱功率归纳偏置；仓库已有实现 | 需保持激活、池化和对数裁剪等结构细节；核对与作者实现的差异 | [P02](https://doi.org/10.1002/hbm.23730)，[论文团队代码](https://github.com/braindecode/braindecode) |
| DeepConvNet | 2017，Human Brain Mapping | 多层时空卷积与逐层池化 | 补足纯深层 CNN 对照，避免只比较轻量 CNN 与 Transformer | 21 通道和长窗口需重算最终特征尺寸；较大容量的收益待实际验证 | [P02](https://doi.org/10.1002/hbm.23730)，[论文团队代码](https://github.com/braindecode/braindecode) |
| EEG Conformer | 2023，TNSRE | 局部时空卷积、池化 token、Transformer | 与现有 Conformer/MDTF 开发路线直接衔接 | 原论文模型、项目适配版和 MDTF 改进模型须分开命名；固定输入长度时核对 token 数及分类头 | [P03](https://doi.org/10.1109/TNSRE.2022.3230250)，[官方实现](https://github.com/eeyhsong/EEG-Conformer) |
| FBCNet | 2021，arXiv 本文版本 | 窄频带滤波器组、各频带空间卷积、时间方差层 | 检验显式频带分解与功率特征是否适合本任务 | 滤波器组属于输入变换/模型流程的一部分；频带需符合原始信号有效带宽与 Nyquist 限制，不能把它当作仅替换分类骨干 | [P04](https://arxiv.org/abs/2104.01233)，[官方 PyTorch 实现](https://github.com/ravikiran-mane/FBCNet) |
| EEG-TCNet | 2020，IEEE SMC | 紧凑 EEGNet 式前端、扩张因果 TCN | 检验低成本时间建模，对长时间窗口有结构代表性 | 原始任务为 MI；长窗口的有效感受野需核对。作者代码与现有 PyTorch 训练器的接口需适配 | [P05](https://doi.org/10.1109/SMC42975.2020.9283028)，[官方实现](https://github.com/iis-eth-zurich/eeg-tcnet) |
| ATCNet | 2023 卷期，2022 early access，TII | 卷积前端、滑动窗口、注意力、TCN | 仓库已有；补足注意力与 TCN 混合路线 | 作者原实现与项目移植版要区分；模型内部窗口与数据集外部切窗是两层不同配置 | [P06](https://doi.org/10.1109/TII.2022.3197419)，[官方实现](https://github.com/Altaheri/EEG-ATCNet) |
| TCFormer | 2025，Scientific Reports | 多卷积核前端、分组查询注意力 Transformer、TCN 分类头 | 仓库已有的较新混合结构，可作为强候选 | 多尺度卷积、注意力组数和 TCN 感受野需记录；论文 MI 结果不能外推到视频刺激条件 | [P07](https://doi.org/10.1038/s41598-025-16219-7)，[官方实现](https://github.com/Altaheri/TCFormer) |
| CTNet | 2024，Scientific Reports | EEGNet 式卷积前端、Transformer、分类头 | 与 EEG Conformer 形成另一种 CNN–Transformer 设计对照 | 与 Conformer 的结构相似性应说明；原文重组增强不能只对 CTNet 开启后声称仅比较骨干 | [P08](https://doi.org/10.1038/s41598-024-71118-7)，[论文指定实现](https://github.com/snailpt/CTNet) |
| EEGNeX | 2024 卷期，2023 online；预印本 2022，BSPC | EEGNet 启发的纯 CNN，扩展卷积与扩张时间卷积设计 | 较强纯 CNN 候选，验证复杂注意力模块是否带来稳定收益 | 作者仓库是 Keras 风格实现；PyTorch 移植需核对 padding、扩张率、归一化与约束 | [P09](https://doi.org/10.1016/j.bspc.2023.105475)，[官方实现](https://github.com/chenxiachan/EEGNeX) |
| EEGSimpleConv | 2024，TNSRE；预印本 2023 | 标准 1D 卷积组成的简单骨干 | 实现简洁，可作为效率与跨受试者学习对照 | 作者实验包含重采样/训练规则，应审计输入采样率和训练配方；不能只照抄公开分数 | [P10](https://doi.org/10.1109/TNSRE.2024.3451010)，[官方 PyTorch 实现](https://github.com/elouayas/EEGSimpleConv) |
| EEG-ITNet | 2022，IEEE Access | Inception 多尺度卷积、扩张因果时间卷积 | 检验多尺度频谱与时间建模，代表性区别于单尺度前端 | 作者 TensorFlow/Notebook 实现；需接口适配及感受野核对。应在引入代码前检查明确授权条件 | [P11](https://doi.org/10.1109/ACCESS.2022.3161489)，[官方实现](https://github.com/AbbasSalami/EEG-ITNet) |
| TSception V2 | 2023 卷期，2022 early access；预印本 2021，TAFFC | 多尺度时间卷积、全局/左右半球空间卷积、融合 | 原任务来自视频刺激相关情绪 EEG，补充 MI 之外的设计来源 | 需真实电极位置、左右半球顺序及中线通道处理，21 通道不能直接套用 DEAP 的 28 通道顺序。原任务标签也不等于本项目标签 | [P12](https://doi.org/10.1109/TAFFC.2022.3169001)，[官方 PyTorch V2 实现](https://github.com/yi-ding-cs/TSception) |
| EEG-Deformer | 2025 卷期，2024 online，JBHI | 浅层卷积、分层粗到细 Transformer、密集信息净化 DIP | 论文覆盖注意、疲劳与心理负荷，补充接近持续状态/认知任务的结构来源 | 仍须用本项目刺激条件标签重新训练与验证；核对各层时间缩减、DIP 汇聚与最终特征尺寸 | [P13](https://doi.org/10.1109/JBHI.2024.3504604)，[官方 PyTorch 实现](https://github.com/yi-ding-cs/EEG-Deformer) |

## 本地实现状态与接入建议

在本次阅读的 `EEG-Conformer/comparison_models.py` 中，已存在 `eegnet`、`shallowconvnet`、`atcnet`、`tcformer` 四个外部模型。这些是基于作者 TCFormer 仓库特定提交的项目适配实现，不能称为从各论文作者原始仓库原封不动运行。其来源、提交和许可信息应随新实验保存。

EEG Conformer 的项目模型与训练主入口也已存在。MDTF 是项目改进模型，不占用文献中一个新的外部架构名额。

- **已有四个外部适配实现**：EEGNet、ShallowConvNet、ATCNet、TCFormer。
- **已有基础 Conformer 项目实现**：EEG Conformer；仍应核对任务适配与原文配置的差异。
- **待接入并验证的九个架构**：DeepConvNet、FBCNet、EEG-TCNet、CTNet、EEGNeX、EEGSimpleConv、EEG-ITNet、TSception V2、EEG-Deformer。

建议把正式队列中的候选按结构覆盖与接入代价预先选定，以下仅是工程优先级，尚非用户决定：

1. 保留已有五种基础架构，形成结果与历史实验的联系。
2. 优先补充 DeepConvNet、EEGSimpleConv、EEGNeX，增强纯 CNN 对照；优先评估 EEG-Deformer 的适配成本。
3. 在确认电极布局后加入 TSception V2。没有可信电极映射时，应明确暂缓，而不是随意把通道排列解释成左右半球。
4. FBCNet、EEG-TCNet、EEG-ITNet、CTNet 可补足滤波器组、TCN、多尺度和另一种 Transformer 前端的结构覆盖；若全部纳入，正式运行前冻结名单与调参范围。

选定新模型后，适配阶段只验证接口、参数量、输出有限性、梯度与小批次训练是否正常。正式排名必须来自统一数据与验证协议；本轮没有开展这些实现验证。

## 实验设计应保留的区分

- **相同可用信号与数据划分**：模型都使用同一批原始记录、同一有效通道和时间范围。独立泛化估计要求测试受试者不参与 epoch 或超参数选择；用户已明确本轮继续旧测试选轮开发协议，因此结果只作探索性架构比较，不称独立测试估计。
- **结构自带变换**：FBCNet 滤波器组、模型内部窗口、多尺度卷积等要完整记录。若额外添加 FFT/时频图/图结构，则属于明确的流程差异。
- **训练配方与结构**：先比较相同预算下的可调模型；对选中骨干做模块消融时固定其余训练配方。公开实现中的重采样、重组增强、max-norm 与早停规则不能遗漏。
- **输入适配与模型改造**：把输出类别改成 3、按实际通道数设置空间卷积是任务适配；新增分支、改变归纳偏置、修改关键感受野则应记为模型变体。
- **论文分数与项目分数**：MI、情绪、注意任务的数据和评估不同，论文准确率不参与本项目模型排名。

## 精确论文目录

- **P01 — EEGNet**：Lawhern et al. *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*. Journal of Neural Engineering 15(5), 056013, 2018. [DOI](https://doi.org/10.1088/1741-2552/aace8c)，[作者预印本](https://arxiv.org/abs/1611.08024)。
- **P02 — ShallowConvNet / DeepConvNet**：Schirrmeister et al. *Deep learning with convolutional neural networks for EEG decoding and visualization*. Human Brain Mapping 38(11), 5391–5420, 2017. [DOI](https://doi.org/10.1002/hbm.23730)，[作者预印本](https://arxiv.org/abs/1703.05051)。这不是同年题名含 “EEG pathology” 的另一篇会议论文。
- **P03 — EEG Conformer**：Song et al. *EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization*. IEEE TNSRE 31, 710–719, 2023. [DOI](https://doi.org/10.1109/TNSRE.2022.3230250)，[作者实现与说明](https://github.com/eeyhsong/EEG-Conformer)。
- **P04 — FBCNet**：Mane et al. *FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface*. arXiv:2104.01233, 2021. [作者预印本](https://arxiv.org/abs/2104.01233)，[作者实现](https://github.com/ravikiran-mane/FBCNet)。此条仅确认该预印本/实现版本，不冒称其为期刊论文；较早同族会议工作亦不应混为此版本。
- **P05 — EEG-TCNet**：Ingolfsson et al. *EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain-Machine Interfaces*. IEEE SMC, 2958–2965, 2020. [DOI](https://doi.org/10.1109/SMC42975.2020.9283028)，[作者机构存档](https://cris.unibo.it/handle/11585/800231)，[作者预印本](https://arxiv.org/abs/2006.00622)。
- **P06 — ATCNet**：Altaheri, Muhammad, Alsulaiman. *Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification*. IEEE Transactions on Industrial Informatics 19(2), 2249–2258, 2023（2022 early access）. [DOI](https://doi.org/10.1109/TII.2022.3197419)，[作者实现](https://github.com/Altaheri/EEG-ATCNet)。
- **P07 — TCFormer**：Altaheri, Karray, Karimi. *Temporal convolutional transformer for EEG based motor imagery decoding*. Scientific Reports 15, 32959, 2025. [出版方全文](https://www.nature.com/articles/s41598-025-16219-7)，[作者实现](https://github.com/Altaheri/TCFormer)。
- **P08 — CTNet**：Zhao et al. *CTNet: a convolutional transformer network for EEG-based motor imagery classification*. Scientific Reports 14, 20237, 2024. [出版方全文](https://www.nature.com/articles/s41598-024-71118-7)，[论文指定实现](https://github.com/snailpt/CTNet)。
- **P09 — EEGNeX**：Chen et al. *Toward reliable signals decoding for electroencephalogram: A benchmark study to EEGNeX*. Biomedical Signal Processing and Control 87, 105475, 2024（2023 online）. [DOI](https://doi.org/10.1016/j.bspc.2023.105475)，[作者论文页](https://chenxiachan.github.io/publication/2022-07-15-EEGNeX)，[作者预印本](https://arxiv.org/abs/2207.12369)。
- **P10 — EEGSimpleConv**：El Ouahidi et al. *A Strong and Simple Deep Learning Baseline for BCI Motor Imagery Decoding*. IEEE TNSRE 32, 3338–3347, 2024. [DOI](https://doi.org/10.1109/TNSRE.2024.3451010)，[作者预印本](https://arxiv.org/abs/2309.07159)，[作者实现](https://github.com/elouayas/EEGSimpleConv)。仓库中的 2023 引用指预印本。
- **P11 — EEG-ITNet**：Salami, Andreu-Perez, Gillmeister. *EEG-ITNet: An Explainable Inception Temporal Convolutional Network for Motor Imagery Classification*. IEEE Access 10, 36672–36685, 2022. [DOI](https://doi.org/10.1109/ACCESS.2022.3161489)，[出版方摘要](https://ieeexplore.ieee.org/document/9739771/)，[作者预印本](https://arxiv.org/abs/2204.06947)。
- **P12 — TSception V2**：Ding et al. *TSception: Capturing Temporal Dynamics and Spatial Asymmetry From EEG for Emotion Recognition*. IEEE Transactions on Affective Computing 14(3), 2238–2250, 2023（2022 early access）. [DOI](https://doi.org/10.1109/TAFFC.2022.3169001)，[作者预印本](https://arxiv.org/abs/2104.02935)，[作者 V2 实现](https://github.com/yi-ding-cs/TSception)。区分 [IJCNN 2020 旧版](https://github.com/deepBrains/TSception)。
- **P13 — EEG-Deformer**：Ding et al. *EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces*. IEEE Journal of Biomedical and Health Informatics 29(3), 1909–1918, 2025（2024 online）. [DOI](https://doi.org/10.1109/JBHI.2024.3504604)，[作者预印本](https://arxiv.org/abs/2405.00719)，[作者实现](https://github.com/yi-ding-cs/EEG-Deformer)。作者 README 标注 2024，Crossref 卷期记录为 2025-03，二者指不同出版阶段。

## 尚未决定

跨受试者目标已由用户确认。用户将审阅完整比较结果，再决定最佳骨干；本文件不预设自动裁定胜者的评分公式。

正式候选数量、是否全部从零训练、训练规模和GPU分配仍待讨论后冻结。队列、现有输入、旧测试选轮协议已确认，详见同目录experiment_plan.md。哪些结构进入后续优化，由用户看实验数据后决定。本文件未启动任何 GPU 任务。
