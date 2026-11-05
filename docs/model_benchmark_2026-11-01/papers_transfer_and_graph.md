# EEG 补充候选：预训练模型、图网络与视频识别

研究日期：2026-11-01（America/New_York）。本文件是文献与实现可行性调查，尚未运行模型；不构成最终模型名单、实验协议或性能排名。用户已明确继续研究跨受试者视频刺激条件分类，最终模型由用户结合完整结果选择。

## 检索与核验范围

- 已实际调用 Paper Search MCP：`search_papers`，查询 BENDR、BIOT、LaBraM、EEGPT、CBraMod、EEG2Rep、RGNN、DGCNN，来源指定 `arxiv,crossref`。聚合调用长时间未完成后停止等待，不把它记作检索成功。
- 独立调用 `search_arxiv(query='ti:"CBraMod"', max_results=1)` 返回 HTTP 429，工具注明重试三次仍受限。
- 独立调用 `search_crossref` 查询 RGNN，首条返回的是 2014 年另一篇极限学习机论文，已因题名/作者不匹配而剔除。说明学术 MCP 的模糊结果仍需逐篇校验。
- 以下事实用实际打开的会议正式发表页、作者原论文、作者官方仓库交叉核验；没有使用综述的性能表作排名。核验深度在每项注明；本轮没有下载 checkpoint、安装依赖或运行训练，也不声称完成逐篇全文精读。

## 八项候选索引

| 候选 | 精确论文题名与发表状态 | 核心结构/本轮核验深度 | 官方实现与权重情况 | 建议用途 |
|---|---|---|---|---|
| BENDR | **BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data**，Frontiers in Human Neuroscience，2021；DOI `10.3389/fnhum.2021.653659`。[发表原文](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full) | 卷积压缩器、Transformer 上下文模型、对比自监督；核查原文方法相关段落与官方 README | [SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR)；README 指向 [v0.1-alpha 权重发布页](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha)，本轮未下载 | 较早的自监督迁移参照，优先级低于较轻且输入更灵活的模型 |
| BIOT | **BIOT: Biosignal Transformer for Cross-data Learning in the Wild**，NeurIPS 2023 正式发表。[正式发表页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract.html) | 通道分别进行频谱 token 化，加入通道与位置表示，再用 Transformer 编码；核查会议摘要、官方 README | [ycq091044/BIOT](https://github.com/ycq091044/BIOT)；提供多个命名明确的 EEG checkpoint 和从头训练/迁移示例 | 可同时提供监督从头训练与外部预训练两种模式，分开报告 |
| LaBraM | **Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI**，ICLR 2024 正式发表、Spotlight。[会议原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)；[作者提交页](https://openreview.net/forum?id=QzTpTRVtrP) | EEG 通道 patch、向量量化神经频谱 tokenizer、掩码神经码预测；核查会议摘要、官方 README | [935963004/LaBraM](https://github.com/935963004/LaBraM)；README 明确要求加载预训练 checkpoint，并示例 `labram-base.pth`；权重下载文件入口需在实施前复核 | 外部预训练迁移候选，优先尝试 base 版本；不计划用本项目小队列从头重做大型预训练 |
| EEGPT | **EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals**，NeurIPS 2024 正式发表。[会议原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf) | 掩码自监督、时空表征对齐与重建，分层时空处理；核查会议摘要、官方 README | [BINE022/EEGPT](https://github.com/BINE022/EEGPT)；README 提供 Figshare 权重入口、模型文件名与分类适配路径；Figshare 本轮访问返回 403，未完成下载验证 | 优先列入迁移可行性测试；需解决固定预训练窗口与本项目长窗口的对应 |
| CBraMod | **CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding**，ICLR 2025 正式发表。[会议发表页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html) | 空间/时间分离的 criss-cross 注意力、条件位置编码、掩码 patch 重建；核查会议摘要、官方 README 与示例输入 | [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod)；官方 README 链接 [weighting666/CBraMod 权重](https://huggingface.co/weighting666/CBraMod)，并给出自定义分类器示例 | 迁移赛道优先候选；也可抽取其分离时空注意力思路供后续架构优化 |
| EEG2Rep | **EEG2Rep: Enhancing Self-supervised EEG Representation Through Informative Masked Inputs**，KDD 2024 正式发表。[作者原文](https://arxiv.org/abs/2402.17772) | 在潜在表征空间预测掩码片段，使用 semantic subsequence preserving 掩码；核查作者摘要和官方 README | [Navidfoumani/EEG2Rep](https://github.com/Navidfoumani/EEG2Rep)；提供预训练与任务代码；本轮 README 未找到通用预训练 checkpoint 下载入口，不能据此断言权重不存在 | 适合独立研究“仅训练受试者的无标签预训练是否有用”；不自动纳入最低成本第一轮 |
| RGNN | **EEG-Based Emotion Recognition Using Regularized Graph Neural Networks**，IEEE Transactions on Affective Computing，2020 online（作者 repo 的引用年份）；arXiv 初稿 2019。[作者原文](https://arxiv.org/html/1907.07835v4) | 具有神经拓扑先验的图网络、NodeDAT 域对抗与 EmotionDL；核查原文相关方法/输入段和作者代码声明 | [zhongpeixiang/RGNN](https://github.com/zhongpeixiang/RGNN)；官方只提供简化模型示例，明确未包含 EmotionDL；无通用权重下载入口被本轮核实 | 可作为图建模补充；完整方法移植成本、目标域访问和软标签语义均需审查 |
| Video-GCNN（本文工作标识） | **EEG-based video identification using graph signal modeling and graph convolutional neural network**，ICASSP 2018，DOI `10.1109/ICASSP.2018.8462207`。[作者原文](https://arxiv.org/html/1809.04229v1)；[作者实验室发表清单](https://mcml.yonsei.ac.kr/publications) | 分频带功率/熵特征，空间或相关图，跨频带连边，Chebyshev 图卷积；核查原文方法和实验设计段 | 本轮未找到作者官方实现/预训练权重，不能把第三方代码当官方复现；“Video-GCNN”仅为本文件简称 | 任务最接近“观看哪段视频”的文献参考；若纳入实验需自实现且另行标识输入特征差异 |

## 输入和实验协议的实际障碍

1. **BENDR**：原文按约 256 Hz 对齐，使用共同电极布局并引入振幅相关表示。应核对官方 encoder 的通道映射与输入缩放，不能直接把现有 21 通道数组按任意顺序塞入已训练权重。[原文预处理方法](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full)
2. **BIOT**：官方预训练模型统一到 200 Hz；公开 checkpoint 有 16 通道双极导联配置及扩展的 18 通道配置。架构支持不同通道/长度，不代表任意通道名和参考方式已与 checkpoint 对齐。需登记通道 embedding 的映射/新增方案，且把 vanilla 与 pretrained 结果分别标注。[官方输入与 checkpoint 说明](https://github.com/ycq091044/BIOT)
3. **LaBraM**：官方数据处理建议 200 Hz、单位 µV、0.1–75 Hz 带通和 50 Hz 陷波，且强调输入通道顺序。现有逐整段归一化后的 `X.npy` 是否还能对应这些幅度约定需核对。不能只改采样率就声称完整复现。官方从头训练 tokenizer 的建议配置为 8 张 RTX 3090 或更好，因此本轮四 GPU 方案宜加载 base checkpoint 迁移，具体可运行规模须由单批显存测试决定。[官方数据与训练说明](https://github.com/935963004/LaBraM)
4. **EEGPT**：官方大模型 checkpoint 的预训练输入是 58 通道、256 Hz、4 秒、64 点 patch。支持通道适配不等于支持任意长窗口。对现有长窗口，必须先确定保持完整覆盖的分块/表征汇总策略或可验证的时间位置适配；这一适配也属于方法的一部分。[官方权重和输入说明](https://github.com/BINE022/EEGPT)
5. **CBraMod**：官方示例输入 `[batch, channels, segments, 200]`，示例为 `[8,22,4,200]`；分类头按通道数和段数构造。示例维度并不要求所有任务固定 22 通道。实施时需明确 200 点 patch 与重采样/秒数的对应，换成三分类头并核对长窗口显存；这些需要实测，尚未验证本项目可用。[官方 quick start](https://github.com/wjq-learning/CBraMod)
6. **EEG2Rep**：适配可保留项目原始通道与窗口语义，但自监督阶段也必须在每个外层折内部进行。若用到了留出受试者的无标签记录，该结果须标明允许目标域无标签数据的适配，不能与完全不接触新受试者的协议混算。[作者论文](https://arxiv.org/abs/2402.17772)、[官方实现](https://github.com/Navidfoumani/EEG2Rep)
7. **RGNN**：原文在 SEED 上使用预计算并经过 LDS 平滑的 DE 特征；NodeDAT 方法明确建立源/目标域的域分类器。重做跨受试者实验要先核对是否访问留出人的无标签数据。另有 EmotionDL 对情绪标签相似性的假设，本项目视频标签不自动继承此关系。可选“图骨干 + 统一监督训练”或单设域适配试验，但应准确命名修改版。[原文方法](https://arxiv.org/html/1907.07835v4)、[官方简化实现声明](https://github.com/zhongpeixiang/RGNN)
8. **Video-GCNN**：原文用 DEAP 的 32 通道、128 Hz 信号，三秒窗口、一步一秒；用八频带功率/熵形成图节点特征。原文描述将信号片段 80% 用于训练、其余测试，未给出本文所需的受试者完全分离证明。它的论文分数不能当成本项目 LOSO 的可比目标。架构/图构造可参考，新的实验必须按本项目统一受试者分组重做。[原文方法与实验设计](https://arxiv.org/html/1809.04229v1)

## 建议在方案讨论中区分的研究问题

- **监督架构比较**：每个候选只使用当前训练受试者的数据，保留相同信息范围；如果 BIOT/CBraMod 等从头训练，必须明确这不是加载作者预训练模型。
- **外部预训练迁移**：优先做 CBraMod、EEGPT、LaBraM-base、BIOT 的可运行性与 checkpoint 核验。完整报告预训练数据来源、固定/微调参数、输入适配、训练预算；这些数据的额外收益不能全部归于结构。
- **图与频带输入研究**：RGNN/Video-GCNN 提供通道拓扑和跨频带连接假设，但它们改变输入表示/训练方法，适合作为单独实验或后续消融。
- 最终选择由用户查看统一协议下的完整结果决定。本调查只提供候选及成本/适配信息，不用外部公开数据集的准确率预先裁定胜者。

## 需要实施前补齐、当前未完成的核验

- 固定所采用的官方仓库 commit、代码与权重许可、checkpoint 文件 SHA256；本轮只核验网页提供的发布声明和入口。
- 完成 EEGPT Figshare 的可下载验证、LaBraM 实际权重文件入口验证、EEG2Rep 是否有可复用公共权重验证。
- 每种输入适配的输出形状、数值/单位、标签映射与预测汇总方法验证。
- 每模型一次前向/反向与显存计时，再估计四 GPU 并发吞吐；模型参数规模或 README 多 GPU 示例不能替代实际测量。
