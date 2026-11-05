# 四个 EEG 模型官方实现快照与接口审计

日期：2026-11-02（America/New_York）。范围：实际获取作者源代码、静态审计构造器、输入输出和数据适配要求。本轮未修改项目生产代码，未安装依赖，未执行训练或模型前向；文中的中间形状由源代码静态推导。

## 官方快照与许可文件

全部快照位于 `eeg/local_artifacts/model_sources_2026-11-02/`。四个仓库均以 `git clone --depth 1` 获取；审计后 `git status --porcelain` 均为空，保留原始源代码状态。

| 模型 | 作者仓库 | 锁定 commit | 本地核心文件 | 快照中的许可文件 |
| --- | --- | --- | --- | --- |
| EEGSimpleConv | [elouayas/EEGSimpleConv](https://github.com/elouayas/EEGSimpleConv) | `a732d9a3758ede6f31ddd5de1fece2e790ef4d0f` | `EEGSimpleConv/models/EEGSimpleConv.py` | 未发现 LICENSE/COPYING 文件；不据此推断授权范围 |
| EEG-Deformer | [yi-ding-cs/EEG-Deformer](https://github.com/yi-ding-cs/EEG-Deformer) | `47df1c4d8ce375ae21117f84e81e969041d907f6` | `EEG-Deformer/models/EEGDeformer.py` | 未发现 LICENSE/COPYING 文件；不据此推断授权范围 |
| TSception V2 | [yi-ding-cs/TSception](https://github.com/yi-ding-cs/TSception) | `9efd666b618d006e32e6da1d30dbc79b1d190604` | `TSception/code/networks.py` | `LICENSE-CBCR`，CBCR License 1.0，允许非商业使用及修改，要求保留声明 |
| CTNet | [snailpt/CTNet](https://github.com/snailpt/CTNet) | `fb83814abc55bafaea6e71cc18475bf331c8d28b` | `CTNet/main_subject_specific.py`；另有 `main_cross_subject_LOSO.ipynb` | `LICENSE`，MIT，Copyright 2024 snailpt |

下文路径以本快照目录为根；这些代码快照作为实现证据保存，不应将上游训练脚本直接导入项目训练流程。

## EEGSimpleConv

源码：`EEGSimpleConv/models/EEGSimpleConv.py:4`。类名及精确签名：

```python
EEGSimpleConv(fm, n_convs, resampling, kernel_size, n_chan,
              n_classes, sfreq, n_subjects=None)
```

- 前七项均无默认值；`fm` 是第一层特征数，`n_convs` 是卷积块数，`resampling` 是内部重采样频率。
- 输入 `[B, C, T]`；首层 `torchaudio.transforms.Resample(orig_freq=sfreq, new_freq=resampling)` 后进入 Conv1d、BatchNorm、ReLU 和若干双卷积块，最后沿时间均值池化。
- `n_subjects=None` 时输出为分类 logits `[B,n_classes]`；设置该参数会新增受试者分类头，输出 `(class_logits, subject_logits)`。本次纯分类任务应明确关闭该辅助头，避免引入未约定的辅助目标。
- 纯模型文件仅依赖 `torch` 和 `torchaudio.transforms.Resample`。上游完整 requirements 的 torch/torchaudio 为 `2.0.0/2.0.1`，其 MOABB、MNE 等是完整实验流程依赖，不是该模型前向依赖。
- 官方 `run_results.ipynb` 第一个跨受试者示例使用 `p=[109,4,70,8]`，对应 `fm=109,n_convs=4,resampling=70,kernel_size=8`，后续由 `scripts/scripts.py:54` 附加通道、类别、原采样率。它是示例配置，不是构造器默认值；同 notebook 还有其他配置，不能泛称为唯一原始配置。
- 128 Hz / 21 通道 / 1920 点可以直接作为 `sfreq=128,n_chan=21,n_classes=3` 输入。内部若选 `resampling=70`，15 秒变成约 1050 点；若保持 128 Hz 则显式设为 128。应在模型配置中记录该选择。
- 原码 `MaxPool1d(2) if i > 0 - 1 else ...` 的条件实际是 `i>-1`，因此每一个块（包括第 0 个）都会池化。复现时不要无记录地改成仅第 1 块以后池化。
- 原码 Conv1d 对偶数核使用 `padding=kernel_size//2`，每次卷积会让时间长度增加 1；不要直接替换为严格 same padding 后仍宣称逐层一致。时间均值池化使分类头不依赖最终长度。

进一步核对作者 [arXiv v2 全文附录 D / Table IX](https://arxiv.org/html/2309.07159#A4) 后，作者给出的**跨受试者推荐搜索范围**为：`fm(W)=64–144`、`n_convs(K)=2–4`、`kernel_size=5–8`、重采样频率 `50–80 Hz`；不是一组通用固定默认值。对照组内受试者设置则为核 12–17、K=1、重采样 70–100 Hz。因此 notebook 中 `109/4/70/8` 是符合跨受试者区间的具体示例，`84/1/95/13` 更接近组内受试者路线。若本轮保留 128 Hz，可使用同一骨干宽度/深度/核并将 `resampling=128` 记录为本任务的 identity resampling 适配；不能称 128 Hz 为作者推荐值。原始模型没有 Dropout 层。

## EEG-Deformer

源码：`EEG-Deformer/models/EEGDeformer.py:120`。真正模型类是 `Deformer`，精确签名：

```python
Deformer(*, num_chan, num_time, temporal_kernel, num_kernel=64,
         num_classes, depth=4, heads=16, mlp_dim=16,
         dim_head=16, dropout=0.)
```

- 输入 `[B,C,T]`；模型自行增加单通道维度，输出分类 logits `[B,num_classes]`，没有 softmax。
- 模型前向只需要 `torch`、`einops.rearrange`、`einops.layers.torch.Rearrange`。Lightning、torchmetrics 和上游数据处理库只属于模型外训练封装；不要为接入核心模型导入 `models/model_handler.py`。
- 可直接设置 `num_chan=21,num_time=1920,num_classes=3`。`temporal_kernel` 应使用奇数，原码按 `(kernel-1)//2` 填充；偶数核会造成实际时间长度与 `num_time//2` 推定值不一致，首先在位置编码处出错。
- 原文件末尾示例是 `temporal_kernel=11,num_kernel=64,depth=4,heads=16,mlp_dim=16,dim_head=16,dropout=0.5`。各任务训练入口又分别使用 21/51 等核及不同 Dropout，因此这些是不同任务配置，不能混用后宣称唯一原设定。
- 以奇数核、64 特征、4 层为例：1920 点经 stem 池化为 960；分层时间嵌入依次为 `480→240→120→60`；最终 `64*60+64*4=4096` 个特征进入三分类线性头。
- 注意力的 token 轴是 **64 个卷积特征**，每个 token 的 embedding 是逐层缩短的时间向量；不要把它移植成沿 960 个时间 token 做同一种注意力。位置参数形状为 `[1,64,960]`。
- FTL 支路与 coarse 支路均将时间减半；最后拼接各层 `log(mean(x**2,time))` 的 DIP 特征。原码这处没有 epsilon/clamp，若实现数值保护，应在适配说明中明确记录。
- stem 的两个空间/时间卷积使用 max-norm=2，每次 forward 执行 `torch.renorm(...,p=2,dim=0,maxnorm=2)`；不能遗漏。
- 上游 Lightning 封装的 Adam、weight decay 和 cosine scheduler 与本轮统一训练配置不同。接入的是骨干，不应将这些上游训练设置悄悄带入当前实验。

## TSception V2 与 21 通道

源码：`TSception/code/networks.py:6`。本次应采用 `TSception`，同文件中的 `TSceptionIJCNN` 是旧版。精确签名：

```python
TSception(num_classes, input_size, sampling_rate, num_T,
          num_S, hidden, dropout_rate)
```

- 所有项均无默认值；官方 README 示例为 `num_T=15,num_S=15,hidden=32,dropout_rate=0.5`，`sampling_rate=128`。
- 输入 `[B,1,C,T]`；`input_size=(1,C,T)`；输出 logits `[B,num_classes]`。纯模型仅依赖 torch。
- 时间核为 `[0.5,0.25,0.125]*sampling_rate`，128 Hz 对应 `[64,32,16]` 点。1920 点的三个支路在池化后长度分别为 `[232,236,238]`，沿时间拼接成 706；空间支路池化后 353，融合池化后 88，随后时间均值池化，再做 `num_S→hidden→3` 分类。
- 全局空间支路的核覆盖全部通道；半球支路的核和步长均为 `int(C*0.5)`；两支路输出拼成 global/left/right 三行，再由 `(3,1)` 融合核聚合。

项目现有公共通道在 `eeg-data-processing/data_to_list/8.xlsx_to_npy_dataset.py:23` 中定义为：

```text
原顺序：Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,M1,M2,Fz,Cz,Pz
左侧：  Fp1,F3,C3,P3,O1,F7,T3,T5,M1       （索引0,2,4,6,8,10,12,14,16）
右侧：  Fp2,F4,C4,P4,O2,F8,T4,T6,M2       （索引1,3,5,7,9,11,13,15,17）
中线：  Fz,Cz,Pz                          （索引18,19,20）
```

`M1/M2` 是当前输入通道名中的一对侧向通道；其采集/参考含义应按真实数据记录描述，不能随意改称为其他头皮电极。

**官方行为事实**：`TSception/code/utils.py:105` 的 `generate_TS_channel_order` 只保留带数字的通道，按编号配对；因此本项目调用它会输出左 9 + 右 9，共 18 通道，删除 `Fz/Cz/Pz`。README 的 DEAP 示例也从 32 通道变成 28 通道，不是保留所有中线。将其直接作用于本数据后，应称为 18 通道输入。

**直接塞入 21 通道的后果**：半球核/步长为 10，输出两行分别取连续的 10+10 个位置，第 21 个位置不进入该支路。即使先排成左 9、右 9、中线 3，也会混入对侧/中线通道；代码能运行不代表实现了正确的两半球分组。

**若要求每个模型均可使用全部 21 通道，一种显式任务适配方案**：全局支路仍使用全部 21；半球支路先 gather 左 9 + 右 9 的 18 通道，再以空间核/步长 9 处理，三条中线只通过全局支路。融合仍得到 global/left/right 三行。这是本审计提出的适配建议，**不是官方现成实现**，必须记录为 21 通道布局适配，不得静默称为原版参数。

本审计没有替主实验作出采用 18 通道原版还是 21 通道适配的决定。

## CTNet

官方纯 `.py` 中模型位于 `CTNet/main_subject_specific.py:226`，模型类名 `EEGTransformer`；精确签名如下：

```python
EEGTransformer(heads=2, emb_size=16, depth=6, database_type='A',
               eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
               eeg1_pooling_size1=8, eeg1_pooling_size2=8,
               eeg1_dropout_rate=0.3, eeg1_number_channel=22,
               flatten_eeg1=600, **kwargs)
```

- 输入 `[B,1,C,T]`；返回 `(features, logits)`，features 是 `[B,n_tokens,emb_size]`，logits 是 `[B,n_classes]`。
- 模型构件依赖 torch、einops、math；整训练脚本额外依赖 pandas、torchsummary、utils（又导入 cv2 等）。脚本顶层还覆盖 `CUDA_VISIBLE_DEVICES='0'`、修改 cuDNN 设置，因此不应直接 import 整个训练脚本来注册模型。
- `eeg1_number_channel` 在模型构造中**被忽略**：真正通道和类别来自 `utils.numberClassChannel(database_type)`，仅支持 A=`4类/22通道`、B=`2类/3通道`。本任务必须通过明确的接口适配设置 `21通道/3类`，单独传 `eeg1_number_channel=21` 不会生效。
- `PositioinalEncoding.forward` 内硬编码 `.cuda()`，需按输入/参数 device 管理，才能正确支持 CPU 验证与队列设备分配。
- `emb_size` 必须等于 `eeg1_f1*eeg1_D` 且可整除 `heads`。PatchEmbedding 的 `emb_size` 参数没有额外投影层，不会替你改变前端的实际输出宽度。
- 两层池化均为 8 时，1920 点变为 30 tokens；应设置 `flatten_eeg1=30*emb_size`。例如 `f1=8,D=2,emb=16` 对应 480，`f1=20,D=2,emb=40` 对应 1200。原 600 或 240 都不能直接用于本任务。
- 位置表默认长度 100，30 tokens 在范围内；保持 pooling 8/8 可直接覆盖当前输入。
- 原代码 CNN 时间核 64 在 128 Hz 下是 0.5 秒；在原 MI 采样率下对应不同时间跨度。保留 64 点还是按物理时间缩放，应作为明确的模型配置记录。
- `.py` 类构造默认 CNN Dropout=0.3，位置编码 Dropout=0.1，Transformer 的注意力/残差/FFN Dropout=0.5，分类头 Dropout=0.5；main 的配置又按是否 LOSO 将 CNN Dropout 设为 0.25/0.5。
- 原注意力 logits 除以 `sqrt(emb_size)`，并非 `sqrt(head_dim)`；原残差是 post LayerNorm，最后 `features=cnn+trans(cnn)` 还有一次外部相加。以通用 Transformer 直接替换会改变结构，需核对。
- `.py` 的 main 实际使用 `f1=8,D=2,emb=16,heads=2,depth=6,flatten=240`（原 1000 点输入）。`main_cross_subject_LOSO.ipynb` 的类声明默认又有 `f1=20,emb=40,heads=4`，但其执行配置也覆盖成 16/8/2。应选定并记录一套配置，不混合不同默认值。
- 作者完整训练流程包含 S&R 增强及训练/验证划分。本轮已约定无增强及统一训练规则，应仅接入其架构，避免把上游流程带入。

## 主任务选定配置的事实核对

主任务已选择的 CTNet `F1=8,D=2,emb=16,heads=2,depth=6,k=64,pool=8/8,CNN Dropout=0.25` 与上游主脚本的实际 LOSO 执行配置一致；本任务仍需将通道、类别与 flatten 数改为 21、3、480，并处理 device。

主任务已选择的 Deformer `temporal_kernel=51,num_kernel=64,depth=4,heads=16,dim_head=16,mlp_dim=16,dropout=0.25` 与 `main_MWL.py` 经 `model_handler.py` 映射后的配置一致（`num_layers=6` 在封装里减 2 得到内部 depth=4）。51 是奇数，1920 点输入的形状推导成立。以上属于静态一致性核对，未运行前向或训练。

## 给实现者的最小核对清单

1. 外层接口正确区分 `[B,C,T]`（SimpleConv、Deformer）与 `[B,1,C,T]`（TSception、CTNet）。
2. 不导入任何会设置 GPU、读数据或运行训练的上游顶层脚本。
3. 保存作者 URL、锁定 commit、具体构造参数、许可声明、输入布局/通道适配说明。
4. 在正式训练前另行验证 batch=1/2 的有限 logits、梯度和分类头尺寸；此审计未执行这些验证，也不将静态推导写成实测。
