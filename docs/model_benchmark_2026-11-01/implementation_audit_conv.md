# CNN / TCN 候选模型实现审计

日期：2026-11-02。这里只获取并阅读源码，没有执行作者训练脚本，没有安装依赖、改训练环境或修改生产模型代码。作者的默认函数参数、论文运行脚本的参数、Braindecode 的默认参数分别记录，避免把不同实现静默混在一起。

## 已落地源码快照

路径基准：`/Volumes/S70/Desktop/lunwen/eeg/local_artifacts/model_sources_2026-11-02/`。

| 模型 | 作者仓库固定 commit | 本地权威入口 | 代码授权记录 |
| --- | --- | --- | --- |
| FBCNet | `de1bbdd8a54cb1e466830e3d47070e0e56761a37` | `FBCNet/codes/centralRepo/networks.py`，滤波在 `transforms.py` / `saveData.py` | MIT，保留 `LICENSE` |
| EEG-TCNet | `363c7e52d19285b0bac9a8846bbdbd6aac5aa851` | `EEG-TCNet/utils/models.py`，论文运行配置见 `sample_train.py` | Apache-2.0，根目录和模型目录 LICENSE 均保存 |
| EEG-ITNet | `5afcd39b28e832b662bd234af4f43ec7acd3d514` | `EEG-ITNet/Main.ipynb`；另有 `Main_extracted.py` 仅提取 code cells 方便阅读 | 作者 README 有明确的非商业、禁止衍生、未经同意不再分发限制；没有宽松许可证 |
| EEGNeX | `044d7fb2a5b953c7004f30fe45db40d1ce57af5c` | `EEGNeX/BenchmarkModels.py::EEGNeX_8_32` | MIT，保留 `LICENSE.md` |

四个仓库是浅克隆中的选择性源码快照，不含训练数据或模型权重。每个目录的 `SNAPSHOT.json` 和根目录 `SOURCE_MANIFEST_CONV.json` 保存作者仓库、commit、所选文件、文件 SHA-256 和来源链接。

另存了一份 Braindecode 官方 PyTorch 参考：

`braindecode_conv_reference/`，commit **`d19b024aa37a334f783b4873285b1b52044721d5`**，许可证 **BSD-3-Clause**。

已保存 `braindecode/models/{fbcnet,eegtcnet,eegitnet,eegnex,deep4}.py`、仓库 `LICENSE.txt` 及部分支持模块。`SNAPSHOT.json` 记录下载文件的固定 commit URL 与 SHA-256。它是第二实现来源，不能自动等同于作者原始实现。

### 原始来源

- [FBCNet 作者仓库](https://github.com/ravikiran-mane/FBCNet/tree/de1bbdd8a54cb1e466830e3d47070e0e56761a37)
- [EEG-TCNet 作者机构仓库](https://github.com/iis-eth-zurich/eeg-tcnet/tree/363c7e52d19285b0bac9a8846bbdbd6aac5aa851)
- [EEG-ITNet 作者仓库](https://github.com/AbbasSalami/EEG-ITNet/tree/5afcd39b28e832b662bd234af4f43ec7acd3d514)
- [EEGNeX 作者仓库](https://github.com/chenxiachan/EEGNeX/tree/044d7fb2a5b953c7004f30fe45db40d1ce57af5c)
- [Braindecode 固定版本](https://github.com/braindecode/braindecode/tree/d19b024aa37a334f783b4873285b1b52044721d5)

## FBCNet

### 作者实际接口和结构

构造器：

`FBCNet(nChan, nTime, nClass=2, nBands=9, m=32, temporalLayer='LogVarLayer', strideFactor=4, doWeightNorm=True)`。

输入为 **`[B, 1, C, T, bands]`**，已经完成频带滤波；作者网络内部不执行滤波。内部转换成 `[B, bands, C, T]`。

1. 9 个频带分组空间卷积：`Conv2d(9, 9×32, (C,1), groups=9)`，默认包含 bias，空间权重 max-norm=2。
2. `BatchNorm2d(288)` 与 Swish（`x * sigmoid(x)`，等价 SiLU）。
3. 将时间分成 **4 个连续等长段**，本项目 T=1920 对应每段 480 点；沿每段时间计算 LogVar。这里的 `strideFactor` 不是数据集切窗步长。
4. 展平 288×4=1152 个特征，约束线性分类器 max-norm=0.5，最后 `LogSoftmax(dim=1)`。

作者 `LogVarLayer` 使用 `torch.var` 的默认无偏估计，即在 PyTorch 1.12 中 `unbiased=True`；方差先截断至 `[1e-6,1e6]` 再取自然对数。应保留该约定或显式记录改变。

作者没有另写初始化函数，网络层沿用 PyTorch 自身初始化；不应不加说明地套用全局 Xavier 初始化。默认 `nTime` 没有用于自动改变模型结构。T 必须能被4整除，原实现没有补零处理。

本项目 C=21、三分类、9频带、m=32 时，从上述定义推导可训练参数约 **10,371**（不含 BN running buffers）；最终应再由实际实现统计核验。

### 滤波事实

`saveData.py::transformAndSave` 的默认频带为 `[4,8], [8,12], …, [36,40]` Hz，共9个；原数据采样率默认250 Hz。移植本项目时必须显式设为 **128 Hz**。

`transforms.py::filterBank`：Chebyshev Type II IIR，通带最大损失3 dB，阻带衰减30 dB，过渡带2 Hz；用 `cheb2ord` / `cheby2` 确定滤波器。默认 `filtType='filter'`，实际采用 **`scipy.signal.lfilter`**，不是 `filtfilt`。可支持 `filtfilt`，但这是不同输入处理选择。

应冻结在每个15秒窗口独立滤波还是在整记录上滤波后切窗；两者边界状态不同。频带滤波可以作为无标签的固定模型输入变换，但不能将 FBCNet 的完整流程称为“只替换分类头”。

### Braindecode 差异与接入点

Braindecode 接收 `[B,C,T]`，内部 `FilterBankLayer` 生成频带；最终输出 logits，且时间不能整除4时会补零。其滤波器参数、边界处理必须和本轮选择明确对应。对于当前 T=1920，无需补零。

本项目通用模型返回 `(features, logits)` 时，可以暴露1152维 LogVar 特征；选择作者 logits 前的线性输出搭配 `CrossEntropyLoss`，或保留 log-probabilities 搭配 `NLLLoss`。避免把已经归一化的概率当原始 logits。

## EEG-TCNet

### 两种不同的作者“默认”

`utils/models.py` 函数签名：

`EEGTCNet(nb_classes, Chans=64, Samples=128, layers=3, kernel_s=10, filt=10, dropout=0, activation='relu', F1=4, D=2, kernLength=64, dropout_eeg=0.1)`。

但是作者 `sample_train.py` 真正使用的论文运行配置是：

| 参数 | 论文运行脚本值 |
| --- | ---: |
| F1 / D / F2 | 8 / 2 / 16 |
| EEG 时间核 `kernLength` | 32 |
| EEG 前端 Dropout | 0.2 |
| TCN residual blocks | 2 |
| 每个 TCN 卷积核 | 4 |
| TCN filters | 12 |
| TCN Dropout | 0.3 |
| TCN activation | ELU |

不能把函数签名默认值误当作论文主配置。

### 张量和逐层结构

作者输入 `[B,1,C,T]`，内部转 `[B,T,C,1]`。前端：

1. 时间 Conv2D，kernel `(32,1)`、8滤波器、same、无bias，随后BN。
2. **此处没有ELU**；直接深度空间卷积 `(1,C)`，D=2、max-norm=1，随后BN、ELU、AvgPool `(8,1)`、Dropout0.2。
3. SeparableConv2D，**固定时间核16**，same、无bias，随后BN、ELU、AvgPool `(8,1)`、Dropout0.2。

T=1920 时得到30个时间位置、16维特征。TCN 为2个残差块，每块有2次 `Conv1D→BN→ELU→Dropout`，dilation依次1、2，kernel4；采用**左侧因果 padding**。第一块的16→12残差路径有带bias的1×1卷积。两块的普通TCN卷积均沿用 Keras 默认 **bias=True**，`kernel_initializer='he_uniform'`。

TCN 之后取**最后一个时间位置**，得到12维特征，Dense3、max-norm0.25、Softmax。适配通用CE训练应返回Dense之前的12维features及Dense logits。

本项目C=21/三分类，按作者论文运行脚本结构推导为 **4,067** 可训练参数；原作者C=22/四分类相同配置为4,096。实际实现应再次统计。

### 第二实现的差异

当前 Braindecode `EEGTCNet` 默认使用8前端滤波器、时间核64、TCN depth2/kernel4/filters12、统一Dropout0.5。前端会额外在首个时间卷积BN之后使用ELU；其分离时间核与第二池化核的长度还由F2/F1推导。其TCN卷积bias=False。因此不能直接实例化默认参数后声称与上述作者脚本逐层完全相同。

建议纯PyTorch移植时以作者 `sample_train.py` 和 `utils/models.py` 配套为主，保留 Apache-2.0 头和许可证。Braindecode可以辅助核对因果填充和返回接口。

## EEG-ITNet

### 来源和授权边界

作者只提供 notebook 主入口，其 README 规定非商业、禁止衍生和未经同意不再分发。后续生产接入建议基于 **Braindecode BSD-3 的独立 PyTorch 实现**并保留其来源及版权，论文与作者 notebook 用于核对结构事实；不要把作者 notebook 直接复制或逐行移植进生产文件。是否获得作者另外授权未在本次调查中确认。

### 作者结构事实

`Network(Chans, Samples, out_type='single')` 依赖 notebook 全局 `Net_name=4`、`Data_name`、`analysis_type`；`Data_name` 决定原始输出2类或4类，需要在本项目显式设3类。输入 `[B,C,T,1]`；`out_type='multi'` 返回 `[probabilities, embedded]`。

三个 Inception 分支的时间核为 **16 / 32 / 64**、滤波器数 **2 / 4 / 8**，每个分支：时间卷积无bias、BN、跨全部电极的depthwise空间卷积（D=1、无bias、max-norm1）、BN、ELU。拼接后共14个特征图，AvgPool时间4。

TCN共 **4个深度时间残差块**，每个块有2次depthwise时间卷积，kernel4、depth multiplier1，dilation **1/2/4/8**，分别左填充 **3/6/12/24** 点；每次卷积后是BN、ELU、Dropout，第二次Dropout后与块输入相加，**加法之后没有额外ELU**。

随后维度压缩为 **14→14 的1×1卷积**（不是14→28）、BN、ELU、时间AvgPool4、Dropout、Flatten，再Dense类别数，max-norm0.25。

作者 notebook 跨受试者方案（`analysis_type=2`）Dropout为 **0.2**；被试内方案为0.4。本项目T=1920，两次时间池化后120个位置，最终特征为14×120=1680。

作者代码中的 `ZeroPadding2D(data_format='channels_first')` 和其他channels-last层混用：原始 `[B,1,T,14]` 上它实际仍是给时间轴左侧加零。移植时应按实际张量轴实现左侧padding，不直接照抄不同数据格式参数。

### Braindecode差异

已保存的当前实现保留14→14维度压缩、空间max-norm1、分类max-norm0.25；默认Dropout0.4，跨受试者实验要显式改0.2。其TCN depthwise卷积bias=False，作者notebook为Keras默认bias=True；BN默认也遵循PyTorch约定。模型文档明确说原作者没有审核其正确性，应如实标记来源为“Braindecode实现及本项目适配”。

## EEGNeX

作者入口：`EEGNeX_8_32(n_timesteps, n_features, n_outputs)`，输入 `[B,1,C,T]`，输出Softmax。函数名中的8/32对应前两个block的滤波器宽度，作者该函数实际时间核长度是64。

| Block | 作者实际操作 |
| --- | --- |
| 1 | Conv 1→8，kernel `(1,64)`，same，无bias；BN；ELU |
| 2 | Conv 8→32，kernel `(1,64)`，same，无bias；BN；**无ELU** |
| 3 | Depthwise空间Conv 32→64，kernel `(C,1)`，D=2，无bias，max-norm1；BN；ELU；时间AvgPool4（same）；Dropout0.5 |
| 4 | Conv 64→32，kernel `(1,16)`，dilation `(1,2)`，same，无bias；BN；**无ELU** |
| 5 | Conv32→8，kernel `(1,16)`，dilation `(1,4)`，same，无bias；BN；ELU；时间AvgPool4（same）；Dropout0.5 |
| Head | Flatten；Dense3，max-norm0.25；Softmax |

本项目T=1920的时间池化为1920→480→120，flatten960维。两次池化的padding对当前可被16整除的输入不额外添点。

**原作者文件的BN轴有一个必须明确处理的差异：**卷积显式channels-first，但 `BatchNormalization()` 没有设置axis，因此Keras默认axis=-1，是时间轴，而非卷积feature轴。常规PyTorch `BatchNorm2d(features)` 是另一种归一化语义。应明确选作者代码语义或规范通道BN适配并记录，不能声称逐层完全一致。

当前Braindecode版本和作者函数的其他差异：默认最后池化为8（总缩减32倍），block1后没有作者的ELU，BN按feature通道进行，池化还带固定padding1。它只能作为独立参考，不能与作者8/32函数默认参数混用。

本项目生产实现如果采用作者卷积结构与常规通道BN，应命名/元数据注明“EEGNeX作者结构的PyTorch通道BN适配”，同时记录pool4/4和block1的ELU。

## DeepConvNet / Deep4Net 补充

权威源码为 Braindecode `braindecode_conv_reference/braindecode/models/deep4.py`，作者标注Robin Schirrmeister，BSD-3，论文 DOI [10.1002/hbm.23730](https://doi.org/10.1002/hbm.23730)。

默认结构：

- 首层分解为时间卷积 **1→25、时间核10、bias=True**，以及空间卷积 **25→25、跨全部C电极**；BN开启时空间bias=False。两个卷积之间**没有非线性或BN**。
- 第一个卷积块之后依次为BN、ELU、时间MaxPool3/stride3。
- 后续三个块宽度 **50 / 100 / 200**，每块为 **Dropout0.5→valid时间Conv10→BN→ELU→MaxPool3/stride3**。BN开启时卷积bias=False。
- 所有BN使用eps `1e-5`、momentum `0.1`，gamma=1/beta=0。
- 所有卷积（包括分类器）使用Xavier uniform、gain1；存在的bias初始化0。
- **该实现没有max-norm约束**。不要从Keras其他baseline默认添加约束。
- `final_conv_length='auto'`：最后Conv将剩余整个时间范围映射到3类，带bias；当前官方版本输出logits。

T=1920时的实际时间尺寸：

`1920 → conv 1911 → pool 637 → conv 628 → pool 209 → conv 200 → pool 66 → conv 57 → pool 19`。

因此最终特征200×19=3800，分类器 `Conv2d(200,3,(19,1))`（原代码时间轴在第2维）。转为本项目 `[B,1,C,T]` 的等价实现，可以将时空卷积核轴交换，flatten后Linear(3800,3)与整时间分类卷积等价。

当前Braindecode的首个时空卷积用 `CombinedConv` 合并计算，但分别保留时间和空间权重；朴素纯Torch实现可分别执行两个线性卷积。对T=1920不会触发其短输入自动缩核分支。

## 共同的PyTorch1.12接入注意

1. 固定本项目公共输入 `[B,1,21,1920]`，输出 `(features, logits)`；TF Softmax/作者LogSoftmax需要和训练损失协调。
2. Keras stride1的same padding在偶数核时不对称：左侧 `floor((effective_kernel-1)/2)`、右侧剩余。直接对称填充k//2会多出一个时间点。因果卷积仅左侧填充 `(k-1)×dilation`。
3. TensorFlow BN默认eps=1e-3、momentum=.99，与PyTorch momentum=.01对应；TF模型直接改成PyTorch默认eps1e-5/momentum.1会改变训练。另需区分EEGNeX作者BN轴问题。
4. Keras kernel默认Glorot uniform、bias0；EEG-TCNet TCN卷积显式He uniform。PyTorch自身Conv默认初始化不同。应逐模型确定，不能用一个共享初始化覆盖全部候选。
5. Keras约束通常在optimizer更新后应用，作者FBCNet的约束在forward之前应用。若本项目采用forward约束应记录实现约定，使用`torch.no_grad()`和`copy_`避免在autograd中错误覆盖。
6. 约束应按每个输出滤波器/类别的权重向量求范数；depthwise和grouped层不能把所有滤波器合并求一个全局范数。
7. 最新Braindecode文件包含现代type annotation、mixin、einops、parametrization及其他包依赖。它们是审计参考，不应在现有PyTorch1.12共享环境直接安装最新版整个库；提取所需纯Torch算子时保留其BSD来源并进行接口/数值检查。
8. 是否精确照搬作者采样率下的核时长，或把核长度按128Hz重标定，是实验设计选择。当前审计只记录源码中的采样点数，不静默改变核。
