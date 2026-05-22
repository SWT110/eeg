# `8.xlsx_to_npy_dataset.py` 使用说明

## 1. 脚本定位

`8.xlsx_to_npy_dataset.py` 的作用是：把 `list_normalization_fixed_duration` 目录中的 EEG `.xlsx` 文件，转换成适合深度学习训练直接读取的 `.npy` 数据集。

这个脚本解决的是一个非常具体的问题：

- 原始输入是按受试者和类别组织的 Excel 文件；
- 每个 Excel 文件本质上是一段**连续 EEG 时间序列**；
- 训练模型时，模型并不直接吃整段连续信号，而是吃很多个**固定长度窗口样本**；
- 因此需要一个中间步骤，把 Excel 里的连续 EEG 数据切成统一形状的窗口，再导出成 NumPy 数组。

你可以把它理解成一条标准的数据准备流水线：

```text
归一化后的 EEG xlsx
-> 筛选 EEG 列
-> 解析时间轴
-> 按窗口切片
-> 生成 X / y / groups
-> 导出为 .npy
```

---

## 2. 这个脚本在整个训练流程中的作用

这个脚本不负责训练模型，也不负责特征工程或模型评估。它的职责非常清晰：

- **输入：** 已经整理好的 EEG `.xlsx`
- **输出：** 模型训练可直接使用的 `.npy`

也就是说，它属于**数据工程（data engineering）**中的“训练数据集构建”步骤。

在你当前项目中，整体流程可以理解为：

```text
EDF / 原始记录
-> 列表化 / 时间切段 / 归一化
-> list_normalization_fixed_duration
-> 8.xlsx_to_npy_dataset.py
-> subject_x_X.npy / y.npy / groups.npy
-> 后续训练脚本读取
```

它在训练中的核心价值主要有 4 点：

1. **统一输入格式**  
   把 Excel 这种适合人工检查的格式，变成适合程序高速读取的 `.npy`。

2. **统一样本形状**  
   把连续 EEG 信号切成固定长度窗口，满足模型输入对统一维度的要求。

3. **显式保留标签信息**  
   每个窗口都会带有类别标签，直接用于分类训练。

4. **保留分组信息**  
   通过 `groups` 记录每个窗口来自哪一份原始 xlsx，方便后续做分组交叉验证，尽量避免数据泄漏。

---

## 3. 输入数据要求

### 3.1 输入根目录

脚本默认读取：

```text
data_to_list/list_normalization_fixed_duration
```

也就是当前脚本所在目录下的：

```text
eeg-data-processing/data_to_list/list_normalization_fixed_duration
```

### 3.2 目录结构要求

输入目录需要按“受试者编号 / 文件”的形式组织，例如：

```text
list_normalization_fixed_duration/
  1/
    1_e_1.xlsx
    1_e_2.xlsx
    1_e_3.xlsx
  2/
    2_e_1.xlsx
    2_e_2.xlsx
    2_e_3.xlsx
  3/
    ...
```

其中：

- `1`、`2`、`3`：受试者编号目录
- `1_e_1.xlsx`：受试者 `1` 的第 `1` 类 EEG 文件
- `1_e_2.xlsx`：受试者 `1` 的第 `2` 类 EEG 文件
- `1_e_3.xlsx`：受试者 `1` 的第 `3` 类 EEG 文件

### 3.3 文件名要求

脚本只处理匹配下面格式的文件：

```text
^\d+_e_[123]\.xlsx$
```

也就是只会处理：

- `1_e_1.xlsx`
- `1_e_2.xlsx`
- `1_e_3.xlsx`
- `11_e_1.xlsx`

不会处理：

- `1_1.xlsx`
- `1_ppg_1.xlsx`
- `note.xlsx`
- `tmp.csv`

### 3.4 表头要求

每个 xlsx 至少需要满足以下条件：

1. 有 `Time` 列
2. 至少有一列列名以 `EEG` 开头
3. 至少有 2 行有效数据

脚本会优先筛选 **21 个公共 EEG 通道**（`Fp1/Fp2/F3/F4/C3/C4/P3/P4/O1/O2/F7/F8/T3/T4/T5/T6/M1/M2/Fz/Cz/Pz`），并按固定顺序输出。例如：

- `EEG Fp1-REF`
- `EEG Fp2-REF`
- `EEG Cz-REF`
- `EEG M1-REF`

以下列会被自动忽略：

- `Time`
- `ECG ...`
- `EMG ...`
- `SaO2 ...`
- `Pulse Rate`
- `EEG A1-REF`
- `EEG A2-REF`
- 其他不以 `EEG` 开头的辅助列

---

## 4. 输入数据在脚本中的语义

这是理解脚本最重要的一点：

**一份 xlsx 文件不是一个训练样本，而是一整段连续 EEG 信号。**

例如：

```text
1_e_1.xlsx
```

它表示的是：

- 受试者 `1`
- 类别 `e_1`
- 一整段连续采样的 EEG 记录

模型训练并不是直接使用这一整段记录，而是把它切成很多小窗口。  
这些窗口才是真正进入模型的训练样本。

因此，从训练视角看：

```text
一份 xlsx
-> 多个固定长度窗口
-> 多个训练样本
```

---

## 5. 脚本的运行方式

这个脚本现在支持两种运行方式：

1. **交互式运行（推荐）**
2. **命令行参数运行**

### 5.1 交互式运行

直接运行：

```bash
python3 data_to_list/8.xlsx_to_npy_dataset.py
```

脚本会依次提示你输入：

1. 输入目录
2. 输出目录
3. 窗口秒数
4. 步长秒数

交互效果大致如下：

```text
Input directory [/path/to/list_normalization_fixed_duration]:
Output directory [/path/to/npy_dataset]:
Window seconds:
Stride seconds (press Enter to use window seconds) [1.0]:
```

其中：

- 目录问题可以直接回车使用默认值
- `Window seconds` 必须输入正数
- `Stride seconds` 可以回车，回车后默认等于窗口秒数

这种方式适合你在服务器上手动运行，因为不需要每次都记命令参数。

### 5.2 参数运行

如果你希望写成脚本或批处理，也可以直接传参数：

```bash
python3 data_to_list/8.xlsx_to_npy_dataset.py \
  --input-root /path/to/input \
  --output-root /path/to/output \
  --window-seconds 1 \
  --stride-seconds 1
```

---

## 6. 程序内部运行逻辑

下面按执行顺序介绍脚本内部到底做了什么。

### 6.1 解析运行配置

程序启动后，先确定 4 个关键参数：

- `input_root`
- `output_root`
- `window_seconds`
- `stride_seconds`

如果你没有通过命令行传入这些参数，脚本就会进入交互式提示流程。

这一步的目的，是把“运行环境配置”收集完整，形成后续处理所需的上下文。

### 6.2 遍历受试者目录

脚本会扫描输入目录下所有名字为数字的子目录，例如：

- `1`
- `2`
- `3`

这些目录会被当作不同受试者。

### 6.3 识别每个受试者的 EEG 文件

在每个受试者目录中，脚本只保留匹配：

```text
<subject_id>_e_<label>.xlsx
```

的文件。

然后从文件名里提取标签：

- `e_1 -> 0`
- `e_2 -> 1`
- `e_3 -> 2`

这里采用的是 **0-based 标签编码**，因为后续多数分类模型、损失函数和训练脚本都更适合使用 `0 / 1 / 2` 这种连续类别编号。

### 6.4 读取单个 xlsx 文件

对于每个 EEG 文件，脚本会做以下处理：

#### 1. 读取 Excel

使用 `pandas.read_excel()` 加载整张表。

#### 2. 检查数据行数

如果数据不足 2 行，直接跳过，因为：

- 无法估算采样间隔
- 也无法形成有效窗口

#### 3. 检查 `Time` 列

如果缺少 `Time` 列，直接跳过。

#### 4. 筛选 EEG 列

只保留列名以 `EEG` 开头的列。

#### 5. 解析时间轴

脚本兼容两种时间格式：

##### 格式 A：字符串时间

例如：

```text
00:00:00.00
00:00:00.50
00:00:01.00
```

这类数据会被解析为 timedelta，再转成秒数。

##### 格式 B：数值秒数

例如：

```text
0.0
0.5
1.0
1.5
```

如果 `Time` 列本身就是数值型，脚本会直接把它当成秒数使用，而不会再走字符串解析流程。

这是一个很重要的兼容性设计，因为不同导出流程产生的 `Time` 列格式并不总是统一的。

#### 6. 检查无效值

脚本会检查 EEG 数据中是否存在：

- `NaN`
- `Inf`
- `-Inf`

如果存在，会跳过该文件，防止污染训练数据。

### 6.5 估算采样间隔

时间轴解析完成后，脚本通过相邻时间差：

```text
intervals = np.diff(times)
```

估算采样间隔。

这里做了两层保护：

1. **要求至少有 2 个时间点**
2. **要求所有相邻间隔都必须大于 0**

也就是说，只要时间轴中出现：

- 重复时间戳
- 倒序时间戳
- 局部非递增时间戳

该文件都会被判定为时间轴无效并跳过。

这一步非常重要，因为如果时间轴错误，后面计算窗口长度、样本数和滑动步长都会出问题。

### 6.6 计算窗口长度和步长对应的采样点数

窗口不是直接按“秒”切，而是要先换算成“采样点数”。

例如：

- `window_seconds = 1`
- 估算出的采样间隔约为 `0.008` 秒

那么每个窗口的点数大约是：

```text
1 / 0.008 = 125
```

脚本会取最近整数，得到：

```text
window_samples = 125
```

同理，步长也会被换算成对应采样点数。

如果窗口秒数或步长秒数过小，小到换算后不到 1 个采样点，脚本会直接报错并跳过该文件，而不是进入死循环或产生空窗口。

### 6.7 切窗

切窗逻辑是标准滑窗：

```text
start_idx = 0
while start_idx + window_samples <= len(data):
    取 data[start_idx : start_idx + window_samples]
    start_idx += stride_samples
```

这表示：

- 每次取一个固定长度窗口
- 然后按步长向前移动
- 直到剩余长度不足一个完整窗口

### 6.8 尾部不足窗口的处理

如果最后剩下的数据长度不足一个完整窗口：

- **不会补零**
- **不会强行保留**
- **直接丢弃**

这样做的好处是可以保证所有训练样本形状一致，并避免引入人工补齐造成的伪特征。

### 6.9 聚合为受试者级数据集

对同一个受试者的多个文件，脚本会把所有切出来的窗口拼起来。

例如受试者 `1`：

- `1_e_1.xlsx` -> 一批窗口
- `1_e_2.xlsx` -> 一批窗口
- `1_e_3.xlsx` -> 一批窗口

最后拼成一套受试者级别的数据：

- `X`
- `y`
- `groups`

---

## 7. 输出文件详细说明

每个受试者会生成一个目录：

```text
subject_1/
  subject_1_X.npy
  subject_1_y.npy
  subject_1_groups.npy
```

### 7.1 `X.npy`

`X` 是模型真正读取的特征数据。

形状为：

```text
(N, C, T)
```

其中：

- `N`：窗口总数
- `C`：EEG 通道数
- `T`：每个窗口的采样点数

例如：

```text
(1653, 21, 125)
```

表示：

- 共 `1653` 个训练样本
- 每个样本有 `21` 个公共 EEG 通道（已去除 `A1/A2`）
- 每个样本长度为 `125` 个时间点

这里特别要注意：

**脚本输出的维度顺序是 `(N, C, T)`，不是 `(N, T, C)`。**

这个顺序更适合大多数 EEG 模型的输入习惯。

### 7.2 `y.npy`

`y` 是每个窗口对应的类别标签。

形状为：

```text
(N,)
```

取值为：

- `0`：来自 `e_1`
- `1`：来自 `e_2`
- `2`：来自 `e_3`

例如：

```text
[0, 0, 0, ..., 1, 1, ..., 2, 2, ...]
```

### 7.3 `groups.npy`

`groups` 是每个窗口所属的“原始文件分组”。

形状为：

```text
(N,)
```

当前设计中，`groups` 的编码粒度是：

**按整份 xlsx 文件分组**

例如同一受试者下：

- `1_e_1.xlsx -> group 0`
- `1_e_2.xlsx -> group 1`
- `1_e_3.xlsx -> group 2`

因此来自同一份 xlsx 的所有窗口，其 `groups` 值完全相同。

这在训练阶段的作用是：

- 你可以按 `groups` 做分组划分
- 尽量避免把同一份连续信号切出的相似窗口，同时分到训练集和验证集

这对于降低数据泄漏风险非常有帮助。

---

## 8. 这个脚本为什么对训练很重要

如果没有这一步，后续训练会遇到几个根本问题：

### 8.1 模型无法直接读取 Excel

训练脚本通常更适合直接读取：

- `numpy.ndarray`
- `torch.Tensor`

而不是循环打开很多 Excel 文件。

Excel 更适合人工查看，不适合高频训练读取。

### 8.2 连续信号长度不统一

不同 xlsx 文件长度不同，模型无法直接处理不定长输入。

切窗后，所有样本统一为固定形状，训练才可行。

### 8.3 标签和分组会很难管理

如果不把标签和分组提前整理出来，后面训练脚本会变得很混乱。

现在这个脚本已经把：

- 特征 `X`
- 标签 `y`
- 分组 `groups`

全部整理好，训练阶段就只需要“读取并划分”。

### 8.4 训练会更快

`.npy` 的读取速度和运行效率明显优于反复读取 `.xlsx`。

这意味着：

- 更快的数据加载
- 更少的 I/O 开销
- 更稳定的训练前处理

---

## 9. 推荐的训练使用方式

生成 `.npy` 后，后续训练最典型的用法是：

1. 读取 `subject_<id>_X.npy`
2. 读取 `subject_<id>_y.npy`
3. 读取 `subject_<id>_groups.npy`
4. 按 `groups` 做训练/验证划分
5. 把 `X` 喂给模型，把 `y` 用于监督学习

也就是说，这个脚本生成的输出，已经非常接近训练输入接口。

你后面训练时，一般只需要再做：

- `numpy -> torch`
- `train/val split`
- `DataLoader` 封装

就可以进入模型训练。

---

## 10. 交互式运行示例

### 10.1 直接运行

```bash
cd /Users/swt/Desktop/eeg/eeg-data-processing
python3 data_to_list/8.xlsx_to_npy_dataset.py
```

### 10.2 按提示输入

例如：

```text
Input directory [/Users/swt/Desktop/eeg/eeg-data-processing/data_to_list/list_normalization_fixed_duration]:
Output directory [/tmp/eeg-npy-interactive]:
Window seconds: 1
Stride seconds (press Enter to use window seconds) [1.0]:
```

含义是：

- 输入目录直接回车，使用默认值
- 输出目录手动指定到 `/tmp/eeg-npy-interactive`
- 窗口长度设为 `1 s`
- 步长直接回车，默认等于 `1 s`

运行结束后会输出类似：

```text
Subject 1: exported 1653 windows from 3 files
Subject 2: exported 1653 windows from 3 files
...
Done.
Output directory: /tmp/eeg-npy-interactive
```

---

## 11. 参数化运行示例

### 11.1 导出 1 s 数据集

```bash
python3 data_to_list/8.xlsx_to_npy_dataset.py \
  --window-seconds 1 \
  --output-root /tmp/eeg-npy-1s
```

### 11.2 导出 2 s 数据集

```bash
python3 data_to_list/8.xlsx_to_npy_dataset.py \
  --window-seconds 2 \
  --output-root /tmp/eeg-npy-2s
```

### 11.3 使用重叠滑窗

例如：

- 窗口长度 `2 s`
- 步长 `1 s`

```bash
python3 data_to_list/8.xlsx_to_npy_dataset.py \
  --window-seconds 2 \
  --stride-seconds 1 \
  --output-root /tmp/eeg-npy-2s-stride1s
```

这会产生重叠窗口，样本数通常会更多。

---

## 12. 典型输出解释

假设你运行后，某个受试者得到：

```text
X: (1653, 21, 125)
y: (1653,)
groups: (1653,)
```

可以这样理解：

- 该受试者共生成了 `1653` 个窗口样本
- 每个窗口有 `21` 个公共 EEG 通道（已去除 `A1/A2`）
- 每个窗口长 `125` 个采样点
- 每个窗口对应一个类别标签
- 每个窗口还带有来源文件分组编号

其中 `125` 不是写死的常数，而是脚本根据 `Time` 列估算出来的每秒采样点数。

这意味着脚本不是把采样率硬编码成 `128 Hz` 或 `200 Hz`，而是尽量从数据本身推断。

---

## 13. 异常与跳过策略

脚本对坏数据采取的策略是：

**尽量跳过问题文件，而不是让整个批处理直接崩掉。**

以下情况会跳过当前文件：

- 无法读取 Excel
- 缺少 `Time` 列
- 没有 EEG 列
- 数据行数不足 2 行
- `Time` 列无法解析
- `Time` 列包含 `NaN`
- 时间轴存在重复或倒序
- EEG 数据存在 `NaN`
- EEG 数据存在 `Inf / -Inf`
- 窗口秒数或步长秒数太小，换算后不足 1 个采样点

跳过时，脚本会在终端输出明确原因，方便你定位异常文件。

---

## 14. 测试覆盖情况

这个脚本配套有单元测试，主要覆盖了以下行为：

- 正常导出
- 非 EEG 列过滤
- 尾部不完整窗口丢弃
- `stride` 默认等于 `window`
- 数值时间列处理
- 字符串时间列处理
- `NaN / Inf / -Inf` 拦截
- 无效时间轴拦截
- 部分重复时间戳拦截
- 交互式配置解析

这些测试的意义是：不仅验证“能跑通”，也验证脚本在坏数据和边界条件下不会默默生成错误数据集。

---

## 15. 实际使用建议

### 15.1 第一次使用建议

第一次建议先导出一版 `1 s` 数据集：

```bash
python3 data_to_list/8.xlsx_to_npy_dataset.py
```

然后在交互里输入：

- 窗口长度：`1`
- 步长：回车

### 15.2 导出完成后先检查 1 个受试者

例如：

```bash
python3 - <<'PY'
import numpy as np
from pathlib import Path

root = Path('/tmp/eeg-npy-interactive/subject_1')
X = np.load(root / 'subject_1_X.npy')
y = np.load(root / 'subject_1_y.npy')
g = np.load(root / 'subject_1_groups.npy')

print('X:', X.shape)
print('y:', y.shape, sorted(set(y.tolist())))
print('groups:', g.shape, sorted(set(g.tolist())))
PY
```

### 15.3 再考虑比较不同窗口长度

例如后续可以比较：

- `1 s`
- `2 s`
- `2 s + 1 s stride`

这样可以观察不同窗口设计对训练效果的影响。

---

## 16. 一句话总结

`8.xlsx_to_npy_dataset.py` 的核心价值，是把“便于人工查看的连续 EEG Excel 文件”，转换成“便于模型训练直接读取的窗口化 NumPy 数据集”。

它完成了训练前最关键的数据对齐工作：

- 对齐输入格式
- 对齐样本长度
- 对齐标签
- 对齐分组信息

因此，它是你从 `list_normalization_fixed_duration` 走向后续训练脚本之间的关键桥梁。
