# Mac 测试训练环境清单

本文档用于在 Apple Silicon Mac 上配置本项目的本地测试和小规模训练环境。这个环境面向 Mac ARM，不用于 CUDA 服务器；CUDA x86 服务器应单独创建自己的 Linux/CUDA 环境。

## 适用范围

| 项目 | 约定 |
| --- | --- |
| 机器架构 | macOS arm64 / Apple Silicon |
| Python 版本 | Python 3.10 |
| 环境类型 | 项目内 Conda 环境 |
| 环境目录 | `.conda-envs/eeg-mac-test-train` |
| Jupyter kernel | `Python (eeg-mac-test-train)` |
| 默认数据目录 | `local_artifacts/` |
| 训练设备 | 优先 `cpu` 做 smoke test；可按需尝试 `mps` |
| 不适用场景 | CUDA 训练、服务器正式长时间训练 |

## 依赖清单

可安装依赖写在根目录的 `requirements-mac-test-train.txt` 中，主要覆盖以下几类：

| 类别 | 包 |
| --- | --- |
| 数值计算 | `numpy<2`、`scipy` |
| 表格和 Excel | `pandas`、`openpyxl` |
| 机器学习 | `scikit-learn` |
| 深度学习 | `torch`、`torchvision`、`torchaudio`、`einops`、`torchsummary` |
| EEG / EDF 处理 | `mne` |
| 可视化 | `matplotlib`、`plotly`、`pillow` |
| API | `fastapi`、`uvicorn[standard]`、`python-multipart`、`httpx` |
| 测试 | `pytest` |
| Notebook | `ipykernel` |

当前本机已验证的主要版本如下：

| 包 | 版本 |
| --- | --- |
| Python | `3.10.20` |
| NumPy | `1.26.4` |
| pandas | `2.3.3` |
| openpyxl | `3.1.5` |
| SciPy | `1.15.3` |
| scikit-learn | `1.7.2` |
| PyTorch | `2.12.0` |
| torchvision | `0.27.0` |
| torchaudio | `2.11.0` |
| einops | `0.8.2` |
| torchsummary | `1.5.1` |
| matplotlib | `3.10.9` |
| Pillow | `12.2.0` |
| MNE | `1.12.1` |
| Plotly | `6.7.0` |
| FastAPI | `0.136.1` |
| Uvicorn | `0.47.0` |
| pytest | `9.0.3` |
| ipykernel | `7.2.0` |

视频切割脚本还依赖系统命令 `ffmpeg`。如果只做数据处理、测试和训练，可以暂时不安装；需要切视频时再通过 Homebrew 安装：

```bash
brew install ffmpeg
```

## 创建环境

在项目根目录运行：

```bash
mkdir -p .conda-envs .conda-pkgs
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda create --prefix "$PWD/.conda-envs/eeg-mac-test-train" python=3.10 -y
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python -m pip install --upgrade pip setuptools wheel
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python -m pip install -r requirements-mac-test-train.txt
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python -m ipykernel install --user --name eeg-mac-test-train --display-name "Python (eeg-mac-test-train)"
```

激活环境：

```bash
conda activate "$PWD/.conda-envs/eeg-mac-test-train"
```

退出环境：

```bash
conda deactivate
```

## 验证环境

在项目根目录运行：

```bash
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python -c "import platform, torch; print('python ok'); print('machine:', platform.machine()); print('torch:', torch.__version__); print('mps built:', torch.backends.mps.is_built()); print('mps available:', torch.backends.mps.is_available())"
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python -m compileall -q -x '(^|/)(\.venv|\.venv_debug|\.venv_jupyter|\.conda-envs|\.conda-pkgs)(/|$)' EEG-Conformer eeg-data-processing eeg_project_paths.py
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python -m pytest EEG-Conformer/tests/ eeg-data-processing/tests/ -q
```

如果只想快速确认训练脚本能启动，可以先跑帮助命令：

```bash
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python EEG-Conformer/train_activity_loso.py --help
```

## Mac 小规模训练建议

Mac 环境主要用于检查数据链路、跑单元测试、做短 epoch smoke test，不建议替代 CUDA 服务器跑完整实验。

建议先使用 `cpu` 验证最小链路：

```bash
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda run --prefix "$PWD/.conda-envs/eeg-mac-test-train" python EEG-Conformer/train_activity_loso.py \
  --dataset-root local_artifacts/data_to_list/global_activity_dataset \
  --output-dir local_artifacts/outputs/activity_loso_mac_smoke \
  --epochs 1 \
  --batch-size 8 \
  --device cpu
```

如果 `mps available: True`，可以在小样本上尝试：

```bash
--device mps
```

如果遇到 MPS 算子不支持或数值问题，回退到 `--device cpu`。正式训练仍建议放到 CUDA x86 服务器。

## 与 CUDA 服务器环境的关系

代码、数据格式和 `local_artifacts/` 目录结构可以在 Mac 和服务器之间保持一致。需要分开的只有运行环境：

- Mac：使用本文档的 ARM 测试训练环境。
- 服务器：使用 CUDA x86 环境，例如 `EEG-Conformer/setup_env_jupyter.py` 创建的 Jupyter kernel。
- 不要同步 `.conda-envs/`、`.conda-pkgs/`、`.venv/`、`site-packages/`。
- 数据和输出继续通过 `local_artifacts/`、zip 或定时同步工具传输。