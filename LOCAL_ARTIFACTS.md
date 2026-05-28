# Local Artifacts 管理说明

本项目将不适合通过 GitHub 同步的数据、训练输出和缓存统一放在根目录 `local_artifacts/` 下。该目录已加入 `.gitignore`，适合压缩成 zip 或用定时任务在本机和训练服务器之间同步。

## 目录结构

```text
local_artifacts/
├── data_to_list/
│   ├── data/
│   ├── list/
│   ├── list_cut_fixed_duration/
│   ├── list_normalization_fixed_duration/
│   ├── npy_dataset/
│   └── global_activity_dataset/
├── data_to_sorted/
│   └── sort2/
│       ├── time_data/
│       └── data_cut*/
├── outputs/
│   ├── activity_loso/
│   ├── activity_final/
│   └── activity_api_cache/
└── videos/
```

## 旧路径映射

| 旧路径 | 新路径 |
| --- | --- |
| `EEG-Conformer/outputs/` | `local_artifacts/outputs/` |
| `eeg-data-processing/data_to_list/data/` | `local_artifacts/data_to_list/data/` |
| `eeg-data-processing/data_to_list/list/` | `local_artifacts/data_to_list/list/` |
| `eeg-data-processing/data_to_list/list_cut_fixed_duration/` | `local_artifacts/data_to_list/list_cut_fixed_duration/` |
| `eeg-data-processing/data_to_list/list_normalization_fixed_duration/` | `local_artifacts/data_to_list/list_normalization_fixed_duration/` |
| `eeg-data-processing/data_to_list/npy_dataset*/` | `local_artifacts/data_to_list/npy_dataset*/` |
| `eeg-data-processing/data_to_list/global_activity_dataset*/` | `local_artifacts/data_to_list/global_activity_dataset*/` |
| `eeg-data-processing/data_to_sorted/sort2/time_data/` | `local_artifacts/data_to_sorted/sort2/time_data/` |
| `eeg-data-processing/data_to_sorted/sort2/data_cut*/` | `local_artifacts/data_to_sorted/sort2/data_cut*/` |
| `eeg-data-processing/videos/` | `local_artifacts/videos/` |

## 环境变量覆盖

默认情况下，脚本会自动使用项目根目录下的 `local_artifacts/`。训练服务器如果把数据解压到其他位置，只需要设置：

```bash
export EEG_LOCAL_ARTIFACTS_ROOT=/path/to/local_artifacts
```

也可以按目录单独覆盖，例如：

```bash
export EEG_GLOBAL_ACTIVITY_DATASET_DIR=/path/to/global_activity_dataset
export EEG_ACTIVITY_LOSO_OUTPUT_DIR=/path/to/activity_loso
export EEG_ACTIVITY_FINAL_OUTPUT_DIR=/path/to/activity_final
```

## 同步建议

本地打包：

```bash
zip -r local_artifacts.zip local_artifacts
```

服务器解压后保持目录名为 `local_artifacts/`，或设置 `EEG_LOCAL_ARTIFACTS_ROOT` 指向解压目录。代码和文档通过 GitHub 同步，`local_artifacts/` 通过 zip 或定时任务同步。
