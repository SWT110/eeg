"""Central path configuration for EEG project code and local artifacts."""

from __future__ import annotations

import os
from pathlib import Path


EEG_ROOT = Path(__file__).resolve().parent
CONFORMER_ROOT = EEG_ROOT / "EEG-Conformer"
DATA_PROCESSING_ROOT = EEG_ROOT / "eeg-data-processing"
DATA_TO_LIST_ROOT = DATA_PROCESSING_ROOT / "data_to_list"
SORT2_ROOT = DATA_PROCESSING_ROOT / "data_to_sorted" / "sort2"


def _path_from_env(name: str, default: Path) -> Path:
    raw_value = os.environ.get(name)
    if raw_value:
        return Path(raw_value).expanduser()
    return default.expanduser()


LOCAL_ARTIFACTS_ROOT = _path_from_env(
    "EEG_LOCAL_ARTIFACTS_ROOT",
    EEG_ROOT / "local_artifacts",
)
DATA_TO_LIST_ARTIFACTS_ROOT = LOCAL_ARTIFACTS_ROOT / "data_to_list"
OUTPUTS_ROOT = LOCAL_ARTIFACTS_ROOT / "outputs"
SORT2_ARTIFACTS_ROOT = LOCAL_ARTIFACTS_ROOT / "data_to_sorted" / "sort2"

DATA_TO_LIST_DATA_DIR = _path_from_env(
    "EEG_DATA_TO_LIST_DATA_DIR",
    DATA_TO_LIST_ARTIFACTS_ROOT / "data",
)
LIST_DIR = _path_from_env(
    "EEG_LIST_DIR",
    DATA_TO_LIST_ARTIFACTS_ROOT / "list",
)
LIST_CUT_DIR = _path_from_env(
    "EEG_LIST_CUT_DIR",
    DATA_TO_LIST_ARTIFACTS_ROOT / "list_cut_fixed_duration",
)
LIST_NORMALIZED_DIR = _path_from_env(
    "EEG_LIST_NORMALIZED_DIR",
    DATA_TO_LIST_ARTIFACTS_ROOT / "list_normalization_fixed_duration",
)
NPY_DATASET_DIR = _path_from_env(
    "EEG_NPY_DATASET_DIR",
    DATA_TO_LIST_ARTIFACTS_ROOT / "npy_dataset",
)
GLOBAL_ACTIVITY_DATASET_DIR = _path_from_env(
    "EEG_GLOBAL_ACTIVITY_DATASET_DIR",
    DATA_TO_LIST_ARTIFACTS_ROOT / "global_activity_dataset",
)
WINDOW_STRIDE_CONFIG = _path_from_env(
    "EEG_WINDOW_STRIDE_CONFIG",
    DATA_TO_LIST_ROOT / "window_stride_configs.json",
)

ACTIVITY_LOSO_OUTPUT_DIR = _path_from_env(
    "EEG_ACTIVITY_LOSO_OUTPUT_DIR",
    OUTPUTS_ROOT / "activity_loso",
)
ACTIVITY_FINAL_OUTPUT_DIR = _path_from_env(
    "EEG_ACTIVITY_FINAL_OUTPUT_DIR",
    OUTPUTS_ROOT / "activity_final",
)
ACTIVITY_API_CACHE_DIR = _path_from_env(
    "EEG_ACTIVITY_API_CACHE_DIR",
    OUTPUTS_ROOT / "activity_api_cache",
)

SORT2_TIME_DATA_DIR = _path_from_env(
    "EEG_SORT2_TIME_DATA_DIR",
    SORT2_ARTIFACTS_ROOT / "time_data",
)
SORT2_DATA_CUT_ROOT = _path_from_env(
    "EEG_SORT2_DATA_CUT_ROOT",
    SORT2_ARTIFACTS_ROOT,
)
VIDEOS_DIR = _path_from_env(
    "EEG_VIDEOS_DIR",
    LOCAL_ARTIFACTS_ROOT / "videos",
)


def sort2_data_cut_dir(name: str) -> Path:
    return SORT2_DATA_CUT_ROOT / name
