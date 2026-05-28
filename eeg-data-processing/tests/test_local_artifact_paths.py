from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


EEG_ROOT = Path(__file__).resolve().parents[2]
DATA_TO_LIST_ROOT = EEG_ROOT / "eeg-data-processing" / "data_to_list"
SORT2_ROOT = EEG_ROOT / "eeg-data-processing" / "data_to_sorted" / "sort2"
LOCAL_ARTIFACTS_ROOT = EEG_ROOT / "local_artifacts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LocalArtifactPathDefaultsTests(unittest.TestCase):
    def test_environment_overrides_artifact_root_and_specific_paths(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EEG_LOCAL_ARTIFACTS_ROOT": "/tmp/eeg-artifacts-smoke",
                "EEG_ACTIVITY_LOSO_OUTPUT_DIR": "/tmp/custom-loso-output",
            },
        ):
            module = load_module("eeg_project_paths_env", EEG_ROOT / "eeg_project_paths.py")

        self.assertEqual(
            module.GLOBAL_ACTIVITY_DATASET_DIR,
            Path("/tmp/eeg-artifacts-smoke/data_to_list/global_activity_dataset"),
        )
        self.assertEqual(module.ACTIVITY_LOSO_OUTPUT_DIR, Path("/tmp/custom-loso-output"))

    def test_log_to_list_defaults_use_local_artifacts(self) -> None:
        module = load_module("log_to_list", DATA_TO_LIST_ROOT / "1.log_to_list.py")
        args = module.parse_args([])

        self.assertEqual(args.input_dir, LOCAL_ARTIFACTS_ROOT / "data_to_list" / "data")
        self.assertEqual(args.output_dir, LOCAL_ARTIFACTS_ROOT / "data_to_list" / "list")

    def test_edf_to_list_defaults_use_local_artifacts(self) -> None:
        module = load_module("edf_to_list", DATA_TO_LIST_ROOT / "2.edf_to_list.py")
        args = module.parse_args([])

        self.assertEqual(args.input_dir, LOCAL_ARTIFACTS_ROOT / "data_to_list" / "data")
        self.assertEqual(args.output_dir, LOCAL_ARTIFACTS_ROOT / "data_to_list" / "list")

    def test_sort2_defaults_use_local_artifacts(self) -> None:
        module = load_module("cut_by_feedback_time", SORT2_ROOT / "cut_by_feedback_time.py")

        self.assertEqual(
            module.TIME_DATA_DIR,
            LOCAL_ARTIFACTS_ROOT / "data_to_sorted" / "sort2" / "time_data",
        )
        self.assertEqual(
            module.INPUT_ROOT,
            LOCAL_ARTIFACTS_ROOT / "data_to_list" / "list_cut_fixed_duration",
        )
        self.assertEqual(module.VIDEOS_ROOT, LOCAL_ARTIFACTS_ROOT / "videos")


if __name__ == "__main__":
    unittest.main()
