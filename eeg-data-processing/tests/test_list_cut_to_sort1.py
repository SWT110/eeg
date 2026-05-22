from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "data_to_sorted" / "1.list_cut_to_sort1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("list_cut_to_sort1", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ListCutToSort1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="list-cut-to-sort1-"))
        self.fixed_root = self.temp_dir / "list_cut_fixed_duration"
        self.normalized_root = self.temp_dir / "list_normalization_fixed_duration"
        self.output_root = self.temp_dir / "output"
        self.module = load_module()

        self._write_original_inputs()
        self._write_normalized_inputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_original_inputs(self) -> None:
        subject_dir = self.fixed_root / "1"
        subject_dir.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "Signal": [10, 11, 12, 13, 14],
                "Time": [
                    "00:00:00",
                    "00:00:01",
                    "00:00:02",
                    "00:00:03",
                    "00:00:04",
                ],
            }
        )
        frame.to_excel(subject_dir / "1_1.xlsx", index=False)
        frame.to_excel(subject_dir / "1_e_1.xlsx", index=False)

    def _write_normalized_inputs(self) -> None:
        subject_dir = self.normalized_root / "1"
        subject_dir.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "EEG Fp1-REF": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Time": [
                    "00:00:00",
                    "00:00:01",
                    "00:00:02",
                    "00:00:03",
                    "00:00:04",
                ],
            }
        )
        frame.to_excel(subject_dir / "1_e_1.xlsx", index=False)

    def test_process_all_sources_writes_original_and_normalized_outputs(self) -> None:
        sources = [
            {
                "input_root": self.fixed_root,
                "output_suffix": "",
                "signals": ("ppg", "eeg"),
            },
            {
                "input_root": self.normalized_root,
                "output_suffix": "_normalization",
                "signals": ("eeg",),
            },
        ]

        self.module.process_all_sources(
            segment_seconds=2,
            output_base=self.output_root,
            sources=sources,
        )

        self.assertTrue((self.output_root / "data_cut_2s" / "1" / "ppg" / "1_1_seg001.xlsx").exists())
        self.assertTrue((self.output_root / "data_cut_2s" / "1" / "eeg" / "1_e_1_seg001.xlsx").exists())
        self.assertTrue(
            (self.output_root / "data_cut_2s_normalization" / "1" / "eeg" / "1_e_1_seg001.xlsx").exists()
        )
        self.assertFalse((self.output_root / "data_cut_2s_normalization" / "1" / "ppg").exists())


if __name__ == "__main__":
    unittest.main()
