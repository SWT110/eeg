from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "data_to_list" / "6.edf_list_normalization_fixed_duration.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fixed_duration_normalization", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedDurationNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="fixed-duration-normalization-")
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.output_dir = self.root / "output"
        self.module = load_module()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_path(self, subject_id: int, index: int) -> Path:
        return self.input_dir / str(subject_id) / f"{subject_id}_e_{index}.xlsx"

    def output_path(self, subject_id: int, index: int) -> Path:
        return self.output_dir / str(subject_id) / f"{subject_id}_e_{index}.xlsx"

    def write_subject(self, subject_id: int) -> None:
        directory = self.input_dir / str(subject_id)
        directory.mkdir()
        for index in (1, 2, 3):
            pd.DataFrame({
                "EEG Test-REF": np.array([1.0, 3.0, 8.0]) + 100 * index,
                "ECG Test": [5.0, 6.0, 9.0],
                "EMG Test": [1.0, 7.0, 9.0],
                "SaO2 Test": [90.0, 95.0, 100.0],
                "Unchanged": [12, 13, 14],
                "Time": ["15:00:00.000", "15:00:00.008", "15:00:00.016"],
            }).to_excel(self.source_path(subject_id, index), index=False)

    def run_selected(self, *subject_ids: int) -> None:
        self.module.main([
            "--input-dir", str(self.input_dir),
            "--output-dir", str(self.output_dir),
            "--subject-ids", *map(str, subject_ids),
        ])

    def test_default_paths_and_mtime_behavior_remain_available(self) -> None:
        args = self.module.parse_args([])
        self.assertEqual(args.input_dir, self.module.SOURCE_DIR)
        self.assertEqual(args.output_dir, self.module.TARGET_DIR)
        self.assertIsNone(args.subject_ids)

    def test_selected_new_subjects_preserve_old_subject_and_only_use_three_records(self) -> None:
        self.write_subject(12)
        self.write_subject(17)
        old_source = self.input_dir / "3" / "3_e_3.xlsx"
        old_source.parent.mkdir()
        old_source.write_bytes(b"old shorter source must not be read")
        old_output = self.output_path(3, 3)
        old_output.parent.mkdir(parents=True)
        old_output.write_bytes(b"original standardized S3 record")
        old_timestamp = old_output.stat().st_mtime_ns
        extra_source = self.source_path(17, 4)
        extra_source.write_bytes(b"unused later experiment")
        self.run_selected(12, 17)
        self.assertEqual(old_output.read_bytes(), b"original standardized S3 record")
        self.assertEqual(old_output.stat().st_mtime_ns, old_timestamp)
        self.assertFalse(self.output_path(17, 4).exists())
        for subject_id in (12, 17):
            for index in (1, 2, 3):
                frame = pd.read_excel(self.output_path(subject_id, index))
                for column in ("EEG Test-REF", "ECG Test", "EMG Test", "SaO2 Test"):
                    self.assertAlmostEqual(float(frame[column].mean()), 0.0, places=12)
                    self.assertAlmostEqual(float(frame[column].std(ddof=0)), 1.0, places=12)
                self.assertEqual(frame["Unchanged"].tolist(), [12, 13, 14])
                self.assertEqual(frame["Time"].tolist(), ["15:00:00.000", "15:00:00.008", "15:00:00.016"])

    def test_missing_selected_record_fails_before_output_creation(self) -> None:
        self.write_subject(12)
        self.write_subject(17)
        self.source_path(17, 3).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "17_e_3.xlsx"):
            self.run_selected(12, 17)
        self.assertFalse(self.output_dir.exists())

    def test_current_valid_target_is_skipped_by_mtime(self) -> None:
        self.write_subject(12)
        self.run_selected(12)
        targets = [self.output_path(12, index) for index in (1, 2, 3)]
        before_bytes = [path.read_bytes() for path in targets]
        before_timestamps = [path.stat().st_mtime_ns for path in targets]
        with patch.object(pd, "read_excel", side_effect=AssertionError("current target should skip reading")):
            self.run_selected(12)
        self.assertEqual([path.read_bytes() for path in targets], before_bytes)
        self.assertEqual([path.stat().st_mtime_ns for path in targets], before_timestamps)

    def test_stale_target_is_rebuilt(self) -> None:
        self.write_subject(12)
        target = self.output_path(12, 1)
        target.parent.mkdir(parents=True)
        pd.DataFrame({"old": [99]}).to_excel(target, index=False)
        os.utime(target, (0, 0))
        self.run_selected(12)
        frame = pd.read_excel(target)
        self.assertIn("EEG Test-REF", frame)
        self.assertAlmostEqual(float(frame["EEG Test-REF"].mean()), 0.0, places=12)

    def test_zero_byte_and_damaged_targets_are_rebuilt_even_with_newer_mtime(self) -> None:
        self.write_subject(12)
        target1, target2 = self.output_path(12, 1), self.output_path(12, 2)
        target1.parent.mkdir(parents=True)
        target1.touch()
        target2.write_bytes(b"interrupted nonzero target")
        self.run_selected(12)
        self.assertIn("EEG Test-REF", pd.read_excel(target1))
        self.assertIn("EEG Test-REF", pd.read_excel(target2))

    def test_invalid_signal_values_produce_failure_exit_and_no_target(self) -> None:
        self.write_subject(12)
        source = self.source_path(12, 1)
        frame = pd.read_excel(source)
        frame.loc[1, "EEG Test-REF"] = np.nan
        frame.to_excel(source, index=False)
        with self.assertRaisesRegex(RuntimeError, "Failed to normalize 1"):
            self.run_selected(12)
        self.assertFalse(self.output_path(12, 1).exists())

    def test_malformed_source_produces_failure_exit(self) -> None:
        self.write_subject(12)
        self.source_path(12, 1).write_bytes(b"broken source workbook")
        with self.assertRaisesRegex(RuntimeError, "Failed to normalize 1"):
            self.run_selected(12)
        self.assertFalse(self.output_path(12, 1).exists())

    def test_missing_signal_columns_produce_failure_exit(self) -> None:
        self.write_subject(12)
        pd.DataFrame({"Time": ["15:00:00"]}).to_excel(self.source_path(12, 1), index=False)
        with self.assertRaisesRegex(RuntimeError, "Failed to normalize 1"):
            self.run_selected(12)
        self.assertFalse(self.output_path(12, 1).exists())

    def test_write_error_or_keyboard_interrupt_cleans_temp_and_preserves_previous_output(self) -> None:
        self.write_subject(12)
        target = self.output_path(12, 1)
        target.parent.mkdir(parents=True)
        pd.DataFrame({"old": [99]}).to_excel(target, index=False)
        os.utime(target, (0, 0))
        original = target.read_bytes()
        for failure in (OSError("simulated disk failure"), KeyboardInterrupt()):
            with self.subTest(exception=type(failure).__name__):
                def interrupted_write(frame, path, **kwargs):
                    Path(path).write_bytes(b"incomplete temporary workbook")
                    raise failure

                with patch.object(pd.DataFrame, "to_excel", interrupted_write):
                    if isinstance(failure, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            self.run_selected(12)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "Failed to normalize 3"):
                            self.run_selected(12)
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(list(self.output_dir.rglob(".*.tmp.xlsx")), [])

    def test_no_selection_keeps_default_recursive_scan(self) -> None:
        self.write_subject(12)
        self.module.main(["--input-dir", str(self.input_dir), "--output-dir", str(self.output_dir)])
        self.assertEqual(len(list(self.output_dir.rglob("*.xlsx"))), 3)


if __name__ == "__main__":
    unittest.main()
