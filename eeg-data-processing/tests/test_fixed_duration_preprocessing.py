from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "data_to_list"


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedDurationPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="fixed-duration-preprocessing-")
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.output_dir = self.root / "output"
        self.time_path = self.root / "time_data.txt"
        self.module = load_script("fixed_duration_preprocessing", "3.time_edit_fixed_duration.py")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, text: str) -> None:
        self.time_path.write_text(text, encoding="utf-8")

    def write_eeg(self, subject_id: int) -> None:
        times = pd.date_range("2026-10-09 15:00:00", periods=2000, freq="s")
        pd.DataFrame({
            "EEG Test-REF": range(len(times)),
            "Time": times.strftime("%H:%M:%S.%f").str[:-3],
        }).to_excel(self.input_dir / f"{subject_id}_e.xlsx", index=False)

    def run_selected(self, *subject_ids: int, overwrite: bool = False) -> None:
        argv = [
            "--time-data-path", str(self.time_path),
            "--input-dir", str(self.input_dir),
            "--output-dir", str(self.output_dir),
            "--subject-ids", *map(str, subject_ids),
        ]
        if overwrite:
            argv.append("--overwrite")
        self.module.main(argv)

    def test_defaults_keep_existing_artifact_paths(self) -> None:
        args = self.module.parse_args([])
        self.assertEqual(args.time_data_path, self.module.TIME_DATA_PATH)
        self.assertEqual(args.input_dir, self.module.LIST_DIR)
        self.assertEqual(args.output_dir, self.module.OUTPUT_DIR)
        self.assertIsNone(args.subject_ids)
        self.assertFalse(args.overwrite)

    def test_reads_legacy_bare_starts_colon_header_and_control_characters(self) -> None:
        self.write_config(
            "\ufeff1\n15:00:00-15:05:07\n15:06:00-15:11:20\n15:12:00-15:28:50\n\n"
            "15\n15:00:00\x03\n15:06:00\n15:12:00\n\n"
            "17:\n15:00:00\n15:06:00\n15:12:00\n15:30:00\n15:36:00\x03\n15:42:00\n"
        )
        starts = self.module.read_time_starts(self.time_path)
        self.assertEqual(starts["1"], ["15:00:00", "15:06:00", "15:12:00"])
        self.assertEqual(starts["15"], starts["1"])
        self.assertEqual(starts["17"], [*starts["1"], "15:30:00", "15:36:00", "15:42:00"])

    def test_fixed_durations_use_only_first_three_and_ignore_legacy_end(self) -> None:
        self.write_config("17:\n15:00:00-15:07:00\n15:06:00\n15:12:00\n15:30:00\n15:36:00\n15:42:00\n")
        starts = self.module.read_time_starts(self.time_path)
        self.assertEqual(self.module.build_fixed_ranges("17", starts["17"]), [
            ("15:00:00", "15:05:00"),
            ("15:06:00", "15:11:18"),
            ("15:12:00", "15:28:38"),
        ])

    def test_invalid_start_fails_instead_of_renumbering_remaining_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid start time"):
            self.module.build_fixed_ranges("15", ["invalid", "15:06:00", "15:12:00"])

    def test_missing_third_start_fails_instead_of_creating_partial_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "e_1/e_2/e_3"):
            self.module.build_fixed_ranges("12", ["15:00:00", "15:06:00"])

    def test_duplicate_subject_header_rejected_with_line_number(self) -> None:
        self.write_config("17\n15:00:00\n17:\n15:06:00\n")
        with self.assertRaisesRegex(ValueError, r"time_data.txt:3: duplicate subject 17"):
            self.module.read_time_starts(self.time_path)

    def test_malformed_time_record_rejected_with_line_number(self) -> None:
        self.write_config("15\n25:00:00\n15:06:00\n15:12:00\n")
        with self.assertRaisesRegex(ValueError, r"time_data.txt:2: invalid time record"):
            self.module.read_time_starts(self.time_path)

    def test_time_record_without_subject_header_is_rejected(self) -> None:
        self.write_config("15:00:00\n")
        with self.assertRaisesRegex(ValueError, "no subject header"):
            self.module.read_time_starts(self.time_path)

    def test_overlapping_or_midnight_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.module.build_fixed_ranges("12", ["15:00:00", "15:04:00", "15:12:00"])
        with self.assertRaisesRegex(ValueError, "midnight"):
            self.module.build_fixed_ranges("12", ["23:58:00", "00:06:00", "00:12:00"])

    def test_s15_control_character_keeps_original_class_order(self) -> None:
        self.write_config("15\n15:00:00\x03\n15:06:00\n15:12:00\n")
        self.write_eeg(15)
        self.run_selected(15)
        expected = [(1, "15:00:00.000", "15:05:00.000", 301),
                    (2, "15:06:00.000", "15:11:18.000", 319),
                    (3, "15:12:00.000", "15:28:38.000", 999)]
        for index, start, end, count in expected:
            frame = pd.read_excel(self.output_dir / "15" / f"15_e_{index}.xlsx")
            self.assertEqual(frame["Time"].iloc[0], start)
            self.assertEqual(frame["Time"].iloc[-1], end)
            self.assertEqual(len(frame), count)

    def test_selected_subject_17_writes_exactly_three_experiments(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n\n"
                          "17:\n15:00:00\n15:06:00\n15:12:00\n15:30:00\n15:36:00\n15:42:00\n")
        self.write_eeg(17)
        self.run_selected(17)
        self.assertEqual(sorted(path.name for path in (self.output_dir / "17").iterdir()),
                         ["17_e_1.xlsx", "17_e_2.xlsx", "17_e_3.xlsx"])
        self.assertFalse((self.output_dir / "12").exists())

    def test_all_selected_configurations_are_checked_before_any_output(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n\n13\n15:00:00\n15:06:00\n")
        self.write_eeg(12)
        self.write_eeg(13)
        with self.assertRaisesRegex(ValueError, "Dataset 13"):
            self.run_selected(12, 13)
        self.assertFalse(self.output_dir.exists())

    def test_requested_missing_subject_is_rejected(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        with self.assertRaisesRegex(ValueError, "missing from time configuration"):
            self.run_selected(17)
        self.assertFalse(self.output_dir.exists())

    def test_requested_missing_eeg_excel_is_rejected(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        with self.assertRaisesRegex(FileNotFoundError, "Missing EEG input"):
            self.run_selected(12)
        self.assertFalse(self.output_dir.exists())

    def test_eeg_only_completed_subject_skips_and_overwrite_rebuilds(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        self.write_eeg(12)
        self.run_selected(12)
        paths = sorted((self.output_dir / "12").glob("*.xlsx"))
        before = [path.stat().st_mtime_ns for path in paths]
        with patch.object(self.module, "slice_excel_file") as cutter:
            self.run_selected(12)
            cutter.assert_not_called()
        self.assertEqual(before, [path.stat().st_mtime_ns for path in paths])
        paths[0].write_bytes(b"stale existing output")
        self.run_selected(12, overwrite=True)
        self.assertEqual(len(pd.read_excel(paths[0])), 301)

    def test_invalid_excel_time_fails_before_any_labelled_output(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        pd.DataFrame({"EEG Test": [1, 2], "Time": ["15:00:00", "invalid"]}).to_excel(
            self.input_dir / "12_e.xlsx", index=False)
        with self.assertRaisesRegex(ValueError, "invalid time column"):
            self.run_selected(12)
        self.assertEqual(list(self.output_dir.rglob("*.xlsx")), [])

    def test_absent_later_time_range_does_not_leave_first_class_output(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        pd.DataFrame({"EEG Test": [1, 2], "Time": ["15:00:00", "15:05:00"]}).to_excel(
            self.input_dir / "12_e.xlsx", index=False)
        with self.assertRaisesRegex(ValueError, "does not cover full range #2"):
            self.run_selected(12)
        self.assertEqual(list(self.output_dir.rglob("*.xlsx")), [])

    def test_partial_last_experiment_is_rejected_before_first_output(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        times = pd.date_range("2026-10-09 15:00:00", periods=1650, freq="s")
        pd.DataFrame({"EEG Test": range(len(times)), "Time": times.strftime("%H:%M:%S")}).to_excel(
            self.input_dir / "12_e.xlsx", index=False)
        with self.assertRaisesRegex(ValueError, "does not cover full range #3"):
            self.run_selected(12)
        self.assertEqual(list(self.output_dir.rglob("*.xlsx")), [])

    def test_zero_byte_or_damaged_cut_outputs_are_rebuilt(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        self.write_eeg(12)
        subject_output = self.output_dir / "12"
        subject_output.mkdir(parents=True)
        (subject_output / "12_e_1.xlsx").touch()
        (subject_output / "12_e_2.xlsx").write_bytes(b"interrupted Excel")
        self.run_selected(12)
        self.assertEqual(len(pd.read_excel(subject_output / "12_e_1.xlsx")), 301)
        self.assertEqual(len(pd.read_excel(subject_output / "12_e_2.xlsx")), 319)

    def test_cut_write_failure_preserves_existing_output_and_cleans_temporary(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        self.write_eeg(12)
        self.run_selected(12)
        existing_path = self.output_dir / "12" / "12_e_1.xlsx"
        existing_bytes = existing_path.read_bytes()

        def interrupted_write(frame, path, **kwargs):
            Path(path).write_bytes(b"interrupted write")
            raise OSError("simulated write failure")

        with patch.object(pd.DataFrame, "to_excel", interrupted_write):
            with self.assertRaisesRegex(OSError, "simulated write failure"):
                self.run_selected(12, overwrite=True)
        self.assertEqual(existing_path.read_bytes(), existing_bytes)
        self.assertEqual(list(self.output_dir.rglob(".*.tmp.xlsx")), [])

    def test_cut_keyboard_interrupt_cleans_temporary_and_preserves_existing_output(self) -> None:
        self.write_config("12\n15:00:00\n15:06:00\n15:12:00\n")
        self.write_eeg(12)
        self.run_selected(12)
        existing_path = self.output_dir / "12" / "12_e_1.xlsx"
        existing_bytes = existing_path.read_bytes()

        def interrupted_write(frame, path, **kwargs):
            Path(path).write_bytes(b"interrupted write")
            raise KeyboardInterrupt()

        with patch.object(pd.DataFrame, "to_excel", interrupted_write):
            with self.assertRaises(KeyboardInterrupt):
                self.run_selected(12, overwrite=True)
        self.assertEqual(existing_path.read_bytes(), existing_bytes)
        self.assertEqual(list(self.output_dir.rglob(".*.tmp.xlsx")), [])


class EdFSubjectSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="edf-subject-selection-")
        self.root = Path(self.temporary.name)
        self.module = load_script("edf_subject_selection", "2.edf_to_list.py")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_requested_subjects_and_keeps_all_files_default(self) -> None:
        for name in ("1_e.edf", "12_e.edf", "17_e.EDF", "other.edf", "12.log"):
            (self.root / name).touch()
        self.assertEqual([path.name for path in self.module.select_edf_files(self.root, [12, 17])],
                         ["12_e.edf", "17_e.EDF"])
        self.assertEqual(len(self.module.select_edf_files(self.root)), 4)

    def test_missing_requested_edf_fails_before_any_conversion(self) -> None:
        (self.root / "12_e.edf").touch()
        output = self.root / "output"
        with patch.object(self.module, "convert_edf_to_excel") as converter:
            with self.assertRaisesRegex(FileNotFoundError, "17"):
                self.module.main(["--input-dir", str(self.root), "--output-dir", str(output),
                                  "--subject-ids", "12", "17"])
            converter.assert_not_called()
        self.assertFalse(output.exists())

    def test_selection_does_not_convert_unrequested_or_existing_files(self) -> None:
        for name in ("1_e.edf", "12_e.edf", "17_e.edf"):
            (self.root / name).touch()
        output = self.root / "output"
        output.mkdir()
        pd.DataFrame({"EEG Test": [1], "Time": ["15:00:00"]}).to_excel(output / "12_e.xlsx", index=False)
        with patch.object(self.module, "convert_edf_to_excel", return_value=True) as converter:
            self.module.main(["--input-dir", str(self.root), "--output-dir", str(output),
                              "--subject-ids", "12", "17"])
        converter.assert_called_once_with(self.root / "17_e.edf", output)

    def test_nonpositive_subject_id_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.module.parse_args(["--subject-ids", "0"])

    def test_zero_byte_and_damaged_existing_excel_are_not_skipped(self) -> None:
        (self.root / "12_e.xlsx").touch()
        (self.root / "13_e.xlsx").write_bytes(b"interrupted Excel archive")
        pd.DataFrame({"Time": ["15:00:00"]}).to_excel(self.root / "17_e.xlsx", index=False)
        self.assertEqual(self.module.discover_processed_files(self.root), {"17_e"})

    def test_conversion_failures_make_main_fail(self) -> None:
        (self.root / "12_e.edf").touch()
        with patch.object(self.module, "convert_edf_to_excel", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Failed to convert 1"):
                self.module.main(["--input-dir", str(self.root), "--output-dir", str(self.root / "output"),
                                  "--subject-ids", "12"])

    def fake_mne(self):
        raw = SimpleNamespace(
            get_data=lambda: np.array([[1.0, 2.0]]),
            ch_names=["EEG Test-REF"],
            info={"sfreq": 128.0, "meas_date": datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)},
        )
        return SimpleNamespace(io=SimpleNamespace(read_raw_edf=Mock(return_value=raw)))

    def test_successful_conversion_publishes_readable_excel_without_temporary_file(self) -> None:
        output = self.root / "output"
        with patch.dict("sys.modules", {"mne": self.fake_mne()}):
            self.assertTrue(self.module.convert_edf_to_excel(self.root / "12_e.edf", output))
        frame = pd.read_excel(output / "12_e.xlsx")
        self.assertEqual(frame["EEG Test-REF"].tolist(), [1.0, 2.0])
        self.assertEqual(list(output.glob(".*.tmp.xlsx")), [])

    def test_interrupted_conversion_preserves_existing_excel_and_cleans_temporary_file(self) -> None:
        output = self.root / "output"
        output.mkdir()
        existing_path = output / "12_e.xlsx"
        pd.DataFrame({"old": [99]}).to_excel(existing_path, index=False)
        existing_bytes = existing_path.read_bytes()

        def interrupted_write(frame, path, **kwargs):
            Path(path).write_bytes(b"interrupted write")
            raise OSError("simulated write failure")

        with patch.dict("sys.modules", {"mne": self.fake_mne()}):
            with patch.object(pd.DataFrame, "to_excel", interrupted_write):
                self.assertFalse(self.module.convert_edf_to_excel(self.root / "12_e.edf", output))
        self.assertEqual(existing_path.read_bytes(), existing_bytes)
        self.assertEqual(list(output.glob(".*.tmp.xlsx")), [])

    def test_keyboard_interrupt_during_conversion_cleans_temporary_and_preserves_previous_excel(self) -> None:
        output = self.root / "output"
        output.mkdir()
        existing_path = output / "12_e.xlsx"
        pd.DataFrame({"old": [99]}).to_excel(existing_path, index=False)
        existing_bytes = existing_path.read_bytes()

        def interrupted_write(frame, path, **kwargs):
            Path(path).write_bytes(b"interrupted write")
            raise KeyboardInterrupt()

        with patch.dict("sys.modules", {"mne": self.fake_mne()}):
            with patch.object(pd.DataFrame, "to_excel", interrupted_write):
                with self.assertRaises(KeyboardInterrupt):
                    self.module.convert_edf_to_excel(self.root / "12_e.edf", output)
        self.assertEqual(existing_path.read_bytes(), existing_bytes)
        self.assertEqual(list(output.glob(".*.tmp.xlsx")), [])


if __name__ == "__main__":
    unittest.main()
