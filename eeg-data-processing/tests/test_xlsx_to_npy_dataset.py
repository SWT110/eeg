from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "data_to_list" / "8.xlsx_to_npy_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("xlsx_to_npy_dataset", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "M1",
    "M2",
    "Fz",
    "Cz",
    "Pz",
]


class XlsxToNpyDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xlsx-to-npy-"))
        self.input_root = self.temp_dir / "input"
        self.output_root = self.temp_dir / "output"
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_default_paths_use_local_artifacts(self) -> None:
        expected_root = MODULE_PATH.resolve().parents[2] / "local_artifacts" / "data_to_list"
        self.assertEqual(self.module.DEFAULT_INPUT_ROOT, expected_root / "list_normalization_fixed_duration")
        self.assertEqual(self.module.DEFAULT_OUTPUT_ROOT, expected_root / "npy_dataset")

    def _write_test_xlsx(self, subject_id: str, label: str, num_samples: int, 
                         has_nan: bool = False, has_inf: bool = False, invalid_time: bool = False) -> None:
        """Write a test xlsx file with EEG columns and Time column."""
        subject_dir = self.input_root / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Sampling interval of 0.5 seconds
        if invalid_time:
            times = [f"00:00:00.00"] * num_samples  # All same time (invalid)
        else:
            times = [f"00:00:{i*0.5:05.2f}" for i in range(num_samples)]
        
        ch1_data = list(range(num_samples))
        ch2_data = list(range(100, 100 + num_samples))
        
        if has_nan:
            if num_samples >= 2:
                ch1_data[1] = np.nan  # Inject NaN
        
        if has_inf:
            if num_samples >= 2:
                ch1_data[1] = np.inf  # Inject Inf
        
        data = {
            "EEG_Ch1": ch1_data,
            "EEG_Ch2": ch2_data,
            "NonEEG_Col": list(range(200, 200 + num_samples)),  # Should be filtered out
            "Time": times,
        }
        df = pd.DataFrame(data)
        filename = f"{subject_id}_e_{label}.xlsx"
        df.to_excel(subject_dir / filename, index=False)

    def _write_dataframe_xlsx(self, subject_id: str, label: str, df: pd.DataFrame) -> Path:
        subject_dir = self.input_root / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        path = subject_dir / f"{subject_id}_e_{label}.xlsx"
        df.to_excel(path, index=False)
        return path

    def test_basic_export_with_two_files(self) -> None:
        """Test basic export: two xlsx files, 2 EEG cols, 0.5s sampling, 1s window/stride."""
        # Create two files for subject 1: e_1 (4 samples = 2s) and e_2 (6 samples = 3s)
        self._write_test_xlsx("1", "1", 4)  # 2 seconds total, expect 2 windows of 1s each
        self._write_test_xlsx("1", "2", 6)  # 3 seconds total, expect 3 windows of 1s each

        # Run the conversion
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )

        # Check output structure - MUST be subject_<id> not just <id>
        subject_out = self.output_root / "subject_1"
        self.assertTrue(subject_out.exists(), "Output directory should be subject_1 not 1")
        
        X_path = subject_out / "subject_1_X.npy"
        y_path = subject_out / "subject_1_y.npy"
        groups_path = subject_out / "subject_1_groups.npy"
        
        self.assertTrue(X_path.exists())
        self.assertTrue(y_path.exists())
        self.assertTrue(groups_path.exists())

        # Load and check arrays
        X = np.load(X_path)
        y = np.load(y_path)
        groups = np.load(groups_path)

        # Expected: 2 windows from e_1 + 3 windows from e_2 = 5 total windows
        # Each window is 1 second with 0.5s sampling = 2 samples per window
        # 2 EEG channels
        # Shape MUST be (N, C, T) not (N, T, C)
        self.assertEqual(X.shape, (5, 2, 2))  # (5 windows, 2 channels, 2 samples)
        self.assertEqual(y.shape, (5,))
        self.assertEqual(groups.shape, (5,))

        # Check labels: first 2 should be 0 (e_1), next 3 should be 1 (e_2)
        np.testing.assert_array_equal(y[:2], [0, 0])
        np.testing.assert_array_equal(y[2:], [1, 1, 1])

        # Check groups: first 2 should be group 0, next 3 should be group 1
        np.testing.assert_array_equal(groups[:2], [0, 0])
        np.testing.assert_array_equal(groups[2:], [1, 1, 1])
        
        # Verify (N, C, T) shape by checking first window's channel data
        # First window from e_1: samples 0,1 from EEG_Ch1 should be [0, 1]
        # Shape is (2 channels, 2 samples), so X[0, 0, :] is Ch1, X[0, 1, :] is Ch2
        np.testing.assert_array_equal(X[0, 0, :], [0, 1])  # Ch1: [0, 1]
        np.testing.assert_array_equal(X[0, 1, :], [100, 101])  # Ch2: [100, 101]

    def test_stride_defaults_to_window(self) -> None:
        """Test that stride_seconds defaults to window_seconds when None."""
        self._write_test_xlsx("2", "1", 6)  # 3 seconds

        # Call with stride_seconds=None
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=None,
        )

        subject_out = self.output_root / "subject_2"
        X = np.load(subject_out / "subject_2_X.npy")
        
        # 3 seconds / 1 second window = 3 windows (non-overlapping)
        self.assertEqual(X.shape[0], 3)

    def test_non_eeg_columns_filtered_out(self) -> None:
        """Test that columns not starting with 'EEG' are filtered out."""
        subject_dir = self.input_root / "3"
        subject_dir.mkdir(parents=True)
        
        df = pd.DataFrame({
            "EEG_A": [1, 2, 3, 4],
            "EEG_B": [5, 6, 7, 8],
            "Signal": [9, 10, 11, 12],  # Should be filtered
            "PPG": [13, 14, 15, 16],  # Should be filtered
            "Time": ["00:00:00.00", "00:00:00.50", "00:00:01.00", "00:00:01.50"],
        })
        df.to_excel(subject_dir / "3_e_1.xlsx", index=False)

        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )

        X = np.load(self.output_root / "subject_3" / "subject_3_X.npy")
        
        # Shape is (N, C, T): Should have 2 EEG channels
        self.assertEqual(X.shape[1], 2)  # 2 EEG channels
        self.assertEqual(X.shape[2], 2)  # 2 samples per window

    def test_load_xlsx_file_keeps_only_21_common_channels_and_excludes_a1_a2(self) -> None:
        times = ["00:00:00.00", "00:00:00.50", "00:00:01.00", "00:00:01.50"]
        data = {
            f"EEG {name}-REF": [idx * 10 + row for row in range(4)]
            for idx, name in enumerate(COMMON_CHANNELS)
        }
        data["EEG A1-REF"] = [900 + row for row in range(4)]
        data["EEG A2-REF"] = [950 + row for row in range(4)]
        data["ECG EKG-REF"] = [1000 + row for row in range(4)]
        data["Time"] = times

        xlsx_path = self._write_dataframe_xlsx("9", "1", pd.DataFrame(data))
        loaded = self.module.load_xlsx_file(xlsx_path)

        self.assertIsNotNone(loaded)
        eeg_data, parsed_times = loaded
        self.assertEqual(eeg_data.shape, (4, 21))
        np.testing.assert_array_equal(parsed_times, [0.0, 0.5, 1.0, 1.5])
        np.testing.assert_array_equal(
            eeg_data[0],
            np.array([idx * 10 for idx in range(len(COMMON_CHANNELS))], dtype=float),
        )

    def test_incomplete_window_at_tail_is_discarded(self) -> None:
        """Test that incomplete windows at the end are discarded."""
        # Create 5 samples (2.5 seconds) with 0.5s sampling interval
        # With 1.0s window, we should get exactly 2 windows (0-1s, 1-2s)
        # The last 0.5s (sample at 2.0s and 2.5s mark) should be discarded
        self._write_test_xlsx("4", "1", 5)

        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )

        X = np.load(self.output_root / "subject_4" / "subject_4_X.npy")
        
        # Should have only 2 complete windows, not 3
        self.assertEqual(X.shape[0], 2)

    def test_nan_in_eeg_data_raises_error(self) -> None:
        """Test that NaN values in EEG data cause clear error."""
        self._write_test_xlsx("5", "1", 4, has_nan=True)

        # Should print error and skip file (not crash or silently write NaN)
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_5"
        self.assertFalse(subject_out.exists())

    def test_invalid_time_column_raises_error(self) -> None:
        """Test that invalid Time column (all same values, median diff <= 0) raises error."""
        self._write_test_xlsx("6", "1", 4, invalid_time=True)

        # Should skip file with clear error (not divide by zero)
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_6"
        self.assertFalse(subject_out.exists())

    def test_xlsx_with_only_headers_is_skipped(self) -> None:
        """Test that xlsx with only headers (0 data rows) is skipped with clear error."""
        subject_dir = self.input_root / "7"
        subject_dir.mkdir(parents=True)
        
        df = pd.DataFrame({
            "EEG_Ch1": [],
            "EEG_Ch2": [],
            "Time": [],
        })
        df.to_excel(subject_dir / "7_e_1.xlsx", index=False)

        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created
        subject_out = self.output_root / "subject_7"
        self.assertFalse(subject_out.exists())

    def test_xlsx_with_one_data_row_is_skipped(self) -> None:
        """Test that xlsx with only 1 data row is skipped (need >= 2 for diff)."""
        subject_dir = self.input_root / "8"
        subject_dir.mkdir(parents=True)
        
        df = pd.DataFrame({
            "EEG_Ch1": [1.0],
            "EEG_Ch2": [2.0],
            "Time": ["00:00:00.00"],
        })
        df.to_excel(subject_dir / "8_e_1.xlsx", index=False)

        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created
        subject_out = self.output_root / "subject_8"
        self.assertFalse(subject_out.exists())

    def test_window_too_small_relative_to_sampling_interval(self) -> None:
        """Test that window/stride smaller than sampling interval is handled gracefully."""
        subject_dir = self.input_root / "9"
        subject_dir.mkdir(parents=True)
        
        # Create file with 0.5s sampling interval (as per _write_test_xlsx default)
        df = pd.DataFrame({
            "EEG_Ch1": [1.0, 2.0, 3.0, 4.0],
            "EEG_Ch2": [5.0, 6.0, 7.0, 8.0],
            "Time": ["00:00:00.00", "00:00:00.50", "00:00:01.00", "00:00:01.50"],
        })
        df.to_excel(subject_dir / "9_e_1.xlsx", index=False)

        # Try with 0.2s window (smaller than 0.5s sampling interval)
        # This should skip the file with clear error, not hang or create empty output
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=0.2,
            stride_seconds=0.5,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_9"
        self.assertFalse(subject_out.exists())

    def test_stride_too_small_relative_to_sampling_interval(self) -> None:
        """Test that stride smaller than sampling interval is handled gracefully."""
        subject_dir = self.input_root / "10"
        subject_dir.mkdir(parents=True)
        
        # Create file with 0.5s sampling interval
        df = pd.DataFrame({
            "EEG_Ch1": [1.0, 2.0, 3.0, 4.0],
            "EEG_Ch2": [5.0, 6.0, 7.0, 8.0],
            "Time": ["00:00:00.00", "00:00:00.50", "00:00:01.00", "00:00:01.50"],
        })
        df.to_excel(subject_dir / "10_e_1.xlsx", index=False)

        # Try with 1.0s window but 0.2s stride (stride smaller than 0.5s sampling)
        # This should skip the file with clear error, not hang
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=0.2,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_10"
        self.assertFalse(subject_out.exists())

    def test_numeric_time_column_in_seconds(self) -> None:
        """Test that numeric Time column (in seconds) is handled correctly."""
        subject_dir = self.input_root / "11"
        subject_dir.mkdir(parents=True)
        
        # Create file with numeric Time column in seconds (0.5s sampling interval)
        # This mimics the real-world scenario where Time is already numeric seconds
        df = pd.DataFrame({
            "EEG_Ch1": [1.0, 2.0, 3.0, 4.0],
            "EEG_Ch2": [5.0, 6.0, 7.0, 8.0],
            "Time": [0.0, 0.5, 1.0, 1.5],  # Numeric seconds, not string timestamps
        })
        df.to_excel(subject_dir / "11_e_1.xlsx", index=False)

        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        subject_out = self.output_root / "subject_11"
        self.assertTrue(subject_out.exists())
        
        X = np.load(subject_out / "subject_11_X.npy")
        y = np.load(subject_out / "subject_11_y.npy")
        
        # Should extract 2 windows: [0.0, 0.5] and [1.0, 1.5]
        self.assertEqual(X.shape[0], 2)  # 2 windows
        self.assertEqual(X.shape[1], 2)  # 2 channels
        self.assertEqual(X.shape[2], 2)  # 2 samples per window
        
        # Verify first window data
        np.testing.assert_array_equal(X[0, 0, :], [1.0, 2.0])  # Ch1: [1.0, 2.0]
        np.testing.assert_array_equal(X[0, 1, :], [5.0, 6.0])  # Ch2: [5.0, 6.0]

    def test_numeric_time_column_with_nan(self) -> None:
        """Test that numeric Time column with NaN is handled gracefully."""
        subject_dir = self.input_root / "12"
        subject_dir.mkdir(parents=True)
        
        # Create file with numeric Time column containing NaN
        df = pd.DataFrame({
            "EEG_Ch1": [1.0, 2.0, 3.0, 4.0],
            "EEG_Ch2": [5.0, 6.0, 7.0, 8.0],
            "Time": [0.0, np.nan, 1.0, 1.5],  # NaN in Time column
        })
        df.to_excel(subject_dir / "12_e_1.xlsx", index=False)

        # Should skip file with clear error about unparseable Time values
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_12"
        self.assertFalse(subject_out.exists())

    def test_millisecond_rounded_128hz_timestamps_still_yield_128_sample_windows(self) -> None:
        """Test that ms-rounded 128 Hz timestamps do not collapse 1s windows to 125 samples."""
        subject_dir = self.input_root / "16"
        subject_dir.mkdir(parents=True)

        sample_count = 257
        rounded_times = [f"00:00:{value:06.3f}" for value in np.round(np.arange(sample_count) / 128, 3)]
        df = pd.DataFrame({
            "EEG_Ch1": np.arange(sample_count, dtype=float),
            "EEG_Ch2": np.arange(1000, 1000 + sample_count, dtype=float),
            "Time": rounded_times,
        })
        df.to_excel(subject_dir / "16_e_1.xlsx", index=False)

        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )

        X = np.load(self.output_root / "subject_16" / "subject_16_X.npy")

        self.assertEqual(X.shape, (2, 2, 128))
        np.testing.assert_array_equal(X[0, 0, :], np.arange(128, dtype=float))
        np.testing.assert_array_equal(X[1, 0, :], np.arange(128, 256, dtype=float))

    def test_inf_in_eeg_data_is_rejected(self) -> None:
        """Test that Inf values in EEG data cause file to be skipped."""
        self._write_test_xlsx("13", "1", 4, has_inf=True)

        # Should print error and skip file (not write Inf to npy)
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_13"
        self.assertFalse(subject_out.exists())

    def test_negative_inf_in_eeg_data_is_rejected(self) -> None:
        """Test that -Inf values in EEG data cause file to be skipped."""
        subject_dir = self.input_root / "14"
        subject_dir.mkdir(parents=True)
        
        # Create file with -Inf in EEG data
        df = pd.DataFrame({
            "EEG_Ch1": [1.0, -np.inf, 3.0, 4.0],
            "EEG_Ch2": [5.0, 6.0, 7.0, 8.0],
            "Time": ["00:00:00.00", "00:00:00.50", "00:00:01.00", "00:00:01.50"],
        })
        df.to_excel(subject_dir / "14_e_1.xlsx", index=False)

        # Should print error and skip file
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_14"
        self.assertFalse(subject_out.exists())

    def test_partial_duplicate_timestamps_are_rejected(self) -> None:
        """Test that partial duplicate timestamps (some intervals == 0) cause file to be skipped.
        
        This is the key bug: when only SOME adjacent intervals are 0 (not all),
        the median interval can still be > 0, causing the file to incorrectly pass validation.
        The correct behavior is to reject any file with ANY interval <= 0.
        """
        subject_dir = self.input_root / "15"
        subject_dir.mkdir(parents=True)
        
        # Create file with partial duplicate timestamps:
        # intervals: [0.5, 0.5, 0.0, 0.0, 0.5]
        # median would be 0.5 (still positive!), but file should be rejected
        df = pd.DataFrame({
            "EEG_Ch1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "EEG_Ch2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Time": [0.0, 0.5, 1.0, 1.0, 1.0, 1.5],  # Two duplicate timestamps at 1.0
        })
        df.to_excel(subject_dir / "15_e_1.xlsx", index=False)

        # Should skip file with error about invalid time sequence
        self.module.convert_xlsx_to_npy(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=1.0,
        )
        
        # Output directory should not be created if file was skipped
        subject_out = self.output_root / "subject_15"
        self.assertFalse(subject_out.exists(), 
                        "File with partial duplicate timestamps should be rejected")

    def test_resolve_runtime_config_prompts_for_missing_values(self) -> None:
        """Test that running without CLI args uses interactive prompts and defaults."""
        with patch("builtins.input", side_effect=["", "", "1.5", ""]):
            input_root, output_root, window_seconds, stride_seconds = self.module.resolve_runtime_config(
                input_root=None,
                output_root=None,
                window_seconds=None,
                stride_seconds=None,
            )

        self.assertEqual(input_root, self.module.DEFAULT_INPUT_ROOT)
        self.assertEqual(output_root, self.module.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(window_seconds, 1.5)
        self.assertEqual(stride_seconds, 1.5)

    def test_resolve_runtime_config_accepts_custom_paths_and_stride(self) -> None:
        """Test that interactive prompts accept custom input/output paths and stride."""
        custom_input = self.temp_dir / "custom_input"
        custom_output = self.temp_dir / "custom_output"
        custom_input.mkdir(parents=True)

        with patch(
            "builtins.input",
            side_effect=[str(custom_input), str(custom_output), "2", "0.5"],
        ):
            input_root, output_root, window_seconds, stride_seconds = self.module.resolve_runtime_config(
                input_root=None,
                output_root=None,
                window_seconds=None,
                stride_seconds=None,
            )

        self.assertEqual(input_root, custom_input)
        self.assertEqual(output_root, custom_output)
        self.assertEqual(window_seconds, 2.0)
        self.assertEqual(stride_seconds, 0.5)


if __name__ == "__main__":
    unittest.main()
