from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_to_list"
    / "build_activity_global_index.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("build_activity_global_index", MODULE_PATH)
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


def _write_xlsx(
    directory: Path,
    subject_id: str,
    label: str,
    n_samples: int,
    *,
    interval_seconds: float = 0.5,
) -> None:
    """Write a minimal valid EEG xlsx file under *directory*/<subject_id>/."""
    directory.mkdir(parents=True, exist_ok=True)
    times = [f"00:00:{(i * interval_seconds):012.9f}" for i in range(n_samples)]
    df = pd.DataFrame(
        {
            "EEG_Ch1": list(range(n_samples)),
            "EEG_Ch2": list(range(100, 100 + n_samples)),
            "NonEEG_Col": list(range(200, 200 + n_samples)),
            "Time": times,
        }
    )
    df.to_excel(directory / f"{subject_id}_e_{label}.xlsx", index=False)


def _write_common_channel_xlsx(
    directory: Path,
    subject_id: str,
    label: str,
    *,
    suffix: str,
    include_a1_a2: bool,
    n_samples: int = 4,
    interval_seconds: float = 0.5,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    times = [f"00:00:{(i * interval_seconds):012.9f}" for i in range(n_samples)]
    df_dict = {
        f"EEG {channel}-{suffix}": [idx * 10 + row for row in range(n_samples)]
        for idx, channel in enumerate(COMMON_CHANNELS)
    }
    if include_a1_a2:
        df_dict[f"EEG A1-{suffix}"] = [900 + row for row in range(n_samples)]
        df_dict[f"EEG A2-{suffix}"] = [950 + row for row in range(n_samples)]
    df_dict["ECG EKG"] = [1000 + row for row in range(n_samples)]
    df_dict["Time"] = times
    pd.DataFrame(df_dict).to_excel(directory / f"{subject_id}_e_{label}.xlsx", index=False)


class BuildActivityGlobalIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="build-global-"))
        self.input_root = self.temp_dir / "input"
        self.output_root = self.temp_dir / "output"
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build(self, window: float = 1.0, stride: float = 1.0) -> None:
        self.module.build_global_dataset(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=window,
            stride_seconds=stride,
        )

    def _load_outputs(self) -> dict:
        return {
            "X": np.load(self.output_root / "X.npy"),
            "y": np.load(self.output_root / "y.npy"),
            "subject_ids": np.load(self.output_root / "subject_ids.npy"),
            "record_ids": np.load(self.output_root / "record_ids.npy"),
            "window_indices": np.load(self.output_root / "window_indices.npy"),
            "metadata": json.loads((self.output_root / "metadata.json").read_text()),
        }

    # ------------------------------------------------------------------
    # Test 1: build from 2 subjects, each with 2-3 xlsx files
    # ------------------------------------------------------------------

    def test_build_from_two_subjects_produces_all_outputs(self) -> None:
        """Build from subject 1 (2 files) and subject 2 (3 files); all outputs exist."""
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"

        # subject 1: e_1 (4 samples -> 2 windows), e_2 (6 samples -> 3 windows)
        _write_xlsx(subj1, "1", "1", 4)
        _write_xlsx(subj1, "1", "2", 6)

        # subject 2: e_1 (4), e_2 (4), e_3 (4) -> 2+2+2 = 6 windows
        _write_xlsx(subj2, "2", "1", 4)
        _write_xlsx(subj2, "2", "2", 4)
        _write_xlsx(subj2, "2", "3", 4)

        self._build()

        for fname in ("X.npy", "y.npy", "subject_ids.npy", "record_ids.npy",
                      "window_indices.npy", "metadata.json"):
            self.assertTrue((self.output_root / fname).exists(), f"Missing {fname}")

    # ------------------------------------------------------------------
    # Test 2: output dimensions correct (N, C, T)
    # ------------------------------------------------------------------

    def test_output_shape_is_N_C_T(self) -> None:
        """X must be (N, C, T); each window: 2 channels, 2 samples (1s / 0.5s interval)."""
        subj1 = self.input_root / "1"
        _write_xlsx(subj1, "1", "1", 4)  # 2 windows
        _write_xlsx(subj1, "1", "2", 6)  # 3 windows

        self._build()
        arrs = self._load_outputs()

        X = arrs["X"]
        # 5 windows total, 2 EEG channels, 2 samples per 1-second window at 0.5s interval
        self.assertEqual(X.shape, (5, 2, 2))

    # ------------------------------------------------------------------
    # Test 3: side arrays have same length as X first dim
    # ------------------------------------------------------------------

    def test_all_arrays_have_consistent_first_dimension(self) -> None:
        """y, subject_ids, record_ids, window_indices must all match X.shape[0]."""
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"
        _write_xlsx(subj1, "1", "1", 4)
        _write_xlsx(subj1, "1", "3", 6)
        _write_xlsx(subj2, "2", "2", 4)

        self._build()
        arrs = self._load_outputs()

        n = arrs["X"].shape[0]
        self.assertEqual(arrs["y"].shape, (n,))
        self.assertEqual(arrs["subject_ids"].shape, (n,))
        self.assertEqual(arrs["record_ids"].shape, (n,))
        self.assertEqual(arrs["window_indices"].shape, (n,))

    # ------------------------------------------------------------------
    # Test 4: metadata.json contains all required fields
    # ------------------------------------------------------------------

    def test_metadata_contains_required_fields(self) -> None:
        """metadata.json must include the 7 required keys."""
        _write_xlsx(self.input_root / "1", "1", "1", 4)
        self._build()

        meta = self._load_outputs()["metadata"]
        required = {
            "label_map",
            "record_id_map",
            "window_seconds",
            "stride_seconds",
            "n_subjects",
            "n_records",
            "n_samples",
        }
        for key in required:
            self.assertIn(key, meta, f"metadata missing key: {key}")

    # ------------------------------------------------------------------
    # Test 5: label mapping e_1/e_2/e_3 -> 0/1/2
    # ------------------------------------------------------------------

    def test_label_mapping_is_correct(self) -> None:
        """e_1 -> 0, e_2 -> 1, e_3 -> 2; label_map in metadata matches."""
        subj = self.input_root / "1"
        _write_xlsx(subj, "1", "1", 4)  # 2 windows, label 0
        _write_xlsx(subj, "1", "2", 4)  # 2 windows, label 1
        _write_xlsx(subj, "1", "3", 4)  # 2 windows, label 2

        self._build()
        arrs = self._load_outputs()

        y = arrs["y"]
        # Files are sorted by label; first 2 windows label 0, next 2 label 1, last 2 label 2
        np.testing.assert_array_equal(y[:2], [0, 0])
        np.testing.assert_array_equal(y[2:4], [1, 1])
        np.testing.assert_array_equal(y[4:], [2, 2])

        meta_label_map = arrs["metadata"]["label_map"]
        self.assertEqual(meta_label_map["e_1"], 0)
        self.assertEqual(meta_label_map["e_2"], 1)
        self.assertEqual(meta_label_map["e_3"], 2)

    # ------------------------------------------------------------------
    # Test 6: record_ids encode subject/activity identity
    # ------------------------------------------------------------------

    def test_record_id_map_entries_match_source_files(self) -> None:
        """record_id_map must map integer IDs to '<subject>_e_<label>' strings."""
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"
        _write_xlsx(subj1, "1", "1", 4)
        _write_xlsx(subj2, "2", "2", 4)

        self._build()
        meta = self._load_outputs()["metadata"]

        record_id_map = meta["record_id_map"]
        values = set(record_id_map.values())
        self.assertIn("1_e_1", values)
        self.assertIn("2_e_2", values)

    # ------------------------------------------------------------------
    # Test 7: window_indices track position within each record
    # ------------------------------------------------------------------

    def test_window_indices_reset_per_record(self) -> None:
        """window_indices should start from 0 for each distinct record file."""
        subj = self.input_root / "1"
        _write_xlsx(subj, "1", "1", 4)  # 2 windows -> indices 0, 1
        _write_xlsx(subj, "1", "2", 6)  # 3 windows -> indices 0, 1, 2

        self._build()
        arrs = self._load_outputs()

        wi = arrs["window_indices"]
        # First record: indices 0, 1
        np.testing.assert_array_equal(wi[:2], [0, 1])
        # Second record: indices 0, 1, 2
        np.testing.assert_array_equal(wi[2:], [0, 1, 2])

    # ------------------------------------------------------------------
    # Test 8: subject_ids correspond to directory integer names
    # ------------------------------------------------------------------

    def test_subject_ids_are_correct_integers(self) -> None:
        """subject_ids must equal int(subject_dir.name) for each window."""
        subj1 = self.input_root / "1"
        subj3 = self.input_root / "3"
        _write_xlsx(subj1, "1", "1", 4)  # 2 windows -> subject_id 1
        _write_xlsx(subj3, "3", "2", 4)  # 2 windows -> subject_id 3

        self._build()
        arrs = self._load_outputs()

        sid = arrs["subject_ids"]
        self.assertEqual(list(sid[:2]), [1, 1])
        self.assertEqual(list(sid[2:]), [3, 3])

    # ------------------------------------------------------------------
    # Test 9: metadata n_samples / n_subjects / n_records consistency
    # ------------------------------------------------------------------

    def test_metadata_counts_are_consistent(self) -> None:
        """n_samples, n_subjects, n_records in metadata must be accurate."""
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"
        _write_xlsx(subj1, "1", "1", 4)  # 2 windows
        _write_xlsx(subj1, "1", "2", 6)  # 3 windows
        _write_xlsx(subj2, "2", "3", 4)  # 2 windows

        self._build()
        arrs = self._load_outputs()
        meta = arrs["metadata"]

        self.assertEqual(meta["n_samples"], arrs["X"].shape[0])  # 7
        self.assertEqual(meta["n_subjects"], 2)
        self.assertEqual(meta["n_records"], 3)

    # ------------------------------------------------------------------
    # Test 10: stride defaults to window_seconds when not supplied
    # ------------------------------------------------------------------

    def test_stride_defaults_to_window_seconds(self) -> None:
        """When stride_seconds is None, non-overlapping windows are produced."""
        subj = self.input_root / "1"
        _write_xlsx(subj, "1", "1", 6)  # 3s of data -> 3 non-overlapping 1s windows

        self.module.build_global_dataset(
            input_root=self.input_root,
            output_root=self.output_root,
            window_seconds=1.0,
            stride_seconds=None,  # should default to 1.0
        )

        X = np.load(self.output_root / "X.npy")
        self.assertEqual(X.shape[0], 3)

    # ------------------------------------------------------------------
    # Test 11: interactive default for stride (missing-param behavior)
    # ------------------------------------------------------------------

    def test_resolve_runtime_config_defaults_stride_to_window(self) -> None:
        """When stride input is blank, resolve_runtime_config uses window_seconds."""
        subj = self.input_root / "1"
        _write_xlsx(subj, "1", "1", 6)

        # Simulate user pressing Enter for stride (blank input -> default)
        inputs = [""]  # blank -> accept default (= window_seconds = 1.0)
        with patch("builtins.input", side_effect=inputs):
            _, _, w, s = self.module.resolve_runtime_config(
                input_root=self.input_root,
                output_root=self.output_root,
                window_seconds=1.0,
                stride_seconds=None,  # will prompt
            )

        self.assertEqual(w, 1.0)
        self.assertEqual(s, 1.0)

    # ------------------------------------------------------------------
    # Test 12: non-EEG columns are excluded from X
    # ------------------------------------------------------------------

    def test_non_eeg_columns_excluded(self) -> None:
        """Only columns starting with 'EEG' should appear in X channels."""
        subj = self.input_root / "1"
        _write_xlsx(subj, "1", "1", 4)  # writes EEG_Ch1, EEG_Ch2, NonEEG_Col

        self._build()
        X = np.load(self.output_root / "X.npy")

        # 2 EEG channels, not 3
        self.assertEqual(X.shape[1], 2)

    def test_common_21_channel_filter_allows_ref_and_av_files_together(self) -> None:
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"
        _write_common_channel_xlsx(
            subj1,
            "1",
            "1",
            suffix="REF",
            include_a1_a2=True,
        )
        _write_common_channel_xlsx(
            subj2,
            "2",
            "1",
            suffix="AV",
            include_a1_a2=False,
        )

        self._build()
        arrs = self._load_outputs()
        self.assertEqual(arrs["X"].shape, (4, 21, 2))

    # ------------------------------------------------------------------
    # Test 13: missing input directory raises FileNotFoundError
    # ------------------------------------------------------------------

    def test_missing_input_root_raises(self) -> None:
        """build_global_dataset must raise FileNotFoundError for nonexistent input."""
        with self.assertRaises(FileNotFoundError):
            self.module.build_global_dataset(
                input_root=self.temp_dir / "nonexistent",
                output_root=self.output_root,
                window_seconds=1.0,
                stride_seconds=1.0,
            )

    # ------------------------------------------------------------------
    # Test 14: mismatched EEG channel counts fail with a clear error
    # ------------------------------------------------------------------

    def test_mismatched_eeg_channel_counts_raise_clear_error(self) -> None:
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"
        _write_xlsx(subj1, "1", "1", 4)

        subj2.mkdir(parents=True, exist_ok=True)
        times = [f"00:00:{i * 0.5:05.2f}" for i in range(4)]
        df = pd.DataFrame(
            {
                "EEG_Ch1": [0, 1, 2, 3],
                "EEG_Ch2": [10, 11, 12, 13],
                "EEG_Ch3": [20, 21, 22, 23],
                "Time": times,
            }
        )
        df.to_excel(subj2 / "2_e_1.xlsx", index=False)

        with self.assertRaisesRegex(ValueError, "Inconsistent EEG channel count"):
            self._build()

    def test_mismatched_sampling_intervals_raise_clear_error(self) -> None:
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"
        _write_xlsx(subj1, "1", "1", 4)

        subj2.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "EEG_Ch1": [0, 1, 2, 3],
                "EEG_Ch2": [10, 11, 12, 13],
                "Time": ["00:00:00.00", "00:00:00.25", "00:00:00.50", "00:00:00.75"],
            }
        )
        df.to_excel(subj2 / "2_e_1.xlsx", index=False)

        with self.assertRaisesRegex(ValueError, "Inconsistent sampling interval"):
            self._build()

    def test_tiny_sampling_interval_jitter_is_allowed(self) -> None:
        subj1 = self.input_root / "1"
        subj2 = self.input_root / "2"
        _write_xlsx(subj1, "1", "1", 4, interval_seconds=0.0078125)
        _write_xlsx(subj2, "2", "1", 4, interval_seconds=0.007812498)

        self._build(window=0.015625, stride=0.015625)
        arrs = self._load_outputs()
        self.assertEqual(arrs["X"].shape[0], 4)

    # ------------------------------------------------------------------
    # Test 14: two subjects; separate subject_ids are preserved
    # ------------------------------------------------------------------

    def test_two_subjects_with_three_records_each(self) -> None:
        """Full 2-subject, 3-record-per-subject scenario."""
        for sid in ("1", "2"):
            subj_dir = self.input_root / sid
            for lbl in ("1", "2", "3"):
                _write_xlsx(subj_dir, sid, lbl, 4)  # 2 windows each

        self._build()
        arrs = self._load_outputs()

        # 2 subjects x 3 records x 2 windows = 12 total
        self.assertEqual(arrs["X"].shape[0], 12)
        self.assertEqual(set(arrs["subject_ids"].tolist()), {1, 2})
        self.assertEqual(arrs["metadata"]["n_subjects"], 2)
        self.assertEqual(arrs["metadata"]["n_records"], 6)
        self.assertEqual(arrs["metadata"]["n_samples"], 12)


if __name__ == "__main__":
    unittest.main()
