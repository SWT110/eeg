"""
Tests for train_activity_loso_batch.py
=======================================
Covers:
  1. discover_subject_ids_from_global_dataset – reads unique IDs from subject_ids.npy
  2. fold_is_complete – detects metrics.json / best_model.pt
  3. run_loso_batch – skip_existing skips complete folds
  4. run_loso_batch – calls train_loso_fold for incomplete folds
  5. parse_args defaults – EEG-relative paths + cuda:0
  6. parse_subject_id_list – comma-separated parsing + validation
  7. maybe_rerun_in_project_env – auto-restart key behaviour
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch, MagicMock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_activity_loso_batch.py"
_PROJECT_ROOT = MODULE_PATH.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location("train_activity_loso_batch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_fake_global_dataset(
    root: Path,
    n_subjects: int = 3,
    n_per_subject: int = 4,
) -> None:
    total = n_subjects * n_per_subject
    rng = np.random.default_rng(0)
    X = rng.random((total, 21, 640), dtype=np.float32)
    y = np.tile([0, 1, 2, 0], n_subjects)[:total].astype(np.int64)
    subject_ids = np.repeat(np.arange(1, n_subjects + 1), n_per_subject).astype(np.int64)

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X.npy", X)
    np.save(root / "y.npy", y)
    np.save(root / "subject_ids.npy", subject_ids)
    np.save(root / "record_ids.npy", np.zeros(total, dtype=np.int64))
    np.save(root / "window_indices.npy", np.tile(np.arange(n_per_subject), n_subjects).astype(np.int64))

    metadata = {
        "label_map": {"e_1": 0, "e_2": 1, "e_3": 2},
        "n_subjects": n_subjects,
        "n_samples": total,
    }
    with open(root / "metadata.json", "w") as fh:
        json.dump(metadata, fh)


class TestDiscoverSubjectIds(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="loso-batch-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_global_dataset(self.dataset_root, n_subjects=4, n_per_subject=3)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_discovers_sorted_unique_ids(self) -> None:
        ids = self.module.discover_subject_ids_from_global_dataset(self.dataset_root)
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_raises_when_file_missing(self) -> None:
        bad_root = self.temp_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            self.module.discover_subject_ids_from_global_dataset(bad_root)

    def test_deduplicates_repeated_ids(self) -> None:
        # write custom subject_ids.npy with duplicates and gaps
        arr = np.array([5, 5, 7, 7, 7, 3], dtype=np.int64)
        np.save(self.dataset_root / "subject_ids.npy", arr)
        ids = self.module.discover_subject_ids_from_global_dataset(self.dataset_root)
        self.assertEqual(ids, [3, 5, 7])


class TestFoldIsComplete(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="loso-batch-"))
        self.output_dir = self.temp_dir / "outputs"
        self.output_dir.mkdir()
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_false_when_fold_dir_absent(self) -> None:
        self.assertFalse(self.module.fold_is_complete(self.output_dir, subject_id=1))

    def test_true_when_metrics_json_exists(self) -> None:
        fold_dir = self.output_dir / "fold_subject_2"
        fold_dir.mkdir()
        (fold_dir / "metrics.json").write_text("{}")
        self.assertTrue(self.module.fold_is_complete(self.output_dir, subject_id=2))

    def test_true_when_best_model_pt_exists(self) -> None:
        fold_dir = self.output_dir / "fold_subject_3"
        fold_dir.mkdir()
        (fold_dir / "best_model.pt").write_bytes(b"\x00")
        self.assertTrue(self.module.fold_is_complete(self.output_dir, subject_id=3))

    def test_false_when_fold_dir_empty(self) -> None:
        fold_dir = self.output_dir / "fold_subject_4"
        fold_dir.mkdir()
        self.assertFalse(self.module.fold_is_complete(self.output_dir, subject_id=4))


class TestRunLosoBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="loso-batch-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_global_dataset(self.dataset_root, n_subjects=3)
        self.output_dir = self.temp_dir / "outputs"
        self.output_dir.mkdir()
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_skip_existing_skips_complete_folds(self) -> None:
        # Pre-create fold for subject 2
        (self.output_dir / "fold_subject_2").mkdir()
        (self.output_dir / "fold_subject_2" / "metrics.json").write_text("{}")

        with patch.object(
            self.module, "train_loso_fold",
            return_value=self.output_dir / "fold_subject_1" / "metrics.json",
        ) as mock_train:
            results = self.module.run_loso_batch(
                subject_ids=[1, 2, 3],
                dataset_root=self.dataset_root,
                epochs=1,
                batch_size=8,
                lr=2e-4,
                device="cpu",
                output_dir=self.output_dir,
                skip_existing=True,
            )

        statuses = {r.subject_id: r.status for r in results}
        self.assertEqual(statuses[1], "trained")
        self.assertEqual(statuses[2], "skipped")
        self.assertEqual(statuses[3], "trained")
        # train_loso_fold called for subject 1 and 3, not 2
        called_subjects = [
            call.kwargs["test_subject_id"]
            for call in mock_train.call_args_list
        ]
        self.assertIn(1, called_subjects)
        self.assertNotIn(2, called_subjects)
        self.assertIn(3, called_subjects)

    def test_without_skip_existing_runs_all_folds(self) -> None:
        def fake_train(*, test_subject_id, **kwargs):
            fold_dir = self.output_dir / f"fold_subject_{test_subject_id}"
            fold_dir.mkdir(exist_ok=True)
            p = fold_dir / "metrics.json"
            p.write_text("{}")
            return p

        with patch.object(self.module, "train_loso_fold", side_effect=fake_train):
            results = self.module.run_loso_batch(
                subject_ids=[1, 2],
                dataset_root=self.dataset_root,
                epochs=1,
                batch_size=8,
                lr=2e-4,
                device="cpu",
                output_dir=self.output_dir,
                skip_existing=False,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.status == "trained" for r in results))

    def test_failed_fold_raises_runtime_error(self) -> None:
        def boom(**kwargs):
            raise RuntimeError("simulated failure")

        with patch.object(self.module, "train_loso_fold", side_effect=boom):
            with self.assertRaises(RuntimeError) as ctx:
                self.module.run_loso_batch(
                    subject_ids=[1],
                    dataset_root=self.dataset_root,
                    epochs=1,
                    batch_size=8,
                    lr=2e-4,
                    device="cpu",
                    output_dir=self.output_dir,
                    skip_existing=False,
                )
        self.assertIn("failed folds", str(ctx.exception))


class TestParseArgs(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_default_dataset_root_points_to_global_activity_dataset(self) -> None:
        args = self.module.parse_args([])
        expected = MODULE_PATH.resolve().parents[1] / "eeg-data-processing" / "data_to_list" / "global_activity_dataset"
        self.assertEqual(args.dataset_root, expected)

    def test_default_output_dir_is_activity_loso(self) -> None:
        args = self.module.parse_args([])
        expected = MODULE_PATH.resolve().parent / "outputs" / "activity_loso"
        self.assertEqual(args.output_dir, expected)

    def test_default_device_is_cuda0(self) -> None:
        args = self.module.parse_args([])
        self.assertEqual(args.device, "cuda:0")

    def test_skip_existing_defaults_to_false(self) -> None:
        args = self.module.parse_args([])
        self.assertFalse(args.skip_existing)

    def test_skip_existing_flag_sets_true(self) -> None:
        args = self.module.parse_args(["--skip-existing"])
        self.assertTrue(args.skip_existing)

    def test_subject_ids_defaults_to_none(self) -> None:
        args = self.module.parse_args([])
        self.assertIsNone(args.subject_ids)


class TestParseSubjectIdList(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_none_returns_none(self) -> None:
        self.assertIsNone(self.module.parse_subject_id_list(None))

    def test_comma_separated_integers(self) -> None:
        self.assertEqual(self.module.parse_subject_id_list("1,3,5"), [1, 3, 5])

    def test_deduplicates(self) -> None:
        self.assertEqual(self.module.parse_subject_id_list("2,2,3"), [2, 3])

    def test_raises_on_zero(self) -> None:
        with self.assertRaises(ValueError):
            self.module.parse_subject_id_list("0,1")

    def test_raises_on_non_integer(self) -> None:
        with self.assertRaises(ValueError):
            self.module.parse_subject_id_list("1,abc")


class TestMaybeRerunInProjectEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="loso-batch-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_reruns_when_cuda_unavailable_and_env_prefix_exists(self) -> None:
        fake_env_prefix = self.temp_dir / ".conda-envs" / self.module.DEFAULT_ENV_NAME
        fake_env_prefix.mkdir(parents=True, exist_ok=True)

        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.object(self.module, "project_env_prefix", return_value=fake_env_prefix):
                with patch.object(
                    self.module.subprocess, "run", return_value=CompletedProcess([], 0)
                ) as mock_run:
                    with self.assertRaises(SystemExit) as ctx:
                        self.module.maybe_rerun_in_project_env([], "cuda:0")

        self.assertEqual(ctx.exception.code, 0)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[0], str(fake_env_prefix / "bin" / "python"))
        self.assertEqual(
            mock_run.call_args.kwargs["env"][self.module.AUTO_RERUN_ENV_VAR], "1"
        )

    def test_no_rerun_for_cpu_device(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.object(
                self.module.subprocess, "run", side_effect=AssertionError("should not run")
            ):
                self.module.maybe_rerun_in_project_env([], "cpu")

    def test_no_rerun_when_cuda_available(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=True):
            with patch.object(
                self.module.subprocess, "run", side_effect=AssertionError("should not run")
            ):
                self.module.maybe_rerun_in_project_env([], "cuda:0")

    def test_no_rerun_when_env_var_set(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.dict(
                os.environ, {self.module.AUTO_RERUN_ENV_VAR: "1"}, clear=False
            ):
                with patch.object(
                    self.module.subprocess, "run", side_effect=AssertionError("should not run")
                ):
                    self.module.maybe_rerun_in_project_env([], "cuda:0")


if __name__ == "__main__":
    unittest.main()
