"""
Tests for train_activity_final.py
===================================
Covers:
  1. final_train_val_split – 80/20 chronological split within each record_id
  2. final_train_val_split – chronological order is respected (shuffled window_indices)
  3. final_train_val_split – multiple records each get their own split
  4. final_train_val_split – conv-channel dim added, val_fraction bounds check
  5. standardize_for_deployment – returns correct mean/std + normalised arrays
  6. normalization_stats.json round-trip – values survive JSON serialise/load
  7. load_global_dataset_full – reads all 5 arrays + metadata correctly
  8. load_global_dataset_full – raises FileNotFoundError on missing files
  9. parse_args defaults – EEG-relative paths + cuda:0 + 0.2 val_fraction
 10. maybe_rerun_in_project_env – auto-restart key behaviour (no CUDA)
 11. maybe_rerun_in_project_env – no rerun when device is cpu
 12. maybe_rerun_in_project_env – no rerun when env var already set
 13. resolve_runtime_config – missing args in non-interactive mode raises
 14. resolve_runtime_config – explicit values accepted without prompting
 15. main – maybe_rerun called before resolve
 16. main – no-argv path passes Nones to resolve
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
from unittest.mock import patch

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_activity_final.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_activity_final", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_fake_dataset(
    root: Path,
    n_records: int = 3,
    n_per_record: int = 10,
    n_channels: int = 21,
    n_times: int = 640,
) -> None:
    """Write minimal global dataset files to *root*."""
    total = n_records * n_per_record
    rng = np.random.default_rng(42)
    X = rng.random((total, n_channels, n_times), dtype=np.float32)
    y = np.tile([0, 1, 2], total // 3 + 1)[:total].astype(np.int64)
    record_ids = np.repeat(np.arange(n_records), n_per_record).astype(np.int64)
    window_indices = np.tile(np.arange(n_per_record), n_records)[:total].astype(np.int64)
    subject_ids = np.repeat(np.arange(1, n_records + 1), n_per_record).astype(np.int64)

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X.npy", X)
    np.save(root / "y.npy", y)
    np.save(root / "record_ids.npy", record_ids)
    np.save(root / "window_indices.npy", window_indices)
    np.save(root / "subject_ids.npy", subject_ids)

    metadata = {
        "label_map": {"e_1": 0, "e_2": 1, "e_3": 2},
        "n_samples": total,
        "window_seconds": 5.0,
        "stride_seconds": 5.0,
    }
    with open(root / "metadata.json", "w") as fh:
        json.dump(metadata, fh)


class TestFinalTrainValSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_80_20_split_single_record(self) -> None:
        """10 windows in one record → 8 train, 2 val (80/20)."""
        n = 10
        X = np.zeros((n, 5, 20), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        record_ids = np.zeros(n, dtype=np.int64)
        window_indices = np.arange(n, dtype=np.int64)

        train_X, train_y, val_X, val_y = self.module.final_train_val_split(
            X, y, record_ids, window_indices, val_fraction=0.2
        )
        self.assertEqual(len(train_X), 8)
        self.assertEqual(len(val_X), 2)

    def test_chronological_ordering_respected(self) -> None:
        """Shuffled window_indices: last 20% chronologically become val."""
        n = 5
        X = np.zeros((n, 3, 10), dtype=np.float32)
        y = np.arange(n, dtype=np.int64)
        record_ids = np.zeros(n, dtype=np.int64)
        # Shuffled order: original positions 0..4 but stored as 4,0,3,1,2
        window_indices = np.array([4, 0, 3, 1, 2], dtype=np.int64)

        _, train_y, _, val_y = self.module.final_train_val_split(
            X, y, record_ids, window_indices, val_fraction=0.2
        )
        # After sorting by window_indices: order is 0,1,2,3,4 → original rows 1,3,4,2,0
        # n_train = max(1, int(5*0.8)) = 4  → train rows 1,3,4,2 → labels y[1]=1,y[3]=3,y[4]=4,y[2]=2
        # val row 0 → label y[0]=0
        self.assertEqual(set(val_y.tolist()), {0})   # chronologically last = original row with window_idx=4 → y[0]
        self.assertEqual(len(train_y), 4)

    def test_multiple_records_split_independently(self) -> None:
        """Each record gets its own chronological 80/20 split."""
        n_rec, n_win = 3, 10
        total = n_rec * n_win
        X = np.zeros((total, 5, 20), dtype=np.float32)
        y = np.zeros(total, dtype=np.int64)
        record_ids = np.repeat(np.arange(n_rec), n_win).astype(np.int64)
        window_indices = np.tile(np.arange(n_win), n_rec).astype(np.int64)

        train_X, _, val_X, _ = self.module.final_train_val_split(
            X, y, record_ids, window_indices, val_fraction=0.2
        )
        # Each record: 10 windows → 8 train + 2 val
        self.assertEqual(len(train_X), n_rec * 8)
        self.assertEqual(len(val_X), n_rec * 2)

    def test_conv_channel_dim_added(self) -> None:
        """Output shapes have (N, 1, C, T)."""
        n = 5
        X = np.zeros((n, 7, 30), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        record_ids = np.zeros(n, dtype=np.int64)
        window_indices = np.arange(n, dtype=np.int64)

        train_X, _, val_X, _ = self.module.final_train_val_split(
            X, y, record_ids, window_indices, val_fraction=0.2
        )
        self.assertEqual(train_X.ndim, 4)
        self.assertEqual(train_X.shape[1], 1)
        self.assertEqual(train_X.shape[2], 7)
        self.assertEqual(train_X.shape[3], 30)

    def test_invalid_val_fraction_raises(self) -> None:
        X = np.zeros((5, 2, 10), dtype=np.float32)
        y = np.zeros(5, dtype=np.int64)
        record_ids = np.zeros(5, dtype=np.int64)
        window_indices = np.arange(5, dtype=np.int64)
        with self.assertRaises(ValueError):
            self.module.final_train_val_split(X, y, record_ids, window_indices, val_fraction=1.5)

    def test_at_least_one_train_sample_per_record(self) -> None:
        """Even a record with 1 window should put that window in train."""
        X = np.zeros((1, 3, 10), dtype=np.float32)
        y = np.zeros(1, dtype=np.int64)
        record_ids = np.zeros(1, dtype=np.int64)
        window_indices = np.zeros(1, dtype=np.int64)

        train_X, _, val_X, _ = self.module.final_train_val_split(
            X, y, record_ids, window_indices, val_fraction=0.2
        )
        self.assertEqual(len(train_X), 1)
        self.assertEqual(len(val_X), 0)


class TestStandardizeForDeployment(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_returns_mean_std(self) -> None:
        rng = np.random.default_rng(1)
        train = rng.random((20, 1, 5, 30), dtype=np.float32) * 4 + 2
        val = rng.random((5, 1, 5, 30), dtype=np.float32)

        _, _, mean, std = self.module.standardize_for_deployment(train, val)

        self.assertAlmostEqual(mean, float(train.mean()), places=5)
        self.assertAlmostEqual(std, float(train.std()), places=5)

    def test_train_is_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(2)
        train = rng.random((30, 1, 5, 20), dtype=np.float32) * 10 + 5
        val = rng.random((5, 1, 5, 20), dtype=np.float32)

        train_norm, _, _, _ = self.module.standardize_for_deployment(train, val)

        self.assertAlmostEqual(float(train_norm.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(train_norm.std()), 1.0, places=4)

    def test_val_uses_train_stats(self) -> None:
        # train all zeros+noise, val all zeros+100
        rng = np.random.default_rng(3)
        train = rng.random((20, 1, 5, 20), dtype=np.float32)
        val = train[:5] + 100.0

        train_norm, val_norm, mean, std = self.module.standardize_for_deployment(train, val)

        expected_val_mean = (float(train.mean()) + 100.0 - mean) / std
        self.assertAlmostEqual(float(val_norm.mean()), expected_val_mean, delta=0.1)

    def test_zero_std_raises(self) -> None:
        train = np.full((10, 1, 5, 20), 5.0, dtype=np.float32)
        val = np.full((5, 1, 5, 20), 5.0, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.module.standardize_for_deployment(train, val)

    def test_normalization_stats_json_round_trip(self) -> None:
        """mean/std survive JSON serialisation and can be used for normalisation."""
        rng = np.random.default_rng(4)
        train = rng.random((20, 1, 5, 20), dtype=np.float32) * 3 + 1
        val = rng.random((5, 1, 5, 20), dtype=np.float32) * 3 + 1

        _, val_norm_orig, mean, std = self.module.standardize_for_deployment(train, val)

        # Simulate JSON round-trip
        stats = {"mean": mean, "std": std}
        stats_json = json.dumps(stats)
        loaded = json.loads(stats_json)

        val_norm_loaded = (val - loaded["mean"]) / loaded["std"]
        np.testing.assert_allclose(val_norm_orig, val_norm_loaded, rtol=1e-5)


class TestLoadGlobalDatasetFull(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-final-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root, n_records=2, n_per_record=5)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_loads_all_arrays_and_metadata(self) -> None:
        X, y, record_ids, window_indices, metadata = self.module.load_global_dataset_full(
            self.dataset_root
        )
        total = 2 * 5
        self.assertEqual(X.shape, (total, 21, 640))
        self.assertEqual(y.shape, (total,))
        self.assertEqual(record_ids.shape, (total,))
        self.assertEqual(window_indices.shape, (total,))
        self.assertIn("label_map", metadata)
        self.assertIn("window_seconds", metadata)

    def test_raises_on_missing_file(self) -> None:
        bad = self.temp_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            self.module.load_global_dataset_full(bad)

    def test_raises_on_missing_record_ids(self) -> None:
        """Missing record_ids.npy alone should raise."""
        import shutil as _shutil
        bad_root = self.temp_dir / "bad_dataset"
        _shutil.copytree(self.dataset_root, bad_root)
        (bad_root / "record_ids.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            self.module.load_global_dataset_full(bad_root)


class TestParseArgsDefaults(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_default_dataset_root_is_eeg_relative(self) -> None:
        args = self.module.parse_args([])
        expected_eeg_root = MODULE_PATH.resolve().parents[1]
        self.assertEqual(
            args.dataset_root,
            expected_eeg_root / "local_artifacts" / "data_to_list" / "global_activity_dataset",
        )

    def test_default_output_dir_is_activity_final(self) -> None:
        args = self.module.parse_args([])
        expected = MODULE_PATH.resolve().parents[1] / "local_artifacts" / "outputs" / "activity_final"
        self.assertEqual(args.output_dir, expected)

    def test_default_device_is_cuda0(self) -> None:
        args = self.module.parse_args([])
        self.assertEqual(args.device, "cuda:0")

    def test_default_val_fraction_is_0_2(self) -> None:
        args = self.module.parse_args([])
        self.assertAlmostEqual(args.val_fraction, 0.2)

    def test_default_epochs_and_lr(self) -> None:
        args = self.module.parse_args([])
        self.assertEqual(args.epochs, 200)
        self.assertAlmostEqual(args.lr, 0.0002)
        self.assertEqual(args.batch_size, 72)

    def test_explicit_args_override_defaults(self) -> None:
        args = self.module.parse_args(
            ["--epochs", "5", "--device", "cpu", "--val-fraction", "0.3"]
        )
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.device, "cpu")
        self.assertAlmostEqual(args.val_fraction, 0.3)


class TestMaybeRerunInProjectEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-final-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_reexecutes_when_base_env_lacks_cuda(self) -> None:
        project_root = self.temp_dir / "EEG-Conformer"
        env_prefix = project_root / ".conda-envs" / self.module.DEFAULT_ENV_NAME
        env_prefix.mkdir(parents=True, exist_ok=True)
        self.module.PROJECT_ROOT = project_root
        self.module.EEG_ROOT = project_root.parent

        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.dict(os.environ, {"CONDA_EXE": "/opt/miniconda/bin/conda"}, clear=False):
                with patch.object(
                    self.module.subprocess, "run", return_value=CompletedProcess([], 0)
                ) as mock_run:
                    with self.assertRaises(SystemExit) as ctx:
                        self.module.maybe_rerun_in_project_env([], "cuda:0")

        self.assertEqual(ctx.exception.code, 0)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[0], str(env_prefix / "bin" / "python"))
        self.assertEqual(mock_run.call_args.kwargs["env"][self.module.AUTO_RERUN_ENV_VAR], "1")

    def test_no_rerun_when_cpu_device(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.object(
                self.module.subprocess, "run", side_effect=AssertionError("should not run")
            ):
                self.module.maybe_rerun_in_project_env([], "cpu")

    def test_no_rerun_when_cuda_already_usable(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=True):
            with patch.object(
                self.module.subprocess, "run", side_effect=AssertionError("should not run")
            ):
                self.module.maybe_rerun_in_project_env([], "cuda:0")

    def test_no_rerun_when_env_var_already_set(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.dict(
                os.environ, {self.module.AUTO_RERUN_ENV_VAR: "1"}, clear=False
            ):
                with patch.object(
                    self.module.subprocess, "run", side_effect=AssertionError("should not run")
                ):
                    self.module.maybe_rerun_in_project_env([], "cuda:0")


class TestResolveRuntimeConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-final-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_raises_in_noninteractive_mode_when_args_missing(self) -> None:
        with patch("sys.stdin.isatty", return_value=False):
            with patch("builtins.input", side_effect=AssertionError("should not prompt")):
                with self.assertRaises(ValueError) as ctx:
                    self.module.resolve_runtime_config(
                        dataset_root=None,
                        epochs=None,
                        batch_size=None,
                        lr=None,
                        device=None,
                        output_dir=None,
                        seed=None,
                        val_fraction=None,
                    )

        msg = str(ctx.exception)
        self.assertIn("Missing required arguments", msg)
        self.assertIn("--dataset-root", msg)

    def test_explicit_values_accepted_without_prompting(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            config = self.module.resolve_runtime_config(
                dataset_root=self.dataset_root,
                epochs=3,
                batch_size=8,
                lr=2e-4,
                device="cuda:0",
                output_dir=self.temp_dir / "out",
                seed=7,
                val_fraction=0.2,
            )

        self.assertEqual(config.epochs, 3)
        self.assertAlmostEqual(config.val_fraction, 0.2)
        self.assertEqual(config.seed, 7)


class TestTrainFinalModelSaving(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-final-save-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root, n_records=2, n_per_record=3, n_channels=2, n_times=8)
        self.output_dir = self.temp_dir / "out"
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_saves_model_when_first_val_accuracy_is_zero(self) -> None:
        class TinyActivityModel(torch.nn.Module):
            def __init__(self, *args, n_classes: int = 3, **kwargs) -> None:
                super().__init__()
                self.classifier = torch.nn.Linear(1, n_classes)

            def forward(self, inputs: torch.Tensor):
                features = inputs.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
                return features, self.classifier(features)

        with patch.object(self.module, "ActivityConformer", TinyActivityModel):
            with patch.object(self.module, "evaluate", return_value=(1.0, 0.0)):
                with patch.object(
                    self.module,
                    "collect_predictions",
                    return_value=(np.array([0, 1]), np.array([1, 1])),
                ):
                    self.module.train_final_model(
                        dataset_root=self.dataset_root,
                        epochs=1,
                        batch_size=2,
                        lr=2e-4,
                        device="cpu",
                        output_dir=self.output_dir,
                        seed=42,
                        val_fraction=0.5,
                    )

        self.assertTrue((self.output_dir / "final_model.pt").exists())


class TestMainWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-final-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_main_calls_rerun_before_resolve(self) -> None:
        """maybe_rerun_in_project_env should be called before resolve_runtime_config."""
        from argparse import Namespace

        fake_args = Namespace(
            dataset_root=self.dataset_root,
            epochs=1,
            batch_size=8,
            lr=2e-4,
            device="cuda:0",
            output_dir=self.temp_dir / "out",
            seed=42,
            val_fraction=0.2,
        )

        with patch.object(self.module, "parse_args", return_value=fake_args):
            with patch.object(
                self.module, "maybe_rerun_in_project_env", side_effect=SystemExit(0)
            ) as mock_rerun:
                with patch.object(
                    self.module,
                    "resolve_runtime_config",
                    side_effect=AssertionError("should not reach resolve"),
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        self.module.main(["--epochs", "1"])

        self.assertEqual(ctx.exception.code, 0)
        mock_rerun.assert_called_once()

    def test_main_no_args_passes_nones_to_resolve(self) -> None:
        """When called with no argv, main passes None for all config args."""
        fake_config = self.module.RuntimeConfig(
            dataset_root=self.dataset_root,
            epochs=1,
            batch_size=8,
            lr=2e-4,
            device="cpu",
            output_dir=self.temp_dir / "out",
            seed=42,
            val_fraction=0.2,
        )

        with patch.object(self.module, "maybe_rerun_in_project_env", return_value=None):
            with patch.object(
                self.module, "resolve_runtime_config", return_value=fake_config
            ) as mock_resolve:
                with patch.object(
                    self.module,
                    "train_final_model",
                    return_value=self.temp_dir / "val_metrics.json",
                ):
                    self.module.main([])

        mock_resolve.assert_called_once_with(
            dataset_root=None,
            epochs=None,
            batch_size=None,
            lr=None,
            device=None,
            output_dir=None,
            seed=None,
            val_fraction=None,
        )


if __name__ == "__main__":
    unittest.main()
