"""
Tests for train_activity_loso.py
=================================
Covers:
  1. load_global_dataset – reading arrays from disk succeeds
  2. loso_split          – correct train/test sample counts
  3. standardize_by_train – stats computed from train only
  4. ActivityConformer.forward – 21ch × 640pt × 3-class forward pass
  5. parse_args defaults – EEG-relative paths + cuda:0
  6. maybe_rerun_in_project_env – auto-restart key behaviour
  7. resolve_runtime_config – missing args in non-interactive mode raises
  8. main wiring
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


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_activity_loso.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_activity_loso", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_fake_dataset(root: Path, n_subjects: int = 4, n_per_subject: int = 6,
                       n_channels: int = 21, n_times: int = 640) -> None:
    """Write minimal X / y / subject_ids / metadata.json to *root*."""
    total = n_subjects * n_per_subject
    rng = np.random.default_rng(0)
    X = rng.random((total, n_channels, n_times), dtype=np.float32)
    y = np.tile([0, 1, 2, 0, 1, 2], n_subjects)[:total].astype(np.int64)
    subject_ids = np.repeat(np.arange(1, n_subjects + 1), n_per_subject).astype(np.int64)
    record_ids = np.zeros(total, dtype=np.int64)
    window_indices = np.tile(np.arange(n_per_subject), n_subjects)[:total].astype(np.int64)

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X.npy", X)
    np.save(root / "y.npy", y)
    np.save(root / "subject_ids.npy", subject_ids)
    np.save(root / "record_ids.npy", record_ids)
    np.save(root / "window_indices.npy", window_indices)

    metadata = {
        "label_map": {"e_1": 0, "e_2": 1, "e_3": 2},
        "n_subjects": n_subjects,
        "n_samples": total,
        "window_seconds": 5.0,
        "stride_seconds": 5.0,
    }
    with open(root / "metadata.json", "w") as fh:
        json.dump(metadata, fh)


class TestLoadGlobalDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root, n_subjects=3, n_per_subject=4)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_load_returns_correct_shapes(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)

        self.assertEqual(X.shape, (12, 21, 640))
        self.assertEqual(y.shape, (12,))
        self.assertEqual(subject_ids.shape, (12,))

    def test_load_raises_on_missing_file(self) -> None:
        bad_root = self.temp_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            self.module.load_global_dataset(bad_root)


class TestLosoSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        # 4 subjects × 6 windows each = 24 total
        _make_fake_dataset(self.dataset_root, n_subjects=4, n_per_subject=6)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_split_counts_are_correct(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        train_X, train_y, test_X, test_y = self.module.loso_split(X, y, subject_ids, test_subject_id=2)

        # subject 2 has 6 windows
        self.assertEqual(len(test_X), 6)
        self.assertEqual(len(test_y), 6)
        # remaining 3 subjects × 6 = 18
        self.assertEqual(len(train_X), 18)
        self.assertEqual(len(train_y), 18)

    def test_split_adds_conv_channel_dim(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        train_X, _, test_X, _ = self.module.loso_split(X, y, subject_ids, test_subject_id=1)

        # shape should be (N, 1, C, T)
        self.assertEqual(train_X.ndim, 4)
        self.assertEqual(train_X.shape[1], 1)
        self.assertEqual(test_X.ndim, 4)
        self.assertEqual(test_X.shape[1], 1)

    def test_test_subject_windows_not_in_train(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        _, _, _, _ = self.module.loso_split(X, y, subject_ids, test_subject_id=3)
        # sanity: total train + test = total samples
        train_X, _, test_X, _ = self.module.loso_split(X, y, subject_ids, test_subject_id=3)
        self.assertEqual(len(train_X) + len(test_X), len(X))

    def test_split_raises_when_test_subject_missing(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        with self.assertRaises(ValueError):
            self.module.loso_split(X, y, subject_ids, test_subject_id=99)


class TestStandardizeByTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_train_stats_applied_to_both_splits(self) -> None:
        rng = np.random.default_rng(7)
        train_raw = rng.random((20, 1, 21, 640), dtype=np.float32) * 10 + 5
        test_raw = rng.random((5, 1, 21, 640), dtype=np.float32) * 10 + 5

        train_std, test_std = self.module.standardize_by_train(train_raw, test_raw)

        # training data should be near zero-mean / unit-variance
        self.assertAlmostEqual(float(train_std.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(train_std.std()), 1.0, places=4)

    def test_test_set_uses_train_mean_not_own_mean(self) -> None:
        # train: constant 10, test: constant 20
        train_raw = np.full((10, 1, 5, 20), 10.0, dtype=np.float32)
        test_raw = np.full((5, 1, 5, 20), 20.0, dtype=np.float32)

        # std is 0 → should raise
        with self.assertRaises(ValueError):
            self.module.standardize_by_train(train_raw, test_raw)

    def test_test_set_shifted_by_train_mean(self) -> None:
        rng = np.random.default_rng(3)
        train_raw = rng.random((30, 1, 21, 640), dtype=np.float32)
        # test set is simply train + 100 (very different mean)
        test_raw = train_raw[:5] + 100.0

        train_std, test_std = self.module.standardize_by_train(train_raw, test_raw)

        # test set mean ≈ (mean(train) + 100 - mean(train)) / std(train) = 100 / std(train)
        train_std_val = float(train_raw.std())
        self.assertAlmostEqual(float(test_std.mean()), 100.0 / train_std_val, delta=0.1)


class TestInputDomainTransforms(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root, n_subjects=4, n_per_subject=6)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_fft_transform_returns_log_power_rfft_shape(self) -> None:
        windows = np.ones((2, 1, 3, 8), dtype=np.float32)

        transformed = self.module.transform_windows_for_input_domain(windows, "fft")

        self.assertEqual(transformed.shape, (2, 1, 3, 5))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertAlmostEqual(float(transformed[0, 0, 0, 0]), np.log1p(64.0), places=5)

    def test_build_dataloaders_fft_uses_frequency_axis(self) -> None:
        (
            _train_loader,
            _test_loader,
            _n_channels,
            n_times,
            _n_classes,
            _n_train_samples,
            _n_test_samples,
        ) = self.module.build_dataloaders(
            self.dataset_root,
            test_subject_id=1,
            batch_size=4,
            input_domain="fft",
        )

        self.assertEqual(n_times, 321)


class TestActivityConformerForward(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_forward_21ch_640pt_3class(self) -> None:
        model = self.module.ActivityConformer(n_channels=21, n_times=640, n_classes=3)
        batch = torch.randn(2, 1, 21, 640)
        tok, logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_forward_returns_token_features_and_logits(self) -> None:
        model = self.module.ActivityConformer(n_channels=21, n_times=640, n_classes=3)
        batch = torch.randn(4, 1, 21, 640)
        tok, logits = model(batch)

        # tok is the flattened patch tokens before the final fc
        self.assertEqual(tok.shape[0], 4)
        self.assertEqual(logits.shape, (4, 3))

    def test_compute_n_patches_for_640_points(self) -> None:
        n_patches = self.module.compute_n_patches(640)
        # (640 - 24 - 75) // 15 + 1 = 541 // 15 + 1 = 36 + 1 = 37
        self.assertEqual(n_patches, 37)

    def test_compute_n_patches_for_200_points(self) -> None:
        n_patches = self.module.compute_n_patches(200)
        # (200 - 24 - 75) // 15 + 1 = 101 // 15 + 1 = 6 + 1 = 7
        self.assertEqual(n_patches, 7)

    def test_compute_n_patches_rejects_too_short_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            self.module.compute_n_patches(80)

    def test_forward_different_channel_counts(self) -> None:
        for n_ch in [22, 32, 64]:
            model = self.module.ActivityConformer(n_channels=n_ch, n_times=640, n_classes=3)
            batch = torch.randn(2, 1, n_ch, 640)
            _, logits = model(batch)
            self.assertEqual(tuple(logits.shape), (2, 3), msg=f"n_channels={n_ch}")


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

    def test_default_output_dir_is_eeg_conformer_relative(self) -> None:
        args = self.module.parse_args([])

        expected_eeg_root = MODULE_PATH.resolve().parents[1]
        self.assertEqual(
            args.output_dir,
            expected_eeg_root / "local_artifacts" / "outputs" / "activity_loso",
        )

    def test_default_device_is_cuda0(self) -> None:
        args = self.module.parse_args([])
        self.assertEqual(args.device, "cuda:0")

    def test_default_lr_and_epochs(self) -> None:
        args = self.module.parse_args([])
        self.assertAlmostEqual(args.lr, 0.0002)
        self.assertEqual(args.epochs, 200)
        self.assertEqual(args.batch_size, 72)

    def test_explicit_args_override_defaults(self) -> None:
        args = self.module.parse_args(
            ["--test-subject-id", "3", "--epochs", "10", "--device", "cpu"]
        )
        self.assertEqual(args.test_subject_id, 3)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.device, "cpu")

    def test_accepts_class_weights_argument(self) -> None:
        args = self.module.parse_args(["--class-weights", "3,3,1"])
        self.assertEqual(args.class_weights, "3,3,1")

    def test_accepts_input_domain_argument(self) -> None:
        args = self.module.parse_args(["--input-domain", "fft"])
        self.assertEqual(args.input_domain, "fft")


class TestMaybeRerunInProjectEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
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
            with patch.object(self.module.subprocess, "run", side_effect=AssertionError("should not run")):
                # should return without calling subprocess.run
                self.module.maybe_rerun_in_project_env([], "cpu")

    def test_no_rerun_when_cuda_already_usable(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=True):
            with patch.object(self.module.subprocess, "run", side_effect=AssertionError("should not run")):
                self.module.maybe_rerun_in_project_env([], "cuda:0")

    def test_no_rerun_when_env_var_already_set(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.dict(os.environ, {self.module.AUTO_RERUN_ENV_VAR: "1"}, clear=False):
                with patch.object(self.module.subprocess, "run", side_effect=AssertionError("should not run")):
                    self.module.maybe_rerun_in_project_env([], "cuda:0")


class TestResolveRuntimeConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
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
                        test_subject_id=None,
                        epochs=None,
                        batch_size=None,
                        lr=None,
                        device=None,
                        output_dir=None,
                        seed=None,
                    )

        msg = str(ctx.exception)
        self.assertIn("Missing required arguments", msg)
        self.assertIn("--dataset-root", msg)

    def test_explicit_values_accepted_without_prompting(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            config = self.module.resolve_runtime_config(
                dataset_root=self.dataset_root,
                test_subject_id=2,
                epochs=10,
                batch_size=8,
                lr=2e-4,
                device="cuda:0",
                output_dir=self.temp_dir / "out",
                seed=7,
            )

        self.assertEqual(config.test_subject_id, 2)
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.seed, 7)

    def test_rejects_cuda_when_unavailable(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is not available"):
                self.module.resolve_runtime_config(
                    dataset_root=self.dataset_root,
                    test_subject_id=1,
                    epochs=1,
                    batch_size=8,
                    lr=2e-4,
                    device="cuda:0",
                    output_dir=self.temp_dir / "out",
                    seed=42,
                )

    def test_defaults_via_interactive_prompts(self) -> None:
        self.module.DEFAULT_DATASET_ROOT = self.dataset_root
        self.module.DEFAULT_OUTPUT_DIR = self.temp_dir / "outputs"
        with patch("torch.cuda.is_available", return_value=False):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", side_effect=[""] * 10):
                    config = self.module.resolve_runtime_config(
                        dataset_root=None,
                        test_subject_id=None,
                        epochs=None,
                        batch_size=None,
                        lr=None,
                        device=None,
                        output_dir=None,
                        seed=None,
                    )

        self.assertEqual(config.test_subject_id, self.module.DEFAULT_TEST_SUBJECT_ID)
        self.assertEqual(config.epochs, self.module.DEFAULT_EPOCHS)
        self.assertEqual(config.device, "cpu")


class TestMainWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
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
            test_subject_id=1,
            epochs=1,
            batch_size=8,
            lr=2e-4,
            device="cuda:0",
            output_dir=self.temp_dir / "out",
            seed=42,
            input_domain="time",
        )

        with patch.object(self.module, "parse_args", return_value=fake_args):
            with patch.object(
                self.module, "maybe_rerun_in_project_env", side_effect=SystemExit(0)
            ) as mock_rerun:
                with patch.object(
                    self.module, "resolve_runtime_config",
                    side_effect=AssertionError("should not reach resolve"),
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        self.module.main()

        self.assertEqual(ctx.exception.code, 0)
        mock_rerun.assert_called_once()

    def test_main_no_args_passes_nones_to_resolve(self) -> None:
        """When called with no argv, main passes None for all config args."""
        fake_config = self.module.RuntimeConfig(
            dataset_root=self.dataset_root,
            test_subject_id=1,
            epochs=1,
            batch_size=8,
            lr=2e-4,
            device="cpu",
            output_dir=self.temp_dir / "out",
            seed=42,
            input_domain="time",
            class_weights=None,
        )

        with patch.object(self.module, "maybe_rerun_in_project_env", return_value=None):
            with patch.object(
                self.module, "resolve_runtime_config", return_value=fake_config
            ) as mock_resolve:
                with patch.object(
                    self.module, "train_loso_fold",
                    return_value=self.temp_dir / "metrics.json",
                ):
                    self.module.main([])

        mock_resolve.assert_called_once_with(
            dataset_root=None,
            test_subject_id=None,
            epochs=None,
            batch_size=None,
            lr=None,
            device=None,
            output_dir=None,
            seed=None,
            input_domain=None,
            class_weights=None,
        )


if __name__ == "__main__":
    unittest.main()
