"""
Tests for predict_activity_signal.py
======================================
Covers:
  1. softmax_average_probabilities – multi-window averaging is mathematically correct
  2. softmax_average_probabilities – single window matches plain softmax
  3. softmax_average_probabilities – normalisation applied before model call
  4. extract_windows_from_xlsx – returns (N, C, T) float32 from synthetic xlsx
  5. extract_windows_from_xlsx – returns None when load_xlsx_file returns None
  6. load_model_artifacts – loads all 4 files correctly
  7. load_model_artifacts – raises FileNotFoundError for missing files
  8. build_model_from_artifacts – creates model with correct class count
  9. predict_signal – output dict has all required keys
 10. predict_signal – predicted_label matches argmax of average_probabilities
 11. predict_signal – saves JSON to output_json when specified
 12. predict_signal – raises ValueError when no windows extracted
 13. parse_args – --input is required; --model-dir has default; --device defaults to cpu
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = Path(__file__).resolve().parents[1] / "predict_activity_signal.py"


def load_module():
    spec = importlib.util.spec_from_file_location("predict_activity_signal", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Minimal stub models for testing
# ---------------------------------------------------------------------------

class FixedLogitsModel(nn.Module):
    """Always returns fixed logits regardless of input."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        logits = self._logits.unsqueeze(0).expand(B, -1)
        return x, logits


def _make_fake_artifacts(
    root: Path,
    n_channels: int = 5,
    n_times: int = 100,
    n_classes: int = 3,
    window_seconds: float = 1.0,
    stride_seconds: float = 1.0,
    input_domain: str = "time",
) -> None:
    """Write minimal final-model artifact files to *root*.

    Note: final_model.pt is NOT written here – tests that need the full
    model file write it separately.
    """
    root.mkdir(parents=True, exist_ok=True)

    train_config = {
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "emb_size": 40,
        "depth": 6,
        "num_heads": 5,
        "dropout": 0.5,
        "epochs": 1,
        "batch_size": 8,
        "lr": 0.0002,
        "seed": 42,
        "val_fraction": 0.2,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "input_domain": input_domain,
    }
    with open(root / "train_config.json", "w") as fh:
        json.dump(train_config, fh)

    with open(root / "normalization_stats.json", "w") as fh:
        json.dump({"mean": 0.5, "std": 0.3}, fh)

    label_map = {"e_1": 0, "e_2": 1, "e_3": 2}
    with open(root / "label_map.json", "w") as fh:
        json.dump(label_map, fh)


def _write_fake_model_pt(
    root: Path,
    n_channels: int = 5,
    n_times: int = 100,
    n_classes: int = 3,
) -> None:
    """Instantiate a tiny ActivityConformer and save its weights."""
    # Import from the predict module to reuse the same ActivityConformer
    mod = load_module()
    model = mod.ActivityConformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        emb_size=40,
        depth=6,
        num_heads=5,
        dropout=0.0,
    )
    torch.save(
        {
            "epoch": 0,
            "state_dict": model.state_dict(),
            "n_channels": n_channels,
            "n_times": n_times,
            "n_classes": n_classes,
            "emb_size": 40,
            "depth": 6,
            "num_heads": 5,
        },
        root / "final_model.pt",
    )


def _make_fake_xlsx_data(
    n_samples: int = 200,
    n_channels: int = 5,
    sampling_hz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, times) suitable for passing to extract_windows."""
    rng = np.random.default_rng(0)
    times = np.arange(n_samples) / sampling_hz
    data = rng.random((n_samples, n_channels), dtype=np.float64)
    return data, times


# ---------------------------------------------------------------------------
# 1-3. softmax_average_probabilities
# ---------------------------------------------------------------------------

class TestSoftmaxAverageProbabilities(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def _avg(self, model, n_windows: int = 3, n_ch: int = 5, n_t: int = 50):
        windows = np.zeros((n_windows, n_ch, n_t), dtype=np.float32)
        return self.module.softmax_average_probabilities(
            model, windows, mean=0.0, std=1.0, device="cpu"
        )

    def test_fixed_logits_avg_equals_plain_softmax(self) -> None:
        """With identical logits per window, avg == softmax(logits)."""
        logits = [1.5, 0.3, -0.8]
        model = FixedLogitsModel(logits)
        avg = self._avg(model)
        expected = F.softmax(torch.tensor(logits), dim=0).numpy()
        np.testing.assert_allclose(avg, expected, atol=1e-5)

    def test_single_window_matches_softmax(self) -> None:
        """Single window: avg probs == softmax of logits for that window."""
        logits = [0.1, 2.5, -1.0]
        model = FixedLogitsModel(logits)
        windows = np.zeros((1, 3, 20), dtype=np.float32)
        avg = self.module.softmax_average_probabilities(
            model, windows, mean=0.0, std=1.0, device="cpu"
        )
        expected = F.softmax(torch.tensor(logits), dim=0).numpy()
        np.testing.assert_allclose(avg, expected, atol=1e-5)

    def test_normalisation_applied_before_model(self) -> None:
        """Check that mean/std are actually applied: raw windows != normalised windows."""

        class CaptureInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.received: list[torch.Tensor] = []

            def forward(self, x):
                self.received.append(x.detach().clone())
                B = x.shape[0]
                return x, torch.zeros(B, 3)

        spy = CaptureInputModel()
        raw_windows = np.full((2, 3, 10), 10.0, dtype=np.float32)
        self.module.softmax_average_probabilities(
            spy, raw_windows, mean=5.0, std=2.0, device="cpu"
        )
        seen = spy.received[0]
        # After normalisation: (10 - 5) / 2 = 2.5
        self.assertAlmostEqual(float(seen[0, 0, 0, 0]), 2.5, places=4)

    def test_multi_window_average_is_mean_of_per_window_probs(self) -> None:
        """Two windows with different logits: avg == mean of two softmaxes."""

        call_count = [0]

        class AlternatingModel(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                results = []
                for i in range(B):
                    idx = call_count[0] % 2
                    call_count[0] += 1
                    if idx == 0:
                        results.append(torch.tensor([2.0, 0.0, 0.0]))
                    else:
                        results.append(torch.tensor([0.0, 2.0, 0.0]))
                return x, torch.stack(results, dim=0)

        model = AlternatingModel()
        windows = np.zeros((2, 3, 10), dtype=np.float32)
        avg = self.module.softmax_average_probabilities(
            model, windows, mean=0.0, std=1.0, device="cpu", batch_size=1
        )

        p0 = F.softmax(torch.tensor([2.0, 0.0, 0.0]), dim=0).numpy()
        p1 = F.softmax(torch.tensor([0.0, 2.0, 0.0]), dim=0).numpy()
        expected = (p0 + p1) / 2.0
        np.testing.assert_allclose(avg, expected, atol=1e-5)

    def test_fft_input_domain_transforms_to_frequency_axis(self) -> None:
        class CaptureInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.received: list[torch.Tensor] = []

            def forward(self, x):
                self.received.append(x.detach().clone())
                return x, torch.zeros(x.shape[0], 3)

        spy = CaptureInputModel()
        raw_windows = np.ones((1, 2, 8), dtype=np.float32)

        self.module.softmax_average_probabilities(
            spy,
            raw_windows,
            mean=0.0,
            std=1.0,
            device="cpu",
            input_domain="fft",
        )

        seen = spy.received[0]
        self.assertEqual(tuple(seen.shape), (1, 1, 2, 5))
        self.assertAlmostEqual(float(seen[0, 0, 0, 0]), np.log1p(64.0), places=5)


# ---------------------------------------------------------------------------
# 4-5. extract_windows_from_xlsx
# ---------------------------------------------------------------------------

class TestExtractWindowsFromXlsx(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_returns_nct_float32_array(self) -> None:
        data, times = _make_fake_xlsx_data(n_samples=200, n_channels=5, sampling_hz=100.0)
        with patch.object(self.module, "load_xlsx_file", return_value=(data, times)):
            result = self.module.extract_windows_from_xlsx(
                "dummy.xlsx", window_seconds=1.0, stride_seconds=1.0
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.ndim, 3)
        # 200 samples at 100 Hz → 2 windows of 100 samples
        n_windows, n_ch, n_t = result.shape
        self.assertEqual(n_windows, 2)
        self.assertEqual(n_ch, 5)
        self.assertEqual(n_t, 100)

    def test_returns_none_when_load_fails(self) -> None:
        with patch.object(self.module, "load_xlsx_file", return_value=None):
            result = self.module.extract_windows_from_xlsx(
                "bad.xlsx", window_seconds=1.0, stride_seconds=1.0
            )
        self.assertIsNone(result)

    def test_nct_transpose_correct(self) -> None:
        """extract_windows returns (T, C) per window; our wrapper transposes to (C, T)."""
        n_samples, n_ch = 100, 7
        data = np.arange(n_samples * n_ch, dtype=np.float64).reshape(n_samples, n_ch)
        times = np.arange(n_samples) / 100.0  # 100 Hz

        with patch.object(self.module, "load_xlsx_file", return_value=(data, times)):
            result = self.module.extract_windows_from_xlsx(
                "dummy.xlsx", window_seconds=1.0, stride_seconds=1.0
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape[1], n_ch)   # channels dim


# ---------------------------------------------------------------------------
# 6-7. load_model_artifacts
# ---------------------------------------------------------------------------

class TestLoadModelArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="predict-test-"))
        self.model_dir = self.temp_dir / "model"
        _make_fake_artifacts(self.model_dir, n_channels=5, n_times=100, n_classes=3)
        # write a dummy final_model.pt so all files are present
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "final_model.pt").write_bytes(b"dummy")
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_loads_all_artifacts(self) -> None:
        arts = self.module.load_model_artifacts(self.model_dir)
        self.assertIn("train_config", arts)
        self.assertIn("norm_stats", arts)
        self.assertIn("label_map", arts)
        self.assertIn("model_dir", arts)

    def test_train_config_has_required_keys(self) -> None:
        arts = self.module.load_model_artifacts(self.model_dir)
        for key in ("n_channels", "n_times", "n_classes", "window_seconds"):
            self.assertIn(key, arts["train_config"])

    def test_norm_stats_has_mean_and_std(self) -> None:
        arts = self.module.load_model_artifacts(self.model_dir)
        self.assertIn("mean", arts["norm_stats"])
        self.assertIn("std", arts["norm_stats"])

    def test_raises_on_missing_file(self) -> None:
        bad = self.temp_dir / "empty_model"
        bad.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.module.load_model_artifacts(bad)

    def test_raises_on_missing_label_map(self) -> None:
        (self.model_dir / "label_map.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.module.load_model_artifacts(self.model_dir)


# ---------------------------------------------------------------------------
# 8. build_model_from_artifacts
# ---------------------------------------------------------------------------

class TestBuildModelFromArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="predict-test-"))
        self.model_dir = self.temp_dir / "model"
        _make_fake_artifacts(self.model_dir, n_channels=5, n_times=100, n_classes=3)
        _write_fake_model_pt(self.model_dir, n_channels=5, n_times=100, n_classes=3)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_model_has_correct_output_dim(self) -> None:
        arts = self.module.load_model_artifacts(self.model_dir)
        model = self.module.build_model_from_artifacts(arts, device="cpu")
        dummy = torch.zeros(1, 1, 5, 100)
        with torch.no_grad():
            _, logits = model(dummy)
        self.assertEqual(logits.shape, (1, 3))

    def test_model_is_in_eval_mode(self) -> None:
        arts = self.module.load_model_artifacts(self.model_dir)
        model = self.module.build_model_from_artifacts(arts, device="cpu")
        self.assertFalse(model.training)


# ---------------------------------------------------------------------------
# 9-12. predict_signal (end-to-end with mocked xlsx + real model)
# ---------------------------------------------------------------------------

class TestPredictSignal(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="predict-test-"))
        self.model_dir = self.temp_dir / "model"
        # 5 channels, 100 time points, 3 classes
        _make_fake_artifacts(
            self.model_dir, n_channels=5, n_times=100, n_classes=3,
            window_seconds=1.0, stride_seconds=1.0,
        )
        _write_fake_model_pt(self.model_dir, n_channels=5, n_times=100, n_classes=3)
        self.module = load_module()

        # Fake xlsx data: 200 samples at 100 Hz → 2 windows of 100 samples × 5 ch
        self._data, self._times = _make_fake_xlsx_data(
            n_samples=200, n_channels=5, sampling_hz=100.0
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _patch_load(self):
        return patch.object(
            self.module, "load_xlsx_file",
            return_value=(self._data, self._times),
        )

    def test_result_has_required_keys(self) -> None:
        with self._patch_load():
            result = self.module.predict_signal(
                xlsx_path="dummy.xlsx",
                model_dir=self.model_dir,
                device="cpu",
            )
        for key in ("predicted_label", "predicted_activity", "average_probabilities", "n_windows"):
            self.assertIn(key, result)

    def test_predicted_label_matches_argmax_of_avg_probs(self) -> None:
        with self._patch_load():
            result = self.module.predict_signal(
                xlsx_path="dummy.xlsx",
                model_dir=self.model_dir,
                device="cpu",
            )
        probs = result["average_probabilities"]
        expected_label = int(np.argmax(probs))
        self.assertEqual(result["predicted_label"], expected_label)

    def test_n_windows_matches_extracted_count(self) -> None:
        # 200 samples at 100 Hz with 1s windows → 2 windows
        with self._patch_load():
            result = self.module.predict_signal(
                xlsx_path="dummy.xlsx",
                model_dir=self.model_dir,
                device="cpu",
            )
        self.assertEqual(result["n_windows"], 2)

    def test_predicted_activity_is_string(self) -> None:
        with self._patch_load():
            result = self.module.predict_signal(
                xlsx_path="dummy.xlsx",
                model_dir=self.model_dir,
                device="cpu",
            )
        self.assertIsInstance(result["predicted_activity"], str)

    def test_average_probabilities_sum_to_one(self) -> None:
        with self._patch_load():
            result = self.module.predict_signal(
                xlsx_path="dummy.xlsx",
                model_dir=self.model_dir,
                device="cpu",
            )
        total = sum(result["average_probabilities"])
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_output_json_saved(self) -> None:
        out_json = self.temp_dir / "prediction.json"
        with self._patch_load():
            self.module.predict_signal(
                xlsx_path="dummy.xlsx",
                model_dir=self.model_dir,
                device="cpu",
                output_json=out_json,
            )
        self.assertTrue(out_json.exists())
        with open(out_json) as fh:
            saved = json.load(fh)
        self.assertIn("predicted_label", saved)

    def test_raises_when_no_windows_extracted(self) -> None:
        with patch.object(self.module, "load_xlsx_file", return_value=None):
            with self.assertRaises(ValueError):
                self.module.predict_signal(
                    xlsx_path="empty.xlsx",
                    model_dir=self.model_dir,
                    device="cpu",
                )

    def test_uses_input_domain_from_train_config(self) -> None:
        fft_model_dir = self.temp_dir / "model_fft"
        _make_fake_artifacts(
            fft_model_dir,
            n_channels=5,
            n_times=101,
            n_classes=3,
            window_seconds=1.0,
            stride_seconds=1.0,
            input_domain="fft",
        )
        (fft_model_dir / "final_model.pt").write_bytes(b"dummy")

        with self._patch_load():
            with patch.object(self.module, "build_model_from_artifacts", return_value=MagicMock()):
                with patch.object(
                    self.module,
                    "softmax_average_probabilities",
                    return_value=np.array([1.0, 0.0, 0.0]),
                ) as mock_probs:
                    self.module.predict_signal(
                        xlsx_path="dummy.xlsx",
                        model_dir=fft_model_dir,
                        device="cpu",
                    )

        self.assertEqual(mock_probs.call_args.kwargs["input_domain"], "fft")


# ---------------------------------------------------------------------------
# 13. parse_args
# ---------------------------------------------------------------------------

class TestParseArgs(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_input_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            self.module.parse_args([])

    def test_model_dir_default_is_activity_final(self) -> None:
        args = self.module.parse_args(["--input", "/some/file.xlsx"])
        expected = MODULE_PATH.resolve().parents[1] / "local_artifacts" / "outputs" / "activity_final"
        self.assertEqual(args.model_dir, expected)

    def test_device_defaults_to_cpu(self) -> None:
        args = self.module.parse_args(["--input", "/some/file.xlsx"])
        self.assertEqual(args.device, "cpu")

    def test_explicit_args_parsed(self) -> None:
        args = self.module.parse_args([
            "--input", "/path/to/signal.xlsx",
            "--model-dir", "/path/to/model",
            "--device", "cuda:0",
            "--window-seconds", "2.5",
        ])
        self.assertEqual(str(args.input), "/path/to/signal.xlsx")
        self.assertEqual(str(args.model_dir), "/path/to/model")
        self.assertEqual(args.device, "cuda:0")
        self.assertAlmostEqual(args.window_seconds, 2.5)


if __name__ == "__main__":
    unittest.main()
