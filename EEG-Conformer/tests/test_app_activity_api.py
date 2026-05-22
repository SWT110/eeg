from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


MODULE_PATH = Path(__file__).resolve().parents[1] / "app_activity_api.py"


def load_module():
    spec = importlib.util.spec_from_file_location("app_activity_api", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDiscoverBestLosoArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="best-loso-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_candidate(self, window_name: str, subject_id: int, best_acc: float, summary_macro_f1: float) -> Path:
        window_dir = self.temp_dir / window_name
        window_dir.mkdir(parents=True, exist_ok=True)
        with open(window_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "macro_f1": summary_macro_f1,
                    "mean_best_test_acc": best_acc,
                    "std_best_test_acc": 0.05,
                },
                fh,
            )
        fold_dir = self.temp_dir / window_name / f"fold_subject_{subject_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "test_subject_id": subject_id,
                    "best_test_acc": best_acc,
                    "macro_f1": summary_macro_f1,
                    "n_channels": 21,
                    "n_times": 128,
                    "n_classes": 3,
                },
                fh,
            )
        (fold_dir / "best_model.pt").write_bytes(b"placeholder")
        return fold_dir

    def test_selects_highest_accuracy_candidate(self) -> None:
        self._write_candidate("window_5_stride_5", subject_id=2, best_acc=0.61, summary_macro_f1=0.48)
        self._write_candidate("window_9_stride_9", subject_id=4, best_acc=0.67, summary_macro_f1=0.50)

        fake_checkpoint = {
            "n_channels": 21,
            "n_times": 128,
            "n_classes": 3,
            "emb_size": 40,
            "depth": 6,
            "num_heads": 5,
        }
        with patch.object(self.module.torch, "load", return_value=fake_checkpoint):
            selected = self.module.discover_best_loso_ensemble(self.temp_dir)

        self.assertEqual(selected.window_dir.name, "window_9_stride_9")
        self.assertAlmostEqual(selected.summary_macro_f1, 0.50)
        self.assertEqual(selected.window_seconds, 9.0)
        self.assertEqual(selected.stride_seconds, 9.0)
        self.assertEqual(len(selected.checkpoints), 1)


class TestApiRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.client = TestClient(self.module.app)

    def test_index_route_serves_html(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("EEG 信号三分类判定", response.text)

    def test_predict_route_rejects_non_xlsx(self) -> None:
        response = self.client.post(
            "/predict/xlsx",
            files={"file": ("bad.txt", b"oops", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only .xlsx files are supported", response.text)

    def test_predict_route_returns_prediction_payload(self) -> None:
        fake_result = {
            "predicted_label": 0,
            "predicted_activity": "e_1",
            "average_probabilities": [0.9, 0.05, 0.05],
            "probabilities_by_activity": {"e_1": 0.9, "e_2": 0.05, "e_3": 0.05},
            "n_windows": 3,
            "model": {"best_test_acc": 0.86},
        }
        with patch.object(self.module, "predict_uploaded_xlsx", return_value=fake_result):
            response = self.client.post(
                "/predict/xlsx",
                files={
                    "file": (
                        "sample.xlsx",
                        b"dummy-xlsx-content",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_activity"], "e_1")
        self.assertEqual(payload["n_windows"], 3)


if __name__ == "__main__":
    unittest.main()
