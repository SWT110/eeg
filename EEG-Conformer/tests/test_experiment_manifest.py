from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiment_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_manifest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def create_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    X = np.zeros((6, 3, 200), dtype=np.float32)
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    subject_ids = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    record_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    window_indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    np.save(root / "X.npy", X)
    np.save(root / "y.npy", y)
    np.save(root / "subject_ids.npy", subject_ids)
    np.save(root / "record_ids.npy", record_ids)
    np.save(root / "window_indices.npy", window_indices)
    with open(root / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "label_map": {"e_1": 0, "e_2": 1, "e_3": 2},
                "record_id_map": {"0": "1_e_1", "1": "2_e_1"},
                "window_seconds": 3.0,
                "stride_seconds": 3.0,
                "n_records": 2,
            },
            fh,
        )


class TestExperimentManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="manifest-test-"))
        self.dataset_root = self.temp_dir / "datasets" / "window_3_stride_3"
        self.output_dir = self.temp_dir / "outputs" / "window_3_stride_3"
        create_dataset(self.dataset_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fold_dir = self.output_dir / "fold_subject_1"
        fold_dir.mkdir()
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "test_subject_id": 1,
                    "input_domain": "time_fft",
                    "model_type": "dual_branch",
                    "epochs": 5,
                    "batch_size": 4,
                    "lr": 0.0002,
                    "seed": 42,
                    "class_weights": [3.0, 3.0, 1.0],
                    "n_classes": 3,
                    "emb_size": 40,
                    "depth": 6,
                    "num_heads": 5,
                    "dropout": 0.5,
                    "time_n_times": 200,
                    "fft_n_times": 101,
                    "optimizer": "Adam",
                    "optimizer_betas": [0.5, 0.999],
                    "loss": "CrossEntropyLoss",
                },
                fh,
            )
        (self.output_dir / "summary.json").write_text("{}", encoding="utf-8")
        (self.output_dir / "summary.csv").write_text("subject_id\n", encoding="utf-8")
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_collect_dataset_info_includes_counts(self) -> None:
        info = self.module.collect_dataset_info(self.dataset_root)

        self.assertEqual(info["dataset_name"], "window_3_stride_3")
        self.assertEqual(info["X_shape"], [6, 3, 200])
        self.assertEqual(info["class_counts"], {"0": 2, "1": 2, "2": 2})
        self.assertEqual(info["per_subject_counts"], {"1": 3, "2": 3})
        self.assertEqual(info["per_subject_class_counts"]["1"], {"0": 1, "1": 1, "2": 1})

    def test_write_loso_experiment_manifest_json_and_markdown(self) -> None:
        json_path, md_path = self.module.write_loso_experiment_manifest(
            dataset_root=self.dataset_root,
            output_dir=self.output_dir,
            dataset_name="window_3_stride_3",
            input_domain="time_fft",
            training_config={
                "epochs": 5,
                "lr": 0.0002,
                "seed": 42,
                "class_weights": [3.0, 3.0, 1.0],
                "emb_size": 40,
                "depth": 6,
                "num_heads": 5,
                "dropout": 0.5,
                "optimizer_betas": [0.5, 0.999],
            },
            runtime_config={"device": "cpu", "batch_size": 4, "skip_existing": False},
            command_line=["train.py", "--input-domain", "time_fft"],
            run_status="trained",
            run_started_at="2026-01-01T00:00:00+00:00",
            run_ended_at="2026-01-01T00:01:00+00:00",
            project_root=self.temp_dir,
        )

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset"]["window_seconds"], 3.0)
        self.assertEqual(manifest["model"]["model_type"], "dual_branch")
        self.assertEqual(manifest["model"]["fft_n_times"], 101)
        self.assertEqual(manifest["training"]["epochs"], 5)
        self.assertEqual(manifest["split"]["n_folds"], 2)
        self.assertEqual(manifest["runtime"]["batch_size"], 4)
        self.assertIn("# Experiment Manifest", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
