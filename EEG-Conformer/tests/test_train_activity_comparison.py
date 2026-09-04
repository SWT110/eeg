from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from train_activity_comparison import (  # noqa: E402
    build_dry_run_plan,
    completed_fold_matches,
    parse_int_list,
    parse_model_list,
    run_comparisons,
)


def make_dataset(root: Path) -> None:
    rng = np.random.default_rng(7)
    n_subjects = 3
    samples_per_subject = 6
    total = n_subjects * samples_per_subject
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X.npy", rng.normal(size=(total, 21, 512)).astype(np.float32))
    np.save(root / "y.npy", np.tile([0, 1, 2, 0, 1, 2], n_subjects).astype(np.int64))
    np.save(
        root / "subject_ids.npy",
        np.repeat(np.arange(1, n_subjects + 1), samples_per_subject).astype(np.int64),
    )


class TestComparisonRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="comparison-runner-"))
        self.dataset = self.temp_dir / "dataset"
        self.output = self.temp_dir / "output"
        make_dataset(self.dataset)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_list_parsers(self) -> None:
        self.assertEqual(
            parse_model_list("EEGNet, shallow-net,TCFormer"),
            ["eegnet", "shallowconvnet", "tcformer"],
        )
        self.assertEqual(parse_int_list("42,43", "seeds"), [42, 43])
        with self.assertRaises(ValueError):
            parse_int_list("42,42", "seeds")

    def test_dry_run_uses_real_dataset_shape(self) -> None:
        plan = build_dry_run_plan(
            self.dataset,
            ["eegnet", "tcformer"],
            [1, 2, 3],
            [42],
            200,
            72,
            0.0002,
            [3.0, 3.0, 1.0],
            "cuda:0",
        )
        self.assertEqual(plan["dataset_shape"], [18, 21, 512])
        self.assertEqual(plan["planned_folds"], 6)
        self.assertEqual(len(plan["model_plans"]), 2)
        self.assertFalse(plan["protocol"]["eligible_for_manuscript_table"])

    def test_one_epoch_end_to_end_writes_compatible_artifacts(self) -> None:
        json_path, csv_path = run_comparisons(
            dataset_root=self.dataset,
            output_dir=self.output,
            models=["eegnet"],
            subject_ids=[1],
            seeds=[42],
            epochs=1,
            batch_size=3,
            lr=0.0002,
            class_weights=[3.0, 3.0, 1.0],
            device="cpu",
            skip_existing=False,
            resume=True,
        )
        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        metrics_path = self.output / "eegnet" / "seed_42" / "fold_subject_1" / "metrics.json"
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        self.assertEqual(metrics["architecture"], "eegnet")
        self.assertEqual(metrics["input_domain"], "time")
        self.assertEqual(metrics["class_weights"], [3.0, 3.0, 1.0])
        self.assertIn("macro_f1", metrics)
        with open(json_path, encoding="utf-8") as handle:
            comparison_summary = json.load(handle)
        self.assertFalse(
            comparison_summary["protocol"]["eligible_for_manuscript_table"]
        )
        self.assertTrue(
            completed_fold_matches(
                metrics_path,
                architecture="eegnet",
                seed=42,
                epochs=1,
                batch_size=3,
                lr=0.0002,
                class_weights=[3.0, 3.0, 1.0],
            )
        )
        (metrics_path.parent / "test_predictions.npz").unlink()
        self.assertFalse(
            completed_fold_matches(
                metrics_path,
                architecture="eegnet",
                seed=42,
                epochs=1,
                batch_size=3,
                lr=0.0002,
                class_weights=[3.0, 3.0, 1.0],
            )
        )


if __name__ == "__main__":
    unittest.main()
