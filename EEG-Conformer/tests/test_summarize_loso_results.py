"""
Tests for summarize_loso_results.py
=====================================
Covers:
  1. discover_fold_dirs – finds fold_subject_* directories
  2. load_fold_result   – reads metrics.json and test_predictions.npz
  3. confusion_matrix_from_arrays – 3×3 dimension correctness
  4. per_class_metrics_from_cm – precision/recall/f1 per class
  5. macro_f1_from_per_class – average of per-class F1
  6. summarize – produces all required top-level fields
  7. summarize – confusion matrix is 3×3 for 3-class data
  8. summarize – falls back to stored confusion_matrix in metrics.json
  9. write_summary – creates summary.json and summary.csv
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize_loso_results.py"
_PROJECT_ROOT = MODULE_PATH.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_loso_results", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_fold(
    output_dir: Path,
    subject_id: int,
    best_acc: float = 0.6,
    n_classes: int = 3,
    with_predictions: bool = True,
    with_stored_cm: bool = True,
) -> Path:
    """Helper: create a realistic fold directory."""
    fold_dir = output_dir / f"fold_subject_{subject_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(subject_id)
    n_test = 9  # 3 per class
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    y_pred = rng.choice(n_classes, size=n_test).astype(np.int64)

    # Build stored confusion matrix from these predictions
    cm: list[list[int]] = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t)][int(p)] += 1

    metrics: dict = {
        "test_subject_id": subject_id,
        "best_test_acc": best_acc,
        "average_test_acc": best_acc - 0.05,
        "n_train_samples": 60,
        "n_test_samples": n_test,
        "n_classes": n_classes,
        "epochs": 5,
        "batch_size": 8,
        "lr": 0.0002,
        "seed": 42,
    }
    if with_stored_cm:
        metrics["confusion_matrix"] = cm
        metrics["per_class_metrics"] = [
            {"precision": 0.5, "recall": 0.5, "f1": 0.5}
        ] * n_classes
        metrics["macro_f1"] = 0.5

    with open(fold_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh)

    if with_predictions:
        np.savez(fold_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred)

    return fold_dir


class TestHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_confusion_matrix_from_arrays_shape(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 1, 2, 0], dtype=np.int64)
        cm = self.module.confusion_matrix_from_arrays(y_true, y_pred, n_classes=3)
        self.assertEqual(len(cm), 3)
        self.assertTrue(all(len(row) == 3 for row in cm))

    def test_confusion_matrix_diagonal_for_perfect_predictions(self) -> None:
        y = np.array([0, 1, 2], dtype=np.int64)
        cm = self.module.confusion_matrix_from_arrays(y, y, n_classes=3)
        for i in range(3):
            self.assertEqual(cm[i][i], 1)
            for j in range(3):
                if i != j:
                    self.assertEqual(cm[i][j], 0)

    def test_per_class_metrics_keys_present(self) -> None:
        cm = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        per_class = self.module.per_class_metrics_from_cm(cm)
        self.assertEqual(len(per_class), 3)
        for d in per_class:
            self.assertIn("precision", d)
            self.assertIn("recall", d)
            self.assertIn("f1", d)

    def test_per_class_perfect_predictions_gives_f1_one(self) -> None:
        cm = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        per_class = self.module.per_class_metrics_from_cm(cm)
        for d in per_class:
            self.assertAlmostEqual(d["f1"], 1.0, places=5)

    def test_macro_f1_is_mean_of_per_class_f1(self) -> None:
        per_class = [{"f1": 0.4}, {"f1": 0.6}, {"f1": 0.8}]
        macro = self.module.macro_f1_from_per_class(per_class)
        self.assertAlmostEqual(macro, round((0.4 + 0.6 + 0.8) / 3, 6), places=5)

    def test_macro_f1_empty_returns_zero(self) -> None:
        self.assertEqual(self.module.macro_f1_from_per_class([]), 0.0)


class TestDiscoverFoldDirs(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="summarize-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_discovers_fold_dirs_in_sorted_order(self) -> None:
        for sid in [3, 1, 2]:
            (self.temp_dir / f"fold_subject_{sid}").mkdir()
        (self.temp_dir / "not_a_fold").mkdir()

        dirs = self.module.discover_fold_dirs(self.temp_dir)
        self.assertEqual([d.name for d in dirs], [
            "fold_subject_1", "fold_subject_2", "fold_subject_3"
        ])

    def test_returns_empty_list_when_no_folds(self) -> None:
        dirs = self.module.discover_fold_dirs(self.temp_dir)
        self.assertEqual(dirs, [])


class TestLoadFoldResult(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="summarize-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_loads_metrics_json(self) -> None:
        fold_dir = _write_fold(self.temp_dir, subject_id=1, with_predictions=False)
        result = self.module.load_fold_result(fold_dir)
        self.assertEqual(result["test_subject_id"], 1)
        self.assertIn("best_test_acc", result)

    def test_attaches_predictions_when_npz_present(self) -> None:
        fold_dir = _write_fold(self.temp_dir, subject_id=2, with_predictions=True)
        result = self.module.load_fold_result(fold_dir)
        self.assertIn("_y_true", result)
        self.assertIn("_y_pred", result)
        self.assertEqual(len(result["_y_true"]), 9)

    def test_no_predictions_key_when_npz_absent(self) -> None:
        fold_dir = _write_fold(self.temp_dir, subject_id=3, with_predictions=False)
        result = self.module.load_fold_result(fold_dir)
        self.assertNotIn("_y_true", result)

    def test_raises_when_metrics_json_missing(self) -> None:
        fold_dir = self.temp_dir / "fold_subject_99"
        fold_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.module.load_fold_result(fold_dir)


class TestSummarize(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="summarize-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _create_folds(self, n_folds: int = 3, with_predictions: bool = True) -> None:
        for i in range(1, n_folds + 1):
            _write_fold(self.temp_dir, subject_id=i, best_acc=0.5 + i * 0.05,
                        with_predictions=with_predictions)

    def test_raises_when_no_fold_dirs(self) -> None:
        with self.assertRaises(ValueError):
            self.module.summarize(self.temp_dir)

    def test_summary_contains_required_top_level_keys(self) -> None:
        self._create_folds(n_folds=3)
        summary = self.module.summarize(self.temp_dir)

        for key in ["n_folds", "mean_best_test_acc", "std_best_test_acc",
                    "per_fold", "overall_confusion_matrix", "macro_f1",
                    "per_class_metrics"]:
            self.assertIn(key, summary, msg=f"Missing key: {key}")

    def test_n_folds_is_correct(self) -> None:
        self._create_folds(n_folds=4)
        summary = self.module.summarize(self.temp_dir)
        self.assertEqual(summary["n_folds"], 4)

    def test_mean_best_test_acc_is_correct(self) -> None:
        self._create_folds(n_folds=3)
        summary = self.module.summarize(self.temp_dir)
        expected_accs = [0.55, 0.60, 0.65]
        expected_mean = round(sum(expected_accs) / len(expected_accs), 6)
        self.assertAlmostEqual(summary["mean_best_test_acc"], expected_mean, places=4)

    def test_std_best_test_acc_is_present(self) -> None:
        self._create_folds(n_folds=3)
        summary = self.module.summarize(self.temp_dir)
        self.assertIsInstance(summary["std_best_test_acc"], float)

    def test_overall_confusion_matrix_is_3x3(self) -> None:
        self._create_folds(n_folds=3, with_predictions=True)
        summary = self.module.summarize(self.temp_dir)
        cm = summary["overall_confusion_matrix"]
        self.assertIsNotNone(cm)
        self.assertEqual(len(cm), 3, msg="Confusion matrix must have 3 rows")
        self.assertTrue(all(len(row) == 3 for row in cm),
                        msg="Every row must have 3 columns")

    def test_macro_f1_is_not_none(self) -> None:
        self._create_folds(n_folds=3, with_predictions=True)
        summary = self.module.summarize(self.temp_dir)
        self.assertIsNotNone(summary["macro_f1"])
        self.assertIsInstance(summary["macro_f1"], float)

    def test_per_fold_has_correct_subject_ids(self) -> None:
        self._create_folds(n_folds=3)
        summary = self.module.summarize(self.temp_dir)
        ids = [f["subject_id"] for f in summary["per_fold"]]
        self.assertEqual(sorted(ids), [1, 2, 3])

    def test_falls_back_to_stored_confusion_matrix(self) -> None:
        """When test_predictions.npz is absent, use confusion_matrix from metrics.json."""
        self._create_folds(n_folds=2, with_predictions=False)
        summary = self.module.summarize(self.temp_dir)
        # Should still compute overall_cm from stored matrices
        self.assertIsNotNone(summary["overall_confusion_matrix"])
        self.assertEqual(len(summary["overall_confusion_matrix"]), 3)

    def test_per_class_metrics_has_three_entries(self) -> None:
        self._create_folds(n_folds=3, with_predictions=True)
        summary = self.module.summarize(self.temp_dir)
        self.assertEqual(len(summary["per_class_metrics"]), 3)

    def test_per_class_metrics_has_required_fields(self) -> None:
        self._create_folds(n_folds=3, with_predictions=True)
        summary = self.module.summarize(self.temp_dir)
        for entry in summary["per_class_metrics"]:
            self.assertIn("precision", entry)
            self.assertIn("recall", entry)
            self.assertIn("f1", entry)


class TestWriteSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="summarize-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _fake_summary(self) -> dict:
        return {
            "n_folds": 2,
            "mean_best_test_acc": 0.65,
            "std_best_test_acc": 0.05,
            "per_fold": [
                {"subject_id": 1, "best_test_acc": 0.60, "average_test_acc": 0.55,
                 "n_train_samples": 60, "n_test_samples": 9},
                {"subject_id": 2, "best_test_acc": 0.70, "average_test_acc": 0.65,
                 "n_train_samples": 60, "n_test_samples": 9},
            ],
            "overall_confusion_matrix": [[3, 1, 0], [1, 2, 1], [0, 1, 3]],
            "macro_f1": 0.72,
            "per_class_metrics": [
                {"precision": 0.75, "recall": 0.75, "f1": 0.75},
                {"precision": 0.50, "recall": 0.50, "f1": 0.50},
                {"precision": 0.75, "recall": 0.75, "f1": 0.75},
            ],
        }

    def test_creates_summary_json(self) -> None:
        json_path, _ = self.module.write_summary(self._fake_summary(), self.temp_dir)
        self.assertTrue(json_path.exists())

    def test_creates_summary_csv(self) -> None:
        _, csv_path = self.module.write_summary(self._fake_summary(), self.temp_dir)
        self.assertTrue(csv_path.exists())

    def test_json_is_valid_and_contains_macro_f1(self) -> None:
        json_path, _ = self.module.write_summary(self._fake_summary(), self.temp_dir)
        with open(json_path) as fh:
            data = json.load(fh)
        self.assertIn("macro_f1", data)
        self.assertAlmostEqual(data["macro_f1"], 0.72, places=5)

    def test_csv_contains_per_fold_rows(self) -> None:
        _, csv_path = self.module.write_summary(self._fake_summary(), self.temp_dir)
        with open(csv_path, newline="") as fh:
            rows = list(csv.reader(fh))
        # Header + 2 data rows + blank + summary rows
        subject_ids_in_csv = []
        for row in rows[1:]:
            if row and row[0].isdigit():
                subject_ids_in_csv.append(int(row[0]))
        self.assertEqual(sorted(subject_ids_in_csv), [1, 2])

    def test_json_overall_confusion_matrix_is_3x3(self) -> None:
        json_path, _ = self.module.write_summary(self._fake_summary(), self.temp_dir)
        with open(json_path) as fh:
            data = json.load(fh)
        cm = data["overall_confusion_matrix"]
        self.assertEqual(len(cm), 3)
        self.assertTrue(all(len(row) == 3 for row in cm))


if __name__ == "__main__":
    unittest.main()
