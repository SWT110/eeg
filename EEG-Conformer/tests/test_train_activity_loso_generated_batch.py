from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_activity_loso_generated_batch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_activity_loso_generated_batch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def create_dataset(root: Path, subject_ids: list[int] | None = None) -> None:
    subject_ids = [1, 2] if subject_ids is None else subject_ids
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X.npy", np.zeros((len(subject_ids), 21, 640), dtype=np.float32))
    np.save(root / "y.npy", np.zeros(len(subject_ids), dtype=np.int64))
    np.save(root / "subject_ids.npy", np.array(subject_ids, dtype=np.int64))
    with open(root / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"n_subjects": len(set(subject_ids))}, fh)


class TestLoadConfigJobs(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="generated-batch-"))
        self.config_path = self.temp_dir / "configs.json"
        self.dataset_base = self.temp_dir / "datasets"
        self.output_base = self.temp_dir / "outputs"
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_skips_invalid_items_and_deduplicates_dataset_names(self) -> None:
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {"window_seconds": 1.0, "stride_seconds": 1.0},
                    {"window_seconds": 1, "stride_seconds": 1},
                    {"window_seconds": 3.0},
                    {"stride_seconds": 2.0},
                    {"window_seconds": -1.0, "stride_seconds": 1.0},
                    "bad-item",
                ],
                fh,
            )

        jobs = self.module.load_config_jobs(self.config_path, self.dataset_base, self.output_base)

        self.assertEqual([job.dataset_name for job in jobs], ["window_1_stride_1", "window_3_stride_3"])
        self.assertEqual(jobs[0].dataset_root, self.dataset_base / "window_1_stride_1")
        self.assertEqual(jobs[1].output_dir, self.output_base / "window_3_stride_3")


class TestDatasetOutputIsComplete(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="generated-batch-"))
        self.dataset_root = self.temp_dir / "dataset"
        self.output_dir = self.temp_dir / "output"
        self.module = load_module()
        create_dataset(self.dataset_root, subject_ids=[1, 1, 2])

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_requires_summary_and_all_fold_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text("{}")
        (self.output_dir / "summary.csv").write_text("subject_id\n")

        fake_train = SimpleNamespace(
            discover_subject_ids_from_global_dataset=lambda _: [1, 2],
            fold_is_complete=lambda output_dir, subject_id: subject_id == 1,
        )
        with patch.object(self.module, "_load_train_batch_module", return_value=fake_train):
            self.assertFalse(self.module.dataset_output_is_complete(self.dataset_root, self.output_dir))

        fake_train = SimpleNamespace(
            discover_subject_ids_from_global_dataset=lambda _: [1, 2],
            fold_is_complete=lambda output_dir, subject_id: True,
        )
        with patch.object(self.module, "_load_train_batch_module", return_value=fake_train):
            self.assertTrue(self.module.dataset_output_is_complete(self.dataset_root, self.output_dir))


class TestRunGeneratedDatasetBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="generated-batch-"))
        self.dataset_root = self.temp_dir / "datasets" / "window_1_stride_1"
        self.output_dir = self.temp_dir / "outputs" / "window_1_stride_1"
        self.module = load_module()
        create_dataset(self.dataset_root, subject_ids=[1, 2, 2])
        self.job = self.module.DatasetJob(
            dataset_name="window_1_stride_1",
            dataset_root=self.dataset_root,
            output_dir=self.output_dir,
            window_seconds=1.0,
            stride_seconds=1.0,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_calls_inner_batch_runner_and_summary_writer(self) -> None:
        fake_train = SimpleNamespace(
            discover_subject_ids_from_global_dataset=lambda _: [1, 2],
            fold_is_complete=lambda output_dir, subject_id: False,
            run_loso_batch=lambda **kwargs: [kwargs],
        )
        fake_summary = SimpleNamespace(
            summarize=lambda output_dir: {"n_folds": 2},
            write_summary=lambda summary, output_dir: (Path(output_dir) / "summary.json", Path(output_dir) / "summary.csv"),
        )

        with patch.object(self.module, "_load_train_batch_module", return_value=fake_train):
            with patch.object(self.module, "_load_summary_module", return_value=fake_summary):
                results = self.module.run_generated_dataset_batch(
                    jobs=[self.job],
                    epochs=5,
                    batch_size=8,
                    lr=2e-4,
                    device="cpu",
                    skip_existing=True,
                    seed=7,
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "trained")
        self.assertEqual(results[0].summary_json, self.output_dir / "summary.json")
        self.assertTrue((self.output_dir / "experiment_manifest.json").exists())
        self.assertTrue((self.output_dir / "experiment_manifest.md").exists())

    def test_skips_fully_complete_dataset(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text("{}")
        (self.output_dir / "summary.csv").write_text("subject_id\n")

        fake_train = SimpleNamespace(
            discover_subject_ids_from_global_dataset=lambda _: [1, 2],
            fold_is_complete=lambda output_dir, subject_id: True,
            run_loso_batch=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not train")),
        )

        with patch.object(self.module, "_load_train_batch_module", return_value=fake_train):
            results = self.module.run_generated_dataset_batch(
                jobs=[self.job],
                epochs=5,
                batch_size=8,
                lr=2e-4,
                device="cpu",
                skip_existing=True,
                seed=7,
            )

        self.assertEqual(results[0].status, "skipped")
        self.assertEqual(results[0].summary_json, self.output_dir / "summary.json")

    def test_missing_dataset_files_raise_batch_error(self) -> None:
        incomplete_job = self.module.DatasetJob(
            dataset_name="window_3_stride_3",
            dataset_root=self.temp_dir / "datasets" / "window_3_stride_3",
            output_dir=self.temp_dir / "outputs" / "window_3_stride_3",
            window_seconds=3.0,
            stride_seconds=3.0,
        )

        fake_train = SimpleNamespace(
            discover_subject_ids_from_global_dataset=lambda _: [1],
            fold_is_complete=lambda output_dir, subject_id: False,
            run_loso_batch=lambda **kwargs: None,
        )

        with patch.object(self.module, "_load_train_batch_module", return_value=fake_train):
            with self.assertRaises(RuntimeError) as ctx:
                self.module.run_generated_dataset_batch(
                    jobs=[incomplete_job],
                    epochs=5,
                    batch_size=8,
                    lr=2e-4,
                    device="cpu",
                    skip_existing=True,
                    seed=7,
                )

        self.assertIn("window_3_stride_3", str(ctx.exception))

    def test_passes_class_weights_to_inner_batch_runner(self) -> None:
        captured_kwargs: dict[str, object] = {}

        def fake_run_loso_batch(**kwargs):
            captured_kwargs.update(kwargs)
            return [kwargs]

        fake_train = SimpleNamespace(
            discover_subject_ids_from_global_dataset=lambda _: [1, 2],
            fold_is_complete=lambda output_dir, subject_id: False,
            run_loso_batch=fake_run_loso_batch,
        )
        fake_summary = SimpleNamespace(
            summarize=lambda output_dir: {"n_folds": 2},
            write_summary=lambda summary, output_dir: (Path(output_dir) / "summary.json", Path(output_dir) / "summary.csv"),
        )

        with patch.object(self.module, "_load_train_batch_module", return_value=fake_train):
            with patch.object(self.module, "_load_summary_module", return_value=fake_summary):
                self.module.run_generated_dataset_batch(
                    jobs=[self.job],
                    epochs=5,
                    batch_size=8,
                    lr=2e-4,
                    device="cpu",
                    skip_existing=True,
                    seed=7,
                    class_weights=[3.0, 3.0, 1.0],
                )

        self.assertEqual(captured_kwargs["class_weights"], [3.0, 3.0, 1.0])

    def test_passes_input_domain_to_inner_batch_runner(self) -> None:
        captured_kwargs: dict[str, object] = {}

        def fake_run_loso_batch(**kwargs):
            captured_kwargs.update(kwargs)
            return [kwargs]

        fake_train = SimpleNamespace(
            discover_subject_ids_from_global_dataset=lambda _: [1, 2],
            fold_is_complete=lambda output_dir, subject_id: False,
            run_loso_batch=fake_run_loso_batch,
        )
        fake_summary = SimpleNamespace(
            summarize=lambda output_dir: {"n_folds": 2},
            write_summary=lambda summary, output_dir: (Path(output_dir) / "summary.json", Path(output_dir) / "summary.csv"),
        )

        with patch.object(self.module, "_load_train_batch_module", return_value=fake_train):
            with patch.object(self.module, "_load_summary_module", return_value=fake_summary):
                self.module.run_generated_dataset_batch(
                    jobs=[self.job],
                    epochs=5,
                    batch_size=8,
                    lr=2e-4,
                    device="cpu",
                    skip_existing=True,
                    seed=7,
                    input_domain="fft",
                )

        self.assertEqual(captured_kwargs["input_domain"], "fft")


class TestParseArgs(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_defaults_point_to_generated_dataset_layout(self) -> None:
        args = self.module.parse_args([])
        expected_eeg_root = MODULE_PATH.resolve().parents[1]
        self.assertEqual(
            args.config,
            expected_eeg_root / "eeg-data-processing" / "data_to_list" / "window_stride_configs.json",
        )
        self.assertEqual(
            args.dataset_base,
            expected_eeg_root / "local_artifacts" / "data_to_list" / "global_activity_dataset",
        )
        self.assertEqual(
            args.output_base,
            expected_eeg_root / "local_artifacts" / "outputs" / "activity_loso",
        )
        self.assertTrue(args.skip_existing)
        self.assertEqual(args.device, "cuda:0")

    def test_accepts_class_weights_argument(self) -> None:
        args = self.module.parse_args(["--class-weights", "3,3,1"])
        self.assertEqual(args.class_weights, "3,3,1")

    def test_accepts_input_domain_argument(self) -> None:
        args = self.module.parse_args(["--input-domain", "fft"])
        self.assertEqual(args.input_domain, "fft")

    def test_accepts_time_fft_input_domain_argument(self) -> None:
        args = self.module.parse_args(["--input-domain", "time_fft"])
        self.assertEqual(args.input_domain, "time_fft")


if __name__ == "__main__":
    unittest.main()
