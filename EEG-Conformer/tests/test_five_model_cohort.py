from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_activity_five_models.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("cohort_runner_test", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_dataset(root: Path, subjects=(1, 2), excluded=(6,)):
    root.mkdir(parents=True)
    ids = np.repeat(np.array(subjects, dtype=np.int64), 3)
    labels = np.tile(np.arange(3, dtype=np.int64), len(subjects))
    x = np.random.default_rng(42).normal(size=(len(ids), 21, 1920)).astype(np.float32)
    np.save(root / "X.npy", x)
    np.save(root / "y.npy", labels)
    np.save(root / "subject_ids.npy", ids)
    np.save(root / "record_ids.npy", np.arange(len(ids), dtype=np.int64))
    np.save(root / "window_indices.npy", np.zeros(len(ids), dtype=np.int64))
    (root / "metadata.json").write_text(json.dumps({
        "window_seconds": 15, "stride_seconds": 3, "excluded_subject_ids": list(excluded),
        "included_subject_ids": list(subjects), "n_samples": len(ids),
    }))


def arguments(dataset: Path, output: Path, *extra):
    return runner.parse_args(["--dataset-root", str(dataset), "--output-base", str(output),
                              "--cohort-subject-ids", "1,2,6", "--device", "cpu", *extra])


def test_s6_in_arrays_is_rejected_before_dispatch(tmp_path):
    dataset = tmp_path / "unfiltered"
    make_dataset(dataset, subjects=(1, 2, 6))
    with pytest.raises(ValueError, match="Excluded subjects still present"):
        runner.build_plan(arguments(dataset, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_dry_run_keeps_output_absent_and_checks_full_training_cohort(tmp_path, capsys):
    dataset = tmp_path / "cohort_exclude_S6" / "window_15_stride_3"
    make_dataset(dataset)
    args = ["--dataset-root", str(dataset), "--output-base", str(tmp_path / "out"),
            "--cohort-subject-ids", "1,2,6", "--test-subject-ids", "2", "--dry-run"]
    runner.main(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["included_subject_ids"] == [1, 2]
    assert plan["test_subject_ids"] == [2]
    assert plan["planned_training_folds"] == 5
    assert plan["excluded_subject_ids"] == [6]
    assert "exclude_S6" in plan["output_dir"]
    assert not (tmp_path / "out").exists()


def test_both_dispatches_use_same_filtered_arrays_and_historical_mdtf(tmp_path):
    dataset = tmp_path / "cohort_exclude_S6" / "window_15_stride_3"
    make_dataset(dataset)
    plan = runner.build_plan(arguments(dataset, tmp_path / "out"))

    def simulate_external(**kwargs):
        for name in kwargs["models"]:
            folder = kwargs["output_dir"] / name / "seed_42"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text('{"n_folds": 2}')

    with patch.object(runner, "run_loso_batch") as mdtf, \
            patch.object(runner, "run_comparisons", side_effect=simulate_external) as external, \
            patch.object(runner, "summarize", return_value={"n_folds": 2, "per_fold": [],
                         "mean_best_test_acc": 0.5, "std_best_test_acc": 0.0}), \
            patch.object(runner, "collect_environment_info", return_value={}):
        runner.run_plan(plan)
    main_args = mdtf.call_args.kwargs
    other_args = external.call_args.kwargs
    assert main_args["subject_ids"] == other_args["subject_ids"] == [1, 2]
    assert main_args["dataset_root"] == other_args["dataset_root"] == dataset
    assert main_args["transformer_depths"] == [11, 10, 8]
    assert main_args["branch_loss_aux_weight"] == 0.2
    assert main_args["classification_mode"] == "flat"
    assert main_args["class_weights"] == other_args["class_weights"] == [3.0, 3.0, 1.0]
    assert other_args["models"] == ["eegnet", "shallowconvnet", "atcnet", "tcformer"]
    assert len(json.loads((Path(plan["output_dir"]) / "five_model_summary.json").read_text())["models"]) == 5


def test_resume_rejects_modified_dataset_content(tmp_path):
    dataset = tmp_path / "cohort_exclude_S6" / "window_15_stride_3"
    make_dataset(dataset)
    args = arguments(dataset, tmp_path / "out")
    plan = runner.build_plan(args)
    root = Path(plan["output_dir"])
    root.mkdir(parents=True)
    (root / "run_plan.json").write_text(json.dumps(plan))
    x = np.load(dataset / "X.npy")
    x[0, 0, 0] += 1
    np.save(dataset / "X.npy", x)
    changed_plan = runner.build_plan(args)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run_plan(changed_plan, resume=True)
