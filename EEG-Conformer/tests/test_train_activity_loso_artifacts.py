from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_activity_loso.py"


def _make_torch_stub_modules() -> dict[str, types.ModuleType]:
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.device = lambda name: name
    torch_mod.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda self: None,
        __exit__=lambda self, exc_type, exc, tb: False,
    )
    torch_mod.manual_seed = lambda seed: None
    torch_mod.einsum = lambda *args, **kwargs: None
    torch_mod.finfo = lambda dtype: types.SimpleNamespace(min=-1e9)
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed=lambda seed: None,
        manual_seed_all=lambda seed: None,
    )
    torch_mod.optim = types.SimpleNamespace(Adam=object)

    nn_mod = types.ModuleType("torch.nn")

    class DummyModule:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def to(self, *args, **kwargs):
            return self

    class DummySequential(DummyModule):
        pass

    for name in [
        "Conv2d",
        "BatchNorm2d",
        "ELU",
        "AvgPool2d",
        "Dropout",
        "Linear",
        "LayerNorm",
        "GELU",
        "CrossEntropyLoss",
    ]:
        setattr(nn_mod, name, DummyModule)
    nn_mod.Module = DummyModule
    nn_mod.Sequential = DummySequential

    functional_mod = types.ModuleType("torch.nn.functional")
    functional_mod.softmax = lambda *args, **kwargs: None

    utils_mod = types.ModuleType("torch.utils")
    utils_data_mod = types.ModuleType("torch.utils.data")
    utils_data_mod.DataLoader = DummyModule
    utils_data_mod.TensorDataset = DummyModule

    return {
        "torch": torch_mod,
        "torch.nn": nn_mod,
        "torch.nn.functional": functional_mod,
        "torch.utils": utils_mod,
        "torch.utils.data": utils_data_mod,
    }


@contextmanager
def torch_stubbed():
    injected = _make_torch_stub_modules()
    original = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        yield
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_module():
    with torch_stubbed():
        spec = importlib.util.spec_from_file_location("train_activity_loso", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


class TestEpochHistoryArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-artifacts-"))
        self.fold_dir = self.temp_dir / "fold_subject_1"
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_write_epoch_history_files_creates_csv_and_json(self) -> None:
        history = [
            {
                "epoch": 1,
                "train_loss": 0.8,
                "train_acc": 0.5,
                "test_loss": 0.9,
                "test_acc": 0.4,
                "best_test_acc": 0.4,
                "is_best_epoch": True,
            },
            {
                "epoch": 2,
                "train_loss": 0.6,
                "train_acc": 0.7,
                "test_loss": 0.7,
                "test_acc": 0.6,
                "best_test_acc": 0.6,
                "is_best_epoch": True,
            },
        ]
        metadata = {"test_subject_id": 1, "epochs": 2}

        csv_path, json_path = self.module.write_epoch_history_files(self.fold_dir, history, metadata)

        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())

        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["epoch"], "1")
        self.assertEqual(rows[1]["best_test_acc"], "0.6")

        with open(json_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload["test_subject_id"], 1)
        self.assertEqual(len(payload["history"]), 2)
        self.assertEqual(payload["history"][1]["epoch"], 2)

    def test_append_training_log_appends_lines(self) -> None:
        log_path = self.fold_dir / "train.log"
        self.fold_dir.mkdir(parents=True, exist_ok=True)

        self.module.append_training_log(log_path, "Epoch 1/2")
        self.module.append_training_log(log_path, "Epoch 2/2")

        content = log_path.read_text(encoding="utf-8")
        self.assertEqual(content, "Epoch 1/2\nEpoch 2/2\n")


if __name__ == "__main__":
    unittest.main()
