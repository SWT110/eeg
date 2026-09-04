from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from comparison_models import (  # noqa: E402
    VALID_COMPARISON_MODELS,
    build_comparison_model,
    count_trainable_parameters,
    normalize_comparison_model_name,
)


class TestComparisonModels(unittest.TestCase):
    def test_aliases_and_unknown_name(self) -> None:
        self.assertEqual(normalize_comparison_model_name("ShallowNet"), "shallowconvnet")
        self.assertEqual(normalize_comparison_model_name("ATC-Net"), "atcnet")
        with self.assertRaises(ValueError):
            normalize_comparison_model_name("invented-model")

    def test_all_published_configurations_forward_and_backward(self) -> None:
        batch = torch.randn(2, 1, 21, 512)
        labels = torch.tensor([0, 2])
        for architecture in VALID_COMPARISON_MODELS:
            with self.subTest(architecture=architecture):
                model = build_comparison_model(
                    architecture, n_channels=21, n_times=512, n_classes=3
                )
                features, logits = model(batch)
                self.assertEqual(logits.shape, (2, 3))
                self.assertEqual(features.shape[0], 2)
                self.assertTrue(torch.isfinite(logits).all())
                torch.nn.functional.cross_entropy(logits, labels).backward()
                self.assertGreater(count_trainable_parameters(model), 0)

    def test_activity_window_defaults_are_locked(self) -> None:
        eegnet = build_comparison_model("eegnet", 21, 1920, 3)
        tcformer = build_comparison_model("tcformer", 21, 1920, 3)
        self.assertEqual(eegnet.architecture_config["kernel_length"], 32)
        self.assertEqual(eegnet.architecture_config["dropout"], 0.25)
        self.assertEqual(
            tcformer.architecture_config["temporal_kernel_lengths"],
            [20, 32, 64],
        )
        self.assertEqual(tcformer.architecture_config["query_heads"], 4)
        self.assertEqual(tcformer.architecture_config["key_value_heads"], 2)
        self.assertEqual(tcformer.architecture_config["transformer_depth"], 5)

    def test_input_contract_rejects_ambiguous_shape(self) -> None:
        model = build_comparison_model("eegnet", 21, 512, 3)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 2, 21, 512))


if __name__ == "__main__":
    unittest.main()
