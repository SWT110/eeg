"""Scientific input/gradient, filter parity, resume and queue exclusion checks."""
from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_models as bm
import run_architecture_benchmark as runner
import train_activity_loso as training


class ArchitectureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_task_dimensions_and_finite_learning_signal(self):
        for name in bm.CLASSES:
            with self.subTest(model=name):
                torch.manual_seed(11)
                try:
                    model = bm.build_benchmark_model(name, 21, 1920, 3)
                except ModuleNotFoundError as exc:
                    if name == "fbcnet" and exc.name == "scipy":
                        continue  # Filter parity is separately required in server environment.
                    raise
                before = [p.detach().clone() for p in model.parameters()]
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
                features, logits = model(torch.randn(2, 1, 21, 1920))
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertEqual(features.shape[0], 2)
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
                model.eval()
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(model(torch.zeros(1, 1, 21, 1920))[1]).all())

    def test_filter_bank_matches_author_causal_scipy_equations(self):
        try:
            from scipy import signal
        except ImportError:
            self.skipTest("SciPy absent locally; required server validation runs this check")
        bank = bm.FixedFilterBank(1920)
        source = np.random.default_rng(6).standard_normal((2, 3, 1920)).astype(np.float32)
        result = bank(torch.from_numpy(source[:, None])).detach().numpy()
        expected = np.stack([signal.lfilter(c["b"], c["a"], source, axis=-1) for c in bank.coefficients], axis=1)
        np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-4)
        # A last-sample impulse cannot wrap around to the beginning.
        impulse = torch.zeros(1, 1, 1, 1920)
        impulse[..., -1] = 1
        self.assertLess(float(bank(impulse)[..., :100].abs().max()), 1e-6)

    def test_temporal_blocks_do_not_read_future_samples_in_eval(self):
        for depthwise in (False, True):
            model = bm.CausalBlock(4, 4, 4, 4, 0.3, depthwise=depthwise).eval()
            x = torch.randn(2, 4, 100)
            altered = x.clone()
            altered[..., 60:] += 10
            with torch.no_grad():
                torch.testing.assert_close(model(x)[..., :60], model(altered)[..., :60], rtol=0, atol=0)

    def test_tsception_hemispheres_exclude_midline_but_global_sees_it(self):
        model = bm.TSception(21, 1920, 3).eval()
        self.assertEqual(model.hemisphere_order.tolist(), list(range(0, 18, 2)) + list(range(1, 18, 2)))
        x = torch.zeros(1, 1, 21, 1920, requires_grad=True)
        model.classifier(model.forward_features(x)).sum().backward()
        self.assertGreater(float(x.grad[:, :, 18:].abs().sum()), 0)

    def test_eegtcnet_sample_configuration_parameter_count(self):
        model = bm.build_benchmark_model("eegtcnet", 21, 1920, 3)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4067)

    def test_only_one_worker_can_claim_a_fold(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                claims = list(pool.map(lambda index: runner.claim(directory, {"worker": index}), range(8)))
            self.assertEqual(sum(item is not None for item in claims), 1)

    def test_interrupted_new_architecture_resumes_same_training_trajectory(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset = root / "data"
            dataset.mkdir()
            random = np.random.default_rng(8)
            np.save(dataset / "X.npy", random.standard_normal((12, 21, 1920)).astype(np.float32))
            np.save(dataset / "y.npy", np.tile([0, 1, 2, 0], 3))
            np.save(dataset / "subject_ids.npy", np.repeat([1, 2, 3], 4))
            common = dict(dataset_root=dataset, test_subject_id=1, epochs=2, batch_size=2,
                          lr=0.0002, device="cpu", seed=42, resume=True,
                          fail_on_nonfinite_loss=True,
                          architecture="eegtcnet", external_model_factory=bm.build_benchmark_model)
            complete = training.train_loso_fold(output_dir=root / "complete", **common)
            original = training.append_training_log
            def interrupt(path, message):
                original(path, message)
                if message.startswith("Epoch 1/"):
                    raise InterruptedError("simulated process interruption after durable epoch checkpoint")
            with patch.object(training, "append_training_log", interrupt):
                with self.assertRaises(InterruptedError):
                    training.train_loso_fold(output_dir=root / "resumed", **common)
            resumed = training.train_loso_fold(output_dir=root / "resumed", **common)
            a, b = json.loads(complete.read_text()), json.loads(resumed.read_text())
            self.assertEqual(a["best_test_acc"], b["best_test_acc"])
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            left = training.load_torch_checkpoint(complete.parent / "best_model.pt", torch.device("cpu"))
            right = training.load_torch_checkpoint(resumed.parent / "best_model.pt", torch.device("cpu"))
            for key in left["state_dict"]:
                torch.testing.assert_close(left["state_dict"][key], right["state_dict"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
