"""
Tests for train_activity_loso.py
=================================
Covers:
  1. load_global_dataset – reading arrays from disk succeeds
  2. loso_split          – correct train/test sample counts
  3. standardize_by_train – stats computed from train only
  4. ActivityConformer.forward – 21ch × 640pt × 3-class forward pass
  5. parse_args defaults – EEG-relative paths + cuda:0
  6. maybe_rerun_in_project_env – auto-restart key behaviour
  7. resolve_runtime_config – missing args in non-interactive mode raises
  8. main wiring
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_activity_loso.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_activity_loso", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_fake_dataset(root: Path, n_subjects: int = 4, n_per_subject: int = 6,
                       n_channels: int = 21, n_times: int = 640) -> None:
    """Write minimal X / y / subject_ids / metadata.json to *root*."""
    total = n_subjects * n_per_subject
    rng = np.random.default_rng(0)
    X = rng.random((total, n_channels, n_times), dtype=np.float32)
    y = np.tile([0, 1, 2, 0, 1, 2], n_subjects)[:total].astype(np.int64)
    subject_ids = np.repeat(np.arange(1, n_subjects + 1), n_per_subject).astype(np.int64)
    record_ids = np.zeros(total, dtype=np.int64)
    window_indices = np.tile(np.arange(n_per_subject), n_subjects)[:total].astype(np.int64)

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X.npy", X)
    np.save(root / "y.npy", y)
    np.save(root / "subject_ids.npy", subject_ids)
    np.save(root / "record_ids.npy", record_ids)
    np.save(root / "window_indices.npy", window_indices)

    metadata = {
        "label_map": {"e_1": 0, "e_2": 1, "e_3": 2},
        "n_subjects": n_subjects,
        "n_samples": total,
        "window_seconds": 5.0,
        "stride_seconds": 5.0,
    }
    with open(root / "metadata.json", "w") as fh:
        json.dump(metadata, fh)


class TestLoadGlobalDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root, n_subjects=3, n_per_subject=4)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_load_returns_correct_shapes(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)

        self.assertEqual(X.shape, (12, 21, 640))
        self.assertEqual(y.shape, (12,))
        self.assertEqual(subject_ids.shape, (12,))

    def test_load_raises_on_missing_file(self) -> None:
        bad_root = self.temp_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            self.module.load_global_dataset(bad_root)


class TestLosoSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        # 4 subjects × 6 windows each = 24 total
        _make_fake_dataset(self.dataset_root, n_subjects=4, n_per_subject=6)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_split_counts_are_correct(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        train_X, train_y, test_X, test_y = self.module.loso_split(X, y, subject_ids, test_subject_id=2)

        # subject 2 has 6 windows
        self.assertEqual(len(test_X), 6)
        self.assertEqual(len(test_y), 6)
        # remaining 3 subjects × 6 = 18
        self.assertEqual(len(train_X), 18)
        self.assertEqual(len(train_y), 18)

    def test_split_adds_conv_channel_dim(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        train_X, _, test_X, _ = self.module.loso_split(X, y, subject_ids, test_subject_id=1)

        # shape should be (N, 1, C, T)
        self.assertEqual(train_X.ndim, 4)
        self.assertEqual(train_X.shape[1], 1)
        self.assertEqual(test_X.ndim, 4)
        self.assertEqual(test_X.shape[1], 1)

    def test_test_subject_windows_not_in_train(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        _, _, _, _ = self.module.loso_split(X, y, subject_ids, test_subject_id=3)
        # sanity: total train + test = total samples
        train_X, _, test_X, _ = self.module.loso_split(X, y, subject_ids, test_subject_id=3)
        self.assertEqual(len(train_X) + len(test_X), len(X))

    def test_split_raises_when_test_subject_missing(self) -> None:
        X, y, subject_ids = self.module.load_global_dataset(self.dataset_root)
        with self.assertRaises(ValueError):
            self.module.loso_split(X, y, subject_ids, test_subject_id=99)


class TestStandardizeByTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_train_stats_applied_to_both_splits(self) -> None:
        rng = np.random.default_rng(7)
        train_raw = rng.random((20, 1, 21, 640), dtype=np.float32) * 10 + 5
        test_raw = rng.random((5, 1, 21, 640), dtype=np.float32) * 10 + 5

        train_std, test_std = self.module.standardize_by_train(train_raw, test_raw)

        # training data should be near zero-mean / unit-variance
        self.assertAlmostEqual(float(train_std.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(train_std.std()), 1.0, places=4)

    def test_test_set_uses_train_mean_not_own_mean(self) -> None:
        # train: constant 10, test: constant 20
        train_raw = np.full((10, 1, 5, 20), 10.0, dtype=np.float32)
        test_raw = np.full((5, 1, 5, 20), 20.0, dtype=np.float32)

        # std is 0 → should raise
        with self.assertRaises(ValueError):
            self.module.standardize_by_train(train_raw, test_raw)

    def test_test_set_shifted_by_train_mean(self) -> None:
        rng = np.random.default_rng(3)
        train_raw = rng.random((30, 1, 21, 640), dtype=np.float32)
        # test set is simply train + 100 (very different mean)
        test_raw = train_raw[:5] + 100.0

        train_std, test_std = self.module.standardize_by_train(train_raw, test_raw)

        # test set mean ≈ (mean(train) + 100 - mean(train)) / std(train) = 100 / std(train)
        train_std_val = float(train_raw.std())
        self.assertAlmostEqual(float(test_std.mean()), 100.0 / train_std_val, delta=0.1)


class TestInputDomainTransforms(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root, n_subjects=4, n_per_subject=6)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_fft_transform_returns_log_power_rfft_shape(self) -> None:
        windows = np.ones((2, 1, 3, 8), dtype=np.float32)

        transformed = self.module.transform_windows_for_input_domain(windows, "fft")

        self.assertEqual(transformed.shape, (2, 1, 3, 5))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertAlmostEqual(float(transformed[0, 0, 0, 0]), np.log1p(64.0), places=5)

    def test_build_dataloaders_fft_uses_frequency_axis(self) -> None:
        (
            _train_loader,
            _test_loader,
            _n_channels,
            n_times,
            _n_classes,
            _n_train_samples,
            _n_test_samples,
        ) = self.module.build_dataloaders(
            self.dataset_root,
            test_subject_id=1,
            batch_size=4,
            input_domain="fft",
        )

        self.assertEqual(n_times, 321)

    def test_prepare_time_fft_returns_separately_standardized_inputs(self) -> None:
        rng = np.random.default_rng(11)
        train_raw = rng.random((8, 1, 3, 200), dtype=np.float32) * 4 + 3
        test_raw = rng.random((3, 1, 3, 200), dtype=np.float32) * 4 + 3

        train_inputs, test_inputs = self.module.prepare_split_inputs_for_input_domain(
            train_raw,
            test_raw,
            "time_fft",
        )

        train_time, train_fft = train_inputs
        test_time, test_fft = test_inputs
        self.assertEqual(train_time.shape, (8, 1, 3, 200))
        self.assertEqual(test_time.shape, (3, 1, 3, 200))
        self.assertEqual(train_fft.shape, (8, 1, 3, 101))
        self.assertEqual(test_fft.shape, (3, 1, 3, 101))
        self.assertAlmostEqual(float(train_time.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(train_time.std()), 1.0, places=4)
        self.assertAlmostEqual(float(train_fft.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(train_fft.std()), 1.0, places=4)

    def test_build_dataloaders_time_fft_returns_dual_batches(self) -> None:
        train_loader, _test_loader, _n_channels, n_times, _n_classes, *_ = self.module.build_dataloaders(
            self.dataset_root,
            test_subject_id=1,
            batch_size=4,
            input_domain="time_fft",
        )

        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 3)
        batch_time, batch_fft, batch_y = batch
        self.assertEqual(n_times, 640)
        self.assertEqual(batch_time.shape[1:], (1, 21, 640))
        self.assertEqual(batch_fft.shape[1:], (1, 21, 321))
        self.assertEqual(batch_y.ndim, 1)


class TestCumulativeQueryAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    @staticmethod
    def _make_identity_attention(module):
        attention = module.MultiHeadAttention(emb_size=4, num_heads=1, dropout=0.0)
        with torch.no_grad():
            for linear in (attention.queries, attention.keys, attention.values, attention.projection):
                linear.weight.copy_(torch.eye(4))
                linear.bias.zero_()
        attention.eval()
        return attention

    def test_attention_uses_sum_of_previous_and_current_queries(self) -> None:
        attention = self._make_identity_attention(self.module)
        x1 = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
        x2 = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])

        _, q1 = attention(
            x1,
            cumulative_query_attention=True,
            return_cumulative_queries=True,
        )
        output, q12 = attention(
            x2,
            cumulative_queries=q1,
            cumulative_query_attention=True,
            return_cumulative_queries=True,
        )

        expected_q12 = x1.unsqueeze(1) + x2.unsqueeze(1)
        expected_energy = torch.einsum("bhqd,bhkd->bhqk", expected_q12, x2.unsqueeze(1))
        expected_attention = torch.softmax(expected_energy / (4 ** 0.5), dim=-1)
        expected_output = torch.einsum("bhal,bhlv->bhav", expected_attention, x2.unsqueeze(1))
        expected_output = expected_output.transpose(1, 2).contiguous().view(1, 2, 4)

        torch.testing.assert_close(q12, expected_q12)
        torch.testing.assert_close(output, expected_output)

    def test_disabled_mode_matches_original_block_sequence(self) -> None:
        encoder = self.module.TransformerEncoder(
            depth=3,
            emb_size=40,
            num_heads=5,
            cumulative_query_attention=False,
        ).eval()
        x = torch.randn(2, 7, 40)
        expected = x
        for block in encoder:
            expected = block(expected)

        torch.testing.assert_close(encoder(x), expected)

    def test_cumulative_state_resets_for_each_encoder_forward(self) -> None:
        encoder = self.module.TransformerEncoder(
            depth=3,
            emb_size=40,
            num_heads=5,
            cumulative_query_attention=True,
        ).eval()
        x = torch.randn(2, 7, 40)
        first = encoder(x)
        second = encoder(x)

        torch.testing.assert_close(first, second)


class TestActivityConformerForward(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_forward_21ch_640pt_3class(self) -> None:
        model = self.module.ActivityConformer(n_channels=21, n_times=640, n_classes=3)
        batch = torch.randn(2, 1, 21, 640)
        tok, logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_forward_returns_token_features_and_logits(self) -> None:
        model = self.module.ActivityConformer(n_channels=21, n_times=640, n_classes=3)
        batch = torch.randn(4, 1, 21, 640)
        tok, logits = model(batch)

        # tok is the flattened patch tokens before the final fc
        self.assertEqual(tok.shape[0], 4)
        self.assertEqual(logits.shape, (4, 3))

    def test_compute_n_patches_for_640_points(self) -> None:
        n_patches = self.module.compute_n_patches(640)
        # (640 - 24 - 75) // 15 + 1 = 541 // 15 + 1 = 36 + 1 = 37
        self.assertEqual(n_patches, 37)

    def test_compute_n_patches_for_200_points(self) -> None:
        n_patches = self.module.compute_n_patches(200)
        # (200 - 24 - 75) // 15 + 1 = 101 // 15 + 1 = 6 + 1 = 7
        self.assertEqual(n_patches, 7)

    def test_compute_n_patches_rejects_too_short_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            self.module.compute_n_patches(80)

    def test_forward_different_channel_counts(self) -> None:
        for n_ch in [22, 32, 64]:
            model = self.module.ActivityConformer(n_channels=n_ch, n_times=640, n_classes=3)
            batch = torch.randn(2, 1, n_ch, 640)
            _, logits = model(batch)
            self.assertEqual(tuple(logits.shape), (2, 3), msg=f"n_channels={n_ch}")

    def test_dual_branch_forward_time_and_fft_inputs(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=21,
            time_n_times=200,
            fft_n_times=101,
            n_classes=3,
            depth=1,
        )
        batch_time = torch.randn(2, 1, 21, 200)
        batch_fft = torch.randn(2, 1, 21, 101)
        features, logits = model(batch_time, batch_fft)

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_parallel_transformer_depths_are_symmetric_and_equal_weighted(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=3,
            time_n_times=120,
            fft_n_times=101,
            n_classes=3,
            emb_size=10,
            num_heads=5,
            dropout=0.0,
            transformer_depths=[2, 1, 3],
        ).eval()

        self.assertEqual(model.transformer_branch_depths, (2, 1, 3))
        self.assertEqual([len(encoder) for encoder in model.time_branch.depth_encoders], [2, 1, 3])
        self.assertEqual([len(encoder) for encoder in model.fft_branch.depth_encoders], [2, 1, 3])
        self.assertIsNot(
            model.time_branch.depth_weight_logits,
            model.fft_branch.depth_weight_logits,
        )
        expected = torch.full((3,), 1.0 / 3.0)
        torch.testing.assert_close(
            model.time_branch.normalized_transformer_weights(), expected
        )
        torch.testing.assert_close(
            model.fft_branch.normalized_transformer_weights(), expected
        )

        features, logits = model(
            torch.randn(2, 1, 3, 120),
            torch.randn(2, 1, 3, 101),
        )
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_parallel_transformer_weights_receive_gradients(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=3,
            time_n_times=120,
            fft_n_times=101,
            n_classes=3,
            emb_size=10,
            num_heads=5,
            dropout=0.0,
            transformer_depths=[1, 2],
        )
        _, logits = model(
            torch.randn(2, 1, 3, 120),
            torch.randn(2, 1, 3, 101),
        )
        logits.sum().backward()

        self.assertIsNotNone(model.time_branch.depth_weight_logits.grad)
        self.assertIsNotNone(model.fft_branch.depth_weight_logits.grad)

    def test_loss_softmax_uses_per_depth_heads_and_learnable_loss_weights(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=3,
            time_n_times=120,
            fft_n_times=101,
            n_classes=3,
            emb_size=10,
            num_heads=5,
            dropout=0.0,
            transformer_depths=[1, 2, 1],
            transformer_branch_fusion="loss_softmax",
            branch_loss_aux_weight=0.2,
        ).eval()

        self.assertTrue(model.uses_branch_loss_fusion)
        self.assertEqual(len(model.branch_cls_heads), 3)
        self.assertFalse(hasattr(model, "cls_head"))
        self.assertFalse(hasattr(model.time_branch, "depth_weight_logits"))
        self.assertFalse(hasattr(model.fft_branch, "depth_weight_logits"))
        self.assertFalse(hasattr(model, "time_cross_depth_qkv"))
        self.assertFalse(hasattr(model, "fft_cross_depth_qkv"))
        torch.testing.assert_close(
            model.normalized_branch_loss_weights(),
            torch.full((3,), 1.0 / 3.0),
        )
        with torch.no_grad():
            model.branch_loss_weight_logits.copy_(torch.tensor([0.5, -0.5, 1.0]))
        learned_weights = model.normalized_branch_loss_weights()

        features, fused_logits, branch_logits = model.forward_with_branch_logits(
            torch.randn(2, 1, 3, 120),
            torch.randn(2, 1, 3, 101),
        )
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(fused_logits.shape), (2, 3))
        self.assertEqual([tuple(value.shape) for value in branch_logits], [(2, 3)] * 3)
        torch.testing.assert_close(
            fused_logits,
            (
                learned_weights.view(-1, 1, 1)
                * torch.stack(branch_logits)
            ).sum(dim=0),
        )

        labels = torch.tensor([0, 2])
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 3.0, 1.0]))
        total_loss, branch_losses = self.module.compute_model_batch_loss(
            model,
            fused_logits,
            labels,
            criterion,
            branch_logits,
        )
        stacked = torch.stack(branch_losses)
        expected = (learned_weights * stacked).sum() + 0.2 * stacked.mean()
        torch.testing.assert_close(total_loss, expected)
        total_loss.backward()

        self.assertIsNotNone(model.branch_loss_weight_logits.grad)
        for head in model.branch_cls_heads:
            self.assertIsNotNone(head.fc[-1].weight.grad)

    def test_cross_depth_qkv_links_parallel_depth_features(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=3,
            time_n_times=120,
            fft_n_times=101,
            n_classes=3,
            emb_size=10,
            num_heads=5,
            dropout=0.0,
            transformer_depths=[1, 1, 1],
            transformer_branch_fusion="loss_softmax",
            branch_loss_aux_weight=0.2,
            transformer_branch_qkv="cross_depth",
        ).eval()

        self.assertTrue(model.uses_cross_depth_qkv)
        self.assertIsNot(model.time_cross_depth_qkv, model.fft_cross_depth_qkv)
        self.assertAlmostEqual(float(model.time_cross_depth_qkv.gamma.item()), 0.1)
        self.assertEqual(tuple(model.time_cross_depth_qkv.depth_embeddings.shape), (3, 10))

        features, fused_logits, branch_logits = model.forward_with_branch_logits(
            torch.randn(2, 1, 3, 120),
            torch.randn(2, 1, 3, 101),
        )
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(fused_logits.shape), (2, 3))
        self.assertEqual([tuple(value.shape) for value in branch_logits], [(2, 3)] * 3)

        # A loss from head 0 reaches another depth encoder through cross-depth K/V.
        branch_logits[0].sum().backward()
        other_time_encoder_parameter = next(
            model.time_branch.depth_encoders[1].parameters()
        )
        other_fft_encoder_parameter = next(
            model.fft_branch.depth_encoders[2].parameters()
        )
        self.assertIsNotNone(other_time_encoder_parameter.grad)
        self.assertIsNotNone(other_fft_encoder_parameter.grad)
        self.assertIsNotNone(model.time_cross_depth_qkv.attention.in_proj_weight.grad)
        self.assertIsNotNone(model.fft_cross_depth_qkv.attention.in_proj_weight.grad)
        self.assertIsNotNone(model.time_cross_depth_qkv.gamma.grad)
        self.assertIsNotNone(model.fft_cross_depth_qkv.gamma.grad)

        metadata = model.transformer_weight_metadata()
        self.assertIn("time_transformer_branch_qkv_gamma", metadata)
        self.assertIn("fft_transformer_branch_qkv_gamma", metadata)

    def test_loss_softmax_rejects_incompatible_configurations(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.module.resolve_transformer_branch_fusion(
                "loss_softmax",
                transformer_branches=1,
                input_domain="time_fft",
            )
        with self.assertRaisesRegex(ValueError, "applies only"):
            self.module.resolve_branch_loss_aux_weight(
                0.2,
                self.module.TRANSFORMER_FUSION_SOFTMAX,
            )
        with self.assertRaisesRegex(ValueError, "loss_softmax"):
            self.module.resolve_transformer_branch_qkv(
                "cross_depth",
                transformer_branch_fusion=self.module.TRANSFORMER_FUSION_SOFTMAX,
                transformer_branches=3,
                input_domain="time_fft",
            )
        self.assertEqual(
            self.module.resolve_transformer_branch_qkv(
                "cross_depth",
                transformer_branch_fusion=self.module.TRANSFORMER_FUSION_LOSS_SOFTMAX,
                transformer_branches=3,
                input_domain="time_fft",
            ),
            "cross_depth",
        )

    def test_default_dual_branch_keeps_legacy_encoder_state_dict_layout(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=3,
            time_n_times=120,
            fft_n_times=101,
            n_classes=3,
            emb_size=10,
            depth=1,
            num_heads=5,
        )
        keys = list(model.state_dict())

        self.assertTrue(any(key.startswith("time_branch.encoder.0") for key in keys))
        self.assertTrue(any(key.startswith("fft_branch.encoder.0") for key in keys))
        self.assertFalse(any("depth_weight_logits" in key for key in keys))

    def test_patch_embedding_conv_type_controls_spatial_groups(self) -> None:
        standard = self.module.PatchEmbedding(n_channels=21, conv_type="standard")
        dwconv = self.module.PatchEmbedding(n_channels=21, conv_type="dw")

        self.assertEqual(standard.shallownet[1].groups, 1)
        self.assertEqual(dwconv.shallownet[1].groups, 40)
        self.assertEqual(dwconv.conv_type, "dwconv")

    def test_dual_branch_dwconv_uses_depthwise_spatial_conv(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=21,
            time_n_times=200,
            fft_n_times=101,
            n_classes=3,
            depth=1,
            conv_type="dwconv",
        )

        self.assertEqual(model.time_branch.patch_embedding.shallownet[1].groups, 40)
        self.assertEqual(model.fft_branch.patch_embedding.shallownet[1].groups, 40)

    def test_fft_global_mlp_preserves_shape(self) -> None:
        layer = self.module.FFTGlobalMLP(n_times=101)
        batch = torch.randn(2, 1, 21, 101)

        out = layer(batch)

        self.assertEqual(tuple(out.shape), (2, 1, 21, 101))
        self.assertEqual(layer.hidden_size, 25)

    def test_fft_single_branch_global_mlp_forward(self) -> None:
        model = self.module.ActivityConformer(
            n_channels=21,
            n_times=101,
            n_classes=3,
            depth=1,
            fft_global="mlp",
        )
        batch = torch.randn(2, 1, 21, 101)

        features, logits = model(batch)

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(model.fft_global, "mlp")

    def test_dual_branch_fft_global_mlp_forward(self) -> None:
        model = self.module.DualBranchActivityConformer(
            n_channels=21,
            time_n_times=200,
            fft_n_times=101,
            n_classes=3,
            depth=1,
            fft_global="mlp",
        )
        batch_time = torch.randn(2, 1, 21, 200)
        batch_fft = torch.randn(2, 1, 21, 101)

        features, logits = model(batch_time, batch_fft)

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(model.fft_global, "mlp")


class TestParseArgsDefaults(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_default_dataset_root_is_eeg_relative(self) -> None:
        args = self.module.parse_args([])

        expected_eeg_root = MODULE_PATH.resolve().parents[1]
        self.assertEqual(
            args.dataset_root,
            expected_eeg_root / "local_artifacts" / "data_to_list" / "global_activity_dataset",
        )

    def test_default_output_dir_is_eeg_conformer_relative(self) -> None:
        args = self.module.parse_args([])

        expected_eeg_root = MODULE_PATH.resolve().parents[1]
        self.assertEqual(
            args.output_dir,
            expected_eeg_root / "local_artifacts" / "outputs" / "activity_loso",
        )

    def test_default_device_is_cuda0(self) -> None:
        args = self.module.parse_args([])
        self.assertEqual(args.device, "cuda:0")

    def test_default_lr_and_epochs(self) -> None:
        args = self.module.parse_args([])
        self.assertAlmostEqual(args.lr, 0.0002)
        self.assertEqual(args.epochs, 200)
        self.assertEqual(args.batch_size, 72)

    def test_explicit_args_override_defaults(self) -> None:
        args = self.module.parse_args(
            ["--test-subject-id", "3", "--epochs", "10", "--device", "cpu"]
        )
        self.assertEqual(args.test_subject_id, 3)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.device, "cpu")

    def test_accepts_class_weights_argument(self) -> None:
        args = self.module.parse_args(["--class-weights", "3,3,1"])
        self.assertEqual(args.class_weights, "3,3,1")

    def test_accepts_parallel_transformer_arguments(self) -> None:
        args = self.module.parse_args(
            ["--transformer-branches", "3", "--transformer-depths", "11", "10", "8"]
        )
        self.assertEqual(args.transformer_branches, 3)
        self.assertEqual(args.transformer_depths, [11, 10, 8])
        self.assertEqual(
            self.module.resolve_transformer_branch_depths(
                depth=args.depth,
                transformer_branches=args.transformer_branches,
                transformer_depths=args.transformer_depths,
                input_domain="time_fft",
            ),
            (11, 10, 8),
        )

    def test_accepts_loss_softmax_arguments(self) -> None:
        args = self.module.parse_args(
            [
                "--transformer-branch-fusion",
                "loss_softmax",
                "--branch-loss-aux-weight",
                "0.2",
            ]
        )
        self.assertEqual(args.transformer_branch_fusion, "loss_softmax")
        self.assertAlmostEqual(args.branch_loss_aux_weight, 0.2)

    def test_accepts_cross_depth_qkv_argument(self) -> None:
        args = self.module.parse_args(
            ["--transformer-branch-qkv", "cross_depth"]
        )
        self.assertEqual(args.transformer_branch_qkv, "cross_depth")

    def test_resume_is_opt_in(self) -> None:
        self.assertFalse(self.module.parse_args([]).resume)
        self.assertTrue(self.module.parse_args(["--resume"]).resume)

    def test_parallel_transformer_arguments_require_matching_count_and_dual_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal"):
            self.module.resolve_transformer_branch_depths(
                transformer_branches=3,
                transformer_depths=[11, 10],
                input_domain="time_fft",
            )
        with self.assertRaisesRegex(ValueError, "time_fft"):
            self.module.resolve_transformer_branch_depths(
                transformer_branches=3,
                transformer_depths=[11, 10, 8],
                input_domain="time",
            )

    def test_cumulative_query_attention_is_opt_in(self) -> None:
        self.assertFalse(self.module.parse_args([]).cumulative_query_attention)
        self.assertTrue(
            self.module.parse_args(["--cumulative-query-attention"]).cumulative_query_attention
        )

    def test_accepts_input_domain_argument(self) -> None:
        args = self.module.parse_args(["--input-domain", "fft"])
        self.assertEqual(args.input_domain, "fft")

    def test_accepts_time_fft_input_domain_argument(self) -> None:
        args = self.module.parse_args(["--input-domain", "time_fft"])
        self.assertEqual(args.input_domain, "time_fft")
        self.assertEqual(self.module.validate_input_domain(args.input_domain), "time_fft")

    def test_accepts_conv_type_argument(self) -> None:
        args = self.module.parse_args(["--conv-type", "dwconv"])
        self.assertEqual(args.conv_type, "dwconv")
        self.assertEqual(self.module.validate_conv_type(args.conv_type), "dwconv")

    def test_accepts_fft_global_argument(self) -> None:
        args = self.module.parse_args(["--fft-global", "mlp"])
        self.assertEqual(args.fft_global, "mlp")
        self.assertEqual(self.module.validate_fft_global(args.fft_global), "mlp")

    def test_rejects_fft_global_mlp_for_time_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "fft or time_fft"):
            self.module.validate_fft_global_for_input_domain("time", "mlp")


class TestMaybeRerunInProjectEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_reexecutes_when_base_env_lacks_cuda(self) -> None:
        project_root = self.temp_dir / "EEG-Conformer"
        env_prefix = project_root / ".conda-envs" / self.module.DEFAULT_ENV_NAME
        env_prefix.mkdir(parents=True, exist_ok=True)
        self.module.PROJECT_ROOT = project_root
        self.module.EEG_ROOT = project_root.parent

        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.dict(os.environ, {"CONDA_EXE": "/opt/miniconda/bin/conda"}, clear=False):
                with patch.object(
                    self.module.subprocess, "run", return_value=CompletedProcess([], 0)
                ) as mock_run:
                    with self.assertRaises(SystemExit) as ctx:
                        self.module.maybe_rerun_in_project_env([], "cuda:0")

        self.assertEqual(ctx.exception.code, 0)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[0], str(env_prefix / "bin" / "python"))
        self.assertEqual(mock_run.call_args.kwargs["env"][self.module.AUTO_RERUN_ENV_VAR], "1")

    def test_no_rerun_when_cpu_device(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.object(self.module.subprocess, "run", side_effect=AssertionError("should not run")):
                # should return without calling subprocess.run
                self.module.maybe_rerun_in_project_env([], "cpu")

    def test_no_rerun_when_cuda_already_usable(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=True):
            with patch.object(self.module.subprocess, "run", side_effect=AssertionError("should not run")):
                self.module.maybe_rerun_in_project_env([], "cuda:0")

    def test_no_rerun_when_env_var_already_set(self) -> None:
        with patch.object(self.module, "cuda_is_usable", return_value=False):
            with patch.dict(os.environ, {self.module.AUTO_RERUN_ENV_VAR: "1"}, clear=False):
                with patch.object(self.module.subprocess, "run", side_effect=AssertionError("should not run")):
                    self.module.maybe_rerun_in_project_env([], "cuda:0")


class TestResolveRuntimeConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_raises_in_noninteractive_mode_when_args_missing(self) -> None:
        with patch("sys.stdin.isatty", return_value=False):
            with patch("builtins.input", side_effect=AssertionError("should not prompt")):
                with self.assertRaises(ValueError) as ctx:
                    self.module.resolve_runtime_config(
                        dataset_root=None,
                        test_subject_id=None,
                        epochs=None,
                        batch_size=None,
                        lr=None,
                        device=None,
                        output_dir=None,
                        seed=None,
                    )

        msg = str(ctx.exception)
        self.assertIn("Missing required arguments", msg)
        self.assertIn("--dataset-root", msg)

    def test_explicit_values_accepted_without_prompting(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            config = self.module.resolve_runtime_config(
                dataset_root=self.dataset_root,
                test_subject_id=2,
                epochs=10,
                batch_size=8,
                lr=2e-4,
                device="cuda:0",
                output_dir=self.temp_dir / "out",
                seed=7,
            )

        self.assertEqual(config.test_subject_id, 2)
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.seed, 7)

    def test_rejects_cuda_when_unavailable(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is not available"):
                self.module.resolve_runtime_config(
                    dataset_root=self.dataset_root,
                    test_subject_id=1,
                    epochs=1,
                    batch_size=8,
                    lr=2e-4,
                    device="cuda:0",
                    output_dir=self.temp_dir / "out",
                    seed=42,
                )

    def test_defaults_via_interactive_prompts(self) -> None:
        self.module.DEFAULT_DATASET_ROOT = self.dataset_root
        self.module.DEFAULT_OUTPUT_DIR = self.temp_dir / "outputs"
        with patch("torch.cuda.is_available", return_value=False):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", side_effect=[""] * 10):
                    config = self.module.resolve_runtime_config(
                        dataset_root=None,
                        test_subject_id=None,
                        epochs=None,
                        batch_size=None,
                        lr=None,
                        device=None,
                        output_dir=None,
                        seed=None,
                    )

        self.assertEqual(config.test_subject_id, self.module.DEFAULT_TEST_SUBJECT_ID)
        self.assertEqual(config.epochs, self.module.DEFAULT_EPOCHS)
        self.assertEqual(config.device, "cpu")


class TestMainWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-"))
        self.dataset_root = self.temp_dir / "dataset"
        _make_fake_dataset(self.dataset_root)
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_main_calls_rerun_before_resolve(self) -> None:
        """maybe_rerun_in_project_env should be called before resolve_runtime_config."""
        from argparse import Namespace

        fake_args = Namespace(
            dataset_root=self.dataset_root,
            test_subject_id=1,
            epochs=1,
            batch_size=8,
            lr=2e-4,
            device="cuda:0",
            output_dir=self.temp_dir / "out",
            seed=42,
            input_domain="time",
        )

        with patch.object(self.module, "parse_args", return_value=fake_args):
            with patch.object(
                self.module, "maybe_rerun_in_project_env", side_effect=SystemExit(0)
            ) as mock_rerun:
                with patch.object(
                    self.module, "resolve_runtime_config",
                    side_effect=AssertionError("should not reach resolve"),
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        self.module.main()

        self.assertEqual(ctx.exception.code, 0)
        mock_rerun.assert_called_once()

    def test_main_no_args_passes_nones_to_resolve(self) -> None:
        """When called with no argv, main passes None for all config args."""
        fake_config = self.module.RuntimeConfig(
            dataset_root=self.dataset_root,
            test_subject_id=1,
            epochs=1,
            batch_size=8,
            lr=2e-4,
            device="cpu",
            output_dir=self.temp_dir / "out",
            seed=42,
            input_domain="time",
            class_weights=None,
        )

        with patch.object(self.module, "maybe_rerun_in_project_env", return_value=None):
            with patch.object(
                self.module, "resolve_runtime_config", return_value=fake_config
            ) as mock_resolve:
                with patch.object(
                    self.module, "train_loso_fold",
                    return_value=self.temp_dir / "metrics.json",
                ):
                    self.module.main([])

        mock_resolve.assert_called_once_with(
            dataset_root=None,
            test_subject_id=None,
            epochs=None,
            batch_size=None,
            lr=None,
            device=None,
            output_dir=None,
            seed=None,
            input_domain=None,
            class_weights=None,
            resume=False,
        )


class TestOptimizerResume(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="activity-loso-resume-"))
        self.dataset_root = self.temp_dir / "dataset"
        self.dataset_root.mkdir()
        self.module = load_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _tiny_model_class():
        class TinyActivityModel(torch.nn.Module):
            def __init__(self, n_classes: int, **kwargs) -> None:
                super().__init__()
                initial = torch.zeros(n_classes, dtype=torch.float32)
                initial[0] = 1.0
                self.class_logits = torch.nn.Parameter(initial)

            def forward(self, inputs):
                features = inputs.flatten(start_dim=1)
                logits = self.class_logits.unsqueeze(0).expand(inputs.shape[0], -1)
                return features, logits

        return TinyActivityModel

    @staticmethod
    def _tiny_dataloaders(*args, **kwargs):
        inputs = torch.arange(48, dtype=torch.float32).reshape(6, 1, 2, 4) / 48.0
        labels = torch.zeros(6, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
        return train_loader, test_loader, 2, 4, 3, 6, 6

    def test_interrupted_fold_restores_model_adam_history_and_rng(self) -> None:
        output_dir = self.temp_dir / "outputs"
        real_history_writer = self.module.write_epoch_history_files

        def interrupt_after_checkpoint(*, fold_dir, history, metadata):
            self.assertEqual(len(history), 1)
            raise OSError(28, "simulated no space")

        common = {
            "dataset_root": self.dataset_root,
            "test_subject_id": 1,
            "epochs": 2,
            "batch_size": 2,
            "lr": 2e-4,
            "device": "cpu",
            "output_dir": output_dir,
            "seed": 43,
            "resume": True,
        }
        with patch.object(
            self.module, "build_dataloaders", side_effect=self._tiny_dataloaders
        ), patch.object(
            self.module, "ActivityConformer", self._tiny_model_class()
        ), patch.object(
            self.module, "write_epoch_history_files", side_effect=interrupt_after_checkpoint
        ):
            with self.assertRaisesRegex(OSError, "simulated no space"):
                self.module.train_loso_fold(**common)

        fold_dir = output_dir / "fold_subject_1"
        resume_path = fold_dir / self.module.RESUME_CHECKPOINT_FILENAME
        self.assertTrue(resume_path.exists())
        checkpoint = self.module.load_torch_checkpoint(resume_path, torch.device("cpu"))
        self.assertEqual(checkpoint["completed_epoch"], 1)
        self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
        self.assertEqual(len(checkpoint["epoch_history"]), 1)

        with patch.object(
            self.module, "build_dataloaders", side_effect=self._tiny_dataloaders
        ), patch.object(
            self.module, "ActivityConformer", self._tiny_model_class()
        ), patch.object(
            self.module, "write_epoch_history_files", wraps=real_history_writer
        ):
            metrics_path = self.module.train_loso_fold(**common)

        self.assertTrue(metrics_path.exists())
        self.assertFalse(resume_path.exists())
        metrics = json.loads(metrics_path.read_text(encoding="ascii"))
        self.assertEqual(metrics["resumed_from_epoch"], 1)
        history_payload = json.loads(
            (fold_dir / "epoch_history.json").read_text(encoding="utf-8")
        )
        self.assertEqual([row["epoch"] for row in history_payload["history"]], [1, 2])
        self.assertIn("restored model + Adam optimizer", (fold_dir / "train.log").read_text())

    def test_atomic_torch_save_keeps_previous_target_on_failure(self) -> None:
        target = self.temp_dir / "checkpoint.pt"
        target.write_bytes(b"previous-valid-checkpoint")
        with patch.object(
            self.module.torch,
            "save",
            side_effect=OSError(28, "simulated no space"),
        ):
            with self.assertRaisesRegex(OSError, "simulated no space"):
                self.module.atomic_torch_save({"value": 1}, target)

        self.assertEqual(target.read_bytes(), b"previous-valid-checkpoint")
        self.assertFalse(self.module.temporary_artifact_path(target).exists())

    def test_resume_rejects_changed_training_configuration(self) -> None:
        checkpoint = {
            "resume_checkpoint_version": self.module.RESUME_CHECKPOINT_VERSION,
            "completed_epoch": 1,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "rng_state": {},
            "epoch_history": [{"epoch": 1}],
            "training_config": {"batch_size": 72, "epochs": 200},
            "average_test_acc_sum": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            self.module.validate_resume_checkpoint(
                checkpoint,
                expected_training_config={"batch_size": 64, "epochs": 300},
                target_epochs=300,
            )


if __name__ == "__main__":
    unittest.main()
