"""Apply a recall checkpoint, including its frozen normalization and decision.

Input must be [N,C,T] windows with the same upstream preprocessing as X.npy.
This does not accept raw EDF, fit a normalizer, or tune on prediction labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import train_activity_recall as recall


def predict_windows(checkpoint, windows, device, batch_size=72):
    if checkpoint.get('protocol') != 'subject_validated_recall_v1':
        raise ValueError('Expected a refit_checkpoint.pt from train_activity_recall.py')
    if windows.ndim != 3 or list(windows.shape[1:]) != checkpoint['shape'] or len(windows) == 0:
        raise ValueError(f"Expected nonempty [N,{checkpoint['shape'][0]},{checkpoint['shape'][1]}] windows")
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    model = recall.build_model(tuple(checkpoint['shape']), {'model': checkpoint['model_config']}).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    log_probs = []
    for start in range(0, len(windows), batch_size):
        inputs = recall.validation.transform_inputs(windows[start:start + batch_size], True, checkpoint['normalizers'])
        loader = recall.validation.make_loader(inputs, np.zeros(len(inputs[0]), dtype=np.int64), batch_size, False, 0)
        _, scores = recall.predict_log_probs(model, loader, device)
        log_probs.append(scores)
    scores = np.concatenate(log_probs)
    return dict(log_probs=scores, probabilities=np.exp(scores),
                y_pred=recall.apply_decision(scores, checkpoint['decision']),
                y_pred_default=recall.apply_decision(scores, checkpoint['neutral_decision']),
                sample_indices=np.arange(len(windows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input-npy', type=Path, required=True)
    parser.add_argument('--output-npz', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=72)
    parser.add_argument('--cpu-threads', type=int, default=12)
    args = parser.parse_args(argv)
    if args.output_npz.exists():
        raise FileExistsError('Refusing to overwrite existing prediction output')
    if args.cpu_threads < 1:
        raise ValueError('cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(recall.core.validate_device(args.device))
    checkpoint = recall.core.load_torch_checkpoint(args.checkpoint, torch.device('cpu'))
    windows = np.load(args.input_npy, mmap_mode='r', allow_pickle=False)
    predictions = predict_windows(checkpoint, windows, device, args.batch_size)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    recall.core.atomic_save_npz(args.output_npz, **predictions)
    print(f'Saved {len(windows)} predictions to {args.output_npz}; labels 0=e_1, 1=e_2, 2=e_3')


if __name__ == '__main__':
    main()
