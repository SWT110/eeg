"""Run 01,00,10,11 for one class-weight vector with the 0.7213 baseline settings."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import train_activity_loso as core
import train_activity_loso_batch as batch

EEG_ROOT = Path(__file__).resolve().parent.parent
BASELINE = dict(
    epochs=200, batch_size=72, lr=0.0002, seed=42,
    input_domain='time_fft', conv_type='standard', fft_global='none',
    transformer_branches=3, transformer_depths=[11, 10, 8],
    transformer_encoder_dropout=0.85, transformer_branch_fusion='loss_softmax',
    branch_loss_aux_weight=0.2, transformer_branch_qkv='cross_depth',
    transformer_branch_qkv_dropout=0.25,
)
ORDER = ('01', '00', '10', '11')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-weights', required=True, help='e_1,e_2,e_3 loss weights, e.g. 3,4,1')
    parser.add_argument('--dataset-root', type=Path, default=EEG_ROOT / 'local_artifacts/data_to_list/global_activity_dataset/window_15_stride_3')
    parser.add_argument('--output-root', type=Path, default=EEG_ROOT / 'local_artifacts/outputs/activity_weight_sweep_seed42')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--cpu-threads', type=int, default=12)
    parser.add_argument('--dry-run', action='store_true', help='Print configurations without loading data or training')
    args = parser.parse_args(argv)
    try:
        args.class_weights = core.parse_class_weights(args.class_weights)
        if args.class_weights is None or len(args.class_weights) != 3 or any(
            not math.isfinite(w) or w <= 0 for w in args.class_weights
        ):
            raise ValueError('Provide three strictly positive finite class weights')
    except ValueError as exc:
        parser.error(str(exc))
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    return args


def experiments(weights, output_root):
    # Preserve float precision so distinct weight vectors never share a folder.
    tag = '_'.join(str(int(w)) if float(w).is_integer() else repr(w) for w in weights)
    for code in ORDER:
        yield dict(BASELINE, class_weights=weights,
                   classification_mode='hierarchical' if code[0] == '1' else 'flat',
                   train_sampling='triple_minority_replacement' if code[1] == '1' else 'original',
                   output_dir=output_root / f'weights_{tag}_hierarchical_{code[0]}_triple_{code[1]}')


def ensure_plan(directory, plan):
    path = directory / 'experiment_plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(f'Configuration/data changed: use a new --output-root ({directory})')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f'Nonempty directory has no experiment plan: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        core.atomic_json_dump(plan, path)


def write_summary(directory, subject_ids):
    results = [json.loads((directory / f'fold_subject_{s}/metrics.json').read_text())
               for s in subject_ids if (directory / f'fold_subject_{s}/metrics.json').exists()]
    if not results:
        return
    summary = dict(
        completed_subject_ids=[r['test_subject_id'] for r in results],
        expected_subject_ids=subject_ids, complete=len(results) == len(subject_ids),
        selection='Highest test accuracy per fold (historical baseline protocol)',
        mean_best_test_acc=float(np.mean([r['best_test_acc'] for r in results])),
        mean_subject_macro_f1=float(np.mean([r['macro_f1'] for r in results])),
        mean_subject_recall=np.mean([[c['recall'] for c in r['per_class_metrics']]
                                    for r in results], axis=0).tolist(),
        class_order=['e_1', 'e_2', 'e_3'],
        class_weights=results[0]['class_weights'],
        classification_mode=results[0]['classification_mode'],
        train_sampling=results[0]['train_sampling'],
    )
    core.atomic_json_dump(summary, directory / 'summary.json')


def main(argv=None):
    args = parse_args(argv)
    jobs = list(experiments(args.class_weights, args.output_root.expanduser().resolve()))
    for code, job in zip(ORDER, jobs):
        print(f'[{code}] {json.dumps(job, default=str)}', flush=True)
    if args.dry_run:
        return
    device = core.validate_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    dataset = args.dataset_root.expanduser().resolve()
    subject_ids = batch.discover_subject_ids_from_global_dataset(dataset)
    # Detect changed dataset files before resuming without an extra full-array copy.
    data_files = {name: dict(size=(dataset / name).stat().st_size,
                             mtime_ns=(dataset / name).stat().st_mtime_ns)
                  for name in ('X.npy', 'y.npy', 'subject_ids.npy', 'metadata.json')}
    for job in jobs:
        plan = dict(version=1, training={k: v for k, v in job.items() if k != 'output_dir'},
                    dataset_root=str(dataset), data_files=data_files, subject_ids=subject_ids,
                    device=device, cpu_threads=args.cpu_threads)
        ensure_plan(job['output_dir'], plan)
    for job in jobs:
        directory = job['output_dir']
        print(f'\n[EXPERIMENT] {directory.name}', flush=True)
        for subject_id in subject_ids:
            fold = directory / f'fold_subject_{subject_id}'
            metrics_path = fold / 'metrics.json'
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                for key in ('class_weights', 'classification_mode', 'train_sampling', 'epochs', 'seed'):
                    if metrics.get(key) != job[key]:
                        raise ValueError(f'Mismatched {key} in {metrics_path}')
                if not (fold / 'best_model.pt').exists():
                    raise ValueError(f'Completed fold missing best_model.pt: {fold}')
                print(f'[SKIP] subject={subject_id}', flush=True)
            else:
                core.train_loso_fold(dataset_root=dataset, test_subject_id=subject_id,
                                     device=device, resume=True, **job)
            write_summary(directory, subject_ids)
        print(f'[DONE] {directory / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
