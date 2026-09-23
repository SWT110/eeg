"""Three-arm EEG resampling comparison using the historical best-test-epoch rule.

Each LOSO fold evaluates the held-out subject after every epoch and selects the
highest test accuracy. This is comparable to historical summaries but optimistic
as a generalization estimate. Original training entrypoints remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

import train_activity_loso as core
import train_activity_loso_batch as batch
import train_activity_subject_validation as validation

VERSION = 3
TRAIN_SAMPLING_CHOICES = ('original', 'triple_minority_replacement')


def parse_args(argv=None):
    parser = batch.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=None)
    parser.add_argument('--cpu-threads', type=int, default=12)
    parser.add_argument('--train-sampling', choices=TRAIN_SAMPLING_CHOICES, default='original',
                        help='Training windows per epoch: original once, or sample e1/e2 3x with replacement and e3 once')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def predict_labels(log_probs, classification_mode):
    scores = np.asarray(log_probs)
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError('Expected finite log probabilities [N,3]')
    if classification_mode == 'flat':
        return scores.argmax(1)
    group_log = np.logaddexp(scores[:, 0], scores[:, 1])
    group = group_log >= scores[:, 2]
    first = scores[:, 0] >= scores[:, 1]
    return np.where(group, np.where(first, 0, 1), 2).astype(np.int64)


@torch.no_grad()
def predict_log_probs(model, loader, device):
    model.eval()
    labels, scores = [], []
    for item in loader:
        logits, y, _ = core.forward_model_batch_with_branches(model, item, device)
        # Flat logits and hierarchical normalized log probabilities share this format.
        log_probs = torch.log_softmax(logits, dim=1)
        if not torch.isfinite(log_probs).all():
            raise RuntimeError('Non-finite prediction scores')
        labels.append(y.cpu().numpy())
        scores.append(log_probs.cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def training_components(model, config, device):
    weights = config['class_weights']
    weight = None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device)
    loss_type = nn.NLLLoss if config['model']['classification_mode'] == 'hierarchical' else nn.CrossEntropyLoss
    criterion = loss_type(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(.5, .999))
    return criterion, optimizer


def build_model(shape, config):
    return core.DualBranchActivityConformer(n_channels=shape[0], time_n_times=shape[1],
                                           fft_n_times=shape[1] // 2 + 1, n_classes=3,
                                           **config['model'])


def training_draw_counts(labels, sampling):
    counts = np.bincount(np.asarray(labels), minlength=3)
    if len(counts) != 3 or np.any(counts == 0):
        raise ValueError('Every training stage needs all three classes')
    if sampling == 'original':
        draws = counts
    elif sampling == 'triple_minority_replacement':
        draws = counts * np.array([3, 3, 1])
    else:
        raise ValueError(f'Unknown train sampling: {sampling}')
    return counts.astype(int).tolist(), draws.astype(int).tolist()


class TripledMinoritySampler(Sampler[int]):
    """Draw each minority class 3x with replacement and every e3 window once."""

    def __init__(self, labels, generator):
        labels = np.asarray(labels)
        self.positions = [torch.as_tensor(np.flatnonzero(labels == label), dtype=torch.int64)
                          for label in range(3)]
        if any(len(position) == 0 for position in self.positions):
            raise ValueError('Every training stage needs all three classes')
        self.generator = generator

    def __len__(self):
        return 3 * len(self.positions[0]) + 3 * len(self.positions[1]) + len(self.positions[2])

    def __iter__(self):
        minority = [positions[torch.randint(len(positions), (3 * len(positions),),
                                            generator=self.generator)]
                    for positions in self.positions[:2]]
        indices = torch.cat([*minority, self.positions[2]])
        return iter(indices[torch.randperm(len(indices), generator=self.generator)].tolist())


def make_training_loader(inputs, labels, config, seed):
    sampling = config['train_sampling']
    if sampling == 'original':
        return validation.make_loader(inputs, labels, config['batch_size'], True, seed)
    if sampling != 'triple_minority_replacement':
        raise ValueError(f'Unknown train sampling: {sampling}')
    generator = torch.Generator().manual_seed(seed)
    sampler = TripledMinoritySampler(labels, generator)
    return DataLoader(core.tensor_dataset_from_inputs(inputs, labels), batch_size=config['batch_size'],
                      sampler=sampler, num_workers=0, generator=generator)


def resolve_config(args):
    if args.output_dir is None:
        raise ValueError('--output-dir is required; use a NEW resampling directory')
    if args.input_domain != 'time_fft':
        raise ValueError('This entrypoint supports --input-domain time_fft (flat or hierarchical)')
    if args.epochs < 1 or args.batch_size < 2 or args.cpu_threads < 1:
        raise ValueError('epochs >= 1, batch-size >= 2, cpu-threads >= 1 required')
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('lr must be finite and positive')
    if not 0 <= args.seed < 2**32:
        raise ValueError('Seed out of supported range')
    weights = core.parse_class_weights(args.class_weights)
    if weights is not None and (len(weights) != 3 or any(not np.isfinite(w) or w <= 0 for w in weights)):
        raise ValueError('Recall training requires three strictly positive finite class weights')
    if args.train_sampling == 'triple_minority_replacement' and weights not in (None, [1., 1., 1.]):
        raise ValueError('Triple minority sampling requires unweighted loss (omit --class-weights or use 1,1,1)')
    depths = core.resolve_transformer_branch_depths(args.depth, args.transformer_branches,
                                                    args.transformer_depths, args.input_domain)
    fusion = core.resolve_transformer_branch_fusion(args.transformer_branch_fusion, len(depths), args.input_domain)
    qkv = core.resolve_transformer_branch_qkv(args.transformer_branch_qkv, fusion, len(depths), args.input_domain)
    model = dict(depth=args.depth, conv_type=core.validate_conv_type(args.conv_type),
                 fft_global=core.validate_fft_global_for_input_domain(args.input_domain, args.fft_global),
                 input_qkv=core.validate_input_qkv(args.input_qkv), input_qkv_dim=args.input_qkv_dim,
                 input_qkv_heads=args.input_qkv_heads, input_qkv_dropout=args.input_qkv_dropout,
                 input_qkv_res_scale=args.input_qkv_res_scale,
                 cumulative_query_attention=args.cumulative_query_attention,
                 transformer_depths=list(depths), transformer_branch_fusion=fusion,
                 branch_loss_aux_weight=core.resolve_branch_loss_aux_weight(args.branch_loss_aux_weight, fusion),
                 transformer_branch_qkv=qkv,
                 transformer_encoder_dropout=core.validate_dropout_probability(args.transformer_encoder_dropout, 'encoder dropout'),
                 transformer_branch_qkv_dropout=core.validate_dropout_probability(args.transformer_branch_qkv_dropout, 'cross dropout'),
                 classification_mode=args.classification_mode)
    if args.input_qkv_dim < 1 or args.input_qkv_heads < 1 or args.input_qkv_dim % args.input_qkv_heads:
        raise ValueError('input-qkv-dim must be positive and divisible by input-qkv-heads')
    core.validate_dropout_probability(args.input_qkv_dropout, 'input dropout')
    if not np.isfinite(args.input_qkv_res_scale):
        raise ValueError('input-qkv-res-scale must be finite')
    return dict(version=VERSION, model=model, class_weights=weights, train_sampling=args.train_sampling,
                lr=args.lr, epochs=args.epochs,
                batch_size=args.batch_size, seed=args.seed, cpu_threads=args.cpu_threads,
                device=core.normalize_device_name(args.device))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_stage_history(directory, history, stage_id, config, epochs):
    """Keep human-readable progress alongside the resumable checkpoint."""
    validation.write_json(directory / 'train_history.json', history)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'test_acc'])
    writer.writeheader()
    writer.writerows(history)
    csv_path = directory / 'epoch_history.csv'
    temporary = csv_path.with_suffix('.csv.tmp')
    temporary.write_text(stream.getvalue(), encoding='utf-8')
    temporary.replace(csv_path)
    log_lines = [f"stage_id={stage_id}",
                 f"sampling={config['train_sampling']} class_weights={config['class_weights']} "
                 f"lr={config['lr']} batch_size={config['batch_size']} epochs={epochs}"]
    for row in history:
        suffix = '' if 'test_acc' not in row else f" test_acc={row['test_acc']:.6f}"
        log_lines.append(f"epoch={row['epoch']}/{epochs} train_loss={row['train_loss']:.6f}{suffix}")
    log_path = directory / 'train.log'
    temporary = log_path.with_suffix('.log.tmp')
    temporary.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
    temporary.replace(log_path)


def assert_identity(saved, expected, context):
    if saved.get('run_id') != expected:
        raise ValueError(f'{context}: configuration/data/code mismatch; choose a new output directory')


def validate_dataset(X, y, subjects):
    if X.ndim != 3 or y.ndim != 1 or subjects.ndim != 1 or not len(X) == len(y) == len(subjects):
        raise ValueError('Expected aligned X[N,C,T], y[N], subject_ids[N]')
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(subjects.dtype, np.integer):
        raise ValueError('Labels and subject IDs must be integers')
    if not np.array_equal(np.unique(y), [0, 1, 2]):
        raise ValueError('Expected labels 0,1,2 corresponding to e1,e2,e3')
    for s in np.unique(subjects):
        if not np.array_equal(np.unique(y[subjects == s]), [0, 1, 2]):
            raise ValueError(f'Subject {s} must contain all three classes')


def fit_stage(X, y, subjects, train_index, eval_index, shape, config, seed, epochs,
              directory, stage_id, device, resume=False, model_factory=build_model):
    """Resumable fit; eval_index is the held-out test subject for selection runs."""
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / 'complete.json'
    checkpoint_path = directory / 'last_checkpoint.pt'
    predictions_path = directory / 'epoch_test_predictions.npz'
    if done_path.exists():
        done = read_json(done_path)
        assert_identity(done, stage_id, str(directory))
        if eval_index is not None and not predictions_path.exists():
            raise ValueError('Completed selection stage is missing test predictions')
        if eval_index is None and not (directory / 'model.pt').exists():
            raise ValueError('Completed refit is missing model.pt')
        return done
    train_inputs, stats = validation.fit_inputs(X[train_index], dual=True)
    validation.set_seed(seed)
    model = model_factory(shape, config).to(device)
    criterion, optimizer = training_components(model, config, device)
    loader = make_training_loader(train_inputs, y[train_index], config, seed)
    eval_loader = None
    if eval_index is not None:
        eval_inputs = validation.transform_inputs(X[eval_index], True, stats)
        eval_loader = validation.make_loader(eval_inputs, y[eval_index], config['batch_size'], False, seed)
    history, score_history, completed = [], [], 0
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError('Incomplete stage exists; pass --resume')
        saved = core.load_torch_checkpoint(checkpoint_path, torch.device('cpu'))
        assert_identity(saved, stage_id, str(checkpoint_path))
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        core.move_optimizer_state_to_device(optimizer, device)
        completed, history, score_history = saved['epoch'], saved['history'], saved['score_history']
        if completed > epochs or len(history) != completed or (eval_index is not None and len(score_history) != completed):
            raise ValueError('Invalid stage checkpoint epoch/history')
        loader.generator.set_state(saved['loader_rng'])
        core.restore_training_rng_state(saved['rng'], device)
        print(f'[RESUME] {directory.name}: epoch {completed}/{epochs}', flush=True)
    original_counts, draw_counts = training_draw_counts(y[train_index], config['train_sampling'])
    validation.write_json(directory / 'effective_training.json', dict(
        run_id=stage_id, model=config['model'], class_weights=config['class_weights'],
        train_sampling=config['train_sampling'], original_class_counts=original_counts,
        class_draws_per_epoch=draw_counts, optimizer_steps_per_epoch=int(np.ceil(sum(draw_counts) / config['batch_size'])),
        criterion=type(criterion).__name__, lr=config['lr'], seed=seed,
        train_subject_ids=np.unique(subjects[train_index]).tolist(),
        evaluation_subject_ids=[] if eval_index is None else np.unique(subjects[eval_index]).tolist(),
        normalizers=stats, epochs=epochs))
    write_stage_history(directory, history, stage_id, config, epochs)
    for epoch in range(completed + 1, epochs + 1):
        loss = validation.train_epoch(model, loader, optimizer, criterion, device)
        history.append(dict(epoch=epoch, train_loss=loss))
        status = ''
        if eval_loader is not None:
            true, log_probs = predict_log_probs(model, eval_loader, device)
            if not np.array_equal(true, y[eval_index]):
                raise RuntimeError('Test sample order changed')
            score_history.append(log_probs)
            test_accuracy = float(np.mean(predict_labels(log_probs, config['model']['classification_mode']) == true))
            history[-1]['test_acc'] = test_accuracy
            status = f' test_acc={test_accuracy:.4f}'
        core.atomic_torch_save(dict(run_id=stage_id, epoch=epoch, state_dict=model.state_dict(),
                                    optimizer=optimizer.state_dict(), history=history, score_history=score_history,
                                    rng=core.capture_training_rng_state(device), loader_rng=loader.generator.get_state()),
                               checkpoint_path)
        write_stage_history(directory, history, stage_id, config, epochs)
        print(f'{directory.parent.name}/{directory.name} epoch={epoch}/{epochs} loss={loss:.4f}{status}', flush=True)
    if eval_index is not None:
        core.atomic_save_npz(predictions_path, log_probs=np.stack(score_history), y_true=y[eval_index],
                             subject_ids=subjects[eval_index], sample_indices=eval_index)
    else:
        core.atomic_torch_save(dict(state_dict=model.state_dict(), model_config=config['model'], shape=list(shape),
                                    normalizers=stats, run_id=stage_id, epochs=epochs), directory / 'model.pt')
    done = dict(run_id=stage_id, epochs=epochs, normalizers=stats)
    validation.write_json(done_path, done)
    # Delete only the checkpoint written by this stage, after durable completion.
    checkpoint_path.unlink(missing_ok=True)
    return done


def select_best_test_epoch(predictions, config):
    """Use the historical per-fold highest held-out test accuracy rule."""
    true = predictions['y_true']
    scores = predictions['log_probs']
    if scores.shape != (config['epochs'], len(true), 3):
        raise ValueError('Incomplete epoch/test predictions')
    accuracies = [float(np.mean(predict_labels(scores[epoch], config['model']['classification_mode']) == true))
                  for epoch in range(config['epochs'])]
    best_index = int(np.argmax(accuracies))  # first epoch wins an exact tie
    return dict(epoch=best_index + 1, best_test_acc=accuracies[best_index],
                average_test_acc=float(np.mean(accuracies)), per_epoch_test_acc=accuracies,
                selection_source='held-out test subject', tie_rule='earliest epoch')


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def run_outer_fold(X, y, subjects, test_subject_id, config, directory, run_id, device,
                   resume=False, skip_existing=False, model_factory=build_model):
    fold_id = fingerprint(dict(run_id=run_id, test_subject_id=int(test_subject_id)))
    metrics_path = directory / 'outer_metrics.json'
    if metrics_path.exists():
        result = read_json(metrics_path)
        assert_identity(result, fold_id, str(directory))
        for name in ('outer_predictions.npz', 'refit_checkpoint.pt', 'selection.json'):
            if not (directory / name).exists():
                raise ValueError(f'Completed fold missing {name}')
        if not skip_existing:
            raise FileExistsError('Completed fold exists; pass --skip-existing or choose new output')
        print(f'[SKIP] subject={test_subject_id}', flush=True)
        return result
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError('Incomplete fold exists; pass --resume')
    directory.mkdir(parents=True, exist_ok=True)
    train_index = np.flatnonzero(subjects != test_subject_id)
    test_index = np.flatnonzero(subjects == test_subject_id)
    if len(train_index) == 0 or len(test_index) == 0:
        raise ValueError('LOSO fold needs both training and test subjects')
    validation.write_json(directory / 'split.json', dict(test_subject_id=int(test_subject_id),
                          train_subject_ids=np.unique(subjects[train_index]).tolist()))
    shape = tuple(X.shape[1:])
    selection_stage = directory / 'selection_fit'
    fit_stage(X, y, subjects, train_index, test_index, shape, config, config['seed'],
              config['epochs'], selection_stage, fingerprint(dict(fold_id=fold_id, stage='selection')),
              device, resume, model_factory)
    epoch_predictions = load_npz(selection_stage / 'epoch_test_predictions.npz')
    if not np.array_equal(epoch_predictions['sample_indices'], test_index):
        raise ValueError('Test prediction sample identity mismatch')
    selection = select_best_test_epoch(epoch_predictions, config)
    validation.write_json(directory / 'selection.json', selection)
    selected_scores = epoch_predictions['log_probs'][selection['epoch'] - 1]
    selected_labels = predict_labels(selected_scores, config['model']['classification_mode'])
    print(f"[SELECT] subject={test_subject_id} epoch={selection['epoch']} "
          f"best_test_acc={selection['best_test_acc']:.4f}", flush=True)

    # Reproduce the selected checkpoint with the same initialization, training
    # indices, sampler seed and epoch count. Selection predictions remain the
    # metric source, as in the historical best-test-epoch summaries.
    refit = directory / 'selected_checkpoint_fit'
    fit_stage(X, y, subjects, train_index, None, shape, config, config['seed'],
              selection['epoch'], refit, fingerprint(dict(fold_id=fold_id, stage='selected_checkpoint',
                                                         epoch=selection['epoch'])),
              device, resume, model_factory)
    checkpoint = core.load_torch_checkpoint(refit / 'model.pt', torch.device('cpu'))
    checkpoint.update(protocol='test_selected_resampling_v1', run_id=fold_id,
                      selected_epoch=selection['epoch'], best_test_acc=selection['best_test_acc'],
                      class_weights=config['class_weights'], train_sampling=config['train_sampling'],
                      test_subject_id=int(test_subject_id), seed=config['seed'])
    core.atomic_torch_save(checkpoint, directory / 'refit_checkpoint.pt')

    outer_path = directory / 'outer_predictions.npz'
    if outer_path.exists():
        outer = load_npz(outer_path)
        if str(outer['run_id']) != fold_id or not np.array_equal(outer['sample_indices'], test_index):
            raise ValueError('Outer prediction identity mismatch')
    else:
        model = model_factory(shape, config).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        test_inputs = validation.transform_inputs(X[test_index], True, checkpoint['normalizers'])
        loader = validation.make_loader(test_inputs, y[test_index], config['batch_size'], False, config['seed'])
        true, refit_scores = predict_log_probs(model, loader, device)
        if not np.array_equal(true, epoch_predictions['y_true']):
            raise RuntimeError('Selected checkpoint test labels changed')
        np.testing.assert_allclose(refit_scores, selected_scores, rtol=1e-5, atol=1e-6,
                                   err_msg='Selected checkpoint differs from the scored epoch')
        outer = dict(run_id=np.asarray(fold_id), y_true=true, log_probs=selected_scores,
                     probabilities=np.exp(selected_scores), y_pred=selected_labels,
                     sample_indices=test_index, subject_ids=subjects[test_index])
        core.atomic_save_npz(outer_path, **outer)
    best_metrics = validation.metric_summary(outer['y_true'], outer['y_pred'], 3)
    if not np.isclose(best_metrics['accuracy'], selection['best_test_acc']):
        raise RuntimeError('Selected predictions do not match best test accuracy')
    original_counts, draw_counts = training_draw_counts(y[train_index], config['train_sampling'])
    result = dict(run_id=fold_id, test_subject_id=int(test_subject_id), seed=config['seed'],
                  best_epoch=selection['epoch'], best_test_acc=selection['best_test_acc'],
                  average_test_acc=selection['average_test_acc'],
                  selection_source='held-out test subject',
                  class_weights=config['class_weights'], train_sampling=config['train_sampling'],
                  model_config=config['model'], epochs=config['epochs'],
                  batch_size=config['batch_size'], learning_rate=config['lr'], device=config['device'],
                  original_class_counts=original_counts, class_draws_per_epoch=draw_counts,
                  classification_mode=config['model']['classification_mode'],
                  n_train_samples=len(train_index), n_test_samples=len(test_index),
                  selection_train_log=str(selection_stage / 'train.log'),
                  selection_epoch_history_csv=str(selection_stage / 'epoch_history.csv'),
                  best_test_metrics=best_metrics,
                  confusion_matrix=best_metrics['confusion_matrix'],
                  per_class_metrics=best_metrics['per_class'], macro_f1=best_metrics['macro_f1'])
    validation.write_json(metrics_path, result)
    return result


def summarize(results):
    metrics = [row['best_test_metrics'] for row in results]
    mean_recall = np.mean([[c['recall'] for c in item['per_class']] for item in metrics], axis=0)
    return dict(completed_subject_ids=sorted(row['test_subject_id'] for row in results),
                n_subjects=len(results), selection_source='held-out test subject',
                mean_best_test_acc=float(np.mean([row['best_test_acc'] for row in results])),
                mean_average_test_acc=float(np.mean([row['average_test_acc'] for row in results])),
                mean_subject_macro_f1=float(np.mean([item['macro_f1'] for item in metrics])),
                mean_subject_recall=mean_recall.tolist(),
                mean_subject_precision=np.mean([[c['precision'] for c in item['per_class']]
                                                for item in metrics], axis=0).tolist(),
                pooled_confusion_matrix=np.sum([item['confusion_matrix'] for item in metrics], axis=0).tolist(),
                best_epochs={str(row['test_subject_id']): row['best_epoch'] for row in results})


def main(argv=None):
    args = parse_args(argv)
    config = resolve_config(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(config['cpu_threads'])
    dataset = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    X = np.load(dataset / 'X.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(dataset / 'y.npy', allow_pickle=False)
    subjects = np.load(dataset / 'subject_ids.npy', allow_pickle=False)
    validate_dataset(X, y, subjects)
    requested = batch.parse_subject_id_list(args.subject_ids) or np.unique(subjects).astype(int).tolist()
    if not set(requested).issubset(set(np.unique(subjects).astype(int).tolist())):
        raise ValueError('Requested subject ID is absent from the dataset')
    source = Path(__file__).resolve().parent
    identity = dict(config=config, dataset_root=str(dataset),
                    data_sha256={name: validation.sha256_file(dataset / name)
                                 for name in ('X.npy', 'y.npy', 'subject_ids.npy', 'metadata.json')},
                    source_sha256={name: validation.sha256_file(source / name)
                                   for name in (Path(__file__).name, 'train_activity_loso.py',
                                                'train_activity_subject_validation.py', 'train_activity_loso_batch.py',
                                                'comparison_models.py')},
                    runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 cuda=torch.version.cuda))
    run_id = fingerprint(identity)
    plan_path = output / 'protocol_plan.json'
    if output.exists():
        if not plan_path.exists():
            raise FileExistsError('Output exists without resampling protocol identity; choose a NEW directory')
        assert_identity(read_json(plan_path), run_id, str(output))
        if not (args.resume or args.skip_existing):
            raise FileExistsError('Output exists; pass --resume/--skip-existing or choose new directory')
    original_counts, draw_counts = training_draw_counts(y, config['train_sampling'])
    plan = dict(run_id=run_id, **identity, requested_subject_ids=requested,
                budget=dict(selection_fits=len(requested), selected_checkpoint_fits=len(requested),
                            maximum_total_epochs=2 * len(requested) * args.epochs),
                full_dataset_original_class_counts=original_counts,
                full_dataset_class_draws_per_epoch=draw_counts,
                selection='Highest held-out test accuracy in each LOSO fold; exact ties use earliest epoch',
                limitations=[
                    'Selecting epochs with the held-out test subject makes mean best test accuracy optimistic.',
                    'These 11 subjects also informed previous architecture and hyperparameter choices.',
                    'X.npy was normalized per complete video record before this training protocol.',
                    'At the same epoch count, triple minority sampling uses more optimizer updates than original sampling.'])
    output.mkdir(parents=True, exist_ok=True)
    validation.write_json(plan_path, plan)
    print(json.dumps(dict(output=str(output), class_weights=config['class_weights'],
                          train_sampling=config['train_sampling'], class_draws_per_epoch=draw_counts,
                          model=config['model'], budget=plan['budget']), indent=2), flush=True)
    if args.dry_run:
        print('[DRY RUN] No model constructed or trained. Use the same command without --dry-run and with --resume.')
        return
    device = torch.device(core.validate_device(config['device']))
    for test_subject_id in requested:
        run_outer_fold(X, y, subjects, test_subject_id, config, output / f'fold_subject_{test_subject_id}',
                       run_id, device, args.resume, args.skip_existing)
        results = []
        for path in sorted(output.glob('fold_subject_*/outer_metrics.json')):
            result = read_json(path)
            assert_identity(result, fingerprint(dict(run_id=run_id, test_subject_id=result['test_subject_id'])),
                            str(path))
            results.append(result)
        validation.write_json(output / 'summary.json', summarize(results))
    print(json.dumps(read_json(output / 'summary.json'), indent=2), flush=True)


if __name__ == '__main__':
    main()
