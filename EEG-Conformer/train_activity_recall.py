"""Subject-validated recall tuning for the existing time/FFT Conformer.

Same model/training flags as train_activity_loso_batch.py, separate artifacts.
An outer subject never selects epochs, loss weights or decision thresholds.
Resume is exact at completed epoch boundaries, including the loader RNG.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import train_activity_loso as core
import train_activity_loso_batch as batch
import train_activity_subject_validation as validation

VERSION = 1


def parse_args(argv=None):
    parser = batch.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=None)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--split-seed', type=int, default=20260906)
    parser.add_argument('--min-val-accuracy', type=float, default=.70)
    parser.add_argument('--gate-thresholds', default='0.35,0.40,0.45,0.50,0.55,0.60',
                        help='Hierarchical P(e1)+P(e2) thresholds; always includes .5')
    parser.add_argument('--within-thresholds', default='0.45,0.50,0.55',
                        help='Hierarchical P(e1 | group12) thresholds; always includes .5')
    parser.add_argument('--flat-biases', default='0,0.25,0.5',
                        help='Independent additive log-probability biases for e1/e2; includes 0')
    parser.add_argument('--cpu-threads', type=int, default=12)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def float_grid(raw, neutral, probability=False):
    values = sorted(set([neutral] + [float(x) for x in raw.split(',')]))
    if any(not np.isfinite(x) or (probability and not 0 < x < 1) for x in values):
        raise ValueError('Decision grids require finite values (thresholds strictly between 0 and 1)')
    return values


def decision_grid(args):
    if args.classification_mode == 'hierarchical':
        rules = [dict(mode='hierarchical', gate_threshold=g, within_threshold=w)
                 for g in float_grid(args.gate_thresholds, .5, True)
                 for w in float_grid(args.within_thresholds, .5, True)]
    else:
        rules = [dict(mode='flat', biases=[b1, b2, 0.])
                 for b1 in float_grid(args.flat_biases, 0.)
                 for b2 in float_grid(args.flat_biases, 0.)]
    # Neutral first, then smallest departure. Deterministic ties favour no calibration.
    return sorted(rules, key=decision_distance)


def decision_distance(rule):
    if rule['mode'] == 'hierarchical':
        return abs(rule['gate_threshold'] - .5) + abs(rule['within_threshold'] - .5)
    return sum(abs(x) for x in rule['biases'])


def apply_decision(log_probs, rule):
    scores = np.asarray(log_probs)
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError('Expected finite log probabilities [N,3]')
    if rule['mode'] == 'flat':
        return (scores + np.asarray(rule['biases'])).argmax(1)
    group_log = np.logaddexp(scores[:, 0], scores[:, 1])
    group = group_log >= np.log(rule['gate_threshold'])
    first = scores[:, 0] - group_log >= np.log(rule['within_threshold'])
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


def resolve_config(args):
    if args.output_dir is None:
        raise ValueError('--output-dir is required; use a NEW recall-specific directory')
    if args.input_domain != 'time_fft':
        raise ValueError('This entrypoint supports --input-domain time_fft (flat or hierarchical)')
    if args.epochs < 1 or args.batch_size < 2 or args.cpu_threads < 1:
        raise ValueError('epochs >= 1, batch-size >= 2, cpu-threads >= 1 required')
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('lr must be finite and positive')
    if not 0 <= args.min_val_accuracy <= 1:
        raise ValueError('min-val-accuracy must be in [0,1]')
    if not 0 <= args.seed < 2**32 - 1000004 or not 0 <= args.split_seed < 2**32:
        raise ValueError('Seed out of supported range')
    weights = core.parse_class_weights(args.class_weights)
    if weights is not None and (len(weights) != 3 or any(not np.isfinite(w) or w <= 0 for w in weights)):
        raise ValueError('Recall training requires three strictly positive finite class weights')
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
    return dict(version=VERSION, model=model, class_weights=weights, lr=args.lr, epochs=args.epochs,
                batch_size=args.batch_size, seed=args.seed, split_seed=args.split_seed,
                inner_folds=args.inner_folds, min_val_accuracy=args.min_val_accuracy,
                decisions=decision_grid(args), cpu_threads=args.cpu_threads,
                device=core.normalize_device_name(args.device))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


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


def fit_stage(X, y, subjects, train_index, val_index, shape, config, seed, epochs,
              directory, stage_id, device, resume=False, model_factory=build_model):
    """A resumable inner fit or refit; only val_index is evaluated during training.

    Atomic checkpoint is authoritative at an epoch boundary. Inner score history
    is small compared with model/Adam states. Completed inner weights are removed
    after a scores-only completion artifact commits; refit weights are retained.
    """
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / 'complete.json'
    checkpoint_path = directory / 'last_checkpoint.pt'
    predictions_path = directory / 'validation_predictions.npz'
    if done_path.exists():
        done = read_json(done_path)
        assert_identity(done, stage_id, str(directory))
        if val_index is not None and not predictions_path.exists():
            raise ValueError('Completed inner stage is missing validation predictions')
        if val_index is None and not (directory / 'model.pt').exists():
            raise ValueError('Completed refit is missing model.pt')
        return done
    train_inputs, stats = validation.fit_inputs(X[train_index], dual=True)
    validation.set_seed(seed)
    model = model_factory(shape, config).to(device)
    criterion, optimizer = training_components(model, config, device)
    loader = validation.make_loader(train_inputs, y[train_index], config['batch_size'], True, seed)
    val_loader = None
    if val_index is not None:
        val_inputs = validation.transform_inputs(X[val_index], True, stats)
        val_loader = validation.make_loader(val_inputs, y[val_index], config['batch_size'], False, seed)
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
        if completed > epochs or len(history) != completed or (val_index is not None and len(score_history) != completed):
            raise ValueError('Invalid stage checkpoint epoch/history')
        loader.generator.set_state(saved['loader_rng'])
        core.restore_training_rng_state(saved['rng'], device)
        print(f'[RESUME] {directory.name}: epoch {completed}/{epochs}', flush=True)
    validation.write_json(directory / 'effective_training.json', dict(
        run_id=stage_id, model=config['model'], class_weights=config['class_weights'],
        criterion=type(criterion).__name__, lr=config['lr'], seed=seed,
        train_subject_ids=np.unique(subjects[train_index]).tolist(),
        validation_subject_ids=[] if val_index is None else np.unique(subjects[val_index]).tolist(),
        normalizers=stats, epochs=epochs))
    for epoch in range(completed + 1, epochs + 1):
        loss = validation.train_epoch(model, loader, optimizer, criterion, device)
        history.append(dict(epoch=epoch, train_loss=loss))
        status = ''
        if val_loader is not None:
            true, log_probs = predict_log_probs(model, val_loader, device)
            if not np.array_equal(true, y[val_index]):
                raise RuntimeError('Validation sample order changed')
            score_history.append(log_probs)
            neutral = subject_mean_metrics(true, apply_decision(log_probs, config['decisions'][0]), subjects[val_index])
            status = (f" val_acc={neutral['accuracy']:.4f}"
                      f" val_R1/R2={neutral['recall'][0]:.4f}/{neutral['recall'][1]:.4f}")
        core.atomic_torch_save(dict(run_id=stage_id, epoch=epoch, state_dict=model.state_dict(),
                                    optimizer=optimizer.state_dict(), history=history, score_history=score_history,
                                    rng=core.capture_training_rng_state(device), loader_rng=loader.generator.get_state()),
                               checkpoint_path)
        print(f'{directory.parent.name}/{directory.name} epoch={epoch}/{epochs} loss={loss:.4f}{status}', flush=True)
    validation.write_json(directory / 'train_history.json', history)
    if val_index is not None:
        core.atomic_save_npz(predictions_path, log_probs=np.stack(score_history), y_true=y[val_index],
                             subject_ids=subjects[val_index], sample_indices=val_index)
    else:
        core.atomic_torch_save(dict(state_dict=model.state_dict(), model_config=config['model'], shape=list(shape),
                                    normalizers=stats, run_id=stage_id, epochs=epochs), directory / 'model.pt')
    done = dict(run_id=stage_id, epochs=epochs, normalizers=stats)
    validation.write_json(done_path, done)
    # Delete only the checkpoint written by this stage, after durable completion.
    checkpoint_path.unlink(missing_ok=True)
    return done


def subject_mean_metrics(y, predicted, subjects):
    metrics = [validation.metric_summary(y[subjects == s], predicted[subjects == s], 3)
               for s in np.unique(subjects)]
    return dict(accuracy=float(np.mean([m['accuracy'] for m in metrics])),
                macro_f1=float(np.mean([m['macro_f1'] for m in metrics])),
                recall=np.mean([[p['recall'] for p in m['per_class']] for m in metrics], axis=0).tolist(),
                precision=np.mean([[p['precision'] for p in m['per_class']] for m in metrics], axis=0).tolist())


def select_operating_point(inner_predictions, expected_subjects, config):
    """Joint epoch/decision selection, with equal subject weights across unequal groups."""
    seen = [int(s) for item in inner_predictions for s in np.unique(item['subject_ids'])]
    if sorted(seen) != sorted(expected_subjects):
        raise ValueError('Validation subjects must cover outer training subjects exactly once')
    scores = np.concatenate([item['log_probs'] for item in inner_predictions], axis=1)
    true = np.concatenate([item['y_true'] for item in inner_predictions])
    subjects = np.concatenate([item['subject_ids'] for item in inner_predictions])
    if scores.shape != (config['epochs'], len(true), 3):
        raise ValueError('Incomplete epoch/validation predictions')
    table = []
    for e in range(config['epochs']):
        for d, rule in enumerate(config['decisions']):
            m = subject_mean_metrics(true, apply_decision(scores[e], rule), subjects)
            table.append(dict(epoch=e + 1, decision_index=d, **m,
                              min_recall12=min(m['recall'][:2]), mean_recall12=float(np.mean(m['recall'][:2])),
                              meets_accuracy_floor=m['accuracy'] >= config['min_val_accuracy']))
    eligible = [r for r in table if r['meets_accuracy_floor']]
    if eligible:
        winner = min(eligible, key=lambda r: (-r['min_recall12'], -r['macro_f1'], -r['accuracy'],
                                               r['epoch'], r['decision_index']))
    else:
        winner = min(table, key=lambda r: (-r['macro_f1'], -r['accuracy'], -r['min_recall12'],
                                            r['epoch'], r['decision_index']))
    return dict(**winner, decision=config['decisions'][winner['decision_index']],
                accuracy_floor=config['min_val_accuracy'], accuracy_floor_met=bool(eligible),
                fallback=None if eligible else 'maximum_mean_subject_macro_f1',
                objective='min(mean_subject_recall_e1, mean_subject_recall_e2)',
                averaging='equal weight per subject, not per validation group or window',
                tie_rule='macro_f1, accuracy, earlier epoch, decision order (neutral first)',
                candidate_scores=table)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def run_outer_fold(X, y, subjects, split, config, directory, run_id, device,
                   resume=False, skip_existing=False, model_factory=build_model):
    fold_id = fingerprint(dict(run_id=run_id, split=split))
    metrics_path = directory / 'outer_metrics.json'
    if metrics_path.exists():
        result = read_json(metrics_path)
        assert_identity(result, fold_id, str(directory))
        for name in ('outer_predictions.npz', 'refit_checkpoint.pt', 'selection.json'):
            if not (directory / name).exists():
                raise ValueError(f'Completed fold missing {name}')
        if not skip_existing:
            raise FileExistsError('Completed fold exists; pass --skip-existing or choose new output')
        print(f'[SKIP] subject={split["test_subject_id"]}', flush=True)
        return result
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError('Incomplete fold exists; pass --resume')
    directory.mkdir(parents=True, exist_ok=True)
    split_path = directory / 'split.json'
    if split_path.exists() and read_json(split_path) != split:
        raise ValueError('Existing fold has a different subject split')
    validation.write_json(split_path, split)
    shape = tuple(X.shape[1:])
    inner_predictions = []
    for inner in split['inner_folds']:
        train_index = np.flatnonzero(np.isin(subjects, inner['train_subject_ids']))
        val_index = np.flatnonzero(np.isin(subjects, inner['validation_subject_ids']))
        stage = directory / f"inner_{inner['inner_fold']}"
        fit_stage(X, y, subjects, train_index, val_index, shape, config,
                  config['seed'] + 1009 * (inner['inner_fold'] + 1), config['epochs'], stage,
                  fingerprint(dict(fold_id=fold_id, inner=inner)), device, resume, model_factory)
        inner_predictions.append(load_npz(stage / 'validation_predictions.npz'))
    selection = select_operating_point(inner_predictions, split['outer_train_subject_ids'], config)
    validation.write_json(directory / 'selection.json', selection)
    # Retain selected-epoch OOF scores explicitly for future calibration diagnostics.
    core.atomic_save_npz(directory / 'selected_validation_predictions.npz',
                         log_probs=np.concatenate([p['log_probs'][selection['epoch'] - 1] for p in inner_predictions]),
                         y_true=np.concatenate([p['y_true'] for p in inner_predictions]),
                         subject_ids=np.concatenate([p['subject_ids'] for p in inner_predictions]),
                         sample_indices=np.concatenate([p['sample_indices'] for p in inner_predictions]))
    del inner_predictions
    print(f"[SELECT] subject={split['test_subject_id']} epoch={selection['epoch']} "
          f"val_acc={selection['accuracy']:.4f} val_recall={selection['recall']} "
          f"floor_met={selection['accuracy_floor_met']} decision={selection['decision']}", flush=True)
    train_index = np.flatnonzero(np.isin(subjects, split['outer_train_subject_ids']))
    refit = directory / 'refit'
    fit_stage(X, y, subjects, train_index, None, shape, config, config['seed'] + 1000003,
              selection['epoch'], refit, fingerprint(dict(fold_id=fold_id, epoch=selection['epoch'])),
              device, resume, model_factory)
    checkpoint = core.load_torch_checkpoint(refit / 'model.pt', torch.device('cpu'))
    checkpoint.update(protocol='subject_validated_recall_v1', run_id=fold_id,
                      decision=selection['decision'], neutral_decision=config['decisions'][0],
                      class_weights=config['class_weights'], test_subject_id=split['test_subject_id'],
                      seed=config['seed'], validation_accuracy_floor_met=selection['accuracy_floor_met'])
    # Freeze the checkpoint and decision before any outer predictions/metrics.
    core.atomic_torch_save(checkpoint, directory / 'refit_checkpoint.pt')
    outer_path = directory / 'outer_predictions.npz'
    test_index = np.flatnonzero(subjects == split['test_subject_id'])
    if outer_path.exists():
        # Recover after a crash between prediction and metric commit, without re-evaluation.
        outer = load_npz(outer_path)
        if str(outer['run_id']) != fold_id or not np.array_equal(outer['sample_indices'], test_index):
            raise ValueError('Outer prediction identity mismatch')
    else:
        model = model_factory(shape, config).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        test_inputs = validation.transform_inputs(X[test_index], True, checkpoint['normalizers'])
        loader = validation.make_loader(test_inputs, y[test_index], config['batch_size'], False, config['seed'])
        true, log_probs = predict_log_probs(model, loader, device)
        outer = dict(run_id=np.asarray(fold_id), y_true=true, log_probs=log_probs, probabilities=np.exp(log_probs),
                     y_pred=apply_decision(log_probs, selection['decision']),
                     y_pred_default=apply_decision(log_probs, config['decisions'][0]),
                     sample_indices=test_index, subject_ids=subjects[test_index])
        core.atomic_save_npz(outer_path, **outer)
    result = dict(run_id=fold_id, test_subject_id=split['test_subject_id'], seed=config['seed'],
                  selected_epochs=selection['epoch'], decision=selection['decision'],
                  class_weights=config['class_weights'], classification_mode=config['model']['classification_mode'],
                  validation_accuracy_floor_met=selection['accuracy_floor_met'],
                  selection_fallback=selection['fallback'], outer_evaluation_count=1,
                  n_train_samples=len(train_index), n_test_samples=len(test_index),
                  calibrated=validation.metric_summary(outer['y_true'], outer['y_pred'], 3),
                  default_decision=validation.metric_summary(outer['y_true'], outer['y_pred_default'], 3))
    # Completion marker is written last.
    validation.write_json(metrics_path, result)
    return result


def summarize(results):
    summary = dict(completed_subject_ids=sorted(r['test_subject_id'] for r in results),
                   n_subjects=len(results), validation_floor_failed_subject_ids=[r['test_subject_id'] for r in results
                                                                               if not r['validation_accuracy_floor_met']])
    for key in ('calibrated', 'default_decision'):
        mean_recall = np.mean([[p['recall'] for p in r[key]['per_class']] for r in results], axis=0)
        summary[key] = dict(mean_subject_accuracy=float(np.mean([r[key]['accuracy'] for r in results])),
                            mean_subject_macro_f1=float(np.mean([r[key]['macro_f1'] for r in results])),
                            mean_subject_recall=mean_recall.tolist(), min_mean_recall12=float(min(mean_recall[:2])),
                            mean_subject_precision=np.mean([[p['precision'] for p in r[key]['per_class']]
                                                            for r in results], axis=0).tolist(),
                            pooled_confusion_matrix=np.sum([r[key]['confusion_matrix'] for r in results], axis=0).tolist())
    return summary


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
    splits = [validation.make_subject_splits(subjects, s, args.inner_folds, args.split_seed) for s in requested]
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
            raise FileExistsError('Output exists without recall protocol identity; choose a NEW directory')
        assert_identity(read_json(plan_path), run_id, str(output))
        if not (args.resume or args.skip_existing):
            raise FileExistsError('Output exists; pass --resume/--skip-existing or choose new directory')
    plan = dict(run_id=run_id, **identity, requested_subject_ids=requested, splits=splits,
                budget=dict(inner_fits=len(splits) * args.inner_folds, refits=len(splits),
                            maximum_total_epochs=len(splits) * (args.inner_folds + 1) * args.epochs),
                selection='Subject-equal min(R1,R2) under validation accuracy floor; fallback macro-F1',
                limitations=validation.LIMITATIONS + [
                    'Validation accuracy is not a guarantee on the unseen outer subject.',
                    'Calibration selected on inner models may transfer imperfectly to the fresh refit.',
                    'Default-decision metrics use the SAME recall-selected refit, not a separately selected baseline.'])
    output.mkdir(parents=True, exist_ok=True)
    validation.write_json(plan_path, plan)
    print(json.dumps(dict(output=str(output), class_weights=config['class_weights'],
                          model=config['model'], budget=plan['budget']), indent=2), flush=True)
    if args.dry_run:
        print('[DRY RUN] No model constructed or trained. Use the same command without --dry-run and with --resume.')
        return
    device = torch.device(core.validate_device(config['device']))
    for split in splits:
        run_outer_fold(X, y, subjects, split, config, output / f"fold_subject_{split['test_subject_id']}",
                       run_id, device, args.resume, args.skip_existing)
        results = []
        for path in sorted(output.glob('fold_subject_*/outer_metrics.json')):
            result = read_json(path)
            expected_split = validation.make_subject_splits(subjects, result['test_subject_id'], args.inner_folds, args.split_seed)
            assert_identity(result, fingerprint(dict(run_id=run_id, split=expected_split)), str(path))
            results.append(result)
        validation.write_json(output / 'summary.json', summarize(results))
    print(json.dumps(read_json(output / 'summary.json'), indent=2), flush=True)


if __name__ == '__main__':
    main()
