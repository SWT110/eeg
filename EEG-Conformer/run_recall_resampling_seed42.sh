#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON_BIN:-./.conda-envs/eegconformer310/bin/python}"
device="${DEVICE:-cuda:0}"
run_mode="${RUN_MODE:-dry-run}"
output_root="${OUTPUT_ROOT:-local_artifacts/outputs/activity_resampling_test_best_seed42}"
run_groups="${RUN_GROUPS:-unweighted weighted_3_3_1 triple_minority}"

if [[ "$run_mode" != dry-run && "$run_mode" != train ]]; then
  echo "RUN_MODE must be dry-run or train" >&2
  exit 2
fi

extra_args=()
if [[ "$run_mode" == dry-run ]]; then
  extra_args+=(--dry-run)
fi
if [[ -n "${SUBJECT_IDS:-}" ]]; then
  extra_args+=(--subject-ids "$SUBJECT_IDS")
fi

for group in $run_groups; do
  case "$group" in
    unweighted)
      class_weights=1,1,1
      sampling=original
      ;;
    weighted_3_3_1)
      class_weights=3,3,1
      sampling=original
      ;;
    triple_minority)
      class_weights=1,1,1
      sampling=triple_minority_replacement
      ;;
    *)
      echo "Unknown group: $group" >&2
      exit 2
      ;;
  esac

  "$python_bin" EEG-Conformer/train_activity_resampling.py \
    --dataset-root local_artifacts/data_to_list/global_activity_dataset/window_15_stride_3 \
    --output-dir "$output_root/$group/window_15_stride_3" \
    --device "$device" \
    --epochs 200 \
    --batch-size 72 \
    --classification-mode hierarchical \
    --class-weights "$class_weights" \
    --train-sampling "$sampling" \
    --input-domain time_fft \
    --conv-type standard \
    --fft-global none \
    --transformer-branches 3 \
    --transformer-depths 11 10 8 \
    --transformer-encoder-dropout 0.85 \
    --transformer-branch-fusion loss_softmax \
    --branch-loss-aux-weight 0.2 \
    --transformer-branch-qkv cross_depth \
    --transformer-branch-qkv-dropout 0.25 \
    --seed 42 \
    --cpu-threads 12 \
    --skip-existing \
    --resume \
    "${extra_args[@]}"
done
