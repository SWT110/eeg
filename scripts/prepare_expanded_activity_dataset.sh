#!/usr/bin/env bash
set -euo pipefail

EEG_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EEG_TASK_PY="${EEG_TASK_PY:-python}"
EEG_EXCLUDED_IDS="6"
EEG_COHORT_IDS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"
EEG_PROCESS_IDS="12,13,14,15,16,17"
EEG_OUTPUT_BASE="$EEG_REPO_ROOT/local_artifacts/data_to_list/global_activity_dataset_1to17_S17_first3"
EEG_REBUILD=0

usage() {
  cat <<'EOF'
Prepare the expanded 15s/3s EEG cohort, retaining old normalized records.
Usage: bash scripts/prepare_expanded_activity_dataset.sh [options]
  --python PATH                  Preprocessing Python (or EEG_TASK_PY environment variable)
  --exclude-subject-ids 6,8       Whole people removed from main arrays (default: 6)
  --cohort-subject-ids 1,2,...     Source cohort IDs (default: 1 through 17)
  --process-subject-ids 12,13,...  Newly converted/cut/normalized IDs (default: 12 through 17)
  --output-base PATH             Main output base; exclusion tag is added automatically
  --rebuild-datasets             Explicitly rebuild existing derived NPY datasets

Prerequisites: new EDFs + original time_data.txt in data/, and old normalized
records for subjects 1-11 in list_normalization_fixed_duration/. Raw and Excel
records are retained. Every excluded person gets a separate Failurecase dataset.
EOF
}

while (($#)); do
  case "$1" in
    --python|--exclude-subject-ids|--cohort-subject-ids|--process-subject-ids|--output-base)
      if (($# < 2)); then usage >&2; exit 2; fi
      case "$1" in
        --python) EEG_TASK_PY="$2" ;;
        --exclude-subject-ids) EEG_EXCLUDED_IDS="$2" ;;
        --cohort-subject-ids) EEG_COHORT_IDS="$2" ;;
        --process-subject-ids) EEG_PROCESS_IDS="$2" ;;
        --output-base) EEG_OUTPUT_BASE="$2" ;;
      esac
      shift 2 ;;
    --rebuild-datasets) EEG_REBUILD=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd -- "$EEG_REPO_ROOT"
EEG_NORMALIZED_ROOT="$("$EEG_TASK_PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from eeg_project_paths import LIST_NORMALIZED_DIR; print(LIST_NORMALIZED_DIR)' "$EEG_REPO_ROOT")"
IFS=',' read -r -a EEG_PROCESS_ARRAY <<< "$EEG_PROCESS_IDS"
IFS=',' read -r -a EEG_EXCLUDED_ARRAY <<< "$EEG_EXCLUDED_IDS"
EEG_BUILD_OPTIONS=()
if ((EEG_REBUILD)); then EEG_BUILD_OPTIONS+=(--overwrite); fi

# Fail before any heavy conversion if the preserved source cohort is unavailable.
"$EEG_TASK_PY" - "$EEG_REPO_ROOT" "$EEG_COHORT_IDS" "$EEG_EXCLUDED_IDS" "$EEG_PROCESS_IDS" <<'PY'
import sys
from pathlib import Path
repo=Path(sys.argv[1])
sys.path.insert(0,str(repo/'eeg-data-processing'/'data_to_list'))
from build_activity_global_index import parse_subject_ids
from eeg_project_paths import DATA_TO_LIST_DATA_DIR, LIST_NORMALIZED_DIR
cohort=parse_subject_ids(sys.argv[2])
excluded=parse_subject_ids(sys.argv[3])
process=parse_subject_ids(sys.argv[4])
if set(excluded+process)-set(cohort):
    raise ValueError('Excluded/processed IDs must belong to the source cohort')
if len(set(cohort)-set(excluded)) < 2:
    raise ValueError('LOSO needs at least two formal subjects')
time_file=DATA_TO_LIST_DATA_DIR/'time_data.txt'
if not time_file.is_file():
    raise FileNotFoundError(time_file)
missing=[str(LIST_NORMALIZED_DIR/str(sid)/f'{sid}_e_{label}.xlsx')
         for sid in cohort if sid not in process for label in (1,2,3)
         if not (LIST_NORMALIZED_DIR/str(sid)/f'{sid}_e_{label}.xlsx').is_file()]
if missing:
    raise FileNotFoundError('Preserved normalized EEG records are missing; restore the input bundle first: '+', '.join(missing))
missing_edfs=[str(DATA_TO_LIST_DATA_DIR/f'{sid}_e.edf') for sid in process
              if not (DATA_TO_LIST_DATA_DIR/f'{sid}_e.edf').is_file()]
if missing_edfs:
    raise FileNotFoundError('New EDFs missing: '+', '.join(missing_edfs))
print('Source cohort:',cohort,'formal cohort:',sorted(set(cohort)-set(excluded)))
print('New preprocessing subjects:',process,'excluded subjects:',excluded)
PY

"$EEG_TASK_PY" -u eeg-data-processing/data_to_list/2.edf_to_list.py \
  --subject-ids "${EEG_PROCESS_ARRAY[@]}"
"$EEG_TASK_PY" -u eeg-data-processing/data_to_list/3.time_edit_fixed_duration.py \
  --subject-ids "${EEG_PROCESS_ARRAY[@]}"
"$EEG_TASK_PY" -u eeg-data-processing/data_to_list/6.edf_list_normalization_fixed_duration.py \
  --subject-ids "${EEG_PROCESS_ARRAY[@]}"

"$EEG_TASK_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
  --output-root "$EEG_OUTPUT_BASE/window_15_stride_3" \
  --window-seconds 15 --stride-seconds 3 \
  --include-subject-ids "$EEG_COHORT_IDS" --exclude-subject-ids "$EEG_EXCLUDED_IDS" \
  --require-complete-records "${EEG_BUILD_OPTIONS[@]}" \
  --input-root "$EEG_NORMALIZED_ROOT"

for EEG_CASE_ID in "${EEG_EXCLUDED_ARRAY[@]}"; do
  "$EEG_TASK_PY" -u eeg-data-processing/data_to_list/build_activity_global_index.py \
    --output-root "$EEG_REPO_ROOT/local_artifacts/data_to_list/global_activity_failurecase_S${EEG_CASE_ID}/window_15_stride_3" \
    --window-seconds 15 --stride-seconds 3 --include-subject-ids "$EEG_CASE_ID" \
    --require-complete-records "${EEG_BUILD_OPTIONS[@]}" \
    --input-root "$EEG_NORMALIZED_ROOT"
done
