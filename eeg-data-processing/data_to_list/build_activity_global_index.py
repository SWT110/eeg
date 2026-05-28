from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
EEG_ROOT = BASE_DIR.parents[1]
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import GLOBAL_ACTIVITY_DATASET_DIR, LIST_NORMALIZED_DIR

# ---------------------------------------------------------------------------
# Reuse helper functions from the sibling script (8.xlsx_to_npy_dataset.py)
# ---------------------------------------------------------------------------

_SIBLING = BASE_DIR / "8.xlsx_to_npy_dataset.py"
_spec = importlib.util.spec_from_file_location("_xlsx_to_npy", _SIBLING)
assert _spec is not None and _spec.loader is not None
_xlsx_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xlsx_mod)

estimate_sampling_interval = _xlsx_mod.estimate_sampling_interval  # type: ignore[attr-defined]
extract_windows = _xlsx_mod.extract_windows  # type: ignore[attr-defined]
load_xlsx_file = _xlsx_mod.load_xlsx_file  # type: ignore[attr-defined]
prompt_path = _xlsx_mod.prompt_path  # type: ignore[attr-defined]
prompt_positive_float = _xlsx_mod.prompt_positive_float  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT_ROOT = LIST_NORMALIZED_DIR
DEFAULT_OUTPUT_ROOT = GLOBAL_ACTIVITY_DATASET_DIR
SAMPLING_INTERVAL_RTOL = 1e-4
SAMPLING_INTERVAL_ATOL = 1e-8

EEG_FILE_PATTERN = re.compile(r"^(?P<subject>\d+)_e_(?P<label>[123])\.xlsx$")
LABEL_MAP: dict[str, int] = {"e_1": 0, "e_2": 1, "e_3": 2}


# ---------------------------------------------------------------------------
# Config resolution (mirrors resolve_runtime_config with local defaults)
# ---------------------------------------------------------------------------

def resolve_runtime_config(
    input_root: Path | str | None,
    output_root: Path | str | None,
    window_seconds: float | None,
    stride_seconds: float | None,
) -> tuple[Path, Path, float, float]:
    if input_root is None:
        input_root_path = prompt_path("Input directory", DEFAULT_INPUT_ROOT, must_exist=True)
    else:
        input_root_path = Path(input_root).expanduser()

    if output_root is None:
        output_root_path = prompt_path("Output directory", DEFAULT_OUTPUT_ROOT)
    else:
        output_root_path = Path(output_root).expanduser()

    if window_seconds is None:
        resolved_window = prompt_positive_float("Window seconds")
    else:
        resolved_window = float(window_seconds)
        if resolved_window <= 0:
            raise ValueError("window_seconds must be positive")

    if stride_seconds is None:
        resolved_stride = prompt_positive_float(
            "Stride seconds (press Enter to use window seconds)",
            default=resolved_window,
        )
    else:
        resolved_stride = float(stride_seconds)
        if resolved_stride <= 0:
            raise ValueError("stride_seconds must be positive")

    return input_root_path, output_root_path, resolved_window, resolved_stride


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_global_dataset(
    input_root: Path | str,
    output_root: Path | str,
    window_seconds: float,
    stride_seconds: float | None = None,
) -> None:
    """Build a global activity dataset suitable for LOSO three-class classification.

    Scans ``input_root`` for subject folders whose names are pure digits.  For
    each subject, it reads files matching ``<subject>_e_[123].xlsx`` and slices
    them into sliding windows.  All windows are concatenated into a single
    global array and saved to ``output_root``.

    Output files
    ------------
    X.npy             (N, C, T) float64 EEG windows
    y.npy             (N,)      int64   activity labels 0/1/2
    subject_ids.npy   (N,)      int64   subject integer ID per window
    record_ids.npy    (N,)      int64   integer key into record_id_map
    window_indices.npy(N,)      int64   per-window position inside its record
    metadata.json               dataset provenance
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    if stride_seconds is None:
        stride_seconds = window_seconds

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    subject_dirs = sorted(
        [d for d in input_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    all_windows: list[np.ndarray] = []
    all_y: list[int] = []
    all_subject_ids: list[int] = []
    all_record_ids: list[int] = []
    all_window_indices: list[int] = []

    record_id_map: dict[int, str] = {}
    record_counter = 0
    n_subjects_contributed = 0
    expected_n_channels: int | None = None
    expected_sampling_interval: float | None = None

    for subject_dir in subject_dirs:
        subject_id_str = subject_dir.name
        subject_id_int = int(subject_id_str)

        xlsx_files: list[tuple[Path, int]] = []
        for path in subject_dir.iterdir():
            if path.suffix.lower() != ".xlsx":
                continue
            m = EEG_FILE_PATTERN.match(path.name)
            if m and m.group("subject") == subject_id_str:
                label = int(m.group("label")) - 1  # 1/2/3 -> 0/1/2
                xlsx_files.append((path, label))

        if not xlsx_files:
            print(f"No matching EEG files found for subject {subject_id_str}")
            continue

        xlsx_files.sort(key=lambda t: t[1])

        subject_contributed = False

        for xlsx_path, label in xlsx_files:
            record_name = f"{subject_id_str}_e_{label + 1}"

            loaded = load_xlsx_file(xlsx_path)
            if loaded is None:
                continue

            data, times = loaded
            sampling_interval = estimate_sampling_interval(times)
            n_channels = int(data.shape[1])
            if expected_n_channels is None:
                expected_n_channels = n_channels
            elif n_channels != expected_n_channels:
                raise ValueError(
                    "Inconsistent EEG channel count across files: "
                    f"expected {expected_n_channels}, got {n_channels} in {xlsx_path.name}"
                )
            if expected_sampling_interval is None:
                expected_sampling_interval = sampling_interval
            elif not np.isclose(
                sampling_interval,
                expected_sampling_interval,
                rtol=SAMPLING_INTERVAL_RTOL,
                atol=SAMPLING_INTERVAL_ATOL,
            ):
                raise ValueError(
                    "Inconsistent sampling interval across files: "
                    f"expected {expected_sampling_interval:.9f}s, "
                    f"got {sampling_interval:.9f}s in {xlsx_path.name}"
                )

            try:
                windows = extract_windows(data, times, window_seconds, stride_seconds)
            except ValueError as exc:
                print(f"Skipping {xlsx_path.name}: {exc}")
                continue

            if not windows:
                print(f"No windows extracted from {xlsx_path.name}")
                continue

            record_int_id = record_counter
            record_id_map[record_int_id] = record_name
            record_counter += 1

            for win_idx, window in enumerate(windows):
                all_windows.append(window)
                all_y.append(label)
                all_subject_ids.append(subject_id_int)
                all_record_ids.append(record_int_id)
                all_window_indices.append(win_idx)

            subject_contributed = True

        if subject_contributed:
            n_subjects_contributed += 1

    if not all_windows:
        print("No windows extracted from any file.  Output not written.")
        return

    # (N, T, C) -> (N, C, T)
    X = np.transpose(np.stack(all_windows, axis=0), (0, 2, 1))
    y = np.array(all_y, dtype=np.int64)
    subject_ids = np.array(all_subject_ids, dtype=np.int64)
    record_ids = np.array(all_record_ids, dtype=np.int64)
    window_indices = np.array(all_window_indices, dtype=np.int64)

    output_root.mkdir(parents=True, exist_ok=True)

    np.save(output_root / "X.npy", X)
    np.save(output_root / "y.npy", y)
    np.save(output_root / "subject_ids.npy", subject_ids)
    np.save(output_root / "record_ids.npy", record_ids)
    np.save(output_root / "window_indices.npy", window_indices)

    metadata: dict = {
        "label_map": LABEL_MAP,
        "record_id_map": {str(k): v for k, v in record_id_map.items()},
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "n_subjects": n_subjects_contributed,
        "n_records": len(record_id_map),
        "n_samples": len(all_windows),
    }
    with open(output_root / "metadata.json", "w", encoding="ascii") as fh:
        json.dump(metadata, fh, indent=2)

    print(
        f"Global dataset: {len(all_windows)} windows from "
        f"{n_subjects_contributed} subjects, {len(record_id_map)} records"
    )
    print(f"Output: {output_root}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a global LOSO activity dataset from normalised EEG xlsx files"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=f"Input directory containing subject folders (interactive default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"Output directory for global dataset (interactive default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Window size in seconds (prompts when omitted)",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=None,
        help="Stride size in seconds (prompts when omitted; defaults to window-seconds if blank)",
    )

    args = parser.parse_args()

    input_root, output_root, window_seconds, stride_seconds = resolve_runtime_config(
        input_root=args.input_root,
        output_root=args.output_root,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    build_global_dataset(
        input_root=input_root,
        output_root=output_root,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )

    print()
    print("Done.")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
