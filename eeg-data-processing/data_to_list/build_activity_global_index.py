from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Sequence

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


def parse_subject_ids(raw: str | Sequence[int] | None) -> list[int] | None:
    """Resolve a subject selection without allowing typos or empty items."""
    if raw is None:
        return None
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    if not values:
        raise ValueError("Subject selection must not be empty")
    resolved = []
    for value in values:
        item = str(value).strip()
        if not item.isdigit() or int(item) < 1:
            raise ValueError(f"Subject IDs must be positive integers: {value!r}")
        resolved.append(int(item))
    return sorted(set(resolved))


def excluded_output_base(output_base: Path | str, exclude_subject_ids: Sequence[int] | None) -> Path:
    """Give cohorts with different exclusion policies different directories."""
    path = Path(output_base).expanduser()
    excluded = parse_subject_ids(exclude_subject_ids) if exclude_subject_ids else []
    if not excluded:
        return path
    suffix = "_exclude_" + "_".join(f"S{sid}" for sid in excluded)
    if path.name.endswith(suffix):
        return path
    if "_exclude_" in path.name:
        raise ValueError(f"Output name {path.name!r} conflicts with exclusion policy {suffix}")
    return path.with_name(path.name + suffix)


def resolve_dataset_output_root(output_root: Path | str, exclude_subject_ids: Sequence[int] | None) -> Path:
    """Keep window directory names compatible with existing batch discovery."""
    path = Path(output_root).expanduser()
    if re.fullmatch(r"window_[0-9.]+_stride_[0-9.]+", path.name):
        return excluded_output_base(path.parent, exclude_subject_ids) / path.name
    return excluded_output_base(path, exclude_subject_ids)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    *,
    include_subject_ids: str | Sequence[int] | None = None,
    exclude_subject_ids: str | Sequence[int] | None = None,
    require_complete_records: bool = False,
    overwrite: bool = False,
) -> Path | None:
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
    included = parse_subject_ids(include_subject_ids)
    excluded = parse_subject_ids(exclude_subject_ids) or []
    output_root = resolve_dataset_output_root(output_root, excluded)

    if stride_seconds is None:
        stride_seconds = window_seconds

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if (not np.isfinite(window_seconds) or window_seconds <= 0
            or not np.isfinite(stride_seconds) or stride_seconds <= 0):
        raise ValueError("Window and stride must be finite positive seconds")
    if not overwrite and output_root.exists() and any(output_root.glob("*.npy")):
        raise FileExistsError(f"Dataset already exists: {output_root}; use a new path or --overwrite")

    subject_dirs = sorted(
        [d for d in input_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    discovered = sorted({int(directory.name) for directory in subject_dirs})
    unknown = sorted(set((included or []) + excluded) - set(discovered))
    if unknown:
        raise ValueError(f"Subject IDs absent from input: {unknown}; available={discovered}")
    selected = set(discovered if included is None else included) - set(excluded)
    if not selected:
        raise ValueError("Subject selection leaves no training cohort")
    subject_dirs = [directory for directory in subject_dirs if int(directory.name) in selected]
    print(f"Included subjects: {sorted(selected)}; excluded subjects: {excluded}")

    all_windows: list[np.ndarray] = []
    all_y: list[int] = []
    all_subject_ids: list[int] = []
    all_record_ids: list[int] = []
    all_window_indices: list[int] = []

    record_id_map: dict[int, str] = {}
    record_sources: dict[str, dict] = {}
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
            if require_complete_records:
                raise ValueError(f"No EEG records for selected subject {subject_id_str}")
            print(f"No matching EEG files found for subject {subject_id_str}")
            continue
        if require_complete_records and {label for _, label in xlsx_files} != {0, 1, 2}:
            raise ValueError(f"Selected subject {subject_id_str} must have e_1, e_2, and e_3")

        xlsx_files.sort(key=lambda t: t[1])

        subject_contributed = False

        for xlsx_path, label in xlsx_files:
            record_name = f"{subject_id_str}_e_{label + 1}"

            loaded = load_xlsx_file(xlsx_path)
            if loaded is None:
                if require_complete_records:
                    raise ValueError(f"Invalid EEG record: {xlsx_path}")
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
                if require_complete_records:
                    raise ValueError(f"Invalid windows in {xlsx_path}: {exc}") from exc
                print(f"Skipping {xlsx_path.name}: {exc}")
                continue

            if not windows:
                if require_complete_records:
                    raise ValueError(f"No complete windows in {xlsx_path}")
                print(f"No windows extracted from {xlsx_path.name}")
                continue

            record_int_id = record_counter
            record_id_map[record_int_id] = record_name
            record_sources[record_name] = {
                "relative_path": str(xlsx_path.relative_to(input_root)),
                "sha256": sha256_file(xlsx_path),
                "n_rows": int(len(data)),
                "n_windows": len(windows),
                "first_timestamp_seconds": float(times[0]),
                "last_timestamp_seconds": float(times[-1]),
            }
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
    # metadata is the completion marker used by generate_datasets.py.
    if (output_root / "metadata.json").exists():
        (output_root / "metadata.json").unlink()

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
        "input_root": str(input_root.resolve()),
        "discovered_subject_ids": discovered,
        "requested_subject_ids": discovered if included is None else included,
        "included_subject_ids": np.unique(subject_ids).astype(int).tolist(),
        "excluded_subject_ids": excluded,
        "excluded_stage": "global_window_dataset",
        "record_sources": record_sources,
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "sampling_rate_hz": 1.0 / expected_sampling_interval,
        "require_complete_records": require_complete_records,
    }
    with open(output_root / "metadata.json", "w", encoding="ascii") as fh:
        json.dump(metadata, fh, indent=2)

    print(
        f"Global dataset: {len(all_windows)} windows from "
        f"{n_subjects_contributed} subjects, {len(record_id_map)} records"
    )
    print(f"Output: {output_root}")
    return output_root


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
    parser.add_argument("--include-subject-ids", help="Only include these IDs, e.g. 1,2,3,12,17")
    parser.add_argument("--exclude-subject-ids", help="Exclude whole subjects, e.g. 6 or 6,8; tags output folder")
    parser.add_argument("--require-complete-records", action="store_true",
                        help="Fail if any selected subject lacks a valid e_1/e_2/e_3 record")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rebuild an existing derived dataset")

    args = parser.parse_args()

    input_root, output_root, window_seconds, stride_seconds = resolve_runtime_config(
        input_root=args.input_root,
        output_root=args.output_root,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    actual_output_root = build_global_dataset(
        input_root=input_root,
        output_root=output_root,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        include_subject_ids=args.include_subject_ids,
        exclude_subject_ids=args.exclude_subject_ids,
        require_complete_records=args.require_complete_records,
        overwrite=args.overwrite,
    )

    print()
    print("Done.")
    print(f"Output directory: {actual_output_root}")


if __name__ == "__main__":
    main()
