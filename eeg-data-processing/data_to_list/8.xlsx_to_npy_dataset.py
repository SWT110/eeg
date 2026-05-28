from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EEG_ROOT = BASE_DIR.parents[1]
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import LIST_NORMALIZED_DIR, NPY_DATASET_DIR

DEFAULT_INPUT_ROOT = LIST_NORMALIZED_DIR
DEFAULT_OUTPUT_ROOT = NPY_DATASET_DIR
EEG_FILE_PATTERN = re.compile(r"^(?P<subject>\d+)_e_(?P<label>[123])\.xlsx$")
TIME_COLUMN = "Time"
COMMON_EEG_CHANNELS = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "M1",
    "M2",
    "Fz",
    "Cz",
    "Pz",
)


def estimate_sampling_interval(times: np.ndarray) -> float:
    """Estimate sampling interval from time array.
    
    Raises:
        ValueError: If less than 2 time points or any interval <= 0
    """
    if len(times) < 2:
        raise ValueError("Need at least 2 time points to estimate sampling interval")
    intervals = np.diff(times)
    
    # Check if ANY interval is <= 0 (catches partial duplicates/decreasing times)
    if np.any(intervals <= 0):
        median_interval = float(np.median(intervals))
        raise ValueError(
            f"Invalid time sequence: found intervals <= 0 (median={median_interval:.6f}). "
            "Check for duplicate or decreasing timestamps."
        )

    # Millisecond-rounded timestamps from higher-rate recordings (for example 128 Hz)
    # alternate between 7 ms and 8 ms steps. Using the full-span average interval
    # preserves the true rate, while the median incorrectly snaps to 8 ms.
    return float((times[-1] - times[0]) / (len(times) - 1))


def extract_windows(
    data: np.ndarray,
    times: np.ndarray,
    window_seconds: float,
    stride_seconds: float,
) -> list[np.ndarray]:
    """Extract sliding windows from data array.
    
    Args:
        data: Array of shape (n_samples, n_channels)
        times: Time array of shape (n_samples,)
        window_seconds: Window size in seconds
        stride_seconds: Stride size in seconds
        
    Returns:
        List of window arrays, each of shape (window_samples, n_channels)
        
    Raises:
        ValueError: If window or stride is too small relative to sampling interval
    """
    if len(data) == 0:
        return []
    
    sampling_interval = estimate_sampling_interval(times)
    window_samples = int(np.round(window_seconds / sampling_interval))
    stride_samples = int(np.round(stride_seconds / sampling_interval))
    
    if window_samples < 1:
        raise ValueError(
            f"window_seconds ({window_seconds}s) is too small relative to sampling "
            f"interval ({sampling_interval}s): results in {window_samples} samples "
            "(need at least 1)"
        )
    
    if stride_samples < 1:
        raise ValueError(
            f"stride_seconds ({stride_seconds}s) is too small relative to sampling "
            f"interval ({sampling_interval}s): results in {stride_samples} samples "
            "(need at least 1)"
        )
    
    windows = []
    start_idx = 0
    
    while start_idx + window_samples <= len(data):
        window = data[start_idx:start_idx + window_samples]
        windows.append(window)
        start_idx += stride_samples
    
    return windows


def normalize_eeg_channel_name(column_name: str) -> str | None:
    if not isinstance(column_name, str) or not column_name.startswith("EEG"):
        return None
    raw_name = column_name[len("EEG"):].lstrip(" _")
    return raw_name.split("-", maxsplit=1)[0].strip()


def resolve_eeg_columns(df: pd.DataFrame, path: Path) -> list[str] | None:
    all_eeg_columns = [
        col for col in df.columns if isinstance(col, str) and col.startswith("EEG")
    ]
    if not all_eeg_columns:
        print(f"Skipping {path.name}: no columns starting with 'EEG'")
        return None

    matched_common_columns: dict[str, str] = {}
    duplicate_common_channels: list[str] = []
    for column in all_eeg_columns:
        channel_name = normalize_eeg_channel_name(column)
        if channel_name not in COMMON_EEG_CHANNELS:
            continue
        if channel_name in matched_common_columns:
            duplicate_common_channels.append(channel_name)
            continue
        matched_common_columns[channel_name] = column

    if matched_common_columns:
        if duplicate_common_channels:
            duplicate_list = ", ".join(sorted(set(duplicate_common_channels)))
            print(
                f"Skipping {path.name}: duplicate common EEG channels detected: "
                f"{duplicate_list}"
            )
            return None
        missing_common_channels = [
            channel for channel in COMMON_EEG_CHANNELS if channel not in matched_common_columns
        ]
        if missing_common_channels:
            missing_list = ", ".join(missing_common_channels)
            print(
                f"Skipping {path.name}: missing common EEG channels: {missing_list}"
            )
            return None
        return [matched_common_columns[channel] for channel in COMMON_EEG_CHANNELS]

    return all_eeg_columns


def load_xlsx_file(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load xlsx file and extract EEG data and times.
    
    Returns:
        Tuple of (data, times) where data is shape (n_samples, n_channels),
        or None if file cannot be loaded.
    """
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        print(f"Failed to read {path.name}: {exc}")
        return None
    
    if len(df) < 2:
        print(f"Skipping {path.name}: need at least 2 data rows, got {len(df)}")
        return None
    
    if TIME_COLUMN not in df.columns:
        print(f"Skipping {path.name}: missing '{TIME_COLUMN}' column")
        return None
    
    eeg_columns = resolve_eeg_columns(df, path)
    if eeg_columns is None:
        return None
    
    # Parse times - handle both numeric (seconds) and string (timedelta) formats
    time_col = df[TIME_COLUMN]
    
    # Check if Time column is numeric (int or float)
    if pd.api.types.is_numeric_dtype(time_col):
        # Numeric column: treat as seconds directly
        if time_col.isna().any():
            print(f"Skipping {path.name}: '{TIME_COLUMN}' column contains NaN values")
            return None
        times_array = time_col.to_numpy(dtype=float)
    else:
        # Non-numeric column: parse as timedelta strings
        times = pd.to_timedelta(time_col.astype(str), errors="coerce").dt.total_seconds()
        if times.isna().any():
            print(f"Skipping {path.name}: unable to parse all values in '{TIME_COLUMN}'")
            return None
        times_array = times.to_numpy(dtype=float)
    data_array = df[eeg_columns].to_numpy(dtype=float)
    
    # Check for NaN, Inf, -Inf in EEG data
    if not np.isfinite(data_array).all():
        print(f"Skipping {path.name}: EEG data contains NaN, Inf, or -Inf values")
        return None
    
    return data_array, times_array


def prompt_path(prompt_text: str, default: Path, must_exist: bool = False) -> Path:
    while True:
        raw = input(f"{prompt_text} [{default}]: ").strip()
        candidate = Path(raw).expanduser() if raw else default
        if must_exist and not candidate.exists():
            print(f"Path does not exist: {candidate}")
            continue
        return candidate


def prompt_positive_float(prompt_text: str, default: float | None = None) -> float:
    default_text = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt_text}{default_text}: ").strip()
        if not raw and default is not None:
            return float(default)

        try:
            value = float(raw)
        except ValueError:
            print("Please enter a positive number.")
            continue

        if value <= 0:
            print("Please enter a positive number.")
            continue
        return value


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
        resolved_window_seconds = prompt_positive_float("Window seconds")
    else:
        resolved_window_seconds = float(window_seconds)
        if resolved_window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

    if stride_seconds is None:
        resolved_stride_seconds = prompt_positive_float(
            "Stride seconds (press Enter to use window seconds)",
            default=resolved_window_seconds,
        )
    else:
        resolved_stride_seconds = float(stride_seconds)
        if resolved_stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive")

    return (
        input_root_path,
        output_root_path,
        resolved_window_seconds,
        resolved_stride_seconds,
    )


def convert_xlsx_to_npy(
    input_root: Path | str,
    output_root: Path | str,
    window_seconds: float,
    stride_seconds: float | None = None,
) -> None:
    """Convert normalized EEG xlsx files to subject-level npy arrays.
    
    Args:
        input_root: Directory containing subject folders with xlsx files
        output_root: Directory to write output npy files
        window_seconds: Window size in seconds
        stride_seconds: Stride size in seconds (defaults to window_seconds if None)
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    if stride_seconds is None:
        stride_seconds = window_seconds
    
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    
    # Process each subject directory
    subject_dirs = sorted(
        [d for d in input_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        
        # Find all EEG xlsx files for this subject
        xlsx_files = []
        for path in subject_dir.iterdir():
            if path.suffix.lower() == ".xlsx":
                match = EEG_FILE_PATTERN.match(path.name)
                if match and match.group("subject") == subject_id:
                    label = int(match.group("label"))
                    xlsx_files.append((path, label - 1))  # Convert label 1/2/3 to 0/1/2
        
        if not xlsx_files:
            print(f"No matching EEG files found for subject {subject_id}")
            continue
        
        # Sort by label to ensure consistent ordering
        xlsx_files.sort(key=lambda x: x[1])
        
        # Process all files and collect windows
        all_windows = []
        all_labels = []
        all_groups = []
        
        for group_idx, (xlsx_path, label) in enumerate(xlsx_files):
            loaded = load_xlsx_file(xlsx_path)
            if loaded is None:
                continue
            
            data, times = loaded
            
            # Extract windows - catch errors from invalid time sequences
            try:
                windows = extract_windows(data, times, window_seconds, stride_seconds)
            except ValueError as e:
                print(f"Skipping {xlsx_path.name}: {e}")
                continue
            
            for window in windows:
                all_windows.append(window)
                all_labels.append(label)
                all_groups.append(group_idx)
        
        if not all_windows:
            print(f"No windows extracted for subject {subject_id}")
            continue
        
        # Stack all windows into arrays and transpose to (N, C, T) format
        # windows are (window_samples, n_channels), stack gives (N, T, C)
        X_stacked = np.stack(all_windows, axis=0)  # Shape: (n_windows, window_samples, n_channels)
        X = np.transpose(X_stacked, (0, 2, 1))  # Shape: (n_windows, n_channels, window_samples) = (N, C, T)
        y = np.array(all_labels, dtype=np.int64)
        groups = np.array(all_groups, dtype=np.int64)
        
        # Write output to subject_<id> directory
        subject_out = output_root / f"subject_{subject_id}"
        subject_out.mkdir(parents=True, exist_ok=True)
        
        np.save(subject_out / f"subject_{subject_id}_X.npy", X)
        np.save(subject_out / f"subject_{subject_id}_y.npy", y)
        np.save(subject_out / f"subject_{subject_id}_groups.npy", groups)
        
        print(f"Subject {subject_id}: exported {len(all_windows)} windows from {len(xlsx_files)} files")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert normalized EEG xlsx files to subject-level npy arrays"
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
        help=f"Output directory for npy files (interactive default: {DEFAULT_OUTPUT_ROOT})",
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
    
    convert_xlsx_to_npy(
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
