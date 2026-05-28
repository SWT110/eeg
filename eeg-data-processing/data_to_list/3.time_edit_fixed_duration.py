import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
EEG_ROOT = BASE_DIR.parents[1]
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import DATA_TO_LIST_DATA_DIR, LIST_CUT_DIR, LIST_DIR as DEFAULT_LIST_DIR

TIME_DATA_PATH = DATA_TO_LIST_DATA_DIR / "time_data.txt"
LIST_DIR = DEFAULT_LIST_DIR
OUTPUT_DIR = LIST_CUT_DIR

TimeRange = Tuple[str, str]
FIXED_DURATIONS: Tuple[timedelta, ...] = (
    timedelta(minutes=5),
    timedelta(minutes=5, seconds=18),
    timedelta(minutes=16, seconds=38),
)


def parse_time(time_str: str) -> Optional[datetime]:
    """Parse HH:MM:SS(.fff) strings into datetime objects."""
    if not time_str:
        return None
    cleaned = str(time_str).strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def format_time(time_obj: datetime) -> str:
    """Format a datetime object back to HH:MM:SS."""
    return time_obj.strftime("%H:%M:%S")


def is_complete_output(path: Path) -> bool:
    """Treat zero-byte files as incomplete so reruns can rebuild them."""
    return path.exists() and path.stat().st_size > 0


def read_time_starts(file_path: Path) -> Dict[str, List[str]]:
    """Return a dict {id: [start_time, ...]} parsed from time_data.txt."""
    starts: Dict[str, List[str]] = {}
    current_id: Optional[str] = None

    with open(file_path, encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            if line.isdigit():
                current_id = line
                starts[current_id] = []
                continue
            if current_id is None or "-" not in line:
                continue
            start, _ = line.split("-", 1)
            starts[current_id].append(start.strip())

    return {key: value for key, value in starts.items() if value}


def build_fixed_ranges(dataset_id: str, start_times: Sequence[str]) -> List[TimeRange]:
    """Build fixed-duration time ranges from the configured start times."""
    if len(start_times) < len(FIXED_DURATIONS):
        print(
            f"  Warning: dataset {dataset_id} has {len(start_times)} start times, "
            f"expected {len(FIXED_DURATIONS)}."
        )
    if len(start_times) > len(FIXED_DURATIONS):
        print(
            f"  Warning: dataset {dataset_id} has extra start times; "
            f"only the first {len(FIXED_DURATIONS)} will be used."
        )

    ranges: List[TimeRange] = []
    for start_str, duration in zip(start_times[: len(FIXED_DURATIONS)], FIXED_DURATIONS):
        start_time = parse_time(start_str)
        if not start_time:
            print(f"  Warning: dataset {dataset_id} has invalid start time: {start_str}")
            continue
        end_time = start_time + duration
        ranges.append((start_str, format_time(end_time)))

    return ranges


def find_time_column(df: pd.DataFrame, preferred_name: Optional[str] = None, preferred_index: Optional[int] = None):
    """Try name first, then fallback to column index for the time column."""
    if preferred_name and preferred_name in df.columns:
        return preferred_name
    for column in df.columns:
        if isinstance(column, str) and column.strip().lower() == "time":
            return column
    if preferred_index is not None and 0 <= preferred_index < len(df.columns):
        return df.columns[preferred_index]
    return df.columns[0]


def cut_data_by_time(
    df: pd.DataFrame,
    df_times: Sequence[Optional[datetime]],
    start_time_str: str,
    end_time_str: str,
) -> pd.DataFrame:
    """Slice rows whose time column is within [start, end], inclusive."""
    start_time = parse_time(start_time_str)
    end_time = parse_time(end_time_str)
    if not start_time or not end_time:
        return pd.DataFrame()

    start_idx = None
    for idx, current in enumerate(df_times):
        if current and current >= start_time:
            start_idx = idx
            break

    end_idx = None
    for idx in range(len(df_times) - 1, -1, -1):
        current = df_times[idx]
        if current and current <= end_time:
            end_idx = idx
            break

    if start_idx is not None and end_idx is not None and start_idx <= end_idx:
        return df.iloc[start_idx : end_idx + 1]
    return pd.DataFrame()


def slice_excel_file(
    excel_path: Path,
    output_dir: Path,
    prefix: str,
    time_ranges: Sequence[TimeRange],
    preferred_name: Optional[str] = None,
    preferred_index: Optional[int] = None,
) -> None:
    """Read an Excel file, slice by the provided ranges, and write results."""
    if not excel_path.exists():
        print(f"  Missing file: {excel_path.name}")
        return

    try:
        df = pd.read_excel(excel_path)
    except Exception as exc:
        print(f"  Failed to open {excel_path.name}: {exc}")
        return

    time_col = find_time_column(df, preferred_name, preferred_index)
    df_times = [parse_time(value) for value in df[time_col]]

    for idx, (start, end) in enumerate(time_ranges, 1):
        output_path = output_dir / f"{prefix}_{idx}.xlsx"
        if is_complete_output(output_path):
            print(f"    Skipping existing {output_path.name}")
            continue
        if output_path.exists():
            print(f"    Rebuilding incomplete file {output_path.name}")

        print(f"    Range #{idx}: {start} -> {end}")
        cut_df = cut_data_by_time(df, df_times, start, end)
        if cut_df.empty:
            print(f"    {excel_path.name}: no rows for range #{idx} ({start}->{end})")
            continue
        cut_df.to_excel(output_path, index=False)
        print(f"    Saved {output_path.name} ({len(cut_df)} rows)")


def dataset_already_processed(dataset_id: str, ranges: Sequence[TimeRange]) -> bool:
    """Check if all expected cut files already exist for a dataset."""
    dest_dir = OUTPUT_DIR / dataset_id
    if not dest_dir.exists():
        return False

    expected_normals = [dest_dir / f"{dataset_id}_{idx}.xlsx" for idx in range(1, len(ranges) + 1)]
    if not all(is_complete_output(path) for path in expected_normals):
        return False

    e_input = LIST_DIR / f"{dataset_id}_e.xlsx"
    if e_input.exists():
        expected_e = [dest_dir / f"{dataset_id}_e_{idx}.xlsx" for idx in range(1, len(ranges) + 1)]
        if not all(is_complete_output(path) for path in expected_e):
            return False

    return True


def main():
    if not TIME_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing time configuration: {TIME_DATA_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    time_starts = read_time_starts(TIME_DATA_PATH)
    if not time_starts:
        print("time_data.txt does not contain any valid start times.")
        return

    for dataset_id, start_times in sorted(time_starts.items(), key=lambda item: int(item[0])):
        ranges = build_fixed_ranges(dataset_id, start_times)
        if not ranges:
            print(f"Skipping dataset {dataset_id}: no valid fixed-duration ranges.")
            continue

        if dataset_already_processed(dataset_id, ranges):
            print(f"Skipping dataset {dataset_id}: already cut.")
            continue

        print(f"Processing dataset {dataset_id}")
        dest_dir = OUTPUT_DIR / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        normal_path = LIST_DIR / f"{dataset_id}.xlsx"
        slice_excel_file(normal_path, dest_dir, dataset_id, ranges, preferred_name="Time")

        e_path = LIST_DIR / f"{dataset_id}_e.xlsx"
        slice_excel_file(e_path, dest_dir, f"{dataset_id}_e", ranges, preferred_index=25)

    print("Fixed-duration cutting complete.")


if __name__ == "__main__":
    main()
