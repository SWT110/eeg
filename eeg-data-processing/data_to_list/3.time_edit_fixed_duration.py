import argparse
import re
import zipfile

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
DATASET_ID_RE = re.compile(r"^(\d+):?$")
CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x1f\x7f\ufeff]")


def clean_time_text(value: str) -> str:
    """Remove pasted control characters without changing experiment order."""
    return CONTROL_CHARACTERS_RE.sub("", str(value)).strip()


def parse_time(time_str: str) -> Optional[datetime]:
    """Parse HH:MM:SS(.fff) strings into datetime objects."""
    if not time_str:
        return None
    cleaned = clean_time_text(time_str)
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
    """Do not reuse zero-byte or interrupted Excel archives."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return {"[Content_Types].xml", "xl/workbook.xml", "xl/worksheets/sheet1.xml"}.issubset(
                archive.namelist()
            )
    except (OSError, zipfile.BadZipFile):
        return False


def read_time_starts(file_path: Path) -> Dict[str, List[str]]:
    """Read numeric subject headers and bare starts or legacy start-end lines.

    Keep all configured starts, including later experiments, in their original
    order. Malformed records fail rather than silently shifting class labels.
    """
    starts: Dict[str, List[str]] = {}
    current_id: Optional[str] = None

    with open(file_path, encoding="utf-8") as fp:
        for line_number, raw_line in enumerate(fp, 1):
            line = clean_time_text(raw_line)
            if not line:
                continue
            id_match = DATASET_ID_RE.fullmatch(line)
            if id_match:
                current_id = id_match.group(1)
                if current_id in starts:
                    raise ValueError(f"{file_path}:{line_number}: duplicate subject {current_id}")
                starts[current_id] = []
                continue
            if current_id is None:
                raise ValueError(f"{file_path}:{line_number}: time record has no subject header")
            parts = [clean_time_text(part) for part in line.split("-")]
            if len(parts) not in (1, 2) or any(parse_time(part) is None for part in parts):
                raise ValueError(
                    f"{file_path}:{line_number}: invalid time record for subject {current_id}: {line!r}"
                )
            starts[current_id].append(parts[0])

    return starts


def build_fixed_ranges(dataset_id: str, start_times: Sequence[str]) -> List[TimeRange]:
    """Build fixed-duration time ranges from the configured start times."""
    if len(start_times) < len(FIXED_DURATIONS):
        raise ValueError(
            f"Dataset {dataset_id} has {len(start_times)} start times; "
            f"expected at least {len(FIXED_DURATIONS)} to preserve e_1/e_2/e_3 labels."
        )
    if len(start_times) > len(FIXED_DURATIONS):
        print(
            f"  Warning: dataset {dataset_id} has extra start times; "
            f"only the first {len(FIXED_DURATIONS)} will be used."
        )

    ranges: List[TimeRange] = []
    for start_str, duration in zip(start_times[: len(FIXED_DURATIONS)], FIXED_DURATIONS):
        start_str = clean_time_text(start_str)
        start_time = parse_time(start_str)
        if not start_time:
            raise ValueError(f"Dataset {dataset_id} has invalid start time: {start_str!r}")
        end_time = start_time + duration
        if end_time.date() != start_time.date():
            raise ValueError(f"Dataset {dataset_id}: ranges crossing midnight are not supported")
        if ranges and start_time <= parse_time(ranges[-1][1]):
            raise ValueError(f"Dataset {dataset_id}: fixed-duration experiment ranges overlap")
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
    overwrite: bool = False,
) -> None:
    """Read an Excel file, slice by the provided ranges, and write results."""
    if not excel_path.exists():
        print(f"  Missing file: {excel_path.name}")
        return

    try:
        df = pd.read_excel(excel_path)
    except Exception as exc:
        raise ValueError(f"Failed to open {excel_path}: {exc}") from exc

    time_col = find_time_column(df, preferred_name, preferred_index)
    df_times = [parse_time(value) for value in df[time_col]]
    if not df_times or any(value is None for value in df_times):
        raise ValueError(f"{excel_path}: empty or invalid time column {time_col!r}")
    if any(left > right for left, right in zip(df_times, df_times[1:])):
        raise ValueError(f"{excel_path}: time column must be nondecreasing")
    sampling_step = next(
        (right - left for left, right in zip(df_times, df_times[1:]) if right > left),
        timedelta(0),
    )
    sampling_step = min(sampling_step, timedelta(seconds=1))

    pending_outputs = []
    for idx, (start, end) in enumerate(time_ranges, 1):
        output_path = output_dir / f"{prefix}_{idx}.xlsx"
        if not overwrite and is_complete_output(output_path):
            print(f"    Skipping existing {output_path.name}")
            continue
        if output_path.exists():
            print(f"    Rebuilding file {output_path.name}")

        print(f"    Range #{idx}: {start} -> {end}")
        start_time, end_time = parse_time(start), parse_time(end)
        if not start_time or not end_time:
            raise ValueError(f"{excel_path}: invalid range #{idx} ({start}->{end})")
        if df_times[0] - sampling_step > start_time or df_times[-1] + sampling_step < end_time:
            raise ValueError(f"{excel_path}: does not cover full range #{idx} ({start}->{end})")
        cut_df = cut_data_by_time(df, df_times, start, end)
        if cut_df.empty:
            raise ValueError(f"{excel_path}: no rows for range #{idx} ({start}->{end})")
        pending_outputs.append((output_path, cut_df))

    for output_path, cut_df in pending_outputs:
        temporary_path = output_path.with_name(f".{output_path.stem}.tmp.xlsx")
        try:
            cut_df.to_excel(temporary_path, index=False)
            temporary_path.replace(output_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"    Saved {output_path.name} ({len(cut_df)} rows)")


def dataset_already_processed(
    dataset_id: str,
    ranges: Sequence[TimeRange],
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> bool:
    """Check if all expected cut files already exist for a dataset."""
    input_dir = LIST_DIR if input_dir is None else input_dir
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    dest_dir = output_dir / dataset_id
    if not dest_dir.exists():
        return False

    normal_input = input_dir / f"{dataset_id}.xlsx"
    if normal_input.exists():
        expected_normals = [dest_dir / f"{dataset_id}_{idx}.xlsx" for idx in range(1, len(ranges) + 1)]
        if not all(is_complete_output(path) for path in expected_normals):
            return False

    e_input = input_dir / f"{dataset_id}_e.xlsx"
    if e_input.exists():
        expected_e = [dest_dir / f"{dataset_id}_e_{idx}.xlsx" for idx in range(1, len(ranges) + 1)]
        if not all(is_complete_output(path) for path in expected_e):
            return False

    return normal_input.exists() or e_input.exists()


def positive_subject_id(value: str) -> int:
    try:
        subject_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Subject IDs must be positive integers") from exc
    if subject_id < 1:
        raise argparse.ArgumentTypeError("Subject IDs must be positive integers")
    return subject_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按前三段实验起始时间裁切固定时长 Excel")
    parser.add_argument("--time-data-path", type=Path, default=TIME_DATA_PATH)
    parser.add_argument("--input-dir", type=Path, default=LIST_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--subject-ids", type=positive_subject_id, nargs="+", default=None,
                        help="只处理指定受试者；省略时处理时间表中的全部受试者")
    parser.add_argument("--overwrite", action="store_true",
                        help="重新裁切已有输出；默认保留已有非空 Excel")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    time_data_path = args.time_data_path.expanduser()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not time_data_path.exists():
        raise FileNotFoundError(f"Missing time configuration: {time_data_path}")

    time_starts = read_time_starts(time_data_path)
    if not time_starts:
        print("time_data.txt does not contain any valid start times.")
        return

    selected_ids = None if args.subject_ids is None else set(args.subject_ids)
    if selected_ids is not None:
        missing = selected_ids - {int(dataset_id) for dataset_id in time_starts}
        if missing:
            raise ValueError(f"Requested subjects missing from time configuration: {sorted(missing)}")
        time_starts = {
            dataset_id: starts for dataset_id, starts in time_starts.items()
            if int(dataset_id) in selected_ids
        }
        for dataset_id in time_starts:
            if not (input_dir / f"{dataset_id}_e.xlsx").exists():
                raise FileNotFoundError(f"Missing EEG input for requested subject {dataset_id}: {input_dir}")

    # Validate every selected subject before creating any output. This prevents
    # a malformed later record from leaving a partly updated labelled dataset.
    selected_ranges = {
        dataset_id: build_fixed_ranges(dataset_id, starts)
        for dataset_id, starts in sorted(time_starts.items(), key=lambda item: int(item[0]))
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id, ranges in selected_ranges.items():

        if not args.overwrite and dataset_already_processed(dataset_id, ranges, input_dir, output_dir):
            print(f"Skipping dataset {dataset_id}: already cut.")
            continue

        print(f"Processing dataset {dataset_id}")
        dest_dir = output_dir / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        normal_path = input_dir / f"{dataset_id}.xlsx"
        slice_excel_file(normal_path, dest_dir, dataset_id, ranges, preferred_name="Time",
                         overwrite=args.overwrite)

        e_path = input_dir / f"{dataset_id}_e.xlsx"
        slice_excel_file(e_path, dest_dir, f"{dataset_id}_e", ranges, preferred_index=25,
                         overwrite=args.overwrite)

    print("Fixed-duration cutting complete.")


if __name__ == "__main__":
    main()
