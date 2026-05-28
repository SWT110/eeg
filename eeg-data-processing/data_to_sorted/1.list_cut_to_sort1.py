from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EEG_ROOT = BASE_DIR.parents[1]
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import LIST_CUT_DIR, LIST_NORMALIZED_DIR, SORT2_DATA_CUT_ROOT


def format_seconds_label(segment_seconds: float) -> str:
    return f"{segment_seconds:g}"


def infer_signal_type(path: Path) -> str | None:
    stem = path.stem
    if "_e_" in stem:
        return "eeg"
    if "_" in stem:
        return "ppg"
    return None


def parse_relative_seconds(series: pd.Series) -> pd.Series:
    parsed = pd.to_timedelta(series.astype(str), errors="coerce")
    if parsed.isna().all():
        parsed_datetime = pd.to_datetime(series, errors="coerce")
        parsed = parsed_datetime - parsed_datetime.min()
    return parsed.dt.total_seconds()


def iter_segments(dataframe: pd.DataFrame, segment_seconds: float) -> Iterable[tuple[int, pd.DataFrame]]:
    if "Time" not in dataframe.columns:
        return

    seconds = parse_relative_seconds(dataframe["Time"])
    valid = dataframe.loc[seconds.notna()].copy()
    seconds = seconds.loc[seconds.notna()]
    if valid.empty:
        return

    start = float(seconds.min())
    end = float(seconds.max())
    segment_index = 1
    cursor = start
    while cursor + segment_seconds <= end:
        next_cursor = cursor + segment_seconds
        mask = (seconds >= cursor) & (seconds < next_cursor)
        segment = valid.loc[mask]
        if not segment.empty:
            yield segment_index, segment
        segment_index += 1
        cursor = next_cursor


def process_source(input_root: Path, output_root: Path, signals: tuple[str, ...], segment_seconds: float) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    for source_path in sorted(input_root.rglob("*.xlsx")):
        signal_type = infer_signal_type(source_path)
        if signal_type not in signals:
            continue

        try:
            subject_id = source_path.parent.relative_to(input_root).parts[0]
        except (IndexError, ValueError):
            subject_id = source_path.stem.split("_", 1)[0]

        dataframe = pd.read_excel(source_path)
        output_dir = output_root / subject_id / signal_type
        output_dir.mkdir(parents=True, exist_ok=True)

        for segment_index, segment in iter_segments(dataframe, segment_seconds):
            output_path = output_dir / f"{source_path.stem}_seg{segment_index:03d}.xlsx"
            segment.to_excel(output_path, index=False)


def process_all_sources(
    segment_seconds: float,
    output_base: Path | str = SORT2_DATA_CUT_ROOT,
    sources: list[dict] | None = None,
) -> None:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")

    output_base = Path(output_base).expanduser()
    if sources is None:
        sources = [
            {"input_root": LIST_CUT_DIR, "output_suffix": "", "signals": ("ppg", "eeg")},
            {"input_root": LIST_NORMALIZED_DIR, "output_suffix": "_normalization", "signals": ("eeg",)},
        ]

    seconds_label = format_seconds_label(float(segment_seconds))
    for source in sources:
        input_root = Path(source["input_root"]).expanduser()
        output_suffix = str(source.get("output_suffix", ""))
        signals = tuple(source.get("signals", ("ppg", "eeg")))
        output_root = output_base / f"data_cut_{seconds_label}s{output_suffix}"
        process_source(input_root, output_root, signals, float(segment_seconds))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按固定秒数切分 list_cut 数据")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--output-base", type=Path, default=SORT2_DATA_CUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    process_all_sources(segment_seconds=args.segment_seconds, output_base=args.output_base)


if __name__ == "__main__":
    main()
