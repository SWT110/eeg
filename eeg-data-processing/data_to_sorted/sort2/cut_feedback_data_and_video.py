from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
TIME_DATA_DIR = BASE_DIR / "time_data"
INPUT_ROOT = PROJECT_ROOT / "data_to_list" / "list_cut_fixed_duration"
VIDEOS_ROOT = PROJECT_ROOT / "videos"
OUTPUT_ROOT = BASE_DIR / "data_cut"

TIME_COLUMN = "Time"
EXPERIMENT_IDS = ("1", "2", "3")
NANOSECONDS_PER_SECOND = Decimal("1000000000")
BLOCK_HEADER_PATTERN = re.compile(r"^(?P<experiment>\d+)-(?P<subject>\d+)\s*$")
RANGE_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclass(frozen=True)
class TimeRange:
    subject_id: str
    experiment_id: str
    range_index: int
    start_ns: int
    end_ns: int
    start_text: str
    end_text: str


@dataclass(frozen=True)
class LoadedTable:
    dataframe: pd.DataFrame
    relative_time_ns: np.ndarray


@dataclass
class ProcessStats:
    created: int = 0
    skipped: int = 0
    warnings: int = 0

    def merge(self, other: "ProcessStats") -> None:
        self.created += other.created
        self.skipped += other.skipped
        self.warnings += other.warnings


def warn(message: str) -> int:
    print(f"Warning: {message}")
    return 1


def read_text_with_fallbacks(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def parse_feedback_timestamp_to_ns(timestamp: str) -> int:
    hours, minutes, seconds = timestamp.split(":")
    second_part, milliseconds = seconds.split(",")
    total_seconds = (
        Decimal(hours) * Decimal(3600)
        + Decimal(minutes) * Decimal(60)
        + Decimal(second_part)
        + (Decimal(milliseconds) / Decimal(1000))
    )
    return int((total_seconds * NANOSECONDS_PER_SECOND).to_integral_value())


def format_feedback_timestamp_for_ffmpeg(timestamp: str) -> str:
    return timestamp.replace(",", ".")


def iter_subject_ids(input_root: Path) -> list[str]:
    if not input_root.exists():
        return []
    return [
        path.name
        for path in sorted(
            (candidate for candidate in input_root.iterdir() if candidate.is_dir() and candidate.name.isdigit()),
            key=lambda candidate: int(candidate.name),
        )
    ]


def ensure_output_directories(output_root: Path) -> None:
    for experiment_id in EXPERIMENT_IDS:
        (output_root / experiment_id).mkdir(parents=True, exist_ok=True)


def is_complete_output(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def build_temp_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def resolve_ffmpeg_path() -> str:
    env_path = os.environ.get("FFMPEG_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"FFMPEG_PATH 指向的文件不存在: {candidate}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    raise FileNotFoundError("未找到 ffmpeg。请先设置 FFMPEG_PATH 或将 ffmpeg 加入 PATH。")


def prompt_video_cutting_enabled() -> bool:
    prompt = "是否切割视频？[Y/n]: "
    enabled_values = {"y", "yes", "1"}
    disabled_values = {"n", "no", "0"}

    while True:
        raw = input(prompt).strip().lower()
        if raw == "":
            return True
        if raw in enabled_values:
            return True
        if raw in disabled_values:
            return False
        print("输入无效，请输入 y/yes/1 或 n/no/0，直接回车默认为是。")


def parse_feedback_file(path: Path, expected_experiment_id: str) -> tuple[dict[str, list[TimeRange]], int]:
    feedback_ranges: dict[str, list[TimeRange]] = {}
    current_subject_id: str | None = None
    warnings = 0

    for line_number, raw_line in enumerate(read_text_with_fallbacks(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        header_match = BLOCK_HEADER_PATTERN.match(line)
        if header_match:
            current_subject_id = header_match.group("subject")
            header_experiment_id = header_match.group("experiment")
            if header_experiment_id != expected_experiment_id:
                warnings += warn(
                    f"{path.name}:{line_number} 的分组头 {line} 与目标实验 {expected_experiment_id} 不一致。"
                )
            feedback_ranges.setdefault(current_subject_id, [])
            continue

        range_match = RANGE_PATTERN.search(line)
        if range_match is None:
            continue

        if current_subject_id is None:
            warnings += warn(f"{path.name}:{line_number} 出现了时间区间，但前面没有受试者分组头。")
            continue

        start_text = range_match.group("start")
        end_text = range_match.group("end")
        start_ns = parse_feedback_timestamp_to_ns(start_text)
        end_ns = parse_feedback_timestamp_to_ns(end_text)

        if end_ns <= start_ns:
            warnings += warn(
                f"{path.name}:{line_number} 的时间区间无效: {start_text} - {end_text}。"
            )
            continue

        subject_ranges = feedback_ranges.setdefault(current_subject_id, [])
        subject_ranges.append(
            TimeRange(
                subject_id=current_subject_id,
                experiment_id=expected_experiment_id,
                range_index=len(subject_ranges) + 1,
                start_ns=start_ns,
                end_ns=end_ns,
                start_text=start_text,
                end_text=end_text,
            )
        )

    return feedback_ranges, warnings


def load_feedback_ranges(time_data_dir: Path = TIME_DATA_DIR) -> tuple[dict[str, dict[str, list[TimeRange]]], int]:
    all_ranges: dict[str, dict[str, list[TimeRange]]] = {}
    warnings = 0

    for experiment_id in EXPERIMENT_IDS:
        feedback_path = time_data_dir / f"反馈时间{experiment_id}.txt"
        if not feedback_path.exists():
            warnings += warn(f"缺少反馈时间文件: {feedback_path}")
            all_ranges[experiment_id] = {}
            continue

        experiment_ranges, experiment_warnings = parse_feedback_file(feedback_path, experiment_id)
        all_ranges[experiment_id] = experiment_ranges
        warnings += experiment_warnings

    return all_ranges, warnings


def load_table(path: Path) -> LoadedTable | None:
    try:
        dataframe = pd.read_excel(path)
    except Exception as exc:
        warn(f"读取 Excel 失败 {path}: {exc}")
        return None

    if TIME_COLUMN not in dataframe.columns:
        warn(f"{path} 缺少 '{TIME_COLUMN}' 列。")
        return None

    timedelta_series = pd.to_timedelta(dataframe[TIME_COLUMN].astype(str), errors="coerce")
    if timedelta_series.isna().any():
        warn(f"{path} 的 '{TIME_COLUMN}' 列存在无法解析的时间。")
        return None

    time_ns = timedelta_series.array.asi8
    if len(time_ns) == 0:
        warn(f"{path} 没有数据行。")
        return None

    relative_time_ns = time_ns - int(time_ns[0])
    if not np.all(relative_time_ns[:-1] <= relative_time_ns[1:]):
        warn(f"{path} 的 '{TIME_COLUMN}' 列不是单调递增。")
        return None

    return LoadedTable(dataframe=dataframe, relative_time_ns=relative_time_ns)


def align_range_to_samples(relative_time_ns: np.ndarray, time_range: TimeRange) -> tuple[int, int] | None:
    start_index = int(np.searchsorted(relative_time_ns, time_range.start_ns, side="left"))
    end_index = int(np.searchsorted(relative_time_ns, time_range.end_ns, side="right")) - 1

    if start_index >= len(relative_time_ns) or end_index < 0 or start_index > end_index:
        return None

    return start_index, end_index


def write_dataframe_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    temp_path = build_temp_output_path(output_path)
    remove_if_exists(temp_path)
    try:
        dataframe.to_excel(temp_path, index=False)
        temp_path.replace(output_path)
    except Exception:
        remove_if_exists(temp_path)
        raise


def build_video_command(
    ffmpeg_path: str,
    source_path: Path,
    output_path: Path,
    time_range: TimeRange,
) -> list[str]:
    duration_seconds = (time_range.end_ns - time_range.start_ns) / NANOSECONDS_PER_SECOND
    duration_text = format(duration_seconds.normalize(), "f")

    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-nostdin",
        "-i",
        str(source_path),
        "-ss",
        format_feedback_timestamp_for_ffmpeg(time_range.start_text),
        "-t",
        duration_text,
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def cut_video_atomically(
    ffmpeg_path: str,
    source_path: Path,
    output_path: Path,
    time_range: TimeRange,
) -> None:
    temp_path = build_temp_output_path(output_path)
    remove_if_exists(temp_path)

    command = build_video_command(
        ffmpeg_path=ffmpeg_path,
        source_path=source_path,
        output_path=temp_path,
        time_range=time_range,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not is_complete_output(temp_path):
        remove_if_exists(temp_path)
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(detail)

    temp_path.replace(output_path)


def process_signal_file(
    source_path: Path,
    output_stem: str,
    time_ranges: list[TimeRange],
    output_dir: Path,
) -> ProcessStats:
    stats = ProcessStats()
    loaded = load_table(source_path)
    if loaded is None:
        stats.warnings += 1
        return stats

    for time_range in time_ranges:
        output_path = output_dir / f"{output_stem}_range{time_range.range_index:02d}.xlsx"
        if is_complete_output(output_path):
            stats.skipped += 1
            continue

        aligned = align_range_to_samples(loaded.relative_time_ns, time_range)
        if aligned is None:
            stats.warnings += warn(
                f"{source_path.name} 的 range{time_range.range_index:02d} "
                f"({time_range.start_text} - {time_range.end_text}) 没有可对齐的数据点。"
            )
            continue

        start_index, end_index = aligned
        sliced = loaded.dataframe.iloc[start_index : end_index + 1]
        if sliced.empty:
            stats.warnings += warn(
                f"{source_path.name} 的 range{time_range.range_index:02d} 对齐后为空。"
            )
            continue

        try:
            write_dataframe_atomically(sliced, output_path)
        except Exception as exc:
            stats.warnings += warn(f"写入 {output_path.name} 失败: {exc}")
            continue

        stats.created += 1

    return stats


def process_video_file(
    source_path: Path,
    output_stem: str,
    time_ranges: list[TimeRange],
    output_dir: Path,
    ffmpeg_path: str,
) -> ProcessStats:
    stats = ProcessStats()

    if not source_path.exists():
        stats.warnings += warn(f"缺少视频文件: {source_path}")
        return stats

    for time_range in time_ranges:
        output_path = output_dir / f"{output_stem}_range{time_range.range_index:02d}.mp4"
        if is_complete_output(output_path):
            stats.skipped += 1
            continue

        try:
            cut_video_atomically(
                ffmpeg_path=ffmpeg_path,
                source_path=source_path,
                output_path=output_path,
                time_range=time_range,
            )
        except Exception as exc:
            stats.warnings += warn(
                f"切割视频 {output_path.name} 失败 "
                f"({time_range.start_text} - {time_range.end_text}): {exc}"
            )
            continue

        stats.created += 1

    return stats


def process_subject_experiment(
    subject_id: str,
    experiment_id: str,
    time_ranges: list[TimeRange],
    input_root: Path,
    videos_root: Path,
    output_root: Path,
    ffmpeg_path: str | None,
    include_video: bool,
) -> ProcessStats:
    stats = ProcessStats()
    subject_dir = input_root / subject_id
    experiment_output_dir = output_root / experiment_id

    ppg_path = subject_dir / f"{subject_id}_{experiment_id}.xlsx"
    eeg_path = subject_dir / f"{subject_id}_e_{experiment_id}.xlsx"
    video_path = videos_root / f"{experiment_id}.mp4"

    if not ppg_path.exists():
        stats.warnings += warn(f"缺少 PPG 文件: {ppg_path}")
    else:
        stats.merge(
            process_signal_file(
                source_path=ppg_path,
                output_stem=f"{subject_id}_{experiment_id}",
                time_ranges=time_ranges,
                output_dir=experiment_output_dir,
            )
        )

    if not eeg_path.exists():
        stats.warnings += warn(f"缺少 EEG 文件: {eeg_path}")
    else:
        stats.merge(
            process_signal_file(
                source_path=eeg_path,
                output_stem=f"{subject_id}_e_{experiment_id}",
                time_ranges=time_ranges,
                output_dir=experiment_output_dir,
            )
        )

    if include_video:
        if ffmpeg_path is None:
            raise RuntimeError("include_video=True 时 ffmpeg_path 不能为空。")
        stats.merge(
            process_video_file(
                source_path=video_path,
                output_stem=f"{subject_id}_{experiment_id}",
                time_ranges=time_ranges,
                output_dir=experiment_output_dir,
                ffmpeg_path=ffmpeg_path,
            )
        )

    print(
        f"Experiment {experiment_id} subject {subject_id}: "
        f"created {stats.created}, skipped {stats.skipped}, warnings {stats.warnings}."
    )
    return stats


def process_all(
    include_video: bool = True,
    input_root: Path = INPUT_ROOT,
    time_data_dir: Path = TIME_DATA_DIR,
    videos_root: Path = VIDEOS_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> ProcessStats:
    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")
    if not time_data_dir.exists():
        raise FileNotFoundError(f"反馈时间目录不存在: {time_data_dir}")

    ffmpeg_path = resolve_ffmpeg_path() if include_video else None
    feedback_ranges, parse_warnings = load_feedback_ranges(time_data_dir)
    available_subject_ids = set(iter_subject_ids(input_root))

    ensure_output_directories(output_root)
    print(f"video_cutting_enabled={include_video}")

    total_stats = ProcessStats(warnings=parse_warnings)
    for experiment_id in EXPERIMENT_IDS:
        experiment_feedback = feedback_ranges.get(experiment_id, {})
        subject_ids = sorted(set(experiment_feedback.keys()) | available_subject_ids, key=int)

        for subject_id in subject_ids:
            time_ranges = experiment_feedback.get(subject_id)
            if not time_ranges:
                if subject_id in available_subject_ids:
                    total_stats.warnings += warn(
                        f"受试者 {subject_id} 的实验 {experiment_id} 存在源数据，但没有反馈时间区间。"
                    )
                continue

            subject_stats = process_subject_experiment(
                subject_id=subject_id,
                experiment_id=experiment_id,
                time_ranges=time_ranges,
                input_root=input_root,
                videos_root=videos_root,
                output_root=output_root,
                ffmpeg_path=ffmpeg_path,
                include_video=include_video,
            )
            total_stats.merge(subject_stats)

    print(
        f"Finished. Output: {output_root}. "
        f"Created {total_stats.created}, skipped {total_stats.skipped}, warnings {total_stats.warnings}."
    )
    return total_stats


def main() -> None:
    try:
        include_video = prompt_video_cutting_enabled()
        process_all(include_video=include_video)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
