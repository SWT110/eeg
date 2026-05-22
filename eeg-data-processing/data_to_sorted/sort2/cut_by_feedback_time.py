from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
TIME_DATA_DIR = BASE_DIR / "time_data"
INPUT_ROOT = PROJECT_ROOT / "data_to_list" / "list_cut_fixed_duration"
VIDEOS_ROOT = PROJECT_ROOT / "videos"
TIME_COLUMN = "Time"
EXPERIMENT_IDS = ("1", "2", "3")
SIGNAL_TYPES = ("ppg", "eeg")
OUTPUT_SUBDIRS = SIGNAL_TYPES + ("video",)
NANOSECONDS_PER_SECOND = Decimal("1000000000")
BLOCK_HEADER_PATTERN = re.compile(r"^(?P<experiment>\d+)-(?P<subject>\d+)$")
RANGE_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
VIDEO_CUT_CHOICES = {"1": True, "2": False}
VIDEO_MODE_CHOICES = {"1": "precise", "2": "fast"}
VIDEO_MODE_LABELS = {"precise": "精确模式", "fast": "快速模式"}


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


@dataclass(frozen=True)
class VideoRuntime:
    ffmpeg_path: str
    ffprobe_path: str
    mode: str


@dataclass
class ProcessSummary:
    created: int = 0
    skipped: int = 0
    discarded: int = 0

    def add(self, other: "ProcessSummary") -> None:
        self.created += other.created
        self.skipped += other.skipped
        self.discarded += other.discarded

    def describe(self, label: str) -> str:
        return (
            f"{label} created {self.created}, "
            f"skipped {self.skipped}, discarded {self.discarded}"
        )


@dataclass(frozen=True)
class RunSummary:
    data: ProcessSummary
    video: ProcessSummary | None


def prompt_segment_seconds() -> tuple[Decimal, str]:
    print("第一步：设置切割秒数。")
    print("作用：决定每个数据片段的长度；如果启用视频切割，也会决定每个视频片段的长度。")

    while True:
        raw = input("请输入切割秒数（例如 1 或 0.5）: ").strip()
        try:
            seconds = Decimal(raw)
        except InvalidOperation:
            print("输入无效，请输入正数，例如 1 或 0.5。")
            continue

        if seconds <= 0:
            print("切割秒数必须大于 0。")
            continue

        return seconds, format_seconds_label(seconds)


def prompt_cut_video() -> bool:
    print("第二步：选择是否切割视频。")
    print("1. 切割视频：会额外生成 video 目录，需要 ffmpeg 和 ffprobe。")
    print("2. 不切割视频：只处理数据，不检查 ffmpeg，也不会生成新的视频片段。")

    while True:
        raw = input("请输入选项编号 (1/2): ").strip()
        cut_video = VIDEO_CUT_CHOICES.get(raw)
        if cut_video is not None:
            return cut_video
        print("输入无效，请输入 1 或 2。")


def prompt_video_mode() -> str:
    print("第三步：选择视频切片模式。")
    print("作用：只影响视频切片方式，不影响数据切片。")
    print("1. 精确模式：重新编码，边界更准确，速度更慢。")
    print("2. 快速模式：流复制，速度更快，但边界可能略不精确。")

    while True:
        raw = input("请输入模式编号 (1/2): ").strip()
        mode = VIDEO_MODE_CHOICES.get(raw)
        if mode:
            return mode
        print("输入无效，请输入 1 或 2。")


def format_seconds_label(seconds: Decimal) -> str:
    return format(seconds.normalize(), "f")


def seconds_to_nanoseconds(seconds: Decimal) -> int:
    return int((seconds * NANOSECONDS_PER_SECOND).to_integral_value())


def parse_feedback_timestamp_to_ns(timestamp: str) -> int:
    hours, minutes, seconds = timestamp.split(":")
    second_part, milliseconds = seconds.split(",")
    total_seconds = (
        Decimal(hours) * Decimal(3600)
        + Decimal(minutes) * Decimal(60)
        + Decimal(second_part)
        + (Decimal(milliseconds) / Decimal(1000))
    )
    return seconds_to_nanoseconds(total_seconds)


def format_ns_for_ffmpeg(ns_value: int) -> str:
    total_milliseconds = (ns_value + 500_000) // 1_000_000
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def read_text_with_fallbacks(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def parse_feedback_file(path: Path, expected_experiment_id: str) -> dict[str, list[TimeRange]]:
    feedback_ranges: dict[str, list[TimeRange]] = {}
    current_subject_id: str | None = None

    for raw_line in read_text_with_fallbacks(path).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_match = BLOCK_HEADER_PATTERN.match(line)
        if header_match:
            current_subject_id = header_match.group("subject")
            if header_match.group("experiment") != expected_experiment_id:
                print(
                    f"Warning: {path.name} contains block {line}, "
                    f"which does not match expected experiment {expected_experiment_id}."
                )
            feedback_ranges.setdefault(current_subject_id, [])
            continue

        range_match = RANGE_PATTERN.search(line)
        if range_match and current_subject_id is not None:
            subject_ranges = feedback_ranges.setdefault(current_subject_id, [])
            subject_ranges.append(
                TimeRange(
                    subject_id=current_subject_id,
                    experiment_id=expected_experiment_id,
                    range_index=len(subject_ranges) + 1,
                    start_ns=parse_feedback_timestamp_to_ns(range_match.group("start")),
                    end_ns=parse_feedback_timestamp_to_ns(range_match.group("end")),
                    start_text=range_match.group("start"),
                    end_text=range_match.group("end"),
                )
            )

    return feedback_ranges


def load_feedback_ranges(time_data_dir: Path = TIME_DATA_DIR) -> dict[str, dict[str, list[TimeRange]]]:
    all_ranges: dict[str, dict[str, list[TimeRange]]] = {}

    for experiment_id in EXPERIMENT_IDS:
        feedback_path = time_data_dir / f"反馈时间{experiment_id}.txt"
        if not feedback_path.exists():
            print(f"Warning: missing feedback file {feedback_path}.")
            all_ranges[experiment_id] = {}
            continue
        all_ranges[experiment_id] = parse_feedback_file(feedback_path, experiment_id)

    return all_ranges


def iter_subject_ids(input_root: Path) -> Iterable[str]:
    return [
        path.name
        for path in sorted(
            (candidate for candidate in input_root.iterdir() if candidate.is_dir() and candidate.name.isdigit()),
            key=lambda candidate: int(candidate.name),
        )
    ]


def ensure_output_directories(output_root: Path, cut_video: bool) -> None:
    for experiment_id in EXPERIMENT_IDS:
        subdir_names = list(SIGNAL_TYPES)
        if cut_video:
            subdir_names.append("video")
        for subdir_name in subdir_names:
            (output_root / experiment_id / subdir_name).mkdir(parents=True, exist_ok=True)


def is_complete_output(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def load_table(path: Path) -> LoadedTable | None:
    try:
        dataframe = pd.read_excel(path)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None

    if TIME_COLUMN not in dataframe.columns:
        print(f"Warning: {path} is missing the '{TIME_COLUMN}' column.")
        return None

    timedelta_series = pd.to_timedelta(dataframe[TIME_COLUMN].astype(str), errors="coerce")
    if timedelta_series.isna().any():
        print(f"Warning: {path} contains unparsable '{TIME_COLUMN}' values.")
        return None

    time_ns = timedelta_series.array.asi8
    if len(time_ns) == 0:
        print(f"Warning: {path} is empty.")
        return None

    relative_time_ns = time_ns - int(time_ns[0])
    if not np.all(relative_time_ns[:-1] <= relative_time_ns[1:]):
        print(f"Warning: {path} has a non-monotonic '{TIME_COLUMN}' column.")
        return None

    return LoadedTable(dataframe=dataframe, relative_time_ns=relative_time_ns)


def align_range_to_samples(relative_time_ns: np.ndarray, time_range: TimeRange) -> tuple[int, int] | None:
    start_index = int(np.searchsorted(relative_time_ns, time_range.start_ns, side="left"))
    end_index = int(np.searchsorted(relative_time_ns, time_range.end_ns, side="right")) - 1

    if start_index >= len(relative_time_ns) or end_index < 0 or start_index > end_index:
        return None

    return start_index, end_index


def build_segment_boundaries(start_ns: int, end_ns: int, segment_ns: int) -> list[tuple[int, int, int]]:
    segment_count = int((end_ns - start_ns) // segment_ns)
    return [
        (
            segment_index,
            start_ns + (segment_index - 1) * segment_ns,
            start_ns + segment_index * segment_ns,
        )
        for segment_index in range(1, segment_count + 1)
    ]


def slice_dataframe_by_window(
    dataframe: pd.DataFrame,
    relative_time_ns: np.ndarray,
    window_start_ns: int,
    window_end_ns: int,
) -> pd.DataFrame:
    left = int(np.searchsorted(relative_time_ns, window_start_ns, side="left"))
    right = int(np.searchsorted(relative_time_ns, window_end_ns, side="left"))
    return dataframe.iloc[left:right]


def ensure_video_runtime(
    videos_root: Path = VIDEOS_ROOT,
    video_mode: str = "precise",
) -> VideoRuntime:
    if video_mode not in VIDEO_MODE_LABELS:
        raise ValueError(f"Unsupported video mode: {video_mode}")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("未找到 ffmpeg，请先安装 ffmpeg 并加入 PATH 后再运行脚本。")

    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("未找到 ffprobe，请先安装 ffprobe 并加入 PATH 后再运行脚本。")

    if not videos_root.exists():
        raise FileNotFoundError(f"视频目录不存在: {videos_root}")

    for experiment_id in EXPERIMENT_IDS:
        video_path = videos_root / f"{experiment_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"缺少实验 {experiment_id} 的视频文件: {video_path}")

        probe_result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe_result.returncode != 0:
            detail = probe_result.stderr.strip() or probe_result.stdout.strip() or "unknown ffprobe error"
            raise RuntimeError(f"ffprobe 无法读取 {video_path.name}: {detail}")
        if not probe_result.stdout.strip():
            raise RuntimeError(f"ffprobe 未返回 {video_path.name} 的时长信息。")

    return VideoRuntime(
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        mode=video_mode,
    )


def build_ffmpeg_command(
    ffmpeg_path: str,
    source_path: Path,
    output_path: Path,
    start_ns: int,
    duration_ns: int,
    video_mode: str,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        format_ns_for_ffmpeg(start_ns),
        "-i",
        str(source_path),
        "-t",
        format_ns_for_ffmpeg(duration_ns),
    ]

    if video_mode == "precise":
        command.extend(
            [
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
            ]
        )
    else:
        command.extend(["-c", "copy"])

    command.append(str(output_path))
    return command


def run_ffmpeg_command(command: list[str], output_path: Path) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not is_complete_output(output_path):
        if output_path.exists():
            output_path.unlink()
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"ffmpeg 切片失败: {detail}")


def process_signal_file(
    source_path: Path,
    output_dir: Path,
    output_stem: str,
    time_ranges: list[TimeRange],
    segment_ns: int,
) -> ProcessSummary:
    loaded = load_table(source_path)
    if loaded is None:
        return ProcessSummary()

    summary = ProcessSummary()

    for time_range in time_ranges:
        aligned = align_range_to_samples(loaded.relative_time_ns, time_range)
        if aligned is None:
            summary.discarded += 1
            print(
                f"Discarded {source_path.name} range{time_range.range_index:02d}: "
                f"{time_range.start_text} - {time_range.end_text} has no aligned samples."
            )
            continue

        start_index, end_index = aligned
        aligned_start_ns = int(loaded.relative_time_ns[start_index])
        aligned_end_ns = int(loaded.relative_time_ns[end_index])
        effective_duration_ns = aligned_end_ns - aligned_start_ns

        if effective_duration_ns <= segment_ns:
            summary.discarded += 1
            print(
                f"Discarded {source_path.name} range{time_range.range_index:02d}: "
                f"{time_range.start_text} - {time_range.end_text} "
                f"aligned duration <= requested segment length."
            )
            continue

        segment_boundaries = build_segment_boundaries(aligned_start_ns, aligned_end_ns, segment_ns)
        for segment_index, window_start_ns, window_end_ns in segment_boundaries:
            output_path = output_dir / (
                f"{output_stem}_range{time_range.range_index:02d}_seg{segment_index:03d}.xlsx"
            )
            if is_complete_output(output_path):
                summary.skipped += 1
                continue

            segment_df = slice_dataframe_by_window(
                loaded.dataframe,
                loaded.relative_time_ns,
                window_start_ns,
                window_end_ns,
            )
            if segment_df.empty:
                continue

            segment_df.to_excel(output_path, index=False)
            summary.created += 1

    return summary


def process_video_file(
    source_path: Path,
    output_dir: Path,
    output_stem: str,
    time_ranges: list[TimeRange],
    segment_ns: int,
    runtime: VideoRuntime,
) -> ProcessSummary:
    summary = ProcessSummary()

    for time_range in time_ranges:
        raw_duration_ns = time_range.end_ns - time_range.start_ns
        if raw_duration_ns <= segment_ns:
            summary.discarded += 1
            print(
                f"Discarded video {output_stem} range{time_range.range_index:02d}: "
                f"{time_range.start_text} - {time_range.end_text} "
                f"original duration <= requested segment length."
            )
            continue

        segment_boundaries = build_segment_boundaries(time_range.start_ns, time_range.end_ns, segment_ns)
        for segment_index, window_start_ns, window_end_ns in segment_boundaries:
            output_path = output_dir / (
                f"{output_stem}_range{time_range.range_index:02d}_seg{segment_index:03d}.mp4"
            )
            if is_complete_output(output_path):
                summary.skipped += 1
                continue

            ffmpeg_command = build_ffmpeg_command(
                ffmpeg_path=runtime.ffmpeg_path,
                source_path=source_path,
                output_path=output_path,
                start_ns=window_start_ns,
                duration_ns=window_end_ns - window_start_ns,
                video_mode=runtime.mode,
            )
            run_ffmpeg_command(ffmpeg_command, output_path)
            summary.created += 1

    return summary


def process_subject_experiment(
    subject_id: str,
    experiment_id: str,
    time_ranges: list[TimeRange],
    input_root: Path,
    videos_root: Path,
    output_root: Path,
    segment_ns: int,
    cut_video: bool,
    runtime: VideoRuntime | None,
) -> RunSummary:
    subject_dir = input_root / subject_id
    ppg_path = subject_dir / f"{subject_id}_{experiment_id}.xlsx"
    eeg_path = subject_dir / f"{subject_id}_e_{experiment_id}.xlsx"

    if not ppg_path.exists():
        print(f"Warning: missing PPG file {ppg_path}.")
    if not eeg_path.exists():
        print(f"Warning: missing EEG file {eeg_path}.")
    if not ppg_path.exists() and not eeg_path.exists() and not cut_video:
        print(
            f"Warning: subject {subject_id} experiment {experiment_id} "
            f"is missing both data files; nothing to process."
        )
    elif not ppg_path.exists() and not eeg_path.exists() and cut_video:
        print(
            f"Warning: subject {subject_id} experiment {experiment_id} "
            f"is missing both data files; only video segments will be generated."
        )

    data_summary = ProcessSummary()
    video_summary = ProcessSummary() if cut_video else None

    if ppg_path.exists():
        data_summary.add(process_signal_file(
            source_path=ppg_path,
            output_dir=output_root / experiment_id / "ppg",
            output_stem=f"{subject_id}_{experiment_id}",
            time_ranges=time_ranges,
            segment_ns=segment_ns,
        ))

    if eeg_path.exists():
        data_summary.add(process_signal_file(
            source_path=eeg_path,
            output_dir=output_root / experiment_id / "eeg",
            output_stem=f"{subject_id}_e_{experiment_id}",
            time_ranges=time_ranges,
            segment_ns=segment_ns,
        ))

    if cut_video and runtime is not None:
        video_path = videos_root / f"{experiment_id}.mp4"
        video_summary = process_video_file(
            source_path=video_path,
            output_dir=output_root / experiment_id / "video",
            output_stem=f"{subject_id}_{experiment_id}",
            time_ranges=time_ranges,
            segment_ns=segment_ns,
            runtime=runtime,
        )

    log_parts = [
        f"Subject {subject_id} experiment {experiment_id}:",
        data_summary.describe("data"),
    ]
    if video_summary is not None:
        log_parts.append(video_summary.describe("video"))
    print("  ".join(log_parts))

    return RunSummary(data=data_summary, video=video_summary)


def process_all(
    segment_seconds: Decimal,
    cut_video: bool,
    video_mode: str | None = None,
    input_root: Path = INPUT_ROOT,
    time_data_dir: Path = TIME_DATA_DIR,
    videos_root: Path = VIDEOS_ROOT,
    output_root: Path | None = None,
) -> RunSummary:
    if output_root is None:
        output_root = BASE_DIR / f"data_cut_{format_seconds_label(segment_seconds)}s"

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if not time_data_dir.exists():
        raise FileNotFoundError(f"Time data directory does not exist: {time_data_dir}")

    runtime: VideoRuntime | None = None
    if cut_video:
        runtime = ensure_video_runtime(videos_root=videos_root, video_mode=video_mode or "precise")

    segment_ns = seconds_to_nanoseconds(segment_seconds)
    feedback_ranges = load_feedback_ranges(time_data_dir)
    available_subject_ids = set(iter_subject_ids(input_root))

    ensure_output_directories(output_root, cut_video=cut_video)

    if cut_video and runtime is not None:
        print(f"视频切片模式: {VIDEO_MODE_LABELS[runtime.mode]}")
    else:
        print("视频切割未启用，仅处理数据。")

    total_data = ProcessSummary()
    total_video = ProcessSummary() if cut_video else None

    for experiment_id in EXPERIMENT_IDS:
        experiment_feedback = feedback_ranges.get(experiment_id, {})
        subject_ids = sorted(set(experiment_feedback.keys()) | available_subject_ids, key=int)

        for subject_id in subject_ids:
            time_ranges = experiment_feedback.get(subject_id)
            if not time_ranges:
                if subject_id in available_subject_ids:
                    print(
                        f"Warning: subject {subject_id} experiment {experiment_id} "
                        f"has source files but no feedback ranges."
                    )
                continue

            run_summary = process_subject_experiment(
                subject_id=subject_id,
                experiment_id=experiment_id,
                time_ranges=time_ranges,
                input_root=input_root,
                videos_root=videos_root,
                output_root=output_root,
                segment_ns=segment_ns,
                cut_video=cut_video,
                runtime=runtime,
            )
            total_data.add(run_summary.data)
            if total_video is not None and run_summary.video is not None:
                total_video.add(run_summary.video)

    log_parts = [f"Finished. Output: {output_root}.", total_data.describe("data")]
    if total_video is not None:
        log_parts.append(total_video.describe("video"))
    print("  ".join(log_parts))

    return RunSummary(data=total_data, video=total_video)


def main() -> None:
    segment_seconds, seconds_label = prompt_segment_seconds()
    cut_video = prompt_cut_video()
    video_mode: str | None = None
    if cut_video:
        video_mode = prompt_video_mode()
    output_root = BASE_DIR / f"data_cut_{seconds_label}s"
    process_all(
        segment_seconds=segment_seconds,
        cut_video=cut_video,
        video_mode=video_mode,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
