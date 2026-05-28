from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EEG_ROOT = BASE_DIR.parents[1]
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import DATA_TO_LIST_DATA_DIR, LIST_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量将 EDF 文件转换为 Excel")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_TO_LIST_DATA_DIR,
        help=f"EDF 输入目录（默认: {DATA_TO_LIST_DATA_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LIST_DIR,
        help=f"Excel 输出目录（默认: {LIST_DIR}）",
    )
    return parser.parse_args(argv)


def discover_processed_files(output_folder: Path) -> set[str]:
    processed_files: set[str] = set()
    if output_folder.exists():
        for file_path in output_folder.iterdir():
            if file_path.suffix == ".xlsx":
                processed_files.add(file_path.stem)
    return processed_files


def convert_edf_to_excel(edf_path: Path, output_folder: Path) -> bool:
    import mne

    base_name = edf_path.stem

    try:
        print(f"\n正在处理: {edf_path.name}")
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        signal_data = raw.get_data()
        labels = raw.ch_names

        sfreq = raw.info["sfreq"]
        start_time = raw.info["meas_date"]
        n_samples = signal_data.shape[1]

        time_step = 1.0 / sfreq
        times = [i * time_step for i in range(n_samples)]

        if start_time is not None:
            start_time = start_time.replace(tzinfo=None)
            absolute_times = [start_time + timedelta(seconds=t) for t in times]
            time_strings = [t.strftime("%H:%M:%S.%f")[:-3] for t in absolute_times]
            print(f"  开始时间: {start_time}")
            print(f"  采样率: {sfreq} Hz (时间精度: {time_step * 1000:.3f} ms)")
            print(f"  时间格式示例: {time_strings[0]}")
        else:
            print("  警告: 未找到开始时间,使用相对时间")
            time_strings = [f"{t:.3f}" for t in times]

        df = pd.DataFrame(signal_data.T, columns=labels)
        df["Time"] = time_strings

        output_file = output_folder / f"{base_name}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"  ✓ 成功导出: {output_file.name}")
        return True
    except Exception as exc:
        print(f"  ✗ 处理 {edf_path.name} 时出错: {exc}")
        return False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_folder = args.input_dir.expanduser()
    output_folder = args.output_dir.expanduser()

    print(f"脚本目录: {BASE_DIR}")
    print(f"输入目录: {input_folder}")
    print(f"输出目录: {output_folder}")

    if not input_folder.exists():
        print(f"✗ 输入目录不存在: {input_folder}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    processed_files = discover_processed_files(output_folder)
    print(f"\n已处理的文件数量: {len(processed_files)}")

    total_files = 0
    skipped_files = 0
    processed_count = 0
    error_count = 0

    for edf_path in sorted(input_folder.iterdir()):
        if edf_path.suffix.lower() != ".edf":
            continue
        total_files += 1
        base_name = edf_path.stem

        if base_name in processed_files:
            print(f"\n⊗ 跳过(已处理): {edf_path.name}")
            skipped_files += 1
            continue

        if convert_edf_to_excel(edf_path, output_folder):
            processed_count += 1
        else:
            error_count += 1

    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"总文件数: {total_files}")
    print(f"已跳过: {skipped_files}")
    print(f"新处理: {processed_count}")
    print(f"错误数: {error_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
