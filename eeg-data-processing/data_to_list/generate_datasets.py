#!/usr/bin/env python3
"""根据 window_stride_configs.json 批量生成不同参数的活动数据集。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_activity_global_index import build_global_dataset

# ---- 默认路径（都在 data_to_list 目录下） ----
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG      = BASE_DIR / "window_stride_configs.json"
DEFAULT_INPUT_ROOT  = BASE_DIR / "list_normalization_fixed_duration"
DEFAULT_OUTPUT_BASE = BASE_DIR / "global_activity_dataset"


def subdir_name(window_seconds: float, stride_seconds: float) -> str:
    """根据参数生成子目录名，例如 'window_2.0_stride_1.0'"""
    # 去掉多余的末尾 0，保持可读性
    ws = f"{window_seconds:g}"
    ss = f"{stride_seconds:g}"
    return f"window_{ws}_stride_{ss}"


def is_dataset_present(output_dir: Path) -> bool:
    """简单判断这个参数组合是否已经生成过（有 metadata.json 即认为已完成）"""
    return (output_dir / "metadata.json").exists()


def main():
    parser = argparse.ArgumentParser(description="批量生成不同窗口/步长的全局活动数据集")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON 配置文件路径（默认 window_stride_configs.json）")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                        help="输入根目录（默认 list_normalization_fixed_duration）")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE,
                        help="输出根目录（默认 global_activity_dataset）")
    args = parser.parse_args()

    # 1. 读取配置
    if not args.config.exists():
        print(f"配置文件不存在：{args.config}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        print("配置文件应当是一个 JSON 数组")
        return

    # 2. 遍历每一个组合
    for i, cfg in enumerate(configs):
        if not isinstance(cfg, dict):
            print(f"第 {i+1} 个配置项格式错误，跳过：{cfg}")
            continue

        window = cfg.get("window_seconds")
        if window is None:
            print(f"第 {i+1} 个配置项缺少 'window_seconds'，跳过：{cfg}")
            continue
        window = float(window)
        if window <= 0:
            print(f"第 {i+1} 个配置项 window_seconds 必须为正数，跳过：{cfg}")
            continue

        stride = cfg.get("stride_seconds")
        if stride is None:
            stride = window
        else:
            stride = float(stride)
            if stride <= 0:
                print(f"第 {i+1} 个配置项 stride_seconds 必须为正数，跳过：{cfg}")
                continue

        # 子目录名和完整输出路径
        sub_dir = subdir_name(window, stride)
        out_dir = args.output_base / sub_dir

        if is_dataset_present(out_dir):
            print(f"[跳过] 已存在：{sub_dir}")
            continue

        print(f"[生成] {sub_dir} ...")
        try:
            build_global_dataset(
                input_root=args.input_root,
                output_root=out_dir,
                window_seconds=window,
                stride_seconds=stride,
            )
        except Exception as e:
            print(f"[失败] {sub_dir}：{e}")
        else:
            print(f"[完成] {sub_dir}")

    print("\n全部任务结束。")


if __name__ == "__main__":
    main()