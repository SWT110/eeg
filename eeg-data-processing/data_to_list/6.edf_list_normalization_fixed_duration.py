from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
EEG_ROOT = BASE_DIR.parents[1]
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import LIST_CUT_DIR, LIST_NORMALIZED_DIR

SOURCE_DIR = LIST_CUT_DIR
TARGET_DIR = LIST_NORMALIZED_DIR
TARGET_NAME_RE = re.compile(r"^\d+_e_\d+\.xlsx$", re.IGNORECASE)
NORMALIZE_PREFIXES = ("EEG", "ECG", "EMG", "SaO2")


def target_is_current(source_path: Path, target_path: Path) -> bool:
    if not target_path.is_file() or target_path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(target_path) as archive:
            complete = {"[Content_Types].xml", "xl/workbook.xml", "xl/worksheets/sheet1.xml"}.issubset(
                archive.namelist()
            )
    except (OSError, zipfile.BadZipFile):
        return False
    return complete and target_path.stat().st_mtime >= source_path.stat().st_mtime


def process_file(
    source_path: Path,
    source_dir: Path | None = None,
    target_dir: Path | None = None,
) -> str:
    source_path = source_path.resolve()
    source_dir = (SOURCE_DIR if source_dir is None else source_dir).expanduser().resolve()
    target_dir = (TARGET_DIR if target_dir is None else target_dir).expanduser()

    if not TARGET_NAME_RE.match(source_path.name):
        return "skipped"

    relative_dir = source_path.parent.relative_to(source_dir)
    output_dir = target_dir / relative_dir
    output_path = output_dir / source_path.name
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.xlsx")

    if target_is_current(source_path, output_path):
        print(f"Skip (up-to-date): {output_path}")
        return "skipped"

    try:
        df = pd.read_excel(source_path)
        columns_to_normalize = [
            col for col in df.columns if isinstance(col, str) and col.startswith(NORMALIZE_PREFIXES)
        ]
        if not columns_to_normalize:
            raise ValueError("No EEG/ECG/EMG/SaO2 columns")
        if df.empty or not np.isfinite(df[columns_to_normalize].to_numpy(dtype=float)).all():
            raise ValueError("Signal columns are empty or contain NaN/Inf")

        scaler = StandardScaler()
        df_normalized = df.copy()
        df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

        output_dir.mkdir(parents=True, exist_ok=True)
        df_normalized.to_excel(temporary_path, index=False)
        temporary_path.replace(output_path)
        print(f"Saved: {output_path}")
        return "processed"
    except Exception as exc:
        print(f"Failed to normalize {source_path}: {exc}")
        return "error"
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def positive_subject_id(value: str) -> int:
    try:
        subject_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Subject IDs must be positive integers") from exc
    if subject_id < 1:
        raise argparse.ArgumentTypeError("Subject IDs must be positive integers")
    return subject_id


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按实验段分别标准化 EEG/ECG/EMG/SaO2 信号")
    parser.add_argument("--input-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=TARGET_DIR)
    parser.add_argument("--subject-ids", type=positive_subject_id, nargs="+", default=None,
                        help="只处理指定受试者的前三段；省略时扫描全部 Excel")
    return parser.parse_args(argv)


def select_source_paths(source_dir: Path, subject_ids: list[int] | None) -> list[Path]:
    if subject_ids is None:
        return sorted(source_dir.rglob("*.xlsx"))
    paths = [
        source_dir / str(subject_id) / f"{subject_id}_e_{index}.xlsx"
        for subject_id in sorted(set(subject_ids)) for index in (1, 2, 3)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing selected experiment Excel files: {missing}")
    return paths


def main(argv=None) -> None:
    args = parse_args(argv)
    source_dir = args.input_dir.expanduser().resolve()
    target_dir = args.output_dir.expanduser()
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {source_dir}")

    source_paths = select_source_paths(source_dir, args.subject_ids)
    target_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    for source_path in source_paths:
        status = process_file(source_path, source_dir, target_dir)
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    print(
        "Fixed-duration normalization complete. "
        f"processed={processed}, skipped={skipped}, errors={errors}"
    )
    if errors:
        raise RuntimeError(f"Failed to normalize {errors} experiment file(s); incomplete outputs were not published")


if __name__ == "__main__":
    main()
