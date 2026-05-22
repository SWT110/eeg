import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "list_cut_fixed_duration"
TARGET_DIR = BASE_DIR / "list_normalization_fixed_duration"
TARGET_NAME_RE = re.compile(r"^\d+_e_\d+\.xlsx$", re.IGNORECASE)
NORMALIZE_PREFIXES = ("EEG", "ECG", "EMG", "SaO2")


def target_is_current(source_path: Path, target_path: Path) -> bool:
    return target_path.exists() and target_path.stat().st_mtime >= source_path.stat().st_mtime


def process_file(source_path: Path) -> str:
    source_path = source_path.resolve()

    if not TARGET_NAME_RE.match(source_path.name):
        return "skipped"

    relative_dir = source_path.parent.relative_to(SOURCE_DIR)
    output_dir = TARGET_DIR / relative_dir
    output_path = output_dir / source_path.name

    if target_is_current(source_path, output_path):
        print(f"Skip (up-to-date): {output_path}")
        return "skipped"

    try:
        df = pd.read_excel(source_path)
    except Exception as exc:
        print(f"Failed to read {source_path}: {exc}")
        return "error"

    columns_to_normalize = [
        col for col in df.columns if isinstance(col, str) and col.startswith(NORMALIZE_PREFIXES)
    ]
    if not columns_to_normalize:
        print(f"Skip (no EEG/ECG/EMG/SaO2 columns): {source_path}")
        return "skipped"

    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    output_dir.mkdir(parents=True, exist_ok=True)
    df_normalized.to_excel(output_path, index=False)
    print(f"Saved: {output_path}")
    return "processed"


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {SOURCE_DIR}")

    TARGET_DIR.mkdir(exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    for source_path in sorted(SOURCE_DIR.rglob("*.xlsx")):
        status = process_file(source_path)
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


if __name__ == "__main__":
    main()
