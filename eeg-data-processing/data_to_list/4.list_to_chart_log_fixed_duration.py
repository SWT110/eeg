from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
EEG_ROOT = BASE_DIR.parents[1]
if str(EEG_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ROOT))

from eeg_project_paths import LIST_CUT_DIR

DATA_DIR = LIST_CUT_DIR
TIME_COL = "Time"
GSR_COL = "GSR(V)"
PPG_COL = "PPG(BPM)"


def parse_time_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%H:%M:%S.%f", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def process_file(file_path: Path) -> str:
    file_path = file_path.resolve()

    if "_e_" in file_path.stem:
        return "skipped"

    output_path = file_path.with_suffix(".html")
    if output_path.exists():
        print(f"Skip (html exists): {output_path}")
        return "skipped"

    try:
        df = pd.read_excel(file_path)
    except Exception as exc:
        print(f"Failed to read {file_path}: {exc}")
        return "error"

    missing_cols = [col for col in (TIME_COL, GSR_COL, PPG_COL) if col not in df.columns]
    if missing_cols:
        print(f"Skip (missing columns {missing_cols}): {file_path}")
        return "error"

    df[TIME_COL] = parse_time_column(df[TIME_COL])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(GSR_COL, PPG_COL),
    )

    fig.add_trace(
        go.Scatter(x=df[TIME_COL], y=df[GSR_COL], mode="lines", name=GSR_COL),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df[TIME_COL], y=df[PPG_COL], mode="lines", name=PPG_COL),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{file_path.stem} - GSR & PPG",
        hovermode="x unified",
        height=700,
    )
    fig.update_xaxes(rangeslider=dict(visible=True), type="date", row=2, col=1)
    fig.update_yaxes(title_text=GSR_COL, row=1, col=1)
    fig.update_yaxes(title_text=PPG_COL, row=2, col=1)

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Saved: {output_path}")
    return "processed"


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {DATA_DIR}")

    processed = 0
    skipped = 0
    errors = 0

    for file_path in sorted(DATA_DIR.rglob("*.xlsx")):
        status = process_file(file_path)
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    print(
        "Fixed-duration log chart generation complete. "
        f"processed={processed}, skipped={skipped}, errors={errors}"
    )


if __name__ == "__main__":
    main()
