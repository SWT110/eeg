import re
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "list_cut_fixed_duration"
TIME_COL = "Time"
TRACE_COLOR = "#1f77b4"
TARGET_NAME_RE = re.compile(r"^\d+_e_\d+\.xlsx$", re.IGNORECASE)
CUSTOM_COLOR_VALUE = "__custom__"
OVERWRITE_CHOICES = {"1": True, "2": False}
COLOR_PRESETS = [
    ("Blue", "#1f77b4"),
    ("Green", "#2ca02c"),
    ("Red", "#d62728"),
    ("Orange", "#ff7f0e"),
    ("Purple", "#9467bd"),
    ("Cyan", "#17becf"),
    ("Brown", "#8c564b"),
    ("Pink", "#e377c2"),
    ("Gray", "#7f7f7f"),
    ("Black", "#111111"),
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <title>__TITLE__</title>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            overflow-y: scroll;
        }
        #toolbar {
            position: fixed;
            left: 0;
            right: 0;
            top: 0;
            z-index: 1000;
            background: #f8f9fa;
            border-bottom: 1px solid #ccc;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .group-box {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        #toolbar input[type="range"] {
            width: 150px;
        }
        #toolbar input[type="text"],
        #toolbar input[type="number"] {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
            width: 70px;
        }
        button {
            cursor: pointer;
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 12px;
            border-radius: 4px;
        }
        button:hover {
            background-color: #0056b3;
        }
        #settings-toggle {
            margin-left: auto;
            background-color: #6c757d;
        }
        #settings-panel {
            position: fixed;
            top: 50px;
            right: 10px;
            width: 430px;
            max-height: 80vh;
            background: white;
            border: 1px solid #ccc;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            z-index: 2000;
            display: none;
            flex-direction: column;
            border-radius: 6px;
        }
        .panel-header {
            padding: 10px;
            background: #f1f1f1;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .panel-content {
            padding: 10px;
            overflow-y: auto;
            flex: 1;
        }
        .batch-control {
            padding-bottom: 10px;
            border-bottom: 2px dashed #eee;
            margin-bottom: 10px;
        }
        .batch-control label {
            display: block;
            margin-bottom: 5px;
            color: #555;
        }
        .channel-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 5px;
            flex-wrap: wrap;
        }
        .ch-name {
            width: 86px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 12px;
            font-weight: bold;
        }
        .channel-row input {
            width: 60px;
            padding: 3px;
            font-size: 12px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .color-select {
            width: 82px;
            padding: 3px;
            font-size: 12px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .color-picker {
            width: 34px;
            height: 24px;
            padding: 0;
            border: 1px solid #ddd;
            border-radius: 3px;
            background: transparent;
        }
        .color-preview {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 1px solid #cbd5e1;
            flex: 0 0 auto;
        }
        .apply-btn-small {
            padding: 3px 8px;
            font-size: 11px;
            background-color: #17a2b8;
        }
        .apply-btn-small:hover {
            background-color: #138496;
        }
        .batch-control-row {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            align-items: center;
        }
        #plot-wrapper {
            margin-top: 60px;
            padding: 10px;
        }
        .sep {
            color: #999;
        }
    </style>
</head>
<body>
    <div id="toolbar">
        <div class="group-box">
            <strong>时间控制:</strong>
            <label>窗口(s) <input id="window-len" type="number" min="0.1" value="10"></label>
            <label>进度 <input id="start-slider" type="range" min="0" max="100" value="0" step="0.1"></label>
            <input id="time-start" type="text" placeholder="Start">
            <span class="sep">~</span>
            <input id="time-end" type="text" placeholder="End">
            <button id="apply-time-btn">跳转</button>
            <span id="current-window"></span>
        </div>
        <button id="settings-toggle">通道量程设置</button>
    </div>

    <div id="settings-panel">
        <div class="panel-header">
            <span>各通道纵轴范围</span>
            <button id="settings-close" style="background:none;color:#333;font-size:16px;padding:0;">&times;</button>
        </div>
        <div class="panel-content">
            <div class="batch-control">
                <label>批量设置所有通道:</label>
                <div class="batch-control-row">
                    <input id="batch-min" type="number" placeholder="Min" style="width:60px;">
                    <input id="batch-max" type="number" placeholder="Max" style="width:60px;">
                    <button id="apply-batch-btn" style="font-size:12px;">全部应用</button>
                    <button id="reset-all-btn" style="font-size:12px; background:#6c757d;">重置自动</button>
                </div>
                <label style="margin-top:8px;">Batch color:</label>
                <div class="batch-control-row">
                    <select id="batch-color-preset" class="color-select">__BATCH_COLOR_OPTIONS__</select>
                    <input id="batch-color-custom" class="color-picker" type="color" value="__DEFAULT_COLOR__">
                    <span id="batch-color-preview" class="color-preview" style="background: __DEFAULT_COLOR__;"></span>
                    <button id="apply-batch-color-btn" style="font-size:12px;">Apply color</button>
                    <button id="reset-all-colors-btn" style="font-size:12px; background:#6c757d;">Default colors</button>
                </div>
            </div>
            <div id="channels-container">
                __CHANNEL_CONTROLS__
            </div>
        </div>
    </div>

    <div id="plot-wrapper">
        __FIG_HTML__
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
            var numChannels = __NUM_CHANNELS__;
            var gd = document.getElementById("edf_plot");

            if (!gd) {
                console.error("Missing plot container: edf_plot");
                return;
            }

            var settingsPanel = document.getElementById("settings-panel");
            var settingsToggle = document.getElementById("settings-toggle");
            var settingsClose = document.getElementById("settings-close");
            var applyBatchBtn = document.getElementById("apply-batch-btn");
            var resetAllBtn = document.getElementById("reset-all-btn");
            var batchMinInput = document.getElementById("batch-min");
            var batchMaxInput = document.getElementById("batch-max");
            var batchColorPreset = document.getElementById("batch-color-preset");
            var batchColorCustom = document.getElementById("batch-color-custom");
            var batchColorPreview = document.getElementById("batch-color-preview");
            var applyBatchColorBtn = document.getElementById("apply-batch-color-btn");
            var resetAllColorsBtn = document.getElementById("reset-all-colors-btn");

            var startSlider = document.getElementById("start-slider");
            var windowInput = document.getElementById("window-len");
            var timeStartInput = document.getElementById("time-start");
            var timeEndInput = document.getElementById("time-end");
            var applyTimeBtn = document.getElementById("apply-time-btn");
            var currentWindow = document.getElementById("current-window");
            var defaultColor = "__DEFAULT_COLOR__";
            var customColorValue = "__CUSTOM_COLOR_VALUE__";
            var presetColors = __COLOR_PRESETS_JSON__;

            var fullStartMs = null;
            var fullEndMs = null;
            var windowMs = null;

            function initFullRange() {
                if (!gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.xaxis.range) {
                    return false;
                }
                var xr = gd._fullLayout.xaxis.range;
                if (!Array.isArray(xr) || xr.length < 2) {
                    return false;
                }
                fullStartMs = new Date(xr[0]).getTime();
                fullEndMs = new Date(xr[1]).getTime();
                return Number.isFinite(fullStartMs) && Number.isFinite(fullEndMs) && fullEndMs > fullStartMs;
            }

            function normalizeWindowLength(totalMs) {
                var sec = parseFloat(windowInput.value);
                if (isNaN(sec) || sec <= 0) {
                    sec = 10;
                    windowInput.value = "10";
                }
                windowMs = Math.min(sec * 1000, totalMs);
            }

            function formatTime(ms) {
                return new Date(ms).toTimeString().slice(0, 8);
            }

            function updateFromSlider() {
                if ((fullStartMs === null || fullEndMs === null) && !initFullRange()) {
                    return;
                }

                var totalMs = fullEndMs - fullStartMs;
                if (totalMs <= 0) {
                    return;
                }

                normalizeWindowLength(totalMs);

                var percent = parseFloat(startSlider.value) / 100.0;
                if (isNaN(percent)) {
                    percent = 0;
                }

                var startMs = fullStartMs + totalMs * percent;
                var endMs = startMs + windowMs;

                if (endMs > fullEndMs) {
                    endMs = fullEndMs;
                    startMs = endMs - windowMs;
                }
                if (startMs < fullStartMs) {
                    startMs = fullStartMs;
                    endMs = startMs + windowMs;
                }

                Plotly.relayout(gd, { "xaxis.range": [new Date(startMs), new Date(endMs)] });

                var startText = formatTime(startMs);
                var endText = formatTime(endMs);
                currentWindow.textContent = "当前窗口: " + startText + " ~ " + endText;
                timeStartInput.value = startText;
                timeEndInput.value = endText;
            }

            function parseTimeToMs(text) {
                if ((fullStartMs === null || fullEndMs === null) && !initFullRange()) {
                    return null;
                }

                text = (text || "").trim();
                if (!text) {
                    return null;
                }

                var parts = text.split(":");
                if (parts.length < 2 || parts.length > 3) {
                    return null;
                }

                var hour = parseInt(parts[0], 10);
                var minute = parseInt(parts[1], 10);
                var second = parts.length === 3 ? parseInt(parts[2], 10) : 0;

                if (isNaN(hour) || isNaN(minute) || isNaN(second)) {
                    return null;
                }

                var base = new Date(fullStartMs);
                base.setHours(hour, minute, second, 0);
                return base.getTime();
            }

            function applyRangeFromInputs() {
                if ((fullStartMs === null || fullEndMs === null) && !initFullRange()) {
                    return;
                }

                var startMs = parseTimeToMs(timeStartInput.value);
                var endMs = parseTimeToMs(timeEndInput.value);

                if (startMs === null || endMs === null) {
                    alert("时间格式应为 HH:MM:SS，例如 12:34:56");
                    return;
                }
                if (startMs >= endMs) {
                    alert("起点时间必须小于终点时间");
                    return;
                }

                if (startMs < fullStartMs) {
                    startMs = fullStartMs;
                }
                if (endMs > fullEndMs) {
                    endMs = fullEndMs;
                }

                windowMs = endMs - startMs;
                windowInput.value = (windowMs / 1000).toFixed(3).replace(/\\.?0+$/, "");

                var totalMs = fullEndMs - fullStartMs;
                var startPercent = ((startMs - fullStartMs) / totalMs) * 100.0;
                startSlider.value = Math.max(0, Math.min(100, startPercent));

                Plotly.relayout(gd, { "xaxis.range": [new Date(startMs), new Date(endMs)] });

                var startText = formatTime(startMs);
                var endText = formatTime(endMs);
                currentWindow.textContent = "当前窗口: " + startText + " ~ " + endText;
                timeStartInput.value = startText;
                timeEndInput.value = endText;
            }

            function axisKey(index) {
                return index === 0 ? "yaxis" : "yaxis" + (index + 1);
            }

            function updateColorPreview(previewEl, color) {
                previewEl.style.backgroundColor = color;
            }

            function syncPresetSelect(selectEl, color) {
                var matched = false;
                for (var i = 0; i < presetColors.length; i += 1) {
                    if (presetColors[i].toLowerCase() === color.toLowerCase()) {
                        selectEl.value = presetColors[i];
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    selectEl.value = customColorValue;
                }
            }

            function syncChannelColorControls(index, color) {
                var selectEl = document.getElementById("color-preset-" + index);
                var colorEl = document.getElementById("color-custom-" + index);
                var previewEl = document.getElementById("color-preview-" + index);
                colorEl.value = color;
                syncPresetSelect(selectEl, color);
                updateColorPreview(previewEl, color);
            }

            function syncBatchColorControls(color) {
                batchColorCustom.value = color;
                syncPresetSelect(batchColorPreset, color);
                updateColorPreview(batchColorPreview, color);
            }

            function selectedColor(selectEl, colorEl) {
                if (selectEl.value !== customColorValue) {
                    colorEl.value = selectEl.value;
                }
                return colorEl.value;
            }

            function applySingleChannel(index) {
                var minInput = document.getElementById("y-min-" + index);
                var maxInput = document.getElementById("y-max-" + index);
                var minValue = minInput.value.trim();
                var maxValue = maxInput.value.trim();
                var axis = axisKey(index);
                var updates = {};

                if (!minValue && !maxValue) {
                    updates[axis + ".autorange"] = true;
                    Plotly.relayout(gd, updates);
                    return;
                }
                if (!minValue || !maxValue) {
                    alert("请同时填写最小值和最大值");
                    return;
                }

                var minNumber = parseFloat(minValue);
                var maxNumber = parseFloat(maxValue);
                if (isNaN(minNumber) || isNaN(maxNumber)) {
                    alert("量程必须是数字");
                    return;
                }
                if (minNumber >= maxNumber) {
                    alert("最小值必须小于最大值");
                    return;
                }

                updates[axis + ".range"] = [minNumber, maxNumber];
                updates[axis + ".autorange"] = false;
                Plotly.relayout(gd, updates);
            }

            function applySingleChannelColor(index) {
                var selectEl = document.getElementById("color-preset-" + index);
                var colorEl = document.getElementById("color-custom-" + index);
                var color = selectedColor(selectEl, colorEl);
                Plotly.restyle(gd, { "line.color": color }, [index]);
                syncChannelColorControls(index, color);
            }

            function applyBatch() {
                var minValue = batchMinInput.value.trim();
                var maxValue = batchMaxInput.value.trim();
                if (!minValue || !maxValue) {
                    alert("请先填写批量最小值和最大值");
                    return;
                }

                var minNumber = parseFloat(minValue);
                var maxNumber = parseFloat(maxValue);
                if (isNaN(minNumber) || isNaN(maxNumber) || minNumber >= maxNumber) {
                    alert("批量量程输入无效");
                    return;
                }

                var updates = {};
                for (var i = 0; i < numChannels; i += 1) {
                    document.getElementById("y-min-" + i).value = minNumber;
                    document.getElementById("y-max-" + i).value = maxNumber;
                    var axis = axisKey(i);
                    updates[axis + ".range"] = [minNumber, maxNumber];
                    updates[axis + ".autorange"] = false;
                }
                Plotly.relayout(gd, updates);
            }

            function applyBatchColor() {
                var color = selectedColor(batchColorPreset, batchColorCustom);
                var indices = [];
                for (var i = 0; i < numChannels; i += 1) {
                    indices.push(i);
                    syncChannelColorControls(i, color);
                }
                Plotly.restyle(gd, { "line.color": Array(numChannels).fill(color) }, indices);
                syncBatchColorControls(color);
            }

            function resetAll() {
                var updates = {};
                batchMinInput.value = "";
                batchMaxInput.value = "";

                for (var i = 0; i < numChannels; i += 1) {
                    document.getElementById("y-min-" + i).value = "";
                    document.getElementById("y-max-" + i).value = "";
                    updates[axisKey(i) + ".autorange"] = true;
                }
                Plotly.relayout(gd, updates);
            }

            function resetAllColors() {
                var indices = [];
                for (var i = 0; i < numChannels; i += 1) {
                    indices.push(i);
                    syncChannelColorControls(i, defaultColor);
                }
                Plotly.restyle(gd, { "line.color": Array(numChannels).fill(defaultColor) }, indices);
                syncBatchColorControls(defaultColor);
            }

            function bindColorInputs() {
                syncBatchColorControls(defaultColor);
                for (var i = 0; i < numChannels; i += 1) {
                    syncChannelColorControls(i, defaultColor);

                    (function(index) {
                        var selectEl = document.getElementById("color-preset-" + index);
                        var colorEl = document.getElementById("color-custom-" + index);
                        var previewEl = document.getElementById("color-preview-" + index);

                        selectEl.addEventListener("change", function() {
                            var color = selectedColor(selectEl, colorEl);
                            updateColorPreview(previewEl, color);
                        });
                        colorEl.addEventListener("input", function() {
                            syncChannelColorControls(index, colorEl.value);
                        });
                    })(i);
                }

                batchColorPreset.addEventListener("change", function() {
                    var color = selectedColor(batchColorPreset, batchColorCustom);
                    updateColorPreview(batchColorPreview, color);
                });
                batchColorCustom.addEventListener("input", function() {
                    syncBatchColorControls(batchColorCustom.value);
                });
            }

            function bindControls() {
                settingsToggle.addEventListener("click", function() {
                    settingsPanel.style.display = settingsPanel.style.display === "flex" ? "none" : "flex";
                });
                settingsClose.addEventListener("click", function() {
                    settingsPanel.style.display = "none";
                });
                applyBatchBtn.addEventListener("click", applyBatch);
                resetAllBtn.addEventListener("click", resetAll);
                applyBatchColorBtn.addEventListener("click", applyBatchColor);
                resetAllColorsBtn.addEventListener("click", resetAllColors);
                startSlider.addEventListener("input", updateFromSlider);
                windowInput.addEventListener("change", updateFromSlider);
                windowInput.addEventListener("keydown", function(event) {
                    if (event.key === "Enter") {
                        updateFromSlider();
                    }
                });
                applyTimeBtn.addEventListener("click", applyRangeFromInputs);
                timeStartInput.addEventListener("keydown", function(event) {
                    if (event.key === "Enter") {
                        applyRangeFromInputs();
                    }
                });
                timeEndInput.addEventListener("keydown", function(event) {
                    if (event.key === "Enter") {
                        applyRangeFromInputs();
                    }
                });
            }

            function waitForPlotReady() {
                if (initFullRange()) {
                    updateFromSlider();
                    bindColorInputs();
                    bindControls();
                } else {
                    setTimeout(waitForPlotReady, 100);
                }
            }

            waitForPlotReady();
            window.applySingleChannel = applySingleChannel;
            window.applySingleChannelColor = applySingleChannelColor;
        });
    </script>
</body>
</html>
"""


def parse_time_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%H:%M:%S.%f", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def inject_div_id(fig_html: str, div_id: str = "edf_plot") -> str:
    match = re.search(r'id="([^"]+)"', fig_html)
    if not match:
        raise RuntimeError("Could not find Plotly div id")
    return fig_html.replace(match.group(1), div_id)


def output_has_bound_plot(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        html = output_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return (
        bool(re.search(r'Plotly\.newPlot\(\s*"edf_plot"', html))
        and "applySingleChannelColor" in html
        and "batch-color-preset" in html
    )


def load_dataframe(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if TIME_COL not in df.columns:
        raise KeyError(f"Missing time column: {TIME_COL}")
    df[TIME_COL] = parse_time_column(df[TIME_COL])
    return df


def build_figure(df: pd.DataFrame, title: str):
    signal_cols = [col for col in df.columns if col != TIME_COL]
    if not signal_cols:
        raise ValueError("No signal columns found")

    fig = make_subplots(
        rows=len(signal_cols),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=signal_cols,
    )

    for index, col in enumerate(signal_cols, start=1):
        fig.add_trace(
            go.Scatter(
                x=df[TIME_COL],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(color=TRACE_COLOR, width=1.2),
            ),
            row=index,
            col=1,
        )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=100 * len(signal_cols) + 160,
    )
    return fig, signal_cols


def build_color_options(selected_color: str) -> str:
    options = []
    matched = False
    for label, value in COLOR_PRESETS:
        selected = ""
        if value.lower() == selected_color.lower():
            selected = " selected"
            matched = True
        options.append(f'<option value="{value}"{selected}>{label}</option>')

    custom_selected = "" if matched else " selected"
    options.append(f'<option value="{CUSTOM_COLOR_VALUE}"{custom_selected}>Custom</option>')
    return "".join(options)


def build_channel_controls(signal_cols) -> str:
    rows = []
    for index, col in enumerate(signal_cols):
        rows.append(
            """
        <div class="channel-row">
            <span class="ch-name" title="{title}">{label}</span>
            <input id="y-min-{index}" type="number" step="10" placeholder="Min">
            <span class="sep">~</span>
            <input id="y-max-{index}" type="number" step="10" placeholder="Max">
            <button class="apply-btn-small" onclick="applySingleChannel({index})">应用</button>
            <select id="color-preset-{index}" class="color-select">{color_options}</select>
            <input id="color-custom-{index}" class="color-picker" type="color" value="{default_color}">
            <span id="color-preview-{index}" class="color-preview" style="background: {default_color};"></span>
            <button class="apply-btn-small" onclick="applySingleChannelColor({index})">颜色</button>
        </div>
            """.format(
                title=escape(str(col)),
                label=escape(str(col)),
                index=index,
                color_options=build_color_options(TRACE_COLOR),
                default_color=TRACE_COLOR,
            )
        )
    return "".join(rows)


def build_page_html(fig_html: str, title: str, signal_cols) -> str:
    return (
        HTML_TEMPLATE.replace("__TITLE__", escape(title))
        .replace("__FIG_HTML__", fig_html)
        .replace("__CHANNEL_CONTROLS__", build_channel_controls(signal_cols))
        .replace("__NUM_CHANNELS__", str(len(signal_cols)))
        .replace("__BATCH_COLOR_OPTIONS__", build_color_options(TRACE_COLOR))
        .replace("__DEFAULT_COLOR__", TRACE_COLOR)
        .replace("__CUSTOM_COLOR_VALUE__", CUSTOM_COLOR_VALUE)
        .replace("__COLOR_PRESETS_JSON__", str([value for _, value in COLOR_PRESETS]).replace("'", '"'))
    )


def process_file(file_path: Path, overwrite: bool = False) -> str:
    file_path = file_path.resolve()

    if not TARGET_NAME_RE.match(file_path.name):
        return "skipped"

    output_path = file_path.with_name(f"{file_path.stem}_floating_multiY.html")
    if not overwrite and output_has_bound_plot(output_path):
        print(f"Skip (html exists): {output_path}")
        return "skipped"
    if output_path.exists():
        print(f"Rebuilding: {output_path}")

    try:
        df = load_dataframe(file_path)
        fig, signal_cols = build_figure(df, f"{file_path.stem} (点击右上角可调节通道量程)")
        fig_html = inject_div_id(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        output_path.write_text(
            build_page_html(fig_html, f"{file_path.stem} - 通道量程控制", signal_cols),
            encoding="utf-8",
        )
        print(f"Saved: {output_path}")
        return "processed"
    except Exception as exc:
        print(f"Failed to process {file_path}: {exc}")
        return "error"


def prompt_overwrite_mode() -> bool:
    print("请选择生成模式：")
    print("1. 全部覆盖：重新生成所有 HTML 文件（已有的会被覆盖）。")
    print("2. 跳过已有：只生成缺失的 HTML 文件。")
    while True:
        raw = input("请输入选项编号 (1/2): ").strip()
        choice = OVERWRITE_CHOICES.get(raw)
        if choice is not None:
            return choice
        print("输入无效，请输入 1 或 2。")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {DATA_DIR}")

    overwrite = prompt_overwrite_mode()

    processed = 0
    skipped = 0
    errors = 0

    for file_path in sorted(DATA_DIR.rglob("*.xlsx")):
        status = process_file(file_path, overwrite=overwrite)
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    print(
        "Fixed-duration EDF multi-range chart generation complete. "
        f"processed={processed}, skipped={skipped}, errors={errors}"
    )


if __name__ == "__main__":
    main()
