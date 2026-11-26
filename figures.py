"""绘制供论文引用的三张静态图。"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = ["#6B7280", "#1D4ED8", "#B45309"]
MARGIN = (134, 75, 1135, 495)
LABELS = ["全天相关", "隔夜筛选", "合成因子"]


def chart_font(size):
    candidates = [
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for file in candidates:
        if file.is_file():
            return ImageFont.truetype(str(file), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def create_axes(series, title, extremes, tick_labels, formatter):
    canvas = Image.new("RGB", (1200, 600), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((134, 20), title, fill="#14213D", font=chart_font(31))
    left, top, right, bottom = MARGIN
    low, high = extremes
    def locate(x, y):
        return left + x * (right - left), bottom - (y - low) / (high - low) * (bottom - top)

    for value in np.linspace(low, high, 6):
        _, y = locate(0, value)
        draw.line((left, y, right, y), fill="#DCE3EB", width=1)
        draw.text((left - 15, y), formatter(value), fill="#333D4F", font=chart_font(19), anchor="rm")
    if low < 0 < high:
        _, zero = locate(0, 0)
        draw.line((left, zero, right, zero), fill="#93A0B2", width=2)
    draw.line((left, top, left, bottom, right, bottom), fill="#8793A2", width=2)
    for x_value, label in tick_labels:
        x, _ = locate(x_value, low)
        draw.text((x, bottom + 15), label, fill="#333D4F", font=chart_font(19), anchor="mt")

    for pos, ((label, _), color) in enumerate(zip(series.items(), PALETTE)):
        x = 149 + pos * 260
        draw.line((x, 566, x + 26, 566), fill=color, width=5)
        draw.text((x + 36, 566), label, fill="#14213D", font=chart_font(21), anchor="lm")
    return canvas, draw, locate


def line_chart(path, series, title, labels):
    arrays = [np.asarray(values) for values in series.values()]
    low = min(array.min() for array in arrays)
    high = max(array.max() for array in arrays)
    padding = max((high - low) * .09, .01)
    canvas, draw, locate = create_axes(
        series, title, (low - padding, high + padding), labels, lambda value: f"{value:.2f}"
    )
    for array, color in zip(arrays, PALETTE):
        points = [locate(x, y) for x, y in zip(np.linspace(0, 1, len(array)), array)]
        draw.line(points, fill=color, width=4, joint="curve")
    canvas.save(path)


def date_ticks(dates):
    indices = np.linspace(0, len(dates) - 1, 6, dtype=int)
    return [(i / (len(dates) - 1), f"{dates[i][:4]}-{dates[i][4:6]}") for i in indices]


def make_figures(output, nav, ic, deciles):
    line_chart(
        output / "net_value.png",
        {"沪深 300": nav.HS300, "基准模型": nav.Size_BP, "研究模型": nav.Size_BP_Corr},
        "周频多因子组合净值", date_ticks(nav.index),
    )
    groups = list(ic.groupby("factor", sort=False))
    line_chart(
        output / "rank_ic.png",
        {label: group.rank_ic.cumsum() for label, (_, group) in zip(LABELS, groups)},
        "单因子累计 Rank IC", date_ticks(groups[0][1].signal_date.tolist()),
    )
    series = dict(zip(LABELS, (
        group.annual_return.to_numpy() for _, group in deciles.groupby("factor", sort=False)
    )))
    bounds = np.array(list(series.values()))
    canvas, draw, locate = create_axes(
        series, "十分组净交易成本年化收益率",
        (min(0, bounds.min() - .025), max(0, bounds.max() + .025)),
        [((q + .5) / 10, f"Q{q+1}") for q in range(10)], lambda value: f"{value:.0%}",
    )
    for column, (values, color) in enumerate(zip(series.values(), PALETTE)):
        for q, value in enumerate(values):
            center = (q + .5) / 10 + (column - 1) * .018
            x, zero = locate(center, 0)
            _, y = locate(center, value)
            draw.rectangle((x - 8, min(y, zero), x + 8, max(y, zero)), fill=color)
    canvas.save(output / "decile_returns.png")
