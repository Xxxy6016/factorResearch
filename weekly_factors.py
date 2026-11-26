"""从本地 Tushare 日线计算《基于隔夜相关性的因子研究》的周频因子。

课程设定：默认用 60 个交易日的 Pearson 相关系数，也可用 --window 21。
参加相关性计算的股票在信号日满足股票池筛选，且整个窗口的收益完整、
有波动。涨跌停筛选按信号日收盘价判断。每个子因子先在当期截面做
3 倍 MAD 去极值（以 1.4826 缩放 MAD）和 Z-Score，再以等权合成。
CSV 保存两个原始子因子及合成值。收益标签与实际调仓由后续回测计算。

运行：python3 weekly_factors.py
依赖：NumPy（当前 Python 缺少时运行 python3 -m pip install numpy）
"""

import argparse
import csv
import gzip
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


HERE = Path(__file__).resolve().parent
TOP_COUNT = 50
FIELDS = (
    "trade_date",
    "ts_code",
    "factor_top50_allday",
    "factor_small50_corr_by_lag",
    "corr_combined1",
)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def calendar_date(value):
    return datetime.strptime(value.replace("-", ""), "%Y%m%d").date()


def parse_date(value):
    return calendar_date(value).strftime("%Y%m%d")


def get_calendar(data_dir):
    days = {
        row["trade_date"]
        for path in (data_dir / "index_daily").glob("000300.SH_*.csv.gz")
        for row in read_rows(path)
    }
    if not days:
        raise ValueError("找不到沪深 300 日线，无法确定交易日。")
    return sorted(days)


def get_listings(data_dir):
    listings = {}
    for status in ("L", "D", "P"):
        for row in read_rows(data_dir / "stock_basic" / f"{status}.csv.gz"):
            listed = row["list_date"]
            if not listed or not row["ts_code"].endswith((".SH", ".SZ", ".BJ")):
                continue
            day = calendar_date(listed)
            try:
                anniversary = day.replace(year=day.year + 1)
            except ValueError:
                anniversary = day.replace(year=day.year + 1, day=28)
            listings[row["ts_code"]] = (
                anniversary.strftime("%Y%m%d"),
                row["delist_date"] or "99991231",
            )
    return listings


def number(value):
    return float(value) if value else np.nan


def get_prices(data_dir, day, stock_index):
    factors = {
        row["ts_code"]: number(row["adj_factor"])
        for row in read_rows(data_dir / "adj_factor" / f"{day}.csv.gz")
    }
    prices = np.full((3, len(stock_index)), np.nan)
    for row in read_rows(data_dir / "daily" / f"{day}.csv.gz"):
        index = stock_index.get(row["ts_code"])
        if index is None:
            continue
        opening, closing, volume = (number(row[key]) for key in ("open", "close", "vol"))
        adjustment = factors.get(row["ts_code"], np.nan)
        if all(np.isfinite(value) and value > 0 for value in (
            opening, closing, volume, adjustment
        )):
            prices[:, index] = opening * adjustment, closing * adjustment, closing
    return prices


def eligible_stocks(data_dir, day, listings, stock_index, raw_close):
    st_codes = {
        row["ts_code"] for row in read_rows(data_dir / "stock_st" / f"{day}.csv.gz")
    }
    eligible = np.zeros(len(stock_index), dtype=bool)
    for row in read_rows(data_dir / "stk_limit" / f"{day}.csv.gz"):
        code = row["ts_code"]
        index = stock_index.get(code)
        if index is not None and code not in st_codes:
            anniversary, delisted = listings[code]
            up, down = number(row["up_limit"]), number(row["down_limit"])
            eligible[index] = (
                anniversary <= day < delisted
                and down < raw_close[index] < up
            )
    return eligible


def calculate_factors(history, eligible):
    """history 的三列收益类型依次为全天、日间、隔夜，按交易日排列。"""
    valid = eligible & np.isfinite(history).all(axis=(0, 1))
    indices = np.flatnonzero(valid)
    values = history[:, :, indices].transpose(1, 2, 0)
    centered = values - values.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(centered, axis=2)
    varying = (norms > 1e-12).all(axis=0)
    indices = indices[varying]
    if len(indices) <= TOP_COUNT:
        raise ValueError(f"仅 {len(indices)} 只有效股票，每只至少需要 {TOP_COUNT} 只其他股票。")

    full_day, intraday, overnight = centered[:, varying] / norms[:, varying, None]
    top = np.empty(len(indices))
    screened = np.empty(len(indices))
    for start in range(0, len(indices), 256):
        stop = min(start + 256, len(indices))
        group = slice(start, stop)
        row = np.arange(stop - start)
        own_stock = start + row
        daily_corr = full_day[group] @ full_day.T
        lag_corr = intraday[group] @ overnight.T
        daily_corr[row, own_stock] = -np.inf
        lag_corr[row, own_stock] = np.inf
        top_cols = np.argpartition(-daily_corr, TOP_COUNT - 1, axis=1)[:, :TOP_COUNT]
        bottom_cols = np.argpartition(lag_corr, TOP_COUNT - 1, axis=1)[:, :TOP_COUNT]
        top[group] = np.take_along_axis(daily_corr, top_cols, axis=1).mean(axis=1)
        screened[group] = np.take_along_axis(daily_corr, bottom_cols, axis=1).mean(axis=1)
    combined = (standardize(top) + standardize(screened)) / 2
    return indices, top, screened, combined


def standardize(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        width = 3 * 1.4826 * mad
        values = np.clip(values, median - width, median + width)
    scale = values.std()
    return (values - values.mean()) / scale if scale > 0 else np.zeros_like(values)


def export_weeks(data_dir, start, end, window, output):
    days = get_calendar(data_dir)
    if start < days[0] or end > days[-1]:
        raise ValueError(f"本地交易日历覆盖 {days[0]} 至 {days[-1]}，请指定范围内日期。")
    week_ends = {}
    for day in days:
        year, week, _ = calendar_date(day).isocalendar()
        week_ends[(year, week)] = day
    signals = [day for day in week_ends.values() if start <= day <= end]
    if not signals:
        raise ValueError("所选区间没有周末信号日。")
    first = days.index(signals[0])
    if first < window:
        raise ValueError(f"首个信号日 {signals[0]} 以前的预热数据不足 {window} 个交易日。")

    listings = get_listings(data_dir)
    codes = sorted(listings)
    stock_index = {code: index for index, code in enumerate(codes)}
    history = deque(maxlen=window)
    begin = first - window
    stop = days.index(signals[-1])
    _, previous_close, _ = get_prices(data_dir, days[begin], stock_index)
    signal_set = set(signals)
    rows = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=output.parent,
        prefix=output.name + ".", suffix=".tmp", delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            writer = csv.writer(temporary)
            writer.writerow(FIELDS)
            for day in days[begin + 1:stop + 1]:
                opening, closing, raw_close = get_prices(data_dir, day, stock_index)
                history.append(np.array((
                    closing / previous_close - 1,
                    closing / opening - 1,
                    opening / previous_close - 1,
                )))
                previous_close = closing
                if day not in signal_set:
                    continue
                eligible = eligible_stocks(data_dir, day, listings, stock_index, raw_close)
                try:
                    indices, top, screened, combined = calculate_factors(np.asarray(history), eligible)
                except ValueError as exc:
                    raise ValueError(f"{day}：{exc}") from exc
                writer.writerows(
                    (day, codes[index], f"{a:.10g}", f"{b:.10g}", f"{c:.10g}")
                    for index, a, b, c in zip(indices, top, screened, combined)
                )
                rows += len(indices)
                print(f"{day}：{len(indices):,} 只股票", flush=True)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"已导出 {len(signals)} 周、{rows:,} 行：{output}")


def main():
    parser = argparse.ArgumentParser(description="计算并导出全 A 股周频隔夜相关性因子")
    parser.add_argument("--data-dir", type=Path, default=HERE / "Tushare数据")
    parser.add_argument("--start", type=parse_date, default="20230101")
    parser.add_argument("--end", type=parse_date, default="20251231")
    parser.add_argument("--window", type=int, choices=(21, 60), default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = HERE / ("weekly_factors.csv" if args.window == 60 else "weekly_factors_21d.csv")
    try:
        export_weeks(args.data_dir, args.start, args.end, args.window, output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"导出失败：{exc}\n")


if __name__ == "__main__":
    main()
