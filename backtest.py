"""周频因子评价与多因子回测。依赖 NumPy、pandas 和 Pillow。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from weekly_factors import FIELDS, get_calendar, get_listings, standardize
from figures import make_figures

HERE = Path(__file__).resolve().parent
FACTORS = list(FIELDS[2:])
MODELS = ["HS300", "Size_BP", "Size_BP_Corr"]
SELL_FEE = 0.003


def defaults(names):
    locations = [folder / name for folder in (HERE, HERE.parent) for name in names]
    return next((path for path in locations if path.exists()), locations[0])


def read_part(data, interface, key, fields):
    return pd.read_csv(data / interface / f"{key}.csv.gz", usecols=["ts_code", *fields])


def market_data(data, days, execution, codes):
    """每个成交日的复权 VWAP、估值、交易约束及可知的基准价格。"""
    closes = pd.concat(
        [pd.read_csv(path, usecols=["trade_date", "close"], dtype={"trade_date": str})
         for path in sorted((data / "index_daily").glob("000300.SH_*.csv.gz"))]
    ).drop_duplicates("trade_date").set_index("trade_date")["close"]
    last_close = pd.Series(np.nan, index=codes)
    tape = {}
    execution = set(execution)
    for day in days:
        daily = read_part(data, "daily", day, ["close", "vol", "amount", "high", "low"])
        adj = read_part(data, "adj_factor", day, ["adj_factor"])
        day_prices = daily.merge(adj, on="ts_code", validate="one_to_one").set_index("ts_code")
        day_prices = day_prices.loc[(day_prices.vol > 0) & (day_prices.adj_factor > 0)]
        adjusted_close = (day_prices.close * day_prices.adj_factor).reindex(codes)
        if day in execution:
            limits = read_part(data, "stk_limit", day, ["up_limit", "down_limit"])
            quotes = day_prices.join(limits.set_index("ts_code"), validate="one_to_one")
            vwap = (10 * quotes.amount / quotes.vol * quotes.adj_factor).where(quotes.amount > 0)
            vwap = vwap.reindex(codes)
            can_buy = ((quotes.low < quotes.up_limit) & (vwap.reindex(quotes.index) > 0))
            can_sell = ((quotes.high > quotes.down_limit) & (vwap.reindex(quotes.index) > 0))
            tape[day] = (
                vwap.to_numpy(),
                vwap.combine_first(last_close).to_numpy(),
                can_buy.reindex(codes, fill_value=False).to_numpy(),
                can_sell.reindex(codes, fill_value=False).to_numpy(),
            )
        last_close.update(adjusted_close)
    return tape, last_close.to_numpy(), closes


def hac_t(values, lag=4):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan
    centered = values - values.mean()
    variance = centered @ centered / n
    for step in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - step / (lag + 1)) * (centered[step:] @ centered[:-step]) / n
    return values.mean() / np.sqrt(variance / n) if variance > 0 else np.nan


def score_groups(frame, field):
    order = frame.sort_values(field, kind="stable").index.to_numpy()
    return np.array_split(order, 10)


def build_signals(data, signal_dates, factors, tape, execution, codes):
    weights, diagnostics = [], []
    names = [*MODELS[1:], *(f"{f}_Q{q}" for f in FACTORS for q in range(1, 11))]
    by_date = factors.groupby("trade_date", sort=True)
    for i, day in enumerate(signal_dates):
        panel = by_date.get_group(day).set_index("ts_code").sort_index()
        basic = read_part(data, "daily_basic", day, ["total_mv", "pb"]).set_index("ts_code")
        panel = panel.join(basic, validate="one_to_one")
        panel["Size"] = np.log(panel.total_mv.where(panel.total_mv > 0))
        panel["BP"] = 1 / panel.pb.where(panel.pb > 0)
        for factor in FACTORS:
            panel[factor] = standardize(panel[factor].to_numpy())

        target = pd.DataFrame(0.0, index=codes, columns=names)
        # 两模型在同一截面打分。小市值和高 BP 作为预先设定的风格方向。
        common = panel.dropna(subset=["Size", "BP", FACTORS[-1]]).copy()
        common["size_score"] = -standardize(common.Size.to_numpy())
        common["bp_score"] = standardize(common.BP.to_numpy())
        common["corr_score"] = standardize(common[FACTORS[-1]].to_numpy())
        common["base_score"] = (common.size_score + common.bp_score) / 2
        common["study_score"] = (common.size_score + common.bp_score + common.corr_score) / 3
        for model, field in zip(MODELS[1:], ("base_score", "study_score")):
            picks = score_groups(common, field)[-1]
            if len(picks) == 0:
                raise ValueError(f"{day} 共同股票池不足十只")
            target.loc[picks, model] = 1 / len(picks)
        for factor in FACTORS:
            for q, picks in enumerate(score_groups(panel, factor), start=1):
                target.loc[picks, f"{factor}_Q{q}"] = 1 / len(picks)
        weights.append(target.to_numpy())

        if i + 1 == len(execution):
            continue
        entry = pd.Series(tape[execution[i]][0], index=codes).reindex(panel.index)
        exit_price = pd.Series(tape[execution[i + 1]][0], index=codes).reindex(panel.index)
        panel["forward_return"] = exit_price / entry - 1
        labeled = panel.dropna(subset=["forward_return"])
        regression = labeled.dropna(subset=["Size", "BP"])
        y = regression.forward_return.to_numpy()
        for factor in FACTORS:
            rank_ic = labeled[factor].rank().corr(labeled.forward_return.rank())
            design = np.column_stack([
                np.ones(len(y)),
                standardize(regression.Size.to_numpy()),
                standardize(regression.BP.to_numpy()),
                standardize(regression[factor].to_numpy()),
            ])
            coeff = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coeff
            r_squared = 1 - (residual @ residual) / np.square(y - y.mean()).sum()
            diagnostics.append({
                "signal_date": day, "entry_date": execution[i], "exit_date": execution[i + 1],
                "factor": factor, "rank_ic": rank_ic, "n_labeled": len(labeled),
                "n_regression": len(regression), "coef": coeff[-1], "r_squared": r_squared,
            })
    return names, weights, pd.DataFrame(diagnostics)


def simulate(weights, quotes, closes, end, execution, listings, codes, columns):
    """成交失败保留现金或旧仓。名义份额按复权价格计价，初始现金为 1。"""
    cash = np.ones(len(columns))
    shares = np.zeros((len(codes), len(columns)))
    delists = np.array([listings.get(code, ("", "99991231"))[1] for code in codes])
    weekly = []
    last_trade_day = ""
    for day, target in zip(execution, weights):
        _, mark, buyable, sellable = quotes[day]
        mark = np.nan_to_num(mark, nan=0)
        retired = (delists > last_trade_day) & (delists <= day)
        cash += (shares[retired] * mark[retired, None]).sum(axis=0)
        shares[retired] = 0
        value = shares * mark[:, None]
        opening_nav = cash + value.sum(axis=0)
        weekly.append(opening_nav)
        target_value = target * opening_nav
        sells = np.maximum(value - target_value, 0) * sellable[:, None]
        cash += sells.sum(axis=0) * (1 - SELL_FEE)
        remaining = value - sells
        asks = np.maximum(target_value - remaining, 0) * buyable[:, None]
        gross = asks.sum(axis=0)
        budget = np.divide(cash, gross, out=np.ones_like(cash), where=gross > 0)
        buys = asks * np.clip(budget, 0, 1)
        cash -= buys.sum(axis=0)
        share_change = np.divide(buys - sells, mark[:, None], out=np.zeros_like(shares),
                                 where=mark[:, None] > 0)
        shares += share_change
        last_trade_day = day
    # 期末盯市，不模拟样本外调仓或强制清仓。
    final_mark = np.nan_to_num(closes, nan=0)
    weekly.append(cash + (shares * final_mark[:, None]).sum(axis=0))
    return pd.DataFrame(weekly, index=pd.Index([*execution, end], name="date"), columns=columns)


def series_stats(returns, years):
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    vol = returns.std(ddof=1) * np.sqrt(52)
    return {
        "annual_return": wealth.iloc[-1] ** (1 / years) - 1,
        "annual_vol": vol,
        "sharpe_rf0": returns.mean() * 52 / vol if vol > 0 else np.nan,
        "max_drawdown": drawdown.min(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=defaults(["Tushare数据", "Tushare- data"]))
    parser.add_argument("--factors", type=Path, default=defaults(["weekly_factors.csv"]))
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()

    factors = pd.read_csv(args.factors, dtype={"trade_date": str, "ts_code": str})
    if factors.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("因子文件有重复的股票日期")
    days = get_calendar(args.data_dir)
    signal_dates = sorted(factors.trade_date.unique())
    loc = {day: i for i, day in enumerate(days)}
    end = signal_dates[-1]
    # 若终点本身是下一成交日，则该周仅做期末盯市。
    signal_dates = [day for day in signal_dates if loc[day] + 1 < loc[end]]
    execution = [days[loc[day] + 1] for day in signal_dates]
    if len(execution) < 3:
        raise ValueError("需要至少三个完整调仓日")
    codes = pd.Index(sorted(factors.ts_code.unique()))
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(execution)} 个成交日，载入日线与复权价格", flush=True)
    tape, final_close, index_close = market_data(
        args.data_dir, days[loc[execution[0]]:loc[end] + 1], execution, codes
    )
    names, weights, ic = build_signals(args.data_dir, signal_dates, factors, tape, execution, codes)
    nav = simulate(weights, tape, final_close, end, execution, get_listings(args.data_dir), codes, names)
    benchmark = index_close.loc[nav.index]
    nav.insert(0, "HS300", (benchmark / benchmark.iloc[0]).to_numpy())
    returns = nav.pct_change().iloc[1:]
    active = pd.DataFrame({
        "base_minus_HS300": returns.Size_BP - returns.HS300,
        "study_minus_HS300": returns.Size_BP_Corr - returns.HS300,
        "study_minus_base": returns.Size_BP_Corr - returns.Size_BP,
    })
    years = (pd.Timestamp(end) - pd.Timestamp(execution[0])).days / 365.25
    metrics = pd.DataFrame({model: series_stats(returns[model], years) for model in MODELS}).T
    metrics["annual_excess_vs_HS300"] = metrics.annual_return - metrics.loc["HS300", "annual_return"]
    metrics["annual_excess_vs_base"] = metrics.annual_return - metrics.loc["Size_BP", "annual_return"]
    yearly = (1 + returns[MODELS]).groupby(returns.index.str[:4]).prod() - 1
    yearly["study_minus_base"] = yearly.Size_BP_Corr - yearly.Size_BP

    single, deciles = [], []
    for factor, sliced in ic.groupby("factor", sort=False):
        spread = returns[f"{factor}_Q10"] - returns[f"{factor}_Q1"]
        single.append({
            "factor": factor, "weeks": len(sliced), "mean_rank_ic": sliced.rank_ic.mean(),
            "sd_rank_ic": sliced.rank_ic.std(), "positive_ic": (sliced.rank_ic > 0).mean(),
            "ic_hac_t": hac_t(sliced.rank_ic), "mean_ols_coef": sliced.coef.mean(),
            "ols_hac_t": hac_t(sliced.coef), "mean_r_squared": sliced.r_squared.mean(),
            "q10_annual": series_stats(returns[f"{factor}_Q10"], years)["annual_return"],
            "spread_mean_week": spread.mean(), "spread_hac_t": hac_t(spread),
        })
        for q in range(1, 11):
            deciles.append({"factor": factor, "decile": q, **series_stats(returns[f"{factor}_Q{q}"], years)})
    single = pd.DataFrame(single)
    deciles = pd.DataFrame(deciles)
    active_summary = pd.DataFrame({
        col: {"mean_week": active[col].mean(), "hac_t": hac_t(active[col])} for col in active
    }).T
    weekly = nav[MODELS].join(returns[MODELS].add_suffix("_return")).join(active)

    for filename, table, include_index in [
        ("factor_summary", single, False), ("factor_weekly", ic, False), ("deciles", deciles, False),
        ("model_summary", metrics, True), ("model_weekly", weekly, True),
        ("model_yearly", yearly, True), ("active_summary", active_summary, True),
    ]:
        table.to_csv(args.output / f"{filename}.csv", index=include_index, encoding="utf-8-sig")
    make_figures(args.output, nav[MODELS], ic, deciles)
    print(single.to_string(index=False), metrics.to_string(), yearly.to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
