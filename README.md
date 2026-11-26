# 基于隔夜相关性的 A 股选股因子复刻

复刻广发证券《基于隔夜相关性的因子研究》第四节的全天相关因子、隔夜筛选因子和合成因子。使用 2022 年 9 月起的 Tushare 日线预热，在 2023—2025 年进行周频检验和回测。

## 复现

安装 Python 3.9 及以上，然后在项目目录执行。以下例子使用当前课程目录的数据布局。

```sh
python3 -m pip install -r requirements.txt
python3 weekly_factors.py --data-dir "../Tushare- data"
python3 backtest.py --data-dir "../Tushare- data" --factors weekly_factors.csv
```

已有周频因子 CSV 时可直接运行第二个脚本。原始数据目录需要包含 `daily`、`adj_factor`、`daily_basic`、`stock_basic`、`stock_st`、`stk_limit`、`index_daily` 分片。脚本读取本地数据，不需要 Token。

21 日窗口对照：

```sh
python3 weekly_factors.py --data-dir "../Tushare- data" --window 21
python3 backtest.py --data-dir "../Tushare- data" --factors weekly_factors_21d.csv --output results/window_21
```

## 结果

- `factor_summary.csv` 和 `factor_weekly.csv` 包含 Rank IC 和控制 Size、BP 的每周截面回归。
- `deciles.csv` 汇总三个因子的十组净交易成本收益和风险指标，Q10 分值最高。
- `model_summary.csv`、`model_weekly.csv` 和 `model_yearly.csv` 包含沪深 300、基准模型和研究模型的净值及收益指标。
- `active_summary.csv` 报告周超额均值和 4 阶 Newey–West t 值。程序的年化超额为两个组合几何年化收益率之差。
- PNG 图展示累计 Rank IC、十分组年化收益、三条净值。

信号在每周末收盘后确定，组合在下一交易日按复权均价成交。仅卖出收取 0.3% 的单边费用。全部因子值用于排序，Rank IC 只用能得到入场及出场 VWAP 的股票。周频模型结果包含最后一次成交至样本终点的短区间。

[原研报微信链接](https://mp.weixin.qq.com/s?__biz=MzA4MDE0MDMwOA%3D%3D&mid=2650802441&idx=1&sn=48c7f4477884729f81fd1db7c3236963)
