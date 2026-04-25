# NVDL Grid Backtest

本目录是一个本地 NVDL 网格策略回测环境。

## 目标
1.维持定期收割
2.核心目标是实现在中间震荡区间反复网格交易获得收益，在超额上涨区间逐渐减仓，预防暴跌，在超跌区间维持两个平衡：控制超跌区间手中资产的降低幅度的同时，持续加仓

## 当前策略

- 标的：`NVDL`
- 数据：`1h` 小时 K 线
- 参数文件：`config/default.yaml`
- 初始资金：`10000`
- 初始买入：`200` 股
- 网格步长：`2` 美元
- 手续费：`0.1%`
- 允许碎股：最小 `0.001` 股
- 仅限支持碎股交易的券商假设
- 初始买入在第一根小时 K 线执行，使用 `Close` 价格
- 交易信号只使用每根小时 K 线的 `Close` 价格，不模拟盘中 `High/Low` 触发顺序
- 每根小时 K 线最多交易一次；如果触发买入，买入后直接结束本根 K 线判断
- 交易后，网格基准价重置为当前成交价
- 每月收割：每月最多一次，如果现金超过 `5000`，从账户中取出 `500`
- 收割优先选现金处于近 `5` 个交易日最高值的时点
- 如果本月没有触发现金高点收割，则在本月最后一个交易日收割
- 每月补贴：现金低于 `100` 时，如果当月还没补贴过，向账户补贴 `1000`
- 加码买入：触发买入时，如果现金超过 `5000`，买入数量增加 `5` 股

趋势判断使用日线过去 1 年均线：

- 日线均线窗口：`252` 个交易日
- 小时线每根 K 线按当天对应的日线 1 年均线判断在均线上方或下方

## 买卖规则

设 1 年日均线价格为 `x`。

| 价格区间 | 买入触发 | 买入数量 | 卖出触发 | 卖出数量 | 最低持仓 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `price > 1.25x` | 下跌 `2` 美元 | `15` 股 | 上涨 `2` 美元 | `20` 股 | `80` 股 |
| `0.50x <= price <= 1.25x` | 下跌 `2` 美元 | `20` 股 | 上涨 `2` 美元 | `15` 股 | `120` 股 |
| `price < 0.50x` | 下跌 `3` 美元 | `15` 股 | 上涨 `2` 美元 | `10` 股 | `120` 股 |

## 执行顺序和假设

- 每根小时 K 线使用 `Close` 作为信号判断价格和成交价格。
- 不使用小时 K 线的 `High` / `Low` 模拟盘中触发。
- 初始买入在第一根小时 K 线执行，使用该 K 线的 `Close` 价格，并扣除 `0.1%` 手续费。
- 如果现金不足以买入完整的初始 `200` 股，则按可买的最大碎股数量买入。
- 初始 `grid_base` 等于初始买入时的 `Close` 价格。
- 网格触发只和 `grid_base` 比较，不和上一根 K 线价格或均线价格比较。
- 即使一根小时 K 线跨过多个网格，也最多只执行一笔交易。
- 买入判断优先于卖出判断；如果同一根 K 线满足买入条件，执行买入后不再检查卖出。
- 买入时如果现金超过 `5000`，买入数量增加 `5` 股。
- 买入允许部分成交：如果现金不足以买入目标股数，则按现金可买的最大碎股数量买入。
- 卖出允许部分成交，但成交后持仓不得低于对应区间的最低持仓。
- 如果可买或可卖数量小于 `0.001` 股，则跳过该笔交易。
- 成交后，`grid_base` 重置为本次成交价格。
- 手续费按成交金额的 `0.1%` 计算，买入时从现金扣除，卖出时从卖出收入扣除。

每根小时 K 线按下面顺序执行：

1. 读取当前小时 K 线 `Close` 价格。
2. 读取当前日期对应的日线 `252` 日均线。
3. 先执行月度收割规则。
4. 再执行月度补贴规则。
5. 如果还没有初始买入，则执行初始买入并设置 `grid_base`。
6. 根据价格和 1 年日均线判断区间。
7. 检查是否触发买入；如触发，则执行买入并更新 `grid_base`。
8. 如果没有买入，再检查是否触发卖出；如触发，则执行卖出并更新 `grid_base`。
9. 记录资产曲线。

均线对齐方式：

- 日线均线使用 `data/NVDL.csv` 中的 `Close` 计算。
- 小时 K 线使用其日期对应的日线 `252` 日均线。
- 当前实现会使用同一日期的日线收盘价参与当日均线计算，这是回测简化假设。

## 数据文件

- `data/NVDL.csv`：日线数据
- `data/NVDL_1h.csv`：小时线数据
- `data/NVDL_1m.csv`：分钟线数据，仅 Yahoo 可提供的最近几天

Yahoo 当前可用范围：

- 日线：`2023-01-03` 至今
- 小时线：`2024-04-25` 至今
- 分钟线：最近几天

## 输出文件

- `logs/NVDL_trade_log.csv`：对齐后的交易日志，适合直接用文本方式查看
- `plots/NVDL_1h_equity.png`：价格、买卖点、现金/持仓市值、回撤图
- `plots/NVDL_1h_backtest.html`：Backtesting.py 交互图
- `results/parameter_sweep_results.csv`：参数实验结果表
- `reports/top_strategies.md`：参数实验排名报告

## 当前回测结果

回测区间：

- `2024-04-25` 到 `2026-04-24`

结果：

- Final Account Equity：`25868.10`
- Real Final Assets：`36368.10`
- Subsidized Cash：`8000.00`
- Total Invested Cash：`18000.00`
- Profit On Invested Cash：`18368.10`
- Return On Invested Cash：`102.04%`
- Return On Initial Cash：`263.68%`
- Benchmark Initial Cash Assets：`30960.32`
- Benchmark Invested Cash Assets：`55728.57`
- Buy & Hold Return：`209.60%`
- Max Drawdown：`-53.53%`
- Real Asset Max Drawdown：`-32.93%`
- Benchmark Initial Max Drawdown：`-69.89%`
- Trades：`492`
- Win Rate：`55.89%`
- Commissions：`711.23`
- Final Cash：`10126.74`
- Final Shares：`153.698`
- Final Position Value：`15707.94`
- Realized P/L：`15110.99`
- Harvested Cash：`10500.00`
- Net External Cash Flow：`+2500.00`

说明：

- `Final Account Equity = Final Cash + Final Shares * Final Price`
- `Real Final Assets = Final Account Equity + Harvested Cash`
- `Total Invested Cash = Initial Cash + Subsidized Cash`
- `Profit On Invested Cash = Real Final Assets - Total Invested Cash`
- `Return On Invested Cash = Profit On Invested Cash / Total Invested Cash`
- `Return On Initial Cash = (Real Final Assets - Initial Cash) / Initial Cash`
- `Net External Cash Flow = Harvested Cash - Subsidized Cash`
- `Benchmark Initial Cash Assets` 表示只用初始 `10000` 买入并持有到期的资产金额
- `Benchmark Invested Cash Assets` 表示假设从起点一次性投入 `18000` 买入并持有到期的资产金额

## 运行

```bash
python -m pip install -r requirements.txt
python playground.py
```

修改策略参数时，优先改 `config/default.yaml`，不要直接改 `playground.py` 顶部常量。

## 参数实验

```bash
python run_experiments.py
```

实验参数写在 `config/experiments.yaml`。当前版本做单因素测试：每次只改变一个参数，其他参数保持 `config/default.yaml` 的 baseline。

输出：

- `results/parameter_sweep_results.csv`
- `reports/top_strategies.md`

当前 sweep 的 baseline：

- Score：`65.49`
- Return On Invested Cash：`102.04%`
- Real Asset Max Drawdown：`-32.93%`
- Subsidized Cash：`8000.00`
- Harvested Cash：`10500.00`

当前 sweep 分数最高的单因素变体：

- `zones.mid.sell_shares=10`
- Score：`80.60`
- Return On Invested Cash：`118.68%`
- Real Asset Max Drawdown：`-35.91%`
- Subsidized Cash：`10000.00`
- Harvested Cash：`9000.00`
