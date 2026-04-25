from pathlib import Path

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from backtesting import Strategy
from backtesting.lib import FractionalBacktest


SYMBOL = "NVDL"
START = "2023-01-01"
END = None
CASH = 10_000
COMMISSION = 0.001
SHOW_PLOT = True
DATA_DIR = Path("data")
LOG_DIR = Path("logs")
PLOT_DIR = Path("plots")
DATA_INTERVAL = "1h"
MA_DAYS = 252
INITIAL_SHARES = 200
GRID_STEP = 2
DEEP_GRID_STEP = 3
FRACTIONAL_UNIT = 0.001
MID_BUY_SHARES = 20
MID_SELL_SHARES = 15
MID_MIN_SHARES = 120
HIGH_BUY_SHARES = 15
HIGH_SELL_SHARES = 20
HIGH_MIN_SHARES = 80
DEEP_BUY_SHARES = 15
DEEP_SELL_SHARES = 10
DEEP_MIN_SHARES = 120
HIGH_PRICE_MULTIPLE = 1.25
DEEP_LOWER_MULTIPLE = 0.50
HARVEST_CASH_THRESHOLD = 5_000
HARVEST_AMOUNT = 500
HARVEST_LOOKBACK_DAYS = 5
SUBSIDY_AMOUNT = 1_000
SUBSIDY_CASH_THRESHOLD = 100
EXTRA_BUY_CASH_THRESHOLD = 5_000
EXTRA_BUY_SHARES = 5


def SMA(values, n):
    return pd.Series(values).rolling(n).mean()


def load_data(symbol: str, start: str, end: str):
    DATA_DIR.mkdir(exist_ok=True)
    data_file = DATA_DIR / f"{symbol}_{DATA_INTERVAL}.csv" if DATA_INTERVAL else DATA_DIR / f"{symbol}.csv"

    if data_file.exists():
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        data = yf.download(symbol, start=start, end=end, auto_adjust=False)
        if data.empty:
            raise RuntimeError(f"No data returned for {symbol}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.to_csv(data_file)

    return data[["Open", "High", "Low", "Close", "Volume"]]


def load_daily_data(symbol: str, start: str, end: str):
    data_file = DATA_DIR / f"{symbol}.csv"
    if data_file.exists():
        daily = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        daily = yf.download(symbol, start=start, end=end, auto_adjust=False)
        if daily.empty:
            raise RuntimeError(f"No daily data returned for {symbol}")
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        daily = daily[["Open", "High", "Low", "Close", "Volume"]]
        daily.to_csv(data_file)

    return daily[["Open", "High", "Low", "Close", "Volume"]]


def format_table(df):
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    headers = list(df.columns)
    widths = []

    for index, header in enumerate(headers):
        values = [row[index] for row in rows]
        widths.append(max(len(header), *(len(value) for value in values)))

    lines = ["  ".join(header.ljust(widths[index])
                       for index, header in enumerate(headers))]
    lines.extend("  ".join(value.ljust(widths[index])
                           for index, value in enumerate(row))
                 for row in rows)
    return "\n".join(lines)


class GridByMA(Strategy):
    def init(self):
        self.grid_price = None
        self.initialized = False
        self.planned_buys = 0
        self.executed_buys = 0
        self.skipped_buys = 0
        self.planned_sells = 0
        self.executed_sells = 0
        self.skipped_sells = 0
        self.cash_budget = 0
        self.ledger_cash = CASH
        self.ledger_shares = 0.0
        self.ledger_cost_basis = 0.0
        self.ledger_realized_pnl = 0.0
        self.ledger_harvested = 0.0
        self.ledger_subsidized = 0.0
        self.harvested_months = set()
        self.subsidized_months = set()
        self.daily_cash_highs = {}
        self.trade_log = []
        self.net_worth_log = []

    def actual_price(self):
        return self.data.Close[-1] / FRACTIONAL_UNIT

    def actual_ma20(self):
        timestamp = pd.Timestamp(self.data.index[-1])
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        date = timestamp.normalize()
        return DAILY_MA_BY_DATE.get(date)

    def order_size(self, shares):
        return int(shares / FRACTIONAL_UNIT)

    def trading_regime(self, price, ma):
        if ma is None:
            return {
                "buy_step": GRID_STEP,
                "buy_shares": MID_BUY_SHARES,
                "sell_step": GRID_STEP,
                "sell_shares": MID_SELL_SHARES,
                "min_shares": 0,
            }
        if price > ma * HIGH_PRICE_MULTIPLE:
            return {
                "buy_step": GRID_STEP,
                "buy_shares": HIGH_BUY_SHARES,
                "sell_step": GRID_STEP,
                "sell_shares": HIGH_SELL_SHARES,
                "min_shares": HIGH_MIN_SHARES,
            }
        if price >= ma * DEEP_LOWER_MULTIPLE:
            return {
                "buy_step": GRID_STEP,
                "buy_shares": MID_BUY_SHARES,
                "sell_step": GRID_STEP,
                "sell_shares": MID_SELL_SHARES,
                "min_shares": MID_MIN_SHARES,
            }
        return {
            "buy_step": DEEP_GRID_STEP,
            "buy_shares": DEEP_BUY_SHARES,
            "sell_step": GRID_STEP,
            "sell_shares": DEEP_SELL_SHARES,
            "min_shares": DEEP_MIN_SHARES,
        }

    def log_trade(self, side, shares, price, amount, fee, realized_pnl=0.0):
        position_value = self.ledger_shares * price
        unrealized_pnl = position_value - self.ledger_cost_basis
        account_asset = self.ledger_cash + position_value
        real_asset = account_asset + self.ledger_harvested
        self.trade_log.append({
            "Date": self.data.index[-1].strftime("%y/%m/%d"),
            "Action": side,
            "Price": round(price, 1),
            "Shares": round(shares, 1),
            "TradeAmt": round(amount, 1),
            "Fee": round(fee, 1),
            "TradeGain": round(realized_pnl, 1),
            "TotalGain": round(self.ledger_realized_pnl, 1),
            "Position": round(self.ledger_shares, 1),
            "PosValue": round(position_value, 1),
            "FloatGain": round(unrealized_pnl, 1),
            "Cash": round(self.ledger_cash, 1),
            "AcctAsset": round(account_asset, 1),
            "RealAsset": round(real_asset, 1),
        })

    def log_net_worth(self):
        price = self.actual_price()
        real_asset = (
            self.ledger_cash
            + self.ledger_shares * price
            + self.ledger_harvested
        )
        self.net_worth_log.append({
            "Date": self.data.index[-1],
            "RealAsset": real_asset,
        })

    def harvest_cash(self):
        timestamp = pd.Timestamp(self.data.index[-1])
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        date = timestamp.normalize()
        month_key = timestamp.strftime("%Y-%m")
        self.daily_cash_highs[date] = max(self.daily_cash_highs.get(date, 0), self.ledger_cash)

        if month_key in self.harvested_months:
            return

        if self.ledger_cash <= HARVEST_CASH_THRESHOLD:
            return

        recent_dates = sorted(day for day in self.daily_cash_highs if day <= date)[-HARVEST_LOOKBACK_DAYS:]
        recent_high = max(self.daily_cash_highs[day] for day in recent_dates)
        is_recent_cash_high = (
            len(recent_dates) >= HARVEST_LOOKBACK_DAYS
            and self.ledger_cash >= recent_high
        )
        is_month_last_trading_day = date == LAST_TRADING_DATE_BY_MONTH.get(month_key)
        if not is_recent_cash_high and not is_month_last_trading_day:
            return

        self.harvested_months.add(month_key)
        amount = min(HARVEST_AMOUNT, self.ledger_cash)
        self.ledger_cash -= amount
        self.ledger_harvested += amount
        self._broker._cash -= amount
        self.log_trade("HARVEST", 0, self.actual_price(), amount, 0)

    def subsidize_cash_if_needed(self):
        if self.ledger_cash >= SUBSIDY_CASH_THRESHOLD:
            return

        timestamp = pd.Timestamp(self.data.index[-1])
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        month_key = timestamp.strftime("%Y-%m")
        if month_key in self.subsidized_months:
            return

        self.subsidized_months.add(month_key)
        self.ledger_cash += SUBSIDY_AMOUNT
        self.ledger_subsidized += SUBSIDY_AMOUNT
        self._broker._cash += SUBSIDY_AMOUNT
        self.log_trade("SUBSIDY", 0, self.actual_price(), SUBSIDY_AMOUNT, 0)

    def buy_shares(self, shares):
        price = self.actual_price()
        affordable_shares = self.ledger_cash / (price * (1 + COMMISSION))
        actual_shares = min(shares, affordable_shares)
        size = self.order_size(actual_shares)
        self.planned_buys += 1
        if size > 0:
            actual_shares = size * FRACTIONAL_UNIT
            amount = actual_shares * price
            fee = amount * COMMISSION
            self.ledger_cash -= amount + fee
            self.ledger_shares += actual_shares
            self.ledger_cost_basis += amount + fee
            self.log_trade("BUY", actual_shares, price, amount, fee)

            self.buy(size=size)
            self.executed_buys += 1
        else:
            self.skipped_buys += 1

    def sell_shares(self, shares, min_shares):
        sellable_shares = max(0, self.ledger_shares - min_shares)
        actual_shares = min(shares, sellable_shares)
        size = self.order_size(actual_shares)
        self.planned_sells += 1
        if size > 0:
            price = self.actual_price()
            actual_shares = size * FRACTIONAL_UNIT
            amount = actual_shares * price
            fee = amount * COMMISSION
            average_cost = self.ledger_cost_basis / self.ledger_shares
            removed_cost_basis = average_cost * actual_shares
            realized_pnl = amount - fee - removed_cost_basis

            self.ledger_cash += amount - fee
            self.ledger_shares -= actual_shares
            self.ledger_cost_basis -= removed_cost_basis
            self.ledger_realized_pnl += realized_pnl
            self.log_trade("SELL", actual_shares, price, amount, fee, realized_pnl)

            self.sell(size=size)
            self.executed_sells += 1
        else:
            self.skipped_sells += 1

    def next(self):
        price = self.actual_price()
        ma20 = self.actual_ma20()
        regime = self.trading_regime(price, ma20)
        self.cash_budget = self._broker.margin_available
        self.harvest_cash()
        self.subsidize_cash_if_needed()

        if not self.initialized:
            self.buy_shares(INITIAL_SHARES)
            self.grid_price = price
            self.initialized = True
            self.log_net_worth()
            return

        if regime is None:
            self.log_net_worth()
            return

        if price <= self.grid_price - regime["buy_step"]:
            buy_shares = regime["buy_shares"]
            if self.ledger_cash > EXTRA_BUY_CASH_THRESHOLD:
                buy_shares += EXTRA_BUY_SHARES
            self.buy_shares(buy_shares)
            self.grid_price = price
            self.log_net_worth()
            return

        if price >= self.grid_price + regime["sell_step"]:
            self.sell_shares(regime["sell_shares"], regime["min_shares"])
            self.grid_price = price

        self.log_net_worth()


data = load_data(SYMBOL, START, END)
daily_data = load_daily_data(SYMBOL, START, END)
daily_ma = daily_data["Close"].rolling(MA_DAYS, min_periods=1).mean()
DAILY_MA_BY_DATE = {index.normalize(): value for index, value in daily_ma.items()}
DATA_DATES = pd.Series(data.index).map(
    lambda index: pd.Timestamp(index).tz_convert(None).normalize()
    if pd.Timestamp(index).tzinfo is not None
    else pd.Timestamp(index).normalize()
)
LAST_TRADING_DATE_BY_MONTH = DATA_DATES.groupby(DATA_DATES.dt.strftime("%Y-%m")).max().to_dict()
start_price = data["Close"].iloc[0]
initial_cost = INITIAL_SHARES * start_price * (1 + COMMISSION)
affordable_initial_shares = CASH / (start_price * (1 + COMMISSION))

bt = FractionalBacktest(
    data,
    GridByMA,
    cash=CASH,
    commission=COMMISSION,
    exclusive_orders=False,
    finalize_trades=True,
    fractional_unit=FRACTIONAL_UNIT,
    trade_on_close=True,
)

stats = bt.run()
strategy = stats["_strategy"]
LOG_DIR.mkdir(exist_ok=True)
trade_log_file = LOG_DIR / f"{SYMBOL}_trade_log.csv"
trade_log = pd.DataFrame(strategy.trade_log)
min_logged_shares = trade_log["Position"].min() if not trade_log.empty else strategy.ledger_shares
below_global_floor = (
    trade_log[(trade_log["Action"] == "SELL") & (trade_log["Position"] < HIGH_MIN_SHARES)]
    if not trade_log.empty
    else pd.DataFrame()
)
trade_log_file.write_text(format_table(trade_log.astype(str)) + "\n")

PLOT_DIR.mkdir(exist_ok=True)
equity_plot_file = PLOT_DIR / f"{SYMBOL}_{DATA_INTERVAL or '1d'}_equity.png"
equity_curve = stats["_equity_curve"]
net_worth_curve = pd.DataFrame(strategy.net_worth_log).set_index("Date")
real_asset_drawdown = net_worth_curve["RealAsset"] / net_worth_curve["RealAsset"].cummax() - 1
max_real_asset_drawdown = real_asset_drawdown.min() * 100
benchmark_initial_curve = data["Close"] / start_price * CASH
benchmark_initial_drawdown = benchmark_initial_curve / benchmark_initial_curve.cummax() - 1
max_benchmark_initial_drawdown = benchmark_initial_drawdown.min() * 100
plot_log = trade_log.copy()
if not plot_log.empty:
    plot_log["Date"] = pd.to_datetime(plot_log["Date"], format="%y/%m/%d")

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
data["Close"].plot(ax=axes[0], color="black", linewidth=1, title=f"{SYMBOL} Price and Trades")
if not plot_log.empty:
    buys = plot_log[plot_log["Action"] == "BUY"]
    sells = plot_log[plot_log["Action"] == "SELL"]
    axes[0].scatter(buys["Date"], buys["Price"], color="tab:green", marker="^", s=35, label="Buy")
    axes[0].scatter(sells["Date"], sells["Price"], color="tab:red", marker="v", s=35, label="Sell")
    axes[0].legend(loc="best")
if not plot_log.empty:
    plot_log.set_index("Date")[["Cash", "PosValue"]].plot(
        ax=axes[1],
        linewidth=1,
        title="Cash vs Position Value",
    )
    axes[1].axhline(1000, color="tab:orange", linestyle="--", linewidth=1, label="$1,000")
    axes[1].axhline(0, color="tab:red", linestyle="--", linewidth=1, label="$0")
    axes[1].legend(loc="best")
else:
    axes[1].set_title("Cash vs Position Value")
net_worth_curve["RealAsset"].plot(
    ax=axes[2],
    color="tab:blue",
    linewidth=1,
    label="RealAsset",
    title="Real Asset vs Benchmark",
)
benchmark_initial_curve.plot(ax=axes[2], color="tab:gray", linewidth=1, linestyle="--", label="Benchmark 10k Buy&Hold")
axes[2].legend(loc="best")
real_asset_drawdown.mul(100).plot(
    ax=axes[3],
    color="tab:red",
    linewidth=1,
    title="Real Asset Drawdown %",
)
axes[0].set_ylabel("Price")
axes[1].set_ylabel("Value")
axes[2].set_ylabel("Asset")
axes[3].set_ylabel("Drawdown %")
axes[3].set_xlabel("Date")
fig.tight_layout()
fig.savefig(equity_plot_file, dpi=150)
plt.close(fig)

print("\n=== Raw Data: First 5 Rows ===")
print(data.head())
print("\n=== Raw Data: Last 5 Rows ===")
print(data.tail())

print("\n=== Backtest Summary ===")
print(f"Symbol: {SYMBOL}")
print(f"Strategy: {GridByMA.__name__}")
print(f"Period: {data.index[0].date()} -> {data.index[-1].date()}")
print(f"Initial Cash: ${CASH:,.2f}")
print(f"Start Price: ${start_price:,.2f}")
print(f"Configured Initial Shares: {INITIAL_SHARES:,.3f}")
print(f"Cash Needed for Initial Shares: ${initial_cost:,.2f}")
print(f"Affordable Initial Shares: {affordable_initial_shares:,.3f}")
total_final_assets = stats["Equity Final [$]"] + strategy.ledger_harvested
total_invested = CASH + strategy.ledger_subsidized
profit_on_invested = total_final_assets - total_invested
return_on_invested = (total_final_assets / total_invested - 1) * 100
return_on_initial = (total_final_assets / CASH - 1) * 100
benchmark_initial_assets = CASH * data["Close"].iloc[-1] / start_price
benchmark_invested_assets = total_invested * data["Close"].iloc[-1] / start_price
print(f"Final Account Equity: ${stats['Equity Final [$]']:,.2f}")
print(f"Real Final Assets: ${total_final_assets:,.2f}")
print(f"Subsidized Cash: ${strategy.ledger_subsidized:,.2f}")
print(f"Total Invested Cash: ${total_invested:,.2f}")
print(f"Profit On Invested Cash: ${profit_on_invested:,.2f}")
print(f"Return On Invested Cash: {return_on_invested:.2f}%")
print(f"Return On Initial Cash: {return_on_initial:.2f}%")
print(f"Benchmark Initial Cash Assets: ${benchmark_initial_assets:,.2f}")
print(f"Benchmark Invested Cash Assets: ${benchmark_invested_assets:,.2f}")
print(f"Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%")
print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
print(f"Real Asset Max Drawdown: {max_real_asset_drawdown:.2f}%")
print(f"Benchmark Initial Max Drawdown: {max_benchmark_initial_drawdown:.2f}%")
print(f"Trades: {stats['# Trades']}")
print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"Commissions: ${stats['Commissions [$]']:,.2f}")
print(f"Ledger Cash: ${strategy.ledger_cash:,.2f}")
print(f"Ledger Shares: {strategy.ledger_shares:,.3f}")
print(f"Minimum Logged Shares: {min_logged_shares:,.3f}")
print(f"Ledger Position Value: ${strategy.ledger_shares * data['Close'].iloc[-1]:,.2f}")
print(f"Ledger Realized P/L: ${strategy.ledger_realized_pnl:,.2f}")
print(f"Harvested Cash: ${strategy.ledger_harvested:,.2f}")
print(f"Trade Log: {trade_log_file}")
print(f"Equity Plot: {equity_plot_file}")
print("\n=== Trade Log Preview ===")
if trade_log.empty:
    print("(no trades)")
else:
    preview = trade_log.tail(20).astype(str)
    print(format_table(preview))
print(f"Planned Buys: {strategy.planned_buys}")
print(f"Submitted Buys: {strategy.executed_buys}")
print(f"Skipped Buys: {strategy.skipped_buys}")
print(f"Planned Sells: {strategy.planned_sells}")
print(f"Submitted Sells: {strategy.executed_sells}")
print(f"Skipped Sells: {strategy.skipped_sells}")
if initial_cost > CASH:
    print("\nCONFIG WARNING: Initial cash cannot buy the configured initial shares.")
if not below_global_floor.empty:
    print("CONFIG WARNING: A sell left position below the global 80-share keep floor.")

if SHOW_PLOT:
    html_plot_file = PLOT_DIR / f"{SYMBOL}_{DATA_INTERVAL or '1d'}_backtest.html"
    bt.plot(filename=str(html_plot_file), superimpose="M", open_browser=False)
    print(f"Backtest HTML Plot: {html_plot_file}")
