# NVDL Grid Parameter Experiment Plan

## Goal

Turn the current single-strategy backtest into a lightweight parameter experiment system.

The first version should stay small enough to run on a phone or in Claude:

- keep `playground.py` usable as the single-run script
- move strategy parameters into config files
- add one experiment runner for parameter sweeps
- output CSV results and a short ranked report

Do not start with a large `src/` refactor. The current priority is fast iteration and readable results.

## Fixed Experiment Boundary

Keep these constant so every run is comparable:

- Symbol: `NVDL`
- Data interval: `1h`
- Backtest period: available local `data/NVDL_1h.csv`
- Daily moving-average data: local `data/NVDL.csv`
- Execution price: hourly `Close`
- Commission: `0.1%`
- Fractional shares: enabled, minimum `0.001`
- Benchmark: buy and hold from first hourly close to final hourly close

## Minimal File Structure

First implementation should add only these files:

```text
config/
  default.yaml
  experiments.yaml
results/
  parameter_sweep_results.csv
reports/
  top_strategies.md
run_experiments.py
```

Keep these existing files:

```text
playground.py
README.md
backtesting/
data/
```

`logs/` and `plots/` remain generated output.

## Step 1: Config Current Strategy

Create `config/default.yaml` with the current strategy values:

```yaml
symbol: NVDL
data_interval: 1h
initial_cash: 10000
initial_shares: 200
commission: 0.001
fractional_unit: 0.001

ma_window_days: 252

zones:
  high:
    condition: price > high_multiple * ma
    high_multiple: 1.25
    buy_step: 2
    buy_shares: 15
    sell_step: 2
    sell_shares: 20
    min_shares: 80

  mid:
    condition: deep_multiple * ma <= price <= high_multiple * ma
    deep_multiple: 0.50
    buy_step: 2
    buy_shares: 20
    sell_step: 2
    sell_shares: 15
    min_shares: 120

  deep:
    condition: price < deep_multiple * ma
    buy_step: 3
    buy_shares: 15
    sell_step: 2
    sell_shares: 10
    min_shares: 120

cash_rules:
  harvest_threshold: 5000
  harvest_amount: 500
  harvest_lookback_days: 5
  subsidy_threshold: 100
  subsidy_amount: 1000
  extra_buy_cash_threshold: 5000
  extra_buy_shares: 5
```

Implementation rule: config values should reproduce the current `playground.py` result before any experiments start.

## Step 2: Single-Factor Experiments

Create `config/experiments.yaml`.

First pass should test one variable family at a time, not every possible combination.

Recommended first sweep:

```yaml
single_factor:
  ma_window_days: [126, 189, 252, 315, 378]
  high_multiple: [1.15, 1.20, 1.25, 1.30]
  deep_multiple: [0.40, 0.50, 0.60]
  mid_buy_shares: [10, 15, 20, 25]
  mid_sell_shares: [10, 15, 20]
  harvest_threshold: [4000, 5000, 6000, 8000]
  harvest_amount: [300, 500, 800, 1000]
```

Do not sweep every zone parameter at first. The current pain points are:

- cash gets exhausted too often
- subsidy dependence is too high
- harvest timing changes compounding
- middle zone may overreact after trend changes

So the first sweep should focus on these.

## Step 3: Experiment Runner

Add `run_experiments.py`.

Requirements:

- load `config/default.yaml`
- load `config/experiments.yaml`
- run baseline config first
- run each single-factor variation
- save one row per run to `results/parameter_sweep_results.csv`
- generate `reports/top_strategies.md`

The runner can reuse the same strategy logic as `playground.py`.

Avoid duplicate strategy implementations if possible. If refactoring is needed, keep it small:

- move shared backtest execution into a helper function
- keep plotting optional
- keep `playground.py` as the easy manual entrypoint

## Metrics Per Run

Every row in `parameter_sweep_results.csv` should include:

```text
run_name
changed_parameter
changed_value
final_account_equity
real_final_assets
total_invested_cash
profit_on_invested_cash
return_on_invested_cash
return_on_initial_cash
buy_hold_return
benchmark_initial_assets
benchmark_invested_assets
max_drawdown
real_asset_max_drawdown
benchmark_initial_max_drawdown
trades
win_rate
commissions
final_cash
final_shares
final_position_value
harvested_cash
subsidized_cash
net_external_cash_flow
score
```

## Ranking Score

Use a simple first-pass score:

```text
score =
  return_on_invested_cash
  - 0.8 * abs(real_asset_max_drawdown)
  - 0.2 * abs(max_drawdown)
  - 0.02 * subsidized_cash / 100
  + 0.02 * harvested_cash / 100
```

This is not a permanent formula. It is a practical sorting tool.

The report should still show top rows by separate views:

- top by score
- top by return on invested cash
- top by lowest real asset drawdown
- top by lowest subsidy
- worst by score

## Report Format

`reports/top_strategies.md` should contain:

```text
Baseline Result
Top 10 By Score
Top 10 By Return
Top 10 By Risk Control
Lowest Subsidy Strategies
Worst 10
Notes
```

Notes should answer:

- Which parameters reduce subsidy dependence?
- Which parameters improve harvested cash?
- Which parameters reduce drawdown?
- Which parameters only improve return by taking too much risk?
- Does the strategy beat buy and hold on real assets, return on invested cash, or drawdown?

## Implementation Order

1. Add `config/default.yaml`.
2. Make `playground.py` load config while preserving current behavior.
3. Verify baseline result matches current README numbers.
4. Add `config/experiments.yaml`.
5. Add `run_experiments.py`.
6. Run a small single-factor sweep.
7. Save `results/parameter_sweep_results.csv`.
8. Generate `reports/top_strategies.md`.
9. Update README with the new experiment workflow.

## What Not To Do Yet

- Do not add machine learning.
- Do not optimize on minute data.
- Do not split the project into many modules unless the runner becomes hard to maintain.
- Do not delete `playground.py`; it is still the easiest manual test entrypoint.
- Do not rank only by final return.

## Next Codex Task

Implement the lightweight parameter experiment system described in this plan.

Keep current strategy behavior unchanged for the baseline run. The first success condition is:

```text
python playground.py
python run_experiments.py
```

Both commands should run successfully, and the baseline row in `results/parameter_sweep_results.csv` should match the single-run summary.
