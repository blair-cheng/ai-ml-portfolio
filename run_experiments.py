import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("config/default.yaml")
EXPERIMENT_CONFIG = Path("config/experiments.yaml")
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
RESULTS_FILE = RESULTS_DIR / "parameter_sweep_results.csv"
REPORT_FILE = REPORTS_DIR / "top_strategies.md"


SUMMARY_PREFIX = "SUMMARY_JSON "


def load_yaml(path):
    with path.open() as file:
        return yaml.safe_load(file)


def set_nested(config, dotted_key, value):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def run_backtest(config, run_name, changed_parameter="", changed_value=""):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_config = copy.deepcopy(config)
        run_config["show_plot"] = False
        run_config["paths"]["log_dir"] = str(temp_path / "logs")
        run_config["paths"]["plot_dir"] = str(temp_path / "plots")
        config_path = temp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(run_config, sort_keys=False))

        result = subprocess.run(
            [sys.executable, "playground.py", "--config", str(config_path), "--no-plot"],
            check=True,
            capture_output=True,
            text=True,
        )

    summary_line = None
    for line in result.stdout.splitlines():
        if line.startswith(SUMMARY_PREFIX):
            summary_line = line[len(SUMMARY_PREFIX):]

    if summary_line is None:
        raise RuntimeError(f"No summary JSON found for run {run_name}")

    row = json.loads(summary_line)
    row["run_name"] = run_name
    row["changed_parameter"] = changed_parameter
    row["changed_value"] = changed_value
    row["score"] = score(row)
    return row


def score(row):
    return (
        row["return_on_invested_cash"]
        - 0.8 * abs(row["real_asset_max_drawdown"])
        - 0.2 * abs(row["max_drawdown"])
        - 0.02 * row["subsidized_cash"] / 100
        + 0.02 * row["harvested_cash"] / 100
    )


def write_results(rows):
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "run_name",
        "changed_parameter",
        "changed_value",
        "score",
        "final_account_equity",
        "real_final_assets",
        "total_invested_cash",
        "profit_on_invested_cash",
        "return_on_invested_cash",
        "return_on_initial_cash",
        "buy_hold_return",
        "benchmark_initial_assets",
        "benchmark_invested_assets",
        "max_drawdown",
        "real_asset_max_drawdown",
        "benchmark_initial_max_drawdown",
        "trades",
        "win_rate",
        "commissions",
        "final_cash",
        "final_shares",
        "final_position_value",
        "harvested_cash",
        "subsidized_cash",
        "net_external_cash_flow",
        "planned_buys",
        "submitted_buys",
        "skipped_buys",
        "planned_sells",
        "submitted_sells",
        "skipped_sells",
    ]

    with RESULTS_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, fields):
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                value = f"{value:.2f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(rows):
    REPORTS_DIR.mkdir(exist_ok=True)
    fields = [
        "run_name",
        "score",
        "return_on_invested_cash",
        "real_asset_max_drawdown",
        "subsidized_cash",
        "harvested_cash",
        "trades",
    ]

    baseline = next(row for row in rows if row["run_name"] == "baseline")
    by_score = sorted(rows, key=lambda row: row["score"], reverse=True)[:10]
    by_return = sorted(rows, key=lambda row: row["return_on_invested_cash"], reverse=True)[:10]
    by_drawdown = sorted(rows, key=lambda row: abs(row["real_asset_max_drawdown"]))[:10]
    by_subsidy = sorted(rows, key=lambda row: row["subsidized_cash"])[:10]
    worst = sorted(rows, key=lambda row: row["score"])[:10]

    content = [
        "# NVDL Parameter Sweep Report",
        "",
        "## Baseline",
        "",
        markdown_table([baseline], fields),
        "",
        "## Top 10 By Score",
        "",
        markdown_table(by_score, fields),
        "",
        "## Top 10 By Return On Invested Cash",
        "",
        markdown_table(by_return, fields),
        "",
        "## Top 10 By Lowest Real Asset Drawdown",
        "",
        markdown_table(by_drawdown, fields),
        "",
        "## Top 10 By Lowest Subsidy",
        "",
        markdown_table(by_subsidy, fields),
        "",
        "## Worst 10 By Score",
        "",
        markdown_table(worst, fields),
        "",
    ]
    REPORT_FILE.write_text("\n".join(content))


def main():
    base_config = load_yaml(DEFAULT_CONFIG)
    experiment_config = load_yaml(EXPERIMENT_CONFIG)

    rows = [run_backtest(base_config, "baseline")]
    for parameter, values in experiment_config["single_factor"].items():
        for value in values:
            config = copy.deepcopy(base_config)
            set_nested(config, parameter, value)
            run_name = f"{parameter}={value}"
            print(f"Running {run_name}")
            rows.append(run_backtest(config, run_name, parameter, value))

    write_results(rows)
    write_report(rows)
    print(f"Wrote {RESULTS_FILE}")
    print(f"Wrote {REPORT_FILE}")


if __name__ == "__main__":
    main()
