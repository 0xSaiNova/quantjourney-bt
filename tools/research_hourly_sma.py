#!/usr/bin/env python3
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Chronological hourly-SMA research using the production weight ledger.

Run from the repository: python tools/research_hourly_sma.py --input DATA.csv
Input is yfinance's two-header OHLCV CSV, with timezone-aware timestamps.
Downloads are deliberately separate: one frozen data file serves every trial.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtester.execution.contract_spec import ContractSpec  # noqa: E402
from backtester.portfolio.accounting import build_weight_ledger  # noqa: E402
from backtester.portfolio.rebalance import RebalanceEngine, RebalancePolicy  # noqa: E402
from backtester.portfolio.weight_cost import FixedBpsWeightCostModel  # noqa: E402
from backtester.version import __version__  # noqa: E402
from strategies.example_weights_09_intraday_1h_sma_trend import (  # noqa: E402
    HourlyTrendParameters,
    hourly_trend_signals,
)

UNIVERSE = ["SPY", "QQQ", "IWM"]
PERIODS = {
    "train": ("2025-01-01", "2026-01-01"),
    "validation": ("2026-01-01", "2026-05-11"),
    "published_holdout": ("2026-05-11", "2026-07-11"),
    "later_holdout": ("2026-07-11", "2026-09-24"),
}


def download_hourly_data(path: Path) -> None:
    """Fetch one adjusted regular-session snapshot for every trial to share."""
    import yfinance as yf

    path.parent.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(path.parent / "yfinance-cache"))
    data = yf.download(
        UNIVERSE,
        start=PERIODS["train"][0],
        end=PERIODS["later_holdout"][1],
        interval="1h",
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError("No hourly data returned; check network and provider retention limits")
    data.to_csv(path)


def load_close(path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path, header=[0, 1], index_col=0)
    close = frame["Close"].loc[:, UNIVERSE].astype(float)
    close.index = pd.to_datetime(close.index, utc=True)
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError("Input timestamps must be unique and increasing")
    valid = (np.isfinite(close) & (close > 0)).all(axis=1)
    dropped = int((~valid).sum())
    close = close.loc[valid]
    if close.empty:
        raise ValueError("No complete positive price bars")
    gaps = close.index.to_series().diff().dropna()
    # Yahoo labels the final, shorter regular-session bar as 15:30. Overnight
    # and weekend gaps are expected; sub-hour bars imply a different strategy.
    if (gaps < pd.Timedelta(hours=1)).any():
        raise ValueError("Expected hourly bars, not daily or sub-hour input")
    if not (gaps == pd.Timedelta(hours=1)).any():
        raise ValueError("No hourly observations found; daily sample data is unsuitable")
    return close, dropped


def weights_for(close: pd.DataFrame, parameters: HourlyTrendParameters) -> pd.DataFrame:
    active = hourly_trend_signals(close, parameters).eq(1.0)
    return active.div(active.sum(axis=1), axis=0).fillna(0.0).clip(upper=0.25)


def evaluate(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    start: str,
    end: str,
    cost_bps: float = 1.0,
) -> dict:
    """Reset capital per fold; preserve past indicator warm-up, never future bars.

    Mirrors Backtester fast weights: shift targets once, 5% cash buffer,
    signal-change rebalances, close-to-close accounting, recursive trade costs.
    Each fold starts flat; no unreported pre-period holdings or liquidation.
    """
    local_dates = close.index.tz_convert("America/New_York")
    mask = (local_dates >= pd.Timestamp(start, tz="America/New_York")) & (
        local_dates < pd.Timestamp(end, tz="America/New_York")
    )
    prices = close.loc[mask]
    if len(prices) < 30:
        raise ValueError(f"Insufficient hourly bars for {start} through {end}")
    # Shift before slicing to make the previous completed bar available.
    targets = weights.shift(1).fillna(0.0).loc[mask] * 0.95
    returns = prices.pct_change(fill_method=None)
    returns.iloc[0] = 0.0
    engine = RebalanceEngine(
        RebalancePolicy(frequency=None, rebalance_on_signal_change=True),
        periods_per_year=252 * 7,
    )
    actual, flags = engine.run(targets, returns)
    result, costs, _ = build_weight_ledger(
        actual_weights=actual,
        portfolio_returns=engine.portfolio_returns,
        prices=prices,
        initial_capital=100_000,
        rebalance_flags=flags,
        cost_model=FixedBpsWeightCostModel(total_bps=cost_bps),
        contract_spec_resolver=ContractSpec.equity,
    )
    pd.testing.assert_series_equal(
        result.nav, result.cash + result.position_values.sum(axis=1), check_names=False
    )
    drawdown = result.nav.div(result.nav.cummax()) - 1.0
    return {
        "return_pct": float((result.nav.iloc[-1] / 100_000 - 1) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "cost_usd": float(costs.total_cost.sum()),
        "rebalances": int(flags.sum()),
        "average_exposure_pct": float(result.weights.sum(axis=1).mean() * 100),
        "bars": len(prices),
        "first_bar": str(prices.index[0]),
        "last_bar": str(prices.index[-1]),
    }


def score(metrics: dict) -> float:
    """Return penalized by drawdown, without unstable short-window annualization."""
    return metrics["return_pct"] + metrics["max_drawdown_pct"]


def run_slope_study(path: Path, output: Path) -> dict:
    """Isolate the entry filter at fixed SMA(10/30); no retuning or promotion."""
    close, dropped = load_close(path)
    cutoff = pd.Timestamp(PERIODS["later_holdout"][1], tz="America/New_York")
    close = close.loc[close.index < cutoff]
    variants = {}
    # One bar is the primary hypothesis. Three/seven bars are sensitivity
    # checks, not additional chances to select a winner on inspected dates.
    for lookback in (0, 1, 3, 7):
        params = HourlyTrendParameters(slope_lookback_bars=lookback)
        weights = weights_for(close, params)
        periods = {}
        for name, bounds in PERIODS.items():
            periods[name] = {
                str(cost): evaluate(close, weights, *bounds, cost_bps=cost) for cost in (1, 5, 10)
            }
        individual = {}
        for instrument in UNIVERSE:
            sleeve = weights.copy()
            sleeve.loc[:, sleeve.columns != instrument] = 0.0
            individual[instrument] = {
                name: evaluate(close, sleeve, *PERIODS[name])
                for name in ("published_holdout", "later_holdout")
            }
        variants[str(lookback)] = {
            "parameters": asdict(params),
            "periods": periods,
            "individual_etfs": individual,
        }
    report = {
        "engine_version": __version__,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data_end": str(close.index[-1]),
        "dropped_incomplete_bars": dropped,
        "status": "exploratory: all periods were previously inspected; no unseen holdout",
        "hypothesis": "slow SMA rises over one bar before entry; bearish exits unchanged",
        "assumptions": "SMA 10/30; 25% per ETF; 5% cash buffer; next-bar close; overnight holds",
        "variants": variants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"])
    for lookback, variant in variants.items():
        print("Slope bars:", lookback)
        for period, metrics in variant["periods"].items():
            print(
                period,
                "return / drawdown / rebalances:",
                metrics["1"]["return_pct"],
                metrics["1"]["max_drawdown_pct"],
                metrics["1"]["rebalances"],
            )
    return report


def run_research(path: Path, output: Path) -> dict:
    close, dropped = load_close(path)
    # September 23 snapshot: later observations never enter this experiment.
    cutoff = pd.Timestamp(PERIODS["later_holdout"][1], tz="America/New_York")
    close = close.loc[close.index < cutoff]
    baseline = HourlyTrendParameters()
    trials = []
    # Freeze the small search space and split boundaries before reading holdouts.
    for fast, slow, buffer, confirmation in itertools.product(
        (5, 10, 20), (30, 50, 80), (0.0, 10.0), (1, 2)
    ):
        params = HourlyTrendParameters(fast, slow, buffer, confirmation)
        weights = weights_for(close, params)
        train = evaluate(close, weights, *PERIODS["train"])
        trials.append({"parameters": asdict(params), "train": train})
    trials.sort(key=lambda trial: score(trial["train"]), reverse=True)
    # Only eight training finalists reach validation. Published/later holdouts
    # never influence ranking or parameter selection.
    finalists = trials[:8]
    baseline_weights = weights_for(close, baseline)
    baseline_train = evaluate(close, baseline_weights, *PERIODS["train"])
    baseline_validation = evaluate(close, baseline_weights, *PERIODS["validation"])
    for trial in finalists:
        trial["validation"] = evaluate(
            close,
            weights_for(close, HourlyTrendParameters(**trial["parameters"])),
            *PERIODS["validation"],
        )
    eligible = [
        trial
        for trial in finalists
        if trial["train"]["return_pct"] >= baseline_train["return_pct"]
        and trial["validation"]["return_pct"] > baseline_validation["return_pct"]
        and score(trial["validation"]) > score(baseline_validation)
    ]
    selected = (
        HourlyTrendParameters(**max(eligible, key=lambda t: score(t["validation"]))["parameters"])
        if eligible
        else baseline
    )
    comparisons = {}
    for label, params in (("baseline", baseline), ("selected", selected)):
        weights = weights_for(close, params)
        comparisons[label] = {
            name: evaluate(close, weights, *bounds) for name, bounds in PERIODS.items()
        }
        comparisons[label]["published_holdout_5bps"] = evaluate(
            close, weights, *PERIODS["published_holdout"], cost_bps=5
        )
        comparisons[label]["later_holdout_5bps"] = evaluate(
            close, weights, *PERIODS["later_holdout"], cost_bps=5
        )
    report = {
        "engine_version": __version__,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": "Yahoo Finance adjusted regular-session 1h closes",
        "data_start": str(close.index[0]),
        "data_end": str(close.index[-1]),
        "dropped_incomplete_bars": dropped,
        "assumptions": {
            "cash_buffer": 0.05,
            "max_position_weight": 0.25,
            "cost_bps_per_trade": 1,
            "execution": "production fast weights; next-bar close; flat fold starts",
            "overnight_positions": True,
            "cash_interest": 0,
        },
        "selection": "top 8 training return-minus-drawdown; validation improvement gate",
        "n_trials": len(trials),
        "selected_parameters": asdict(selected),
        "recommendation": (
            "candidate passed both holdout comparisons; further validation still required"
            if selected != baseline
            and all(
                comparisons["selected"][period]["return_pct"]
                > comparisons["baseline"][period]["return_pct"]
                and score(comparisons["selected"][period]) >= score(comparisons["baseline"][period])
                for period in (
                    "published_holdout",
                    "later_holdout",
                    "published_holdout_5bps",
                    "later_holdout_5bps",
                )
            )
            else "retain baseline defaults; no validated replacement"
        ),
        "comparisons": comparisons,
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("recommendation", "selected_parameters", "comparisons")},
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/w09_hourly_ohlcv_20260923.csv"))
    parser.add_argument("--download", action="store_true", help="download a fresh hourly CSV first")
    parser.add_argument("--slope-study", action="store_true", help="isolate the entry slope filter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.download:
        download_hourly_data(args.input)
    output = args.output or Path(
        "reports/w09-research/slope-results.json"
        if args.slope_study
        else "reports/w09-research/results.json"
    )
    runner = run_slope_study if args.slope_study else run_research
    runner(args.input, output)


if __name__ == "__main__":
    main()
