# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""
Example Weights 09 - Intraday 1h SMA Trend
==========================================

Mode: weights.
Idea: on hourly bars, hold each name only while its SMA(10) is above its
SMA(30), equal weight across active names. A slower intraday-to-swing trend
template.
Universe: three predeclared liquid ETFs: SPY, QQQ and IWM.

Timeframe-grid note: 1h is the slowest cadence in the intraday grid
(1m -> 5m -> 30m -> 1h). With ~7 bars per session it behaves like a fast swing
strategy: lower turnover than the 1m/5m templates, and the SMA(10/30) cross is
a clean hourly trend filter. Use this as the "calm" end of the grid to compare
turnover and cost drag against the faster templates.

Usage:
    ./strategy.sh example_weights_09_intraday_1h_sma_trend

The default remains the published SMA(10/30) rule. Optional QJ_SMA_FAST,
QJ_SMA_SLOW, QJ_SMA_ENTRY_BPS and QJ_SMA_CONFIRM_BARS environment variables
configure research variants. Entry confirmation and a positive spread buffer
can reduce whipsaw; neither delays exits after a bearish crossover. Positions
may be held overnight: "intraday" describes bar frequency, not forced daily
liquidation. No variant is assumed to improve out-of-sample returns.
QJ_SMA_SLOPE_BARS optionally requires the slow SMA to exceed its value that
many completed bars ago before entering (0 disables). This is an entry-only
filter; a falling slope alone does not close an existing position.

For local CSV research without cloud credentials:
    python tools/research_hourly_sma.py --input data/w09_hourly_ohlcv_20260923.csv
"""

import asyncio
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester import Backtester
from backtester.portfolio.rebalance import RebalancePolicy


def _credentials() -> dict:
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": api_key,
        "email": None if api_key else os.environ.get("QJ_EMAIL"),
        "password": None if api_key else os.environ.get("QJ_PASSWORD"),
    }


@dataclass(frozen=True)
class HourlyTrendParameters:
    """Windows count observed hourly bars, including overnight holding periods."""

    fast_window: int = 10
    slow_window: int = 30
    entry_buffer_bps: float = 0.0
    confirmation_bars: int = 1
    slope_lookback_bars: int = 0

    def __post_init__(self) -> None:
        for name in ("fast_window", "slow_window", "confirmation_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        if (
            isinstance(self.slope_lookback_bars, bool)
            or not isinstance(self.slope_lookback_bars, int)
            or self.slope_lookback_bars < 0
        ):
            raise ValueError("slope_lookback_bars must be a non-negative integer")
        if not np.isfinite(self.entry_buffer_bps) or self.entry_buffer_bps < 0:
            raise ValueError("entry_buffer_bps must be finite and non-negative")


def hourly_trend_signals(close: pd.DataFrame, parameters: HourlyTrendParameters) -> pd.DataFrame:
    """Confirm entries above a spread buffer; exit immediately on a down-cross.

    Only completed bars enter rolling windows. The execution engine, rather
    than this signal function, applies the next-bar execution delay. Invalid
    prices reset the position intent and require a complete warm-up again.
    """
    prices = close.where(np.isfinite(close) & (close > 0))
    fast = prices.rolling(parameters.fast_window).mean()
    slow = prices.rolling(parameters.slow_window).mean()
    valid = fast.notna() & slow.notna()
    spread = fast.div(slow) - 1.0
    enter = spread.gt(parameters.entry_buffer_bps / 10_000)
    if parameters.slope_lookback_bars:
        enter &= slow.gt(slow.shift(parameters.slope_lookback_bars))
    enter = enter.rolling(parameters.confirmation_bars).sum().eq(parameters.confirmation_bars)
    events = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    events = events.mask(enter & valid, 1.0)
    events = events.mask(spread.le(0) | ~valid, 0.0)
    return events.ffill().fillna(0.0)


class IntradaySMATrend1h(Backtester):
    """Hourly SMA(10/30) trend/cash strategy."""

    def __init__(self, *, trend_parameters: HourlyTrendParameters | None = None, **kwargs):
        self.trend_parameters = trend_parameters or HourlyTrendParameters()
        super().__init__(**kwargs)

    def _compute_signals(self) -> pd.DataFrame:
        return hourly_trend_signals(
            self.instruments_data.get_feature("close"), self.trend_parameters
        )

    def _compute_weights(self) -> pd.DataFrame:
        active = self.signals == 1.0
        counts = active.sum(axis=1)
        return active.div(counts, axis=0).fillna(0.0).clip(upper=0.25)


async def main() -> None:
    parameters = HourlyTrendParameters(
        fast_window=int(os.environ.get("QJ_SMA_FAST", "10")),
        slow_window=int(os.environ.get("QJ_SMA_SLOW", "30")),
        entry_buffer_bps=float(os.environ.get("QJ_SMA_ENTRY_BPS", "0")),
        confirmation_bars=int(os.environ.get("QJ_SMA_CONFIRM_BARS", "1")),
        slope_lookback_bars=int(os.environ.get("QJ_SMA_SLOPE_BARS", "0")),
    )
    strategy = IntradaySMATrend1h(
        **_credentials(),
        trend_parameters=parameters,
        strategy_name="ExampleWeights09_IntradaySMATrend1h",
        strategy_type="Long / Cash",
        initial_capital=100_000,
        instruments=["SPY", "QQQ", "IWM"],
        backtest_period={"start": "2026-05-11", "end": "2026-07-11"},
        granularity="1h",
        benchmark_symbol="SPY",
        benchmark_name="SPDR S&P 500 ETF Trust",
        source="yfinance",
        execution_mode="weights",
        max_position_size=0.25,
        rebalance_policy=RebalancePolicy(frequency=None, rebalance_on_signal_change=True),
        indicators_config=[
            {
                "function": "SMA",
                "price_cols": ["close"],
                "params": {"periods": [parameters.fast_window, parameters.slow_window]},
            },
        ],
        show_text_reports=True,
        save_text_reports=True,
        save_portfolio_plots=True,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
