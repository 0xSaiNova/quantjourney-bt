# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Hourly trend causality, risk-budget and research-accounting checks."""

import numpy as np
import pandas as pd
import pytest

from strategies.example_weights_09_intraday_1h_sma_trend import (
    HourlyTrendParameters,
    IntradaySMATrend1h,
    hourly_trend_signals,
)
from tools.research_hourly_sma import evaluate, load_close, weights_for


def market_frame(periods=160):
    rng = np.random.default_rng(2026)
    returns = rng.normal(0.0002, 0.004, size=(periods, 3))
    return pd.DataFrame(
        100 * np.exp(returns.cumsum(axis=0)),
        index=pd.date_range("2026-01-02 14:30", periods=periods, freq="h", tz="UTC"),
        columns=["SPY", "QQQ", "IWM"],
    )


def test_default_signals_preserve_the_published_rule():
    close = market_frame()
    fast, slow = close.rolling(10).mean(), close.rolling(30).mean()
    expected = (fast > slow).astype(float).where(fast.notna() & slow.notna(), 0.0)
    pd.testing.assert_frame_equal(hourly_trend_signals(close, HourlyTrendParameters()), expected)


def test_slope_filter_blocks_falling_sma_entries_without_forcing_early_exits():
    close = pd.DataFrame({"SPY": [110, 100, 90, 96, 97, 98, 105, 104, 104.8, 104.6, 90]})
    params = HourlyTrendParameters(1, 3, slope_lookback_bars=1)
    actual = hourly_trend_signals(close, params)["SPY"]
    # At bars 3/4 price exceeds the slow SMA, but that SMA is falling.
    # At bar 9 the slope is falling again: preserve the existing position.
    # The bearish crossover at bar 10 still exits immediately.
    assert actual.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]


def test_slope_filter_requires_history_and_never_enters_on_a_flat_sma():
    close = pd.DataFrame({"SPY": [100.0] * 20})
    params = HourlyTrendParameters(1, 3, slope_lookback_bars=5)
    assert hourly_trend_signals(close, params).eq(0).all().all()
    close["SPY"] = np.arange(100.0, 120.0)
    signals = hourly_trend_signals(close, params)["SPY"]
    assert signals.iloc[:7].eq(0).all()
    assert signals.iloc[7:].eq(1).all()


def test_buffer_holds_existing_trend_and_confirmation_never_delays_exit():
    close = pd.DataFrame({"SPY": [100, 100, 102, 104, 104.1, 103, 105, 107]})
    params = HourlyTrendParameters(1, 2, entry_buffer_bps=10, confirmation_bars=2)
    assert hourly_trend_signals(close, params)["SPY"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_missing_or_invalid_price_resets_intent_and_requires_full_warmup(invalid):
    close = pd.DataFrame({"SPY": [100, 101, 102, invalid, 104, 105, 106, 107]})
    params = HourlyTrendParameters(1, 3, confirmation_bars=2)
    assert hourly_trend_signals(close, params)["SPY"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_future_bars_cannot_change_existing_signals_or_weights():
    close = market_frame()
    params = HourlyTrendParameters(5, 30, 10, 2, slope_lookback_bars=3)
    changed = close.copy()
    changed.iloc[90:] *= 3
    pd.testing.assert_frame_equal(
        hourly_trend_signals(close, params).iloc[:90],
        hourly_trend_signals(changed, params).iloc[:90],
    )
    pd.testing.assert_frame_equal(
        weights_for(close, params).iloc[:90], weights_for(close.iloc[:90], params)
    )
    weights = weights_for(close, params)
    assert weights.ge(0).all().all()
    assert weights.le(0.25).all().all()
    assert weights.sum(axis=1).le(0.75).all()


def test_production_strategy_uses_the_same_research_signals():
    from backtester.portfolio.instr_data import InstrumentData

    close = market_frame()
    params = HourlyTrendParameters(5, 30, 10, 2, slope_lookback_bars=3)
    strategy = IntradaySMATrend1h(
        api_key="unit-test",
        instruments=list(close.columns),
        trend_parameters=params,
        backtest_period={"start": "2026-01-02", "end": "2026-01-10"},
        show_text_reports=False,
    )
    prices = pd.concat({"close": close, "adj_close": close}, axis=1).swaplevel(axis=1)
    empty = pd.DataFrame(index=close.index)
    strategy.instruments_data = InstrumentData(
        group_data=pd.Series("equity", index=close.columns),
        group_order=["equity"],
        strategies=empty,
        prices=prices,
        metrics=empty,
        parameters=empty,
        _skip_validation=True,
    )
    expected_signals = hourly_trend_signals(close, params)
    expected_signals.columns.name = "instrument"
    pd.testing.assert_frame_equal(strategy._compute_signals(), expected_signals)

    from backtester.portfolio.portf_data import PortfolioData

    strategy.instruments_data.add_strategy_data(strategy.strategy_name, "signals", expected_signals)
    weights = strategy._compute_weights()
    strategy.instruments_data.add_strategy_data(strategy.strategy_name, "weights", weights)
    strategy.portfolio_data = PortfolioData(
        instruments=strategy.instruments_data,
        net_asset_value=pd.Series(100_000.0, index=close.index),
        periods_per_year=252 * 7,
        _skip_initial_metrics=True,
    )
    # The CLI runner must agree with the actual Backtester, including costs,
    # drift, cash buffer, and the single execution lag.
    from backtester.portfolio.rebalance import RebalancePolicy

    strategy._rebalance_policy = RebalancePolicy(frequency=None, rebalance_on_signal_change=True)
    strategy._compute_strategy_performance()
    metrics = evaluate(close, weights, "2026-01-02", "2026-01-10")
    assert metrics["return_pct"] == pytest.approx(
        (strategy.portfolio_data.net_asset_value.iloc[-1] / 100_000 - 1) * 100
    )
    assert metrics["cost_usd"] == pytest.approx(
        strategy.portfolio_data.total_transaction_costs.sum()
    )


def test_cost_stress_reduces_net_return_and_splits_ignore_future_prices():
    close = market_frame()
    params = HourlyTrendParameters(5, 30)
    weights = weights_for(close, params)
    cheap = evaluate(close, weights, "2026-01-02", "2026-01-06", cost_bps=1)
    expensive = evaluate(close, weights, "2026-01-02", "2026-01-06", cost_bps=5)
    assert cheap["rebalances"] > 0
    assert expensive["cost_usd"] > cheap["cost_usd"]
    assert expensive["return_pct"] < cheap["return_pct"]
    changed = close.copy()
    changed.loc[changed.index >= pd.Timestamp("2026-01-06", tz="America/New_York")] *= 3
    assert evaluate(changed, weights_for(changed, params), "2026-01-02", "2026-01-06") == cheap


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast_window": 0},
        {"slow_window": 10},
        {"confirmation_bars": True},
        {"confirmation_bars": 0},
        {"entry_buffer_bps": np.nan},
        {"entry_buffer_bps": -1},
        {"slope_lookback_bars": -1},
        {"slope_lookback_bars": True},
        {"slope_lookback_bars": 1.5},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HourlyTrendParameters(**kwargs)


def test_daily_data_is_not_accepted_as_an_hourly_backtest(tmp_path):
    close = market_frame(40)
    close.index = pd.bdate_range("2026-01-01", periods=len(close), tz="UTC")
    path = tmp_path / "daily.csv"
    pd.concat({"Close": close}, axis=1).to_csv(path)
    with pytest.raises(ValueError, match="daily sample data"):
        load_close(path)
