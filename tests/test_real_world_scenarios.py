# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Deterministic market stress scenarios with independently calculated outcomes."""

import pandas as pd
import pytest

from backtester.execution import (
    ExecutionSimulator,
    FillEngine,
    FixedBpsCommission,
    FixedBpsSlippage,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from backtester.execution.contract_spec import ContractSpec
from backtester.portfolio.accounting import PortfolioLedger
from backtester.risk import PreTradeRisk


def run_scenario(prices, schedule, *, volume=10_000.0, risk=None, **engine_options):
    """Submit decisions after each close; execute against subsequent bars."""
    dates = pd.bdate_range("2024-01-02", periods=len(prices))
    close = pd.DataFrame({"SPY": prices}, index=dates)
    engine = FillEngine(fill_at="open", **engine_options)
    simulator = ExecutionSimulator(
        fill_engine=engine,
        ledger=PortfolioLedger(
            initial_cash=10_000.0,
            instruments=["SPY"],
            contract_spec_resolver=ContractSpec.equity,
        ),
        pre_trade_risk=risk,
    )

    def on_bar(date, bars, positions, nav, entry_prices):
        orders = schedule.get(dates.get_loc(date), [])
        for order in orders:
            order.created_at = date
        simulator.submit_batch(orders)

    result = simulator.run(
        close=close,
        open_=close,
        high=close * 1.01,
        low=close * 0.99,
        volume=pd.DataFrame(volume, index=dates, columns=["SPY"]),
        on_bar=on_bar,
    )
    # Reconcile every day's portfolio independently of reported returns.
    pd.testing.assert_series_equal(
        result.nav,
        result.cash + result.positions.mul(close).sum(axis=1),
        check_names=False,
    )
    assert result.nav.iloc[0] == 10_000.0
    return result, engine


@pytest.mark.parametrize("gap_price", [90.0, 75.0, 50.0])
def test_stop_loss_cannot_guarantee_stop_price_during_overnight_crash(gap_price):
    entry = Order("SPY", OrderSide.BUY, 50)
    stop = Order("SPY", OrderSide.SELL, 50, order_type=OrderType.STOP, stop_price=95)
    result, engine = run_scenario([100, 100, gap_price, gap_price], {0: [entry], 1: [stop]})

    assert [fill.fill_price for fill in engine.fill_history] == [100, gap_price]
    assert result.positions.iloc[-1]["SPY"] == 0
    assert result.cash.iloc[-1] == pytest.approx(10_000 + 50 * (gap_price - 100))
    assert stop.status == OrderStatus.FILLED


def test_round_trip_profit_deducts_both_entry_and_exit_commissions():
    result, engine = run_scenario(
        [100, 100, 110, 110],
        {0: [Order("SPY", OrderSide.BUY, 50)], 1: [Order("SPY", OrderSide.SELL, 50)]},
        commission=FixedBpsCommission(bps=10),
    )

    # $500 gross profit minus $5 entry and $5.50 exit fees.
    assert sum(fill.commission for fill in engine.fill_history) == pytest.approx(10.5)
    assert result.nav.iloc[-1] == pytest.approx(10_489.5)
    assert result.cash.iloc[-1] == pytest.approx(10_489.5)
    assert result.positions.iloc[-1]["SPY"] == 0


def test_thin_volume_entry_takes_three_sessions_without_overfilling():
    order = Order("SPY", OrderSide.BUY, 25)
    result, engine = run_scenario(
        [100] * 5,
        {0: [order]},
        volume=100,
        max_volume_participation=0.10,
    )

    assert [fill.quantity for fill in engine.fill_history] == [10, 10, 5]
    assert [fill.timestamp for fill in engine.fill_history] == list(result.nav.index[1:4])
    assert result.positions["SPY"].tolist() == [0, 10, 20, 25, 25]
    assert result.cash.iloc[-1] == 7_500
    assert result.nav.tolist() == [10_000] * 5
    assert order.status == OrderStatus.FILLED


def test_overnight_gap_rechecks_leverage_before_spending_cash():
    # Affordable at the decision close, but 150% leverage at the next open.
    order = Order("SPY", OrderSide.BUY, 100)
    result, engine = run_scenario(
        [100, 150, 150],
        {0: [order]},
        risk=PreTradeRisk(max_gross_leverage=1.0, allow_short=False),
    )

    assert order.status == OrderStatus.REJECTED
    assert engine.fill_history == []
    assert result.positions["SPY"].tolist() == [0, 0, 0]
    assert result.cash.tolist() == [10_000] * 3


def test_cash_account_rejects_sale_of_unowned_shares():
    order = Order("SPY", OrderSide.SELL, 50)
    result, engine = run_scenario([100, 90, 80], {0: [order]}, risk=PreTradeRisk(allow_short=False))

    assert order.status == OrderStatus.REJECTED
    assert engine.fill_history == []
    assert result.nav.tolist() == [10_000] * 3


def test_flat_market_round_trip_loses_spread_and_fees_on_both_legs():
    result, engine = run_scenario(
        [100] * 4,
        {0: [Order("SPY", OrderSide.BUY, 50)], 1: [Order("SPY", OrderSide.SELL, 50)]},
        slippage=FixedBpsSlippage(bps=20),
        commission=FixedBpsCommission(bps=10),
    )

    # Buy at $100.20, sell at $99.80: $20 slippage and $10 total fees.
    assert [fill.fill_price for fill in engine.fill_history] == pytest.approx([100.2, 99.8])
    assert sum(fill.commission for fill in engine.fill_history) == pytest.approx(10)
    assert result.nav.iloc[-1] == pytest.approx(9_970)
    assert result.cash.iloc[-1] == pytest.approx(9_970)
    assert result.positions.iloc[-1]["SPY"] == 0
