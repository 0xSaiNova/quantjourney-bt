# Hourly SMA trend research — 2026-09-23

The parameter search did **not** establish a replacement for SMA(10/30).
The strategy keeps its original defaults and 25% per-instrument weight cap.
It now supports configurable SMA windows, a positive entry spread buffer, and
consecutive-bar entry confirmation, with immediate bearish-cross exits.

## Data and comparison contract

The [published W09 packet](https://backtester.quantjourney.cloud/strategies/example-weights-09-intraday-1h-sma-trend)
reports 1.50% over May 11–July 10, 2026 using engine 0.10.0. A fresh run with
engine 0.12.5 and newly downloaded adjusted Yahoo Finance bars does **not**
reproduce that figure. Neither its original input bars nor its complete
execution state are available locally. Treat the results below as a new
same-engine comparison, not a demonstrated improvement over the published run.

- Universe: SPY, QQQ, IWM; adjusted regular-session hourly closes.
- Data snapshot through September 23, 2026: January 2, 2025–September 23,
  2026; 3,002 common complete bars. Later observations are excluded.
  One bar with incomplete prices was excluded across the universe; no forward
  filling or synthetic bars were used. Indicators warm up on past bars.
- Input CSV SHA-256:
  `27351b725530f74f43e7c272dcd841295739efe69f1464506d148042b9e3a53e`.
- Capital: $100,000 per fold, starting flat; 25% maximum target per ETF,
  followed by the engine's unchanged 5% cash buffer.
- Execution: production fast weight accounting, one-bar signal delay,
  close-to-close returns, signal-change rebalancing, drift, recursive costs.
  This is an implied-trade model, without discrete share fills or volume caps.
- Cost: 1 bp per traded notional; separate 5 bp sensitivity runs.
- Positions may remain open overnight. No leverage, interest on idle cash,
  or end-of-fold liquidation is included.

## Selection and untouched periods

Freeze 36 candidates before examining holdouts: fast SMA 5/10/20, slow SMA
30/50/80, entry buffer 0/10 bp, and entry confirmation 1/2 bars. Rank on 2025
return minus absolute maximum drawdown; only the top eight enter validation.
Choose the candidate with best validation return-minus-drawdown, requiring
training return at least baseline and improved validation return and score.
May–September data never enter selection. Do not retune against these holdouts.

The selected research candidate is **SMA(20/50), no entry buffer, two-bar entry
confirmation**. The table reports raw percentage returns with identical sizing
and costs; annualized ratios are omitted.

| Period | Baseline 10/30 | Selected 20/50 | Baseline max drawdown | Selected max drawdown |
|---|---:|---:|---:|---:|
| Training: 2025 | 3.66% | 6.52% | -9.91% | -5.39% |
| Validation: Jan 2–May 8, 2026 | 3.16% | 7.60% | -8.16% | -4.42% |
| Holdout: May 11–July 10, 2026 | -0.39% | -2.82% | -2.47% | -4.19% |
| Later holdout: July 13–Sept 23, 2026 | -0.14% | -0.11% | -2.46% | -3.06% |

The candidate reduces holdout rebalance counts from 33 to 18 and from 49 to
27, respectively, but lower turnover does not make its signal more profitable.
At 5 bp costs, May–July returns are -0.76% baseline and -3.02% candidate;
later returns are -0.65% and -0.37%. There is no consistent holdout advantage.
Keep SMA(10/30) as the default and use fresh, unseen data for further research.

## Reproduce

Install the package with its `data` extra, then run from the repository:

```bash
# Download once and run the fixed chronological experiment.
.venv/bin/python tools/research_hourly_sma.py --download

# Reuse exactly the frozen CSV for subsequent comparisons.
.venv/bin/python tools/research_hourly_sma.py --input data/w09_hourly_ohlcv_20260923.csv

# Run causality, invalid-input, production-parity and cost-sensitivity checks.
.venv/bin/python -m pytest tests/test_hourly_sma_research.py -q
```

The ignored local `reports/w09-research/results.json` contains every training
trial, finalist validation results, holdout metrics, cost sensitivity, input
hash and execution assumptions. The downloader requests the fixed dates above;
provider retention and later price revisions may affect future downloads.
No cloud credentials are required for this research runner.

For a cloud-backed strategy run, optional environment settings are
`QJ_SMA_FAST`, `QJ_SMA_SLOW`, `QJ_SMA_ENTRY_BPS`, and `QJ_SMA_CONFIRM_BARS`.
Their defaults preserve SMA(10/30), zero buffer and one-bar confirmation.

## Exploratory slow-SMA slope filter

An optional entry-only filter requires `SMA(30)[t] > SMA(30)[t - lookback]`.
It uses completed bars, retains the engine's next-bar execution delay, and
does not force an exit when the slope turns negative. Bearish crossovers
still exit immediately. Entry confirmation, when configured, requires both
the crossover and slope condition on each confirmation bar. The default
lookback is zero (disabled); set `QJ_SMA_SLOPE_BARS=1` for the primary experiment.

This follow-up uses the same frozen September 23 data snapshot. All periods
were inspected previously, so these results are **exploratory**, including
the periods named "holdout" by the original experiment. SMA(10/30), position
caps, cash buffer and execution assumptions remain fixed. One completed bar
is the primary hypothesis; three and seven bars are sensitivity checks.
No new winner is selected or promoted using these observations.

| Period, 1 bp costs | No slope filter | One-bar slope filter |
|---|---:|---:|
| Training: 2025 | 3.66% | 7.24% |
| Validation: Jan 2–May 8, 2026 | 3.16% | 5.18% |
| May 11–July 10, 2026 | -0.39% | -1.56% |
| July 13–Sept 23, 2026 | -0.14% | 1.23% |

On the later period, rebalances fall from 49 to 37 and max drawdown moves
from -2.46% to -2.40%. At 10 bp costs the later return is 0.41% versus -1.29%
without the filter. May–July remains worse at 10 bp: -2.36% versus -1.22%.
Three/seven-bar slope lookbacks also improve the later period but worsen
May–July. Standalone ETF diagnostics show IWM remains a losing sleeve in
both periods under the primary filter. The evidence is mixed; keep the filter
disabled by default pending evaluation on unseen data.

```bash
.venv/bin/python tools/research_hourly_sma.py --slope-study
```

This produces `reports/w09-research/slope-results.json`, with all four slope
settings, 1/5/10 bp cost sensitivities, and individual ETF diagnostics. ETF
diagnostics run each sleeve separately at the same cap, rather than attributing
the full portfolio's returns. No network access is needed with the cached CSV.
