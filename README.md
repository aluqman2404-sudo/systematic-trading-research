# Systematic Trading Research

A collection of systematic strategy backtests spanning macro, trend-following, and
fundamental factor investing. Each strategy is self-contained, runs off Yahoo Finance
data, and produces a full performance tearsheet.

---

## Strategies

| File | Alpha source | Universe | Rebalance |
|------|-------------|----------|-----------|
| `macro_system1_optimized.py` | Trend + Carry + Vol Risk Premium | 5 ETFs + FX/vol proxies | Monthly |
| `asset_class_trend_following.py` | 10-month SMA momentum | 5 asset-class ETFs | Monthly |
| `net_payout_yield_terminal.py` | Net payout yield (div + buyback) | S&P 500 / custom | Annual |
| `ncav_effect_terminal.py` | NCAV/Market Value (Graham net-nets) | US equities (1,500 liquid) | Annual |
| `trading2.py` | SMA crossover trend | EURUSD spot | Daily |

---

## Macro System — Trend + Carry + Volatility

**File:** `macro_system1_optimized.py`

### Architecture

Three independently-sized sleeves combined into a single monthly-rebalanced portfolio:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Trend Sleeve (40% budget)                                           │
│  Universe: SPY, IEF, GLD, DBC, EFA                                  │
│  Signal:   12-month time-series momentum                             │
│  Mode:     Long-only (top-3 positive momentum) or long/short         │
├─────────────────────────────────────────────────────────────────────┤
│  Carry Sleeve (30% budget)                                           │
│  Instrument: DBV (G10 FX carry proxy)                                │
│  Filter:    Risk-on when SPY 12m mom > 0 AND VIXY 12m mom < 0        │
│             Else: SHY (short-duration cash proxy)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Vol Sleeve (30% budget)                                             │
│  Instrument: SVXY (short-volatility / VRP proxy)                     │
│  Filter:    Same risk-on filter as Carry                             │
│             Else: SHY                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Risk Controls

- **Sleeve risk balancing** — inverse-volatility weights normalise each sleeve's contribution
- **Portfolio vol targeting** — scales returns to 10% annualised vol target (cap 1.5×)
- **Gross leverage cap** — maximum 1.25× notional exposure
- **Drawdown guard** — blends 50% to defensive (SHY) when drawdown breaches −12%

### Walk-Forward Validation

The optimiser searches ~200+ parameter configurations on the **in-sample (IS) period**
(2012–2019). The single best IS config is then applied unchanged to the
**out-of-sample (OOS) period** (2020–present), with IS and OOS results reported
side-by-side. A Sharpe decay > 50% is flagged as an overfitting warning.

```bash
python macro_system1_optimized.py
```

---

## Asset Class Trend Following

**File:** `asset_class_trend_following.py`

Implements Faber (2007) tactical asset allocation rule:

- Hold each ETF only when its price is above its trailing **210-day SMA** (~10 months)
- Equal-weight across all in-trend ETFs; move to T-bills (BIL) when none are trending
- Monthly rebalance with a strict 1-month signal lag

```bash
python asset_class_trend_following.py
```

---

## Net Payout Yield

**File:** `net_payout_yield_terminal.py`

Ranks S&P 500 stocks annually by **net payout yield** = dividend yield + buyback yield
(proxy: year-over-year reduction in shares outstanding). Holds the top decile, equal-weighted.

Key design choices:
- 180-day fundamental lag reduces look-ahead bias on share counts
- Liquidity filter: top 500 by trailing dollar volume
- Annual rebalance (June year-end) to match fiscal calendar

```bash
python net_payout_yield_terminal.py --tickers-file tickers_us_1500.txt
```

---

## NCAV / Market Value (Graham Net-Nets)

**File:** `ncav_effect_terminal.py`

Implements the Ben Graham net-net screen:

```
NCAV = Current Assets − Total Liabilities
Signal = NCAV / Market Cap  (higher is cheaper)
```

Select the top N stocks by NCAV/MV ratio, equal-weight, rebalance annually.

Engineering highlights:
- **Point-in-time** fundamentals — 180-day lag prevents look-ahead
- **Multi-level cache** (Parquet/pickle + per-ticker CSV) — fast re-runs
- **Parallel prefetch** — ThreadPoolExecutor for balance-sheet downloads
- **Liquidity filter** — top 1,500 by trailing dollar volume
- Excludes financial sector (NCAV metric not meaningful for banks)

```bash
python ncav_effect_terminal.py --tickers-file tickers_us_1500.txt \
       --fundamentals-file fundamentals.csv
```

---

## FX Trend Following

**File:** `trading2.py`

Classic SMA crossover on EURUSD spot:
- Long when 50-day SMA > 200-day SMA; short otherwise
- 1-day signal lag, 1 pip round-trip cost modelled
- Compared against buy-and-hold EURUSD benchmark

```bash
python trading2.py
```

---

## Shared Performance Module

**File:** `performance.py`

All strategies import from this module, which provides:

| Function | Description |
|----------|-------------|
| `compute_perf_stats()` | CAGR, Vol, Sharpe, Sortino, Calmar, Max DD, Avg DD Duration, Skewness, Excess Kurtosis, VaR 95%, CVaR 95%, Win Rate, Beta, Correlation, Tracking Error, Information Ratio |
| `print_stats_table()` | Side-by-side comparison table (strategy vs benchmark) |
| `plot_tearsheet()` | 3-panel figure: equity curve + drawdown + rolling Sharpe |

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib scipy
```

Python 3.11+ recommended.

---

## Limitations & Disclaimers

- **Survivorship bias** — all strategies use current ticker universes and exclude
  companies that went bankrupt, were delisted, or were acquired during the sample
  period. Live returns will be lower, particularly for the net-net strategy.
- **Transaction costs** — modelled as flat bps on turnover; actual slippage, market
  impact, and borrow costs are not captured.
- **Data quality** — Yahoo Finance balance-sheet data has gaps and restatements.
  The fundamental strategies (NCAV, NPY) should be run with a verified point-in-time
  data vendor for production use.
- **Parameter risk** — macro system parameters are optimised on historical data.
  Walk-forward validation reduces but does not eliminate overfitting risk.
- **For research and educational purposes only. Not investment advice.**
