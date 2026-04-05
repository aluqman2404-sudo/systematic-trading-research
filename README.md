# Algorithmic Trading Strategies

A collection of quantitative trading strategy backtests spanning macro momentum, trend-following, volatility, statistical arbitrage, and fundamental value investing. Each strategy is self-contained, pulls data from Yahoo Finance, and produces a full performance tearsheet.

---

## Strategies

| File | Alpha Source | Universe | Rebalance |
|------|-------------|----------|-----------|
| `macro_system1_optimized1.py` | Trend + Carry + Vol Risk Premium | 5 ETFs + FX/vol proxies | Monthly |
| `macro_system1_v2_1.py` | Same as above with caching & fast mode | 5 ETFs + FX/vol proxies | Monthly |
| `asset_class_trend.py` | 10-month SMA momentum | 5 asset-class ETFs | Monthly |
| `vol_term_structure1.py` | VIX futures term structure (contango) | VIX/SVXY/VIXY | Daily |
| `pairs_trading1.py` | Cointegration / spread mean-reversion | US equities | Daily |
| `intraday_vwap_reversion1.py` | Intraday VWAP mean-reversion | Single equity (1h bars) | Intraday |
| `net_payout_yield_terminal1.py` | Net payout yield (div + buyback) | S&P 500 / custom | Annual |
| `ncav_effect_terminal1.py` | NCAV/Market Value (Graham net-nets) | US equities (1,500 liquid) | Annual |
| `trading.py` | SMA crossover trend | EURUSD spot | Daily |

---

## Macro System — Trend + Carry + Volatility

**Files:** `macro_system1_optimized1.py`, `macro_system1_v2_1.py`

Three independently-sized sleeves combined into a single monthly-rebalanced portfolio:

```
┌────────────────────────────────────────────────────────────────────┐
│  Trend Sleeve (40% budget)                                          │
│  Universe: SPY, IEF, GLD, DBC, EFA                                 │
│  Signal:   12-month time-series momentum                            │
│  Mode:     Long-only (top-3 positive momentum) or long/short        │
├────────────────────────────────────────────────────────────────────┤
│  Carry Sleeve (30% budget)                                          │
│  Instrument: DBV (G10 FX carry proxy)                               │
│  Filter:    Risk-on when SPY 12m mom > 0 AND VIXY 12m mom < 0       │
│             Else: SHY (short-duration cash proxy)                   │
├────────────────────────────────────────────────────────────────────┤
│  Vol Sleeve (30% budget)                                            │
│  Instrument: SVXY (short-volatility / VRP proxy)                    │
│  Filter:    Same risk-on filter as Carry                            │
│             Else: SHY                                               │
└────────────────────────────────────────────────────────────────────┘
```

**Risk controls:**
- Inverse-volatility weighting across sleeves
- Portfolio vol targeting to 10% annualised (cap 1.5×)
- Gross leverage cap at 1.25×
- Drawdown guard: blends 50% to SHY when drawdown breaches −12%

**Walk-forward validation:** optimises ~200+ parameter configs on 2012–2019 (IS), applies best config unchanged to 2020–present (OOS). Sharpe decay > 50% is flagged as an overfitting warning.

`v2` adds CSV price caching, command-line argument parsing, and a `--fast_mode` flag that reduces the parameter sweep for quicker iteration.

```bash
python macro_system1_optimized1.py
python macro_system1_v2_1.py --fast_mode
```

---

## Asset Class Trend Following

**File:** `asset_class_trend.py`

Implements Faber (2007) tactical asset allocation:
- Hold each ETF only when its price is above its trailing 210-day SMA (~10 months)
- Equal-weight across all in-trend ETFs; move to T-bills (BIL) when none are trending
- Monthly rebalance with a strict 1-month signal lag

```bash
python asset_class_trend.py
```

---

## Volatility Term Structure

**File:** `vol_term_structure1.py`

Exploits the persistent contango in VIX futures by holding short-volatility instruments when the term structure is in contango (VIX futures > spot VIX).

- Two sub-strategies with distinct entry/exit logic
- Regime analysis with contango/backwardation breakdowns
- Fallback data handling for longer-dated VIX instruments
- Rolling Sharpe and regime statistics in diagnostics output

```bash
python vol_term_structure1.py
```

---

## Pairs Trading

**File:** `pairs_trading1.py`

Statistical arbitrage based on Gatev et al. (2006). Identifies cointegrated equity pairs and trades the mean-reversion of their spread.

- Rolling formation/trading windows (walk-forward, no lookahead)
- Engle-Granger cointegration testing
- Ornstein-Uhlenbeck half-life estimation for spread dynamics
- Entry/exit on z-score thresholds; stop-loss on wide spread divergence

```bash
python pairs_trading1.py
```

---

## Intraday VWAP Reversion

**File:** `intraday_vwap_reversion1.py`

Fades intraday deviations from session VWAP on 1-hour bars.

- Session-aware VWAP resets each trading day
- Market-hours filtering (US/Eastern)
- Trade-level statistics (win rate, avg holding time, P&L per trade)
- Note: yfinance limits 1-hour data to ~2 years

```bash
python intraday_vwap_reversion1.py
```

---

## Net Payout Yield

**File:** `net_payout_yield_terminal1.py`

Ranks S&P 500 stocks annually by **net payout yield** = dividend yield + buyback yield (proxied by year-over-year reduction in shares outstanding). Holds the top decile, equal-weighted.

- 180-day fundamental lag to reduce look-ahead bias
- Liquidity filter: top 500 by trailing dollar volume
- Annual rebalance (June year-end) to match fiscal calendar
- Multi-source ticker loading: CSV → GitHub → Wikipedia fallback

```bash
python net_payout_yield_terminal1.py
```

---

## NCAV Effect (Graham Net-Nets)

**File:** `ncav_effect_terminal1.py`

Implements the Benjamin Graham net current asset value screen:

```
NCAV = Current Assets − Total Liabilities
Signal = NCAV / Market Cap  (higher = cheaper)
```

Selects top-N stocks by NCAV/MV ratio, equal-weighted, rebalanced annually.

- Point-in-time fundamentals with 180-day lag
- Multi-level cache (Parquet/pickle + per-ticker CSV) for fast re-runs
- Parallel prefetch via ThreadPoolExecutor
- Liquidity filter: top 1,500 by trailing dollar volume
- Excludes financial sector (NCAV metric not meaningful for banks)

```bash
python ncav_effect_terminal1.py
```

---

## FX SMA Crossover

**File:** `trading.py`

Classic SMA crossover on EURUSD spot:
- Long when short SMA > long SMA; short (or flat) otherwise
- 1-day signal lag, configurable transaction cost
- Compared against buy-and-hold EURUSD benchmark

```bash
python trading.py
```

---

## Shared Modules

| File | Description |
|------|-------------|
| `performance1.py` | CAGR, Vol, Sharpe, Sortino, Calmar, Max DD, VaR, CVaR, Win Rate, Beta, Information Ratio, tearsheet plots |
| `build_fundamentals_yahoo1.py` | Downloads and caches balance sheet fundamentals from yfinance for NCAV screening |

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib scipy statsmodels requests pyarrow
```

Python 3.9+ recommended.

---

## Limitations & Disclaimers

- **Survivorship bias** — strategies use current ticker universes and exclude companies that went bankrupt, were delisted, or acquired during the sample period. Live returns will be lower, particularly for the net-net strategy.
- **Transaction costs** — modelled as flat bps on turnover; actual slippage, market impact, and borrow costs are not captured.
- **Data quality** — Yahoo Finance balance-sheet data has gaps and restatements. Fundamental strategies (NCAV, NPY) should be validated against a point-in-time data vendor before production use.
- **Parameter risk** — macro system parameters are optimised on historical data. Walk-forward validation reduces but does not eliminate overfitting risk.

**For research and educational purposes only. Not investment advice.**
