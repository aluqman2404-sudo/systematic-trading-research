"""
pairs_trading.py — Statistical Pairs Trading / Statistical Arbitrage Backtest
==============================================================================

Strategy Overview
-----------------
This module implements a **mean-reversion statistical arbitrage** strategy using
cointegrated equity pairs.  For each candidate pair we:

  1. Test for cointegration via the Engle-Granger two-step procedure
     (statsmodels ``coint``).  Only pairs with p-value < 0.05 are traded.
  2. Estimate a hedge ratio β via OLS on log prices: log(P₁) = α + β·log(P₂).
  3. Model the log-spread S_t = log(P₁) − β·log(P₂) as an
     Ornstein-Uhlenbeck (OU) process:
         dS_t = θ(μ − S_t)dt + σ dW_t
     and estimate θ (mean-reversion speed) and half-life τ = ln(2)/θ.
  4. Generate trading signals from the rolling 60-day z-score of S_t:
       z < −2  → long spread  (buy leg-1, sell leg-2)
       z > +2  → short spread (sell leg-1, buy leg-2)
       |z| < 0.5 → exit
       |z| > 3  → stop-loss (forced exit)
  5. Apply a 1-day signal lag to avoid look-ahead bias.
  6. Allocate equal capital across all simultaneously active pairs.
  7. Deduct 5 bps per leg per trade (10 bps round-trip per pair) for
     transaction costs.

Academic Reference
------------------
Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006).
  "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
  *The Review of Financial Studies*, 19(3), 797–827.

Usage
-----
    python pairs_trading.py
    python pairs_trading.py --start 2010-01-01 --end 2023-12-31
    python pairs_trading.py --entry-z 2.0 --exit-z 0.5 --stop-z 3.0
    python pairs_trading.py --no-plot
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf

from performance1 import compute_perf_stats, print_stats_table, plot_tearsheet

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_START: str = "2005-01-01"
DEFAULT_END: str = "2026-01-01"
COST_PER_LEG: float = 0.0005          # 5 bps per leg per trade
COINT_PVAL_THRESHOLD: float = 0.10    # 10% — appropriate for rolling windows
FORMATION_DAYS: int = 252             # 1-year look-back to test cointegration
RETEST_EVERY: int = 63                # re-estimate parameters every quarter
DEFAULT_LOOKBACK: int = 60            # rolling z-score window (trading days)
DEFAULT_ENTRY_Z: float = 2.0
DEFAULT_EXIT_Z: float = 0.5
DEFAULT_STOP_Z: float = 3.0

# Expanded universe: includes sector ETF pairs that are structurally linked
CANDIDATE_PAIRS: list[tuple[str, str]] = [
    ("SPY",  "QQQ"),    # S&P 500 vs Nasdaq-100
    ("XOM",  "CVX"),    # Integrated oil majors
    ("GLD",  "GDX"),    # Gold bullion vs gold miners
    ("GDX",  "GDXJ"),   # Senior vs junior gold miners
    ("KO",   "PEP"),    # Beverage duopoly
    ("JPM",  "BAC"),    # Money-centre banks
    ("WMT",  "TGT"),    # Big-box retail
    ("XLF",  "KBE"),    # Financials ETF vs bank ETF
    ("XLE",  "OIH"),    # Energy ETF vs oil-services ETF
    ("TLT",  "IEF"),    # Long-duration vs medium-duration Treasuries
    ("IWM",  "MDY"),    # Small-cap vs mid-cap US equities
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairSpec:
    """Metadata for a single candidate pair."""
    ticker1: str
    ticker2: str

    @property
    def label(self) -> str:
        return f"{self.ticker1}/{self.ticker2}"


@dataclass
class CointResult:
    """Output of the cointegration and OU estimation for one pair."""
    pair: PairSpec
    pvalue: float
    hedge_ratio: float      # β in log-price OLS
    intercept: float        # α in log-price OLS
    ou_theta: float         # mean-reversion speed (per day)
    half_life: float        # τ = ln(2) / θ  (days)
    is_cointegrated: bool


@dataclass
class TradeStats:
    """Aggregate trade statistics collected during a pair backtest."""
    pair_label: str
    total_trades: int = 0
    winning_trades: int = 0
    total_holding_days: int = 0
    pnl_list: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else float("nan")

    @property
    def avg_holding_days(self) -> float:
        return self.total_holding_days / self.total_trades if self.total_trades > 0 else float("nan")


# ---------------------------------------------------------------------------
# 1. Data download
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Download adjusted-close prices from Yahoo Finance with retries.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start, end : str
        Date range strings (YYYY-MM-DD).
    max_retries : int
        Number of download attempts before giving up.
    retry_delay : float
        Seconds to wait between retry attempts.

    Returns
    -------
    pd.DataFrame
        Daily adjusted-close prices, columns = tickers.
        Tickers that fail to download are silently omitted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # yfinance returns MultiIndex columns when multiple tickers
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]] if "Close" in raw.columns else raw
                if len(tickers) == 1:
                    prices.columns = tickers

            prices = prices.dropna(how="all")
            if prices.empty:
                raise ValueError("Downloaded price DataFrame is empty.")
            return prices.ffill().dropna(how="all")

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                print(f"  [download] Attempt {attempt}/{max_retries} failed: {exc}. Retrying in {retry_delay}s…")
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Failed to download data after {max_retries} attempts. Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# 2. Cointegration testing & OU estimation
# ---------------------------------------------------------------------------

def estimate_hedge_ratio(log_p1: pd.Series, log_p2: pd.Series) -> tuple[float, float]:
    """
    Estimate OLS hedge ratio: log(P1) = α + β·log(P2) + ε.

    Returns
    -------
    (intercept α, slope β)
    """
    x = sm.add_constant(log_p2.values)
    model = sm.OLS(log_p1.values, x).fit()
    intercept: float = float(model.params[0])
    beta: float = float(model.params[1])
    return intercept, beta


def estimate_ou_params(spread: pd.Series) -> tuple[float, float]:
    """
    Estimate Ornstein-Uhlenbeck mean-reversion speed θ and half-life τ.

    Method: regress ΔS_t on S_{t-1} (discrete AR(1) approximation).
      ΔS_t = a + b·S_{t-1} + ε_t
      θ ≈ −b  (per period),  τ = ln(2)/θ

    Returns
    -------
    (theta, half_life)  — both in units of the time step (days)
    """
    s = spread.dropna()
    if len(s) < 20:
        return np.nan, np.nan

    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna()
    # Align
    idx = ds.index.intersection(s_lag.index)
    ds, s_lag = ds.loc[idx], s_lag.loc[idx]

    x = sm.add_constant(s_lag.values)
    try:
        res = sm.OLS(ds.values, x).fit()
    except Exception:  # noqa: BLE001
        return np.nan, np.nan

    b: float = float(res.params[1])
    theta: float = max(-b, 1e-8)          # θ must be positive for mean-reversion
    half_life: float = np.log(2) / theta
    return theta, half_life


def test_pair_cointegration(
    prices: pd.DataFrame,
    pair: PairSpec,
    min_obs: int = 252,
) -> Optional[CointResult]:
    """
    Run Engle-Granger cointegration test and estimate OU parameters.

    Parameters
    ----------
    prices : pd.DataFrame
        Full adjusted-close price history with tickers as columns.
    pair : PairSpec
    min_obs : int
        Minimum number of joint non-NaN observations required.

    Returns
    -------
    CointResult or None if data is insufficient.
    """
    if pair.ticker1 not in prices.columns or pair.ticker2 not in prices.columns:
        print(f"  [coint] Missing price data for {pair.label}. Skipping.")
        return None

    p = prices[[pair.ticker1, pair.ticker2]].dropna()
    if len(p) < min_obs:
        print(f"  [coint] Insufficient data for {pair.label} ({len(p)} obs < {min_obs}). Skipping.")
        return None

    log_p1 = np.log(p[pair.ticker1])
    log_p2 = np.log(p[pair.ticker2])

    # Engle-Granger test: H₀ = no cointegration
    try:
        _, pvalue, _ = coint(log_p1.values, log_p2.values, trend="c", autolag="AIC")
    except Exception as exc:  # noqa: BLE001
        print(f"  [coint] Cointegration test failed for {pair.label}: {exc}. Skipping.")
        return None

    intercept, beta = estimate_hedge_ratio(log_p1, log_p2)
    spread = log_p1 - (intercept + beta * log_p2)
    theta, half_life = estimate_ou_params(spread)

    return CointResult(
        pair=pair,
        pvalue=float(pvalue),
        hedge_ratio=float(beta),
        intercept=float(intercept),
        ou_theta=float(theta),
        half_life=float(half_life),
        is_cointegrated=bool(pvalue < COINT_PVAL_THRESHOLD),
    )


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    candidates: list[tuple[str, str]],
) -> list[CointResult]:
    """
    Test all candidate pairs for cointegration over the full price history.

    Returns a list of CointResult objects for every tested pair (used for
    the reporting table).  Callers should filter on ``is_cointegrated``.
    """
    results: list[CointResult] = []
    for t1, t2 in candidates:
        pair = PairSpec(ticker1=t1, ticker2=t2)
        result = test_pair_cointegration(prices, pair)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# 3. Signal generation for a single pair
# ---------------------------------------------------------------------------

def compute_spread_zscore(
    log_p1: pd.Series,
    log_p2: pd.Series,
    intercept: float,
    beta: float,
    lookback: int,
) -> pd.Series:
    """
    Compute the rolling z-score of the log-price spread.

      spread_t = log(P1_t) − α − β·log(P2_t)
      z_t = (spread_t − μ_{t−lookback:t}) / σ_{t−lookback:t}

    The rolling statistics use a lookback-day trailing window and are computed
    on the spread itself (i.e., they adapt over time).

    Parameters
    ----------
    log_p1, log_p2 : pd.Series
    intercept : float
    beta : float
    lookback : int
        Rolling window length in trading days.

    Returns
    -------
    pd.Series  (z-score, same index as inputs)
    """
    spread = log_p1 - (intercept + beta * log_p2)
    roll_mean = spread.rolling(lookback, min_periods=lookback).mean()
    roll_std  = spread.rolling(lookback, min_periods=lookback).std(ddof=1)
    zscore = (spread - roll_mean) / (roll_std + 1e-12)
    return zscore


# ---------------------------------------------------------------------------
# 4. Backtest for a single pair
# ---------------------------------------------------------------------------

def backtest_pair(
    prices: pd.DataFrame,
    cr: CointResult,
    lookback: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    cost_per_leg: float = COST_PER_LEG,
) -> tuple[pd.Series, TradeStats]:
    """
    Simulate the pairs trading strategy for a single cointegrated pair.

    Position encoding (``position`` column):
      +1 → long spread  (long leg-1, short leg-2)
      −1 → short spread (short leg-1, long leg-2)
       0 → flat

    Signal lag: signals are generated on day t and executed on day t+1
    to prevent look-ahead bias.

    Transaction costs are applied on the day a trade opens or closes:
      cost = cost_per_leg × 2 (one cost per leg, two legs per pair)

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted-close prices (full history).
    cr : CointResult
        Cointegration result containing hedge ratio and intercept.
    lookback, entry_z, exit_z, stop_z : float
        Strategy parameters.
    cost_per_leg : float
        One-way transaction cost per leg as a fraction.

    Returns
    -------
    (daily_returns, trade_stats)
    """
    t1, t2 = cr.pair.ticker1, cr.pair.ticker2
    p = prices[[t1, t2]].dropna()
    if len(p) < lookback + 30:
        return pd.Series(dtype=float), TradeStats(pair_label=cr.pair.label)

    log_p1 = np.log(p[t1])
    log_p2 = np.log(p[t2])

    zscore = compute_spread_zscore(log_p1, log_p2, cr.intercept, cr.hedge_ratio, lookback)

    # --- Signal generation (lagged 1 day) ----------------------------------
    raw_signal = pd.Series(0, index=p.index, dtype=float)
    position = 0

    for i in range(1, len(p)):
        z = zscore.iloc[i - 1]         # yesterday's z-score drives today's trade
        if np.isnan(z):
            raw_signal.iloc[i] = 0.0
            continue

        if position == 0:
            if z < -entry_z:
                position = 1
            elif z > entry_z:
                position = -1
        elif position == 1:
            if z > exit_z or z < -stop_z:
                position = 0
        elif position == -1:
            if z < exit_z or z > stop_z:
                position = 0

        raw_signal.iloc[i] = float(position)

    # --- Daily P&L ---------------------------------------------------------
    # Returns of leg-1 (long) and leg-2 (short β shares)
    ret1 = log_p1.diff()
    ret2 = log_p2.diff()

    # Spread return: 1 unit long leg-1, β units short leg-2
    # Normalise by (1 + |β|) so both legs roughly sum to 1 unit of capital
    weight_normaliser = 1.0 + abs(cr.hedge_ratio)
    spread_return = (ret1 - cr.hedge_ratio * ret2) / weight_normaliser

    gross_pnl = raw_signal.shift(0) * spread_return  # position held through day

    # Transaction cost: applied on days when position changes
    trades_mask = raw_signal.diff().abs() > 0
    cost = trades_mask.astype(float) * cost_per_leg * 2.0   # two legs

    daily_returns = (gross_pnl - cost).dropna()

    # --- Trade statistics ---------------------------------------------------
    stats = TradeStats(pair_label=cr.pair.label)
    prev_pos = 0
    trade_open_idx: Optional[int] = None
    cumulative = (1.0 + daily_returns).cumprod()

    pos_array = raw_signal.values
    for i in range(len(pos_array)):
        cur_pos = int(pos_array[i])
        if prev_pos == 0 and cur_pos != 0:
            trade_open_idx = i
        elif prev_pos != 0 and cur_pos == 0 and trade_open_idx is not None:
            stats.total_trades += 1
            holding = i - trade_open_idx
            stats.total_holding_days += holding
            # P&L for the closed trade (cumulative return during holding)
            cum_vals = cumulative.values
            trade_ret = (cum_vals[i] / cum_vals[trade_open_idx]) - 1.0 if trade_open_idx > 0 else 0.0
            stats.pnl_list.append(trade_ret)
            if trade_ret > 0:
                stats.winning_trades += 1
            trade_open_idx = None
        prev_pos = cur_pos

    return daily_returns, stats


# ---------------------------------------------------------------------------
# 5. Rolling formation/trading window backtest (production approach)
# ---------------------------------------------------------------------------

def backtest_pair_rolling(
    prices: pd.DataFrame,
    pair: PairSpec,
    formation_days: int = FORMATION_DAYS,
    retest_every: int = RETEST_EVERY,
    lookback: int = DEFAULT_LOOKBACK,
    entry_z: float = DEFAULT_ENTRY_Z,
    exit_z: float = DEFAULT_EXIT_Z,
    stop_z: float = DEFAULT_STOP_Z,
    cost_per_leg: float = COST_PER_LEG,
    pval_threshold: float = COINT_PVAL_THRESHOLD,
) -> tuple[pd.Series, TradeStats, list[dict]]:
    """
    Rolling formation/trading backtest — the correct approach for stat arb.

    Procedure (matches Gatev, Goetzmann & Rouwenhorst 2006):
    ─────────────────────────────────────────────────────────
    Every `retest_every` trading days:
      1. Look back `formation_days` days.
      2. Run Engle-Granger cointegration test on the formation window.
      3. If p-value < threshold: estimate OLS hedge ratio on formation data.
      4. Apply those parameters to generate z-score signals for the next
         `retest_every` trading days (the "trading period").
      5. Advance to the next window and repeat.

    This avoids the problem of testing cointegration on a 20-year history
    (where structural breaks guarantee failure) and instead validates the
    relationship in a rolling 1-year window — exactly how real desks operate.

    Returns
    -------
    daily_returns : pd.Series
    trade_stats   : TradeStats
    window_log    : list[dict]  — per-window cointegration status
    """
    t1, t2 = pair.ticker1, pair.ticker2
    if t1 not in prices.columns or t2 not in prices.columns:
        return pd.Series(dtype=float), TradeStats(pair_label=pair.label), []

    p = prices[[t1, t2]].dropna()
    if len(p) < formation_days + retest_every:
        return pd.Series(dtype=float), TradeStats(pair_label=pair.label), []

    all_dates = p.index
    position_series = pd.Series(0.0, index=all_dates)
    window_log: list[dict] = []

    # Walk-forward: retest at each quarter boundary
    retest_indices = list(range(formation_days, len(p), retest_every))

    current_beta: float = 1.0
    current_intercept: float = 0.0
    active: bool = False

    for start_idx in retest_indices:
        form_slice = p.iloc[start_idx - formation_days: start_idx]
        lp1 = np.log(form_slice[t1])
        lp2 = np.log(form_slice[t2])

        try:
            _, pvalue, _ = coint(lp1.values, lp2.values, trend="c", autolag="AIC")
        except Exception:
            pvalue = 1.0

        cointegrated = bool(pvalue < pval_threshold)
        if cointegrated:
            current_intercept, current_beta = estimate_hedge_ratio(lp1, lp2)
            active = True
        else:
            active = False

        window_log.append({
            "date": all_dates[start_idx],
            "pvalue": round(pvalue, 4),
            "cointegrated": cointegrated,
            "beta": round(current_beta, 4) if cointegrated else float("nan"),
        })

        if not active:
            continue

        # Trading window
        end_idx = min(start_idx + retest_every, len(p))
        trade_idx = all_dates[start_idx: end_idx]
        if len(trade_idx) < 2:
            continue

        # Z-score: use history from [start_idx - lookback] to end of trading window
        hist_start = max(0, start_idx - lookback)
        hist = p.iloc[hist_start: end_idx]
        spread = np.log(hist[t1]) - (current_intercept + current_beta * np.log(hist[t2]))
        roll_mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
        roll_sd = spread.rolling(lookback, min_periods=lookback // 2).std(ddof=1)
        zscore_full = (spread - roll_mu) / (roll_sd + 1e-12)
        zscore_trade = zscore_full.reindex(trade_idx)

        # Signal generation (1-bar lag)
        pos = 0.0
        for i in range(1, len(trade_idx)):
            dt = trade_idx[i]
            z = zscore_trade.iloc[i - 1]
            if np.isnan(z):
                position_series[dt] = 0.0
                continue
            if pos == 0:
                if z < -entry_z:
                    pos = 1.0
                elif z > entry_z:
                    pos = -1.0
            elif pos == 1:
                if z > exit_z or z < -stop_z:
                    pos = 0.0
            elif pos == -1:
                if z < exit_z or z > stop_z:
                    pos = 0.0
            position_series[dt] = pos

    # Build daily returns from position series
    log_p1 = np.log(p[t1])
    log_p2 = np.log(p[t2])
    ret1 = log_p1.diff()
    ret2 = log_p2.diff()

    weight_norm = 1.0 + abs(current_beta)
    spread_ret = (ret1 - current_beta * ret2) / weight_norm

    gross = position_series * spread_ret
    cost_mask = position_series.diff().abs() > 0
    cost_series = cost_mask.astype(float) * cost_per_leg * 2.0
    daily_returns = (gross - cost_series).dropna()
    daily_returns = daily_returns[daily_returns.index >= all_dates[formation_days]]

    # Trade statistics
    stats = TradeStats(pair_label=pair.label)
    cumulative = (1.0 + daily_returns).cumprod()
    pos_arr = position_series.reindex(daily_returns.index).values
    prev_pos = 0
    open_idx: Optional[int] = None
    for i, cur_pos in enumerate(pos_arr):
        cur_pos = int(cur_pos)
        if prev_pos == 0 and cur_pos != 0:
            open_idx = i
        elif prev_pos != 0 and cur_pos == 0 and open_idx is not None:
            stats.total_trades += 1
            stats.total_holding_days += i - open_idx
            cv = cumulative.values
            tr = (cv[i] / cv[open_idx] - 1.0) if open_idx > 0 else 0.0
            stats.pnl_list.append(tr)
            if tr > 0:
                stats.winning_trades += 1
            open_idx = None
        prev_pos = cur_pos

    return daily_returns, stats, window_log


def summarise_window_log(pair_label: str, window_log: list[dict]) -> None:
    """Print a rolling-window cointegration summary for one pair."""
    if not window_log:
        return
    n_total = len(window_log)
    n_coint = sum(1 for w in window_log if w["cointegrated"])
    pct = n_coint / n_total if n_total else 0.0
    avg_pv = float(np.mean([w["pvalue"] for w in window_log]))
    print(f"  {pair_label:<14}  {n_coint}/{n_total} windows cointegrated "
          f"({pct:.0%})  avg p-value={avg_pv:.3f}")


# ---------------------------------------------------------------------------
# 6. Portfolio combination
# ---------------------------------------------------------------------------

def combine_portfolios(
    pair_returns: list[pd.Series],
    active_pairs_count: Optional[int] = None,
) -> pd.Series:
    """
    Combine per-pair return series into a single equal-weighted portfolio.

    On each date, the portfolio return is the simple average of all pair
    returns that have a non-NaN value on that date (dynamic equal-weighting).

    Parameters
    ----------
    pair_returns : list[pd.Series]
        Daily return series for each active pair.
    active_pairs_count : int, optional
        If provided, divide by this constant (rather than the dynamic count)
        to simulate fixed capital allocation.

    Returns
    -------
    pd.Series  — portfolio daily returns.
    """
    if not pair_returns:
        raise ValueError("No pair return series to combine.")

    combined = pd.concat(pair_returns, axis=1)
    combined.columns = [f"pair_{i}" for i in range(len(pair_returns))]

    if active_pairs_count is not None:
        # Fixed allocation: each pair always gets 1/N of capital
        portfolio = combined.fillna(0).sum(axis=1) / active_pairs_count
    else:
        # Dynamic allocation: equal weight among pairs active that day
        portfolio = combined.mean(axis=1)

    return portfolio.dropna()


# ---------------------------------------------------------------------------
# 6. Reporting helpers
# ---------------------------------------------------------------------------

def print_coint_table(results: list[CointResult]) -> None:
    """Print a formatted cointegration results table."""
    col_w = 14
    header_fmt = f"  {'Pair':<12} {'P-Value':>{col_w}} {'Hedge Ratio':>{col_w}} {'Half-Life (d)':>{col_w}} {'Cointegrated':>{col_w}}"
    sep = "=" * (12 + col_w * 4 + 4)

    print()
    print(sep)
    print("  Cointegration Test Results  (Engle-Granger, p < 0.05 to trade)")
    print(sep)
    print(header_fmt)
    print("-" * len(sep))

    for r in sorted(results, key=lambda x: x.pvalue):
        coint_flag = "YES  ✓" if r.is_cointegrated else "no"
        hl_str = f"{r.half_life:.1f}" if not np.isnan(r.half_life) else "N/A"
        print(
            f"  {r.pair.label:<12}"
            f"  {r.pvalue:>{col_w}.4f}"
            f"  {r.hedge_ratio:>{col_w}.4f}"
            f"  {hl_str:>{col_w}}"
            f"  {coint_flag:>{col_w}}"
        )

    print(sep)
    n_coint = sum(1 for r in results if r.is_cointegrated)
    print(f"  {n_coint}/{len(results)} pairs are cointegrated and will be backtested.")
    print(sep)
    print()


def print_trade_stats(all_stats: list[TradeStats]) -> None:
    """Print per-pair trade statistics."""
    col_w = 16
    sep = "=" * (14 + col_w * 3 + 2)

    print()
    print(sep)
    print("  Per-Pair Trade Statistics")
    print(sep)
    print(
        f"  {'Pair':<12}"
        f"  {'Total Trades':>{col_w}}"
        f"  {'Avg Hold (days)':>{col_w}}"
        f"  {'Win Rate':>{col_w}}"
    )
    print("-" * len(sep))

    for ts in all_stats:
        wr_str = f"{ts.win_rate:.1%}" if not np.isnan(ts.win_rate) else "N/A"
        avg_hold = f"{ts.avg_holding_days:.1f}" if not np.isnan(ts.avg_holding_days) else "N/A"
        print(
            f"  {ts.pair_label:<12}"
            f"  {ts.total_trades:>{col_w}}"
            f"  {avg_hold:>{col_w}}"
            f"  {wr_str:>{col_w}}"
        )

    print(sep)
    total_trades = sum(ts.total_trades for ts in all_stats)
    overall_wins = sum(ts.winning_trades for ts in all_stats)
    overall_wr = overall_wins / total_trades if total_trades > 0 else float("nan")
    print(f"  {'TOTAL / AVG':<12}  {total_trades:>{col_w}}  {'':>{col_w}}  {overall_wr:>{col_w - 1}.1%} ")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairs_trading.py",
        description="Statistical pairs trading backtest (Gatev et al. 2006).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",      type=str,   default=DEFAULT_START,  help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end",        type=str,   default=DEFAULT_END,    help="Backtest end date   (YYYY-MM-DD).")
    parser.add_argument("--entry-z",    type=float, default=DEFAULT_ENTRY_Z, help="Z-score threshold to enter a trade.")
    parser.add_argument("--exit-z",     type=float, default=DEFAULT_EXIT_Z,  help="Z-score threshold to exit a trade.")
    parser.add_argument("--stop-z",     type=float, default=DEFAULT_STOP_Z,  help="Z-score threshold for stop-loss.")
    parser.add_argument("--lookback",   type=int,   default=DEFAULT_LOOKBACK, help="Rolling z-score window (days).")
    parser.add_argument("--plot-file",  type=str,   default="pairs_trading_tearsheet.png", help="Output path for tearsheet PNG.")
    parser.add_argument("--no-plot",    action="store_true", help="Skip generating the tearsheet.")
    return parser


# ---------------------------------------------------------------------------
# 8. Main orchestrator
# ---------------------------------------------------------------------------

def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    End-to-end pipeline: download → test cointegration → backtest → report.
    """
    if args is None:
        args = build_arg_parser().parse_args()

    print("\n" + "=" * 70)
    print("  Statistical Pairs Trading Backtest  (Rolling Formation Windows)")
    print(f"  Period          : {args.start}  →  {args.end}")
    print(f"  Formation window: {FORMATION_DAYS} days (1 year)")
    print(f"  Retest every    : {RETEST_EVERY} days (quarterly)")
    print(f"  Z-score         : entry=±{args.entry_z}, exit=±{args.exit_z}, stop=±{args.stop_z}")
    print(f"  Coint threshold : p < {COINT_PVAL_THRESHOLD}")
    print(f"  Transaction cost: {COST_PER_LEG * 1e4:.0f} bps per leg")
    print("=" * 70 + "\n")

    # -- Collect unique tickers ---------------------------------------------
    all_tickers: list[str] = list(
        dict.fromkeys(t for pair in CANDIDATE_PAIRS for t in pair)
    )
    if "SPY" not in all_tickers:
        all_tickers.append("SPY")

    # -- Download -----------------------------------------------------------
    print(f"Downloading price data for: {', '.join(sorted(all_tickers))}")
    try:
        prices = download_prices(all_tickers, start=args.start, end=args.end)
    except RuntimeError as exc:
        print(f"\n[FATAL] {exc}")
        sys.exit(1)

    print(f"  → {len(prices)} trading days, {prices.shape[1]} tickers loaded.\n")

    # -- Rolling formation/trading backtests --------------------------------
    print(f"Running rolling cointegration tests and backtests "
          f"for {len(CANDIDATE_PAIRS)} candidate pairs…\n")

    all_pair_returns: list[pd.Series] = []
    all_trade_stats: list[TradeStats] = []

    print(f"  {'Pair':<14}  Cointegration windows  avg p-value")
    print("  " + "-" * 58)

    for t1, t2 in CANDIDATE_PAIRS:
        pair = PairSpec(ticker1=t1, ticker2=t2)
        daily_rets, trade_stats, window_log = backtest_pair_rolling(
            prices=prices,
            pair=pair,
            formation_days=FORMATION_DAYS,
            retest_every=RETEST_EVERY,
            lookback=args.lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            cost_per_leg=COST_PER_LEG,
            pval_threshold=COINT_PVAL_THRESHOLD,
        )
        summarise_window_log(pair.label, window_log)
        if daily_rets.empty or trade_stats.total_trades == 0:
            print(f"    → No trades generated for {pair.label}. Skipping.")
            continue
        all_pair_returns.append(daily_rets.rename(pair.label))
        all_trade_stats.append(trade_stats)

    print()

    if not all_pair_returns:
        print("[WARNING] No pairs generated trades. Check data availability.")
        sys.exit(0)

    if not all_pair_returns:
        print("[FATAL] All pair backtests returned empty return series.")
        sys.exit(1)

    # -- Portfolio combination ----------------------------------------------
    portfolio_returns = combine_portfolios(
        all_pair_returns,
        active_pairs_count=len(all_pair_returns),
    )

    # -- Benchmark (SPY buy-and-hold) ---------------------------------------
    if "SPY" in prices.columns:
        spy_close = prices["SPY"].dropna()
        spy_returns = spy_close.pct_change().dropna()
        spy_returns = spy_returns.reindex(portfolio_returns.index).dropna()
        portfolio_returns = portfolio_returns.reindex(spy_returns.index).dropna()
    else:
        spy_returns = None
        print("[WARNING] SPY data unavailable; no benchmark comparison.")

    # -- Performance stats --------------------------------------------------
    print("\nComputing performance statistics…")
    try:
        strat_stats = compute_perf_stats(
            portfolio_returns,
            freq=252,
            benchmark_returns=spy_returns,
        )
    except ValueError as exc:
        print(f"[FATAL] Performance calculation failed: {exc}")
        sys.exit(1)

    stats_table: dict[str, dict] = {"Pairs Strategy": strat_stats}

    if spy_returns is not None:
        try:
            spy_stats = compute_perf_stats(spy_returns, freq=252)
            stats_table["SPY (B&H)"] = spy_stats
        except ValueError:
            pass

    print_stats_table(stats_table, title="Statistical Pairs Trading — Performance Summary")

    # -- Trade statistics ---------------------------------------------------
    print_trade_stats(all_trade_stats)

    # -- Tearsheet ----------------------------------------------------------
    if not args.no_plot:
        print(f"Generating tearsheet → {args.plot_file}")
        try:
            plot_tearsheet(
                returns=portfolio_returns,
                benchmark_returns=spy_returns,
                title="Statistical Pairs Trading Strategy",
                freq=252,
                rolling_window=252,
                save_path=args.plot_file,
                show=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Tearsheet generation failed: {exc}")

    print("\nDone.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
