"""
intraday_vwap_reversion.py — Intraday VWAP Mean-Reversion Backtest (SPY, 1-Hour Bars)
=======================================================================================

Strategy Overview
-----------------
VWAP (Volume-Weighted Average Price) is the ratio of cumulative dollar volume to
cumulative share volume within a trading session.  Institutional desks use VWAP as
an execution benchmark, so price tends to mean-revert toward it intraday:

  * When price dips significantly below VWAP, buy-side algos hunting VWAP fills
    become net buyers, creating upward pressure back toward VWAP.

This backtest is deliberately **long-only**: it buys dips below VWAP and holds
until price reverts.  A short-side mirror would require fading rallies, which
loses money in trending/bull markets.  Long-only VWAP reversion is a common
intraday strategy used by prop desks and market-making firms.

  1. For each trading session, VWAP is computed from the open bar forward
     (cumulative sum of Close × Volume / cumulative Volume).
  2. After 10:30 ET (skip noisy open bars), a long entry fires if
     Close < VWAP × (1 − entry_threshold).
  3. The position is exited when price reverts to VWAP (Close ≥ VWAP), or a
     stop-loss of stop_loss_pct is hit, or at 15:00 ET unconditionally
     (no overnight risk).
  4. Transaction costs are subtracted at 1 bp per side (2 bps round-trip).

Market Microstructure Intuition
--------------------------------
VWAP reversion works best when:
  - Liquidity is high (SPY is the world's most liquid ETF).
  - Trending days are rare — on strong trend days the strategy will hit stops.
  - The mean-reversion assumption holds within a session but NOT across sessions,
    hence the hard EOD flat rule.

Data Limitation
---------------
yfinance provides only ~730 calendar days (~504 trading days) of 1-hour bars.
Results are therefore based on roughly 2 years of intraday data, which is a limited
sample.  Treat conclusions as indicative rather than statistically conclusive.

Usage
-----
    python intraday_vwap_reversion.py
    python intraday_vwap_reversion.py --ticker QQQ --entry-threshold 0.003
    python intraday_vwap_reversion.py --no-plot
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from performance1 import compute_perf_stats, print_stats_table, plot_tearsheet

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARKET_OPEN_HOUR: int = 9   # 09:30 ET
MARKET_OPEN_MIN: int = 30
SIGNAL_START_HOUR: int = 10  # signals only after 10:30 ET
SIGNAL_START_MIN: int = 30
HARD_EXIT_HOUR: int = 15    # hard flatten at 15:00 ET
HARD_EXIT_MIN: int = 0
TRADING_FREQ: int = 252     # daily periods per year for performance module


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def download_hourly(ticker: str) -> pd.DataFrame:
    """
    Download 1-hour OHLCV bars for *ticker* via yfinance.

    Returns a timezone-aware (US/Eastern) DataFrame with columns:
    ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

    yfinance may return a MultiIndex columns frame when a single ticker is
    requested via ``download()``; this function normalises to a flat index.
    """
    print(f"[data] Downloading 1-hour bars for {ticker} (period=2y) …")
    raw: pd.DataFrame = yf.download(
        ticker,
        period="2y",
        interval="1h",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker '{ticker}'. "
            "Check that the ticker is valid and markets are reachable."
        )

    # Flatten MultiIndex columns produced for single-ticker downloads
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Standardise column capitalisation
    raw.columns = [c.capitalize() if c.lower() != "adj close" else "Close"
                   for c in raw.columns]
    if "Adj close" in raw.columns:
        raw = raw.rename(columns={"Adj close": "Close"})

    # Ensure the index is tz-aware in US/Eastern
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert("US/Eastern")

    # Keep only regular trading-session rows (09:30 – 16:00 ET)
    raw = raw[
        (
            (raw.index.hour > MARKET_OPEN_HOUR)
            | ((raw.index.hour == MARKET_OPEN_HOUR) & (raw.index.minute >= MARKET_OPEN_MIN))
        )
        & (
            (raw.index.hour < 16)
        )
    ]

    # Drop rows with missing price or zero volume
    raw = raw.dropna(subset=["Close", "Volume"])
    raw = raw[raw["Volume"] > 0]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    print(
        f"[data] Loaded {len(raw):,} hourly bars | "
        f"{raw.index[0].date()} → {raw.index[-1].date()}"
    )
    return raw


# ---------------------------------------------------------------------------
# VWAP computation
# ---------------------------------------------------------------------------

def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday VWAP for each trading session, resetting at the open.

    VWAP = Σ(Close_i × Volume_i) / Σ(Volume_i)  [cumulative within day]

    Parameters
    ----------
    df : pd.DataFrame
        Hourly OHLCV frame with a tz-aware DatetimeIndex (US/Eastern).

    Returns
    -------
    pd.Series
        VWAP values aligned to df's index.
    """
    df = df.copy()
    df["_date"] = df.index.date
    df["_dollar_vol"] = df["Close"] * df["Volume"]

    df["_cum_dollar_vol"] = df.groupby("_date")["_dollar_vol"].cumsum()
    df["_cum_volume"] = df.groupby("_date")["Volume"].cumsum()

    vwap: pd.Series = df["_cum_dollar_vol"] / df["_cum_volume"]
    vwap.name = "VWAP"
    return vwap


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    vwap: pd.Series,
    entry_threshold: float,
    stop_loss_pct: float,
) -> pd.DataFrame:
    """
    Produce a bar-by-bar signal frame from hourly OHLCV + VWAP.

    Signal convention
    -----------------
    +1  → long  (entered because Close < VWAP × (1 − threshold))
     0  → flat  (long-only: never short)

    Rules applied in order for each bar:
      1. Hard exit at or after 15:00 ET — force flat.
      2. If long, check stop-loss and VWAP reversion exit.
      3. If flat and past 10:30 ET, check long entry condition.
      4. Signals are shifted by 1 bar (executed on the NEXT bar open) to
         avoid intrabar lookahead.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly OHLCV with tz-aware DatetimeIndex.
    vwap : pd.Series
        Session VWAP aligned to df.
    entry_threshold : float
        Fractional deviation from VWAP required to enter (e.g. 0.002 = 0.2 %).
    stop_loss_pct : float
        Fractional adverse move from entry price that triggers stop (e.g. 0.005).

    Returns
    -------
    pd.DataFrame
        Columns: ``signal`` (current bar position), ``entry_price``,
        ``vwap``, ``trade_return``.
    """
    n = len(df)
    close = df["Close"].values
    index = df.index

    signal_arr = np.zeros(n, dtype=np.float64)     # position held THIS bar
    entry_price_arr = np.full(n, np.nan)
    vwap_arr = vwap.values

    position: float = 0.0
    entry_px: float = np.nan

    for i in range(n):
        ts = index[i]
        is_hard_exit = (ts.hour > HARD_EXIT_HOUR) or (
            ts.hour == HARD_EXIT_HOUR and ts.minute >= HARD_EXIT_MIN
        )
        past_signal_start = (ts.hour > SIGNAL_START_HOUR) or (
            ts.hour == SIGNAL_START_HOUR and ts.minute >= SIGNAL_START_MIN
        )

        if position != 0.0:
            # --- Hard exit (EOD) ---
            if is_hard_exit:
                position = 0.0
                entry_px = np.nan

            # --- Stop-loss ---
            elif position == 1.0 and close[i] < entry_px * (1.0 - stop_loss_pct):
                position = 0.0
                entry_px = np.nan

            # --- VWAP reversion exit ---
            elif position == 1.0 and close[i] >= vwap_arr[i]:
                position = 0.0
                entry_px = np.nan

        # --- Long-only entry (only when flat, past signal start, not at EOD) ---
        if position == 0.0 and past_signal_start and not is_hard_exit:
            if close[i] < vwap_arr[i] * (1.0 - entry_threshold):
                position = 1.0
                entry_px = close[i]

        signal_arr[i] = position
        entry_price_arr[i] = entry_px

    result = pd.DataFrame(
        {
            "signal": signal_arr,
            "entry_price": entry_price_arr,
            "vwap": vwap_arr,
            "close": close,
        },
        index=index,
    )
    return result


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_intraday(
    df: pd.DataFrame,
    signal_df: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """
    Translate bar-by-bar signals into a P&L frame.

    Position is executed on the bar FOLLOWING the signal bar (1-bar lag).
    Transaction costs (``cost_bps`` per side) are applied at each position change.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly OHLCV.
    signal_df : pd.DataFrame
        Output of :func:`generate_signals`.
    cost_bps : float
        One-way transaction cost in basis points (e.g. 1.0 = 1 bp).

    Returns
    -------
    pd.DataFrame
        Columns: ``position``, ``bar_return``, ``strategy_return``,
        ``gross_return``, ``cost``.
    """
    cost_frac = cost_bps / 10_000.0

    # Shift signal by 1 bar: position is *entered* on the bar after the signal
    position = signal_df["signal"].shift(1).fillna(0.0)

    # Bar log-return of the underlying
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)

    # Gross strategy return = position held × bar return
    gross_return = position * log_ret

    # Transaction costs: incurred whenever position changes
    pos_change = position.diff().abs().fillna(0.0)
    cost = pos_change * cost_frac

    strategy_return = gross_return - cost

    result = pd.DataFrame(
        {
            "position": position,
            "bar_return": log_ret,
            "gross_return": gross_return,
            "cost": cost,
            "strategy_return": strategy_return,
        },
        index=df.index,
    )
    return result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_to_daily(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum intraday log-returns to daily totals.

    Returns a DataFrame indexed by date with columns:
    ``strategy_return``, ``gross_return``, ``cost``, ``bar_return``.
    """
    daily = (
        pnl_df[["strategy_return", "gross_return", "cost", "bar_return"]]
        .groupby(pnl_df.index.date)
        .sum()
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily


def compute_trade_stats(
    signal_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute trade-level statistics: win rate, avg holding hours, trades/day.

    A *trade* is a complete entry-to-exit cycle identified by contiguous
    non-zero positions in the (shifted) position series.

    Returns
    -------
    dict with keys: total_trades, trades_per_day, pct_days_with_trade,
    avg_holding_hours, win_rate.
    """
    pos = pnl_df["position"]
    dates = pd.Series(pnl_df.index.date, index=pnl_df.index)

    # Identify trade boundaries
    trade_id = (pos != pos.shift(1)).cumsum()
    trade_id = trade_id[pos != 0]

    if trade_id.empty:
        return {
            "total_trades": 0,
            "trades_per_day": 0.0,
            "pct_days_with_trade": 0.0,
            "avg_holding_hours": 0.0,
            "win_rate": np.nan,
        }

    holding_hours: list[float] = []
    trade_pnls: list[float] = []

    for tid, group in pnl_df[pos != 0].groupby(trade_id):
        holding_hours.append(len(group))  # each bar = 1 hour
        trade_pnls.append(group["strategy_return"].sum())

    total_trades = len(trade_pnls)
    n_trading_days = len(dates.unique())
    days_with_trade = (
        pnl_df[pos != 0]
        .groupby(pnl_df[pos != 0].index.date)
        .size()
        .pipe(lambda s: (s > 0).sum())
    )

    return {
        "total_trades": total_trades,
        "trades_per_day": total_trades / n_trading_days if n_trading_days else 0.0,
        "pct_days_with_trade": 100.0 * days_with_trade / n_trading_days
        if n_trading_days
        else 0.0,
        "avg_holding_hours": float(np.mean(holding_hours)) if holding_hours else 0.0,
        "win_rate": 100.0 * np.mean([p > 0 for p in trade_pnls])
        if trade_pnls
        else np.nan,
    }


def compute_hourly_profile(pnl_df: pd.DataFrame) -> pd.Series:
    """
    Average strategy return by hour-of-day (US/Eastern).

    Returns a Series indexed 9, 10, …, 15 (hour integers).
    """
    profile = (
        pnl_df["strategy_return"]
        .groupby(pnl_df.index.hour)
        .mean()
        * 100  # convert to bps × 100 = percent
    )
    profile.index.name = "hour_ET"
    return profile


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_intraday_stats(trade_stats: dict[str, float]) -> None:
    """Pretty-print intraday trade-level statistics."""
    print()
    print("=" * 50)
    print("  Intraday Trade Statistics")
    print("=" * 50)
    print(f"  Total trades           : {trade_stats['total_trades']:>8,.0f}")
    print(f"  Avg trades / day       : {trade_stats['trades_per_day']:>8.2f}")
    print(f"  % days with a trade    : {trade_stats['pct_days_with_trade']:>7.1f} %")
    print(f"  Avg holding time (hrs) : {trade_stats['avg_holding_hours']:>8.2f}")
    print(f"  Win rate (per trade)   : {trade_stats['win_rate']:>7.1f} %")
    print("=" * 50)
    print()


def print_hourly_profile(profile: pd.Series) -> None:
    """Print mean strategy return (in %) by hour of day as a bar chart."""
    print()
    print("=" * 50)
    print("  Avg Strategy Return by Hour (ET)")
    print("=" * 50)
    max_abs = profile.abs().max()
    bar_width = 30

    for hour, val in profile.items():
        bar_len = int(abs(val) / max_abs * bar_width) if max_abs > 0 else 0
        direction = "+" if val >= 0 else "-"
        bar = direction * bar_len
        print(f"  {hour:02d}:00  {val:+.4f}%  |{bar}")
    print("=" * 50)
    print()


# ---------------------------------------------------------------------------
# SPY daily benchmark
# ---------------------------------------------------------------------------

def download_spy_daily(start: str, end: str) -> pd.Series:
    """
    Download SPY daily adjusted close returns over [start, end] (inclusive).

    Returns a pd.Series of log-returns indexed by date.
    """
    raw = yf.download(
        "SPY",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.capitalize() for c in raw.columns]
    close = raw["Close"].dropna()
    returns = np.log(close / close.shift(1)).dropna()
    returns.index = pd.to_datetime(returns.index)
    if returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    returns.name = "SPY_daily"
    return returns


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    ticker: str = "SPY",
    entry_threshold: float = 0.003,
    stop_loss_pct: float = 0.003,
    cost_bps: float = 1.0,
    plot_file: str = "intraday_vwap_tearsheet.png",
    no_plot: bool = False,
) -> None:
    """
    Run the full intraday VWAP reversion backtest pipeline.

    Steps
    -----
    1. Download 1-hour bars.
    2. Compute session VWAP.
    3. Generate signals (entry / stop / reversion exit / EOD exit).
    4. Run backtest with transaction costs.
    5. Aggregate to daily returns.
    6. Print performance & trade stats; optionally plot tearsheet.
    """
    print()
    print("=" * 60)
    print("  Intraday VWAP Reversion Backtest")
    print("=" * 60)
    print(f"  Ticker           : {ticker}")
    print(f"  Entry threshold  : {entry_threshold * 100:.2f}%")
    print(f"  Stop loss        : {stop_loss_pct * 100:.2f}%")
    print(f"  Cost (one-way)   : {cost_bps:.1f} bps")
    print()
    print(
        "  NOTE: yfinance 1-hour data spans ~2 years (~504 trading days).\n"
        "  Results are based on a limited sample. Interpret with caution.\n"
    )

    # ------------------------------------------------------------------
    # 1. Download intraday data
    # ------------------------------------------------------------------
    hourly_df: pd.DataFrame = download_hourly(ticker)

    # ------------------------------------------------------------------
    # 2. Compute VWAP
    # ------------------------------------------------------------------
    print("[signal] Computing session VWAP …")
    vwap: pd.Series = compute_session_vwap(hourly_df)

    # ------------------------------------------------------------------
    # 3. Generate signals
    # ------------------------------------------------------------------
    print("[signal] Generating entry / exit signals …")
    signal_df: pd.DataFrame = generate_signals(
        hourly_df, vwap, entry_threshold, stop_loss_pct
    )

    # ------------------------------------------------------------------
    # 4. Backtest
    # ------------------------------------------------------------------
    print("[backtest] Running backtest …")
    pnl_df: pd.DataFrame = backtest_intraday(hourly_df, signal_df, cost_bps)

    # ------------------------------------------------------------------
    # 5. Aggregate to daily
    # ------------------------------------------------------------------
    daily_df: pd.DataFrame = aggregate_to_daily(pnl_df)
    strategy_daily: pd.Series = daily_df["strategy_return"]
    strategy_daily.name = f"VWAP Reversion ({ticker})"

    # ------------------------------------------------------------------
    # 6. SPY daily benchmark aligned to strategy period
    # ------------------------------------------------------------------
    start_str = str(daily_df.index[0].date())
    end_str = str((daily_df.index[-1] + pd.Timedelta(days=1)).date())
    spy_daily: pd.Series = download_spy_daily(start_str, end_str)

    # Align on common dates
    common_dates = strategy_daily.index.intersection(spy_daily.index)
    strategy_aligned = strategy_daily.loc[common_dates]
    spy_aligned = spy_daily.loc[common_dates]

    # ------------------------------------------------------------------
    # 7. Performance statistics
    # ------------------------------------------------------------------
    print("[perf] Computing performance statistics …")
    strat_stats: dict = compute_perf_stats(
        strategy_aligned, freq=TRADING_FREQ, benchmark_returns=spy_aligned
    )
    spy_stats: dict = compute_perf_stats(spy_aligned, freq=TRADING_FREQ)

    print_stats_table(
        {
            f"VWAP Reversion ({ticker})": strat_stats,
            "SPY Buy-and-Hold": spy_stats,
        },
        title="Intraday VWAP Reversion — Daily Aggregated P&L",
    )

    # ------------------------------------------------------------------
    # 8. Trade-level statistics
    # ------------------------------------------------------------------
    trade_stats = compute_trade_stats(signal_df, pnl_df)
    print_intraday_stats(trade_stats)

    # ------------------------------------------------------------------
    # 9. Hourly return profile
    # ------------------------------------------------------------------
    hourly_profile: pd.Series = compute_hourly_profile(pnl_df)
    print_hourly_profile(hourly_profile)

    # ------------------------------------------------------------------
    # 10. Tearsheet
    # ------------------------------------------------------------------
    if not no_plot:
        print(f"[plot] Saving tearsheet to '{plot_file}' …")
        plot_tearsheet(
            returns=strategy_aligned,
            benchmark_returns=spy_aligned,
            title=f"Intraday VWAP Reversion — {ticker} (1-Hour Bars)",
            freq=TRADING_FREQ,
            rolling_window=TRADING_FREQ,
            save_path=plot_file,
            show=False,
        )
        print(f"[plot] Tearsheet saved.")
    else:
        print("[plot] Skipped (--no-plot).")

    print()
    print("=" * 60)
    print("  Backtest complete.")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="intraday_vwap_reversion",
        description=(
            "Intraday VWAP mean-reversion backtest using 1-hour yfinance bars. "
            "Computes performance statistics vs SPY buy-and-hold and plots a tearsheet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Ticker symbol to backtest (must have liquid 1-hour data).",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.003,
        dest="entry_threshold",
        help=(
            "Fractional deviation from VWAP required for entry. "
            "E.g. 0.002 = enter long when Close < VWAP × 0.998."
        ),
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.003,
        dest="stop_loss_pct",
        help="Fractional adverse move from entry price that triggers a stop-loss exit.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=1.0,
        dest="cost_bps",
        help="One-way transaction cost in basis points (2× round-trip).",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="intraday_vwap_tearsheet.png",
        dest="plot_file",
        help="File path for the saved tearsheet PNG.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        help="Suppress tearsheet generation.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(
        ticker=args.ticker,
        entry_threshold=args.entry_threshold,
        stop_loss_pct=args.stop_loss_pct,
        cost_bps=args.cost_bps,
        plot_file=args.plot_file,
        no_plot=args.no_plot,
    )
