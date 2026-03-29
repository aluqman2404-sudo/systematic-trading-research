"""
FX Trend Following — EURUSD Moving-Average Crossover
=====================================================
Strategy
--------
- Instrument : EUR/USD spot (EURUSD=X via Yahoo Finance)
- Signal     : Dual SMA crossover (50-day vs 200-day)
               Long  when SMA_50 > SMA_200 (uptrend)
               Short when SMA_50 < SMA_200 (downtrend)
               Flat  during the first 200-day warm-up period
- Execution  : Daily rebalancing, 1-day signal lag (no lookahead)
- Costs      : Configurable round-trip spread (default 1 pip = 0.01%)
- Leverage   : Parameterisable; default 1× (unlevered)

Metrics reported
----------------
CAGR, Annualised Vol, Sharpe, Sortino, Calmar, Max Drawdown,
Avg DD Duration, Skewness, Excess Kurtosis, VaR 95%, CVaR 95%, Win Rate

Benchmark  : Buy-and-hold EURUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from performance import compute_perf_stats, print_stats_table, plot_tearsheet


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
TICKER = "EURUSD=X"
START = "2005-01-01"
END: str | None = None          # None → today
SHORT_SMA = 50                  # days
LONG_SMA = 200                  # days
TRANSACTION_COST = 0.0001       # 1 pip per round-trip (0.01%)
LEVERAGE = 1.0                  # 1× = fully-funded, unleveraged
INITIAL_CAPITAL = 100_000


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def download_fx(ticker: str, start: str, end: str | None, max_retries: int = 3) -> pd.Series:
    """
    Download FX daily close from Yahoo Finance with retry logic.
    Returns a pd.Series (Close prices) with a timezone-naive DatetimeIndex.
    """
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned empty DataFrame.")
            col = "Close"
            if col not in raw.columns:
                raise RuntimeError(f"Expected '{col}' column; got {list(raw.columns)}.")
            px = raw[col].dropna()
            px.index = pd.to_datetime(px.index).tz_localize(None)
            px = px.sort_index()
            print(f"Downloaded {len(px)} daily bars for {ticker} "
                  f"({px.index[0].date()} → {px.index[-1].date()})")
            return px
        except Exception as exc:
            print(f"Attempt {attempt}/{max_retries} failed: {exc}", file=sys.stderr)

    raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts.")


# ---------------------------------------------------------------------------
# SIGNAL + BACKTEST
# ---------------------------------------------------------------------------
def compute_signals(px: pd.Series, short_sma: int, long_sma: int) -> pd.DataFrame:
    """
    Compute SMA crossover signals.

    Returns a DataFrame with columns:
      close, sma_short, sma_long, signal (-1 / 0 / +1), position (lagged signal)
    """
    df = px.rename("close").to_frame()
    df["sma_short"] = df["close"].rolling(short_sma).mean()
    df["sma_long"] = df["close"].rolling(long_sma).mean()

    df["signal"] = 0
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
    df.loc[df["sma_short"] < df["sma_long"], "signal"] = -1

    # 1-day lag → no lookahead bias
    df["position"] = df["signal"].shift(1).fillna(0)

    # Flat until both SMAs are warm
    warm_up_mask = df["sma_long"].isna()
    df.loc[warm_up_mask, "position"] = 0

    return df.dropna(subset=["sma_long"])


def backtest(
    df: pd.DataFrame,
    leverage: float = LEVERAGE,
    tx_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Vectorised daily P&L backtest.

    Uses simple (arithmetic) returns for clarity.
    Transaction cost applied on each day a position change occurs.

    Returns the input DataFrame augmented with:
      daily_ret, pos_change, tx_cost_applied, strat_ret, equity
    """
    df = df.copy()
    df["daily_ret"] = df["close"].pct_change().fillna(0.0)

    # Identify trade days (position flips)
    df["pos_change"] = df["position"].diff().abs().fillna(0.0)
    df["tx_cost_applied"] = df["pos_change"] * tx_cost

    # Strategy return: leveraged position × daily return − transaction cost
    df["strat_ret"] = df["position"] * df["daily_ret"] * leverage - df["tx_cost_applied"]

    # Equity curve
    df["equity"] = (1 + df["strat_ret"]).cumprod() * INITIAL_CAPITAL

    return df


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
def build_stats(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build performance dicts for strategy and buy-and-hold benchmark.
    Returns (strategy_stats, benchmark_stats).
    """
    strat_rets = df["strat_ret"].copy()
    bm_rets = df["daily_ret"].copy()   # buy-and-hold EURUSD

    strat_stats = compute_perf_stats(strat_rets, freq=252, benchmark_returns=bm_rets)
    bm_stats = compute_perf_stats(bm_rets, freq=252)
    return strat_stats, bm_stats


def print_signal_summary(df: pd.DataFrame) -> None:
    """Print a summary of position exposure over the backtest."""
    n = len(df)
    n_long = int((df["position"] == 1).sum())
    n_short = int((df["position"] == -1).sum())
    n_flat = int((df["position"] == 0).sum())
    n_trades = int((df["pos_change"] > 0).sum())

    print("\n--- Signal Summary ---")
    print(f"Total days        : {n}")
    print(f"Long days         : {n_long} ({n_long / n:.1%})")
    print(f"Short days        : {n_short} ({n_short / n:.1%})")
    print(f"Flat days         : {n_flat} ({n_flat / n:.1%})")
    print(f"Position changes  : {n_trades}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FX Trend Following Backtest (EURUSD SMA Crossover)")
    p.add_argument("--ticker", default=TICKER)
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--short-sma", type=int, default=SHORT_SMA)
    p.add_argument("--long-sma", type=int, default=LONG_SMA)
    p.add_argument("--leverage", type=float, default=LEVERAGE)
    p.add_argument("--tx-cost", type=float, default=TRANSACTION_COST,
                   help="Round-trip transaction cost as decimal (e.g. 0.0001 = 1 pip).")
    p.add_argument("--plot-file", default="fx_trend_tearsheet.png")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.short_sma >= args.long_sma:
        raise ValueError("--short-sma must be strictly less than --long-sma.")

    # ---- Download ----
    px = download_fx(args.ticker, args.start, args.end)

    # ---- Signals ----
    df_sig = compute_signals(px, args.short_sma, args.long_sma)

    # ---- Backtest ----
    df_res = backtest(df_sig, leverage=args.leverage, tx_cost=args.tx_cost)

    # ---- Stats ----
    strat_stats, bm_stats = build_stats(df_res)

    print_signal_summary(df_res)

    print_stats_table(
        {"Strategy (SMA Cross)": strat_stats, "Buy-and-Hold EURUSD": bm_stats},
        title=f"FX Trend Following — {args.ticker}  "
              f"(SMA {args.short_sma}/{args.long_sma}, "
              f"{df_res.index[0].year}–{df_res.index[-1].year})",
    )

    # ---- Plot ----
    if not args.no_plot:
        plot_tearsheet(
            returns=df_res["strat_ret"],
            benchmark_returns=df_res["daily_ret"],
            title=f"FX Trend Following — {args.ticker} "
                  f"(SMA {args.short_sma}/{args.long_sma})",
            freq=252,
            rolling_window=252,
            save_path=args.plot_file,
            show=True,
        )


if __name__ == "__main__":
    main()
