"""
Asset Class Trend Following
===========================
Strategy
--------
Universe  : SPY (US Equities), EFA (Intl Equities), IEF (Bonds),
            VNQ (REITs), GSG (Commodities)
Rule      : Hold each ETF only when its adjusted close is above its
            trailing 210-day SMA (~10 calendar months of trading days).
            Equal weight across all ETFs currently in trend.
            Move to 100% cash (T-bill proxy: BIL) when no ETF is trending.
Rebalance : Monthly (last trading day of each month).
Lag       : Signal computed at month-end, weights applied the following
            month — zero lookahead bias.
Costs     : 10 bps one-way per rebalance (applied on turnover).

Reference : Faber, M. (2007). "A Quantitative Approach to Tactical Asset
            Allocation." Journal of Wealth Management.

Metrics reported
----------------
CAGR, Ann. Vol, Sharpe, Sortino, Calmar, Max Drawdown, Avg DD Duration,
Skewness, Excess Kurtosis, VaR 95%, CVaR 95%, Win Rate
Plus benchmark-relative: Beta, Correlation, Tracking Error, Info Ratio

Benchmark : SPY buy-and-hold (total return, same period)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from performance1 import compute_perf_stats, print_stats_table, plot_tearsheet


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
UNIVERSE = ["SPY", "EFA", "IEF", "VNQ", "GSG"]
BENCHMARK = "SPY"
CASH_PROXY = "BIL"          # iShares 1–3 Month T-Bill ETF (cash-like)
SMA_DAYS = 210              # ~10-month SMA
COST_BPS = 10               # one-way transaction cost per rebalance
START = "2000-01-01"
END: str | None = None
INITIAL_CAPITAL = 100_000


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def download_prices(tickers: list[str], start: str, end: str | None,
                    max_retries: int = 3) -> pd.DataFrame:
    """
    Download adjusted close prices with retry logic.
    Returns a DataFrame: index=date, columns=tickers.
    Missing tickers are silently dropped after all retries.
    """
    unique = list(dict.fromkeys(tickers))

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                unique,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned empty DataFrame.")

            if "Adj Close" not in raw.columns:
                raise RuntimeError(f"Unexpected columns: {list(raw.columns)}")

            px = raw["Adj Close"]
            if isinstance(px, pd.Series):
                px = px.to_frame(name=unique[0])

            px.index = pd.to_datetime(px.index).tz_localize(None)
            px = px.dropna(axis=1, how="all").ffill().dropna(how="all")

            failed = sorted(set(unique) - set(px.columns))
            if failed:
                print(f"Note: dropped tickers with no data: {failed}", file=sys.stderr)

            return px

        except Exception as exc:
            print(f"Download attempt {attempt}/{max_retries} failed: {exc}", file=sys.stderr)

    raise RuntimeError(f"Failed to download prices after {max_retries} attempts.")


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def run_backtest(
    px_d: pd.DataFrame,
    universe: list[str],
    sma_days: int = SMA_DAYS,
    cost_bps: float = COST_BPS,
    cash_proxy: str | None = CASH_PROXY,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run the SMA trend-following backtest.

    Parameters
    ----------
    px_d : pd.DataFrame
        Daily adjusted close prices.
    universe : list[str]
        Tickers to include in the strategy universe.
    sma_days : int
        Rolling SMA window in trading days.
    cost_bps : float
        One-way transaction cost in basis points.
    cash_proxy : str or None
        Ticker to use as cash substitute when no asset is trending.
        If None or not in px_d, earns 0% when out of the market.

    Returns
    -------
    port_ret_m : pd.Series
        Monthly net strategy returns.
    weights_m : pd.DataFrame
        Monthly end-of-period weights (before applying to next month).
    """
    # Use only universe tickers present in price data
    avail = [t for t in universe if t in px_d.columns]
    if len(avail) < 2:
        raise RuntimeError(f"Fewer than 2 universe tickers have price data: {avail}")

    if len(avail) < len(universe):
        missing = sorted(set(universe) - set(avail))
        print(f"Note: universe tickers missing from price data: {missing}", file=sys.stderr)

    px_univ = px_d[avail].copy()

    # Daily 210-day SMA on universe assets
    sma_d = px_univ.rolling(sma_days).mean()

    # Monthly resampling (last trading day of each month)
    px_m = px_univ.resample("ME").last()
    sma_m = sma_d.resample("ME").last()
    rets_m = px_m.pct_change()

    # Cash proxy monthly returns
    if cash_proxy and cash_proxy in px_d.columns:
        cash_m = px_d[cash_proxy].resample("ME").last().pct_change()
    else:
        cash_m = pd.Series(0.0, index=px_m.index)

    # Trend signal: 1 if price > SMA, else 0
    sig_m = (px_m > sma_m).astype(float)

    # Equal weight across trending assets; 100% cash when none trending
    n_long = sig_m.sum(axis=1)
    weights_m = sig_m.div(n_long.replace(0, np.nan), axis=0).fillna(0.0)
    in_cash = n_long == 0

    # 1-month lag: use last month's weights for this month's returns (no lookahead)
    w_lag = weights_m.shift(1)
    cash_lag = in_cash.shift(1).fillna(True)
    rets_m_aligned = rets_m.loc[w_lag.index]
    cash_aligned = cash_m.reindex(w_lag.index).fillna(0.0)

    # Gross portfolio return
    port_gross = (w_lag * rets_m_aligned).sum(axis=1)
    port_gross = port_gross.where(~cash_lag, cash_aligned)

    # Transaction costs: bps on absolute weight change per rebalance
    cost_decimal = cost_bps / 10_000
    turnover = weights_m.diff().abs().sum(axis=1)
    port_net = port_gross - turnover.shift(1).fillna(0.0) * cost_decimal

    # Trim warm-up period (first SMA window)
    first_valid = sma_m.dropna(how="all").index[0]
    port_net = port_net.loc[port_net.index >= first_valid].dropna()

    return port_net, weights_m


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
def print_weight_summary(weights_m: pd.DataFrame, last_n: int = 12) -> None:
    """Print the most recent monthly weight allocations."""
    print(f"\n--- Recent Portfolio Weights (last {last_n} months) ---")
    print(weights_m.tail(last_n).round(2).to_string())


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asset Class Trend Following (SMA-based)")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--sma-days", type=int, default=SMA_DAYS,
                   help="Trailing SMA window in trading days (default 210 ≈ 10 months).")
    p.add_argument("--cost-bps", type=float, default=COST_BPS,
                   help="One-way transaction cost per rebalance in bps.")
    p.add_argument("--plot-file", default="trend_following_tearsheet.png")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Download ----
    all_tickers = sorted(set(UNIVERSE + [BENCHMARK, CASH_PROXY]))
    print(f"Downloading prices for: {all_tickers}")
    px_d = download_prices(all_tickers, args.start, args.end)
    print(f"Price history: {px_d.index[0].date()} → {px_d.index[-1].date()}, "
          f"{px_d.shape[1]} tickers")

    # ---- Backtest ----
    port_ret_m, weights_m = run_backtest(
        px_d=px_d,
        universe=UNIVERSE,
        sma_days=args.sma_days,
        cost_bps=args.cost_bps,
    )

    # ---- Benchmark: SPY buy-and-hold (monthly returns, same period) ----
    if BENCHMARK in px_d.columns:
        spy_px_m = px_d[BENCHMARK].resample("ME").last()
        spy_ret_m = spy_px_m.pct_change().reindex(port_ret_m.index).dropna()
        port_aligned = port_ret_m.reindex(spy_ret_m.index).dropna()
        spy_aligned = spy_ret_m.reindex(port_aligned.index)
    else:
        print("Warning: SPY not available for benchmark comparison.", file=sys.stderr)
        port_aligned = port_ret_m
        spy_aligned = None

    # ---- Stats ----
    strat_stats = compute_perf_stats(
        port_aligned, freq=12, benchmark_returns=spy_aligned
    )

    all_stats: dict[str, dict] = {"Trend Following": strat_stats}

    if spy_aligned is not None:
        bm_stats = compute_perf_stats(spy_aligned, freq=12)
        all_stats["SPY (B&H)"] = bm_stats

    print_stats_table(
        all_stats,
        title=f"Asset Class Trend Following  (SMA {args.sma_days}d, "
              f"{port_aligned.index[0].year}–{port_aligned.index[-1].year})",
    )

    print_weight_summary(weights_m)

    # ---- Plot ----
    if not args.no_plot:
        plot_tearsheet(
            returns=port_aligned,
            benchmark_returns=spy_aligned,
            title=f"Asset Class Trend Following (SMA {args.sma_days}d)",
            freq=12,
            rolling_window=12,
            save_path=args.plot_file,
            show=True,
        )


if __name__ == "__main__":
    main()
