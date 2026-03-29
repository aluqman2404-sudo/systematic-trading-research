import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_frame(x):
    if isinstance(x, pd.Series):
        return x.to_frame()
    return x


def normalize_index_to_naive_utc(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    idx = pd.to_datetime(series.index, errors="coerce", utc=True)
    mask = ~idx.isna()
    s = series.loc[mask].copy()
    s.index = idx[mask].tz_localize(None)
    return s.sort_index()


def to_naive_ts(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return t.tz_convert("UTC").tz_localize(None)
    return t


def load_tickers(tickers_file: str | None) -> list[str]:
    if tickers_file:
        path = Path(tickers_file)
        if not path.exists():
            raise FileNotFoundError(f"Tickers file not found: {tickers_file}")
        tickers = [line.strip().upper().replace(".", "-") for line in path.read_text().splitlines() if line.strip()]
        if not tickers:
            raise ValueError("Ticker file is empty.")
        return sorted(list(dict.fromkeys(tickers)))

    # Default universe proxy: S&P 500 constituents (CSV sources first, no lxml needed)
    csv_sources = [
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    ]
    for url in csv_sources:
        try:
            df = pd.read_csv(url)
            if "Symbol" in df.columns:
                tickers = (
                    df["Symbol"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                tickers = sorted(list(dict.fromkeys([t for t in tickers if t])))
                if tickers:
                    return tickers
        except Exception:
            continue

    # Last fallback: Wikipedia table parsing (requires lxml/html parser extras)
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers = sorted(list(dict.fromkeys([t for t in tickers if t])))
        if tickers:
            return tickers
    except Exception:
        pass

    raise RuntimeError(
        "Could not load default ticker universe from online sources.\n"
        "Use --tickers-file with one ticker per line, or install parser deps: pip install lxml html5lib."
    )


def download_prices_and_actions(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        actions=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("No data downloaded from Yahoo Finance.")

    close = to_frame(data["Close"]).sort_index().dropna(how="all")
    volume = to_frame(data["Volume"]).sort_index().reindex(close.index)

    if "Dividends" in data.columns:
        dividends = to_frame(data["Dividends"]).sort_index().reindex(close.index).fillna(0.0)
    else:
        dividends = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # Keep only tickers with price history
    valid_cols = close.columns[close.notna().sum() > 252]
    close = close[valid_cols].ffill()
    volume = volume[valid_cols].fillna(0.0)
    dividends = dividends[valid_cols].fillna(0.0)

    if close.shape[1] == 0:
        raise RuntimeError("No ticker has sufficient price history after cleaning.")

    return close, volume, dividends


def load_or_fetch_shares(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    memory_cache: dict[str, pd.Series],
) -> pd.Series:
    if ticker in memory_cache:
        return memory_cache[ticker]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}.csv"

    if cache_path.exists():
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        s = normalize_index_to_naive_utc(s)
        memory_cache[ticker] = s
        return s

    try:
        s = yf.Ticker(ticker).get_shares_full(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if s is None or len(s) == 0:
            s = pd.Series(dtype=float)
        else:
            s = pd.to_numeric(pd.Series(s), errors="coerce").dropna().sort_index()
            s = normalize_index_to_naive_utc(s)
    except Exception:
        s = pd.Series(dtype=float)

    if len(s) > 0:
        s.to_csv(cache_path, header=["shares_outstanding"])

    memory_cache[ticker] = s
    return s


def asof_value(series: pd.Series, dt: pd.Timestamp) -> float | None:
    if len(series) == 0:
        return None
    dt = to_naive_ts(dt)
    s = series.loc[:dt]
    if len(s) == 0:
        return None
    v = float(s.iloc[-1])
    return v if np.isfinite(v) else None


def compute_net_payout_proxy(
    ticker: str,
    rebalance_dt: pd.Timestamp,
    close: pd.DataFrame,
    dividends: pd.DataFrame,
    shares_cache: dict[str, pd.Series],
    shares_cache_dir: Path,
    use_buyback_proxy: bool,
) -> float | None:
    px = close.at[rebalance_dt, ticker] if ticker in close.columns and rebalance_dt in close.index else np.nan
    if not np.isfinite(px) or px <= 0:
        return None

    div_window_start = rebalance_dt - pd.Timedelta(days=365)
    div_sum = dividends.loc[(dividends.index > div_window_start) & (dividends.index <= rebalance_dt), ticker].sum()
    div_yield = float(div_sum / px)

    buyback_yield = 0.0
    if use_buyback_proxy:
        shares = load_or_fetch_shares(
            ticker=ticker,
            start=rebalance_dt - pd.Timedelta(days=800),
            end=rebalance_dt + pd.Timedelta(days=5),
            cache_dir=shares_cache_dir,
            memory_cache=shares_cache,
        )
        curr = asof_value(shares, rebalance_dt)
        prev = asof_value(shares, rebalance_dt - pd.Timedelta(days=365))
        if curr is not None and prev is not None and prev > 0:
            # Positive if shares outstanding decreased (buybacks), negative if increased (issuance)
            buyback_yield = (prev - curr) / prev

    score = div_yield + buyback_yield
    return float(score) if np.isfinite(score) else None


def compute_stats(returns: pd.Series, benchmark_returns: pd.Series | None = None) -> dict[str, float]:
    from performance import compute_perf_stats
    return compute_perf_stats(returns, freq=252, benchmark_returns=benchmark_returns)


def run_backtest(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    dividends: pd.DataFrame,
    liquid_count: int,
    quantile: int,
    rebalance_month: int,
    cost_bps: float,
    use_buyback_proxy: bool,
    shares_cache_dir: Path,
) -> tuple[pd.Series, pd.DataFrame]:
    daily_rets = close.pct_change().fillna(0.0)
    dollar_vol = (close * volume).replace([np.inf, -np.inf], np.nan)

    month_ends = close.resample("ME").last().dropna(how="all")
    rebalance_month_ends = [d for d in month_ends.index if d.month == rebalance_month]

    trading_days = close.index
    rebalance_days = []
    for d in rebalance_month_ends:
        i = trading_days.searchsorted(d, side="right") - 1
        if i >= 0:
            rebalance_days.append(trading_days[i])

    rebalance_days = sorted(list(dict.fromkeys(rebalance_days)))
    if len(rebalance_days) < 2:
        raise RuntimeError("Not enough rebalance dates to run backtest.")

    portfolio_ret = pd.Series(0.0, index=trading_days)
    prev_w = pd.Series(0.0, index=close.columns)
    shares_cache = {}
    selections = []

    for i, reb_dt in enumerate(rebalance_days[:-1]):
        next_reb_dt = rebalance_days[i + 1]

        # Require at least 1Y history for signals
        lookback_start = reb_dt - pd.Timedelta(days=252)
        liquid_window = dollar_vol.loc[(dollar_vol.index > lookback_start) & (dollar_vol.index <= reb_dt)]
        if len(liquid_window) < 120:
            continue

        liquid_rank = liquid_window.mean().dropna().sort_values(ascending=False)
        liquid_universe = liquid_rank.index[: liquid_count]

        scores = {}
        for ticker in liquid_universe:
            score = compute_net_payout_proxy(
                ticker=ticker,
                rebalance_dt=reb_dt,
                close=close,
                dividends=dividends,
                shares_cache=shares_cache,
                shares_cache_dir=shares_cache_dir,
                use_buyback_proxy=use_buyback_proxy,
            )
            if score is not None and np.isfinite(score):
                scores[ticker] = score

        if len(scores) == 0:
            new_w = pd.Series(0.0, index=close.columns)
        else:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            n_long = max(1, int(len(ranked) / quantile))
            longs = [t for t, _ in ranked[:n_long]]
            new_w = pd.Series(0.0, index=close.columns)
            new_w.loc[longs] = 1.0 / len(longs)
            selections.append({"date": reb_dt, "n_liquid": len(liquid_universe), "n_scored": len(scores), "n_longs": len(longs)})

        # Weights become active next trading day
        start_pos = trading_days.searchsorted(reb_dt, side="right")
        end_pos = trading_days.searchsorted(next_reb_dt, side="right") - 1
        if start_pos > end_pos or start_pos >= len(trading_days):
            prev_w = new_w
            continue

        period_days = trading_days[start_pos : end_pos + 1]
        period_rets = daily_rets.loc[period_days, close.columns]
        period_port = (period_rets * new_w).sum(axis=1)

        # Apply turnover cost on first day of the new period
        turnover = float((new_w - prev_w).abs().sum())
        cost = turnover * (cost_bps / 10000.0)
        first_day = period_days[0]
        period_port.loc[first_day] -= cost

        portfolio_ret.loc[period_days] = period_port
        prev_w = new_w

    trade_returns = portfolio_ret[portfolio_ret.index >= rebalance_days[0]].copy()
    trade_returns = trade_returns.loc[trade_returns.index <= rebalance_days[-1]]

    if len(selections) == 0:
        raise RuntimeError("No valid annual selections were formed. Try reducing liquid_count or check data quality.")

    selection_df = pd.DataFrame(selections)
    return trade_returns, selection_df


def main():
    parser = argparse.ArgumentParser(description="Terminal backtest: Net Payout Yield Effect (proxy implementation).")
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--tickers-file", type=str, default=None, help="Optional text file with one ticker per line.")
    parser.add_argument("--liquid-count", type=int, default=500)
    parser.add_argument("--quantile", type=int, default=10)
    parser.add_argument("--rebalance-month", type=int, default=6)
    parser.add_argument("--cost-bps", type=float, default=0.5, help="Turnover cost in bps (0.5 = 0.005%%).")
    parser.add_argument("--no-buyback-proxy", action="store_true", help="Disable shares-outstanding buyback/issuance proxy (faster).")
    parser.add_argument("--shares-cache-dir", type=str, default=".cache/net_payout_shares")
    parser.add_argument("--save-equity", type=str, default=None, help="Optional CSV output path for equity curve.")
    parser.add_argument("--plot-file", type=str, default="nyp_equity.png", help="Output path for equity curve PNG.")
    parser.add_argument("--show-plot", action="store_true", help="Display the plot window in addition to saving PNG.")
    args = parser.parse_args()

    if args.quantile <= 1:
        raise ValueError("--quantile must be >= 2")

    tickers = load_tickers(args.tickers_file)
    print(f"Universe tickers loaded: {len(tickers)}")

    # Add one extra year buffer for signal warm-up
    buffer_start = (pd.Timestamp(args.start) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    close, volume, dividends = download_prices_and_actions(tickers=tickers, start=buffer_start, end=args.end)
    print(f"Usable tickers after cleaning: {close.shape[1]}")

    returns, selections = run_backtest(
        close=close,
        volume=volume,
        dividends=dividends,
        liquid_count=args.liquid_count,
        quantile=args.quantile,
        rebalance_month=args.rebalance_month,
        cost_bps=args.cost_bps,
        use_buyback_proxy=not args.no_buyback_proxy,
        shares_cache_dir=Path(args.shares_cache_dir),
    )

    # ---- Benchmark: SPY total-return (daily, matched to backtest period) ----
    spy_rets: pd.Series | None = None
    try:
        spy_raw = yf.download(
            "SPY", start=returns.index[0].strftime("%Y-%m-%d"),
            end=returns.index[-1].strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False,
        )
        if spy_raw is not None and len(spy_raw) > 0:
            spy_px = spy_raw["Close"].dropna()
            spy_rets = spy_px.pct_change().dropna()
    except Exception:
        pass

    stats = compute_stats(returns, benchmark_returns=spy_rets)

    all_report: dict[str, dict] = {"Net Payout Yield": stats}
    if spy_rets is not None:
        from performance import compute_perf_stats
        all_report["SPY (B&H)"] = compute_perf_stats(
            spy_rets.reindex(returns.index).dropna(), freq=252
        )

    from performance import print_stats_table, plot_tearsheet

    print_stats_table(
        all_report,
        title=f"Net Payout Yield Strategy  "
              f"({returns.index[0].date()} → {returns.index[-1].date()})",
    )
    print(f"  Rebalances with valid selections : {len(selections)}")

    print(
        "\nNote on survivorship bias: this backtest uses the current ticker "
        "universe and excludes companies that were delisted, acquired, or went "
        "bankrupt during the sample period. Results represent an upper-bound estimate."
    )

    equity = (1 + returns).cumprod()

    if args.save_equity:
        out = pd.DataFrame({"equity": equity, "returns": returns})
        out.to_csv(args.save_equity)
        print(f"Saved equity curve to: {args.save_equity}")

    plot_tearsheet(
        returns=returns,
        benchmark_returns=spy_rets,
        title="Net Payout Yield Strategy",
        freq=252,
        rolling_window=252,
        save_path=args.plot_file,
        show=args.show_plot,
    )


if __name__ == "__main__":
    main()
