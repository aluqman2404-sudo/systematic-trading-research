"""
Macro System Backtest (Monthly): Trend + Carry + Volatility (VRP proxy)

- Trend sleeve: long/short cross-sectional momentum (12m), long top half & short bottom half
- Carry sleeve: DBV vs SHY (risk-on filter), falls back to SHY if DBV missing
- Vol sleeve: SVXY vs SHY (risk-on filter), falls back to SHY if SVXY missing
- Sleeve risk balancing: allocates sleeve weights inversely to trailing vol
- Portfolio volatility targeting: scales portfolio returns to target vol (e.g., 10%)

Data: Yahoo Finance via yfinance (Adj Close)
Execution: Simulated monthly rebalancing, 1-month lag (avoid lookahead)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
START = "2012-01-01"
END = None  # None = today
INITIAL_CAPITAL = 100_000

# Preferred sleeve "risk budgets" (used before risk balancing)
BUDGET_TREND = 0.40
BUDGET_CARRY = 0.30
BUDGET_VOL = 0.30

# Trend universe (diversified ETFs)
TREND_TICKERS = ["SPY", "IEF", "GLD", "DBC", "EFA"]

# Carry + Vol proxies
CARRY_TICKER = "DBV"
VOL_TICKER = "SVXY"
HEDGE_TICKER = "VIXY"  # used in risk-on filter
DEFENSIVE = "SHY"      # cash-like

# Signals / windows
MOM_LOOKBACK_MONTHS = 12
ROLL_VOL_MONTHS = 12

# Costs
TURNOVER_COST = 0.001  # 10 bps per 1.0 turnover (very rough)

# Portfolio vol targeting
TARGET_PORTFOLIO_VOL = 0.10   # 10% annualized
MAX_VOL_TARGET_LEVERAGE = 2.0 # cap scaling to avoid runaway leverage

# Optional cap on vol sleeve weight (reduces tail risk)
MAX_VOL_SLEEVE_WEIGHT = 0.25  # set None to disable


# ----------------------------
# HELPERS
# ----------------------------
def download_adj_close(tickers, start=START, end=END, max_retries=3):
    """
    Robust downloader:
    - retries on transient Yahoo failures
    - drops tickers that fail
    - returns Adj Close DataFrame with no all-NaN columns
    """
    tickers = list(dict.fromkeys(tickers))  # unique, preserve order
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="column",
                progress=False,
                threads=False,  # fewer Yahoo flake-outs on some machines
            )

            if df is None or len(df) == 0:
                raise RuntimeError("yfinance returned None/empty dataframe")

            if "Adj Close" not in df.columns:
                raise RuntimeError(f"Unexpected columns from yfinance: {df.columns}")

            px = df["Adj Close"]
            if isinstance(px, pd.Series):
                px = px.to_frame()

            # drop all-missing tickers
            px = px.dropna(axis=1, how="all")

            # forward-fill small gaps
            px = px.ffill().dropna(how="all")

            if px.shape[1] == 0:
                raise RuntimeError("All tickers failed download (no usable columns).")

            failed = sorted(set(tickers) - set(px.columns.tolist()))
            if failed:
                print(f"⚠️ Dropped failed tickers: {failed}")

            return px

        except Exception as e:
            last_err = e
            print(f"Download attempt {attempt}/{max_retries} failed: {e}")

    raise RuntimeError(f"Failed to download after {max_retries} retries. Last error: {last_err}")


def to_monthly_last(px_daily: pd.DataFrame) -> pd.DataFrame:
    # pandas uses "ME" for month-end in newer versions
    return px_daily.resample("ME").last()


def monthly_returns(px_m: pd.DataFrame) -> pd.DataFrame:
    return px_m.pct_change().dropna()


def momentum_12m(px_m: pd.DataFrame, lookback_months=MOM_LOOKBACK_MONTHS) -> pd.DataFrame:
    return px_m.pct_change(lookback_months)


def annualized_vol(x: pd.Series, months=ROLL_VOL_MONTHS) -> pd.Series:
    return x.rolling(months).std() * np.sqrt(12)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats(equity: pd.Series, rets_m: pd.Series):
    n_months = len(rets_m)
    if n_months < 12:
        raise ValueError("Not enough data to compute annualized stats (need >= 12 months).")

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (12 / n_months) - 1
    vol = rets_m.std() * np.sqrt(12)
    sharpe = (rets_m.mean() * 12) / (rets_m.std() * np.sqrt(12) + 1e-12)
    mdd = max_drawdown(equity)
    return {"CAGR": cagr, "Ann.Vol": vol, "Sharpe": sharpe, "Max Drawdown": mdd}


# ----------------------------
# BACKTEST
# ----------------------------
def backtest_three_engines():
    # What we request from Yahoo
    requested = sorted(set(TREND_TICKERS + [CARRY_TICKER, VOL_TICKER, HEDGE_TICKER, DEFENSIVE]))
    px_d = download_adj_close(requested, start=START, end=END)

    px_m = to_monthly_last(px_d)
    rets_all = monthly_returns(px_m)

    if len(rets_all) < 24:
        raise ValueError(f"Not enough monthly data after cleaning: only {len(rets_all)} months.")

    mom = momentum_12m(px_m, MOM_LOOKBACK_MONTHS).dropna()
    # Align to momentum dates
    common_idx = rets_all.index.intersection(mom.index)
    rets = rets_all.loc[common_idx]
    mom = mom.loc[common_idx]

    # What we actually have after downloads
    have_carry = CARRY_TICKER in rets.columns
    have_vol = VOL_TICKER in rets.columns
    have_hedge = HEDGE_TICKER in mom.columns  # momentum needs hedge column

    # Risk-on filter
    if have_hedge:
        risk_on = (mom["SPY"] > 0) & (mom[HEDGE_TICKER] < 0)
    else:
        print("⚠️ HEDGE ticker missing; risk_on uses only SPY momentum.")
        risk_on = (mom["SPY"] > 0)

    # ----------------------------
    # Sleeve 1: TREND (long/short cross-sectional momentum)
    # ----------------------------
    trend_tickers_present = [t for t in TREND_TICKERS if t in mom.columns and t in rets.columns]
    if len(trend_tickers_present) < 4:
        raise RuntimeError(
            f"Not enough TREND tickers downloaded to run (need >=4). Got: {trend_tickers_present}"
        )

    trend_w = pd.DataFrame(0.0, index=rets.index, columns=trend_tickers_present)

    for dt in rets.index:
        m = mom.loc[dt, trend_tickers_present].dropna()
        if len(m) < 4:
            continue

        ranks = m.sort_values(ascending=False)
        n = len(ranks) // 2  # top half long, bottom half short
        if n <= 0:
            continue

        longs = ranks.index[:n]
        shorts = ranks.index[-n:]

        # sum(abs(weights)) = 1 within sleeve
        trend_w.loc[dt, longs] = 1.0 / (2 * n)
        trend_w.loc[dt, shorts] = -1.0 / (2 * n)

    # If empty for a month, treat as DEFENSIVE
    trend_empty = trend_w.abs().sum(axis=1) == 0

    # ----------------------------
    # Sleeve 2: CARRY (DBV vs SHY) with fallback
    # ----------------------------
    if have_carry:
        carry_w = pd.DataFrame(0.0, index=rets.index, columns=[CARRY_TICKER, DEFENSIVE])
        carry_w.loc[risk_on, CARRY_TICKER] = 1.0
        carry_w.loc[~risk_on, DEFENSIVE] = 1.0
    else:
        print("⚠️ CARRY ticker missing; carry sleeve allocated to DEFENSIVE (SHY).")
        carry_w = pd.DataFrame(0.0, index=rets.index, columns=[DEFENSIVE])
        carry_w[DEFENSIVE] = 1.0

    # ----------------------------
    # Sleeve 3: VOL (SVXY vs SHY) with fallback
    # ----------------------------
    if have_vol:
        vol_w = pd.DataFrame(0.0, index=rets.index, columns=[VOL_TICKER, DEFENSIVE])
        vol_w.loc[risk_on, VOL_TICKER] = 1.0
        vol_w.loc[~risk_on, DEFENSIVE] = 1.0
    else:
        print("⚠️ VOL ticker missing; vol sleeve allocated to DEFENSIVE (SHY).")
        vol_w = pd.DataFrame(0.0, index=rets.index, columns=[DEFENSIVE])
        vol_w[DEFENSIVE] = 1.0

    # ----------------------------
    # Sleeve returns (for risk balancing)
    # ----------------------------
    trend_ret = (trend_w * rets[trend_tickers_present]).sum(axis=1)
    trend_ret = trend_ret.where(~trend_empty, rets[DEFENSIVE])

    if have_carry:
        carry_ret = (carry_w * rets[[CARRY_TICKER, DEFENSIVE]]).sum(axis=1)
    else:
        carry_ret = rets[DEFENSIVE].copy()

    if have_vol:
        vol_ret = (vol_w * rets[[VOL_TICKER, DEFENSIVE]]).sum(axis=1)
    else:
        vol_ret = rets[DEFENSIVE].copy()

    # Rolling vols for risk balancing
    trend_vol = annualized_vol(trend_ret, ROLL_VOL_MONTHS)
    carry_vol = annualized_vol(carry_ret, ROLL_VOL_MONTHS)
    vol_vol = annualized_vol(vol_ret, ROLL_VOL_MONTHS)

    sleeve_w = pd.DataFrame(index=rets.index, data={
        "trend": BUDGET_TREND / (trend_vol + 1e-12),
        "carry": BUDGET_CARRY / (carry_vol + 1e-12),
        "vol":   BUDGET_VOL   / (vol_vol + 1e-12),
    }).fillna(0.0)

    if MAX_VOL_SLEEVE_WEIGHT is not None:
        sleeve_w["vol"] = sleeve_w["vol"].clip(upper=MAX_VOL_SLEEVE_WEIGHT)

    sleeve_w = sleeve_w.div(sleeve_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # ----------------------------
    # Build full asset weights
    # ----------------------------
    cols = list(trend_tickers_present) + [DEFENSIVE]
    if have_carry:
        cols += [CARRY_TICKER]
    if have_vol:
        cols += [VOL_TICKER]

    all_cols = sorted(set(cols))
    W = pd.DataFrame(0.0, index=rets.index, columns=all_cols)

    # Trend allocation
    for t in trend_tickers_present:
        W[t] += trend_w[t] * sleeve_w["trend"]
    W.loc[trend_empty, DEFENSIVE] += sleeve_w.loc[trend_empty, "trend"]

    # Carry allocation
    if have_carry:
        W[[CARRY_TICKER, DEFENSIVE]] += carry_w.mul(sleeve_w["carry"], axis=0)
    else:
        W[DEFENSIVE] += sleeve_w["carry"]

    # Vol allocation
    if have_vol:
        W[[VOL_TICKER, DEFENSIVE]] += vol_w.mul(sleeve_w["vol"], axis=0)
    else:
        W[DEFENSIVE] += sleeve_w["vol"]

    # Normalize weights to sum to 1
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # ----------------------------
    # Portfolio returns with 1-month lag (avoid lookahead)
    # ----------------------------
    W_lag = W.shift(1).dropna()
    rets2 = rets.loc[W_lag.index, W_lag.columns]

    port_ret_gross = (W_lag * rets2).sum(axis=1)

    # Turnover cost
    turnover = W.diff().abs().sum(axis=1).loc[port_ret_gross.index]
    port_ret_net = port_ret_gross - TURNOVER_COST * turnover

    # ----------------------------
    # Portfolio volatility targeting
    # ----------------------------
    port_vol = annualized_vol(port_ret_net, ROLL_VOL_MONTHS)
    scaling = (TARGET_PORTFOLIO_VOL / (port_vol + 1e-12)).clip(upper=MAX_VOL_TARGET_LEVERAGE)
    port_ret_vt = port_ret_net * scaling.shift(1).fillna(0.0)

    equity = (1 + port_ret_vt).cumprod() * INITIAL_CAPITAL
    stats = perf_stats(equity, port_ret_vt)

    return {
        "prices_monthly": px_m,
        "returns_monthly": rets,
        "momentum": mom,
        "weights": W,
        "sleeve_weights": sleeve_w,
        "portfolio_returns_net": port_ret_net,
        "portfolio_returns_vol_targeted": port_ret_vt,
        "equity": equity,
        "stats": stats,
        "diagnostics": {
            "have_carry": have_carry,
            "have_vol": have_vol,
            "have_hedge": have_hedge,
            "trend_tickers_present": trend_tickers_present,
        },
    }


if __name__ == "__main__":
    res = backtest_three_engines()

    print("\n=== Diagnostics ===")
    for k, v in res["diagnostics"].items():
        print(f"{k}: {v}")

    print("\n=== Performance (Vol-Targeted, Net of turnover cost) ===")
    print(f"CAGR: {res['stats']['CAGR']:.2%}")
    print(f"Ann.Vol: {res['stats']['Ann.Vol']:.2%}")
    print(f"Sharpe (rf~0): {res['stats']['Sharpe']:.2f}")
    print(f"Max Drawdown: {res['stats']['Max Drawdown']:.2%}")

    res["equity"].plot(
        title="Trend (LS) + Carry + Vol (Proxy) — Vol Targeted Equity Curve",
        figsize=(10, 4)
    )
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()