"""
Trend + Carry + Volatility (VRP proxy) monthly backtest using ETF proxies.

Trend sleeve: time-series momentum (12m) on diversified ETFs, long/flat.
Carry sleeve: DBV (G10 carry ETF) risk-managed with a risk-on filter.
Vol sleeve: SVXY (short vol proxy) with risk-on filter; otherwise in SHY.

Requires:
  pip install yfinance pandas numpy matplotlib

Notes:
- Uses Adjusted Close (Adj Close).
- Rebalances monthly (end of month).
- Adds simple transaction cost per 1.0 turnover (e.g., 10 bps per full rebalance).
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

# Sleeve allocations (must sum to 1.0)
W_TREND = 0.40
W_CARRY = 0.30
W_VOL = 0.30

# Trend universe (diversified)
TREND_TICKERS = ["SPY", "TLT", "GLD", "DBC", "EFA"]  # equities, bonds, gold, broad commodities, developed ex-US

# Carry proxy + defensive asset
CARRY_TICKER = "DBV"  # currency carry ETF proxy
DEFENSIVE = "SHY"     # short-term treasuries proxy / "cash-like"

# Volatility / VRP proxy
VOL_TICKER = "SVXY"   # short VIX futures proxy (dangerous in spikes)
HEDGE_TICKER = "VIXY" # long VIX futures proxy (used only for regime filter here)

# Signals
MOM_LOOKBACK_MONTHS = 12
MIN_HISTORY_MONTHS = 13  # must exceed lookback
TURNOVER_COST = 0.001    # 10 bps cost per 1.0 turnover (very rough)


# ----------------------------
# HELPERS
# ----------------------------
def download_adj_close(tickers, start=START, end=END):
    px = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all")
    px = px.ffill().dropna()
    return px

def to_monthly_last(px_daily: pd.DataFrame) -> pd.DataFrame:
    return px_daily.resample("ME").last()   # Month-End

def monthly_returns(px_m: pd.DataFrame) -> pd.DataFrame:
    return px_m.pct_change().dropna()

def momentum_12m(px_m: pd.DataFrame, lookback_months=MOM_LOOKBACK_MONTHS) -> pd.DataFrame:
    return px_m.pct_change(lookback_months)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def perf_stats(equity: pd.Series, rets_m: pd.Series):
    # monthly to annualized
    n_months = len(rets_m)
    if n_months < 12:
        raise ValueError("Not enough data to compute annualized stats.")

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (12 / n_months) - 1
    vol = rets_m.std() * np.sqrt(12)
    sharpe = (rets_m.mean() * 12) / (rets_m.std() * np.sqrt(12) + 1e-12)
    mdd = max_drawdown(equity)
    return {
        "CAGR": cagr,
        "Ann.Vol": vol,
        "Sharpe (rf~0)": sharpe,
        "Max Drawdown": mdd
    }


# ----------------------------
# BACKTEST LOGIC
# ----------------------------
def backtest_three_engines():
    tickers = sorted(set(TREND_TICKERS + [CARRY_TICKER, DEFENSIVE, VOL_TICKER, HEDGE_TICKER]))
    px_d = download_adj_close(tickers)
    px_m = to_monthly_last(px_d)

    # compute signals
    mom = momentum_12m(px_m, MOM_LOOKBACK_MONTHS)
    rets = monthly_returns(px_m)

    # warmup cutoff
    mom = mom.dropna()
    rets = rets.loc[mom.index]  # align

    # --- Risk-on / risk-off regime filter (simple, robust-ish) ---
    # Risk-on if SPY 12m momentum > 0 AND VIXY 12m momentum < 0 (vol not trending up)
    risk_on = (mom["SPY"] > 0) & (mom[HEDGE_TICKER] < 0)

    # --- Trend sleeve weights (long/flat, equal-weight among positive-momentum assets) ---
    trend_w = pd.DataFrame(0.0, index=rets.index, columns=TREND_TICKERS)

    for dt in rets.index:
        m = mom.loc[dt, TREND_TICKERS]
        winners = m[m > 0].index.tolist()
        if len(winners) > 0:
            trend_w.loc[dt, winners] = 1.0 / len(winners)
        # else stay in cash-like defensive via separate sleeve (handled below)

    # --- Carry sleeve ---
    # If risk-on: hold DBV, else hold SHY
    carry_w = pd.DataFrame(0.0, index=rets.index, columns=[CARRY_TICKER, DEFENSIVE])
    carry_w.loc[risk_on, CARRY_TICKER] = 1.0
    carry_w.loc[~risk_on, DEFENSIVE] = 1.0

    # --- Vol sleeve (VRP proxy) ---
    # If risk-on: hold SVXY, else defensive (SHY)
    vol_w = pd.DataFrame(0.0, index=rets.index, columns=[VOL_TICKER, DEFENSIVE])
    vol_w.loc[risk_on, VOL_TICKER] = 1.0
    vol_w.loc[~risk_on, DEFENSIVE] = 1.0

    # Combine sleeves into full portfolio weights
    # Note: trend sleeve has only TREND_TICKERS; cash fallback = DEFENSIVE when trend has no winners
    all_cols = sorted(set(TREND_TICKERS + [CARRY_TICKER, VOL_TICKER, DEFENSIVE]))
    W = pd.DataFrame(0.0, index=rets.index, columns=all_cols)

    # allocate trend sleeve
    W[TREND_TICKERS] += W_TREND * trend_w

    # if trend sleeve has no winners, allocate that trend portion to DEFENSIVE
    trend_sum = trend_w.sum(axis=1)
    W.loc[trend_sum == 0, DEFENSIVE] += W_TREND

    # allocate carry sleeve
    W[[CARRY_TICKER, DEFENSIVE]] += W_CARRY * carry_w

    # allocate vol sleeve
    W[[VOL_TICKER, DEFENSIVE]] += W_VOL * vol_w

    # normalize (should already be 1.0, but due to overlaps on DEFENSIVE it still sums to 1.0)
    W = W.div(W.sum(axis=1), axis=0)

    # Compute portfolio returns with 1-month lag (trade at month-end close, earn next month return)
    W_lag = W.shift(1).dropna()
    rets2 = rets.loc[W_lag.index, W_lag.columns].copy()

    port_ret_gross = (W_lag * rets2).sum(axis=1)

    # Transaction costs based on turnover in weights
    turnover = W.diff().abs().sum(axis=1).loc[port_ret_gross.index]  # L1 turnover
    port_ret_net = port_ret_gross - TURNOVER_COST * turnover

    equity = (1 + port_ret_net).cumprod() * INITIAL_CAPITAL

    stats = perf_stats(equity, port_ret_net)

    return {
        "prices_monthly": px_m,
        "returns_monthly": rets,
        "weights": W,
        "portfolio_returns_gross": port_ret_gross,
        "portfolio_returns_net": port_ret_net,
        "equity": equity,
        "stats": stats
    }


if __name__ == "__main__":
    res = backtest_three_engines()

    print("=== Performance (Net of simple turnover cost) ===")
    for k, v in res["stats"].items():
        if "Drawdown" in k:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.2%}")

    # Plot equity curve
    res["equity"].plot(title="Trend + Carry + Vol (VRP proxy) Equity Curve", figsize=(10, 4))
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()