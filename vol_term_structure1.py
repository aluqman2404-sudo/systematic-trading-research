"""
vol_term_structure.py — VIX Term Structure / Volatility Risk Premium Backtest
==============================================================================

Strategy Overview
-----------------
VIX futures are structurally in contango (futures price > spot VIX) because
market participants pay a persistent insurance premium for downside protection.
Short-volatility strategies harvest this roll yield over time, accepting
exposure to sharp left-tail drawdowns during volatility spikes.

This module implements two complementary sub-strategies:

  Sub-strategy 1 — Term Structure Slope Signal
      Uses the ratio (VIX3M - VIX) / VIX to identify contango regimes.
      Long SVXY when slope > threshold; flat in backwardation or high-VIX.

  Sub-strategy 2 — VIX Momentum Filter
      Multi-condition regime filter requiring: declining VIX SMA, VIX below
      an elevated-vol threshold, and a positive SPY price trend.

Academic References
-------------------
- Carr, P. & Wu, L. (2009). "Variance Risk Premiums." Review of Financial
  Studies, 22(3), 1311–1341.
- Whaley, R. E. (2009). "Understanding the VIX." Journal of Portfolio
  Management, 35(3), 98–105.
- Simon, D. P. & Campasano, J. (2014). "The VIX Futures Basis: Evidence and
  Trading Strategies." Journal of Derivatives, 21(3), 54–74.

Data Sources (Yahoo Finance, free)
-----------------------------------
  ^VIX    — CBOE VIX 30-day implied volatility index (spot)
  ^VIX3M  — CBOE 3-month VIX (primary long-dated IV proxy)
  ^VIX9D  — CBOE 9-day VIX (fallback if VIX3M history is too short)
  SVXY    — ProShares Short VIX Short-Term Futures ETF (-0.5x VIX futures)
  VXX     — iPath Series B S&P 500 VIX Short-Term Futures ETN (long vol)
  SPY     — SPDR S&P 500 ETF (benchmark)

Usage
-----
  python vol_term_structure.py
  python vol_term_structure.py --start 2014-01-01 --end 2024-12-31
  python vol_term_structure.py --contango-threshold 0.07 --vix-crisis-level 28
  python vol_term_structure.py --no-plot
  python vol_term_structure.py --plot-file my_tearsheet.png --cost-bps 15
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

from performance1 import compute_perf_stats, print_stats_table, plot_tearsheet

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_START: str = "2012-01-01"
DEFAULT_END: str = pd.Timestamp.today().strftime("%Y-%m-%d")
DEFAULT_CONTANGO_THRESHOLD: float = 0.05   # 5% contango / backwardation band
DEFAULT_VIX_CRISIS: float = 30.0           # flat above this VIX level
DEFAULT_COST_BPS: float = 10.0             # round-trip transaction cost basis points
DEFAULT_PLOT_FILE: str = "vol_term_structure_tearsheet.png"
FREQ: int = 252                            # trading days per year


# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------

def download_data(
    start: str,
    end: str,
    primary_long_vix: str = "^VIX3M",
    fallback_long_vix: str = "^VIX9D",
) -> pd.DataFrame:
    """
    Download all required price/index series from Yahoo Finance.

    Parameters
    ----------
    start : str
        ISO date string for the start of the backtest window.
    end : str
        ISO date string for the end of the backtest window.
    primary_long_vix : str
        Ticker for the longer-dated VIX index (default ``^VIX3M``).
    fallback_long_vix : str
        Fallback ticker if primary has insufficient history.

    Returns
    -------
    pd.DataFrame
        Daily aligned DataFrame with columns:
        VIX, LONG_VIX, SVXY, VXX, SPY.
        Index is DatetimeIndex (trading days).
    """
    tickers = ["^VIX", primary_long_vix, "SVXY", "VXX", "SPY"]
    print(f"[Data] Downloading {len(tickers)} tickers from {start} to {end} ...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Extract adjusted close; yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()

    close.index = pd.to_datetime(close.index)
    close.sort_index(inplace=True)

    # Rename to canonical names
    rename_map: dict[str, str] = {
        "^VIX": "VIX",
        "SVXY": "SVXY",
        "VXX": "VXX",
        "SPY": "SPY",
    }
    close.rename(columns=rename_map, inplace=True)

    # Handle long-dated VIX ticker
    long_vix_col = primary_long_vix.replace("^", "")
    fallback_col = fallback_long_vix.replace("^", "")

    if primary_long_vix in close.columns:
        close.rename(columns={primary_long_vix: "LONG_VIX"}, inplace=True)
    elif long_vix_col in close.columns:
        close.rename(columns={long_vix_col: "LONG_VIX"}, inplace=True)
    else:
        long_vix_col = None

    if "LONG_VIX" not in close.columns or close["LONG_VIX"].dropna().shape[0] < 252:
        print(
            f"[Data] Warning: {primary_long_vix} has insufficient history. "
            f"Attempting fallback {fallback_long_vix} ..."
        )
        fb_raw = yf.download(
            [fallback_long_vix],
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(fb_raw.columns, pd.MultiIndex):
            fb_close = fb_raw["Close"].copy()
        else:
            fb_close = fb_raw[["Close"]].copy()

        fb_close.index = pd.to_datetime(fb_close.index)
        col_key = fallback_long_vix if fallback_long_vix in fb_close.columns else fallback_col
        if col_key in fb_close.columns:
            close["LONG_VIX"] = fb_close[col_key].reindex(close.index)
            print(f"[Data] Using {fallback_long_vix} as LONG_VIX proxy.")
        else:
            print("[Data] Warning: Fallback also unavailable. LONG_VIX set to NaN.")
            close["LONG_VIX"] = np.nan

    required_cols = ["VIX", "LONG_VIX", "SVXY", "VXX", "SPY"]
    for col in required_cols:
        if col not in close.columns:
            close[col] = np.nan

    df = close[required_cols].copy()

    # Drop rows where the primary tradeable instruments are all NaN
    df.dropna(subset=["VIX", "SVXY", "SPY"], how="any", inplace=True)

    n_total = len(df)
    n_long_vix = df["LONG_VIX"].notna().sum()
    print(
        f"[Data] Final dataset: {n_total} trading days "
        f"({df.index[0].date()} to {df.index[-1].date()}). "
        f"LONG_VIX coverage: {n_long_vix}/{n_total} days "
        f"({n_long_vix / n_total:.0%})."
    )

    return df


# ---------------------------------------------------------------------------
# Term Structure Computation
# ---------------------------------------------------------------------------

def compute_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the VIX term structure slope and derived signals.

    Adds the following columns to a copy of ``df``:

    - ``slope``         : (LONG_VIX - VIX) / VIX  — contango ratio
    - ``svxy_ret``      : daily log-return of SVXY (as simple return)
    - ``spy_ret``       : daily log-return of SPY
    - ``vxx_ret``       : daily log-return of VXX
    - ``vix_sma10``     : 10-day SMA of VIX
    - ``vix_sma60``     : 60-day SMA of VIX
    - ``spy_ret_20d``   : SPY 20-day cumulative simple return

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`download_data`.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame.
    """
    out = df.copy()

    # Term structure slope: positive = contango, negative = backwardation
    out["slope"] = (out["LONG_VIX"] - out["VIX"]) / out["VIX"]

    # Daily simple returns for tradeable instruments
    out["svxy_ret"] = out["SVXY"].pct_change()
    out["spy_ret"] = out["SPY"].pct_change()
    out["vxx_ret"] = out["VXX"].pct_change()

    # VIX moving averages (for momentum sub-strategy)
    out["vix_sma10"] = out["VIX"].rolling(10, min_periods=10).mean()
    out["vix_sma60"] = out["VIX"].rolling(60, min_periods=60).mean()

    # SPY 20-day return (for momentum sub-strategy)
    out["spy_ret_20d"] = out["SPY"].pct_change(20)

    return out


# ---------------------------------------------------------------------------
# Sub-strategy 1: Term Structure Slope Signal
# ---------------------------------------------------------------------------

def run_term_structure_strategy(
    df: pd.DataFrame,
    contango_threshold: float = DEFAULT_CONTANGO_THRESHOLD,
    vix_crisis_level: float = DEFAULT_VIX_CRISIS,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Term Structure Slope Strategy.

    Entry/exit rules (all signals lagged 1 day before execution):
    - Long SVXY if slope > +contango_threshold AND VIX <= vix_crisis_level
    - Flat (cash, 0% return) in all other conditions

    Transaction costs applied on position changes (10 bps one-way =>
    cost_bps per one-way leg; a round-trip = 2 * cost_bps).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_term_structure`.
    contango_threshold : float
        Minimum slope to go long (default 0.05 = 5%).
    vix_crisis_level : float
        VIX level above which all positions are flattened (default 30).
    cost_bps : float
        One-way transaction cost in basis points (default 10).

    Returns
    -------
    (strategy_returns, position_series, diagnostics_dict)
    """
    cost = cost_bps / 10_000.0

    # Raw signal: 1 = long SVXY, 0 = flat
    slope = df["slope"]
    vix = df["VIX"]

    long_signal = (slope > contango_threshold) & (vix <= vix_crisis_level)

    raw_position = long_signal.astype(float)  # 1 = long SVXY, 0 = flat

    # 1-day lag: today's signal determines tomorrow's position
    position = raw_position.shift(1).fillna(0.0)

    # Compute trade-day indicator (position changes)
    trades = (position.diff().abs() > 0).astype(float)
    trades.iloc[0] = 0.0

    # Strategy returns: position * asset_return - cost on trade days
    strategy_ret = position * df["svxy_ret"] - trades * cost

    # Replace NaN returns (e.g., first row) with 0
    strategy_ret = strategy_ret.fillna(0.0)

    # --- Diagnostics ---
    in_trade = position > 0
    pct_days_in_trade = float(in_trade.mean())
    n_trades = int(trades.sum())

    # Average VIX when entering vs. when flat
    avg_vix_in_trade = float(vix[in_trade].mean()) if in_trade.any() else np.nan
    avg_vix_flat = float(vix[~in_trade].mean()) if (~in_trade).any() else np.nan

    # Crises avoided: days where VIX > crisis_level but we were flat
    crisis_days = vix > vix_crisis_level
    crises_avoided = int((crisis_days & ~in_trade).sum())

    # Mark when slope data is unavailable (LONG_VIX missing)
    no_slope_data = df["slope"].isna().sum()

    diagnostics: dict = {
        "pct_days_in_trade": pct_days_in_trade,
        "n_trades": n_trades,
        "avg_vix_in_trade": avg_vix_in_trade,
        "avg_vix_flat": avg_vix_flat,
        "crises_avoided": crises_avoided,
        "days_without_slope_data": int(no_slope_data),
    }

    return strategy_ret, position, diagnostics


# ---------------------------------------------------------------------------
# Sub-strategy 2: VIX Momentum Filter
# ---------------------------------------------------------------------------

def run_momentum_strategy(
    df: pd.DataFrame,
    vix_upper: float = 25.0,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    VIX Momentum / Multi-Condition Filter Strategy.

    Entry conditions (all must hold simultaneously; 1-day lag):
    1. VIX 10-day SMA < VIX 60-day SMA  (vol trending down / stable)
    2. VIX < vix_upper (not in elevated volatility regime)
    3. SPY 20-day return > 0 (equity trend positive)

    Long SVXY when all conditions hold; flat otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_term_structure`.
    vix_upper : float
        VIX level above which we refuse new long exposure (default 25).
    cost_bps : float
        One-way transaction cost in basis points (default 10).

    Returns
    -------
    (strategy_returns, position_series, diagnostics_dict)
    """
    cost = cost_bps / 10_000.0

    vix = df["VIX"]
    vix_sma10 = df["vix_sma10"]
    vix_sma60 = df["vix_sma60"]
    spy_ret_20d = df["spy_ret_20d"]

    # All three conditions must be true to be long
    cond_sma = vix_sma10 < vix_sma60           # vol regime declining
    cond_level = vix < vix_upper                # not in high-vol regime
    cond_spy = spy_ret_20d > 0.0               # equity trend positive

    raw_position = (cond_sma & cond_level & cond_spy).astype(float)

    # 1-day lag
    position = raw_position.shift(1).fillna(0.0)

    # Transaction costs on position changes
    trades = (position.diff().abs() > 0).astype(float)
    trades.iloc[0] = 0.0

    strategy_ret = position * df["svxy_ret"] - trades * cost
    strategy_ret = strategy_ret.fillna(0.0)

    # --- Diagnostics ---
    in_trade = position > 0
    pct_days_in_trade = float(in_trade.mean())
    n_trades = int(trades.sum())
    avg_vix_in_trade = float(vix[in_trade].mean()) if in_trade.any() else np.nan
    avg_vix_flat = float(vix[~in_trade].mean()) if (~in_trade).any() else np.nan

    # Crises avoided: days where VIX > 30 but we were flat
    crisis_days = vix > DEFAULT_VIX_CRISIS
    crises_avoided = int((crisis_days & ~in_trade).sum())

    # Days each condition individually blocked a trade
    days_blocked_sma = int((~cond_sma & raw_position.shift(1).fillna(0).astype(bool) == False).sum())
    cond_sma_blocks = int((~cond_sma).sum())
    cond_level_blocks = int((~cond_level).sum())
    cond_spy_blocks = int((~cond_spy).sum())

    diagnostics: dict = {
        "pct_days_in_trade": pct_days_in_trade,
        "n_trades": n_trades,
        "avg_vix_in_trade": avg_vix_in_trade,
        "avg_vix_flat": avg_vix_flat,
        "crises_avoided": crises_avoided,
        "days_blocked_vix_sma": cond_sma_blocks,
        "days_blocked_vix_level": cond_level_blocks,
        "days_blocked_spy_trend": cond_spy_blocks,
    }

    return strategy_ret, position, diagnostics


# ---------------------------------------------------------------------------
# Regime Analysis
# ---------------------------------------------------------------------------

def compute_regime_stats(
    returns_dict: dict[str, pd.Series],
    vix: pd.Series,
) -> None:
    """
    Print annualised return and Sharpe for each strategy across VIX regimes.

    Regimes:
    - Low:    VIX < 15
    - Medium: 15 <= VIX < 25
    - High:   VIX >= 25

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of strategy label -> daily return series.
    vix : pd.Series
        Spot VIX series aligned to the same index.
    """
    regimes = {
        "Low VIX  (<15)":   vix < 15,
        "Med VIX (15–25)":  (vix >= 15) & (vix < 25),
        "High VIX (>=25)":  vix >= 25,
    }

    col_w = 14
    label_w = 22
    strategy_labels = list(returns_dict.keys())
    n_cols = len(strategy_labels)
    total_w = label_w + col_w * n_cols * 2 + 2

    print()
    print("=" * total_w)
    print("  VIX Regime Analysis — Annualised Return & Sharpe")
    print("=" * total_w)

    # Header
    header = f"  {'Regime':<{label_w}}"
    for lbl in strategy_labels:
        header += f"  {lbl + ' Ret':>{col_w}}{lbl + ' Sharpe':>{col_w}}"
    print(header)
    print("-" * total_w)

    for regime_name, mask in regimes.items():
        row = f"  {regime_name:<{label_w}}"
        for lbl, rets in returns_dict.items():
            aligned_mask = mask.reindex(rets.index).fillna(False)
            r_sub = rets[aligned_mask].dropna()
            n = len(r_sub)
            if n < 20:
                row += f"{'N/A':>{col_w}}{'N/A':>{col_w}}"
                continue
            ann_ret = float((1 + r_sub).prod() ** (FREQ / n) - 1)
            ann_vol = float(r_sub.std(ddof=1) * np.sqrt(FREQ))
            sharpe = ann_ret / (ann_vol + 1e-12)
            row += f"  {ann_ret:>{col_w - 2}.2%}  {sharpe:>{col_w - 2}.2f}"
        print(row)

    print("=" * total_w)
    print()


# ---------------------------------------------------------------------------
# Diagnostic Summary Printer
# ---------------------------------------------------------------------------

def print_strategy_diagnostics(
    label: str,
    diagnostics: dict,
    position: pd.Series,
    vix: pd.Series,
) -> None:
    """
    Print human-readable diagnostics for a given strategy.

    Parameters
    ----------
    label : str
        Display name for the strategy.
    diagnostics : dict
        Output dict from run_*_strategy functions.
    position : pd.Series
        Binary position series (1 = in trade, 0 = flat).
    vix : pd.Series
        Spot VIX series.
    """
    width = 58
    print()
    print("=" * width)
    print(f"  Diagnostics — {label}")
    print("=" * width)
    print(f"  {'Days in trade:':<30} {diagnostics['pct_days_in_trade']:>8.1%}")
    print(f"  {'Number of trades (one-way):':<30} {diagnostics['n_trades']:>8}")
    avg_in = diagnostics.get("avg_vix_in_trade", np.nan)
    avg_out = diagnostics.get("avg_vix_flat", np.nan)
    if not np.isnan(avg_in):
        print(f"  {'Avg VIX when in trade:':<30} {avg_in:>8.1f}")
    if not np.isnan(avg_out):
        print(f"  {'Avg VIX when flat:':<30} {avg_out:>8.1f}")
    print(f"  {'Crisis days avoided (VIX>30):':<30} {diagnostics['crises_avoided']:>8}")

    # Sub-strategy-specific keys
    if "days_without_slope_data" in diagnostics:
        print(
            f"  {'Days w/o slope data:':<30} "
            f"{diagnostics['days_without_slope_data']:>8}"
        )
    if "days_blocked_vix_sma" in diagnostics:
        print(
            f"  {'Days blocked by VIX SMA:':<30} "
            f"{diagnostics['days_blocked_vix_sma']:>8}"
        )
    if "days_blocked_vix_level" in diagnostics:
        print(
            f"  {'Days blocked by VIX level:':<30} "
            f"{diagnostics['days_blocked_vix_level']:>8}"
        )
    if "days_blocked_spy_trend" in diagnostics:
        print(
            f"  {'Days blocked by SPY trend:':<30} "
            f"{diagnostics['days_blocked_spy_trend']:>8}"
        )
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Combined Tearsheet (multi-strategy equity curves)
# ---------------------------------------------------------------------------

def plot_combined_tearsheet(
    returns_map: dict[str, pd.Series],
    vix: pd.Series,
    title: str = "Vol Term Structure — Combined Tearsheet",
    save_path: Optional[str] = DEFAULT_PLOT_FILE,
    show: bool = True,
) -> None:
    """
    Four-panel combined tearsheet:
      1. Equity curves for all strategies + SPY
      2. Drawdowns for all strategies
      3. Rolling 252-day Sharpe for both sub-strategies
      4. VIX spot time series with crisis threshold shading

    Parameters
    ----------
    returns_map : dict[str, pd.Series]
        Keys: display labels (e.g. "Term Structure", "Momentum", "SPY").
        Values: daily return series.
    vix : pd.Series
        VIX spot series.
    title : str
        Figure title.
    save_path : str, optional
        File path to save the figure (PNG recommended).
    show : bool
        Call plt.show() if True.
    """
    COLORS = {
        "Term Structure": "#1f77b4",   # blue
        "Momentum":       "#2ca02c",   # green
        "SPY":            "#7f7f7f",   # grey
    }
    LS = {
        "Term Structure": "-",
        "Momentum":       "-",
        "SPY":            "--",
    }

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1.5, 1.5, 1.2], hspace=0.50)

    # --- Panel 1: Equity curves ---
    ax1 = fig.add_subplot(gs[0])
    first_ret: Optional[pd.Series] = None

    for label, rets in returns_map.items():
        r = rets.dropna()
        if len(r) == 0:
            continue
        eq = (1 + r).cumprod()
        c = COLORS.get(label, None)
        ls = LS.get(label, "-")
        lw = 1.4 if label != "SPY" else 1.0
        alpha = 1.0 if label != "SPY" else 0.75
        ax1.plot(eq.index, eq.values, linewidth=lw, color=c, linestyle=ls,
                 label=label, alpha=alpha)
        if first_ret is None:
            first_ret = r

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax1.set_ylabel("Growth of $1")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.tick_params(labelbottom=False)

    # --- Panel 2: Drawdowns ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for label, rets in returns_map.items():
        r = rets.dropna()
        if len(r) == 0:
            continue
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1.0
        c = COLORS.get(label, None)
        ls = LS.get(label, "-")
        lw = 1.2 if label != "SPY" else 0.8
        alpha = 0.85 if label != "SPY" else 0.55
        ax2.plot(dd.index, dd.values * 100, linewidth=lw, color=c,
                 linestyle=ls, label=label, alpha=alpha)

    ax2.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.tick_params(labelbottom=False)

    # --- Panel 3: Rolling Sharpe (sub-strategies only) ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for label, rets in returns_map.items():
        if label == "SPY":
            continue
        r = rets.dropna()
        if len(r) < FREQ:
            continue
        mu = r.rolling(FREQ).mean() * FREQ
        sig = r.rolling(FREQ).std(ddof=1) * np.sqrt(FREQ)
        rs = mu / (sig + 1e-12)
        c = COLORS.get(label, None)
        ax3.plot(rs.index, rs.values, linewidth=1.0, color=c, label=label, alpha=0.9)

    ax3.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax3.axhline(1.0, color="darkgreen", linewidth=0.7, linestyle=":", alpha=0.6)
    ax3.set_ylabel(f"Rolling Sharpe\n({FREQ}-day)")
    ax3.set_ylim(-4, 5)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.2, linestyle="--")
    ax3.tick_params(labelbottom=False)

    # --- Panel 4: VIX ---
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    vix_aligned = vix.reindex(
        pd.date_range(vix.index.min(), vix.index.max(), freq="B")
    ).ffill()

    # Only plot VIX over the range we have returns for
    if first_ret is not None:
        vix_plot = vix.reindex(first_ret.index).ffill()
    else:
        vix_plot = vix

    ax4.plot(vix_plot.index, vix_plot.values, linewidth=0.9, color="darkorange",
             label="VIX Spot")
    ax4.axhline(DEFAULT_VIX_CRISIS, color="crimson", linewidth=0.8,
                linestyle="--", alpha=0.7, label=f"Crisis ({DEFAULT_VIX_CRISIS})")
    ax4.axhline(25, color="gold", linewidth=0.8, linestyle=":", alpha=0.7,
                label="Elevated (25)")
    ax4.axhline(15, color="green", linewidth=0.8, linestyle=":", alpha=0.7,
                label="Low (15)")
    ax4.fill_between(
        vix_plot.index, vix_plot.values, DEFAULT_VIX_CRISIS,
        where=(vix_plot.values > DEFAULT_VIX_CRISIS),
        color="crimson", alpha=0.15, interpolate=True,
    )
    ax4.set_ylabel("VIX")
    ax4.legend(fontsize=7, loc="upper right", ncol=2)
    ax4.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Tearsheet saved to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VIX Term Structure / Volatility Risk Premium Backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--contango-threshold",
        type=float,
        default=DEFAULT_CONTANGO_THRESHOLD,
        dest="contango_threshold",
        help="Slope threshold for entering long (e.g. 0.05 = 5%%)",
    )
    parser.add_argument(
        "--vix-crisis-level",
        type=float,
        default=DEFAULT_VIX_CRISIS,
        dest="vix_crisis_level",
        help="VIX level above which all positions are flattened",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=DEFAULT_COST_BPS,
        dest="cost_bps",
        help="One-way transaction cost in basis points",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=DEFAULT_PLOT_FILE,
        dest="plot_file",
        help="Filename for the tearsheet PNG",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        help="Suppress all chart output",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """
    Entry point: download data, run strategies, report results, plot tearsheet.
    """
    args = parse_args(argv)

    if args.no_plot:
        matplotlib.use("Agg")

    # ------------------------------------------------------------------
    # 1. Download and prepare data
    # ------------------------------------------------------------------
    df_raw = download_data(start=args.start, end=args.end)
    df = compute_term_structure(df_raw)

    # SPY returns (benchmark)
    spy_ret = df["spy_ret"].dropna()

    # ------------------------------------------------------------------
    # 2. Run sub-strategy 1: Term Structure Slope
    # ------------------------------------------------------------------
    print("\n[Strategy 1] Running Term Structure Slope Signal ...")
    ts_ret, ts_pos, ts_diag = run_term_structure_strategy(
        df,
        contango_threshold=args.contango_threshold,
        vix_crisis_level=args.vix_crisis_level,
        cost_bps=args.cost_bps,
    )
    print_strategy_diagnostics("Term Structure Slope", ts_diag, ts_pos, df["VIX"])

    # ------------------------------------------------------------------
    # 3. Run sub-strategy 2: VIX Momentum Filter
    # ------------------------------------------------------------------
    print("[Strategy 2] Running VIX Momentum Filter ...")
    mom_ret, mom_pos, mom_diag = run_momentum_strategy(
        df,
        cost_bps=args.cost_bps,
    )
    print_strategy_diagnostics("VIX Momentum Filter", mom_diag, mom_pos, df["VIX"])

    # ------------------------------------------------------------------
    # 4. Align all return series to common date range
    # ------------------------------------------------------------------
    common_idx = ts_ret.index.intersection(mom_ret.index).intersection(spy_ret.index)
    ts_ret_aln = ts_ret.reindex(common_idx).fillna(0.0)
    mom_ret_aln = mom_ret.reindex(common_idx).fillna(0.0)
    spy_ret_aln = spy_ret.reindex(common_idx)

    # Drop leading zeros (before SVXY had enough history)
    first_valid = max(
        ts_ret_aln[ts_ret_aln != 0].index[0] if (ts_ret_aln != 0).any() else common_idx[0],
        mom_ret_aln[mom_ret_aln != 0].index[0] if (mom_ret_aln != 0).any() else common_idx[0],
    )
    ts_ret_aln = ts_ret_aln.loc[first_valid:]
    mom_ret_aln = mom_ret_aln.loc[first_valid:]
    spy_ret_aln = spy_ret_aln.loc[first_valid:]

    # ------------------------------------------------------------------
    # 5. Performance stats
    # ------------------------------------------------------------------
    ts_stats = compute_perf_stats(ts_ret_aln, freq=FREQ, benchmark_returns=spy_ret_aln)
    mom_stats = compute_perf_stats(mom_ret_aln, freq=FREQ, benchmark_returns=spy_ret_aln)
    spy_stats = compute_perf_stats(spy_ret_aln, freq=FREQ)

    print_stats_table(
        {
            "Term Structure": ts_stats,
            "Momentum Filter": mom_stats,
            "SPY": spy_stats,
        },
        title="VIX Term Structure Strategies vs SPY Benchmark",
    )

    # ------------------------------------------------------------------
    # 6. VIX regime analysis
    # ------------------------------------------------------------------
    vix_aligned = df["VIX"].reindex(ts_ret_aln.index)
    compute_regime_stats(
        {
            "Term Structure": ts_ret_aln,
            "Momentum": mom_ret_aln,
            "SPY": spy_ret_aln,
        },
        vix=vix_aligned,
    )

    # ------------------------------------------------------------------
    # 7. Individual tearsheets (via performance.py)
    # ------------------------------------------------------------------
    if not args.no_plot:
        print("[Plot] Generating individual tearsheets ...")

        plot_tearsheet(
            ts_ret_aln,
            benchmark_returns=spy_ret_aln,
            title="Sub-Strategy 1: Term Structure Slope (SVXY)",
            freq=FREQ,
            rolling_window=FREQ,
            save_path=None,
            show=True,
        )

        plot_tearsheet(
            mom_ret_aln,
            benchmark_returns=spy_ret_aln,
            title="Sub-Strategy 2: VIX Momentum Filter (SVXY)",
            freq=FREQ,
            rolling_window=FREQ,
            save_path=None,
            show=True,
        )

    # ------------------------------------------------------------------
    # 8. Combined multi-strategy tearsheet
    # ------------------------------------------------------------------
    returns_map = {
        "Term Structure": ts_ret_aln,
        "Momentum": mom_ret_aln,
        "SPY": spy_ret_aln,
    }

    plot_combined_tearsheet(
        returns_map=returns_map,
        vix=df["VIX"].reindex(ts_ret_aln.index),
        title=(
            f"Vol Term Structure / VIX Contango Strategies  "
            f"[{ts_ret_aln.index[0].date()} — {ts_ret_aln.index[-1].date()}]"
        ),
        save_path=args.plot_file,
        show=not args.no_plot,
    )

    print("\n[Done] Backtest complete.")


if __name__ == "__main__":
    main()
