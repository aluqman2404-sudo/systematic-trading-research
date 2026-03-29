"""
performance.py — Shared performance analytics for all backtest strategies.

Metrics
-------
CAGR, Annualized Vol, Sharpe (rf=0), Sortino, Calmar,
Max Drawdown, Avg DD Duration, Skewness, Excess Kurtosis,
95% VaR (historical), 95% CVaR (Expected Shortfall),
Win Rate, Beta, Correlation, Tracking Error, Information Ratio

Usage
-----
    from performance import compute_perf_stats, print_stats_table, plot_tearsheet

    stats = compute_perf_stats(returns, freq=12, benchmark_returns=spy_rets)
    print_stats_table({"Strategy": stats, "SPY": spy_stats})
    plot_tearsheet(returns, benchmark_returns=spy_rets, title="My Strategy")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

__all__ = ["compute_perf_stats", "print_stats_table", "plot_tearsheet"]


def compute_perf_stats(
    returns: pd.Series,
    freq: int = 252,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Compute comprehensive performance statistics for a return series.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns (not cumulative). Index must be datetime.
    freq : int
        Periods per year. Use 252 for daily, 12 for monthly.
    benchmark_returns : pd.Series, optional
        Benchmark periodic returns for relative metrics (Beta, IR, etc.).
    risk_free_rate : float
        Annualised risk-free rate (default 0). Used in Sharpe/Sortino.

    Returns
    -------
    dict[str, float]  — all keys defined in __all__ metric names.
    """
    r = returns.dropna()
    if len(r) < 2:
        raise ValueError("Need at least 2 non-NaN return observations.")

    rf_per_period = risk_free_rate / freq
    excess = r - rf_per_period

    equity = (1 + r).cumprod()
    years = len(r) / freq
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1)

    ann_vol = float(r.std(ddof=1) * np.sqrt(freq))
    sharpe = float((excess.mean() * freq) / (ann_vol + 1e-12))

    # Sortino: penalises only downside deviation
    downside = r[r < rf_per_period] - rf_per_period
    downside_vol = float(downside.std(ddof=1) * np.sqrt(freq)) if len(downside) > 1 else 1e-12
    sortino = float((excess.mean() * freq) / (downside_vol + 1e-12))

    # Drawdown metrics
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else np.nan

    # Average drawdown duration (in periods)
    in_dd = (dd < 0).astype(int).values
    durations: list[int] = []
    streak = 0
    for v in in_dd:
        if v:
            streak += 1
        elif streak > 0:
            durations.append(streak)
            streak = 0
    if streak > 0:
        durations.append(streak)
    avg_dd_dur = float(np.mean(durations)) if durations else 0.0

    # Higher-moment statistics
    skew = float(scipy_stats.skew(r, bias=False))
    kurt = float(scipy_stats.kurtosis(r, bias=False, fisher=True))  # excess kurtosis

    # Tail-risk (historical)
    var_95 = float(np.percentile(r, 5))
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95

    # Win rate (fraction of positive-return periods)
    win_rate = float((r > 0).mean())

    result: dict[str, float] = {
        "CAGR":                      cagr,
        "Ann. Volatility":           ann_vol,
        "Sharpe Ratio":              sharpe,
        "Sortino Ratio":             sortino,
        "Calmar Ratio":              calmar,
        "Max Drawdown":              max_dd,
        "Avg DD Duration (periods)": avg_dd_dur,
        "Skewness":                  skew,
        "Excess Kurtosis":           kurt,
        "VaR 95% (1-period)":        var_95,
        "CVaR 95% (1-period)":       cvar_95,
        "Win Rate":                  win_rate,
    }

    # Benchmark-relative metrics (computed on aligned intersection)
    if benchmark_returns is not None:
        bm = benchmark_returns.dropna().reindex(r.index).dropna()
        r_aln = r.reindex(bm.index).dropna()
        bm = bm.reindex(r_aln.index)

        if len(r_aln) > 10:
            cov_mat = np.cov(r_aln.values, bm.values)
            beta = float(cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12))
            corr = float(np.corrcoef(r_aln.values, bm.values)[0, 1])

            active = r_aln - bm
            te = float(active.std(ddof=1) * np.sqrt(freq))
            ir = float((active.mean() * freq) / (te + 1e-12))

            bm_eq = (1 + bm).cumprod()
            bm_years = len(bm) / freq
            bm_cagr = float(bm_eq.iloc[-1] ** (1.0 / bm_years) - 1) if bm_years > 0 else 0.0
            alpha = cagr - (risk_free_rate + beta * (bm_cagr - risk_free_rate))

            result.update({
                "Alpha (Jensen)":       alpha,
                "Beta":                 beta,
                "Correlation vs BM":    corr,
                "Tracking Error":       te,
                "Information Ratio":    ir,
            })

    return result


def print_stats_table(
    stats_dict: dict[str, dict[str, float]],
    title: str = "Performance Summary",
) -> None:
    """
    Print a formatted side-by-side performance table.

    Parameters
    ----------
    stats_dict : dict
        Keys are column labels (e.g. "Strategy", "SPY"), values are
        dicts returned by compute_perf_stats().
    title : str
        Table heading.
    """
    PCT_FIELDS = {
        "CAGR", "Ann. Volatility", "Max Drawdown",
        "VaR 95% (1-period)", "CVaR 95% (1-period)", "Win Rate",
        "Tracking Error", "Alpha (Jensen)",
    }

    labels = list(stats_dict.keys())
    # Union of all keys, preserving insertion order of first dict
    first_keys = list(next(iter(stats_dict.values())).keys())
    all_keys = list(dict.fromkeys(first_keys + [k for d in stats_dict.values() for k in d]))

    col_w = max(16, max((len(lb) for lb in labels), default=0) + 2)
    key_w = 30
    total_w = key_w + col_w * len(labels) + 2

    print()
    print("=" * total_w)
    print(f"  {title}")
    print("=" * total_w)
    header = f"  {'Metric':<{key_w - 2}}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(header)
    print("-" * total_w)

    for k in all_keys:
        row = f"  {k:<{key_w - 2}}"
        for lb in labels:
            v = stats_dict[lb].get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row += f"{'N/A':>{col_w}}"
            elif k in PCT_FIELDS:
                row += f"{v:>{col_w - 1}.2%} "
            else:
                row += f"{v:>{col_w}.3f}"
        print(row)

    print("=" * total_w)
    print()


def plot_tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Strategy Tearsheet",
    freq: int = 252,
    rolling_window: int | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Three-panel tearsheet: equity curve, drawdown, rolling Sharpe.

    Parameters
    ----------
    returns : pd.Series
        Strategy periodic returns with datetime index.
    benchmark_returns : pd.Series, optional
        Benchmark returns (normalised to same start value on the chart).
    title : str
        Figure title.
    freq : int
        Annualisation factor (252=daily, 12=monthly).
    rolling_window : int, optional
        Window for rolling Sharpe. Defaults to freq (1 year).
    save_path : str, optional
        If given, saves the figure to this path (PNG, PDF, etc.).
    show : bool
        Whether to call plt.show().
    """
    r = returns.dropna()
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1.0

    if rolling_window is None:
        rolling_window = freq

    def _rolling_sharpe(ret: pd.Series, w: int) -> pd.Series:
        mu = ret.rolling(w).mean() * freq
        sigma = ret.rolling(w).std(ddof=1) * np.sqrt(freq)
        return mu / (sigma + 1e-12)

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.0, 1.5], hspace=0.45)

    # ------------------------------------------------------------------
    # Panel 1: Equity curve
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(equity.index, equity.values, linewidth=1.8, color="royalblue", label="Strategy")

    if benchmark_returns is not None:
        bm = benchmark_returns.dropna().reindex(r.index).ffill().dropna()
        bm_eq = (1 + bm).cumprod()
        # Rescale benchmark to same start
        bm_eq = bm_eq / bm_eq.iloc[0] * equity.iloc[0]
        ax1.plot(bm_eq.index, bm_eq.values, linewidth=1.2, color="grey",
                 linestyle="--", label="Benchmark (SPY)", alpha=0.8)
        ax1.legend(fontsize=9, loc="upper left")

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax1.set_ylabel("Growth of $1")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.tick_params(labelbottom=False)

    # ------------------------------------------------------------------
    # Panel 2: Drawdown
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dd.index, dd.values * 100, 0, color="crimson", alpha=0.45, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    lower = min(dd.min() * 110, -1)
    ax2.set_ylim(lower, 2)
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.tick_params(labelbottom=False)

    # ------------------------------------------------------------------
    # Panel 3: Rolling Sharpe
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    rs = _rolling_sharpe(r, rolling_window)
    ax3.plot(rs.index, rs.values, linewidth=1.0, color="teal")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax3.axhline(1.0, color="green", linewidth=0.7, linestyle=":", alpha=0.6)
    ax3.fill_between(rs.index, rs.values, 0,
                     where=(rs >= 0), color="teal", alpha=0.12, interpolate=True)
    ax3.fill_between(rs.index, rs.values, 0,
                     where=(rs < 0), color="crimson", alpha=0.12, interpolate=True)
    ax3.set_ylabel(f"Rolling Sharpe\n({rolling_window}-period)")
    ax3.set_ylim(-3.5, 4.5)
    ax3.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved tearsheet to: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
