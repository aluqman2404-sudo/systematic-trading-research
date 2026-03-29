"""
Macro System Backtest (Monthly): Trend + Carry + Volatility (VRP proxy)

- Trend sleeve: long/short or long-only cross-sectional momentum
- Carry sleeve: DBV vs SHY (risk-on filter), falls back to SHY if DBV missing
- Vol sleeve: SVXY vs SHY (risk-on filter), falls back to SHY if SVXY missing
- Sleeve risk balancing: allocates sleeve weights inversely to trailing vol
- Portfolio volatility targeting + optional drawdown guard overlay

Data: Yahoo Finance via yfinance (Adj Close)
Execution: Simulated monthly rebalancing, 1-month lag (avoid lookahead)
"""

import argparse
import itertools
import os
import numpy as np
import pandas as pd
import yfinance as yf

# Avoid matplotlib cache permission issues on some systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


# ----------------------------
# GLOBAL CONFIG
# ----------------------------
START = "2012-01-01"
END = None  # None = today
INITIAL_CAPITAL = 100_000

# Trend universe (diversified ETFs)
TREND_TICKERS = ["SPY", "IEF", "GLD", "DBC", "EFA"]

# Carry + Vol proxies
CARRY_TICKER = "DBV"
VOL_TICKER = "SVXY"
HEDGE_TICKER = "VIXY"  # used in risk-on filter
DEFENSIVE = "SHY"      # cash-like

# Costs
TURNOVER_COST = 0.001  # 10 bps per 1.0 turnover (very rough)
DEFAULT_CACHE_PATH = "macro_system1_prices_cache.csv"

# Targets requested by user
TARGET_MIN_CAGR = 0.10
TARGET_MIN_SHARPE = 1.00
TARGET_MIN_MAX_DRAWDOWN = -0.20  # e.g. -0.20 means drawdown not worse than -20%


# ----------------------------
# DEFAULT STRATEGY CONFIG
# ----------------------------
DEFAULT_STRATEGY_CONFIG = {
    "budget_trend": 0.40,
    "budget_carry": 0.30,
    "budget_vol": 0.30,
    "mom_lookback_months": 12,
    "roll_vol_months": 12,
    "target_portfolio_vol": 0.10,
    "max_vol_target_leverage": 1.50,
    "max_vol_sleeve_weight": 0.20,
    "trend_mode": "long_only",      # long_only or long_short
    "max_long_assets": 3,            # used only in long_only mode
    "spy_mom_threshold": 0.00,
    "hedge_mom_threshold": 0.00,
    "dd_guard_threshold": -0.12,
    "dd_guard_risk_multiplier": 0.50,  # if DD guard triggered: keep this % of risk strategy
    "max_gross_leverage": 1.25,
    "turnover_cost": TURNOVER_COST,
}


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
                threads=False,
            )

            if df is None or len(df) == 0:
                raise RuntimeError("yfinance returned None/empty dataframe")

            if "Adj Close" not in df.columns:
                raise RuntimeError(f"Unexpected columns from yfinance: {df.columns}")

            px = df["Adj Close"]
            if isinstance(px, pd.Series):
                px = px.to_frame()

            px = px.dropna(axis=1, how="all")
            px = px.ffill().dropna(how="all")

            if px.shape[1] == 0:
                raise RuntimeError("All tickers failed download (no usable columns).")

            failed = sorted(set(tickers) - set(px.columns.tolist()))
            if failed:
                print(f"Dropped failed tickers: {failed}")

            return px

        except Exception as e:
            last_err = e
            print(f"Download attempt {attempt}/{max_retries} failed: {e}")

    raise RuntimeError(f"Failed to download after {max_retries} retries. Last error: {last_err}")


def load_prices_csv(csv_path: str) -> pd.DataFrame:
    """
    Expected CSV format:
    - first column: date/datetime
    - remaining columns: ticker adjusted-close price series
    """
    px = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if px.empty:
        raise RuntimeError(f"Price CSV is empty: {csv_path}")
    px = px.sort_index()
    px = px.apply(pd.to_numeric, errors="coerce")
    px = px.dropna(axis=1, how="all").ffill().dropna(how="all")
    if px.shape[1] == 0:
        raise RuntimeError(f"No usable price columns in CSV: {csv_path}")
    return px


def get_prices_with_fallback(
    tickers,
    start=START,
    end=END,
    max_retries=3,
    prices_csv=None,
    cache_path=DEFAULT_CACHE_PATH,
):
    try:
        px = download_adj_close(tickers, start=start, end=end, max_retries=max_retries)
        if cache_path:
            try:
                px.to_csv(cache_path)
                print(f"Saved price cache: {cache_path}")
            except Exception as e:
                print(f"Warning: could not save cache ({cache_path}): {e}")
        return px
    except Exception as download_err:
        if prices_csv and os.path.exists(prices_csv):
            print(f"Yahoo download failed; loading local CSV: {prices_csv}")
            return load_prices_csv(prices_csv)
        if cache_path and os.path.exists(cache_path):
            print(f"Yahoo download failed; loading cache: {cache_path}")
            return load_prices_csv(cache_path)
        raise RuntimeError(
            "Could not download market data from Yahoo and no local fallback was found.\n"
            f"Download error: {download_err}\n"
            "Run with --prices-csv <path_to_prices.csv> or ensure internet/DNS access to Yahoo."
        ) from download_err


def to_monthly_last(px_daily: pd.DataFrame) -> pd.DataFrame:
    return px_daily.resample("ME").last()


def monthly_returns(px_m: pd.DataFrame) -> pd.DataFrame:
    return px_m.pct_change().dropna()


def momentum_n(px_m: pd.DataFrame, lookback_months: int) -> pd.DataFrame:
    return px_m.pct_change(lookback_months)


def annualized_vol(x: pd.Series, months: int) -> pd.Series:
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


def constraints_met(stats, min_cagr=TARGET_MIN_CAGR, min_sharpe=TARGET_MIN_SHARPE, min_mdd=TARGET_MIN_MAX_DRAWDOWN):
    return (
        stats["CAGR"] >= min_cagr
        and stats["Sharpe"] >= min_sharpe
        and stats["Max Drawdown"] >= min_mdd
    )


def objective_score(stats, min_cagr=TARGET_MIN_CAGR, min_sharpe=TARGET_MIN_SHARPE, min_mdd=TARGET_MIN_MAX_DRAWDOWN):
    cagr_shortfall = max(0.0, min_cagr - stats["CAGR"])
    sharpe_shortfall = max(0.0, min_sharpe - stats["Sharpe"])
    dd_shortfall = max(0.0, min_mdd - stats["Max Drawdown"])

    # Higher is better. Heavy penalties for constraint violations.
    return (
        2.0 * stats["Sharpe"]
        + 1.5 * stats["CAGR"]
        - 6.0 * cagr_shortfall
        - 4.0 * sharpe_shortfall
        - 8.0 * dd_shortfall
    )


def build_candidate_configs(fast_mode=False):
    budgets = [
        (0.50, 0.25, 0.25),
        (0.45, 0.30, 0.25),
        (0.40, 0.35, 0.25),
    ]
    trend_settings = [
        ("long_only", 2),
        ("long_only", 3),
        ("long_short", 0),
    ]
    dd_controls = [
        (-0.10, 0.00),
        (-0.12, 0.25),
        (-0.15, 0.50),
    ]

    if fast_mode:
        lookbacks = [6, 12]
        roll_windows = [6, 12]
        target_vols = [0.10]
        max_levers = [1.25]
        vol_caps = [0.10, 0.20]
        spy_thresholds = [0.00]
        gross_limits = [1.00, 1.25]
    else:
        lookbacks = [6, 12]
        roll_windows = [6, 12]
        target_vols = [0.08, 0.10, 0.12]
        max_levers = [1.25, 1.50]
        vol_caps = [0.10, 0.20]
        spy_thresholds = [0.00, 0.02]
        gross_limits = [1.00, 1.25]

    for lookback, roll_vol, target_vol, max_lev, vol_cap, spy_thr, (bt, bc, bv), (mode, max_longs), (dd_thr, dd_mult), max_gross in itertools.product(
        lookbacks,
        roll_windows,
        target_vols,
        max_levers,
        vol_caps,
        spy_thresholds,
        budgets,
        trend_settings,
        dd_controls,
        gross_limits,
    ):
        cfg = {
            "budget_trend": bt,
            "budget_carry": bc,
            "budget_vol": bv,
            "mom_lookback_months": lookback,
            "roll_vol_months": roll_vol,
            "target_portfolio_vol": target_vol,
            "max_vol_target_leverage": max_lev,
            "max_vol_sleeve_weight": vol_cap,
            "trend_mode": mode,
            "max_long_assets": max_longs,
            "spy_mom_threshold": spy_thr,
            "hedge_mom_threshold": 0.00,
            "dd_guard_threshold": dd_thr,
            "dd_guard_risk_multiplier": dd_mult,
            "max_gross_leverage": max_gross,
            "turnover_cost": TURNOVER_COST,
        }
        yield cfg


def run_single_backtest(px_d: pd.DataFrame, config: dict):
    px_m = to_monthly_last(px_d)
    rets_all = monthly_returns(px_m)

    if DEFENSIVE not in rets_all.columns:
        raise RuntimeError(f"Required defensive ticker {DEFENSIVE} is missing from downloaded data.")

    if len(rets_all) < 24:
        raise ValueError(f"Not enough monthly data after cleaning: only {len(rets_all)} months.")

    lookback = int(config["mom_lookback_months"])
    roll_vol_months = int(config["roll_vol_months"])

    mom = momentum_n(px_m, lookback).dropna()
    common_idx = rets_all.index.intersection(mom.index)
    rets = rets_all.loc[common_idx]
    mom = mom.loc[common_idx]

    if len(rets) < max(roll_vol_months + 12, 24):
        raise ValueError("Insufficient aligned history for selected parameters.")

    have_carry = CARRY_TICKER in rets.columns
    have_vol = VOL_TICKER in rets.columns
    have_hedge = HEDGE_TICKER in mom.columns

    if "SPY" not in mom.columns:
        raise RuntimeError("SPY is required for risk-on filter.")

    spy_thr = float(config["spy_mom_threshold"])
    hedge_thr = float(config["hedge_mom_threshold"])
    if have_hedge:
        risk_on = (mom["SPY"] > spy_thr) & (mom[HEDGE_TICKER] < -hedge_thr)
    else:
        risk_on = mom["SPY"] > spy_thr

    trend_tickers_present = [t for t in TREND_TICKERS if t in mom.columns and t in rets.columns]
    if len(trend_tickers_present) < 4:
        raise RuntimeError(
            f"Not enough TREND tickers downloaded to run (need >=4). Got: {trend_tickers_present}"
        )

    trend_w = pd.DataFrame(0.0, index=rets.index, columns=trend_tickers_present)
    trend_mode = str(config["trend_mode"])
    max_long_assets = int(config["max_long_assets"])

    for dt in rets.index:
        m = mom.loc[dt, trend_tickers_present].dropna()
        if len(m) < 4:
            continue

        ranks = m.sort_values(ascending=False)

        if trend_mode == "long_short":
            n = len(ranks) // 2
            if n <= 0:
                continue
            longs = ranks.index[:n]
            shorts = ranks.index[-n:]
            trend_w.loc[dt, longs] = 1.0 / (2 * n)
            trend_w.loc[dt, shorts] = -1.0 / (2 * n)

        elif trend_mode == "long_only":
            positives = ranks[ranks > 0]
            if len(positives) == 0:
                continue
            if max_long_assets > 0:
                longs = positives.index[:max_long_assets]
            else:
                longs = positives.index
            if len(longs) == 0:
                continue
            trend_w.loc[dt, longs] = 1.0 / len(longs)

        else:
            raise ValueError(f"Unknown trend_mode: {trend_mode}")

    trend_empty = trend_w.abs().sum(axis=1) == 0

    if have_carry:
        carry_w = pd.DataFrame(0.0, index=rets.index, columns=[CARRY_TICKER, DEFENSIVE])
        carry_w.loc[risk_on, CARRY_TICKER] = 1.0
        carry_w.loc[~risk_on, DEFENSIVE] = 1.0
    else:
        carry_w = pd.DataFrame(0.0, index=rets.index, columns=[DEFENSIVE])
        carry_w[DEFENSIVE] = 1.0

    if have_vol:
        vol_w = pd.DataFrame(0.0, index=rets.index, columns=[VOL_TICKER, DEFENSIVE])
        vol_w.loc[risk_on, VOL_TICKER] = 1.0
        vol_w.loc[~risk_on, DEFENSIVE] = 1.0
    else:
        vol_w = pd.DataFrame(0.0, index=rets.index, columns=[DEFENSIVE])
        vol_w[DEFENSIVE] = 1.0

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

    trend_vol = annualized_vol(trend_ret, roll_vol_months)
    carry_vol = annualized_vol(carry_ret, roll_vol_months)
    vol_vol = annualized_vol(vol_ret, roll_vol_months)

    sleeve_w = pd.DataFrame(
        index=rets.index,
        data={
            "trend": float(config["budget_trend"]) / (trend_vol + 1e-12),
            "carry": float(config["budget_carry"]) / (carry_vol + 1e-12),
            "vol": float(config["budget_vol"]) / (vol_vol + 1e-12),
        },
    ).fillna(0.0)

    vol_cap = config["max_vol_sleeve_weight"]
    if vol_cap is not None:
        sleeve_w["vol"] = sleeve_w["vol"].clip(upper=float(vol_cap))

    sleeve_w = sleeve_w.div(sleeve_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    cols = list(trend_tickers_present) + [DEFENSIVE]
    if have_carry:
        cols += [CARRY_TICKER]
    if have_vol:
        cols += [VOL_TICKER]

    all_cols = sorted(set(cols))
    W = pd.DataFrame(0.0, index=rets.index, columns=all_cols)

    for t in trend_tickers_present:
        W[t] += trend_w[t] * sleeve_w["trend"]
    W.loc[trend_empty, DEFENSIVE] += sleeve_w.loc[trend_empty, "trend"]

    if have_carry:
        W[[CARRY_TICKER, DEFENSIVE]] += carry_w.mul(sleeve_w["carry"], axis=0)
    else:
        W[DEFENSIVE] += sleeve_w["carry"]

    if have_vol:
        W[[VOL_TICKER, DEFENSIVE]] += vol_w.mul(sleeve_w["vol"], axis=0)
    else:
        W[DEFENSIVE] += sleeve_w["vol"]

    # Ensure fully invested on net basis.
    net_sum = W.sum(axis=1)
    W.loc[net_sum == 0, DEFENSIVE] = 1.0
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Cap gross leverage and put residual in defensive asset.
    max_gross = float(config["max_gross_leverage"])
    gross = W.abs().sum(axis=1)
    gross_scale = (max_gross / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    W = W.mul(gross_scale, axis=0)
    W[DEFENSIVE] += 1.0 - W.sum(axis=1)

    W_lag = W.shift(1).dropna()
    rets2 = rets.loc[W_lag.index, W_lag.columns]

    port_ret_gross = (W_lag * rets2).sum(axis=1)

    turnover = W.diff().abs().sum(axis=1).loc[port_ret_gross.index]
    port_ret_net = port_ret_gross - float(config["turnover_cost"]) * turnover

    port_vol = annualized_vol(port_ret_net, roll_vol_months)
    scaling = (
        float(config["target_portfolio_vol"]) / (port_vol + 1e-12)
    ).clip(upper=float(config["max_vol_target_leverage"]))
    port_ret_vt = port_ret_net * scaling.shift(1).fillna(0.0)

    # Drawdown guard: if trailing DD breaches threshold, reduce risk next month.
    base_equity = (1 + port_ret_vt).cumprod()
    base_dd = base_equity / base_equity.cummax() - 1.0
    dd_trigger = base_dd.shift(1).fillna(0.0) <= float(config["dd_guard_threshold"])
    dd_mult = float(config["dd_guard_risk_multiplier"])

    port_ret_final = pd.Series(
        np.where(
            dd_trigger,
            dd_mult * port_ret_vt + (1.0 - dd_mult) * rets2[DEFENSIVE],
            port_ret_vt,
        ),
        index=port_ret_vt.index,
    )

    equity = (1 + port_ret_final).cumprod() * INITIAL_CAPITAL
    stats = perf_stats(equity, port_ret_final)

    return {
        "prices_monthly": px_m,
        "returns_monthly": rets,
        "momentum": mom,
        "weights": W,
        "sleeve_weights": sleeve_w,
        "portfolio_returns_net": port_ret_net,
        "portfolio_returns_vol_targeted": port_ret_vt,
        "portfolio_returns_final": port_ret_final,
        "equity": equity,
        "stats": stats,
        "config": config,
        "diagnostics": {
            "have_carry": have_carry,
            "have_vol": have_vol,
            "have_hedge": have_hedge,
            "trend_tickers_present": trend_tickers_present,
        },
    }


def backtest_three_engines(config=None, px_d=None):
    """
    Run a single backtest with the provided config.
    """
    merged_cfg = DEFAULT_STRATEGY_CONFIG.copy()
    if config:
        merged_cfg.update(config)

    if px_d is None:
        requested = sorted(set(TREND_TICKERS + [CARRY_TICKER, VOL_TICKER, HEDGE_TICKER, DEFENSIVE]))
        px_d = get_prices_with_fallback(requested, start=START, end=END)

    return run_single_backtest(px_d=px_d, config=merged_cfg)


def optimize_for_targets(
    min_cagr=TARGET_MIN_CAGR,
    min_sharpe=TARGET_MIN_SHARPE,
    min_mdd=TARGET_MIN_MAX_DRAWDOWN,
    prices_csv=None,
    cache_path=DEFAULT_CACHE_PATH,
    fast_mode=False,
):
    """
    Downloads data once, sweeps candidate configs, and returns:
    - best_feasible: highest-Sharpe config that meets all thresholds
    - best_overall: closest config if none are feasible
    """
    requested = sorted(set(TREND_TICKERS + [CARRY_TICKER, VOL_TICKER, HEDGE_TICKER, DEFENSIVE]))
    px_d = get_prices_with_fallback(
        requested,
        start=START,
        end=END,
        prices_csv=prices_csv,
        cache_path=cache_path,
    )

    best_feasible = None
    best_overall = None
    best_overall_score = -1e99
    best_feasible_key = None

    tested = 0
    for cfg in build_candidate_configs(fast_mode=fast_mode):
        tested += 1
        try:
            res = backtest_three_engines(config=cfg, px_d=px_d)
        except Exception:
            continue

        stats = res["stats"]
        score = objective_score(stats, min_cagr=min_cagr, min_sharpe=min_sharpe, min_mdd=min_mdd)

        if score > best_overall_score:
            best_overall = res
            best_overall_score = score

        if constraints_met(stats, min_cagr=min_cagr, min_sharpe=min_sharpe, min_mdd=min_mdd):
            key = (stats["Sharpe"], stats["CAGR"], stats["Max Drawdown"])
            if best_feasible is None or key > best_feasible_key:
                best_feasible = res
                best_feasible_key = key

    if best_overall is None:
        raise RuntimeError("No valid backtest run completed from candidate configurations.")

    return {
        "tested_configs": tested,
        "best_feasible": best_feasible,
        "best_overall": best_overall,
        "constraints": {
            "min_cagr": min_cagr,
            "min_sharpe": min_sharpe,
            "min_mdd": min_mdd,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constraint-tuned macro system backtest.")
    parser.add_argument("--prices-csv", type=str, default=None, help="Local fallback price CSV (date index + ticker columns).")
    parser.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH, help="Path to save/load downloaded price cache.")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib equity plot.")
    parser.add_argument("--fast", action="store_true", help="Run a smaller candidate grid for faster optimization.")
    args = parser.parse_args()

    scan = optimize_for_targets(
        min_cagr=TARGET_MIN_CAGR,
        min_sharpe=TARGET_MIN_SHARPE,
        min_mdd=TARGET_MIN_MAX_DRAWDOWN,
        prices_csv=args.prices_csv,
        cache_path=args.cache_path,
        fast_mode=args.fast,
    )

    res = scan["best_feasible"] if scan["best_feasible"] is not None else scan["best_overall"]

    print("\n=== Optimization Summary ===")
    print(f"Tested configs: {scan['tested_configs']}")
    print(f"Thresholds: CAGR >= {TARGET_MIN_CAGR:.0%}, Sharpe >= {TARGET_MIN_SHARPE:.2f}, Max Drawdown >= {TARGET_MIN_MAX_DRAWDOWN:.0%}")
    print(f"Meets all thresholds: {scan['best_feasible'] is not None}")

    print("\n=== Selected Config ===")
    for k, v in res["config"].items():
        print(f"{k}: {v}")

    print("\n=== Diagnostics ===")
    for k, v in res["diagnostics"].items():
        print(f"{k}: {v}")

    print("\n=== Performance (Vol-Targeted, Net of turnover cost, DD guard applied) ===")
    print(f"CAGR: {res['stats']['CAGR']:.2%}")
    print(f"Ann.Vol: {res['stats']['Ann.Vol']:.2%}")
    print(f"Sharpe (rf~0): {res['stats']['Sharpe']:.2f}")
    print(f"Max Drawdown: {res['stats']['Max Drawdown']:.2%}")

    if scan["best_feasible"] is None:
        print("\nWARNING: No tested configuration satisfied all constraints; showing closest result.")

    if not args.no_plot:
        res["equity"].plot(
            title="Trend + Carry + Vol (Proxy) - Constraint-Tuned Equity Curve",
            figsize=(10, 4),
        )
        plt.ylabel("Equity ($)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
