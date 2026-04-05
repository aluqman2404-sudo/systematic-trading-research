
import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# UTIL
# ----------------------------
def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")

def _valid_us_symbol(sym: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", sym))

def chunked(values: list[str], n: int):
    for i in range(0, len(values), n):
        yield values[i : i + n]

def tickers_hash(tickers: list[str]) -> str:
    payload = "\n".join(sorted([normalize_ticker(t) for t in tickers])).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

def normalize_index_to_naive_utc(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    idx = pd.to_datetime(series.index, errors="coerce", utc=True)
    mask = ~idx.isna()
    s = series.loc[mask].copy()
    s.index = idx[mask].tz_localize(None)
    return s.sort_index()

def _asof_value(series: pd.Series, dt: pd.Timestamp) -> float | None:
    if len(series) == 0:
        return None
    s = series.loc[:dt]
    if len(s) == 0:
        return None
    v = float(s.iloc[-1])
    if not np.isfinite(v):
        return None
    return v

def compute_stats(returns: pd.Series, benchmark_returns: pd.Series | None = None) -> dict[str, float]:
    from performance1 import compute_perf_stats
    return compute_perf_stats(returns, freq=252, benchmark_returns=benchmark_returns)

def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def load_external_fundamentals(
    fundamentals_file: str,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """
    Load point-in-time fundamentals from CSV/Parquet.
    Required columns:
      - ticker
      - date
      - ncav
    Optional columns:
      - shares_outstanding (or shares)
      - sector
    """
    path = Path(fundamentals_file)
    if not path.exists():
        raise FileNotFoundError(f"Fundamentals file not found: {fundamentals_file}")

    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError("Fundamentals file is empty.")

    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"ticker", "date", "ncav"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fundamentals file missing required columns: {missing}")

    shares_col = None
    if "shares_outstanding" in df.columns:
        shares_col = "shares_outstanding"
    elif "shares" in df.columns:
        shares_col = "shares"

    df["ticker"] = df["ticker"].astype(str).map(normalize_ticker)
    dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.loc[~dt.isna()].copy()
    df["date"] = dt.loc[~dt.isna()].dt.tz_localize(None)
    df["ncav"] = pd.to_numeric(df["ncav"], errors="coerce")
    df = df.loc[df["ticker"].str.len() > 0]
    df = df.dropna(subset=["ncav"]).sort_values(["ticker", "date"])

    if shares_col is not None:
        df[shares_col] = pd.to_numeric(df[shares_col], errors="coerce")

    ncav_map: dict[str, pd.Series] = {}
    shares_map: dict[str, pd.Series] = {}
    sector_map: dict[str, str] = {}

    for ticker, g in df.groupby("ticker", sort=False):
        s_ncav = pd.Series(g["ncav"].values, index=pd.to_datetime(g["date"])).sort_index()
        s_ncav = pd.to_numeric(s_ncav, errors="coerce").dropna()
        if len(s_ncav) > 0:
            ncav_map[ticker] = s_ncav

        if shares_col is not None:
            s_sh = pd.Series(g[shares_col].values, index=pd.to_datetime(g["date"])).sort_index()
            s_sh = pd.to_numeric(s_sh, errors="coerce").dropna()
            if len(s_sh) > 0:
                shares_map[ticker] = s_sh

        if "sector" in g.columns:
            sec = g["sector"].dropna()
            if len(sec) > 0:
                v = str(sec.iloc[-1]).strip()
                if v:
                    sector_map[ticker] = v

    return ncav_map, shares_map, sector_map


# ----------------------------
# TICKERS
# ----------------------------
def load_tickers(tickers_file: str | None) -> list[str]:
    if not tickers_file:
        raise RuntimeError("Use --tickers-file (one ticker per line). This will be MUCH faster and more reliable.")
    path = Path(tickers_file)
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {tickers_file}")
    tickers = [normalize_ticker(line) for line in path.read_text().splitlines() if line.strip()]
    tickers = sorted(list(dict.fromkeys([t for t in tickers if _valid_us_symbol(t)])))
    if not tickers:
        raise ValueError("Ticker file is empty or invalid after cleaning.")
    return tickers


# ----------------------------
# PRICES (CACHED)
# ----------------------------
def download_prices_volume(tickers: list[str], start: str, end: str,
                           batch_size: int, min_price_obs: int,
                           batch_sleep: float, batch_retries: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_parts, volume_parts = [], []

    for i, batch in enumerate(chunked(tickers, batch_size), start=1):
        print(f"Downloading prices batch {i} ({len(batch)} tickers)...")
        data = None
        for attempt in range(batch_retries + 1):
            try:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    actions=False,
                    group_by="column",
                    progress=False,
                    threads=True,
                )
            except Exception:
                data = None

            if data is not None and len(data) > 0:
                break

            if attempt < batch_retries:
                wait_s = max(0.5, batch_sleep * (2 ** attempt))
                print(f"  retry {attempt + 1}/{batch_retries} after {wait_s:.2f}s")
                time.sleep(wait_s)

        if data is None or len(data) == 0:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in data.columns.get_level_values(0):
                continue
            close_chunk = data["Close"].copy()
            volume_chunk = data["Volume"].copy() if "Volume" in data.columns.get_level_values(0) else pd.DataFrame(index=close_chunk.index)
        else:
            # single ticker
            t = batch[0]
            if "Close" not in data.columns:
                continue
            close_chunk = data[["Close"]].rename(columns={"Close": t})
            volume_chunk = data[["Volume"]].rename(columns={"Volume": t}) if "Volume" in data.columns else pd.DataFrame(index=close_chunk.index, columns=[t])

        close_chunk.columns = [normalize_ticker(c) for c in close_chunk.columns]
        volume_chunk.columns = [normalize_ticker(c) for c in volume_chunk.columns]

        close_parts.append(close_chunk)
        volume_parts.append(volume_chunk)

        if batch_sleep > 0:
            time.sleep(batch_sleep)

    if not close_parts:
        raise RuntimeError("No price data downloaded from Yahoo Finance.")

    close = pd.concat(close_parts, axis=1)
    volume = pd.concat(volume_parts, axis=1) if volume_parts else pd.DataFrame(index=close.index, columns=close.columns)

    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.loc[:, ~volume.columns.duplicated()].reindex(close.index).sort_index()

    valid_cols = close.columns[close.notna().sum() >= min_price_obs]
    close = close[valid_cols].ffill()
    volume = volume.reindex(columns=valid_cols).fillna(0.0)

    if close.shape[1] == 0:
        raise RuntimeError("No ticker has sufficient cleaned price history.")
    return close, volume

def load_or_download_prices_volume(tickers: list[str], start: str, end: str,
                                  batch_size: int, min_price_obs: int,
                                  batch_sleep: float, batch_retries: int,
                                  cache_dir: Path, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_p = cache_dir / "close.parquet"
    vol_p = cache_dir / "volume.parquet"
    close_k = cache_dir / "close.pkl"
    vol_k = cache_dir / "volume.pkl"
    meta_p = cache_dir / "prices_meta.json"

    meta = {
        "tickers_hash": tickers_hash(tickers),
        "start": start,
        "end": end,
        "min_price_obs": int(min_price_obs),
    }

    if (not refresh) and close_p.exists() and vol_p.exists() and meta_p.exists():
        try:
            cached = json.loads(meta_p.read_text())
            if all(cached.get(k) == v for k, v in meta.items()):
                print("Loading cached prices...")
                cache_format = cached.get("cache_format", "parquet")
                if cache_format == "pickle":
                    return pd.read_pickle(close_k), pd.read_pickle(vol_k)
                return pd.read_parquet(close_p), pd.read_parquet(vol_p)
        except Exception:
            # Fallback to any valid cache format if metadata/parquet load fails
            try:
                if close_k.exists() and vol_k.exists():
                    print("Loading cached prices (pickle fallback)...")
                    return pd.read_pickle(close_k), pd.read_pickle(vol_k)
            except Exception:
                pass

    close, volume = download_prices_volume(tickers, start, end, batch_size, min_price_obs, batch_sleep, batch_retries)

    cache_format = "parquet"
    try:
        close.to_parquet(close_p)
        volume.to_parquet(vol_p)
        print(f"Cached prices to: {close_p}")
    except Exception:
        cache_format = "pickle"
        close.to_pickle(close_k)
        volume.to_pickle(vol_k)
        print(f"Parquet engine unavailable. Cached prices to: {close_k}")

    meta["cache_format"] = cache_format
    meta_p.write_text(json.dumps(meta))
    return close, volume


# ----------------------------
# FUNDAMENTALS (CACHED PER TICKER)
# ----------------------------
class FundamentalStore:
    """
    Goal: approximate NCAV = Current Assets - Total Liabilities (annual)
    and shares outstanding (for market cap).
    Cache each ticker to disk.
    """
    def __init__(
        self,
        cache_dir: Path,
        start: pd.Timestamp,
        end: pd.Timestamp,
        exclude_financials: bool,
        external_ncav: dict[str, pd.Series] | None = None,
        external_shares: dict[str, pd.Series] | None = None,
        external_sector: dict[str, str] | None = None,
        allow_yahoo_fallback: bool = True,
    ):
        self.cache_dir = cache_dir
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.exclude_financials = exclude_financials
        self.allow_yahoo_fallback = allow_yahoo_fallback

        self.ncav_dir = cache_dir / "ncav"
        self.shares_dir = cache_dir / "shares"
        self.meta_dir = cache_dir / "meta"
        self.ncav_dir.mkdir(parents=True, exist_ok=True)
        self.shares_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self._ncav_cache: dict[str, pd.Series] = {}
        self._shares_cache: dict[str, pd.Series] = {}
        self._sector_cache: dict[str, str | None] = {}
        self._external_ncav = external_ncav or {}
        self._external_shares = external_shares or {}
        self._external_sector = external_sector or {}

    def _load_series_csv(self, path: Path) -> pd.Series:
        if not path.exists():
            return pd.Series(dtype=float)
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        return normalize_index_to_naive_utc(s)

    def _save_series_csv(self, s: pd.Series, path: Path, col: str):
        if len(s) == 0:
            return
        pd.DataFrame({col: s}).to_csv(path)

    def sector(self, ticker: str) -> str | None:
        t = normalize_ticker(ticker)
        if t in self._sector_cache:
            return self._sector_cache[t]
        if t in self._external_sector:
            sec = self._external_sector[t]
            self._sector_cache[t] = sec
            return sec

        path = self.meta_dir / f"{t}.json"
        if path.exists():
            try:
                sec = json.loads(path.read_text()).get("sector")
                sec = sec if isinstance(sec, str) else None
                self._sector_cache[t] = sec
                return sec
            except Exception:
                pass

        sec = None
        if not self.allow_yahoo_fallback:
            self._sector_cache[t] = sec
            return sec
        try:
            info = yf.Ticker(t).get_info()
            if isinstance(info, dict):
                v = info.get("sector")
                sec = v if isinstance(v, str) else None
        except Exception:
            sec = None

        try:
            path.write_text(json.dumps({"sector": sec}))
        except Exception:
            pass
        self._sector_cache[t] = sec
        return sec

    def shares_series(self, ticker: str) -> pd.Series:
        t = normalize_ticker(ticker)
        if t in self._shares_cache:
            return self._shares_cache[t]
        if t in self._external_shares:
            s = self._external_shares[t]
            self._shares_cache[t] = s
            return s
        if not self.allow_yahoo_fallback:
            s = pd.Series(dtype=float)
            self._shares_cache[t] = s
            return s

        path = self.shares_dir / f"{t}.csv"
        if path.exists():
            s = self._load_series_csv(path)
            self._shares_cache[t] = s
            return s

        s = pd.Series(dtype=float)
        try:
            tk = yf.Ticker(t)
            raw = tk.get_shares_full(start=self.start.strftime("%Y-%m-%d"), end=self.end.strftime("%Y-%m-%d"))
            if raw is not None and len(raw) > 0:
                s = pd.to_numeric(pd.Series(raw), errors="coerce").dropna().sort_index()
                s = normalize_index_to_naive_utc(s)

            # fallback
            if len(s) == 0:
                info = tk.get_info()
                sh = info.get("sharesOutstanding") if isinstance(info, dict) else None
                if sh is not None and np.isfinite(sh) and sh > 0:
                    s = pd.Series([float(sh)], index=[self.end])
        except Exception:
            s = pd.Series(dtype=float)

        if len(s) > 0:
            self._save_series_csv(s, path, "shares_outstanding")
        self._shares_cache[t] = s
        return s

    def ncav_series(self, ticker: str) -> pd.Series:
        """
        Try to compute NCAV = Current Assets - Total Liabilities
        from annual balance sheet.
        """
        t = normalize_ticker(ticker)
        if t in self._ncav_cache:
            return self._ncav_cache[t]
        if t in self._external_ncav:
            s = self._external_ncav[t]
            self._ncav_cache[t] = s
            return s
        if not self.allow_yahoo_fallback:
            s = pd.Series(dtype=float)
            self._ncav_cache[t] = s
            return s

        path = self.ncav_dir / f"{t}.csv"
        if path.exists():
            s = self._load_series_csv(path)
            self._ncav_cache[t] = s
            return s

        s = pd.Series(dtype=float)
        try:
            tk = yf.Ticker(t)
            bs = tk.get_balance_sheet(freq="yearly")
            if bs is None or bs.empty:
                bs = tk.balance_sheet

            if bs is not None and not bs.empty:
                rows = {str(r).strip().lower(): r for r in bs.index}

                # current assets candidates
                ca_row = (
                    rows.get("current assets") or rows.get("total current assets")
                    or rows.get("currentassets")  # sometimes squashed
                )

                # total liabilities candidates
                tl_row = (
                    rows.get("total liabilities net minority interest")
                    or rows.get("total liabilities")
                    or rows.get("totalliabilitiesnetminorityinterest")
                )

                if ca_row is not None and tl_row is not None:
                    ca = pd.to_numeric(bs.loc[ca_row], errors="coerce")
                    tl = pd.to_numeric(bs.loc[tl_row], errors="coerce")
                    raw = (ca - tl).dropna()
                else:
                    raw = pd.Series(dtype=float)

                if len(raw) > 0:
                    idx = pd.to_datetime(raw.index, errors="coerce", utc=True)
                    mask = ~idx.isna()
                    s = pd.Series(raw.values[mask], index=idx[mask].tz_localize(None)).sort_index()
                    s = pd.to_numeric(s, errors="coerce").dropna()
        except Exception:
            s = pd.Series(dtype=float)

        if len(s) > 0:
            self._save_series_csv(s, path, "ncav")
        self._ncav_cache[t] = s
        return s

    def market_cap_asof(self, ticker: str, dt: pd.Timestamp, px: float) -> float | None:
        if not np.isfinite(px) or px <= 0:
            return None
        sh_s = self.shares_series(ticker)
        sh = _asof_value(sh_s, dt)
        if sh is None or sh <= 0:
            return None
        cap = float(sh * px)
        return cap if np.isfinite(cap) and cap > 0 else None

    def ncav_asof(self, ticker: str, dt: pd.Timestamp) -> float | None:
        s = self.ncav_series(ticker)
        v = _asof_value(s, dt)
        if v is None or not np.isfinite(v):
            return None
        return float(v)


def _is_financial_sector(sector: str | None) -> bool:
    if sector is None:
        return False
    s = sector.strip().lower()
    return ("financial" in s) or ("bank" in s) or ("insurance" in s)


# ----------------------------
# REBALANCE + LIQUIDITY (POINT-IN-TIME)
# ----------------------------
def build_rebalance_days(index: pd.DatetimeIndex, rebalance_month: int) -> list[pd.Timestamp]:
    """
    Match paper-style: rebalance at END of June each year.
    We select the last trading day of rebalance_month.
    """
    days = []
    years = sorted(pd.Index(index.year).unique())
    for y in years:
        mask = (index.year == y) & (index.month == rebalance_month)
        month_days = index[mask]
        if len(month_days) > 0:
            days.append(month_days[-1])  # LAST trading day of June
    return days

def build_liquid_universe_map(close: pd.DataFrame, volume: pd.DataFrame,
                             rebalance_days: list[pd.Timestamp],
                             liquidity_count: int, liquidity_lookback_days: int) -> tuple[dict[pd.Timestamp, list[str]], list[str]]:
    """
    For each rebalance date, compute trailing average dollar volume over lookback window,
    and keep top liquidity_count tickers.
    """
    dollar_vol = (close * volume).replace([np.inf, -np.inf], np.nan)
    liquid_map: dict[pd.Timestamp, list[str]] = {}
    union: set[str] = set()

    min_points = max(20, int(liquidity_lookback_days / 3))

    for reb_dt in rebalance_days[:-1]:
        lookback_start = reb_dt - pd.Timedelta(days=liquidity_lookback_days)
        window = dollar_vol.loc[(dollar_vol.index > lookback_start) & (dollar_vol.index <= reb_dt)]
        if len(window) < min_points:
            window = dollar_vol.loc[dollar_vol.index <= reb_dt].tail(min_points)

        if len(window) == 0:
            liquid = []
        else:
            rank = window.mean().dropna().sort_values(ascending=False)
            liquid = rank.index[:liquidity_count].tolist()

        liquid_map[reb_dt] = liquid
        union.update(liquid)

    return liquid_map, sorted(list(union))


# ----------------------------
# STRATEGY: NCAV / MV TOP-N
# ----------------------------
def run_backtest(close: pd.DataFrame, volume: pd.DataFrame, store: FundamentalStore,
                 rebalance_days: list[pd.Timestamp], liquid_map: dict[pd.Timestamp, list[str]],
                 top_n: int, universe_size: int,
                 fundamental_lag_days: int, cost_bps: float,
                 max_workers: int, exclude_financials: bool) -> tuple[pd.Series, pd.DataFrame]:

    trading_days = close.index
    daily_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    portfolio_ret = pd.Series(0.0, index=trading_days)
    prev_w = pd.Series(0.0, index=close.columns)
    logs = []

    for i, reb_dt in enumerate(rebalance_days[:-1], start=1):
        next_dt = rebalance_days[i]
        liquid = liquid_map.get(reb_dt, [])
        if len(liquid) == 0:
            continue

        px_today = close.loc[reb_dt, liquid].dropna()
        if len(px_today) == 0:
            continue

        # lag fundamentals
        fund_dt = reb_dt - pd.Timedelta(days=fundamental_lag_days)

        print(f"[{i}/{len(rebalance_days)-1}] Rebalance {reb_dt.date()} liquid={len(liquid)} fund_asof={fund_dt.date()}")

        # 1) market cap (parallel)
        caps: dict[str, float] = {}

        def _cap_one(t: str):
            cap = store.market_cap_asof(t, fund_dt, float(px_today[t]))
            return t, cap

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_cap_one, t) for t in px_today.index]
            for fut in as_completed(futures):
                t, cap = fut.result()
                if cap is not None and cap > 0:
                    caps[t] = float(cap)

        if len(caps) == 0:
            continue

        # 2) restrict to top by cap (paper often uses broad universe; we keep it controlled)
        ranked_caps = sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
        top_universe = [t for t, _ in ranked_caps[:universe_size]]

        # 3) compute NCAV/MV and rank
        ratios: list[tuple[str, float]] = []

        for t in top_universe:
            if exclude_financials and _is_financial_sector(store.sector(t)):
                continue
            ncav = store.ncav_asof(t, fund_dt)
            mv = caps.get(t)
            if ncav is None or mv is None or mv <= 0:
                continue
            r = ncav / mv
            if np.isfinite(r):
                ratios.append((t, float(r)))

        # Keep only positive ratios (net-net-ish). If none, hold cash (strict).
        ratios_pos = [(t, r) for t, r in ratios if r > 0]
        ratios_used = ratios_pos if len(ratios_pos) > 0 else []

        ratios_used.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ratios_used[:top_n]]

        new_w = pd.Series(0.0, index=close.columns)
        if len(selected) > 0:
            new_w.loc[selected] = 1.0 / len(selected)

        logs.append({
            "date": reb_dt,
            "fundamental_asof_date": fund_dt.date(),
            "n_liquid": len(liquid),
            "n_cap_ranked": len(caps),
            "n_top_universe": len(top_universe),
            "n_ratios": len(ratios_used),
            "n_selected": len(selected),
            "avg_ratio_selected": float(np.mean([r for _, r in ratios_used[:top_n]])) if len(selected) else np.nan,
        })

        # Apply from next trading day after rebalance date
        start_pos = trading_days.searchsorted(reb_dt, side="right")
        end_pos = trading_days.searchsorted(next_dt, side="right") - 1
        if start_pos <= end_pos and start_pos < len(trading_days):
            period_days = trading_days[start_pos : end_pos + 1]
            period_port = (daily_returns.loc[period_days] * new_w).sum(axis=1)

            # turnover cost on first day
            turnover = float((new_w - prev_w).abs().sum())
            cost = turnover * (cost_bps / 10000.0)
            period_port.iloc[0] -= cost

            portfolio_ret.loc[period_days] = period_port

        prev_w = new_w

    returns = portfolio_ret.loc[(portfolio_ret.index >= rebalance_days[0]) & (portfolio_ret.index <= rebalance_days[-1])].copy()
    return returns, pd.DataFrame(logs)


# ----------------------------
# MAIN
# ----------------------------
def main():
    p = argparse.ArgumentParser("NCAV/MV (Net-Net) top-N yearly strategy — fast cached version")
    p.add_argument("--tickers-file", type=str, required=True, help="One ticker per line (recommended).")
    p.add_argument(
        "--fundamentals-file",
        type=str,
        default=None,
        help="Optional CSV/Parquet with columns: ticker,date,ncav[,shares_outstanding|shares][,sector].",
    )
    p.add_argument(
        "--yahoo-fallback",
        action="store_true",
        help="When --fundamentals-file is provided, still query Yahoo for missing tickers.",
    )
    p.add_argument("--start", type=str, default="2000-01-01")
    p.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"))

    # Strategy parameters
    p.add_argument("--rebalance-month", type=int, default=6, help="June = 6 (rebalance at end of month).")
    p.add_argument("--top-n", type=int, default=100, help="Hold ~100 stocks (Quantpedia style).")
    p.add_argument("--universe-size", type=int, default=3000, help="Top by market cap within the liquid universe.")
    p.add_argument("--fundamental-lag-days", type=int, default=180, help="Lag fundamentals to reduce lookahead.")
    p.add_argument("--cost-bps", type=float, default=10.0, help="Turnover cost in bps (annual rebalance so 5–20 bps is reasonable).")
    p.add_argument("--include-financials", action="store_true")

    # Liquidity filter (point-in-time)
    p.add_argument("--liquidity-count", type=int, default=1500)
    p.add_argument("--liquidity-lookback-days", type=int, default=126)

    # Price download/caching
    p.add_argument("--cache-dir", type=str, default=".cache/ncav_strategy")
    p.add_argument("--refresh-prices", action="store_true")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--batch-sleep", type=float, default=0.0)
    p.add_argument("--batch-retries", type=int, default=1)
    p.add_argument("--min-price-obs", type=int, default=252)

    # Concurrency
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--min-ncav-files", type=int, default=25, help="Minimum tickers with cached NCAV series required.")
    p.add_argument("--min-ncav-coverage", type=float, default=0.02, help="Minimum NCAV coverage over liquid union (0-1).")
    p.add_argument("--allow-empty-selection", action="store_true", help="Allow run to finish even if no stocks are selected.")

    # Outputs
    p.add_argument("--save-equity", type=str, default="ncav_equity.csv")
    p.add_argument("--save-log", type=str, default="ncav_log.csv")
    p.add_argument("--plot-file", type=str, default="ncav_equity.png")
    p.add_argument("--show-plot", action="store_true")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.min_ncav_files < 0:
        raise ValueError("--min-ncav-files must be >= 0")
    if not (0.0 <= args.min_ncav_coverage <= 1.0):
        raise ValueError("--min-ncav-coverage must be between 0 and 1")

    tickers = load_tickers(args.tickers_file)
    print(f"Tickers loaded: {len(tickers)}")

    external_ncav: dict[str, pd.Series] = {}
    external_shares: dict[str, pd.Series] = {}
    external_sector: dict[str, str] = {}
    if args.fundamentals_file:
        external_ncav, external_shares, external_sector = load_external_fundamentals(args.fundamentals_file)
        print(
            f"Loaded external fundamentals: "
            f"ncav={len(external_ncav)} tickers, shares={len(external_shares)} tickers, sector={len(external_sector)} tickers"
        )

    buffer_start = (pd.Timestamp(args.start) - pd.Timedelta(days=550)).strftime("%Y-%m-%d")

    close, volume = load_or_download_prices_volume(
        tickers=tickers,
        start=buffer_start,
        end=args.end,
        batch_size=args.batch_size,
        min_price_obs=args.min_price_obs,
        batch_sleep=args.batch_sleep,
        batch_retries=args.batch_retries,
        cache_dir=cache_dir,
        refresh=args.refresh_prices,
    )
    print(f"Usable tickers after cleaning: {close.shape[1]}")

    # Rebalance at end-of-June each year
    rebalance_days = [d for d in build_rebalance_days(close.index, args.rebalance_month) if d >= pd.Timestamp(args.start)]
    if len(rebalance_days) < 2:
        raise RuntimeError("Not enough rebalance dates found. Try a longer date range.")

    liquid_map, liquid_union = build_liquid_universe_map(
        close=close,
        volume=volume,
        rebalance_days=rebalance_days,
        liquidity_count=args.liquidity_count,
        liquidity_lookback_days=args.liquidity_lookback_days,
    )
    print(f"Unique liquid symbols across rebalances: {len(liquid_union)}")

    # Restrict matrices to possible liquid union (speed)
    close = close[liquid_union]
    volume = volume[liquid_union]

    store = FundamentalStore(
        cache_dir=cache_dir,
        start=pd.Timestamp(buffer_start),
        end=pd.Timestamp(args.end),
        exclude_financials=not args.include_financials,
        external_ncav=external_ncav,
        external_shares=external_shares,
        external_sector=external_sector,
        allow_yahoo_fallback=(args.yahoo_fallback or not args.fundamentals_file),
    )

    # Prefetch fundamentals once (fills cache; future runs faster)
    print("Prefetching fundamentals (shares + NCAV) for liquid union (first run can still take a while)...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(store.shares_series, liquid_union))
        list(ex.map(store.ncav_series, liquid_union))

    ncav_available = sum(1 for t in liquid_union if len(store._ncav_cache.get(normalize_ticker(t), pd.Series(dtype=float))) > 0)
    shares_available = sum(1 for t in liquid_union if len(store._shares_cache.get(normalize_ticker(t), pd.Series(dtype=float))) > 0)
    ncav_coverage = (ncav_available / max(len(liquid_union), 1))
    print(
        f"Fundamental coverage on liquid union: "
        f"shares={shares_available}/{len(liquid_union)}, "
        f"ncav={ncav_available}/{len(liquid_union)} ({ncav_coverage:.2%})"
    )

    if ncav_available < args.min_ncav_files or ncav_coverage < args.min_ncav_coverage:
        raise RuntimeError(
            "Insufficient NCAV data coverage for strict strategy. "
            "Yahoo is not returning enough balance-sheet rows. "
            "Use a different fundamentals source or lower --min-ncav-files / --min-ncav-coverage."
        )

    returns, log = run_backtest(
        close=close,
        volume=volume,
        store=store,
        rebalance_days=rebalance_days,
        liquid_map=liquid_map,
        top_n=args.top_n,
        universe_size=args.universe_size,
        fundamental_lag_days=args.fundamental_lag_days,
        cost_bps=args.cost_bps,
        max_workers=args.max_workers,
        exclude_financials=not args.include_financials,
    )

    if len(log) == 0 or int(log["n_selected"].sum()) == 0:
        msg = (
            "No stocks were selected under strict NCAV rules. "
            "This usually means fundamentals coverage is still too sparse for the chosen universe/date range."
        )
        if args.allow_empty_selection:
            print(f"Warning: {msg}")
        else:
            raise RuntimeError(msg + " Re-run with --allow-empty-selection to force output files.")

    # ---- Benchmark: SPY total-return (daily, matched to backtest period) ----
    spy_rets: pd.Series | None = None
    try:
        import yfinance as _yf
        spy_raw = _yf.download(
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

    all_report: dict[str, dict] = {"NCAV/MV Net-Net": stats}
    if spy_rets is not None:
        from performance1 import compute_perf_stats
        all_report["SPY (B&H)"] = compute_perf_stats(
            spy_rets.reindex(returns.index).dropna(), freq=252
        )

    from performance1 import print_stats_table, plot_tearsheet

    print_stats_table(
        all_report,
        title=f"NCAV/MV Net-Net Strategy  "
              f"({returns.index[0].date()} → {returns.index[-1].date()})",
    )

    if len(log) > 0:
        print(f"  Average selected names/rebalance : {log['n_selected'].mean():.1f}")
        print(f"  Median ratios computed/rebalance : {log['n_ratios'].median():.0f}")

    print(
        "\nNote on survivorship bias: this backtest uses the ticker universe "
        "provided at run time and does not include companies that were delisted, "
        "went bankrupt, or were acquired during the sample period. "
        "Live returns will differ; treat results as an upper bound."
    )

    equity = (1 + returns).cumprod()
    dd = drawdown_series(returns)

    out = pd.DataFrame({"equity": equity, "returns": returns, "drawdown": dd})
    out.to_csv(args.save_equity)
    log.to_csv(args.save_log, index=False)
    print(f"\nSaved equity/returns/drawdown to: {args.save_equity}")
    print(f"Saved rebalance log to: {args.save_log}")

    plot_tearsheet(
        returns=returns,
        benchmark_returns=spy_rets,
        title="NCAV/MV (Net-Net) Top-N Strategy",
        freq=252,
        rolling_window=252,
        save_path=args.plot_file,
        show=args.show_plot,
    )


if __name__ == "__main__":
    main()
