import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def load_tickers(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    tickers = [normalize_ticker(x) for x in p.read_text().splitlines() if x.strip()]
    tickers = sorted(list(dict.fromkeys(tickers)))
    if not tickers:
        raise ValueError("Ticker file is empty.")
    return tickers


def to_naive_index(idx) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx, errors="coerce", utc=True)
    return dt[~dt.isna()].tz_localize(None)


def parse_ncav_from_balance_sheet(bs: pd.DataFrame) -> pd.Series:
    if bs is None or bs.empty:
        return pd.Series(dtype=float)

    rows = {str(r).strip().lower(): r for r in bs.index}
    ca_row = rows.get("current assets") or rows.get("total current assets") or rows.get("currentassets")
    tl_row = (
        rows.get("total liabilities net minority interest")
        or rows.get("total liabilities")
        or rows.get("totalliabilitiesnetminorityinterest")
    )
    if ca_row is None or tl_row is None:
        return pd.Series(dtype=float)

    ca = pd.to_numeric(bs.loc[ca_row], errors="coerce")
    tl = pd.to_numeric(bs.loc[tl_row], errors="coerce")
    raw = (ca - tl).dropna()
    if len(raw) == 0:
        return pd.Series(dtype=float)

    idx = to_naive_index(raw.index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    vals = raw.loc[~pd.to_datetime(raw.index, errors="coerce", utc=True).isna()].values
    s = pd.Series(vals, index=idx).sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def get_with_retries(fn, retries: int, sleep_s: float):
    for i in range(retries + 1):
        try:
            return fn()
        except Exception:
            if i < retries:
                time.sleep(sleep_s * (2 ** i))
    return None


def asof_shares(shares_series: pd.Series, dt: pd.Timestamp, fallback_shares: float | None) -> float | None:
    if len(shares_series) > 0:
        s = shares_series.loc[:dt]
        if len(s) > 0:
            v = float(s.iloc[-1])
            if np.isfinite(v) and v > 0:
                return v
    if fallback_shares is not None and np.isfinite(fallback_shares) and fallback_shares > 0:
        return float(fallback_shares)
    return None


def build_one_ticker_rows(ticker: str, retries: int, sleep_s: float) -> list[dict]:
    tk = yf.Ticker(ticker)

    bs = get_with_retries(lambda: tk.get_balance_sheet(freq="yearly"), retries, sleep_s)
    if bs is None or bs.empty:
        bs = get_with_retries(lambda: tk.balance_sheet, retries, sleep_s)
    ncav = parse_ncav_from_balance_sheet(bs if bs is not None else pd.DataFrame())
    if len(ncav) == 0:
        return []

    min_dt = (ncav.index.min() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    max_dt = (pd.Timestamp.today() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    raw_shares = get_with_retries(lambda: tk.get_shares_full(start=min_dt, end=max_dt), retries, sleep_s)
    if raw_shares is None or len(raw_shares) == 0:
        shares_series = pd.Series(dtype=float)
    else:
        shares_series = pd.to_numeric(pd.Series(raw_shares), errors="coerce").dropna().sort_index()
        shares_series.index = to_naive_index(shares_series.index)

    info = get_with_retries(lambda: tk.get_info(), retries, sleep_s)
    if isinstance(info, dict):
        sector = info.get("sector")
        sector = sector if isinstance(sector, str) else None
        info_shares = info.get("sharesOutstanding")
        info_shares = float(info_shares) if (info_shares is not None and np.isfinite(info_shares)) else None
    else:
        sector = None
        info_shares = None

    rows = []
    for dt, v in ncav.items():
        sh = asof_shares(shares_series, pd.Timestamp(dt), info_shares)
        if sh is None:
            continue
        rows.append(
            {
                "ticker": ticker,
                "date": pd.Timestamp(dt).date().isoformat(),
                "ncav": float(v),
                "shares_outstanding": float(sh),
                "sector": sector if sector is not None else "",
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser("Build external fundamentals CSV for ncav_effect_terminal.py")
    p.add_argument("--tickers-file", type=str, required=True)
    p.add_argument("--output", type=str, default="fundamentals.csv")
    p.add_argument("--max-tickers", type=int, default=0, help="0 = all")
    p.add_argument("--sleep", type=float, default=0.25, help="Sleep between tickers (seconds)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--save-every", type=int, default=25, help="Persist progress every N processed tickers")
    p.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = p.parse_args()

    tickers = load_tickers(args.tickers_file)
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    out_path = Path(args.output)
    all_rows: list[dict] = []
    done = set()

    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if len(old) > 0 and "ticker" in old.columns:
            old["ticker"] = old["ticker"].astype(str).map(normalize_ticker)
            done = set(old["ticker"].unique().tolist())
            all_rows = old.to_dict("records")
            print(f"Resuming from {out_path}: {len(done)} tickers already present, {len(all_rows)} rows.")

    total = len(tickers)
    processed = 0
    added_tickers = 0

    for i, t in enumerate(tickers, start=1):
        if t in done:
            continue
        processed += 1
        try:
            rows = build_one_ticker_rows(t, retries=args.retries, sleep_s=max(args.sleep, 0.05))
        except Exception:
            rows = []

        if rows:
            all_rows.extend(rows)
            added_tickers += 1
            done.add(t)

        if i % args.save_every == 0:
            if all_rows:
                pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"[{i}/{total}] processed={processed} added_tickers={added_tickers} rows={len(all_rows)}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
    else:
        pd.DataFrame(columns=["ticker", "date", "ncav", "shares_outstanding", "sector"]).to_csv(out_path, index=False)

    print(f"Saved: {out_path.resolve()} rows={len(all_rows)} unique_tickers={len(done)}")


if __name__ == "__main__":
    main()
