"""
Ticker Fetcher & Liquidity Universe
===================================

Fetch S&P 500 + NASDAQ-100 tickers from Wikipedia and optionally build a
liquidity-ranked universe (by ADV = Close * Volume).

Usage (module):
    from tickerfetch import (
        fetch_sp500_tickers, fetch_nasdaq100_tickers,
        get_combined_universe, build_liquidity_universe,
        save_tickers, load_tickers
    )

CLI:
    python tickerfetch.py                # fetch & save combined universe
    python tickerfetch.py --liquidity    # fetch + liquidity universe
    python tickerfetch.py --liquidity --top 250
"""

from __future__ import annotations

import os
import time
import argparse
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = PROJECT_ROOT / "ticker"          # save under "ticker/"
UNIVERSE_DIR = DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------------------
def fetch_sp500_tickers() -> List[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        raw = df["Symbol"].astype(str).str.strip().tolist()
        clean = [t.replace(".", "-") for t in raw]
        print(f"✅ Fetched {len(clean)} S&P 500 tickers")
        return clean
    except Exception as e:
        print(f"❌ Error fetching S&P 500 tickers: {e}")
        return []


def fetch_nasdaq100_tickers() -> List[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        time.sleep(1)
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        for i, df in enumerate(tables):
            lower = [str(c).lower() for c in df.columns]
            if "ticker" in lower or "symbol" in lower:
                col = df.columns[lower.index("ticker")] if "ticker" in lower else df.columns[lower.index("symbol")]
                raw = df[col].astype(str).str.strip().tolist()
                tickers = [t.replace(".", "-") for t in raw]
                valid = [t for t in tickers if t and t != "nan" and len(t) <= 6]
                if len(valid) >= 80:
                    print(f"✅ Using NASDAQ-100 table #{i} with column '{col}' ({len(valid)} tickers)")
                    return valid

        raise ValueError("Could not find NASDAQ-100 ticker column")
    except Exception as e:
        print(f"❌ Error fetching NASDAQ-100 tickers: {e}")
        return []

# --------------------------------------------------------------------------------------
# Combine universe
# --------------------------------------------------------------------------------------
def get_combined_universe(
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    save_components: bool = False,
) -> pd.DataFrame:
    all_tickers, sources = [], []

    if include_sp500:
        spx = fetch_sp500_tickers()
        all_tickers.extend(spx)
        sources.extend(["S&P 500"] * len(spx))

    if include_nasdaq100:
        ndx = fetch_nasdaq100_tickers()
        all_tickers.extend(ndx)
        sources.extend(["NASDAQ 100"] * len(ndx))

    if not all_tickers:
        return pd.DataFrame(columns=["Ticker", "Source", "In_Both", "Fetched_Date", "Total_Count"])

    df = pd.DataFrame({"Ticker": all_tickers, "Source": sources})
    counts = df["Ticker"].value_counts()
    df["In_Both"] = df["Ticker"].map(counts > 1)

    uni = (
        df.groupby("Ticker")
        .agg({"Source": lambda x: " + ".join(sorted(set(x))), "In_Both": "first"})
        .reset_index()
    )
    uni["Fetched_Date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    uni["Total_Count"] = len(uni)

    print("\n📊 Combined Universe Summary:")
    print(f"  Total unique tickers: {len(uni)}")
    print(f"  S&P 500 only: {len(uni[uni['Source'] == 'S&P 500'])}")
    print(f"  NASDAQ 100 only: {len(uni[uni['Source'] == 'NASDAQ 100'])}")
    print(f"  In both indices: {len(uni[uni['In_Both']])}")

    if save_components:
        analysis = {
            "unique_tickers": len(uni),
            "overlap_count": int(uni["In_Both"].sum()),
            "fetch_date": datetime.now(timezone.utc).isoformat(),
        }
        pd.Series(analysis).to_csv(DATA_DIR / "ticker_analysis.csv")
        print(f"💾 Saved analysis to {DATA_DIR / 'ticker_analysis.csv'}")

    return uni

# --------------------------------------------------------------------------------------
# Liquidity ranking
# --------------------------------------------------------------------------------------
def rank_by_liquidity(
    tickers: List[str],
    lookback_days: int = 30,
    min_price: float = 5.0,
    min_adv_usd: float = 10_000_000.0,
    period_pad_days: int = 20,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "avg_traded_value", "last_close"])

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days + period_pad_days)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    rows = []
    for t in tickers:
        try:
            df = raw[t].dropna().tail(lookback_days)
            if df.empty:
                continue
            last_close = float(df["Close"].iloc[-1])
            if last_close < min_price:
                continue
            adv = float((df["Close"] * df["Volume"]).mean())
            if adv < min_adv_usd:
                continue
            rows.append({"Ticker": t, "avg_traded_value": adv, "last_close": last_close})
        except Exception:
            pass

    out = pd.DataFrame(rows)
    return out.sort_values("avg_traded_value", ascending=False).reset_index(drop=True)


def build_liquidity_universe(
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    top_n: int = 200,
    lookback_days: int = 30,
    min_price: float = 5.0,
    min_adv_usd: float = 10_000_000.0,
    save_to_dir: str | None = str(UNIVERSE_DIR),
    save_tag: str = "spx_ndx",
) -> pd.DataFrame:
    uni = get_combined_universe(include_sp500=include_sp500,
                                include_nasdaq100=include_nasdaq100,
                                save_components=False)
    tickers = [str(t).replace(".", "-") for t in uni["Ticker"].tolist()]
    ranked = rank_by_liquidity(tickers, lookback_days, min_price, min_adv_usd)
    if ranked.empty:
        print("⚠️ No tickers passed liquidity filters.")
        return ranked

    top = ranked.head(top_n).copy()

    if save_to_dir:
        os.makedirs(save_to_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        snap = os.path.join(save_to_dir, f"{save_tag}_liq_top{top_n}_{stamp}.csv")
        latest = os.path.join(save_to_dir, f"{save_tag}_liq_top{top_n}_latest.csv")
        top.to_csv(snap, index=False)
        top.to_csv(latest, index=False)
        print(f"💾 Saved: {snap}\n💾 Saved: {latest}")

    return top

# --------------------------------------------------------------------------------------
# Save/load
# --------------------------------------------------------------------------------------
def save_tickers(ticker_df: pd.DataFrame, filepath: str | None = None) -> str:
    if filepath is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        filepath = str(DATA_DIR / f"tickers_{ts}.csv")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    ticker_df.to_csv(filepath, index=False)
    print(f"💾 Saved {len(ticker_df)} tickers to {filepath}")
    return filepath


def load_tickers(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        print(f"📂 Loaded {len(df)} tickers from {filepath}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ticker universes and build liquidity rankings")
    parser.add_argument("--liquidity", action="store_true", help="Also build liquidity-ranked universe")
    parser.add_argument("--top", type=int, default=200, help="Number of top tickers to keep in liquidity universe")
    args = parser.parse_args()

    uni = get_combined_universe(include_sp500=True, include_nasdaq100=True, save_components=True)
    if not uni.empty:
        save_tickers(uni, filepath=str(DATA_DIR / "tickers_combined_latest.csv"))
        print("\n🔎 Sample tickers:")
        print(uni[["Ticker", "Source", "In_Both"]].head(10))

        if args.liquidity:
            topn = build_liquidity_universe(include_sp500=True, include_nasdaq100=True, top_n=args.top)
            if not topn.empty:
                print(f"\n📊 Top {args.top} by ADV:")
                print(topn.head(10))




# # just fetch tickers
# python tickerfetch.py

# # fetch + build liquidity universe (default top 200)
# python tickerfetch.py --liquidity

# # fetch + build liquidity universe with top 250
# python tickerfetch.py --liquidity --top 250
