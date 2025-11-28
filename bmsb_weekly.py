#!/usr/bin/env python3
# bmsb_weekly.py
# Standalone script to fetch weekly BTC/USDT candles from Binance,
# compute 50-week MA and Bull Market Support Band (20W SMA & 21W EMA),
# and save a chart to PNG.

import io
import os
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def fetch_ohlcv_binance(symbol="BTC/USDT", timeframe="1w", since_days=4000, limit=1000) -> pd.DataFrame:
    """
    Pull OHLCV from Binance via ccxt and return a UTC-indexed DataFrame
    with columns: Open, High, Low, Close, Volume.
    since_days: how far back to start (in days). 4000 ~ 10+ years buffer.
    """
    ex = ccxt.binance({"enableRateLimit": True})
    now = datetime.now(timezone.utc)
    since_ms = int((now - timedelta(days=since_days)).timestamp() * 1000)

    all_rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        # Stop if we’re basically “caught up” to the latest candle
        if last_ts >= int(now.timestamp() * 1000) - 60_000:
            break
        since_ms = last_ts + 1

    if not all_rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(all_rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    # Ensure float dtype
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def compute_indicators(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add 50-week SMA and Bull Market Support Band:
    - 50W MA        : 50-week Simple Moving Average of Close
    - BMSB upper    : 20-week SMA of Close
    - BMSB lower    : 21-week EMA of Close
    """
    df = weekly.copy()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=50).mean()

    # Bull Market Support Band (classic definition)
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["BMSB_upper"] = df["SMA20"]
    df["BMSB_lower"] = df["EMA21"]
    return df


def plot_bmsb(df: pd.DataFrame, out_path="btc_bmsb_weekly.png", title_prefix="BTC/USDT — Weekly"):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Price
    ax.plot(df.index, df["Close"], label="Close", lw=1.6, color="#87e5ff")

    # 50-week MA
    if df["MA50"].notna().any():
        ax.plot(df.index, df["MA50"], label="50W MA (SMA)", lw=1.2, linestyle="--", color="#ffffff")

    # BMSB band (20W SMA & 21W EMA)
    have_band = df["BMSB_upper"].notna().any() and df["BMSB_lower"].notna().any()
    if have_band:
        ax.plot(df.index, df["BMSB_upper"], label="BMSB Upper (20W SMA)", lw=1.0, color="#f3b76b")
        ax.plot(df.index, df["BMSB_lower"], label="BMSB Lower (21W EMA)", lw=1.0, color="#f19b2c")
        # Shade between
        ax.fill_between(df.index, df["BMSB_lower"], df["BMSB_upper"], alpha=0.18, color="#ffa726")

    # Formatting
    ax.set_title(f"{title_prefix} — 50W MA & Bull Market Support Band", fontsize=13, weight="bold")
    ax.set_ylabel("Price (USDT)"); ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.legend(loc="best", fontsize=9, framealpha=0.25)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Plot BTC Weekly 50W MA & Bull Market Support Band from Binance data.")
    p.add_argument("--symbol", default="BTC/USDT", help="Market symbol, e.g., BTC/USDT")
    p.add_argument("--timeframe", default="1w", help="Binance timeframe (default: 1w)")
    p.add_argument("--since_days", type=int, default=4000, help="How many days back to fetch (default: 4000)")
    p.add_argument("--out", default="btc_bmsb_weekly.png", help="Output PNG path")
    args = p.parse_args()

    print(f"[INFO] Fetching {args.symbol} {args.timeframe} from Binance…")
    weekly = fetch_ohlcv_binance(args.symbol, timeframe=args.timeframe, since_days=args.since_days)
    if weekly.empty:
        print("[ERROR] No data returned. Check symbol/timeframe/connection.")
        return

    print(f"[INFO] {len(weekly)} candles fetched from {weekly.index.min()} to {weekly.index.max()}")

    df = compute_indicators(weekly)

    # Print latest values
    last = df.dropna().iloc[-1]
    close = float(last["Close"])
    ma50 = float(last["MA50"]) if not pd.isna(last["MA50"]) else None
    sma20 = float(last["SMA20"]) if not pd.isna(last["SMA20"]) else None
    ema21 = float(last["EMA21"]) if not pd.isna(last["EMA21"]) else None

    print("\n=== Latest Weekly Snapshot ===")
    print(f"Close : {close:,.2f} USDT")
    if ma50 is not None:
        print(f"50W MA (SMA): {ma50:,.2f} USDT")
    if sma20 is not None and ema21 is not None:
        upper = sma20
        lower = ema21
        pos = "above band" if close > upper else ("within band" if lower <= close <= upper else "below band")
        print(f"BMSB Upper (20W SMA): {upper:,.2f} USDT")
        print(f"BMSB Lower (21W EMA): {lower:,.2f} USDT")
        print(f"Position vs BMSB: {pos}")

    out_path = plot_bmsb(df, out_path=args.out, title_prefix=args.symbol)
    print(f"\n[OK] Chart saved → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
