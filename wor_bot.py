# WOR Telegram Bot (Weekly Opening Range)
# ---------------------------------------------------------------
# Drop-in Python script built on telebot + yfinance + APScheduler.
# Adapts the forecast/alerts/chart scaffold to WOR logic.
#
# Quick start:
#   1) pip install pyTelegramBotAPI yfinance apscheduler matplotlib pandas numpy python-dateutil
#   2) Put your Telegram Bot token below.
#   3) python wor_bot.py
#
# Commands:
#   /wor set <SYMBOL>                # e.g. BTC-USD or AAPL
#   /wor mode <first4h|mondayclose>  # WOR window definition
#   /wor range                       # Show this week’s OR high/low, width, mid
#   /wor signal                      # Status: inside OR / broke up/down, extension
#   /wor chart [7|14|30|90]          # Chart with WOR bands & breakout markers (days)
#   /wor alert on                    # Enable alerts (first break, extensions)
#   /wor alert off                   # Disable alerts
#   /wor alert level <list>          # e.g. /wor alert level 0.5 1.0 1.5
#   /wor help                        # Help text
#
# Notes:
# - Default timeframe uses 1h data from Yahoo (works for crypto & many tickers).
# - You can later swap yfinance with ccxt for exchange-tight crypto feeds.
# - Minimal state persistence via JSON file per chat_id.

import os
import io
import json
import time
import math
import telebot
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from dateutil import tz

# ============================ CONFIG ============================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
DATA_TIMEFRAME = "1h"          # yfinance interval mapping (we’ll resample when needed)
CHECK_INTERVAL_MIN = 5          # scheduler cadence for alert checks
STATE_FILE = "wor_state.json"   # simple persistence
DEFAULT_SYMBOL = "BTC-USD"
DEFAULT_MODE = "first4h"        # or "mondayclose"
DEFAULT_EXT_LEVELS = [0.5, 1.0, 1.5]

# ============================ STATE ============================
# Persist small per-chat settings: symbol, mode, alert_on, ext_levels, last_fired keys
_state = {}

def load_state():
    global _state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                _state = json.load(f)
        except Exception:
            _state = {}
    else:
        _state = {}

def save_state():
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_state, f)
    os.replace(tmp, STATE_FILE)


def get_chat_cfg(chat_id):
    sid = str(chat_id)
    if sid not in _state:
        _state[sid] = {
            "symbol": DEFAULT_SYMBOL,
            "mode": DEFAULT_MODE,
            "alert_on": False,
            "ext_levels": DEFAULT_EXT_LEVELS,
            "last_week_alerts": {},   # { week_str: {"break":"UP/DOWN/None", "ext_hit": [0.5,..] } }
        }
    return _state[sid]

# ============================ DATA ============================
# yfinance helper; returns OHLCV with UTC index (pd.DatetimeIndex tz-aware)

def fetch_ohlcv(symbol, period_days=90, interval="1h"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=period_days)
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Ensure tz-aware index in UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)
    # Standardize columns
    cols = {"Open":"Open", "High":"High", "Low":"Low", "Close":"Close", "Volume":"Volume"}
    df = df.rename(columns=cols)[["Open","High","Low","Close","Volume"]]
    return df

# ============================ WOR LOGIC ============================

def start_of_week(ts):
    # Align to Monday 00:00 UTC
    dt = pd.to_datetime(ts).tz_convert(timezone.utc)
    dow = dt.weekday()  # Monday=0
    floor = (dt - timedelta(days=dow)).replace(hour=0, minute=0, second=0, microsecond=0)
    return floor


def compute_wor(df, mode="first4h"):
    """
    df: OHLCV at 1h or finer (UTC tz-aware index)
    mode: "first4h" uses Monday 00:00–04:00 window (crypto-friendly)
          "mondayclose" uses Monday daily high/low
    returns: wor DataFrame indexed by week_start with OR_high/OR_low/OR_mid/OR_width
    """
    if df.empty:
        return pd.DataFrame(columns=["OR_high","OR_low","OR_mid","OR_width"]).astype(float)

    df = df.copy()
    df["week"] = df.index.map(start_of_week)

    rows = []
    for wk, g in df.groupby("week"):
        monday = g[g.index.weekday == 0]
        if monday.empty:
            continue
        if mode == "first4h":
            m0 = monday.index.min()
            window = monday[(monday.index >= m0) & (monday.index < m0 + timedelta(hours=4))]
            if window.empty:
                continue
            or_high, or_low = window["High"].max(), window["Low"].min()
        else:  # mondayclose
            day = monday.resample("1D").agg({"High":"max","Low":"min"})
            if day.empty:
                continue
            or_high, or_low = float(day["High"].iloc[0]), float(day["Low"].iloc[0])

        rows.append({
            "week": wk,
            "OR_high": float(or_high),
            "OR_low": float(or_low),
            "OR_mid": float((or_high + or_low)/2.0),
            "OR_width": float(or_high - or_low),
        })

    wor = pd.DataFrame(rows).set_index("week").sort_index()
    return wor


def latest_week_wor(df, mode):
    wor = compute_wor(df, mode)
    if wor.empty:
        return None, None
    wk = wor.index.max()
    return wk, wor.loc[wk]


def breakout_and_extensions(df, week_start, or_high, or_low, ext_levels):
    """Return dict with breakout direction, first_break_time, and which ext_levels have been hit.
    df: hourly (or finer) dataframe, UTC indexed
    """
    g = df[(df.index >= week_start) & (df.index < week_start + timedelta(days=7))]
    if g.empty:
        return {"break": None, "first_time": None, "hits": []}

    broke_up = g["Close"] > or_high
    broke_dn = g["Close"] < or_low

    up_time = broke_up.idxmax() if broke_up.any() else None
    dn_time = broke_dn.idxmax() if broke_dn.any() else None

    direction = None
    first_time = None
    base = None
    if up_time and (not dn_time or up_time < dn_time):
        direction = "UP"; first_time = up_time; base = or_high
    elif dn_time:
        direction = "DOWN"; first_time = dn_time; base = or_low

    hits = []
    if direction:
        sub = g[g.index >= first_time]
        width = or_high - or_low
        if width <= 0:
            return {"break": direction, "first_time": first_time, "hits": hits}
        for k in ext_levels:
            if direction == "UP":
                tgt = base + k*width
                hit = (sub["High"] >= tgt).any()
            else:
                tgt = base - k*width
                hit = (sub["Low"] <= tgt).any()
            if hit:
                hits.append(k)
    return {"break": direction, "first_time": first_time, "hits": hits}

# ============================ CHARTING ============================

def plot_wor_chart(df, wk, or_row, symbol):
    """Return bytes of PNG chart with price + WOR bands."""
    hi, lo, mid = or_row["OR_high"], or_row["OR_low"], or_row["OR_mid"]
    week_df = df[(df.index >= wk) & (df.index < wk + timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(week_df.index, week_df["Close"], label=f"{symbol} Close")

    ax.axhline(hi, linestyle="--", linewidth=1.2, label="OR High")
    ax.axhline(lo, linestyle="--", linewidth=1.2, label="OR Low")
    ax.axhline(mid, linestyle=":", linewidth=1.0, label="OR Mid")

    # Shade OR window (first 4h of Monday) for visual context
    mon = week_df[week_df.index.weekday==0]
    if not mon.empty:
        m0 = mon.index.min()
        ax.axvspan(m0, m0 + timedelta(hours=4), alpha=0.1)

    ax.set_title(f"{symbol} — Weekly Opening Range")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend(loc="best")
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ============================ TELEGRAM BOT ============================
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
load_state()

HELP_TEXT = (
"/wor set <SYMBOL>\n"
"/wor mode <first4h|mondayclose>\n"
"/wor range\n"
"/wor signal\n"
"/wor chart [7|14|30|90]\n"
"/wor alert on | off\n"
"/wor alert level <k1 k2 ...> (e.g. 0.5 1.0 1.5)\n"
"/wor help\n"
)

@bot.message_handler(commands=["wor"])
def wor_router(message):
    args = message.text.split()[1:]
    chat_id = message.chat.id
    cfg = get_chat_cfg(chat_id)

    if not args:
        bot.reply_to(message, HELP_TEXT)
        return

    sub = args[0].lower()

    if sub == "help":
        bot.reply_to(message, HELP_TEXT)
        return

    if sub == "set":
        if len(args) < 2:
            bot.reply_to(message, "Usage: /wor set <SYMBOL>")
            return
        cfg["symbol"] = args[1].strip().upper()
        save_state()
        bot.reply_to(message, f"Symbol set to {cfg['symbol']}")
        return

    if sub == "mode":
        if len(args) < 2 or args[1].lower() not in ("first4h","mondayclose"):
            bot.reply_to(message, "Usage: /wor mode <first4h|mondayclose>")
            return
        cfg["mode"] = args[1].lower()
        save_state()
        bot.reply_to(message, f"Mode set to {cfg['mode']}")
        return

    if sub == "range":
        df = fetch_ohlcv(cfg["symbol"], period_days=30, interval=DATA_TIMEFRAME)
        wk, row = latest_week_wor(df, cfg["mode"])
        if row is None:
            bot.reply_to(message, "No data to compute WOR.")
            return
        msg = (
            f"WOR for week starting {wk.strftime('%Y-%m-%d')} (UTC)\n"
            f"OR High: {row['OR_high']:.2f}\nOR Low: {row['OR_low']:.2f}\n"
            f"Width: {row['OR_width']:.2f} ({(row['OR_width']/row['OR_mid']*100):.2f}%)\n"
            f"Mid: {row['OR_mid']:.2f}"
        )
        bot.reply_to(message, msg)
        return

    if sub == "signal":
        df = fetch_ohlcv(cfg["symbol"], period_days=30, interval=DATA_TIMEFRAME)
        wk, row = latest_week_wor(df, cfg["mode"])
        if row is None:
            bot.reply_to(message, "No data to compute signals.")
            return
        sig = breakout_and_extensions(df, wk, row['OR_high'], row['OR_low'], cfg['ext_levels'])
        if not sig["break"]:
            bot.reply_to(message, "No breakout yet — price inside OR.")
            return
        ext_str = ", ".join([str(k) for k in sig["hits"]]) if sig["hits"] else "None"
        tstr = sig["first_time"].strftime('%Y-%m-%d %H:%M UTC') if sig["first_time"] else "—"
        bot.reply_to(message, f"Break: {sig['break']} at {tstr}\nExtensions hit: {ext_str}")
        return

    if sub == "chart":
        days = 30
        if len(args) >= 2 and args[1] in ("7","14","30","90"):
            days = int(args[1])
        df = fetch_ohlcv(cfg["symbol"], period_days=days, interval=DATA_TIMEFRAME)
        wk, row = latest_week_wor(df, cfg["mode"])
        if row is None:
            bot.reply_to(message, "No data to chart.")
            return
        png = plot_wor_chart(df, wk, row, cfg["symbol"])
        bot.send_photo(chat_id, png)
        return

    if sub == "alert":
        if len(args) < 2:
            bot.reply_to(message, "Usage: /wor alert on|off|level <...>")
            return
        opt = args[1].lower()
        if opt == "on":
            cfg["alert_on"] = True
            save_state()
            bot.reply_to(message, "WOR alerts: ON")
            return
        if opt == "off":
            cfg["alert_on"] = False
            save_state()
            bot.reply_to(message, "WOR alerts: OFF")
            return
        if opt == "level":
            if len(args) < 3:
                bot.reply_to(message, "Usage: /wor alert level <k1 k2 ...>")
                return
            try:
                levels = [float(x) for x in args[2:]]
                levels = [x for x in levels if x > 0]
                if not levels:
                    raise ValueError
                cfg["ext_levels"] = levels
                save_state()
                bot.reply_to(message, f"Extension levels set to {levels}")
            except Exception:
                bot.reply_to(message, "Provide numeric positive levels, e.g. 0.5 1.0 1.5")
            return

    bot.reply_to(message, "Unknown /wor subcommand. Try /wor help")

# ============================ ALERT LOOP ============================

def check_alerts():
    for sid, cfg in list(_state.items()):
        if not cfg.get("alert_on"):
            continue
        chat_id = int(sid)
        try:
            df = fetch_ohlcv(cfg["symbol"], period_days=30, interval=DATA_TIMEFRAME)
            wk, row = latest_week_wor(df, cfg["mode"])
            if row is None:
                continue
            sig = breakout_and_extensions(df, wk, row['OR_high'], row['OR_low'], cfg['ext_levels'])
            week_key = wk.strftime('%Y-%m-%d')
            hist = cfg.setdefault("last_week_alerts", {}).setdefault(week_key, {"break": None, "ext_hit": []})

            # Fire once per week per condition
            if sig["break"] and hist.get("break") is None:
                tstr = sig["first_time"].strftime('%Y-%m-%d %H:%M UTC') if sig["first_time"] else "—"
                bot.send_message(chat_id, f"WOR Break {sig['break']} — {cfg['symbol']} at {tstr}")
                hist["break"] = sig["break"]

            for k in sig["hits"]:
                if k not in hist["ext_hit"]:
                    bot.send_message(chat_id, f"WOR {sig['break']} extension hit: {k}x range — {cfg['symbol']}")
                    hist["ext_hit"].append(k)

            save_state()
        except Exception as e:
            # Be quiet but safe; avoid spamming errors in chat
            print("check_alerts error:", e)

# ============================ MAIN ============================
if __name__ == "__main__":
    if TELEGRAM_BOT_TOKEN == "PUT_YOUR_BOT_TOKEN_HERE":
        print("[WARN] Please set TELEGRAM_BOT_TOKEN env var or replace the placeholder.")
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_alerts, 'interval', minutes=CHECK_INTERVAL_MIN)
    scheduler.start()
    print("WOR bot running…")
    bot.polling(none_stop=True)
