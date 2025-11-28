# WOR Telegram Bot — BTC-only, Monday Close mode, Binance data, journal-enhanced
# Cleaned & fixed: Sheets integration, /enter /exit /flow, /finalize, /push twr

import os, io, json, re, csv, base64
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText

import ccxt
import telebot
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# ============================ CONFIG ============================

# -- load .env once, before reading any vars --
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)
print("[DEBUG] .env loaded from:", ENV_PATH)

def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    raw = raw.split("#", 1)[0].strip()
    try:
        return float(raw)
    except Exception:
        return default

def _mask(t: str) -> str:
    if not t:
        return "<EMPTY>"
    return f"{t[:6]}...{t[-6:]} (len={len(t)})"

# === Cycle definitions & formatter ==========================================
from datetime import date, datetime, timedelta

# Label cycles by their *halving year*
CYCLE_DEFS = {
    2012: {
        "label": "2012-2016",
        "halving": date(2012, 11, 28),
        "peak": date(2013, 11, 30),     # major cycle peak (Nov 2013)
        "bottom": date(2015, 1, 14),    # cycle bottom (Jan 2015)
        "next_halving": date(2016, 7, 9),
        "next_peak": date(2017, 12, 17),
        "next_bottom": date(2018, 12, 15),
        "projected": False,
    },
    2016: {
        "label": "2016-2020",
        "halving": date(2016, 7, 9),
        "peak": date(2017, 12, 17),
        "bottom": date(2018, 12, 15),
        "next_halving": date(2020, 5, 11),
        "next_peak": date(2021, 11, 10),
        "next_bottom": date(2022, 11, 21),
        "projected": False,
    },
    2020: {
        "label": "2020-2024",
        "halving": date(2020, 5, 11),
        "peak": date(2021, 11, 10),
        "bottom": date(2022, 11, 21),
        "next_halving": date(2024, 4, 19),  # use your 19-Apr-2024 anchor
        "next_peak": None,                  # next cycle handled by 2024 block
        "next_bottom": None,
        "projected": False,
    },
    2024: {
        "label": "2024-2028",
        "halving": date(2024, 4, 19),
        "peak": date(2025, 10, 7),   # your projected cycle top
        "bottom": date(2026, 10, 8), # your projected cycle bottom
        "next_halving": date(2028, 4, 14),  # your projected next halving
        "next_peak": None,   # we derive from pnp_days
        "next_bottom": None, # derive from bnb_days

        # These keep your original projection maths:
        "pnp_days": 1445,  # Peak → Next Peak
        "bnb_days": 1445,  # Bottom → Next Bottom
        "bnp_days": 1079,  # Bottom → Next Peak

        "projected": True,  # mark that peak/bottom/next-halving/next-X are projections
    },
}

ACTIVE_CYCLE_YEAR = 2024  # used for countdowns

def _fmt_cycle_date(d: date | None) -> str:
    return d.strftime("%d-%b-%Y") if d else "n/a"


def build_cycle_message(cycle_year: int, today: date | None = None) -> str:
    """
    Build the /cycle text for a given halving year.
    For past cycles: just show dates + durations.
    For the active 2024 cycle: also append a countdown block.
    """
    if today is None:
        today = datetime.utcnow().date()

    if cycle_year not in CYCLE_DEFS:
        raise KeyError(f"Unknown cycle year {cycle_year}")

    cfg = CYCLE_DEFS[cycle_year]
    h = cfg["halving"]
    p = cfg.get("peak")
    b = cfg.get("bottom")
    nh = cfg.get("next_halving")
    np = cfg.get("next_peak")
    nb = cfg.get("next_bottom")

    # Derive projected next peak/bottom from offsets if not explicitly stored
    if np is None and p and cfg.get("pnp_days") is not None:
        np = p + timedelta(days=cfg["pnp_days"])
    if nb is None and b and cfg.get("bnb_days") is not None:
        nb = b + timedelta(days=cfg["bnb_days"])

    def days(a: date | None, b: date | None, fallback_key: str | None = None):
        if a and b:
            return (b - a).days
        if fallback_key and cfg.get(fallback_key) is not None:
            return cfg[fallback_key]
        return None

    lines: list[str] = []

    # Header block
    lines.append(f"📆 Cycle {cfg['label']}")
    lines.append(f"- Halving: {_fmt_cycle_date(h)}")
    if p:
        lines.append(
            f"- Peak:    {_fmt_cycle_date(p)}"
            + (" (projected)" if cfg["projected"] and cycle_year == ACTIVE_CYCLE_YEAR else "")
        )
    if b:
        lines.append(
            f"- Bottom:  {_fmt_cycle_date(b)}"
            + (" (projected)" if cfg["projected"] and cycle_year == ACTIVE_CYCLE_YEAR else "")
        )
    if nh:
        lines.append(
            f"- Next Halving: {_fmt_cycle_date(nh)}"
            + (" (projected)" if cfg["projected"] and cycle_year == ACTIVE_CYCLE_YEAR else "")
        )
    if np:
        lines.append(
            f"- Next Peak:    {_fmt_cycle_date(np)}"
            + (" (projected)" if cfg["projected"] and cycle_year == ACTIVE_CYCLE_YEAR else "")
        )
    if nb:
        lines.append(
            f"- Next Bottom:  {_fmt_cycle_date(nb)}"
            + (" (projected)" if cfg["projected"] and cycle_year == ACTIVE_CYCLE_YEAR else "")
        )

    # Duration block
    lines.append("")
    lines.append("⏱ Durations:")
    dd = days(h, nh)
    if dd is not None:
        lines.append(f"- Halving → Halving: {dd:5d} days")
    dd = days(p, b)
    if dd is not None:
        lines.append(f"- Peak → Bottom:     {dd:5d} days")
    dd = days(b, nh)
    if dd is not None:
        lines.append(f"- Bottom → Halving:  {dd:5d} days")
    dd = days(p, np, "pnp_days")
    if dd is not None:
        lines.append(f"- Peak → Next Peak:      {dd:5d} days")
    dd = days(b, nb, "bnb_days")
    if dd is not None:
        lines.append(f"- Bottom → Next Bottom:  {dd:5d} days")
    dd = days(b, np, "bnp_days")
    if dd is not None:
        lines.append(f"- Bottom → Next Peak:    {dd:5d} days")

    # Countdown only for the active (2024) cycle
    if cycle_year == ACTIVE_CYCLE_YEAR:
        lines.append("")
        lines.append(f"⌛ Time Left (from {today:%d-%b-%Y}):")

        def left(label: str, target: date | None):
            if not target:
                return
            delta = (target - today).days
            if delta >= 0:
                lines.append(f"- To {label}: {delta:4d} days")
            else:
                lines.append(f"- Since {label}: {-delta:4d} days")

        # Local-bottom countdown stays handled by your separate send_cycle_countdown()
        left("Bottom", b)
        left("Halving", nh)
        left("Peak", np)

    return "\n".join(lines)


    # --- Header & anchors ---
    lines = []
    lines.append(f"📆 Cycle {c['name']}")
    lines.append(f"- Halving: {fmt_anchor(c['halving'])}")
    lines.append(f"- Peak:    {fmt_anchor(c['peak'])}")
    lines.append(f"- Bottom:  {fmt_anchor(c['bottom'])}")
    lines.append(f"- Next Halving: {fmt_anchor(c['next_halving'])}")
    lines.append("")
    lines.append("⏱ Durations:")
    lines.append(f"- Halving → Halving: {halving_to_halving} days")
    lines.append(f"- Peak → Bottom:     {peak_to_bottom} days")
    lines.append(f"- Bottom → Halving:  {bottom_to_halving} days")
    if nxt:
        lines.append(f"- Peak → Next Peak:      {peak_to_peak} days")
        lines.append(f"- Bottom → Next Bottom:  {bottom_to_bottom} days")
        lines.append(f"- Bottom → Next Peak:    {bottom_to_next_peak} days")

    # --- Optional time-left block (only for current cycle) ---
    if with_time_left:
        time_to_bottom = days(today, c["bottom"])
        time_to_halving = days(today, c["next_halving"])
        time_to_peak = days(today, nxt["peak"]) if nxt else None

        lines.append("")
        lines.append(f"⌛ Time Left (from {today:%d-%b-%Y}):")
        lines.append(f"- To Bottom:  {time_to_bottom} days")
        lines.append(f"- To Halving: {time_to_halving} days")
        if time_to_peak is not None:
            lines.append(f"- To Peak:    {time_to_peak} days")

    return "\n".join(lines)

def get_current_cycle_idx(today: date | None = None) -> int:
    """
    Return the index in CYCLES for the cycle that 'today' sits in.
    If we're past all known next_halving dates, return the last cycle.
    """
    today = today or date.today()
    current_idx = len(CYCLES) - 1
    for i, c in enumerate(CYCLES):
        if c["halving"] <= today < c["next_halving"]:
            current_idx = i
            break
    return current_idx

def get_local_bottom_for_cycle(idx: int):
    """
    For historical cycles: return explicit local_bottom if present.
    For future cycles: project using the previous cycle's
    (peak -> local_bottom) lag.
    """
    c = CYCLES[idx]

    # If this cycle explicitly has a local_bottom, just use it
    if "local_bottom" in c:
        return c["local_bottom"]

    # Otherwise, try to project from previous cycle
    prev_idx = idx - 1
    if prev_idx >= 0:
        prev = CYCLES[prev_idx]
        if "local_bottom" in prev:
            lag_days = days(prev["peak"], prev["local_bottom"])
            # project: current peak + lag
            from datetime import timedelta
            return c["peak"] + timedelta(days=lag_days)

    return None

def build_cycle_countdown_text(today: date | None = None) -> str:
    """
    Clean, aligned, Telegram-safe countdown block.
    """
    today = today or date.today()
    idx = get_current_cycle_idx(today)
    c = CYCLES[idx]
    nxt = CYCLES[idx + 1] if idx + 1 < len(CYCLES) else None

    def clamp(n: int) -> int:
        return n if n > 0 else 0

    # values
    local_bottom = get_local_bottom_for_cycle(idx)
    to_local = clamp(days(today, local_bottom)) if local_bottom else None

    to_bottom  = clamp(days(today, c["bottom"]))
    to_halving = clamp(days(today, c["next_halving"]))
    to_peak    = clamp(days(today, nxt["peak"])) if nxt else None

    # compute width so all days align
    numbers = [v for v in [to_local, to_bottom, to_halving, to_peak] if v is not None]
    width = max(len(str(v)) for v in numbers)

    # Build lines (monospaced inside ``` block)
    lines = []
    lines.append(f"⌛ Time Left (as of {today:%d-%b-%Y})")
    lines.append("-----------------------------------------------------------")
    if to_local is not None:
        lines.append(f"Local Bottom : {to_local:>{width}} days")
    lines.append(f"Cycle Bottom : {to_bottom:>{width}} days")
    lines.append(f"Next Halving : {to_halving:>{width}} days")
    if to_peak is not None:
        lines.append(f"Next Peak    : {to_peak:>{width}} days")

    # Wrap in triple backticks WITHOUT indentation
    return "\n" + "\n".join(lines) + "\n"



# -------- Google Sheets integration --------
SHEETS_ID  = os.environ.get("WOR_SHEETS_SPREADSHEET_ID", "").strip()
SHEET_TAB  = os.environ.get("WOR_SHEETS_WORKSHEET", "WOR").strip()
CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON", "").strip()
CREDS_B64  = os.environ.get("GOOGLE_CREDS_JSON_B64", "").strip()

print("[Sheets] SHEETS_ID set?", bool(SHEETS_ID))
print("[Sheets] SHEET_TAB:", SHEET_TAB or "<missing>")

_gs_client = None
_gs_ws     = None

# Journal / risk — single source of truth
JOURNAL_HEADERS: List[str] = [
    "week_start_utc", "symbol",
    "or_high", "or_low", "or_mid", "or_width",
    "break_dir", "break_time_utc", "phase",
    "fri_close", "close_pos_vs_or", "result", "pnl_r",
    # --- user inputs / trade details ---
    "size_pct",          # legacy sizing column (kept)
    "risk_pct",          # % of equity risked for this trade (optional)
    "risk_usd",          # $ at risk (optional)
    "entry_time_utc",    # when you entered (optional)
    "entry_price",       # entry price (optional)
    "side",              # long | short (optional)
    "exit_time_utc",     # when you exited (optional)
    "exit_price",        # exit price (optional)
    "regime", "notes",
    "window_start_utc", "window_end_utc",
    "ext_hits", "data_source", "bot_version",
    "snapshot_path",
    # --- cash flow for TWR (manual deposits/withdrawals) ---
    "cash_flow_usd",     # +deposit / -withdrawal (optional)
    "flow_when"          # "before" or "after" the week’s return (default "after")
]

def _sheets_boot():
    global _gs_client, _gs_ws, SHEETS_ID, SHEET_TAB, CREDS_JSON, CREDS_B64
    if not SHEETS_ID:
        print("[Sheets] Missing spreadsheet ID")
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        if _gs_client is not None and _gs_ws is not None:
            return _gs_ws

        info = None
        if CREDS_JSON:
            try:
                info = json.loads(CREDS_JSON)
                print("[Sheets] Parsed GOOGLE_CREDS_JSON as plain JSON")
            except Exception as e:
                print("[Sheets] JSON parse failed; trying GOOGLE_CREDS_JSON_B64…", e)
        if info is None and CREDS_B64:
            info = json.loads(base64.b64decode(CREDS_B64).decode("utf-8"))
            print("[Sheets] Parsed GOOGLE_CREDS_JSON_B64 as base64 JSON")
        if info is None:
            print("[Sheets] No service account creds found in env")
            return None

        # Fix escaped newlines in private key if needed
        if "private_key" in info and "\\n" in info["private_key"]:
            info["private_key"] = info["private_key"].replace("\\n", "\n")

        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        _gs_client = gspread.authorize(creds)
        sh = _gs_client.open_by_key(SHEETS_ID)
        try:
            _gs_ws = sh.worksheet(SHEET_TAB)
        except Exception:
            _gs_ws = sh.add_worksheet(title=SHEET_TAB, rows=1000, cols=30)

        # Ensure header row matches
        header = _gs_ws.row_values(1)
        if header != JOURNAL_HEADERS:
            _gs_ws.clear()
            _gs_ws.update("A1", [JOURNAL_HEADERS])
        print(f"[Sheets] Connected ✅ (tab '{SHEET_TAB}')")
        return _gs_ws

        try:
            _gs_ws = sh.worksheet(SHEET_TAB)
        except Exception:
            _gs_ws = sh.add_worksheet(
                title=SHEET_TAB,
                rows=2000,
                cols=max(60, len(JOURNAL_HEADERS) + 10),
            )

        # make sure the sheet is wide/tall enough
        want_cols = max(60, len(JOURNAL_HEADERS) + 10)
        want_rows = max(2000, _gs_ws.row_count)
        if _gs_ws.col_count < want_cols or _gs_ws.row_count < want_rows:
            _gs_ws.resize(rows=want_rows, cols=want_cols)

        # ensure header matches exactly
        header = _gs_ws.row_values(1)
        if header != JOURNAL_HEADERS:
            _gs_ws.clear()
            _gs_ws.update("A1", [JOURNAL_HEADERS])

    except Exception as e:
        print("[Sheets] boot error:", repr(e))
        return None

def append_to_sheets(row: dict) -> bool:
    ws = _sheets_boot()
    if ws is None:
        print("[Sheets] append: ws is None (boot failed)")
        return False
    try:
        values = [row.get(k, "") for k in JOURNAL_HEADERS]
        ws.append_row(values, value_input_option="USER_ENTERED")
        print("[Sheets] Row appended ✅")
        return True
    except Exception as e:
        # Print the full error so you can see WHY it failed
        import traceback
        print("[Sheets] append error:", repr(e))
        traceback.print_exc()
        return False


# Core secrets / ids
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
print("Loaded TELEGRAM_BOT_TOKEN:", _mask(TELEGRAM_BOT_TOKEN))
TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")
if not TELEGRAM_BOT_TOKEN or not TOKEN_RE.match(TELEGRAM_BOT_TOKEN):
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing or malformed.")

GROUP_ID     = int(os.environ.get("WATERMELON_GROUP_ID", "0") or "0")
WOR_TOPIC_ID = int(os.environ.get("WOR_TOPIC_ID", "0") or "0")
WOR_ARCHIVE_TOPIC_ID = int(os.environ.get("WOR_ARCHIVE_TOPIC_ID", "0") or "0")
TBI_TOPIC_ID = int(os.environ.get("TBI_TOPIC_ID", "0") or "0")
CHANNEL_ID   = int(os.environ.get("WOR_CHANNEL_ID", "0") or "0")

# Strategy defaults
SYMBOL            = "BTC/USDT"
DATA_TIMEFRAME    = "1h"
CHECK_INTERVAL_MIN= 5
STATE_FILE        = "wor_state_btc.json"
EXT_LEVELS        = [0.5, 1.0, 1.5]

# Journal / risk files
JOURNAL_CSV       = os.environ.get("WOR_JOURNAL_CSV", "wor_journal.csv")
LOSS_R            = float(os.environ.get("WOR_RISK_POLICY", "-2.0"))
DEFAULT_SIZE_PCT  = float(os.environ.get("WOR_DEFAULT_SIZE_PCT", "1.0"))
ROLLOVER_MIN      = int(os.environ.get("WOR_WEEK_ROLLOVER_MINUTE", "5"))
SNAP_DIR          = "journal_snaps"

# ============================ JOURNAL HELPERS ============================

def _ensure_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(JOURNAL_HEADERS)

def _append_csv(path: str, row: dict):
    _ensure_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_HEADERS, quoting=csv.QUOTE_ALL)
        w.writerow({k: row.get(k, "") for k in JOURNAL_HEADERS})

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        _ensure_csv(path)
        return pd.DataFrame(columns=JOURNAL_HEADERS)
    try:
        return pd.read_csv(path)
    except Exception as e:
        print("read_csv strict failed:", e)
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as e2:
            print("read_csv fallback failed:", e2)
            return pd.DataFrame(columns=JOURNAL_HEADERS)

# Starting equity (used by perf helpers)
START_EQ_USD = _env_float("WOR_START_EQUITY_USD", 100000.0)
print(f"[WOR] START_EQ_USD = {START_EQ_USD}")

# ============================ STATE ============================
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
            "alert_on": False,
            "last_week_alerts": {},
        }
    return _state[sid]

# ============================ HELPERS ============================
def reply_kwargs_for(m):
    if m.chat.type in ("supergroup","group") and getattr(m, "message_thread_id", None):
        return dict(message_thread_id=m.message_thread_id)
    return {}

def post_to_topic(topic_id: int, *, text=None, photo_bytes=None, document_path=None):
    if not GROUP_ID or not topic_id:
        return False
    try:
        if photo_bytes is not None:
            if isinstance(photo_bytes, (bytes, bytearray)):
                photo_bytes = io.BytesIO(photo_bytes)
                photo_bytes.name = "image.png"
            bot.send_photo(GROUP_ID, photo_bytes, caption=text or None, message_thread_id=topic_id)
        elif document_path is not None:
            with open(document_path, "rb") as f:
                bot.send_document(GROUP_ID, f,
                    visible_file_name=os.path.basename(document_path),
                    caption=text or None, message_thread_id=topic_id)
        elif text:
            bot.send_message(GROUP_ID, text, message_thread_id=topic_id)
        else:
            return False
        return True
    except Exception as e:
        print("post_to_topic error:", e)
        return False

def post_to_tbi(text=None, photo_bytes=None, document_path=None):
    if not GROUP_ID or not TBI_TOPIC_ID:
        return False
    try:
        if photo_bytes is not None:
            bot.send_photo(GROUP_ID, photo_bytes, caption=text or None,
                           message_thread_id=TBI_TOPIC_ID)
        elif document_path is not None:
            with open(document_path, "rb") as f:
                bot.send_document(GROUP_ID, f,
                    visible_file_name=os.path.basename(document_path),
                    caption=text or None,
                    message_thread_id=TBI_TOPIC_ID)
        elif text:
            bot.send_message(GROUP_ID, text, message_thread_id=TBI_TOPIC_ID)
        else:
            return False
        return True
    except Exception as e:
        print("post_to_tbi error:", e)
        return False

def send_cycle_countdown():
    """
    Daily countdown alert into TBI topic (if configured).
    Falls back to console print if no group/topic is set.
    """
    text = build_cycle_countdown_text()
    if GROUP_ID and TBI_TOPIC_ID:
        ok = post_to_tbi(text=text)
        if not ok:
            print("[cycle] countdown send failed:", text)
    else:
        # No group/topic configured – just log it
        print("[cycle] countdown:", text)

def post_to_channel(*, text=None, photo_bytes=None, document_path=None):
    if not CHANNEL_ID:
        return False
    try:
        if photo_bytes is not None:
            if isinstance(photo_bytes, (bytes, bytearray)):
                photo_bytes = io.BytesIO(photo_bytes)
                photo_bytes.name = "image.png"
            bot.send_photo(CHANNEL_ID, photo_bytes, caption=text or None)
        elif document_path is not None:
            with open(document_path, "rb") as f:
                bot.send_document(CHANNEL_ID, f,
                    visible_file_name=os.path.basename(document_path),
                    caption=text or None)
        elif text:
            bot.send_message(CHANNEL_ID, text)
        else:
            return False
        return True
    except Exception as e:
        print("post_to_channel error:", e)
        return False

# ==== PERF HELPERS (ΣR + compounded USD) ====

def _parse_week_safe(s):
    v = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(v):
        v = pd.to_datetime(s, utc=True, errors="coerce", dayfirst=True)
    return v

def load_journal_df():
    df = _read_csv(JOURNAL_CSV)
    if df.empty:
        return df
    df["week_start_utc"] = df["week_start_utc"].apply(_parse_week_safe)
    df = df.dropna(subset=["week_start_utc"]).sort_values("week_start_utc")
    for c in ("pnl_r", "size_pct", "risk_pct", "risk_usd", "equity_usd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "result" not in df.columns:
        df["result"] = np.where(df["pnl_r"]>0,"win", np.where(df["pnl_r"]<0,"loss","void"))
    else:
        miss = df["result"].isna() | (df["result"].astype(str).str.strip()=="")
        df.loc[miss & (df["pnl_r"]>0), "result"] = "win"
        df.loc[miss & (df["pnl_r"]<0), "result"] = "loss"
        df.loc[miss & (df["pnl_r"]==0), "result"] = "void"
    return df

def compute_perf(df):
    if df.empty:
        return df
    trades = df.copy()
    mask_keep = trades["result"].astype(str).str.lower().isin(["win","loss","void","flow"]) | trades["result"].isna() | (trades["result"].astype(str).str.strip()=="")
    trades = trades[mask_keep].copy()
    trades["week_start_utc"] = trades["week_start_utc"].apply(_parse_week_safe)
    trades = trades.sort_values("week_start_utc")

    mask_wl = trades["result"].astype(str).str.lower().isin(["win","loss"])
    pnl_r_series = pd.to_numeric(trades.loc[mask_wl, "pnl_r"], errors="coerce").fillna(0.0)
    cum_r = pnl_r_series.cumsum()
    trades["equity_r"] = np.nan
    trades.loc[mask_wl, "equity_r"] = cum_r

    risk_pct = pd.to_numeric(trades.get("risk_pct"), errors="coerce") if "risk_pct" in trades.columns else None
    risk_usd = pd.to_numeric(trades.get("risk_usd"), errors="coerce") if "risk_usd" in trades.columns else None
    size_pct = pd.to_numeric(trades.get("size_pct"), errors="coerce") if "size_pct" in trades.columns else None

    cash_flow = pd.to_numeric(trades.get("cash_flow_usd"), errors="coerce") if "cash_flow_usd" in trades.columns else None
    flow_when = trades.get("flow_when").astype(str).str.lower() if "flow_when" in trades.columns else None

    eq_prev = START_EQ_USD
    eq_usd, ret_pct = [], []

    for i, row in trades.iterrows():
        cf = float(cash_flow.loc[i]) if (cash_flow is not None and pd.notna(cash_flow.loc[i])) else 0.0
        fw = (flow_when.loc[i] if flow_when is not None and pd.notna(flow_when.loc[i]) else "after")
        if fw == "before":
            eq_prev += cf

        res = str(row.get("result", "")).lower()
        if res in ("win","loss"):
            if risk_pct is not None and pd.notna(risk_pct.loc[i]):
                r_t = float(row.get("pnl_r", 0.0)) * (float(risk_pct.loc[i]) / 100.0)
            elif risk_usd is not None and pd.notna(risk_usd.loc[i]) and eq_prev > 0:
                r_t = float(row.get("pnl_r", 0.0)) * (float(risk_usd.loc[i]) / eq_prev)
            elif size_pct is not None and pd.notna(size_pct.loc[i]) and eq_prev > 0 and float(size_pct.loc[i]) > 5:
                r_t = float(row.get("pnl_r", 0.0)) * (float(size_pct.loc[i]) / eq_prev)
            else:
                r_t = float(row.get("pnl_r", 0.0)) * 0.01
            eq_now = eq_prev * (1.0 + r_t)
            ret_pct.append(r_t)
        else:
            eq_now = eq_prev
            ret_pct.append(0.0)

        if fw != "before":
            eq_now += cf

        eq_usd.append(eq_now)
        eq_prev = eq_now

    trades["equity_usd"] = eq_usd
    trades["ret_pct"] = ret_pct
    return trades

def plot_equity_r(trades):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(trades["week_start_utc"], trades["equity_r"], lw=2, color="cyan")
    ax.set_title("WOR Equity (ΣR)")
    ax.set_xlabel("Week"); ax.set_ylabel("ΣR")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=150); plt.close(fig)
    buf.seek(0); return buf

def plot_equity_usd(trades):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(trades["week_start_utc"], trades["equity_usd"], lw=2, color="deepskyblue")
    ax.set_title("WOR Equity (Compounded USD)")
    ax.set_xlabel("Week"); ax.set_ylabel("Equity (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=150); plt.close(fig)
    buf.seek(0); return buf

def plot_equity_index(trades):
    plt.style.use("dark_background")
    eq = trades["equity_usd"].astype(float)
    base = float(eq.iloc[0]) if len(eq) else 1.0
    idx = eq / (base if base != 0 else 1.0)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(trades["week_start_utc"], idx, lw=2, color="violet")
    ax.set_title("WOR Compounded Equity (Index = 1.00)")
    ax.set_xlabel("Week"); ax.set_ylabel("Index")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=150); plt.close(fig)
    buf.seek(0); return buf

# ============================ INDICATORS (Weekly) ============================
def fetch_weekly(symbol: str = "BTC/USDT", since_days: int = 4000) -> pd.DataFrame:
    """
    Weekly OHLCV from Binance via ccxt, using the same fetch pattern as hourly.
    """
    binance = ccxt.binance({'enableRateLimit': True})
    tf = "1w"
    now = datetime.now(timezone.utc)
    since = int((now - timedelta(days=since_days)).timestamp() * 1000)

    ohlcv = []
    limit = 1000
    while True:
        batch = binance.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= int(now.timestamp() * 1000) - 60_000:
            break
        since = last_ts + 1

    if not ohlcv:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    df = pd.DataFrame(ohlcv, columns=["ts","Open","High","Low","Close","Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[["Open","High","Low","Close","Volume"]].astype(float)


def compute_weekly_indicators(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - 50-week MA (SMA)
      - Bull Market Support Band (BMSB): Upper = 20W SMA, Lower = 21W EMA
    """
    df = df_weekly.copy()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["BMSB_upper"] = df["SMA20"]
    df["BMSB_lower"] = df["EMA21"]
    return df


def latest_weekly_snapshot_text(df: pd.DataFrame) -> str:
    """
    Returns a concise multi-line text block with latest indicator values.
    """
    dfl = df.dropna(subset=["Close"])
    if dfl.empty:
        return "No weekly data."
    last = dfl.iloc[-1]
    close = float(last["Close"])
    ma50  = last.get("MA50")
    up    = last.get("BMSB_upper")
    lo    = last.get("BMSB_lower")

    # Position vs band
    pos = ""
    if pd.notna(up) and pd.notna(lo):
        if close > up: pos = "above band"
        elif close < lo: pos = "below band"
        else: pos = "within band"

    def fmt(x): 
        return f"{float(x):,.2f}" if pd.notna(x) else "—"

    lines = [
        "=== Latest Weekly Snapshot ===",
        f"Close : {fmt(close)} USDT",
        f"50W MA (SMA): {fmt(ma50)} USDT",
        f"BMSB Upper (20W SMA): {fmt(up)} USDT",
        f"BMSB Lower (21W EMA): {fmt(lo)} USDT",
        f"Position vs BMSB: {pos or '—'}"
    ]
    return "\n".join(lines)


def plot_weekly_indicators(df: pd.DataFrame) -> bytes:
    """
    Dark chart with Close, 50W MA, and BMSB shaded band.
    Returns PNG bytes.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["Close"], label="Close", lw=1.6, color="#87e5ff")

    if df["MA50"].notna().any():
        ax.plot(df.index, df["MA50"], label="50W MA (SMA)", lw=1.2, linestyle="--", color="#ffffff")

    have_band = df["BMSB_upper"].notna().any() and df["BMSB_lower"].notna().any()
    if have_band:
        ax.plot(df.index, df["BMSB_upper"], label="BMSB Upper (20W SMA)", lw=1.0, color="#f3b76b")
        ax.plot(df.index, df["BMSB_lower"], label="BMSB Lower (21W EMA)", lw=1.0, color="#f19b2c")
        ax.fill_between(df.index, df["BMSB_lower"], df["BMSB_upper"], alpha=0.18, color="#ffa726")

    ax.set_title("BTC/USDT — Weekly 50W MA & Bull Market Support Band", fontsize=13, weight="bold")
    ax.set_ylabel("Price (USDT)"); ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.legend(loc="best", fontsize=9, framealpha=0.25)
    ax.grid(alpha=0.15)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def weekly_snapshot_text_at(df_weekly: pd.DataFrame, which: int = -2) -> str:
    """
    Return a multi-line text for a specific weekly bar.
    which = -1 (latest bar, may be incomplete)
            -2 (previous fully-closed week)  <-- default
    """
    dfl = df_weekly.dropna(subset=["Close"])
    if len(dfl) < abs(which):
        return "Not enough weekly data."

    row = dfl.iloc[which]
    week_open = row.name  # index is the bar open time (UTC)
    week_label = pd.to_datetime(week_open).strftime("%Y-%m-%d")  # Monday open UTC (Binance)

    close = float(row["Close"])
    ma50  = row.get("MA50")
    up    = row.get("BMSB_upper")
    lo    = row.get("BMSB_lower")

    pos = ""
    if pd.notna(up) and pd.notna(lo):
        if close > up: pos = "above band"
        elif close < lo: pos = "below band"
        else: pos = "within band"

    def fmt(x): return f"{float(x):,.2f}" if pd.notna(x) else "—"

    lines = [
        f"=== Weekly Snapshot (Week start {week_label} UTC) ===",
        f"Close : {fmt(close)} USDT",
        f"50W MA (SMA): {fmt(ma50)} USDT",
        f"BMSB Upper (20W SMA): {fmt(up)} USDT",
        f"BMSB Lower (21W EMA): {fmt(lo)} USDT",
        f"Position vs BMSB: {pos or '—'}",
    ]
    return "\n".join(lines)

# ============================ WEEKLY INDICATORS (for overlay) ============================

def _fetch_weekly_full(symbol: str = "BTC/USDT", since_days: int = 4000) -> pd.DataFrame:
    binance = ccxt.binance({'enableRateLimit': True})
    tf = "1w"
    now = datetime.now(timezone.utc)
    since = int((now - timedelta(days=since_days)).timestamp() * 1000)

    ohlcv, limit = [], 1000
    while True:
        batch = binance.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= int(now.timestamp() * 1000) - 60_000:
            break
        since = last_ts + 1

    if not ohlcv:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    df = pd.DataFrame(ohlcv, columns=["ts","Open","High","Low","Close","Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[["Open","High","Low","Close","Volume"]].astype(float)

def _compute_weekly_indis(dfw: pd.DataFrame) -> pd.DataFrame:
    df = dfw.copy()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["BMSB_upper"] = df["SMA20"]
    df["BMSB_lower"] = df["EMA21"]
    return df

def get_indicator_box_text(symbol: str = "BTC/USDT") -> str:
    """
    Returns a very compact single-week snapshot for overlay.
    Example:
      50W: 102,981 | BMSB: 110,398–113,750 | Pos: below
    """
    try:
        dfw = _compute_weekly_indis(_fetch_weekly_full(symbol, since_days=4000))
        dfl = dfw.dropna(subset=["Close"])
        if dfl.empty:
            return "50W: — | BMSB: — | Pos: —"
        last = dfl.iloc[-1]
        close = float(last["Close"])
        ma50  = last.get("MA50")
        up    = last.get("BMSB_upper")
        lo    = last.get("BMSB_lower")

        def f(x):
            return "—" if pd.isna(x) else f"{float(x):,.0f}"

        pos = "—"
        if pd.notna(up) and pd.notna(lo):
            if close > up: pos = "above"
            elif close < lo: pos = "below"
            else: pos = "within"

        return f"50W: {f(ma50)} | BMSB: {f(lo)}–{f(up)} | Pos: {pos}"
    except Exception:
        return "50W: — | BMSB: — | Pos: —"

# ============================ DATA & WOR LOGIC ============================

def fetch_ohlcv(symbol: str = "BTC/USDT", period_days: int = 10, interval: str = "1h") -> pd.DataFrame:
    """
    Pull the last `period_days` of OHLCV from Binance via ccxt and return a UTC-indexed DataFrame
    with columns: Open, High, Low, Close, Volume.
    """
    binance = ccxt.binance({'enableRateLimit': True})
    tf = interval

    now = datetime.now(timezone.utc)
    since = int((now - timedelta(days=period_days)).timestamp() * 1000)

    ohlcv = []
    limit = 1000
    while True:
        batch = binance.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= int(now.timestamp() * 1000) - 60_000:
            break
        since = last_ts + 1

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
def fetch_ohlcv_between(symbol="BTC/USDT", start: pd.Timestamp = None, end: pd.Timestamp = None, interval="1h"):
    """
    Fetch OHLCV between [start, end) using ccxt Binance.
    Falls back to a modest safety window if dates are missing.
    """
    if start is None or end is None:
        raise ValueError("fetch_ohlcv_between: start and end required")

    binance = ccxt.binance({'enableRateLimit': True})
    tf = interval
    since = int(pd.Timestamp(start).tz_convert(timezone.utc).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).tz_convert(timezone.utc).timestamp() * 1000)

    ohlcv, limit = [], 1000
    while True:
        batch = binance.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= end_ms - 60_000:
            break
        since = last_ts + 1

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["ts","Open","High","Low","Close","Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[["Open","High","Low","Close","Volume"]].astype(float)

def start_of_week(ts):
    dt = pd.to_datetime(ts).tz_convert(timezone.utc)
    dow = dt.weekday()  # Monday=0
    return (dt - timedelta(days=dow)).replace(hour=0, minute=0, second=0, microsecond=0)

def compute_wor_mondayclose(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["OR_high","OR_low","OR_mid","OR_width"]).astype(float)
    df = df.copy()
    df["week"] = df.index.map(start_of_week)
    rows = []
    for wk, g in df.groupby("week"):
        monday = g[g.index.weekday == 0]
        if monday.empty: continue
        or_high, or_low = float(monday["High"].max()), float(monday["Low"].min())
        rows.append({"week": wk, "OR_high": or_high, "OR_low": or_low,
                     "OR_mid": (or_high + or_low)/2.0, "OR_width": or_high - or_low})
    return pd.DataFrame(rows).set_index("week").sort_index()

def latest_week_or(df: pd.DataFrame):
    wor = compute_wor_mondayclose(df)
    if wor.empty: return None, None
    wk = wor.index.max()
    return wk, wor.loc[wk]

def breakout_and_extensions(df, week_start, or_high, or_low, ext_levels):
    g = df[(df.index >= week_start) & (df.index < week_start + timedelta(days=7))]
    if g.empty:
        return {"break": None, "first_time": None, "hits": []}
    broke_up = g["Close"] > or_high
    broke_dn = g["Close"] < or_low
    up_time = broke_up.idxmax() if broke_up.any() else None
    dn_time = broke_dn.idxmax() if broke_dn.any() else None

    direction, first_time, base = None, None, None
    if up_time and (not dn_time or up_time < dn_time):
        direction, first_time, base = "UP", up_time, or_high
    elif dn_time:
        direction, first_time, base = "DOWN", dn_time, or_low

    hits = []
    if direction:
        sub = g[g.index >= first_time]
        width = or_high - or_low
        if width > 0:
            for k in ext_levels:
                if direction == "UP":
                    tgt = base + k*width
                    hit = (sub["High"] >= tgt).any()
                else:
                    tgt = base - k*width
                    hit = (sub["Low"] <= tgt).any()
                if hit: hits.append(k)
    return {"break": direction, "first_time": first_time, "hits": hits}

# ============================ CHART ============================
def plot_week_chart(df, wk, or_row, outcome: str = None):
    hi = float(or_row["OR_high"]); lo = float(or_row["OR_low"]); mid = float(or_row["OR_mid"])
    orl_minus = lo * (1 - 0.02)
    orh_plus  = hi * (1 + 0.02)

    start = wk
    mon_close = wk + timedelta(days=1)
    thu_cutoff = wk + timedelta(days=3, hours=12)
    end = wk + timedelta(days=7)

    week_df = df[(df.index >= start) & (df.index <= end)]
    if week_df.empty:
        buf = io.BytesIO()
        plt.figure(figsize=(8,3)); plt.text(0.5, 0.5, "No data for this week", ha="center", va="center")
        plt.axis("off"); plt.tight_layout(); plt.savefig(buf, format="png", dpi=150); plt.close(); buf.seek(0)
        return buf.getvalue()

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Alternate day shading
    for i in range(7):
        day_start = wk + timedelta(days=i); day_end = day_start + timedelta(days=1)
        if day_start > end: break
        if i % 2 == 0:
            ax.axvspan(day_start, min(day_end, end), color="white", alpha=0.08)

    ax.plot(week_df.index, week_df["Close"], lw=2.0, color="#b7f1e3", label="BTC/USDT")
    ax.axhline(orh_plus, color="white", linestyle="-.", lw=1.2, label="ORH+")    
    ax.axhline(hi,  color="white", linestyle="--", lw=1.4, label="ORH")
    ax.axhline(mid, color="white", linestyle=":",  lw=1.2, label="ORM")
    ax.axhline(lo,  color="white", linestyle="--", lw=1.4, label="ORL")
    ax.axhline(orl_minus, color="white", linestyle="-.", lw=1.2, label="ORL-")


    xr = week_df.index[-1]
    ax.text(xr, orh_plus, f"{orh_plus:.0f}", va="top", ha="left", fontsize=8, color="white")    
    ax.text(xr, hi,  f"{hi:.0f}",  va="bottom", ha="left", fontsize=8, color="white")
    ax.text(xr, mid, f"{mid:.0f}", va="bottom", ha="left", fontsize=8, color="white")
    ax.text(xr, lo,  f"{lo:.0f}",  va="top",    ha="left", fontsize=8, color="white")
    ax.text(xr, orl_minus, f"{orl_minus:.0f}", va="top", ha="left", fontsize=8, color="white")


    def add_rect(x0, x1, y0, y1, label, alpha=0.10):
        rect = Rectangle((x0, y0), (x1 - x0), (y1 - y0), facecolor="white", edgecolor=None, alpha=alpha, zorder=0.6)
        ax.add_patch(rect); ax.text(x0+(x1-x0)/2, y0+(y1-y0)/2, label, color="white", fontsize=10, ha="center", va="center", alpha=0.6)

    tue_start, tue_end = wk + timedelta(days=1), wk + timedelta(days=2)
    wed_start, wed_end = wk + timedelta(days=2), wk + timedelta(days=3)
    ymin, ymax = ax.get_ylim()
    add_rect(tue_start, tue_end, lo,  mid, "Phase 1A", alpha=0.10)
    add_rect(tue_start, tue_end, mid, hi,  "Phase 1B", alpha=0.10)
    add_rect(wed_start, wed_end, hi, ymax, "Phase 1C", alpha=0.10)
    add_rect(wed_start, wed_end, lo, ymin, "Phase 1D", alpha=0.10)

    if start <= mon_close <= end:
        ax.axvline(mon_close, color="orange", linestyle="--", lw=1.4)
    if start <= thu_cutoff <= end:
        ax.axvline(thu_cutoff, color="orange", linestyle="--", lw=1.4)

    title = f"{SYMBOL} — WOR (Monday Close)  |  Week {wk.strftime('%Y-%m-%d')}"
    width_abs = hi - lo; width_pct = (width_abs / mid * 100.0) if mid else 0.0
    ax.set_title(title + f"\nWidth: {width_abs:.0f}  ({width_pct:.2f}%)", fontsize=13, weight="bold")

    ax.set_ylabel("Price (USDT)"); ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d %Hh")); fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=9, framealpha=0.25)
    ax.set_xlim(start, end); ax.set_xmargin(0)

    # === Weekly indicator box (compact overlay) ===
    try:
        dfw = _compute_weekly_indis(_fetch_weekly_full(SYMBOL, since_days=4000))
        last = dfw.dropna(subset=["Close"]).iloc[-1]

        def fmt_full(x):
            return "—" if pd.isna(x) else f"{float(x):,.2f}"

        ma50 = fmt_full(last.get("MA50"))
        up   = fmt_full(last.get("BMSB_upper"))
        lo   = fmt_full(last.get("BMSB_lower"))

        pos = "—"
        if pd.notna(last.get("BMSB_upper")) and pd.notna(last.get("BMSB_lower")):
            c, u, l = float(last["Close"]), float(last["BMSB_upper"]), float(last["BMSB_lower"])
            pos = "above band" if c > u else ("below band" if c < l else "within band")

        text_lines = (
            f"50W: {ma50}\n"
            f"BMSB Upper: {up}\n"
            f"BMSB Lower: {lo}\n"
            f"Position: {pos}"
        )

        at = AnchoredText(
            text_lines,
            loc="upper right",                    # change to "lower right" if you prefer
            prop=dict(size=9, color="white"),
            frameon=True,
            borderpad=0.6,
            pad=0.4,
        )
        at.patch.set_boxstyle("round,pad=0.35")
        at.patch.set_facecolor((0, 0, 0, 0.45))
        at.patch.set_edgecolor("white")
        at.patch.set_linewidth(0.6)
        ax.add_artist(at)
    except Exception as e:
        print("indicator box error:", e)

    # save
    buf = io.BytesIO()
    plt.tight_layout(); plt.savefig(buf, format="png", dpi=150); plt.close(fig); buf.seek(0)
    return buf.getvalue()

def save_week_snapshot(df, wk, or_row, outcome_text=None) -> str:
    os.makedirs(SNAP_DIR, exist_ok=True)
    png_bytes = plot_week_chart(df, wk, or_row, outcome=outcome_text)
    fname = f"{wk.strftime('%Y-%m-%d')}.png"
    fpath = os.path.join(SNAP_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(png_bytes)
    return fpath

# ============================ OUTCOME / JOURNAL ============================
def week_bounds(week_start: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(week_start).tz_convert(timezone.utc)
    end = start + timedelta(days=7)
    return start, end

def week_last_close(df: pd.DataFrame, week_start: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
    start, end = week_bounds(week_start)
    g = df[(df.index >= start) & (df.index < end)]
    if g.empty: return None
    ts = g.index.max()
    return ts, float(g.loc[ts, "Close"])

def _pos_vs_or(close_price: float, hi: float, lo: float) -> str:
    try:
        if pd.isna(close_price) or pd.isna(hi) or pd.isna(lo):
            return "unknown"
    except Exception:
        pass
    if close_price >= hi:
        return "above_orh"
    if close_price <= lo:
        return "below_orl"
    return "inside"

def _infer_phase(break_dir: Optional[str], first_time: Optional[pd.Timestamp]) -> str:
    if not break_dir:
        return "none"
    return "1B" if break_dir.upper() == "UP" else "1A"

def compute_week_outcome(df: pd.DataFrame, wk: pd.Timestamp, or_row: pd.Series) -> dict:
    sig = breakout_and_extensions(df, wk, or_row["OR_high"], or_row["OR_low"], EXT_LEVELS)
    last = week_last_close(df, wk)
    if last is None: return {"finalized": False}
    last_ts, last_close = last
    where = _pos_vs_or(last_close, or_row["OR_high"], or_row["OR_low"])
    if sig["break"] == "UP":
        result = "win" if where == "above_orh" else "loss"
        pnl_r = 1.0 if result == "win" else LOSS_R
    elif sig["break"] == "DOWN":
        result = "win" if where == "below_orl" else "loss"
        pnl_r = 1.0 if result == "win" else LOSS_R
    else:
        result = "void"; pnl_r = 0.0
    phase = _infer_phase(sig["break"], sig["first_time"])
    return {
        "finalized": True,
        "break_dir": sig["break"],
        "break_time": sig["first_time"],
        "phase": phase,
        "fri_close": last_close,
        "close_pos_vs_or": where,
        "result": result,
        "pnl_r": pnl_r,
        "ext_hits": sig["hits"]
    }

# ============================ BOT ============================
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
bot.remove_webhook()
me = bot.get_me()
print(f"Bot online as @{me.username} (id={me.id})")
load_state()

HELP_TEXT_SHORT = """\
/wor range
/wor chart
/wor equity
/wor twr
/wor indi
/wor help
"""

HELP_TEXT = """\
/wor range
/wor signal
/wor chart
/wor alert on | off
/wor journal [n]
/wor stats
/wor equity # ΣR
/wor twr [usd|index] # Compounded
/wor indi # Weekly 50W MA + Bull Market Support Band
/wor export
/wor note <text>
/wor finalize [YYYY-MM-DD]  # auto write a completed week row (Fri close)
/wor enter <YYYY-MM-DD> <long|short> <size_pct|1%|$1000> [@price] [note...]
/wor exit  <YYYY-MM-DD> <win|loss|void> <pnl_r> [@price] [note...]
/wor flow  <+usd|-usd> [before|after] [YYYY-MM-DD]
/wor alive
/wor push <chart|equity|twr|export>
/wor help
"""

def monday_of(ts):
    t = pd.to_datetime(ts, utc=True)
    monday = (t - pd.Timedelta(days=t.weekday())).normalize()
    return monday

def _ensure_or_for(date_utc: pd.Timestamp):
    df = fetch_ohlcv(SYMBOL, period_days=30, interval=DATA_TIMEFRAME)
    wor = compute_wor_mondayclose(df)
    monday = monday_of(date_utc)
    if monday not in wor.index:
        raise RuntimeError(f"No Monday data for {monday.date()}")
    r = wor.loc[monday]
    return monday, float(r["OR_high"]), float(r["OR_low"]), float(r["OR_mid"]), float(r["OR_width"])

@bot.message_handler(commands=["sheets"])
def sheets_router(m):
    parts = m.text.strip().split()
    sub = parts[1].lower() if len(parts) > 1 else "ping"

    if sub == "ping":
        ws = _sheets_boot()
        bot.reply_to(m, "Sheets OK → Spreadsheet ID set, tab '{}' ready.".format(SHEET_TAB) if ws
                       else "Sheets: not connected. Check console for [Sheets] logs.")
        return

    if sub == "appendtest":
        dummy = {k: "" for k in JOURNAL_HEADERS}
        dummy["notes"] = "append-test"
        ok = append_to_sheets(dummy)
        bot.reply_to(m, "Append OK ✅" if ok else "Append failed — see console logs.")
        return

@bot.message_handler(commands=['cycle'])
@bot.message_handler(commands=['cycle'])
def handle_cycle(message):
    """
    Usage:
      /cycle          -> active cycle (2024-2028)
      /cycle 2020     -> 2020-2024 cycle
      /cycle 2016     -> 2016-2020 cycle
      /cycle 2012     -> 2012-2016 cycle
    """
    parts = message.text.split()
    year = ACTIVE_CYCLE_YEAR

    if len(parts) >= 2:
        try:
            year = int(parts[1])
        except ValueError:
            # if user types junk like '/cycle abc', just fall back
            year = ACTIVE_CYCLE_YEAR

    if year not in CYCLE_DEFS:
        available = ", ".join(str(y) for y in sorted(CYCLE_DEFS))
        bot.reply_to(
            message,
            f"Unknown cycle year *{year}*.\n"
            f"Available cycles: {available}",
            parse_mode="Markdown",
        )
        return

    text = build_cycle_message(year)
    bot.reply_to(message, text)


@bot.message_handler(commands=['countdown'])
def handle_countdown(message):
    text = build_cycle_countdown_text()
    if GROUP_ID and TBI_TOPIC_ID:
        post_to_tbi(text=text)
        bot.reply_to(message, "Countdown pushed to TBI topic.", **reply_kwargs_for(message))
    else:
        bot.reply_to(message, text, **reply_kwargs_for(message))

@bot.message_handler(commands=["start"])
def start_cmd(m):
    bot.reply_to(m, "Hi! Watermelon is alive. Try /wor help")

@bot.message_handler(commands=["ping"])
def ping_cmd(m):
    bot.reply_to(m, "Watermelon online")

def monday_of_date_str(date_s: str) -> pd.Timestamp:
    dt = pd.to_datetime(date_s, utc=True, errors="coerce")
    if pd.isna(dt):
        raise ValueError("Bad date; use YYYY-MM-DD")
    return (dt - pd.Timedelta(days=dt.weekday())).normalize()  # Monday 00:00 UTC

def render_week_chart_for_date(date_s: str):
    """
    Returns (caption_str, png_bytes). Uses the same dark plot_week_chart().
    """
    wk = monday_of_date_str(date_s)
    # Fetch 10 days around the week to be safe (Mon..next Mon + 3 days)
    start = wk - pd.Timedelta(days=1)
    end   = wk + pd.Timedelta(days=8)

    df = fetch_ohlcv_between(SYMBOL, start=start, end=end, interval=DATA_TIMEFRAME)
    if df.empty:
        raise RuntimeError(f"No OHLCV data found between {start} and {end}.")

    # Compute WOR for the whole span and grab this Monday
    wor_all = compute_wor_mondayclose(df)
    if wk not in wor_all.index:
        raise RuntimeError(f"No Monday data for {wk.date()}.")

    row = wor_all.loc[wk]
    png = plot_week_chart(df, wk, row)
    cap = f"{SYMBOL} — Week {wk.strftime('%Y-%m-%d')}"
    return cap, png, wk

@bot.message_handler(commands=["wor"])
def wor_router(message):
    text = (message.text or "").strip()
    parts = text.split()
    args = parts[1:] if len(parts) > 1 else []
    sub  = args[0].lower() if args else ""

    chat_id = message.chat.id
    cfg = get_chat_cfg(chat_id)

    if not args:
        bot.reply_to(message, HELP_TEXT_SHORT, **reply_kwargs_for(message))
        return
    
    if sub in ("help", "?"):
        bot.reply_to(message, HELP_TEXT, **reply_kwargs_for(message))
        return

    if sub in ("alive", "status"):
        dfj = _read_csv(JOURNAL_CSV)
        last_entry = "none" if dfj.empty else str(dfj["week_start_utc"].iloc[-1])
        alert_state = "ON" if cfg.get("alert_on") else "OFF"
        try:
            if message.chat.type in ("supergroup", "group"):
                group_title = message.chat.title or "Group"
            elif GROUP_ID:
                group_title = bot.get_chat(GROUP_ID).title or "Group"
            else:
                group_title = "Group not configured"
        except Exception:
            group_title = "Group"
        topic_hint = ""
        mtid = getattr(message, "message_thread_id", None)
        if mtid:
            if WOR_TOPIC_ID and mtid == WOR_TOPIC_ID:
                topic_hint = " (#WOR)"
            elif MOR_TOPIC_ID and mtid == MOR_TOPIC_ID:
                topic_hint = " (#MOR)"
            elif PIE_TOPIC_ID and mtid == PIE_TOPIC_ID:
                topic_hint = " (#PIE)"
        bot.reply_to(
            message,
            f"✅ WOR bot active\n"
            f"Symbol: {SYMBOL}\n"
            f"Mode: Monday Close\n"
            f"Connected to: {group_title}{topic_hint}\n"
            f"Last journal entry: {last_entry}\n"
            f"Alerts: {alert_state}",
            **reply_kwargs_for(message)
        )
        return

    if sub == "push":
        if len(args) < 2:
            bot.reply_to(message, "Usage: /wor push <chart|equity|twr|export>", **reply_kwargs_for(message)); return
        what = args[1].lower()

        if what == "chart":
            df = fetch_ohlcv(SYMBOL, period_days=10, interval=DATA_TIMEFRAME)
            wk, row = latest_week_or(df)
            if row is None:
                bot.reply_to(message, "No data to chart.", **reply_kwargs_for(message)); return
            png = plot_week_chart(df, wk, row)
            ok_topic = post_to_topic(WOR_TOPIC_ID, text=f"WOR Chart {wk.strftime('%Y-%m-%d')}", photo_bytes=png)
            ok_chan  = post_to_channel(text=f"WOR Chart {wk.strftime('%Y-%m-%d')}", photo_bytes=png)
            bot.reply_to(message, "Pushed to topic ✅" if ok_topic or ok_chan else "No topic/channel configured.", **reply_kwargs_for(message))
            return

        if what == "equity":
            dfj = _read_csv(JOURNAL_CSV)
            if dfj.empty:
                bot.reply_to(message, "No equity yet — journal is empty.", **reply_kwargs_for(message)); return
            dfj["pnl_r"] = pd.to_numeric(dfj["pnl_r"], errors="coerce").fillna(0.0)
            dfj = dfj[dfj["result"].isin(["win","loss"])].copy()
            if dfj.empty:
                bot.reply_to(message, "No completed trades yet.", **reply_kwargs_for(message)); return
            dfj["week_start_utc"] = pd.to_datetime(dfj["week_start_utc"], utc=True, errors="coerce")
            dfj = dfj.sort_values("week_start_utc"); dfj["equity"] = dfj["pnl_r"].cumsum()

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(dfj["week_start_utc"], dfj["equity"], lw=2.0, color="cyan")
            ax.set_title("WOR Equity Curve (ΣR)"); ax.set_xlabel("Week"); ax.set_ylabel("ΣR")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d")); fig.autofmt_xdate()
            buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=150); plt.close(fig); buf.seek(0)

            ok_topic = post_to_topic(WOR_TOPIC_ID, text="WOR Equity Curve (ΣR)", photo_bytes=buf)
            ok_chan  = post_to_channel(text="WOR Equity Curve (ΣR)", photo_bytes=buf)
            bot.reply_to(message, "Pushed to topic ✅" if ok_topic or ok_chan else "No topic/channel configured.", **reply_kwargs_for(message))
            return

        if what == "twr":
            dfj = load_journal_df()
            trades = compute_perf(dfj)
            if trades.empty:
                bot.reply_to(message, "No completed trades yet.", **reply_kwargs_for(message)); return
            buf = plot_equity_index(trades)
            cap = "WOR Compounded Equity (Index = 1.00)"
            ok_topic = post_to_topic(WOR_TOPIC_ID, text=cap, photo_bytes=buf)
            ok_chan  = post_to_channel(text=cap, photo_bytes=buf)
            bot.reply_to(message, "Pushed TWR chart ✅" if ok_topic or ok_chan else "No topic/channel configured.", **reply_kwargs_for(message))
            return

        if what == "export":
            ok_topic = post_to_topic(WOR_TOPIC_ID, text="WOR Journal CSV", document_path=JOURNAL_CSV)
            ok_chan  = post_to_channel(text="WOR Journal CSV", document_path=JOURNAL_CSV)
            bot.reply_to(message, "Pushed CSV ✅" if ok_topic or ok_chan else "No topic/channel configured.", **reply_kwargs_for(message))
            return

        bot.reply_to(message, "Usage: /wor push <chart|equity|twr|export>", **reply_kwargs_for(message))
        return

    if sub == "enter":
        if len(args) < 4:
            bot.reply_to(message, ("Usage:\n"
                                   "/wor enter <YYYY-MM-DD> <long|short> <size_pct|1%|$1000> [@price] [note...]"),
                         **reply_kwargs_for(message)); return
        date_s = args[1]
        side   = args[2].lower()
        sizing = args[3]
        rest   = args[4:] if len(args) > 4 else []
        note_txt = ""
        entry_price = ""
        for tok in rest:
            if tok.startswith("@"):
                entry_price = tok[1:]
            else:
                note_txt += (tok + " ")
        note_txt = note_txt.strip()
        try:
            entry_time = pd.to_datetime(date_s, utc=True)
        except Exception:
            bot.reply_to(message, "Bad date. Use YYYY-MM-DD.", **reply_kwargs_for(message)); return
        monday, orh, orl, orm, orw = _ensure_or_for(entry_time)
        risk_pct = ""
        risk_usd = ""
        size_pct = ""
        s = sizing.strip().lower().replace("%","")
        if sizing.startswith("$"):
            risk_usd = s[0:]
            if risk_usd.startswith("$"):
                risk_usd = risk_usd[1:]
        elif sizing.endswith("%"):
            risk_pct = s
        else:
            size_pct = s
        row = {c: "" for c in JOURNAL_HEADERS}
        row.update({
            "week_start_utc": monday.strftime("%Y-%m-%d"),
            "symbol": SYMBOL,
            "or_high": f"{orh:.2f}",
            "or_low": f"{orl:.2f}",
            "or_mid": f"{orm:.2f}",
            "or_width": f"{orw:.2f}",
            "result": "",
            "pnl_r": "",
            "size_pct": size_pct,
            "risk_pct": risk_pct,
            "risk_usd": risk_usd,
            "entry_time_utc": entry_time.strftime("%Y-%m-%d %H:%M UTC"),
            "entry_price": entry_price,
            "side": side,
            "notes": note_txt
        })
        _append_csv(JOURNAL_CSV, row)
        append_to_sheets(row)
        bot.reply_to(message, f"Entered {side.upper()} logged for week {row['week_start_utc']}.", **reply_kwargs_for(message))
        return

    if sub == "exit":
        if len(args) < 4:
            bot.reply_to(message, ("Usage:\n"
                                   "/wor exit <YYYY-MM-DD> <win|loss|void> <pnl_r> [@price] [note...]"),
                         **reply_kwargs_for(message)); return
        date_s = args[1]
        result = args[2].lower()
        pnl_r  = args[3]
        rest   = args[4:] if len(args) > 4 else []
        exit_price = ""
        note_txt = ""
        for tok in rest:
            if tok.startswith("@"):
                exit_price = tok[1:]
            else:
                note_txt += (tok + " ")
        note_txt = note_txt.strip()
        try:
            exit_time = pd.to_datetime(date_s, utc=True)
        except Exception:
            bot.reply_to(message, "Bad date. Use YYYY-MM-DD.", **reply_kwargs_for(message)); return
        monday = monday_of(exit_time).strftime("%Y-%m-%d")
        dfj = _read_csv(JOURNAL_CSV)
        if dfj.empty:
            bot.reply_to(message, "No journal yet—use /wor enter first.", **reply_kwargs_for(message)); return
        mask_week = dfj["week_start_utc"].astype(str) == monday
        mask_open = dfj["result"].astype(str).str.strip().eq("") | dfj["result"].isna()
        idxs = dfj.index[mask_week & mask_open]
        if len(idxs) == 0:
            bot.reply_to(message, f"No open entry found for week {monday}.", **reply_kwargs_for(message)); return
        i = idxs[-1]
        dfj.at[i, "result"] = result
        dfj.at[i, "pnl_r"]  = pnl_r
        dfj.at[i, "exit_time_utc"] = exit_time.strftime("%Y-%m-%d %H:%M UTC")
        if exit_price:
            dfj.at[i, "exit_price"] = exit_price
        if note_txt:
            prev = str(dfj.at[i, "notes"]) if pd.notna(dfj.at[i, "notes"]) else ""
            dfj.at[i, "notes"] = (prev + " | " if prev else "") + note_txt
        dfj.to_csv(JOURNAL_CSV, index=False)
        append_to_sheets({c: str(dfj.at[i, c]) if c in dfj.columns else "" for c in JOURNAL_HEADERS})
        bot.reply_to(message, f"Exit logged for week {monday}: {result.upper()} {pnl_r}R.", **reply_kwargs_for(message))
        return

    if sub == "flow":
        if len(args) < 2:
            bot.reply_to(message, "Usage: /wor flow <+usd|-usd> [before|after] [YYYY-MM-DD]", **reply_kwargs_for(message)); return
        amt_s = args[1]
        when  = args[2].lower() if len(args) >= 3 and args[2].lower() in ("before","after") else "after"
        date_s= args[3] if len(args) >= 4 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            amt = float(amt_s)
        except Exception:
            bot.reply_to(message, "Cash flow must be a number like +2000 or -1500.", **reply_kwargs_for(message)); return
        try:
            dt = pd.to_datetime(date_s, utc=True)
        except Exception:
            bot.reply_to(message, "Bad date. Use YYYY-MM-DD.", **reply_kwargs_for(message)); return
        monday, orh, orl, orm, orw = _ensure_or_for(dt)
        row = {c: "" for c in JOURNAL_HEADERS}
        row.update({
            "week_start_utc": monday.strftime("%Y-%m-%d"),
            "symbol": SYMBOL,
            "or_high": f"{orh:.2f}", "or_low": f"{orl:.2f}", "or_mid": f"{orm:.2f}", "or_width": f"{orw:.2f}",
            "result": "flow",
            "cash_flow_usd": f"{amt}",
            "flow_when": when,
            "notes": f"manual cash flow {amt:+g} ({when})"
        })
        _append_csv(JOURNAL_CSV, row)
        append_to_sheets(row)
        bot.reply_to(message, f"Cash flow {amt:+g} ({when}) recorded for week {row['week_start_utc']}.",
                     **reply_kwargs_for(message))
        return

    if sub == "finalize":
        target = None
        if len(args) >= 2:
            try:
                target = pd.to_datetime(args[1], utc=True).normalize()
            except Exception:
                bot.reply_to(message, "Use /wor finalize YYYY-MM-DD (Monday date).", **reply_kwargs_for(message)); return
        df = fetch_ohlcv(SYMBOL, period_days=30, interval=DATA_TIMEFRAME)
        wk, or_row = latest_week_or(df)
        if or_row is None:
            bot.reply_to(message, "No data available to finalize.", **reply_kwargs_for(message)); return
        if target is not None:
            wor_all = compute_wor_mondayclose(df)
            if target not in wor_all.index:
                bot.reply_to(message, f"No Monday data for {target.date()}.", **reply_kwargs_for(message)); return
            wk, or_row = target, wor_all.loc[target]
        outcome = compute_week_outcome(df, wk, or_row)
        if not outcome.get("finalized"):
            bot.reply_to(message, "Week not complete yet (no Friday close).", **reply_kwargs_for(message)); return
        window_start = wk + timedelta(days=1)
        window_end   = wk + timedelta(days=3, hours=12)
        snap_path = save_week_snapshot(df, wk, or_row, outcome_text=outcome["result"].upper())
        row = {c: "" for c in JOURNAL_HEADERS}
        row.update({
            "week_start_utc": wk.strftime("%Y-%m-%d"),
            "symbol": SYMBOL,
            "or_high": f"{or_row['OR_high']:.2f}",
            "or_low": f"{or_row['OR_low']:.2f}",
            "or_mid": f"{or_row['OR_mid']:.2f}",
            "or_width": f"{or_row['OR_width']:.2f}",
            "break_dir": outcome["break_dir"] or "",
            "break_time_utc": outcome["break_time"].strftime("%Y-%m-%d %H:%M UTC") if outcome["break_time"] else "",
            "phase": outcome["phase"],
            "fri_close": f"{outcome['fri_close']:.2f}",
            "close_pos_vs_or": outcome["close_pos_vs_or"],
            "result": outcome["result"],
            "pnl_r": f"{outcome['pnl_r']:.2f}",
            "size_pct": f"{DEFAULT_SIZE_PCT:.2f}",
            "regime": "",
            "notes": "",
            "window_start_utc": window_start.strftime("%Y-%m-%d %H:%M UTC"),
            "window_end_utc": window_end.strftime("%Y-%m-%d %H:%M UTC"),
            "ext_hits": ",".join(map(str, outcome.get("ext_hits", []))),
            "data_source": "binance_spot",
            "bot_version": "wor-btc-only-ccxt-1.0",
            "snapshot_path": snap_path
        })
        dfj = _read_csv(JOURNAL_CSV)
        exists = (not dfj.empty) and (dfj["week_start_utc"].astype(str) == row["week_start_utc"]).any()
        if exists:
            idx = dfj.index[dfj["week_start_utc"].astype(str) == row["week_start_utc"]][0]
            for k, v in row.items():
                dfj.at[idx, k] = v
            dfj.to_csv(JOURNAL_CSV, index=False)
            append_to_sheets(row)
            bot.reply_to(message, f"Finalized (updated) {row['week_start_utc']}.", **reply_kwargs_for(message))
        else:
            _append_csv(JOURNAL_CSV, row)
            append_to_sheets(row)
            bot.reply_to(message, f"Finalized (added) {row['week_start_utc']}.", **reply_kwargs_for(message))
        return

    if sub == "range":
        df = fetch_ohlcv(SYMBOL, period_days=10, interval=DATA_TIMEFRAME)
        wk, row = latest_week_or(df)
        if row is None:
            bot.reply_to(message, "No data to compute WOR.", **reply_kwargs_for(message)); return
        msg = (
            f"WOR (Monday Close) for week starting {wk.strftime('%Y-%m-%d')} (UTC)\n"
            f"OR High: {row['OR_high']:.2f}\nOR Low: {row['OR_low']:.2f}\n"
            f"Width: {row['OR_width']:.2f} ({(row['OR_width']/row['OR_mid']*100):.2f}%)\n"
            f"Mid: {row['OR_mid']:.2f}"
        )
        bot.reply_to(message, msg, **reply_kwargs_for(message)); return

    if sub == "signal":
        df = fetch_ohlcv(SYMBOL, period_days=10, interval=DATA_TIMEFRAME)
        wk, row = latest_week_or(df)
        if row is None:
            bot.reply_to(message, "No data to compute signals.", **reply_kwargs_for(message)); return
        sig = breakout_and_extensions(df, wk, row["OR_high"], row["OR_low"], EXT_LEVELS)
        if not sig["break"]:
            bot.reply_to(message, "No breakout yet — price inside OR.", **reply_kwargs_for(message)); return
        ext_str = ", ".join([str(k) for k in sig["hits"]]) if sig["hits"] else "None"
        tstr = sig["first_time"].strftime('%Y-%m-%d %H:%M UTC') if sig["first_time"] else "—"
        bot.reply_to(message, f"Break: {sig['break']} at {tstr}\nExtensions hit: {ext_str}", **reply_kwargs_for(message)); return

    if sub == "chart":
        # Case A: /wor chart YYYY-MM-DD  -> ALWAYS post to WOR Archive topic (if configured)
        if len(args) >= 2:
            try:
                cap, png, wk = render_week_chart_for_date(args[1])  # your existing renderer
            except Exception as e:
                bot.reply_to(message, str(e), **reply_kwargs_for(message))
                return

            # Prefer archive topic; fallback to current chat
            tid = int(WOR_ARCHIVE_TOPIC_ID) if str(WOR_ARCHIVE_TOPIC_ID).strip() else None
            if tid is None:
                tid = get_or_create_wor_archive_topic_id()  # optional helper; safe if you added it

            if GROUP_ID and tid:
                bot.send_photo(GROUP_ID, png, caption=cap, message_thread_id=tid)
            else:
                bot.send_photo(message.chat.id, png, caption=cap, **reply_kwargs_for(message))
            return

        # Case B: /wor chart  -> ALWAYS post to WOR topic (if configured)
        df = fetch_ohlcv(SYMBOL, period_days=10, interval=DATA_TIMEFRAME)
        wk, row = latest_week_or(df)
        
        if row is None:
            bot.reply_to(message, "No data to chart.", **reply_kwargs_for(message)); return

        png = plot_week_chart(df, wk, row)

        tid = int(WOR_TOPIC_ID) if str(WOR_TOPIC_ID).strip() else None
        if tid is None:
            tid = get_or_create_wor_topic_id()  # optional helper; safe if you added it

        # --- wrap raw bytes into a file-like object ---
        fileobj = io.BytesIO(png)
        fileobj.name = f"WOR_{wk.strftime('%Y-%m-%d')}.png"

        caption = f"{SYMBOL} — Week {wk.strftime('%Y-%m-%d')}"
        if GROUP_ID and tid:
            bot.send_photo(GROUP_ID, fileobj, caption=caption, message_thread_id=tid)
        else:
            bot.send_photo(message.chat.id, fileobj, caption=caption)
        return

    if sub == "alert":
        if len(args) < 2 or args[1].lower() not in ("on","off"):
            bot.reply_to(message, "Usage: /wor alert on|off", **reply_kwargs_for(message)); return
        opt = args[1].lower(); cfg["alert_on"] = (opt == "on"); save_state()
        bot.reply_to(message, f"WOR alerts: {'ON' if cfg['alert_on'] else 'OFF'}", **reply_kwargs_for(message)); return

    if sub == "journal":
        n = 5
        if len(args) >= 2 and args[1].isdigit():
            n = int(args[1])
        dfj = _read_csv(JOURNAL_CSV)
        if dfj.empty:
            bot.reply_to(message, "Journal is empty.", **reply_kwargs_for(message)); return
        tail = dfj.tail(n)
        lines = []
        for _, r in tail.iterrows():
            lines.append(
                f"{r['week_start_utc']} | {r['symbol']} | {r['result']} | {r['pnl_r']}R | "
                f"Close: {r['fri_close']} ({r['close_pos_vs_or']}) | Phase: {r['phase']}"
            )
        bot.reply_to(message, "Last entries:\n" + "\n".join(lines), **reply_kwargs_for(message)); return

    if sub == "stats":
        dfj = _read_csv(JOURNAL_CSV)
        if dfj.empty:
            bot.reply_to(message, "No stats: journal is empty.", **reply_kwargs_for(message)); return
        dfj["pnl_r"] = pd.to_numeric(dfj["pnl_r"], errors="coerce").fillna(0.0)
        trades = dfj[dfj["result"].isin(["win","loss"])]
        if trades.empty:
            bot.reply_to(message, "No completed trades yet.", **reply_kwargs_for(message)); return
        wins = (trades["result"] == "win").sum(); total = len(trades)
        hit = 100.0 * wins / total if total else 0.0
        avg_r = trades["pnl_r"].mean(); std_r = trades["pnl_r"].std(ddof=0) if total > 1 else 0.0
        eq = trades["pnl_r"].cumsum().iloc[-1]
        bot.reply_to(message,
            f"Stats:\nTrades: {total} | Wins: {wins} | Hit: {hit:.1f}%\n"
            f"Avg R: {avg_r:.2f} | Stdev R: {std_r:.2f}\nEquity (ΣR): {eq:.2f}",
            **reply_kwargs_for(message)
        ); return

    if sub == "export":
        if not os.path.exists(JOURNAL_CSV):
            _ensure_csv(JOURNAL_CSV)
        with open(JOURNAL_CSV, "rb") as f:
            bot.send_document(chat_id, f, visible_file_name=os.path.basename(JOURNAL_CSV), **reply_kwargs_for(message))
        return

    if sub == "equity":
        dfj = load_journal_df()
        trades = compute_perf(dfj)
        if trades.empty:
            bot.reply_to(message, "No completed trades yet.", **reply_kwargs_for(message)); return
        buf = plot_equity_r(trades)
        bot.send_photo(chat_id, buf, **reply_kwargs_for(message))
        return

    if sub == "twr":
        mode = "index"
        if len(args) >= 2 and args[1].lower() in ("usd","index"):
            mode = args[1].lower()
        dfj = load_journal_df()
        trades = compute_perf(dfj)
        if trades.empty:
            bot.reply_to(message, "No completed trades yet.", **reply_kwargs_for(message)); return
        if mode == "usd":
            buf = plot_equity_usd(trades)
            cap = "WOR Compounded Equity (USD)"
        else:
            buf = plot_equity_index(trades)
            cap = "WOR Compounded Equity (Index = 1.00)"
        bot.send_photo(chat_id, buf, caption=cap, **reply_kwargs_for(message))
        return
    
    if sub == "indi":
        try:
            weekly = fetch_weekly(SYMBOL, since_days=4000)
            if weekly.empty:
                bot.reply_to(message, "No weekly data available.", **reply_kwargs_for(message))
                return

            dfw = compute_weekly_indicators(weekly)
            snapshot = latest_weekly_snapshot_text(dfw)
            png = plot_weekly_indicators(dfw)
            fileobj = io.BytesIO(png); fileobj.name = "weekly_indicators.png"

            # Prefer WOR topic if configured; else reply in the current chat/thread
            tid = int(WOR_TOPIC_ID) if str(WOR_TOPIC_ID).strip() else None
            caption = f"{SYMBOL} — Weekly Indicators\n" + snapshot

            if GROUP_ID and tid:
                bot.send_photo(GROUP_ID, fileobj, caption=caption, message_thread_id=tid)
            else:
                bot.send_photo(message.chat.id, fileobj, caption=caption, **reply_kwargs_for(message))
        except Exception as e:
            bot.reply_to(message, f"/wor indi error: {e}", **reply_kwargs_for(message))
        return
   
    bot.reply_to(message, "Unknown /wor subcommand. Try /wor help", **reply_kwargs_for(message))

# ============================ ALERT LOOP ============================
def check_alerts():
    for sid, cfg in list(_state.items()):
        if not cfg.get("alert_on"): continue
        chat_id = int(sid)
        try:
            df = fetch_ohlcv(SYMBOL, period_days=10, interval=DATA_TIMEFRAME)
            wk, row = latest_week_or(df)
            if row is None: continue
            sig = breakout_and_extensions(df, wk, row["OR_high"], row["OR_low"], EXT_LEVELS)
            week_key = wk.strftime("%Y-%m-%d")
            hist = cfg.setdefault("last_week_alerts", {}).setdefault(week_key, {"break": None, "ext_hit": []})
            if sig["break"] and hist.get("break") is None:
                tstr = sig["first_time"].strftime("%Y-%m-%d %H:%M UTC") if sig["first_time"] else "—"
                telebot.TeleBot(TELEGRAM_BOT_TOKEN).send_message(chat_id, f"WOR Break {sig['break']} — {SYMBOL} at {tstr}")
                hist["break"] = sig["break"]
            for k in sig["hits"]:
                if k not in hist["ext_hit"]:
                    telebot.TeleBot(TELEGRAM_BOT_TOKEN).send_message(chat_id, f"WOR {sig['break']} extension hit: {k}x range — {SYMBOL}")
                    hist["ext_hit"].append(k)
            save_state()
        except Exception as e:
            print("check_alerts error:", e)

# ============================ FINALIZER ============================
def finalize_weeks():
    now_utc = datetime.now(timezone.utc)
    for sid, cfg in list(_state.items()):
        try:
            df = fetch_ohlcv(SYMBOL, period_days=10, interval=DATA_TIMEFRAME)
            wk, or_row = latest_week_or(df)
            if or_row is None: continue
            start, end = week_bounds(wk)
            if not (end <= now_utc <= end + timedelta(minutes=ROLLOVER_MIN)):
                continue
            dfj = _read_csv(JOURNAL_CSV)
            wk_str = wk.strftime("%Y-%m-%d")
            if (not dfj.empty):
                have_done = (dfj["week_start_utc"].astype(str) == wk_str) & (dfj["result"].astype(str).str.lower().isin(["win","loss","void"]))
                if have_done.any():
                    continue
            outcome = compute_week_outcome(df, wk, or_row)
            if not outcome.get("finalized"): continue
            window_start = wk + timedelta(days=1)
            window_end   = wk + timedelta(days=3, hours=12)
            snap_path = save_week_snapshot(df, wk, or_row, outcome_text=outcome["result"].upper())
            row = {c: "" for c in JOURNAL_HEADERS}
            row.update({
                "week_start_utc": wk_str,
                "symbol": SYMBOL,
                "or_high": f"{or_row['OR_high']:.2f}",
                "or_low": f"{or_row['OR_low']:.2f}",
                "or_mid": f"{or_row['OR_mid']:.2f}",
                "or_width": f"{or_row['OR_width']:.2f}",
                "break_dir": outcome["break_dir"] or "",
                "break_time_utc": outcome["break_time"].strftime("%Y-%m-%d %H:%M UTC") if outcome["break_time"] else "",
                "phase": outcome["phase"],
                "fri_close": f"{outcome['fri_close']:.2f}",
                "close_pos_vs_or": outcome["close_pos_vs_or"],
                "result": outcome["result"],
                "pnl_r": f"{outcome['pnl_r']:.2f}",
                "size_pct": f"{DEFAULT_SIZE_PCT:.2f}",
                "regime": "",
                "notes": "",
                "window_start_utc": window_start.strftime("%Y-%m-%d %H:%M UTC"),
                "window_end_utc": window_end.strftime("%Y-%m-%d %H:%M UTC"),
                "ext_hits": ",".join(map(str, outcome.get("ext_hits", []))),
                "data_source": "binance_spot",
                "bot_version": "wor-btc-only-ccxt-1.0",
                "snapshot_path": snap_path
            })
            _append_csv(JOURNAL_CSV, row)
            append_to_sheets(row)
            print(f"Journal entry written for {wk_str} with snapshot {snap_path}")
            try:
                if GROUP_ID and WOR_TOPIC_ID:
                    with open(snap_path, "rb") as f:
                        bot.send_photo(GROUP_ID, f,
                            caption=f"WOR Wrap {wk_str} — {row['result'].upper()} | {row['pnl_r']}R",
                            message_thread_id=WOR_TOPIC_ID)
            except Exception as e:
                print("topic wrap post error:", e)
            try:
                if CHANNEL_ID:
                    with open(snap_path, "rb") as f:
                        bot.send_photo(CHANNEL_ID, f,
                            caption=f"WOR Wrap {wk_str} — {row['result'].upper()} | {row['pnl_r']}R")
            except Exception as e:
                print("channel wrap post error:", e)
        except Exception as e:
            print("finalize_weeks error:", e)

# ============================ MAIN ============================
if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_alerts, "interval", minutes=CHECK_INTERVAL_MIN)
    scheduler.add_job(finalize_weeks, "interval", minutes=1)
    # NEW: daily cycle countdown to TBI at 09:00 local time
    scheduler.add_job(send_cycle_countdown, "cron", hour=9, minute=0)
    scheduler.start()
    print("[Scheduler] Jobs:")
    for job in scheduler.get_jobs():
        print("  ", job)
    print("WOR BTC-only bot running…")
    try:
        bot.polling(none_stop=True)
    except telebot.apihelper.ApiTelegramException as e:
        try:
            code = getattr(e, "result_json", {}).get("error_code")
        except Exception:
            code = None
        if code == 401:
            print("Telegram 401 Unauthorized: invalid or revoked token.")
        raise
