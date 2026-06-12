"""
╔══════════════════════════════════════════════════════════════════════════╗
║      INSTITUTIONAL DERIVATIVES WORKSTATION  v3.5                        ║
║      Merged: v3.0 full architecture + v3.2 dual T1/T2 target system    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Engine A — Nifty Options Direction   (4/6 conditions, ADX filter)      ║
║  Engine B — Fresh Crossovers / Strong Trend mode + sector bonus         ║
║  Engine C — F&O Money Flow  (relative volume ≥ 1.3×)                   ║
║  Engine D — Institutional Relative Strength  (20-day RS vs Nifty)       ║
║  Engine E — Event Risk Filter  (earnings / ex-div within 7 days)        ║
║  Alpha Board — Ranked composite + dual targets + position sizing        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TRADE LEVELS (from v3.2):                                              ║
║    SL     = Entry ± 1.5 × ATR                                           ║
║    Target1 = Entry ± 1.5 × risk  (partial exit, 1:1.5 R:R)             ║
║    Target2 = Entry ± 2.5 × risk  (full exit,    1:2.5 R:R)             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Institutional Derivatives Workstation v3.5",
    page_icon="🛡️",
)

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 90

STOCK_UNIVERSE = sorted(set([
    "RELIANCE", "SBIN", "TCS", "INFY", "TATAMOTORS", "ITC", "HDFCBANK", "ICICIBANK",
    "BHARTIARTL", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "MARUTI", "BAJFINANCE",
    "M&M", "SUNPHARMA", "ULTRACEMCO", "POWERGRID", "NTPC", "TITAN", "ADANIENT",
    "JSWSTEEL", "TATASTEEL", "HINDALCO", "COALINDIA", "BPCL", "ONGC", "GRASIM",
    "TECHM", "WIPRO", "HCLTECH", "APOLLOHOSP", "DIVISLAB", "CIPLA", "DRREDDY",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "INDUSINDBK", "BAJAJFINSV", "BRITANNIA",
    "NESTLEIND", "ASIANPAINT", "ADANIPORTS", "HDFCLIFE", "SBILIFE", "BEL", "HAL",
    "ASHOKLEY", "AUBANK", "BALKRISIND", "BANDHANBNK", "BERGEPAINT", "BHARATFORG",
    "CHOLAFIN", "CONCOR", "CUMMINSIND", "DLF", "EXIDEIND", "FEDERALBNK",
    "GODREJCP", "GMRINFRA", "HAVELLS", "IDFCFIRSTB", "IOC", "IRCTC", "JINDALSTEL",
    "LICHSGFIN", "LTIM", "LUPIN", "MRF", "MUTHOOTFIN", "NATIONALUM", "OBEROIRLTY",
    "PEL", "PFC", "PNB", "RECLTD", "SAIL", "SHREECEM", "SRF", "TATACOMM", "TATACONSUM",
    "TATAPOWER", "TVSMOTOR", "VOLTAS", "ZEEL", "ZYDUSLIFE", "PERSISTENT",
]))

SECTOR_MAP = {
    "BANK"    : ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
                 "BANDHANBNK","FEDERALBNK","IDFCFIRSTB","AUBANK","PNB"],
    "IT"      : ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT"],
    "AUTO"    : ["MARUTI","TATAMOTORS","M&M","EICHERMOT","HEROMOTOCO","BAJAJ-AUTO",
                 "ASHOKLEY","TVSMOTOR","EXIDEIND","BALKRISIND","BHARATFORG"],
    "PHARMA"  : ["SUNPHARMA","DIVISLAB","CIPLA","DRREDDY","LUPIN","APOLLOHOSP","ZYDUSLIFE"],
    "METAL"   : ["JSWSTEEL","TATASTEEL","HINDALCO","SAIL","NATIONALUM","JINDALSTEL"],
    "ENERGY"  : ["RELIANCE","BPCL","ONGC","TATAPOWER","POWERGRID","NTPC","IOC",
                 "COALINDIA","PFC","RECLTD"],
    "DEFENCE" : ["BEL","HAL","BHARATFORG"],
    "FMCG"    : ["ITC","HINDUNILVR","NESTLEIND","BRITANNIA","GODREJCP","TATACONSUM"],
    "INFRA"   : ["LT","ADANIPORTS","CONCOR","GMRINFRA","DLF","OBEROIRLTY","IRCTC"],
    "FINANCE" : ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","LICHSGFIN",
                 "HDFCLIFE","SBILIFE","PEL"],
    "CEMENT"  : ["ULTRACEMCO","SHREECEM","GRASIM"],
    "MISC"    : ["TITAN","ASIANPAINT","HAVELLS","BERGEPAINT","VOLTAS","MRF","ADANIENT",
                 "SRF","CUMMINSIND","TATACOMM","ZEEL"],
}

SYMBOL_SECTOR = {sym: sec for sec, syms in SECTOR_MAP.items() for sym in syms}

INDEX_TICKERS = ["^NSEI", "^NSEBANK"]
STOCK_TICKERS = [f"{s}.NS" for s in STOCK_UNIVERSE]
ALL_TICKERS   = INDEX_TICKERS + STOCK_TICKERS


# ═════════════════════════════════════════════════════════════════════════════
# ███  TRADE LEVEL CALCULATOR  (v3.2 dual-target system)  ███
# ═════════════════════════════════════════════════════════════════════════════

def calculate_trade_levels(entry: float, atr: float, bias: str) -> tuple:
    """
    Dual-target trade levels from v3.2.
      risk   = 1.5 × ATR
      SL     = entry ∓ risk
      Target1 = entry ± 1.5 × risk   →  partial exit (1 : 1.5 R:R)
      Target2 = entry ± 2.5 × risk   →  full exit    (1 : 2.5 R:R)
    """
    risk = 1.5 * atr
    if bias == "LONG":
        sl      = entry - risk
        target1 = entry + 1.5 * risk
        target2 = entry + 2.5 * risk
    else:
        sl      = entry + risk
        target1 = entry - 1.5 * risk
        target2 = entry - 2.5 * risk
    return round(sl, 2), round(target1, 2), round(target2, 2)


# ═════════════════════════════════════════════════════════════════════════════
# ███  QUANT UTILITIES  ███
# ═════════════════════════════════════════════════════════════════════════════

def calculate_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["HIGH"], df["LOW"], df["CLOSE"]
    pc      = c.shift(1)
    tr      = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c  = df["HIGH"], df["LOW"], df["CLOSE"]
    pc       = c.shift(1)
    up_move  = h - h.shift(1)
    dn_move  = l.shift(1) - l
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr       = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr      = pd.Series(tr).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    pdi      = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    mdi      = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx       = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ─── Table styling ────────────────────────────────────────────────────────────

def color_signal(val):
    if isinstance(val, str):
        if "LONG"     in val: return "background-color:#083d08;color:#00e676;font-weight:bold;"
        if "SHORT"    in val: return "background-color:#3d0808;color:#ff5252;font-weight:bold;"
        if "BUY CE"   in val: return "background-color:#083d08;color:#00e676;font-weight:bold;"
        if "BUY PE"   in val: return "background-color:#3d0808;color:#ff5252;font-weight:bold;"
        if "WATCH CE" in val: return "background-color:#2d2d00;color:#ffeb3b;font-weight:bold;"
        if "WATCH PE" in val: return "background-color:#2d1500;color:#ff9800;font-weight:bold;"
    return ""


def styled_df(df: pd.DataFrame, bias_col: str = "Bias"):
    if bias_col in df.columns:
        return df.style.map(color_signal, subset=[bias_col])
    return df.style


# ═════════════════════════════════════════════════════════════════════════════
# ███  DATA LAYER  ███
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def download_all_market_data() -> dict:
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    raw   = yf.download(
        tickers     = ALL_TICKERS,
        start       = start.strftime("%Y-%m-%d"),
        end         = end.strftime("%Y-%m-%d"),
        interval    = "1d",
        group_by    = "ticker",
        auto_adjust = True,
        progress    = False,
        threads     = True,
    )
    result = {}
    for ticker in ALL_TICKERS:
        try:
            df = raw[ticker].copy() if len(ALL_TICKERS) > 1 else raw.copy()
            df.dropna(subset=["Close"], inplace=True)
            if len(df) < 30:
                continue
            df.columns = [c.upper() for c in df.columns]
            result[ticker] = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        except Exception:
            continue
    return result


def extract_ticker_dataframe(batch: dict, ticker: str, min_rows: int = 55):
    df = batch.get(ticker)
    if df is None or len(df) < min_rows:
        return None
    return df.copy()


# ═════════════════════════════════════════════════════════════════════════════
# ███  MARKET BREADTH  (EMA50)  ███
# ═════════════════════════════════════════════════════════════════════════════

def calculate_market_breadth(batch: dict) -> float:
    above = total = 0
    for ticker in STOCK_TICKERS:
        df = extract_ticker_dataframe(batch, ticker)
        if df is None:
            continue
        ema50 = df["CLOSE"].ewm(span=50, adjust=False).mean()
        total += 1
        if df["CLOSE"].iloc[-1] > ema50.iloc[-1]:
            above += 1
    return round(above / total * 100, 1) if total else 50.0


# ═════════════════════════════════════════════════════════════════════════════
# ███  SECTOR STRENGTH  ███
# ═════════════════════════════════════════════════════════════════════════════

def calculate_sector_strength(batch: dict) -> dict:
    sector_returns: dict = {s: [] for s in SECTOR_MAP}
    for symbol in STOCK_UNIVERSE:
        df = extract_ticker_dataframe(batch, f"{symbol}.NS")
        if df is None or len(df) < 6:
            continue
        ret = float((df["CLOSE"].iloc[-1] - df["CLOSE"].iloc[-6]) / df["CLOSE"].iloc[-6] * 100)
        sec = SYMBOL_SECTOR.get(symbol)
        if sec:
            sector_returns[sec].append(ret)
    return {sec: round(np.mean(v), 2) if v else 0.0 for sec, v in sector_returns.items()}


def get_sector_bonus(symbol: str, sector_strengths: dict, bias: str) -> float:
    sec = SYMBOL_SECTOR.get(symbol)
    if not sec:
        return 0.0
    sorted_secs = sorted(sector_strengths, key=sector_strengths.get, reverse=(bias == "LONG"))
    return 10.0 if sec in sorted_secs[:3] else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE A — NIFTY OPTIONS DIRECTION  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_a(nifty_df: pd.DataFrame, banknifty_df: pd.DataFrame,
                 breadth: float) -> dict:
    metrics = {}
    for label, df in [("NIFTY", nifty_df), ("BANKNIFTY", banknifty_df)]:
        c    = df["CLOSE"]; h = df["HIGH"]; l = df["LOW"]
        ema5 = c.ewm(span=5,  adjust=False).mean()
        ema20= c.ewm(span=20, adjust=False).mean()
        ema50= c.ewm(span=50, adjust=False).mean()
        rsi  = calculate_wilder_rsi(c, 14)
        adx  = calculate_adx(df, 14)
        metrics[label] = {
            "close"           : round(float(c.iloc[-1]),    2),
            "ema5"            : round(float(ema5.iloc[-1]), 2),
            "ema20"           : round(float(ema20.iloc[-1]),2),
            "ema50"           : round(float(ema50.iloc[-1]),2),
            "rsi"             : round(float(rsi.iloc[-1]),  1),
            "adx"             : round(float(adx.iloc[-1]),  1),
            "prev_high"       : round(float(h.iloc[-2]),    2),
            "prev_low"        : round(float(l.iloc[-2]),    2),
            "above_prev_high" : float(c.iloc[-1]) > float(h.iloc[-2]),
            "below_prev_low"  : float(c.iloc[-1]) < float(l.iloc[-2]),
            "ema_bull"        : float(ema5.iloc[-1]) > float(ema20.iloc[-1]),
            "ema_bear"        : float(ema5.iloc[-1]) < float(ema20.iloc[-1]),
            "trending"        : float(adx.iloc[-1]) > 20,
        }

    n = metrics["NIFTY"]; bn = metrics["BANKNIFTY"]

    ce_conditions = {
        "Nifty > Prev Day High"       : n["above_prev_high"],
        "BankNifty > Prev Day High"   : bn["above_prev_high"],
        "Nifty EMA5 > EMA20"          : n["ema_bull"],
        "Nifty RSI > 55"              : n["rsi"] > 55,
        "Market Breadth (EMA50) > 60%": breadth > 60,
        "Nifty ADX > 20 (trending)"   : n["trending"],
    }
    pe_conditions = {
        "Nifty < Prev Day Low"        : n["below_prev_low"],
        "BankNifty < Prev Day Low"    : bn["below_prev_low"],
        "Nifty EMA5 < EMA20"          : n["ema_bear"],
        "Nifty RSI < 45"              : n["rsi"] < 45,
        "Market Breadth (EMA50) < 40%": breadth < 40,
        "Nifty ADX > 20 (trending)"   : n["trending"],
    }

    ce_met = sum(ce_conditions.values())
    pe_met = sum(pe_conditions.values())

    if   ce_met >= 4: signal, color = "🟢 BUY CE", "green"
    elif pe_met >= 4: signal, color = "🔴 BUY PE", "red"
    else:             signal, color = "⬛ NO TRADE — Await Confirmation", "gray"

    confidence = min(100, int((ce_met if "CE" in signal else pe_met) * 100 / 6))

    return {
        "signal": signal, "confidence": confidence, "color": color,
        "nifty": n, "banknifty": bn, "breadth": breadth,
        "ce_conditions": ce_conditions, "pe_conditions": pe_conditions,
        "ce_met": ce_met, "pe_met": pe_met,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE B — BREAKOUTS + DUAL TARGETS  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_b(batch: dict, nifty_df: pd.DataFrame,
                 sector_strengths: dict, scanner_mode: str = "Fresh Crossovers") -> dict:
    nifty_c  = nifty_df["CLOSE"]
    nifty_20d= float((nifty_c.iloc[-1] - nifty_c.iloc[-21]) / nifty_c.iloc[-21] * 100)

    long_signals = []; short_signals = []

    for ticker in STOCK_TICKERS:
        df = extract_ticker_dataframe(batch, ticker)
        if df is None:
            continue

        symbol = ticker.replace(".NS", "")
        c      = df["CLOSE"]; vol = df["VOLUME"]
        ema20  = c.ewm(span=20, adjust=False).mean()
        ema50  = c.ewm(span=50, adjust=False).mean()
        rsi    = calculate_wilder_rsi(c, 14)
        atr    = calculate_atr(df, 14)

        cur     = float(c.iloc[-1])
        cur_atr = float(atr.iloc[-1])
        cur_rsi = float(rsi.iloc[-1])
        avg_vol = float(vol.tail(20).mean())
        vol_r   = round(float(vol.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 0
        s20d    = float((c.iloc[-1] - c.iloc[-21]) / c.iloc[-21] * 100)
        rs_score= round(s20d - nifty_20d, 2)

        # ── LONG ──────────────────────────────────────────────────────
        if scanner_mode == "Fresh Crossovers":
            long_qual = (float(c.iloc[-2]) <= float(ema20.iloc[-2]) and
                         float(c.iloc[-1])  > float(ema20.iloc[-1]))
        else:
            long_qual = (cur > float(ema20.iloc[-1]) > float(ema50.iloc[-1]) and cur_rsi > 55)

        if long_qual:
            days_above = int((c > ema20).tail(20).sum())
            freshness  = max(0, 20 - days_above) if scanner_mode == "Fresh Crossovers" else 5
            rsi_accel  = max(0.0, float(rsi.iloc[-1]) - float(rsi.iloc[-2]))
            price_acc  = max(0.0, float((c.iloc[-1] - c.iloc[-3]) / c.iloc[-3]) * 100)
            vol_bonus  = 10.0 if vol_r > 1.3 else 0.0
            sec_bonus  = get_sector_bonus(symbol, sector_strengths, "LONG")
            score      = freshness * 2 + rsi_accel * 5 + price_acc + vol_bonus + sec_bonus

            if score > 10:
                sl, t1, t2 = calculate_trade_levels(cur, cur_atr, "LONG")
                long_signals.append({
                    "Symbol"     : symbol,
                    "Sector"     : SYMBOL_SECTOR.get(symbol, "—"),
                    "Score"      : round(score, 1),
                    "Entry"      : round(cur, 2),
                    "SL"         : sl,
                    "Target 1"   : t1,
                    "Target 2"   : t2,
                    "RSI"        : round(cur_rsi, 1),
                    "EMA20"      : round(float(ema20.iloc[-1]), 2),
                    "EMA50"      : round(float(ema50.iloc[-1]), 2),
                    "RS vs Nifty": rs_score,
                    "Vol Ratio"  : vol_r,
                    "Bias"       : "LONG",
                })

        # ── SHORT ─────────────────────────────────────────────────────
        if scanner_mode == "Fresh Crossovers":
            short_qual = (float(c.iloc[-2]) >= float(ema20.iloc[-2]) and
                          float(c.iloc[-1])  < float(ema20.iloc[-1]))
        else:
            short_qual = (cur < float(ema20.iloc[-1]) < float(ema50.iloc[-1]) and cur_rsi < 45)

        if short_qual:
            days_below     = int((c < ema20).tail(20).sum())
            freshness_short= max(0, 20 - days_below) if scanner_mode == "Fresh Crossovers" else 5
            rsi_accel_b    = max(0.0, float(rsi.iloc[-2]) - float(rsi.iloc[-1]))
            price_acc_b    = max(0.0, float((c.iloc[-3] - c.iloc[-1]) / c.iloc[-3]) * 100)
            vol_bonus      = 10.0 if vol_r > 1.3 else 0.0
            sec_bonus      = get_sector_bonus(symbol, sector_strengths, "SHORT")
            score          = freshness_short * 2 + rsi_accel_b * 5 + price_acc_b + vol_bonus + sec_bonus

            if score > 10:
                sl, t1, t2 = calculate_trade_levels(cur, cur_atr, "SHORT")
                short_signals.append({
                    "Symbol"     : symbol,
                    "Sector"     : SYMBOL_SECTOR.get(symbol, "—"),
                    "Score"      : round(score, 1),
                    "Entry"      : round(cur, 2),
                    "SL"         : sl,
                    "Target 1"   : t1,
                    "Target 2"   : t2,
                    "RSI"        : round(cur_rsi, 1),
                    "EMA20"      : round(float(ema20.iloc[-1]), 2),
                    "EMA50"      : round(float(ema50.iloc[-1]), 2),
                    "RS vs Nifty": rs_score,
                    "Vol Ratio"  : vol_r,
                    "Bias"       : "SHORT",
                })

    long_signals.sort( key=lambda x: x["Score"], reverse=True)
    short_signals.sort(key=lambda x: x["Score"], reverse=True)
    return {"long": long_signals[:10], "short": short_signals[:10]}


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE C — F&O MONEY FLOW  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_c(batch: dict) -> list:
    candidates = []
    for ticker in STOCK_TICKERS:
        df = extract_ticker_dataframe(batch, ticker)
        if df is None:
            continue

        symbol    = ticker.replace(".NS", "")
        c         = df["CLOSE"]; vol = df["VOLUME"]
        atr       = calculate_atr(df, 14)
        rsi       = calculate_wilder_rsi(c, 14)

        avg_vol   = float(vol.tail(20).mean())
        vol_ratio = round(float(vol.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 0
        price_acc = float((c.iloc[-1] - c.iloc[-4]) / c.iloc[-4] * 100)
        avg_atr   = float(atr.tail(20).mean())
        atr_exp   = float(atr.iloc[-1]) > avg_atr * 1.15
        cur_rsi   = float(rsi.iloc[-1])
        cur_atr   = float(atr.iloc[-1])
        cur       = float(c.iloc[-1])

        if vol_ratio >= 1.3 and atr_exp:
            if price_acc > 1.0 and 45 < cur_rsi < 80:
                mf_score   = round(vol_ratio * 15 + price_acc * 10 + cur_rsi * 0.2, 1)
                sl, t1, t2 = calculate_trade_levels(cur, cur_atr, "LONG")
                candidates.append({
                    "Symbol"     : symbol,
                    "Sector"     : SYMBOL_SECTOR.get(symbol, "—"),
                    "MF Score"   : mf_score,
                    "Entry"      : round(cur, 2),
                    "SL"         : sl,
                    "Target 1"   : t1,
                    "Target 2"   : t2,
                    "Vol Ratio"  : vol_ratio,
                    "Price Acc%" : round(price_acc, 2),
                    "RSI"        : round(cur_rsi, 1),
                    "ATR Expand" : "✅",
                    "Bias"       : "LONG",
                })
            elif price_acc < -1.0 and 20 < cur_rsi < 55:
                mf_score   = round(vol_ratio * 15 + abs(price_acc) * 10 + (100 - cur_rsi) * 0.2, 1)
                sl, t1, t2 = calculate_trade_levels(cur, cur_atr, "SHORT")
                candidates.append({
                    "Symbol"     : symbol,
                    "Sector"     : SYMBOL_SECTOR.get(symbol, "—"),
                    "MF Score"   : mf_score,
                    "Entry"      : round(cur, 2),
                    "SL"         : sl,
                    "Target 1"   : t1,
                    "Target 2"   : t2,
                    "Vol Ratio"  : vol_ratio,
                    "Price Acc%" : round(price_acc, 2),
                    "RSI"        : round(cur_rsi, 1),
                    "ATR Expand" : "✅",
                    "Bias"       : "SHORT",
                })

    candidates.sort(key=lambda x: x["MF Score"], reverse=True)
    return candidates[:20]


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE D — RELATIVE STRENGTH  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_d(batch: dict, nifty_df: pd.DataFrame) -> list:
    nc      = nifty_df["CLOSE"]
    n20d    = float((nc.iloc[-1] - nc.iloc[-21]) / nc.iloc[-21] * 100)
    n5d     = float((nc.iloc[-1] - nc.iloc[-6])  / nc.iloc[-6]  * 100)
    rs_list = []

    for ticker in STOCK_TICKERS:
        df = extract_ticker_dataframe(batch, ticker)
        if df is None:
            continue
        c      = df["CLOSE"]
        s20d   = float((c.iloc[-1] - c.iloc[-21]) / c.iloc[-21] * 100)
        s5d    = float((c.iloc[-1] - c.iloc[-6])  / c.iloc[-6]  * 100)
        rs_list.append({
            "Symbol"     : ticker.replace(".NS", ""),
            "Sector"     : SYMBOL_SECTOR.get(ticker.replace(".NS", ""), "—"),
            "RS 20d"     : round(s20d - n20d, 2),
            "RS 5d"      : round(s5d  - n5d,  2),
            "Stock 20d%" : round(s20d, 2),
            "RSI"        : round(float(calculate_wilder_rsi(c, 14).iloc[-1]), 1),
            "RS Bias"    : "LONG" if (s20d - n20d) > 0 else "SHORT",
        })

    rs_list.sort(key=lambda x: x["RS 20d"], reverse=True)
    return rs_list


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE E — EVENT RISK  ███
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def run_engine_e(symbols: list) -> dict:
    today   = datetime.today().date()
    horizon = today + timedelta(days=7)
    risk_map= {}

    for symbol in symbols:
        e_risk = d_risk = False
        e_date = d_date = "—"
        try:
            info = yf.Ticker(f"{symbol}.NS").calendar
            if info and hasattr(info, "get"):
                eq = info.get("Earnings Date")
                if eq:
                    for d in (eq if isinstance(eq, list) else [eq]):
                        try:
                            d_obj = pd.Timestamp(d).date()
                            if today <= d_obj <= horizon:
                                e_risk = True; e_date = str(d_obj); break
                        except Exception:
                            pass
                eq_div = info.get("Ex-Dividend Date")
                if eq_div:
                    try:
                        d_obj = pd.Timestamp(eq_div).date()
                        if today <= d_obj <= horizon:
                            d_risk = True; d_date = str(d_obj)
                    except Exception:
                        pass
        except Exception:
            pass
        risk_map[symbol] = {
            "earnings_risk": e_risk, "dividend_risk": d_risk,
            "earnings_date": e_date, "dividend_date": d_date,
            "flagged"      : e_risk or d_risk,
        }
    return risk_map


# ═════════════════════════════════════════════════════════════════════════════
# ███  ALPHA BOARD — COMPOSITE RANKED TRADE SHEET  ███
# ═════════════════════════════════════════════════════════════════════════════

def build_alpha_board(eng_b: dict, eng_c: list, risk_map: dict,
                      capital: float, risk_pct: float) -> pd.DataFrame:
    """
    Alpha Score = 0.60 × B_score + 0.40 × C_mf_score
    Position sizing: qty = floor((capital × risk%) / |entry − SL|)
    Dual targets from calculate_trade_levels() already embedded in B & C rows.
    """
    b_map = {r["Symbol"]: r for r in eng_b["long"] + eng_b["short"]}
    c_map = {r["Symbol"]: r for r in eng_c}
    overlap = set(b_map) & set(c_map)
    if not overlap:
        return pd.DataFrame()

    risk_per_trade = capital * risk_pct / 100
    rows = []

    for sym in overlap:
        b = b_map[sym]; c = c_map[sym]
        alpha   = round(0.60 * b["Score"] + 0.40 * c["MF Score"], 1)
        entry   = b["Entry"]; sl = b["SL"]
        risk_pt = abs(entry - sl)
        qty     = int(risk_per_trade / risk_pt) if risk_pt > 0 else 0
        flag    = "⚠️ Event Risk" if risk_map.get(sym, {}).get("flagged") else "✅ Clean"

        rows.append({
            "Rank"         : 0,
            "Symbol"       : sym,
            "Sector"       : b.get("Sector", "—"),
            "Bias"         : b["Bias"],
            "Alpha Score"  : alpha,
            "B Score"      : b["Score"],
            "C MF Score"   : c["MF Score"],
            "Entry"        : entry,
            "SL"           : sl,
            "Target 1"     : b["Target 1"],
            "Target 2"     : b["Target 2"],
            "Qty (risk)"   : qty,
            "RS vs Nifty"  : b.get("RS vs Nifty", "—"),
            "RSI"          : b.get("RSI", "—"),
            "Event Status" : flag,
        })

    rows.sort(key=lambda x: x["Alpha Score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["Rank"] = i
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# ███  SEBI FOOTER  ███
# ═════════════════════════════════════════════════════════════════════════════

def apply_sebi_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center;font-size:11px;color:#888;padding:6px 0;'>
        ⚖️ <b>SEBI Disclaimer</b> &nbsp;|&nbsp;
        For <b>educational and research purposes only</b>. Not investment advice.
        Derivatives carry substantial risk of loss. Consult a SEBI-registered advisor.
        &nbsp;|&nbsp; Data: Yahoo Finance (delayed). &nbsp;|&nbsp;
        <b>Not SEBI registered.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ███  STREAMLIT UI  ███
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <h1 style='font-family:monospace;letter-spacing:2px;margin-bottom:2px;'>
    🛡️ INSTITUTIONAL DERIVATIVES WORKSTATION
    <span style='font-size:15px;color:#888;'>&nbsp;v3.5</span>
    </h1>
    <p style='color:#aaa;font-size:12px;margin-top:0;'>
    5-Engine Alpha Discovery &nbsp;·&nbsp; NSE F&O Universe &nbsp;·&nbsp;
    Dual Targets (T1 partial exit / T2 full exit) &nbsp;·&nbsp;
    Engine A · B · C · D · E
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Scan Controls")
    run_btn = st.button("🚀 Run Full Alpha Scan", use_container_width=True, type="primary")

    st.markdown("---")
    st.subheader("🔍 Engine B Mode")
    scanner_mode = st.selectbox(
        "Scanner Mode",
        ["Fresh Crossovers", "Strong Trends"],
        help=(
            "Fresh Crossovers: EMA20 crossover in last 1 bar.\n"
            "Strong Trends: Close > EMA20 > EMA50 + RSI > 55."
        ),
    )

    st.markdown("---")
    st.subheader("💼 Position Sizing")
    capital = st.number_input(
        "Trading Capital (₹)", min_value=10_000, max_value=10_000_000,
        value=500_000, step=50_000, format="%d",
    )
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, 0.25)
    st.caption(f"Risk per trade: ₹{capital * risk_pct / 100:,.0f}")

    st.markdown("---")
    st.subheader("🛡️ Event Risk (Engine E)")
    run_event_filter = st.checkbox(
        "Enable Engine E (slower)",
        value=False,
        help="Makes one API call per candidate stock — adds ~30s to scan.",
    )

    st.markdown("---")
    st.subheader("📡 Data Source")
    st.info(
        "**Now:** Yahoo Finance (15-min cache)\n\n"
        "**Upgrade:**\n- Zerodha Kite Connect\n- Upstox API v2\n"
        "- Angel SmartAPI\n- Dhan API"
    )
    st.caption("⚠️ Research only. Not SEBI registered.")

# ── Main Scan ─────────────────────────────────────────────────────────────────
if run_btn:
    prog  = st.progress(0, text="📥 Downloading market data…")
    batch = download_all_market_data()
    prog.progress(15, text="✅ Data downloaded.")

    if not batch:
        st.error("❌ Download failed. Check internet connection.")
        st.stop()

    nifty_df     = extract_ticker_dataframe(batch, "^NSEI")
    banknifty_df = extract_ticker_dataframe(batch, "^NSEBANK")
    if nifty_df is None or banknifty_df is None:
        st.error("❌ Nifty/BankNifty unavailable. Yahoo may be throttling — retry in 60s.")
        st.stop()

    prog.progress(22, text="📊 Market breadth (EMA50)…")
    breadth = calculate_market_breadth(batch)

    prog.progress(32, text="🌐 Sector strength…")
    sector_strengths = calculate_sector_strength(batch)

    prog.progress(42, text="🔭 Engine A: Nifty Options…")
    eng_a = run_engine_a(nifty_df, banknifty_df, breadth)

    prog.progress(55, text=f"🔍 Engine B: {scanner_mode}…")
    eng_b = run_engine_b(batch, nifty_df, sector_strengths, scanner_mode)

    prog.progress(67, text="💰 Engine C: Money Flow…")
    eng_c = run_engine_c(batch)

    prog.progress(77, text="📈 Engine D: Relative Strength…")
    eng_d = run_engine_d(batch, nifty_df)

    risk_map = {}
    if run_event_filter:
        candidates = list(set([r["Symbol"] for r in eng_b["long"] + eng_b["short"] + eng_c]))
        prog.progress(85, text=f"🛡️ Engine E: Event risk ({len(candidates)} stocks)…")
        risk_map = run_engine_e(candidates)

    prog.progress(93, text="🏆 Building Alpha Board…")
    alpha_df = build_alpha_board(eng_b, eng_c, risk_map, capital, risk_pct)

    prog.progress(100, text="✅ Complete.")
    prog.empty()

    st.success(
        f"✅ Scan complete — {datetime.now().strftime('%d %b %Y  %H:%M:%S')}  |  "
        f"Breadth: {breadth}%  |  Capital: ₹{capital:,}  |  "
        f"Risk/trade: ₹{capital * risk_pct / 100:,.0f}"
    )
    st.divider()

    # ── ENGINE A ──────────────────────────────────────────────────────────
    with st.expander("🔭 Engine A — Nifty Options Direction", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Signal",          eng_a["signal"])
        c2.metric("Confidence",      f"{eng_a['confidence']}%")
        c3.metric("CE Met",          f"{eng_a['ce_met']}/6")
        c4.metric("PE Met",          f"{eng_a['pe_met']}/6")
        c5.metric("Breadth (EMA50)", f"{breadth}%")

        idx_rows = []
        for lbl, d in [("NIFTY 50", eng_a["nifty"]), ("BANKNIFTY", eng_a["banknifty"])]:
            idx_rows.append({
                "Index"    : lbl,  "Close": d["close"],
                "EMA5"     : d["ema5"],    "EMA20": d["ema20"], "EMA50": d["ema50"],
                "RSI"      : d["rsi"],     "ADX"  : d["adx"],
                "Prev High": d["prev_high"],"Prev Low": d["prev_low"],
                "> PH?"    : "✅" if d["above_prev_high"] else "❌",
                "< PL?"    : "✅" if d["below_prev_low"]  else "❌",
            })
        st.dataframe(pd.DataFrame(idx_rows), use_container_width=True, hide_index=True)

        ca, cp = st.columns(2)
        with ca:
            st.markdown(f"**CE — {eng_a['ce_met']}/6 met**")
            for cond, met in eng_a["ce_conditions"].items():
                st.markdown(f"{'✅' if met else '❌'} {cond}")
        with cp:
            st.markdown(f"**PE — {eng_a['pe_met']}/6 met**")
            for cond, met in eng_a["pe_conditions"].items():
                st.markdown(f"{'✅' if met else '❌'} {cond}")

    st.divider()

    # ── SECTOR STRENGTH ───────────────────────────────────────────────────
    with st.expander("🌐 Sector Strength — 5-Day Returns"):
        ss_rows = sorted(sector_strengths.items(), key=lambda x: x[1], reverse=True)
        ss_df   = pd.DataFrame(ss_rows, columns=["Sector", "5-Day Return %"])
        ss_df.insert(0, "Rank", range(1, len(ss_df) + 1))

        def color_sector(val):
            if isinstance(val, float):
                return "color:#00e676;" if val > 0 else "color:#ff5252;"
            return ""

        st.dataframe(
            ss_df.style.map(color_sector, subset=["5-Day Return %"]),
            use_container_width=True, hide_index=True, height=380,
        )

    st.divider()

    # ── ENGINE B ──────────────────────────────────────────────────────────
    with st.expander(f"🔍 Engine B — {scanner_mode}", expanded=True):
        b_cols = ["Symbol","Sector","Score","Entry","SL","Target 1","Target 2",
                  "RSI","EMA20","EMA50","RS vs Nifty","Vol Ratio","Bias"]
        tab_l, tab_s = st.tabs(["📈 Top 10 LONG", "📉 Top 10 SHORT"])
        with tab_l:
            if eng_b["long"]:
                st.dataframe(styled_df(pd.DataFrame(eng_b["long"])[b_cols]),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No long setups in current mode.")
        with tab_s:
            if eng_b["short"]:
                st.dataframe(styled_df(pd.DataFrame(eng_b["short"])[b_cols]),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No short setups in current mode.")

    st.divider()

    # ── ENGINE C ──────────────────────────────────────────────────────────
    with st.expander("💰 Engine C — F&O Money Flow", expanded=True):
        c_cols = ["Symbol","Sector","MF Score","Entry","SL","Target 1","Target 2",
                  "Vol Ratio","Price Acc%","RSI","ATR Expand","Bias"]
        if eng_c:
            st.dataframe(styled_df(pd.DataFrame(eng_c)[c_cols]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No money-flow signals today.")

    st.divider()

    # ── ENGINE D ──────────────────────────────────────────────────────────
    with st.expander("📈 Engine D — Institutional Relative Strength (20-day)"):
        d_cols = ["Symbol","Sector","RS 20d","RS 5d","Stock 20d%","RSI","RS Bias"]
        dt1, dt2 = st.tabs(["🏆 Top 15 RS Leaders", "🔻 Bottom 15 RS Laggards"])
        with dt1:
            top15 = eng_d[:15]
            if top15:
                st.dataframe(styled_df(pd.DataFrame(top15)[d_cols], "RS Bias"),
                             use_container_width=True, hide_index=True)
        with dt2:
            bot15 = eng_d[-15:][::-1]
            if bot15:
                st.dataframe(styled_df(pd.DataFrame(bot15)[d_cols], "RS Bias"),
                             use_container_width=True, hide_index=True)

    st.divider()

    # ── ENGINE E ──────────────────────────────────────────────────────────
    if run_event_filter and risk_map:
        with st.expander("🛡️ Engine E — Event Risk Filter"):
            flagged = [s for s, v in risk_map.items() if v["flagged"]]
            if flagged:
                st.warning(
                    f"**{len(flagged)} stocks** flagged for earnings/dividend within 7 days: "
                    + ", ".join(f"`{s}`" for s in sorted(flagged))
                )
                st.dataframe(
                    pd.DataFrame([{
                        "Symbol"        : s,
                        "Earnings Date" : risk_map[s]["earnings_date"],
                        "Dividend Date" : risk_map[s]["dividend_date"],
                        "Flagged"       : "⚠️ Yes",
                    } for s in flagged]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("No event risk in current candidate list.")
        st.divider()

    # ── ALPHA BOARD ───────────────────────────────────────────────────────
    st.subheader("🏆 Alpha Board — Ranked Trade Sheet")
    st.caption(
        "Stocks in **both Engine B ∩ Engine C**. "
        "Alpha Score = 60% Breakout + 40% Money Flow. "
        "T1 = partial exit (1:1.5 R:R) · T2 = full exit (1:2.5 R:R). "
        f"Sizing: ₹{capital:,} capital at {risk_pct}% risk."
    )

    if not alpha_df.empty:
        a_cols = ["Rank","Symbol","Sector","Bias","Alpha Score","B Score","C MF Score",
                  "Entry","SL","Target 1","Target 2","Qty (risk)","RS vs Nifty","RSI","Event Status"]
        st.dataframe(
            alpha_df[a_cols].style.map(color_signal, subset=["Bias"]),
            use_container_width=True, hide_index=True,
            height=min(60 + len(alpha_df) * 36, 520),
        )
        csv = alpha_df[a_cols].to_csv(index=False)
        st.download_button(
            "⬇️ Download Alpha Board CSV", data=csv,
            file_name=f"alpha_board_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No overlap between Engine B and Engine C today. "
            "Normal on range-bound sessions. Try **Strong Trends** mode or review engines above."
        )

else:
    st.markdown(
        """
        ### 👋 Welcome to v3.5

        Click **Run Full Alpha Scan** in the sidebar to begin.

        | Engine | Purpose |
        |--------|---------|
        | **A** | Nifty/BankNifty options — 4/6 conditions + ADX + prev-day H/L break |
        | **B** | Stock breakouts — Fresh Crossover or Strong Trend mode + sector bonus |
        | **C** | F&O money flow — relative volume ≥ 1.3×, ATR expansion |
        | **D** | Institutional RS — 20-day return vs Nifty |
        | **E** | Event risk — flag earnings & ex-div within 7 days (optional) |

        **Alpha Board** ranks B∩C overlap by composite score with:
        - **T1** partial exit at 1.5× risk (1:1.5 R:R)
        - **T2** full exit at 2.5× risk (1:2.5 R:R)
        - Position sizing at your chosen capital & risk%

        > Data: Yahoo Finance (free, 15-min cached).
        """,
        unsafe_allow_html=True,
    )

apply_sebi_footer()
