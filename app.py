import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Institutional Derivatives Workstation v3.0",
    page_icon="🛡️",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 90   # enough for EMA50 + 20-day RS

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

# Sector map — used by Engine B sector-strength bonus (Fix 5)
SECTOR_MAP = {
    "BANK"    : ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
                 "BANDHANBNK","FEDERALBNK","IDFCFIRSTB","AUBANK","PNB"],
    "IT"      : ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT"],
    "AUTO"    : ["MARUTI","TATAMOTORS","M&M","EICHERMOT","HEROMOTOCO","BAJAJ-AUTO",
                 "ASHOKLEY","TVSMOTOR","EXIDEIND","BALKRISIND","BHARATFORG"],
    "PHARMA"  : ["SUNPHARMA","DIVISLAB","CIPLA","DRREDDY","LUPIN","APOLLOHOSP","ZYDUSLIFE"],
    "METAL"   : ["JSWSTEEL","TATASTEEL","HINDALCO","SAIL","NATIONALUM","JINDALSTEL"],
    "ENERGY"  : ["RELIANCE","BPCL","ONGC","TATAPOWER","POWERGRID","NTPC","IOC","COALINDIA","PFC","RECLTD"],
    "DEFENCE" : ["BEL","HAL","BHARATFORG"],
    "FMCG"    : ["ITC","HINDUNILVR","NESTLEIND","BRITANNIA","GODREJCP","TATACONSUM","MARICO"],
    "INFRA"   : ["LT","ADANIPORTS","CONCOR","GMRINFRA","DLF","OBEROIRLTY","IRCTC"],
    "FINANCE" : ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","LICHSGFIN",
                 "HDFCLIFE","SBILIFE","PEL"],
    "CEMENT"  : ["ULTRACEMCO","SHREECEM","GRASIM"],
    "MISC"    : ["TITAN","ASIANPAINT","HAVELLS","BERGEPAINT","VOLTAS","MRF","ADANIENT",
                 "SRF","CUMMINSIND","TATACOMM","ZEEL"],
}

# Build reverse lookup: symbol → sector
SYMBOL_SECTOR = {}
for sec, syms in SECTOR_MAP.items():
    for s in syms:
        SYMBOL_SECTOR[s] = sec

INDEX_TICKERS  = ["^NSEI", "^NSEBANK"]
STOCK_TICKERS  = [f"{s}.NS" for s in STOCK_UNIVERSE]
ALL_TICKERS    = INDEX_TICKERS + STOCK_TICKERS


# ═════════════════════════════════════════════════════════════════════════════
# ███  QUANT UTILITIES  ███
# ═════════════════════════════════════════════════════════════════════════════

def calculate_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """True Wilder RSI using EWM with alpha = 1/period."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    h, l, c  = df["HIGH"], df["LOW"], df["CLOSE"]
    pc       = c.shift(1)
    tr       = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder).
    Returns ADX series; values > 25 indicate trending market.
    """
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
    adx      = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


# ─── Styling helper (Fix 8) ───────────────────────────────────────────────

def color_signal(val):
    """Green cell for LONG, red cell for SHORT, neutral otherwise."""
    if isinstance(val, str):
        if "LONG" in val:
            return "background-color:#083d08; color:#00e676; font-weight:bold;"
        if "SHORT" in val:
            return "background-color:#3d0808; color:#ff5252; font-weight:bold;"
        if "BUY CE" in val:
            return "background-color:#083d08; color:#00e676; font-weight:bold;"
        if "BUY PE" in val:
            return "background-color:#3d0808; color:#ff5252; font-weight:bold;"
        if "WATCH CE" in val:
            return "background-color:#2d2d00; color:#ffeb3b; font-weight:bold;"
        if "WATCH PE" in val:
            return "background-color:#2d1500; color:#ff9800; font-weight:bold;"
    return ""


def styled_df(df: pd.DataFrame, bias_col: str = "Bias") -> object:
    """Return a Styler with bias-column coloring."""
    if bias_col in df.columns:
        return df.style.map(color_signal, subset=[bias_col])
    return df.style


# ═════════════════════════════════════════════════════════════════════════════
# ███  DATA LAYER  ███
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def download_all_market_data() -> dict:
    """
    Batch-download 90 days of daily OHLCV for all tickers.
    Returns {ticker: DataFrame(OPEN, HIGH, LOW, CLOSE, VOLUME)}.
    """
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    raw = yf.download(
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


def extract_ticker_dataframe(batch: dict, ticker: str,
                              min_rows: int = 55) -> "pd.DataFrame | None":
    """Safe single-ticker extractor. Requires min_rows for EMA50 validity."""
    df = batch.get(ticker)
    if df is None or len(df) < min_rows:
        return None
    return df.copy()


# ═════════════════════════════════════════════════════════════════════════════
# ███  MARKET BREADTH  (Fix 2 — EMA50 based)  ███
# ═════════════════════════════════════════════════════════════════════════════

def calculate_market_breadth(batch: dict) -> float:
    """
    % of F&O stocks where Close > EMA50.
    EMA50 is less noisy than EMA20 — better institutional proxy.  [Fix 2]
    """
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
# ███  ENGINE A — NIFTY OPTIONS DIRECTION  (Fix 1 + ADX)  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_a(nifty_df: pd.DataFrame, banknifty_df: pd.DataFrame,
                 breadth: float) -> dict:
    """
    CE: needs >= 4/6 conditions.  Confidence = ce_met * 16.7  [Fix 1]
    PE: needs >= 4/6 conditions.
    Added ADX > 20 as 6th condition (trending market filter).
    """
    metrics = {}
    for label, df in [("NIFTY", nifty_df), ("BANKNIFTY", banknifty_df)]:
        c = df["CLOSE"]; h = df["HIGH"]; l = df["LOW"]
        ema5  = c.ewm(span=5,  adjust=False).mean()
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        rsi   = calculate_wilder_rsi(c, 14)
        adx   = calculate_adx(df, 14)

        metrics[label] = {
            "close"           : round(float(c.iloc[-1]), 2),
            "ema5"            : round(float(ema5.iloc[-1]),  2),
            "ema20"           : round(float(ema20.iloc[-1]), 2),
            "ema50"           : round(float(ema50.iloc[-1]), 2),
            "rsi"             : round(float(rsi.iloc[-1]),   1),
            "adx"             : round(float(adx.iloc[-1]),   1),
            "prev_high"       : round(float(h.iloc[-2]), 2),
            "prev_low"        : round(float(l.iloc[-2]), 2),
            "above_prev_high" : float(c.iloc[-1]) > float(h.iloc[-2]),
            "below_prev_low"  : float(c.iloc[-1]) < float(l.iloc[-2]),
            "ema_bull"        : float(ema5.iloc[-1]) > float(ema20.iloc[-1]),
            "ema_bear"        : float(ema5.iloc[-1]) < float(ema20.iloc[-1]),
            "trending"        : float(adx.iloc[-1]) > 20,
        }

    n  = metrics["NIFTY"]
    bn = metrics["BANKNIFTY"]

    ce_conditions = {
        "Nifty > Prev Day High"      : n["above_prev_high"],
        "BankNifty > Prev Day High"  : bn["above_prev_high"],
        "Nifty EMA5 > EMA20"         : n["ema_bull"],
        "Nifty RSI > 55"             : n["rsi"] > 55,
        "Market Breadth (EMA50) > 60%": breadth > 60,
        "Nifty ADX > 20 (trending)"  : n["trending"],
    }
    pe_conditions = {
        "Nifty < Prev Day Low"       : n["below_prev_low"],
        "BankNifty < Prev Day Low"   : bn["below_prev_low"],
        "Nifty EMA5 < EMA20"         : n["ema_bear"],
        "Nifty RSI < 45"             : n["rsi"] < 45,
        "Market Breadth (EMA50) < 40%": breadth < 40,
        "Nifty ADX > 20 (trending)"  : n["trending"],
    }

    ce_met = sum(ce_conditions.values())
    pe_met = sum(pe_conditions.values())

    # Fix 1: threshold relaxed to >= 4
    if ce_met >= 4:
        signal, color = "🟢 BUY CE", "green"
    elif pe_met >= 4:
        signal, color = "🔴 BUY PE", "red"
    else:
        signal, color = "⬛ NO TRADE — Await Confirmation", "gray"

    # Confidence scales with conditions met
    if "CE" in signal:
        confidence = min(100, int(ce_met * 100 / 6))
    elif "PE" in signal:
        confidence = min(100, int(pe_met * 100 / 6))
    else:
        confidence = max(int(ce_met * 100 / 6), int(pe_met * 100 / 6))

    return {
        "signal"       : signal,
        "confidence"   : confidence,
        "color"        : color,
        "nifty"        : n,
        "banknifty"    : bn,
        "breadth"      : breadth,
        "ce_conditions": ce_conditions,
        "pe_conditions": pe_conditions,
        "ce_met"       : ce_met,
        "pe_met"       : pe_met,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ███  SECTOR STRENGTH  (Fix 5)  ███
# ═════════════════════════════════════════════════════════════════════════════

def calculate_sector_strength(batch: dict) -> dict:
    """
    5-day return for each sector (market-cap proxy = equal weight).
    Returns {sector_name: return_pct}.
    """
    sector_returns: dict[str, list[float]] = {s: [] for s in SECTOR_MAP}
    for symbol in STOCK_UNIVERSE:
        ticker = f"{symbol}.NS"
        df = extract_ticker_dataframe(batch, ticker)
        if df is None or len(df) < 6:
            continue
        ret = float((df["CLOSE"].iloc[-1] - df["CLOSE"].iloc[-6]) / df["CLOSE"].iloc[-6] * 100)
        sec = SYMBOL_SECTOR.get(symbol)
        if sec and sec in sector_returns:
            sector_returns[sec].append(ret)

    return {
        sec: round(np.mean(v), 2) if v else 0.0
        for sec, v in sector_returns.items()
    }


def get_sector_bonus(symbol: str, sector_strengths: dict) -> float:
    """
    +10 if stock's sector is in top-3 by 5d return (long),
    +10 if stock's sector is in bottom-3 (short — implicitly handled by caller).
    """
    sec = SYMBOL_SECTOR.get(symbol)
    if sec is None:
        return 0.0
    sorted_secs = sorted(sector_strengths, key=sector_strengths.get, reverse=True)
    if sec in sorted_secs[:3]:
        return 10.0
    return 0.0


def get_sector_bonus_short(symbol: str, sector_strengths: dict) -> float:
    """Short bonus: stock's sector in bottom-3 by 5d return."""
    sec = SYMBOL_SECTOR.get(symbol)
    if sec is None:
        return 0.0
    sorted_secs = sorted(sector_strengths, key=sector_strengths.get)
    if sec in sorted_secs[:3]:
        return 10.0
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE B — BREAKOUTS + STRONG TREND  (Fix 3 + sector bonus)  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_b(batch: dict, nifty_df: pd.DataFrame,
                 sector_strengths: dict,
                 scanner_mode: str = "Fresh Crossovers") -> dict:
    """
    scanner_mode:
      'Fresh Crossovers' — EMA20 crossover in last 1 bar
      'Strong Trends'    — close > EMA20 > EMA50 AND RSI > 55
    """
    nifty_close = nifty_df["CLOSE"]
    nifty_20d   = float((nifty_close.iloc[-1] - nifty_close.iloc[-21]) / nifty_close.iloc[-21] * 100)

    long_signals  = []
    short_signals = []

    for ticker in STOCK_TICKERS:
        df = extract_ticker_dataframe(batch, ticker)
        if df is None:
            continue

        symbol = ticker.replace(".NS", "")
        c      = df["CLOSE"]
        vol    = df["VOLUME"]
        ema20  = c.ewm(span=20, adjust=False).mean()
        ema50  = c.ewm(span=50, adjust=False).mean()
        rsi    = calculate_wilder_rsi(c, 14)
        atr    = calculate_atr(df, 14)

        cur    = float(c.iloc[-1])
        cur_atr= float(atr.iloc[-1])
        cur_rsi= float(rsi.iloc[-1])
        avg_vol= float(vol.tail(20).mean())
        vol_r  = round(float(vol.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 0

        # 20-day RS vs Nifty
        stock_20d = float((c.iloc[-1] - c.iloc[-21]) / c.iloc[-21] * 100)
        rs_score  = round(stock_20d - nifty_20d, 2)

        # ── LONG qualifier ──────────────────────────────────────────
        if scanner_mode == "Fresh Crossovers":
            long_qual = (float(c.iloc[-2]) <= float(ema20.iloc[-2]) and
                         float(c.iloc[-1])  > float(ema20.iloc[-1]))
        else:  # Strong Trends  [Fix 3]
            long_qual = (cur > float(ema20.iloc[-1]) > float(ema50.iloc[-1]) and
                         cur_rsi > 55)

        if long_qual:
            days_above = int((c > ema20).tail(20).sum())
            freshness  = max(0, 20 - days_above) if scanner_mode == "Fresh Crossovers" else 5
            rsi_accel  = max(0.0, float(rsi.iloc[-1]) - float(rsi.iloc[-2]))
            price_acc  = max(0.0, float((c.iloc[-1] - c.iloc[-3]) / c.iloc[-3]) * 100)
            vol_bonus  = 10.0 if vol_r > 1.3 else 0.0
            sec_bonus  = get_sector_bonus(symbol, sector_strengths)

            score = freshness * 2 + rsi_accel * 5 + price_acc + vol_bonus + sec_bonus

            if score > 10:
                long_signals.append({
                    "Symbol"    : symbol,
                    "Sector"    : SYMBOL_SECTOR.get(symbol, "—"),
                    "Score"     : round(score, 1),
                    "Entry"     : round(cur, 2),
                    "SL"        : round(cur - 1.5 * cur_atr, 2),
                    "Target"    : round(cur + 3.0 * cur_atr, 2),
                    "RSI"       : round(cur_rsi, 1),
                    "EMA20"     : round(float(ema20.iloc[-1]), 2),
                    "EMA50"     : round(float(ema50.iloc[-1]), 2),
                    "RS vs Nifty": rs_score,
                    "Vol Ratio" : vol_r,
                    "Bias"      : "LONG",
                })

        # ── SHORT qualifier ─────────────────────────────────────────
        if scanner_mode == "Fresh Crossovers":
            short_qual = (float(c.iloc[-2]) >= float(ema20.iloc[-2]) and
                          float(c.iloc[-1])  < float(ema20.iloc[-1]))
        else:
            short_qual = (cur < float(ema20.iloc[-1]) < float(ema50.iloc[-1]) and
                          cur_rsi < 45)

        if short_qual:
            days_below     = int((c < ema20).tail(20).sum())
            freshness_short= max(0, 20 - days_below) if scanner_mode == "Fresh Crossovers" else 5
            rsi_accel_b    = max(0.0, float(rsi.iloc[-2]) - float(rsi.iloc[-1]))
            price_acc_b    = max(0.0, float((c.iloc[-3] - c.iloc[-1]) / c.iloc[-3]) * 100)
            vol_bonus      = 10.0 if vol_r > 1.3 else 0.0
            sec_bonus      = get_sector_bonus_short(symbol, sector_strengths)

            score = freshness_short * 2 + rsi_accel_b * 5 + price_acc_b + vol_bonus + sec_bonus

            if score > 10:
                short_signals.append({
                    "Symbol"    : symbol,
                    "Sector"    : SYMBOL_SECTOR.get(symbol, "—"),
                    "Score"     : round(score, 1),
                    "Entry"     : round(cur, 2),
                    "SL"        : round(cur + 1.5 * cur_atr, 2),
                    "Target"    : round(cur - 3.0 * cur_atr, 2),
                    "RSI"       : round(cur_rsi, 1),
                    "EMA20"     : round(float(ema20.iloc[-1]), 2),
                    "EMA50"     : round(float(ema50.iloc[-1]), 2),
                    "RS vs Nifty": rs_score,
                    "Vol Ratio" : vol_r,
                    "Bias"      : "SHORT",
                })

    long_signals.sort( key=lambda x: x["Score"], reverse=True)
    short_signals.sort(key=lambda x: x["Score"], reverse=True)
    return {"long": long_signals[:10], "short": short_signals[:10]}


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE C — F&O MONEY FLOW  (Fix 4 — relative volume 1.3×)  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_c(batch: dict) -> list:
    """
    Relative volume threshold lowered to 1.3× (captures more institutional flow).
    Scoring: vol_ratio*15 + price_acc*10 + rsi*0.2  [Fix 4]
    """
    candidates = []
    for ticker in STOCK_TICKERS:
        df = extract_ticker_dataframe(batch, ticker)
        if df is None:
            continue

        symbol  = ticker.replace(".NS", "")
        c       = df["CLOSE"]; vol = df["VOLUME"]
        atr     = calculate_atr(df, 14)
        rsi     = calculate_wilder_rsi(c, 14)

        avg_vol   = float(vol.tail(20).mean())
        cur_vol   = float(vol.iloc[-1])
        vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0

        price_acc = float((c.iloc[-1] - c.iloc[-4]) / c.iloc[-4] * 100)
        avg_atr   = float(atr.tail(20).mean())
        atr_exp   = float(atr.iloc[-1]) > avg_atr * 1.15
        cur_rsi   = float(rsi.iloc[-1])

        if vol_ratio >= 1.3 and atr_exp:
            if price_acc > 1.0 and 45 < cur_rsi < 80:
                mf_score = round(vol_ratio * 15 + price_acc * 10 + cur_rsi * 0.2, 1)
                candidates.append({
                    "Symbol"     : symbol,
                    "Sector"     : SYMBOL_SECTOR.get(symbol, "—"),
                    "MF Score"   : mf_score,
                    "Vol Ratio"  : vol_ratio,
                    "Price Acc%" : round(price_acc, 2),
                    "RSI"        : round(cur_rsi, 1),
                    "ATR Expand" : "✅",
                    "Bias"       : "LONG",
                })
            elif price_acc < -1.0 and 20 < cur_rsi < 55:
                mf_score = round(vol_ratio * 15 + abs(price_acc) * 10 + (100 - cur_rsi) * 0.2, 1)
                candidates.append({
                    "Symbol"     : symbol,
                    "Sector"     : SYMBOL_SECTOR.get(symbol, "—"),
                    "MF Score"   : mf_score,
                    "Vol Ratio"  : vol_ratio,
                    "Price Acc%" : round(price_acc, 2),
                    "RSI"        : round(cur_rsi, 1),
                    "ATR Expand" : "✅",
                    "Bias"       : "SHORT",
                })

    candidates.sort(key=lambda x: x["MF Score"], reverse=True)
    return candidates[:20]


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE D — INSTITUTIONAL RELATIVE STRENGTH  ███
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_d(batch: dict, nifty_df: pd.DataFrame) -> list:
    """
    20-day Relative Strength = Stock_return_20d − Nifty_return_20d.
    Top RS leaders often become the next breakout setups.
    Returns top 15 long RS + top 15 short RS (weakest).
    """
    nifty_close = nifty_df["CLOSE"]
    nifty_20d   = float((nifty_close.iloc[-1] - nifty_close.iloc[-21]) / nifty_close.iloc[-21] * 100)
    nifty_5d    = float((nifty_close.iloc[-1] - nifty_close.iloc[-6])  / nifty_close.iloc[-6]  * 100)

    rs_list = []
    for ticker in STOCK_TICKERS:
        df = extract_ticker_dataframe(batch, ticker)
        if df is None:
            continue
        symbol    = ticker.replace(".NS", "")
        c         = df["CLOSE"]
        s20d      = float((c.iloc[-1] - c.iloc[-21]) / c.iloc[-21] * 100)
        s5d       = float((c.iloc[-1] - c.iloc[-6])  / c.iloc[-6]  * 100)
        rs_20d    = round(s20d - nifty_20d, 2)
        rs_5d     = round(s5d  - nifty_5d,  2)
        rsi       = round(float(calculate_wilder_rsi(c, 14).iloc[-1]), 1)
        rs_list.append({
            "Symbol"    : symbol,
            "Sector"    : SYMBOL_SECTOR.get(symbol, "—"),
            "RS 20d"    : rs_20d,
            "RS 5d"     : rs_5d,
            "Stock 20d%": round(s20d, 2),
            "RSI"       : rsi,
            "RS Bias"   : "LONG" if rs_20d > 0 else "SHORT",
        })

    rs_list.sort(key=lambda x: x["RS 20d"], reverse=True)
    return rs_list   # caller slices top/bottom


# ═════════════════════════════════════════════════════════════════════════════
# ███  ENGINE E — EVENT RISK FILTER  ███
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def run_engine_e(symbols: list) -> dict:
    """
    Scrapes Yahoo Finance calendar for each symbol to flag:
      • Earnings within 5 trading days
      • Ex-dividend date within 5 trading days

    Returns {symbol: {"earnings_risk": bool, "dividend_risk": bool,
                       "earnings_date": str, "dividend_date": str}}

    NOTE: Yahoo Finance calendar data reliability varies.
    Treat as a soft filter, not a hard block.
    For production use, replace with Refinitiv / Bloomberg / BSE API.
    """
    today    = datetime.today().date()
    horizon  = today + timedelta(days=7)
    risk_map = {}

    for symbol in symbols:
        ticker = f"{symbol}.NS"
        e_risk = False
        d_risk = False
        e_date = "—"
        d_date = "—"
        try:
            info = yf.Ticker(ticker).calendar
            # yfinance returns a dict-like object
            if info is not None and not (isinstance(info, dict) and len(info) == 0):
                if hasattr(info, "get"):
                    eq = info.get("Earnings Date")
                    if eq:
                        # Can be list of Timestamps
                        dates = eq if isinstance(eq, list) else [eq]
                        for d in dates:
                            try:
                                d_obj = pd.Timestamp(d).date()
                                if today <= d_obj <= horizon:
                                    e_risk = True
                                    e_date = str(d_obj)
                                    break
                            except Exception:
                                pass
                    eq_div = info.get("Ex-Dividend Date")
                    if eq_div:
                        try:
                            d_obj = pd.Timestamp(eq_div).date()
                            if today <= d_obj <= horizon:
                                d_risk = True
                                d_date = str(d_obj)
                        except Exception:
                            pass
        except Exception:
            pass
        risk_map[symbol] = {
            "earnings_risk" : e_risk,
            "dividend_risk" : d_risk,
            "earnings_date" : e_date,
            "dividend_date" : d_date,
            "flagged"       : e_risk or d_risk,
        }
    return risk_map


def apply_event_risk_filter(signals: list, risk_map: dict) -> tuple[list, list]:
    """Split signals into clean (safe) and flagged (event risk)."""
    clean   = [s for s in signals if not risk_map.get(s["Symbol"], {}).get("flagged", False)]
    flagged = [s for s in signals if     risk_map.get(s["Symbol"], {}).get("flagged", False)]
    return clean, flagged


# ═════════════════════════════════════════════════════════════════════════════
# ███  ALPHA BOARD — RANKED COMPOSITE  (Fix 6 + Fix 7 position sizing)  ███
# ═════════════════════════════════════════════════════════════════════════════

def build_alpha_board(eng_b: dict, eng_c: list,
                      risk_map: dict, capital: float,
                      risk_pct: float) -> pd.DataFrame:
    """
    Composite score = 0.60 × B_score + 0.40 × C_mf_score  [Fix 6]
    Only stocks appearing in BOTH Engine B and Engine C qualify.
    Position sizing: qty = floor((capital * risk_pct/100) / |entry − SL|)  [Fix 7]
    Event-risk flagged stocks are marked but still included (soft filter).
    """
    b_map: dict[str, dict] = {}
    for row in eng_b["long"] + eng_b["short"]:
        b_map[row["Symbol"]] = row

    c_map: dict[str, dict] = {}
    for row in eng_c:
        c_map[row["Symbol"]] = row

    overlap = set(b_map.keys()) & set(c_map.keys())
    if not overlap:
        return pd.DataFrame()

    rows = []
    risk_per_trade = capital * (risk_pct / 100)

    for sym in overlap:
        b = b_map[sym]
        c = c_map[sym]

        b_score = b["Score"]
        c_score = c["MF Score"]
        alpha   = round(0.60 * b_score + 0.40 * c_score, 1)

        entry   = b["Entry"]
        sl      = b["SL"]
        target  = b["Target"]
        risk_pt = abs(entry - sl)
        qty     = int(risk_per_trade / risk_pt) if risk_pt > 0 else 0

        event   = risk_map.get(sym, {})
        flag    = "⚠️ Event Risk" if event.get("flagged") else "✅ Clean"

        rows.append({
            "Rank"         : 0,           # filled after sort
            "Symbol"       : sym,
            "Sector"       : b.get("Sector", "—"),
            "Bias"         : b["Bias"],
            "Alpha Score"  : alpha,
            "B Score"      : b_score,
            "C MF Score"   : c_score,
            "Entry"        : entry,
            "SL"           : sl,
            "Target"       : target,
            "Qty (1% risk)": qty,
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
        &nbsp;|&nbsp; Data: Yahoo Finance (delayed / best-effort). &nbsp;|&nbsp;
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
    🛡️ INSTITUTIONAL DERIVATIVES WORKSTATION &nbsp;<span style='font-size:16px;color:#888;'>v3.0</span>
    </h1>
    <p style='color:#aaa;font-size:12px;margin-top:0;'>
    5-Engine Alpha Discovery &nbsp;·&nbsp; NSE F&O Universe &nbsp;·&nbsp;
    Engine A: Options Direction &nbsp;·&nbsp; B: Breakouts &nbsp;·&nbsp;
    C: Money Flow &nbsp;·&nbsp; D: Relative Strength &nbsp;·&nbsp; E: Event Risk
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Scan Controls")

    run_btn = st.button("🚀 Run Full Alpha Scan", use_container_width=True, type="primary")

    st.markdown("---")
    st.subheader("🔍 Engine B Mode")       # Fix 3
    scanner_mode = st.selectbox(
        "Scanner Mode",
        ["Fresh Crossovers", "Strong Trends"],
        help=(
            "Fresh Crossovers: EMA20 crossover in last 1 bar (rarer, higher conviction).\n"
            "Strong Trends: Close > EMA20 > EMA50 + RSI > 55 (more signals, for trending names)."
        ),
    )

    st.markdown("---")
    st.subheader("💼 Position Sizing")     # Fix 7
    capital = st.number_input(
        "Trading Capital (₹)",
        min_value=10_000,
        max_value=10_000_000,
        value=500_000,
        step=50_000,
        format="%d",
    )
    risk_pct = st.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.25,
    )
    risk_amt = capital * risk_pct / 100
    st.caption(f"Risk per trade: ₹{risk_amt:,.0f}")

    st.markdown("---")
    st.subheader("🛡️ Event Risk Filter")
    run_event_filter = st.checkbox(
        "Enable Engine E (slower — 1 API call per stock)",
        value=False,
    )

    st.markdown("---")
    st.subheader("📡 Data Source")
    st.info(
        "**Current:** Yahoo Finance (15-min cache)\n\n"
        "**Upgrade path:**\n"
        "- Zerodha Kite Connect\n"
        "- Upstox API v2\n"
        "- Angel SmartAPI\n"
        "- Dhan API"
    )
    st.caption("⚠️ Research only. Not SEBI registered.")

# ── Main Scan ─────────────────────────────────────────────────────────────────
if run_btn:

    prog = st.progress(0, text="📥 Downloading market data…")
    batch = download_all_market_data()
    prog.progress(15, text="📥 Data downloaded.")

    if not batch:
        st.error("❌ Data download failed. Check internet connection.")
        st.stop()

    nifty_df     = extract_ticker_dataframe(batch, "^NSEI")
    banknifty_df = extract_ticker_dataframe(batch, "^NSEBANK")

    if nifty_df is None or banknifty_df is None:
        st.error("❌ Nifty / BankNifty data unavailable. Yahoo may be throttling — retry in 60s.")
        st.stop()

    prog.progress(20, text="📊 Calculating breadth (EMA50)…")
    breadth = calculate_market_breadth(batch)

    prog.progress(30, text="🌐 Calculating sector strength…")
    sector_strengths = calculate_sector_strength(batch)

    prog.progress(40, text="🔭 Engine A: Nifty Options Direction…")
    eng_a = run_engine_a(nifty_df, banknifty_df, breadth)

    prog.progress(55, text=f"🔍 Engine B: {scanner_mode}…")
    eng_b = run_engine_b(batch, nifty_df, sector_strengths, scanner_mode)

    prog.progress(68, text="💰 Engine C: Money Flow…")
    eng_c = run_engine_c(batch)

    prog.progress(78, text="📈 Engine D: Relative Strength…")
    eng_d = run_engine_d(batch, nifty_df)

    # Engine E (optional — slower)
    if run_event_filter:
        all_candidates = list(set(
            [r["Symbol"] for r in eng_b["long"] + eng_b["short"] + eng_c]
        ))
        prog.progress(85, text=f"🛡️ Engine E: Event risk ({len(all_candidates)} stocks)…")
        risk_map = run_engine_e(all_candidates)
    else:
        risk_map = {}

    prog.progress(93, text="🏆 Building Alpha Board…")
    alpha_df = build_alpha_board(eng_b, eng_c, risk_map, capital, risk_pct)

    prog.progress(100, text="✅ Scan complete.")
    prog.empty()

    st.success(f"✅ Scan complete — {datetime.now().strftime('%d %b %Y  %H:%M:%S')}  |  "
               f"Breadth: {breadth}%  |  Capital: ₹{capital:,}  |  Risk/trade: ₹{risk_amt:,.0f}")
    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # ENGINE A
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("🔭 Engine A — Nifty Options Direction  (40% weight)", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Signal",           eng_a["signal"])
        c2.metric("Confidence",       f"{eng_a['confidence']}%")
        c3.metric("CE Met",           f"{eng_a['ce_met']}/6")
        c4.metric("PE Met",           f"{eng_a['pe_met']}/6")
        c5.metric("Breadth (EMA50)",  f"{breadth}%")

        idx_rows = []
        for lbl, d in [("NIFTY 50", eng_a["nifty"]), ("BANKNIFTY", eng_a["banknifty"])]:
            idx_rows.append({
                "Index"     : lbl,
                "Close"     : d["close"],
                "EMA5"      : d["ema5"],
                "EMA20"     : d["ema20"],
                "EMA50"     : d["ema50"],
                "RSI"       : d["rsi"],
                "ADX"       : d["adx"],
                "Prev High" : d["prev_high"],
                "Prev Low"  : d["prev_low"],
                "> PH?"     : "✅" if d["above_prev_high"] else "❌",
                "< PL?"     : "✅" if d["below_prev_low"]  else "❌",
            })
        st.dataframe(pd.DataFrame(idx_rows), use_container_width=True, hide_index=True)

        ca, cp = st.columns(2)
        with ca:
            st.markdown(f"**CE Conditions — {eng_a['ce_met']}/6 met**")
            for cond, met in eng_a["ce_conditions"].items():
                st.markdown(f"{'✅' if met else '❌'} {cond}")
        with cp:
            st.markdown(f"**PE Conditions — {eng_a['pe_met']}/6 met**")
            for cond, met in eng_a["pe_conditions"].items():
                st.markdown(f"{'✅' if met else '❌'} {cond}")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # SECTOR STRENGTH  (Fix 5)
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("🌐 Sector Strength — 5-Day Returns"):
        ss_rows = sorted(sector_strengths.items(), key=lambda x: x[1], reverse=True)
        ss_df   = pd.DataFrame(ss_rows, columns=["Sector", "5-Day Return %"])
        ss_df["Rank"] = range(1, len(ss_df) + 1)

        def color_sector(val):
            if isinstance(val, float):
                if val > 0:  return "color:#00e676;"
                if val < 0:  return "color:#ff5252;"
            return ""

        st.dataframe(
            ss_df[["Rank","Sector","5-Day Return %"]].style.map(
                color_sector, subset=["5-Day Return %"]
            ),
            use_container_width=True, hide_index=True, height=350,
        )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # ENGINE B
    # ─────────────────────────────────────────────────────────────────────
    with st.expander(f"🔍 Engine B — {scanner_mode}  (35% weight)", expanded=True):
        b_cols = ["Symbol","Sector","Score","Entry","SL","Target",
                  "RSI","EMA20","EMA50","RS vs Nifty","Vol Ratio","Bias"]

        tab_long, tab_short = st.tabs(["📈 Top 10 LONG", "📉 Top 10 SHORT"])
        with tab_long:
            if eng_b["long"]:
                df_l = pd.DataFrame(eng_b["long"])[b_cols]
                st.dataframe(styled_df(df_l), use_container_width=True, hide_index=True)
            else:
                st.info("No long setups in current mode.")
        with tab_short:
            if eng_b["short"]:
                df_s = pd.DataFrame(eng_b["short"])[b_cols]
                st.dataframe(styled_df(df_s), use_container_width=True, hide_index=True)
            else:
                st.info("No short setups in current mode.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # ENGINE C
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("💰 Engine C — F&O Money Flow  (25% weight)", expanded=True):
        if eng_c:
            df_c = pd.DataFrame(eng_c)
            st.dataframe(
                styled_df(df_c, "Bias"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No money-flow signals. Try lowering vol-ratio threshold or checking for low-volume day.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # ENGINE D
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("📈 Engine D — Institutional Relative Strength (20-day)"):
        d_top15  = eng_d[:15]
        d_bot15  = eng_d[-15:][::-1]

        dtab1, dtab2 = st.tabs(["🏆 Top 15 RS Leaders", "🔻 Bottom 15 RS Laggards"])
        d_cols = ["Symbol","Sector","RS 20d","RS 5d","Stock 20d%","RSI","RS Bias"]
        with dtab1:
            if d_top15:
                st.dataframe(styled_df(pd.DataFrame(d_top15)[d_cols], "RS Bias"),
                             use_container_width=True, hide_index=True)
        with dtab2:
            if d_bot15:
                st.dataframe(styled_df(pd.DataFrame(d_bot15)[d_cols], "RS Bias"),
                             use_container_width=True, hide_index=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # ENGINE E
    # ─────────────────────────────────────────────────────────────────────
    if run_event_filter and risk_map:
        with st.expander("🛡️ Engine E — Event Risk Filter"):
            flagged_syms = [s for s, v in risk_map.items() if v["flagged"]]
            if flagged_syms:
                st.warning(
                    f"**{len(flagged_syms)} stocks flagged** for upcoming earnings / dividend events: "
                    + ", ".join(f"`{s}`" for s in sorted(flagged_syms))
                )
                e_rows = [
                    {
                        "Symbol"        : s,
                        "Earnings Date" : risk_map[s]["earnings_date"],
                        "Dividend Date" : risk_map[s]["dividend_date"],
                        "Flagged"       : "⚠️ Yes",
                    }
                    for s in flagged_syms
                ]
                st.dataframe(pd.DataFrame(e_rows), use_container_width=True, hide_index=True)
            else:
                st.success("No event risk detected in the current candidate list.")
        st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # 🏆  ALPHA BOARD  (Fix 6 + Fix 7)
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🏆 Alpha Board — Ranked Trade Sheet")
    st.caption(
        "Stocks confirmed by **both Engine B and Engine C**. "
        "Alpha Score = 60% Breakout + 40% Money Flow. "
        f"Position sizing based on ₹{capital:,} capital at {risk_pct}% risk per trade."
    )

    if not alpha_df.empty:
        display_cols = ["Rank","Symbol","Sector","Bias","Alpha Score",
                        "B Score","C MF Score","Entry","SL","Target",
                        "Qty (1% risk)","RS vs Nifty","RSI","Event Status"]
        st.dataframe(
            alpha_df[display_cols].style.map(color_signal, subset=["Bias"]),
            use_container_width=True,
            hide_index=True,
            height=min(60 + len(alpha_df) * 36, 500),
        )

        # Download button
        csv = alpha_df[display_cols].to_csv(index=False)
        st.download_button(
            label="⬇️ Download Alpha Board CSV",
            data=csv,
            file_name=f"alpha_board_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No stocks qualify on both Engine B and Engine C today. "
            "This is normal on low-volatility or range-bound sessions. "
            "Try **Strong Trends** mode in Engine B, or review individual engine outputs above."
        )

else:
    # ─── Landing page ────────────────────────────────────────────────────
    st.markdown(
        """
        ### 👋 Welcome to v3.0

        Click **Run Full Alpha Scan** in the sidebar.

        | Engine | Purpose | Weight |
        |--------|---------|--------|
        | **A** | Nifty / BankNifty options direction — relaxed 4/6 conditions + ADX filter | 40% |
        | **B** | Stock breakouts — Fresh Crossover **or** Strong Trend mode + sector bonus | 35% |
        | **C** | F&O money flow — relative volume ≥ 1.3×, ATR expansion | 25% |
        | **D** | Institutional RS — 20-day return vs Nifty, top / bottom leaders | — |
        | **E** | Event risk filter — flag earnings & ex-div within 7 days | — |

        **Alpha Board** ranks B∩C overlap by composite score with ATR-based SL/Target and
        1%-risk position sizing.

        > **Data:** Yahoo Finance (free, 15-min cached).
        > Upgrade to Zerodha Kite / Upstox / Dhan API for live NSE data.
        """,
        unsafe_allow_html=True,
    )

apply_sebi_footer()
