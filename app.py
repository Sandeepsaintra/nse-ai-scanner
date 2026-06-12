import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Enforce professional mobile-responsive baseline layout configuration
st.set_page_config(layout="wide", page_title="Institutional Derivatives Workstation")

# =====================================================================
# SYSTEM CONFIGURATION & DICTIONARIES
# =====================================================================
WATCH_LIST = ["RELIANCE", "SBIN", "TCS", "INFY", "TATAMOTORS", "ITC", "HDFCBANK", "ICICIBANK"]
TICKERS = ["^NSEI", "^NSEBANK"] + [f"{stock}.NS" for stock in WATCH_LIST]

# =====================================================================
# SEBI COMPLIANT DISCLAIMER
# =====================================================================
def apply_sebi_footer():
    st.markdown("---")
    st.caption(
        "⚖️ **SEBI-Compliant Regulatory Safe Harbor:** This output is generated via deterministic algorithmic analysis "
        "for educational and research purposes only. It does not constitute investment advice, financial planning, "
        "or speculative solicitation. Trading derivatives involves significant capital loss risk."
    )

# =====================================================================
# ARMORED PARALLELIZED BATCH DOWNLOAD & CLEANING LAYER
# =====================================================================
@st.cache_data(ttl=300)
def download_all_market_data():
    """Executes a single cached, multi-threaded batch request to eliminate API rate limiting."""
    try:
        df = yf.download(
            tickers=TICKERS,
            period="1y",
            group_by="ticker",
            progress=False,
            threads=True
        )
        return df
    except Exception:
        return None

def extract_ticker_dataframe(batch_df, ticker_symbol):
    """Safely extracts, flattens, and formats data frames from the multi-indexed batch block."""
    try:
        if batch_df is None or batch_df.empty:
            return None
        if not isinstance(batch_df.columns, pd.MultiIndex):
            return None
        if ticker_symbol not in batch_df.columns.get_level_values(0):
            return None
            
        df = batch_df[ticker_symbol].copy()
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        if not all(col in df.columns for col in required_cols):
            return None
            
        df = df.dropna(subset=['CLOSE'])
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'DATE'}, inplace=True)
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
        return df
    except Exception:
        return None

def calculate_wilder_rsi(series, period=14):
    """Implements true J. Welles Wilder exponential smoothing constants via alpha=1/N."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

# =====================================================================
# INTEGRATED SCORING & FACTOR METRIC LAYER
# =====================================================================
def process_trading_workstation_logic(stock_df, nifty_df, asset_name):
    close = stock_df['CLOSE']
    high = stock_df['HIGH']
    low = stock_df['LOW']
    vol = stock_df['VOLUME']
    current_price = float(close.iloc[-1])
    
    # --- TRUE INSTITUTIONAL ATR CALCULATION ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]
    if atr <= 0 or np.isnan(atr): 
        atr = current_price * 0.015
        
    # --- TECHNICAL DATA GENERATION ---
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    rsi_series = calculate_wilder_rsi(close, 14)
    rsi = rsi_series.iloc[-1]
    
    # --- DIRECT DIRECTIONAL BIAS ANALYSIS ---
    bias_direction = "🟢 BULLISH" if current_price > ema20 else "🔴 BEARISH"

    # --- INDICATIVE ENTRY / DEFENSIVE INVALIDATIONS MATRIX ---
    entry_buffer = current_price * 0.003
    entry_min = current_price - entry_buffer
    entry_max = current_price + entry_buffer
    
    stop_loss_dist = atr * 1.5
    if current_price > ema20:
        stop_loss = current_price - stop_loss_dist
        t1 = current_price + stop_loss_dist
        t2 = current_price + (stop_loss_dist * 2)
    else:
        stop_loss = current_price + stop_loss_dist
        t1 = current_price - stop_loss_dist
        t2 = current_price - (stop_loss_dist * 2)

    # --- POSITION SIZING CALCULATOR ---
    capital_limit_qty = int(100000 / current_price)
    risk_qty = int(1000 / (stop_loss_dist + 1e-9))
    suggested_equity_qty = max(1, min(capital_limit_qty, risk_qty))

    return {
        "Symbol": asset_name,
        "Current Price": round(current_price, 2),
        "Bias Trend": bias_direction,
        "Entry Min": round(entry_min, 2),
        "Entry Max": round(entry_max, 2),
        "Stop Loss": round(stop_loss, 2),
        "Target 1": round(t1, 2),
        "Target 2": round(t2, 2),
        "Suggested Qty": suggested_equity_qty,
        "Event Status": "🟢 Clean (No Event)"
    }

# =====================================================================
# UI LAYOUT FRAMEWORK (THE WORKSTATION FRONT-END)
# =====================================================================
st.title("🛡️ Professional Trader Workstation")

# Initialization step
batch_data = download_all_market_data()

# --- HEALTH MONITOR PANEL ---
with st.expander("🩺 System Infrastructure Diagnostics", expanded=False):
    yfin_health = "🟢 OPERATIONAL" if batch_data is not None and not batch_data.empty else "🔴 FETCH ERROR"
    nifty_raw = extract_ticker_dataframe(batch_data, "^NSEI") if batch_data is not None else None
    bn_raw = extract_ticker_dataframe(batch_data, "^NSEBANK") if batch_data is not None else None
    
    st.write(f"**Data Provider Integration (Yahoo Finance):** {yfin_health}")
    st.write(f"**Nifty 50 Buffer Status:** {'🟢 CONNECTED' if nifty_raw is not None else '🔴 DISCONNECTED'}")
    st.write(f"**Bank Nifty Buffer Status:** {'🟢 CONNECTED' if bn_raw is not None else '🔴 DISCONNECTED'}")

if batch_data is not None and nifty_raw is not None and bn_raw is not None:
    nifty_last = nifty_raw['CLOSE'].iloc[-1]
    nifty_ema = nifty_raw['CLOSE'].ewm(span=20, adjust=False).mean().iloc[-1]
    
    # --- CARD 1: MACRO SYSTEM REG
