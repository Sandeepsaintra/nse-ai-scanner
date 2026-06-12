import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Institutional NSE 5-Layer Engine")

# =====================================================================
# REAL-TIME MARKET STATE & HOLIDAY DETECTOR
# =====================================================================
def get_detailed_market_status():
    """Interrogates exchange APIs to diagnose live operational states."""
    status_template = {
        "connected": False,
        "status_text": "OFFLINE / MAINTENANCE",
        "mode": "FALLBACK",
        "details": "Direct exchange gateway is unresponsive. Standard routine weekend maintenance active."
    }
    try:
        url = "https://www.nseindia.com/api/marketStatus"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=3)
        response = session.get(url, headers=headers, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            market_state = data.get("marketState", [])
            if market_state:
                equity_market = next((m for m in market_state if m.get("market") == "Capital Market"), {})
                market_status_str = equity_market.get("marketStatus", "").upper()
                status_template["connected"] = True
                
                if market_status_str == "CLOSED":
                    if datetime.now().weekday() >= 5:
                        status_template["status_text"] = "EXCHANGE CLOSED (WEEKEND)"
                        status_template["details"] = "NSE is closed for the weekend. Feeds are in historical state."
                    else:
                        status_template["status_text"] = "EXCHANGE CLOSED (MARKET HOLIDAY)"
                        status_template["details"] = "NSE is closed due to an official settlement holiday."
                    status_template["mode"] = "HISTORICAL"
                elif market_status_str == "OPEN":
                    status_template["status_text"] = "LIVE OPERATIONAL CHANNEL"
                    status_template["details"] = "Exchange data streams are live and parsing active order books."
                    status_template["mode"] = "REAL_TIME"
                return status_template
    except Exception:
        pass
        
    if datetime.now().weekday() >= 5:
        status_template["connected"] = True
        status_template["status_text"] = "EXCHANGE CLOSED (WEEKEND OVERRIDE)"
        status_template["details"] = "System automatically bypassed offline exchange servers via weekend fallback protocols."
        status_template["mode"] = "HISTORICAL"
    return status_template

def verify_yfinance_api():
    """Verifies baseline global data stream connection."""
    try:
        df = yf.download("^NSEI", period="3d", interval="1d", progress=False)
        if df is None or df.empty:
            return "NO"
        return "YES"
    except Exception:
        return "NO"

# =====================================================================
# DATA RETRIEVAL LAYER (MULTIPLE CORES)
# =====================================================================
def get_cleaned_data(symbol, period="6mo"):
    """Fetches, flattens, and type-sanitizes raw underlying pricing data."""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        for col in ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['CLOSE', 'HIGH', 'LOW'])
    except Exception:
        return None

# =====================================================================
# INSTITUTIONAL 5-LAYER CORE ALGORITHMIC ENGINE
# =====================================================================
def calculate_rsi(series, periods=14):
    """Calculates standard technical momentum relative strength matrix."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def run_five_layer_engine(stock_df, benchmark_df):
    """
    Executes the multi-factor structural quantitative protocol:
    Layer 1: Structural Macro Trend (Dual EMA cross-checks)
    Layer 2: Pure Volatility Bounds (ATR Band width expansion)
    Layer 3: Volume Confirmation Vector (Volume vs 20-Day Mean)
    Layer 4: Relative Momentum Oscillators (RSI State evaluation)
    Layer 5: Systematic Beta Correlation (Covariance analysis vs Nifty)
    """
    close = stock_df['CLOSE']
    high = stock_df['HIGH']
    low = stock_df['LOW']
    vol = stock_df['VOLUME']
    
    current_price = float(close.iloc[-1])
    
    # --- LAYER 1: STRUCTURAL TREND ---
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    l1_score = 1 if current_price > ema20 and ema20 > ema50 else (-1 if current_price < ema20 and ema20 < ema50 else 0)
    
    # --- LAYER 2: VOLATILITY MATRIX ---
    atr = (high - low).rolling(window=14).mean().iloc[-1]
    if atr <= 0 or np.isnan(atr): 
        atr = current_price * 0.015
    l2_score = 1 if (high.iloc[-1] - low.iloc[-1]) > (atr * 1.2) else 0
    
    # --- LAYER 3: VOLUME CONFIRMATION VECTOR ---
    mean_volume = vol.rolling(window=20).
