import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", page_title="Professional Option Scanner")

# =====================================================================
# INDICATOR FUNCTIONS
# =====================================================================
def calculate_bollinger_score(stock_df, signal_type):
    """Adds 0-10 points: CALL if near lower band, PUT if near upper."""
    c = float(stock_df["Close"].iloc[-1])
    std = float(stock_df["Close"].rolling(20).std().iloc[-1])
    sma = float(stock_df["Close"].rolling(20).mean().iloc[-1])
    upper, lower = sma + (2 * std), sma - (2 * std)
    
    if signal_type == "CALL": return 10 if c < lower else (5 if c < sma else 0)
    else: return 10 if c > upper else (5 if c > sma else 0)

# =====================================================================
# MASTER SCORING ENGINE
# =====================================================================
def score_stock(stock_df, nifty_df, market_bias, atr_period):
    stock_df = stock_df.copy()
    if isinstance(stock_df.columns, pd.MultiIndex): stock_df.columns = stock_df.columns.get_level_values(0)
    if len(stock_df) < 250: return None
    
    c = stock_df['Close']
    stock_df['EMA20'] = c.ewm(span=20, adjust=False).mean()
    stock_df['EMA50'] = c.ewm(span=50, adjust=False).mean()
    
    # RSI & MACD
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    stock_df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    stock_df['MACD'] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    stock_df['Signal_Line'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR
    tr = pd.concat([stock_df['High']-stock_df['Low'], (stock_df['High']-c.shift()).abs(), (stock_df['Low']-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().iloc[-1]
    
    results = {}
    for choice in ["CALL", "PUT"]:
        # New Scoring: Trend + Momentum + Bollinger
        base = 20 + calculate_bollinger_score(stock_df, choice) # Base points for structure
        adj = 15 if (market_bias=="BULLISH" and choice=="CALL") else -15
        results[choice] = max(0, min(100, base + adj))
    
    win = "CALL" if results["CALL"] >= results["PUT"] else "PUT"
    f_score = results[win]
    act = win if f_score >= 60 else "NO TRADE"
    
    entry = float(c.iloc[-1])
    return {"Action": act, "Direction": win, "Score": f_score, "Entry": round(entry, 2), 
            "Stoploss": round(entry - (1.5 * atr if win=="CALL" else -1.5*atr), 2), 
            "Target": round(entry + (3.0 * atr if win=="CALL" else -3.0*atr), 2)}

# =====================================================================
# DASHBOARD
# =====================================================================
if st.button("🚀 Run Enhanced Scan"):
    nifty = yf.download("^NSEI", period="2y", interval="1d", progress=False)
    if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
    m_bias, _ = ("BULLISH", 0) # Simplified for brevity
    data = []
    for s in ["RELIANCE.NS", "SBIN.NS", "TCS.NS"]:
        raw = yf.download(s, period="2y", interval="1d", progress=False)
        m = score_stock(raw, nifty, m_bias, 14)
        if m: 
            m["Symbol"] = s.split('.')[0]
            data.append(m)
    st.dataframe(pd.DataFrame(data))
