import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# 1. Fundamental Cache (Runs only once a day)
@st.cache_data(ttl=86400)
def get_fundamental_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        roe = info.get('returnOnEquity', 0)
        d_e = info.get('debtToEquity', 0)
        # Quality Filter: ROE > 15% and Debt/Equity < 1.0
        return (roe > 0.15) and (d_e < 1.0)
    except:
        return False

# 2. Main Scanner Logic
def run_scan():
    nifty_50 = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "ITC.NS"]
    
    signals = []
    for ticker in nifty_50:
        # Check fundamentals first (will use cache if already run today)
        if not get_fundamental_score(ticker):
            continue
            
        df = yf.download(ticker, period="3mo", progress=False)
        if df.empty: continue
        
        # Flatten structure if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df['Close'].iloc[-1]
        ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        
        if close > ema20:
            signals.append({
                "Symbol": ticker.replace(".NS", ""),
                "Price": round(close, 2),
                "Stop Loss": round(close - (1.5 * atr), 2),
                "Target": round(close + (2.5 * atr), 2)
            })
    return signals

# 3. Streamlit Interface
st.title("🛡️ Institutional Workstation: Fundamental-Trend Hybrid")
if st.button("🚀 Run Optimized Scan"):
    with st.spinner("Analyzing fundamentals and technical trends..."):
        data = run_scan()
        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.warning("No fundamentally strong stocks found in an uptrend today.")
