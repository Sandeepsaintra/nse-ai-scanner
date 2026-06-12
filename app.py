import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC CALCULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_trade_levels(entry, atr, bias):
    """Dynamically calculates SL and Targets based on 1.5x/2.5x ATR."""
    risk = 1.5 * atr
    if bias == "LONG":
        return entry - risk, entry + (1.5 * risk), entry + (2.5 * risk)
    else:
        return entry + risk, entry - (1.5 * risk), entry - (2.5 * risk)

def get_dynamic_signals(batch):
    """
    Scans the universe and calculates dynamic levels for any stock 
    meeting the criteria.
    """
    signals = []
    for ticker in batch:
        df = batch[ticker]
        if len(df) < 20: continue
        
        # Dynamic ATR for volatility-based risk
        high_low = df['HIGH'] - df['LOW']
        high_close = np.abs(df['HIGH'] - df['CLOSE'].shift())
        low_close = np.abs(df['LOW'] - df['CLOSE'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(14).mean().iloc[-1]
        
        # Logic: If price > EMA20 (Simple Trend Filter for demo)
        ema20 = df['CLOSE'].ewm(span=20).mean().iloc[-1]
        close = df['CLOSE'].iloc[-1]
        
        if close > ema20:
            bias = "LONG"
            sl, t1, t2 = calculate_trade_levels(close, atr, bias)
            signals.append({
                "Symbol": ticker.replace(".NS", ""), "Bias": bias,
                "Entry": round(close, 2), "SL": round(sl, 2),
                "Target 1": round(t1, 2), "Target 2": round(t2, 2)
            })
    return signals

# ─────────────────────────────────────────────────────────────────────────────
# UI LAYER
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Institutional Workstation")
st.title("🛡️ Institutional Derivatives Workstation v3.3")

if st.button("🚀 Run Dynamic Scan"):
    # 1. Download Data
    tickers = ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    batch = {t: yf.download(t, period="1mo", progress=False) for t in tickers}
    
    # 2. Get Signals
    results = get_dynamic_signals(batch)
    
    if results:
        df = pd.DataFrame(results)
        st.subheader("📋 Trade Recommendations")
        # Visualizing the risk-reward setup
        st.dataframe(df, use_container_width=True)
        
        # Logic for visual risk management
        st.info("💡 Note: Targets are calculated based on 1.5x and 2.5x of the 14-day ATR.")
    else:
        st.warning("No dynamic signals found with current trend criteria.")

# 
