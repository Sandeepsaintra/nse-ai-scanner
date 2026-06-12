import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def calculate_trade_levels(entry, atr, bias):
    # Dynamic risk based on 1.5x ATR
    risk = 1.5 * atr
    if bias == "LONG":
        return entry - risk, entry + (1.5 * risk), entry + (2.5 * risk)
    else:
        return entry + risk, entry - (1.5 * risk), entry - (2.5 * risk)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Institutional Derivatives Workstation")
st.title("🛡️ Institutional Derivatives Workstation v3.3")

# List of tickers to scan
tickers = ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AXISBANK.NS"]

if st.button("🚀 Run Dynamic Scan"):
    with st.spinner("Downloading and processing market data..."):
        # Fix: Using multi_level_index=False to avoid MultiIndex KeyErrors
        raw_data = yf.download(tickers, period="2mo", progress=False, multi_level_index=False)
        
        # Standardize columns to uppercase to avoid case-sensitivity issues
        raw_data.columns = [c.upper() for c in raw_data.columns]
        
        results = []
        
        # Process each ticker
        for ticker in tickers:
            # Select columns for this specific ticker
            # yfinance returns columns like 'OPEN', 'HIGH', etc., when multi_level_index=False
            df = raw_data.xs(ticker, axis=1, level=1) if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
            
            if len(df) < 20: continue
            
            # Calculate ATR
            high_low = df['HIGH'] - df['LOW']
            high_close = np.abs(df['HIGH'] - df['CLOSE'].shift())
            low_close = np.abs(df['LOW'] - df['CLOSE'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            atr = ranges.max(axis=1).rolling(14).mean().iloc[-1]
            
            # Simple Trend Filter
            ema20 = df['CLOSE'].ewm(span=20).mean().iloc[-1]
            close = df['CLOSE'].iloc[-1]
            
            if close > ema20:
                bias = "LONG"
                sl, t1, t2 = calculate_trade_levels(close, atr, bias)
                results.append({
                    "Symbol": ticker.replace(".NS", ""), 
                    "Bias": bias,
                    "Entry": round(close, 2), 
                    "SL": round(sl, 2),
                    "Target 1": round(t1, 2), 
                    "Target 2": round(t2, 2)
                })

    # Render Table
    if results:
        st.subheader("📋 Trade Recommendations")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.info("💡 Targets calculated using 14-day ATR volatility scaling.")
    else:
        st.warning("No signals found. Try adjusting the trend filter or ticker list.")

# 
