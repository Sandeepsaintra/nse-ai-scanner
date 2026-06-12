import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# 1. SETUP UI
st.set_page_config(layout="wide", page_title="Institutional Derivatives Workstation")
st.title("🛡️ Institutional Derivatives Workstation v3.4")

# 2. CALCULATION UTILITIES
def calculate_trade_levels(entry, atr, bias):
    risk = 1.5 * atr
    if bias == "LONG":
        return entry - risk, entry + (1.5 * risk), entry + (2.5 * risk)
    else:
        return entry + risk, entry - (1.5 * risk), entry - (2.5 * risk)

# 3. MAIN SCANNER
tickers = ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AXISBANK.NS"]

if st.button("🚀 Run Dynamic Scan"):
    with st.spinner("Downloading and processing market data..."):
        # Download data
        raw_data = yf.download(tickers, period="3mo", group_by='ticker', progress=False)
        
        if raw_data.empty:
            st.error("No data downloaded. Check your internet connection or ticker symbols.")
            st.stop()
            
        results = []
        
        # Iterate through each ticker's individual DataFrame
        for ticker in tickers:
            df = raw_data[ticker].copy()
            
            # Ensure columns are uppercase for safety
            df.columns = [c.upper() for c in df.columns]
            
            # Need at least 20 days for EMA/ATR
            if len(df) < 20: continue
            
            # Calculate ATR
            high_low = df['HIGH'] - df['LOW']
            high_close = np.abs(df['HIGH'] - df['CLOSE'].shift())
            low_close = np.abs(df['LOW'] - df['CLOSE'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            atr = ranges.max(axis=1).rolling(14).mean().iloc[-1]
            
            # Trend Filter
            ema20 = df['CLOSE'].ewm(span=20).mean().iloc[-1]
            close = df['CLOSE'].iloc[-1]
            
            # SIGNAL LOGIC
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

        # 4. DISPLAY RESULTS
        if results:
            st.subheader("📋 Trade Recommendations")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.success(f"Scan complete. Found {len(results)} potential setups.")
        else:
            st.warning("No signals found. Try adding more tickers or relaxing trend criteria.")

st.info("💡 Note: Targets are based on volatility-adjusted ATR levels.")
