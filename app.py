import streamlit as st
import pandas as pd
import yfinance as yf

# Nifty 50 Tickers
NIFTY_50 = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS"]

def get_signal(ticker):
    df = yf.download(ticker, period="3mo", progress=False)
    if df.empty: return None
    
    # Standardize Column Names
    df.columns = [c.upper() for c in df.columns]
    
    close = df['CLOSE'].iloc[-1]
    ema20 = df['CLOSE'].ewm(span=20).mean().iloc[-1]
    ema50 = df['CLOSE'].ewm(span=50).mean().iloc[-1]
    atr = (df['HIGH'] - df['LOW']).rolling(14).mean().iloc[-1]
    
    # Logic: CALL (Uptrend) vs PUT (Downtrend)
    if close > ema20 and ema20 > ema50:
        bias = "CALL (BUY)"
        sl = round(close - (1.5 * atr), 2)
        return {"Symbol": ticker.replace(".NS", ""), "Bias": bias, "Entry": round(close, 2), "SL": sl, "T1": round(close + (1.5*atr), 2), "T2": round(close + (2.5*atr), 2)}
    elif close < ema20 and ema20 < ema50:
        bias = "PUT (BUY)"
        sl = round(close + (1.5 * atr), 2)
        return {"Symbol": ticker.replace(".NS", ""), "Bias": bias, "Entry": round(close, 2), "SL": sl, "T1": round(close - (1.5*atr), 2), "T2": round(close - (2.5*atr), 2)}
    return None

st.title("🛡️ Institutional Signal Generator")
if st.button("🚀 Scan for Calls & Puts"):
    results = []
    for ticker in NIFTY_50:
        sig = get_signal(ticker)
        if sig: results.append(sig)
    
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("No stocks currently satisfy the Call/Put trend criteria.")
