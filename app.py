import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(layout="wide")

# =====================================================================
# CORE ENGINE: FIXED & DEBUGGED
# =====================================================================
def get_stock_data(symbol):
    # Fetch data
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    
    # Debug: Check if data is empty
    if df.empty:
        return None
        
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required):
        return None
        
    return df.copy()

def run_scanner():
    watch_list = ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS"]
    results = []
    
    for s in watch_list:
        df = get_stock_data(s)
        if df is not None and len(df) > 50:
            # Simple calculation for test
            close = float(df['Close'].iloc[-1])
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            
            results.append({
                "Symbol": s.split('.')[0],
                "Price": round(close, 2),
                "Entry": round(close, 2),
                "Stoploss": round(close - (1.5 * atr), 2),
                "Target": round(close + (3.0 * atr), 2)
            })
        else:
            st.warning(f"Could not retrieve valid data for {s}")
            
    return pd.DataFrame(results)

# =====================================================================
# INTERFACE
# =====================================================================
st.title("🛡️ Institutional Engine")

if st.button("🚀 Run Scan"):
    df_final = run_scanner()
    if not df_final.empty:
        st.dataframe(df_final)
    else:
        st.error("Scanner returned no data. Check your internet connection or ticker symbols.")
