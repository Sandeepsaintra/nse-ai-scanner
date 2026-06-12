import streamlit as st
import pandas as pd
from nselib import equity_data

st.set_page_config(layout="wide", page_title="Professional NSE Scanner")

def fetch_nse_data(symbol):
    try:
        # nselib fetches official historical data
        # 'period' is not available here, we use dates
        df = equity_data.equity_history(symbol=symbol.split('.')[0], 
                                        series='EQ', 
                                        start_date='01-05-2026', 
                                        end_date='13-06-2026')
        if df is None or df.empty:
            return None
        # Convert columns to numeric for calculation
        df['CLOSE'] = pd.to_numeric(df['CLOSE_PRICE'])
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None

st.title("🛡️ Professional NSE Scanner (Powered by nselib)")

if st.button("🚀 Run Official NSE Scan"):
    watch_list = ["RELIANCE", "SBIN", "TCS"]
    results = []
    
    for s in watch_list:
        df = fetch_nse_data(s)
        if df is not None:
            price = df['CLOSE'].iloc[-1]
            results.append({"Symbol": s, "Close": price})
        
    if results:
        st.dataframe(pd.DataFrame(results))
    else:
        st.error("No data retrieved.")
