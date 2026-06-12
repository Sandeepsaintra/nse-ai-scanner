import streamlit as st
import yfinance as yf
import pandas as pd
from nselib import equity_data  # Ensure you have run: pip install nselib

st.set_page_config(layout="wide", page_title="Data Source Diagnostic")
st.title("🔍 Data Source Verification")

if st.button("Check API Connectivity"):
    # 1. Test yfinance
    st.write("---")
    st.subheader("Testing yfinance (Yahoo)")
    try:
        df_yf = yf.download("^NSEI", period="1mo", progress=False)
        if not df_yf.empty:
            st.success(f"yfinance is working. Received {len(df_yf)} rows.")
        else:
            st.error("yfinance returned an empty DataFrame.")
    except Exception as e:
        st.error(f"yfinance failed: {e}")

    # 2. Test nselib (Official NSE Source)
    st.subheader("Testing nselib (Official NSE)")
    try:
        # Fetching recent data
        df_nse = equity_data.equity_history(symbol='RELIANCE', series='EQ', 
                                            start_date='01-06-2026', end_date='12-06-2026')
        if not df_nse.empty:
            st.success(f"nselib is working. Received {len(df_nse)} rows.")
        else:
            st.error("nselib returned no data.")
    except Exception as e:
        st.error(f"nselib failed: {e}")
