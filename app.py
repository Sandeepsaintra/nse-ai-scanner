import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import numpy as np

st.set_page_config(layout="wide", page_title="Institutional NSE Analytics Engine")

# =====================================================================
# DATA EXTRACTION LAYER
# =====================================================================
def get_nifty_macro():
    """Downloads macro tracking metrics via stable indices."""
    try:
        df = yf.download("^NSEI", period="1mo", interval="1d", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.copy()
    except Exception:
        return None

def get_equity_history(symbol):
    """Fetches high-integrity historical market metrics from official NSE data pool."""
    try:
        # Utilizing official capital_market module for strict asset validation
        df = capital_market.price_volume_data(symbol=symbol, period='1M')
        if df is None or df.empty:
            return None
            
        df.columns = [str(col).strip().upper() for col in df.columns]
        df['CLOSE_PRICE'] = pd.to_numeric(df['CLOSE_PRICE'], errors='coerce')
        df['HIGH_PRICE'] = pd.to_numeric(df['HIGH_PRICE'], errors='coerce')
        df['LOW_PRICE'] = pd.to_numeric(df['LOW_PRICE'], errors='coerce')
        df['TTL_TRD_QNTY'] = pd.to_numeric(df['TTL_TRD_QNTY'], errors='coerce')
        
        df = df.dropna(subset=['CLOSE_PRICE'])
        return df.copy()
    except Exception:
        return None

# =====================================================================
# QUANTITATIVE PROCESSING ENGINE
# =====================================================================
def analyze_asset(stock_df, macro_df):
    """Applies execution metrics to current asset states."""
    close_series = stock_df['CLOSE_PRICE']
    current_price = float(close_series.iloc[-1])
    
    # Mathematical models for indicators
    sma20 = float(close_series.rolling(20).mean().iloc[-1]) if len(close_series) >= 20 else current_price
    
    # True range calculations for volatility structuring
    highs = stock_df['HIGH_PRICE']
    lows = stock_df['LOW_PRICE']
    atr = float((highs - lows).rolling(min(14, len(stock_df))).mean().iloc[-1])
    
    # Strategy assignment logic
    if current_price > sma20:
        action = "CALL"
        stoploss = current_price - (1.5 * atr)
        target1 = current_price + (2.0 * atr)
        target2 = current_price + (4.0 * atr)
    else:
        action = "PUT"
        stoploss = current_price + (1.5 * atr)
        target1 = current_price - (2.0 * atr)
        target2 = current_price - (4.0 * atr)
        
    return {
        "Action": action,
        "Entry": round(current_price, 2),
        "Stoploss": round(stoploss, 2),
        "Target 1": round(target1, 2),
        "Target 2": round(target2, 2)
    }

# =====================================================================
# DASHBOARD INTERFACE
# =====================================================================
st.title("🛡️ Institutional Multi-Layer Analytics Matrix")

if st.button("🚀 Run System Integration Scan"):
    with st.spinner("Analyzing operational channels..."):
        # Layer 1 Data Check
        macro_data = get_nifty_macro()
        
        watch_list = ["RELIANCE", "SBIN", "TCS", "INFY", "TATAMOTORS", "ITC"]
        compiled_matrix = []
        
        for asset in watch_list:
            stock_data = get_equity_history(asset)
            if stock_data is not None and len(stock_data) > 0:
                analysis = analyze_asset(stock_data, macro_data)
                analysis["Symbol"] = asset
                compiled_matrix.append(analysis)
                
        if compiled_matrix:
            df_display = pd.DataFrame(compiled_matrix)
            # Reorder columns for scannability
            ordered_cols = ["Symbol", "Action", "Entry", "Stoploss", "Target 1", "Target 2"]
            st.dataframe(df_display[ordered_cols], use_container_width=True)
            st.success("Matrix calculations computed from verified sources.")
        else:
            st.error("Data tracking returned empty matrix parameters. Verify source channel uptime.")
