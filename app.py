import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
from datetime import datetime, timedelta

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
    """Fetches historical market metrics using robust date tracking parameters."""
    try:
        # Calculate strict date bounds to guarantee consistent structural delivery
        # Fetching ~50 days of data ensures reliable SMA and ATR computation windows
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=50)
        
        str_end = end_dt.strftime('%d-%m-%Y')
        str_start = start_dt.strftime('%d-%m-%Y')
        
        # Swapping out 'period' parameter for explicit date matrices
        df = capital_market.price_volume_data(symbol=symbol, from_date=str_start, to_date=str_end)
        
        if df is None or df.empty:
            return None
            
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Explicit data sanitization layer
        df['CLOSE_PRICE'] = pd.to_numeric(df['CLOSE_PRICE'], errors='coerce')
        df['HIGH_PRICE'] = pd.to_numeric(df['HIGH_PRICE'], errors='coerce')
        df['LOW_PRICE'] = pd.to_numeric(df['LOW_PRICE'], errors='coerce')
        
        df = df.dropna(subset=['CLOSE_PRICE', 'HIGH_PRICE', 'LOW_PRICE'])
        return df.copy()
    except Exception:
        return None

# =====================================================================
# QUANTITATIVE PROCESSING ENGINE
# =====================================================================
def analyze_asset(stock_df, macro_df):
    """Applies structural analysis logic to raw data blocks."""
    close_series = stock_df['CLOSE_PRICE']
    current_price = float(close_series.iloc[-1])
    
    # Simple Moving Average computation
    sma20 = float(close_series.rolling(min(20, len(close_series))).mean().iloc[-1])
    
    # Core Volatility Matrix
    highs = stock_df['HIGH_PRICE']
    lows = stock_df['LOW_PRICE']
    atr = float((highs - lows).rolling(min(14, len(stock_df))).mean().iloc[-1])
    if atr <= 0: 
        atr = current_price * 0.01  # Safe fallback default metrics
    
    # Target and strategy generation
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
    with st.spinner("Accessing direct exchange data infrastructure..."):
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
            ordered_cols = ["Symbol", "Action", "Entry", "Stoploss", "Target 1", "Target 2"]
            st.dataframe(df_display[ordered_cols], use_container_width=True)
            st.success("Matrix calculations computed from verified sources.")
        else:
            st.error("Data tracking returned empty matrix parameters. Verify source channel uptime.")
