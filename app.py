import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Institutional NSE Analytics Engine")

# =====================================================================
# API GATEWAY LAYER (VERIFICATION SIGNALS)
# =====================================================================
def verify_yfinance_api():
    """Checks the health and uptime of the Yahoo Finance data stream."""
    try:
        # Fetching index proxy data to test raw connectivity
        df = yf.download("^NSEI", period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return "NO"
        return "YES"
    except Exception:
        return "NO"

def verify_nse_exchange_api():
    """Checks the health and uptime of the direct NSE settlement/data gateway."""
    try:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=10)
        
        # Test pull using a highly liquid reference asset (RELIANCE)
        df = capital_market.price_volume_data(
            symbol="RELIANCE", 
            from_date=start_dt.strftime('%d-%m-%Y'), 
            to_date=end_dt.strftime('%d-%m-%Y')
        )
        if df is None or df.empty:
            return "NO"
        return "YES"
    except Exception:
        return "NO"

# =====================================================================
# CORE HISTORICAL RETRIEVAL LAYER
# =====================================================================
def get_equity_history(symbol):
    """Fetches full analytical data array from the verified exchange gateway."""
    try:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=50) # Generates ~35 trading sessions
        
        df = capital_market.price_volume_data(
            symbol=symbol, 
            from_date=start_dt.strftime('%d-%m-%Y'), 
            to_date=end_dt.strftime('%d-%m-%Y')
        )
        if df is None or df.empty:
            return None
            
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Cross-build layout mapping matrix
        rename_map = {
            'CLOSE_PRICE': 'CLOSE', 'CLOSEPRICE': 'CLOSE',
            'HIGH_PRICE': 'HIGH', 'HIGHPRICE': 'HIGH',
            'LOW_PRICE': 'LOW', 'LOWPRICE': 'LOW'
        }
        df = df.rename(columns=rename_map)
        
        # Enforce technical float matrix types
        df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')
        df['HIGH'] = pd.to_numeric(df['HIGH'], errors='coerce')
        df['LOW'] = pd.to_numeric(df['LOW'], errors='coerce')
        
        return df.dropna(subset=['CLOSE', 'HIGH', 'LOW'])
    except Exception:
        return None

# =====================================================================
# QUANTITATIVE SCORING ENGINE
# =====================================================================
def analyze_asset(stock_df):
    """Processes pricing data to generate option strategies and volatility guards."""
    close_series = stock_df['CLOSE']
    current_price = float(close_series.iloc[-1])
    
    sma20 = float(close_series.rolling(min(20, len(close_series))).mean().iloc[-1])
    
    # ATR Volatility Core
    highs = stock_df['HIGH']
    lows = stock_df['LOW']
    atr = float((highs - lows).rolling(min(14, len(stock_df))).mean().iloc[-1])
    if atr <= 0: 
        atr = current_price * 0.01 # Volatility baseline override
        
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
# INTERFACE EXECUTION LAYER
# =====================================================================
st.title("🛡️ Institutional Multi-Layer Analytics Engine")

# --- STEP 1 & 2: VERIFICATION PANELS VIA LIVE STREAM STATUS ---
st.subheader("🌐 Core Channel Gateways STATUS")
col1, col2 = st.columns(2)

with col1:
    with st.spinner("Checking yfinance channel..."):
        yf_status = verify_yfinance_api()
    if yf_status == "YES":
        st.success("**FIRST ENTRY:** yfinance API Connection ➔ **YES**")
    else:
        st.error("**FIRST ENTRY:** yfinance API Connection ➔ **NO**")

with col2:
    with st.spinner("Checking Exchange data stream..."):
        # Note: In the Indian ecosystem context, the direct market price feed matches the NSE data pool
        nse_status = verify_nse_exchange_api()
    if nse_status == "YES":
        st.success("**SECOND ENTRY:** Direct Exchange Data Stream ➔ **YES**")
    else:
        st.error("**SECOND ENTRY:** Direct Exchange Data Stream ➔ **NO**")

st.markdown("---")

# --- STEP 3 & 4: CONDITIONAL LINKING PIPELINE ---
if yf_status == "YES" and nse_status == "YES":
    st.info("⚡ **System Status:** All production pipelines are linked. Quantitative Matrix Enabled.")
    
    if st.button("🚀 Run Institutional Matrix Scan"):
        watch_list = ["RELIANCE", "SBIN", "TCS", "INFY", "TATAMOTORS", "ITC"]
        compiled_matrix = []
        
        with st.spinner("Compiling asset vectors through structural data pool..."):
            for asset in watch_list:
                stock_data = get_equity_history(asset)
                if stock_data is not None and len(stock_data) > 0:
                    analysis = analyze_asset(stock_data)
                    analysis["Symbol"] = asset
                    compiled_matrix.append(analysis)
                    
            if compiled_matrix:
                df_display = pd.DataFrame(compiled_matrix)
                ordered_cols = ["Symbol", "Action", "Entry", "Stoploss", "Target 1", "Target 2"]
                st.dataframe(df_display[ordered_cols], use_container_width=True)
                st.success("Option Matrix calculated successfully from institutional feeds.")
            else:
                st.error("Matrix execution halted: Core data blocks missing variables.")
else:
    st.warning("🚨 **System Status:** Scanner disabled. One or more external data bridges are offline.")
