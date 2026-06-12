import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Enforce professional wide layout configuration
st.set_page_config(layout="wide", page_title="Institutional Derivatives Workstation")

# =====================================================================
# SYSTEM CONFIGURATION & AUTOMATED MARKET BASKET
# =====================================================================
# Dynamically tracking highly liquid institutional targets across sectors
DYNAMIC_MARKET_BASKET = [
    "RELIANCE", "SBIN", "TCS", "INFY", "TATAMOTORS", "ITC", "HDFCBANK", "ICICIBANK",
    "BHARTIARTL", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "MARUTI", "BAJFINANCE"
]
TICKERS = ["^NSEI", "^NSEBANK"] + [f"{stock}.NS" for stock in DYNAMIC_MARKET_BASKET]

# Initialize Professional Session State Buffers
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None

# =====================================================================
# SEBI COMPLIANT DISCLAIMER
# =====================================================================
def apply_sebi_footer():
    st.markdown("---")
    st.caption(
        "⚖️ **SEBI-Compliant Regulatory Safe Harbor:** This output is generated via deterministic algorithmic analysis "
        "for educational and research purposes only. It does not constitute investment advice, financial planning, "
        "or speculative solicitation. Trading derivatives involves significant capital loss risk."
    )

# =====================================================================
# ARMORED PARALLELIZED BATCH DOWNLOAD & CLEANING LAYER
# =====================================================================
@st.cache_data(ttl=120)  # Reduced cache TTL to 2 minutes for tighter market tracking
def download_all_market_data():
    """Executes a single cached, multi-threaded batch request to eliminate API rate limiting."""
    try:
        df = yf.download(
            tickers=TICKERS,
            period="1y",
            group_by="ticker",
            progress=False,
            threads=True
        )
        return df
    except Exception:
        return None

def extract_ticker_dataframe(batch_df, ticker_symbol):
    """Safely extracts, flattens, and formats data frames from the multi-indexed batch block."""
    try:
        if batch_df is None or batch_df.empty:
            return None
        if not isinstance(batch_df.columns, pd.MultiIndex):
            return None
        if ticker_symbol not in batch_df.columns.get_level_values(0):
            return None
            
        df = batch_df[ticker_symbol].copy()
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        if not all(col in df.columns for col in required_cols):
            return None
            
        df = df.dropna(subset=['CLOSE'])
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'DATE'}, inplace=True)
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
        return df
    except Exception:
        return None

def calculate_wilder_rsi(series, period=14):
    """Implements true J. Welles Wilder exponential smoothing constants via alpha=1/N."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

# =====================================================================
# AUTOMATED INDICATOR RECOGNITION & ALGORITHMIC MATCHING ENGINE
# =====================================================================
def process_trading_workstation_logic(stock_df, asset_name):
    close = stock_df['CLOSE']
    high = stock_df['HIGH']
    low = stock_df['LOW']
    current_price = float(close.iloc[-1])
    
    # --- CALCULATE INDICATORS ---
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    rsi_series = calculate_wilder_rsi(close, 14)
    current_rsi = rsi_series.iloc[-1]
    
    # --- INSTITUTIONAL ATR FOR PROTECTIVE BANDS ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]
    if atr <= 0 or np.isnan(atr): 
        atr = current_price * 0.015

    # --- STRICT SYSTEMATIC SIGNAL MATRICES ---
    # Buy Criterion: Price > EMA20 and RSI shows building long momentum without being overbought
    is_bullish_setup = (current_price > ema20) and (45 < current_rsi < 70)
    
    # Short Criterion: Price < EMA20 and RSI shows cascading short momentum without being oversold
    is_bearish_setup = (current_price < ema20) and (30 < current_rsi < 55)

    if not (is_bullish_setup or is_bearish_setup):
        return None  # Filtered out: Stock does not meet the necessary technical requirements

    # --- ASSIGN BIAS DIRECTION & PRICE METRICS ---
    if is_bullish_setup:
        bias_direction = "🟢 BUY / LONG"
        entry_buffer = current_price * 0.002
        entry_min = current_price - entry_buffer
        entry_max = current_price + entry_buffer
        stop_loss = current_price - (atr * 1.5)
        t1 = current_price + (atr * 1.5)
        t2 = current_price + (atr * 3.0)
    else:
        bias_direction = "🔴 SELL / SHORT"
        entry_buffer = current_price * 0.002
        entry_min = current_price + entry_buffer
        entry_max = current_price - entry_buffer
        stop_loss = current_price + (atr * 1.5)
        t1 = current_price - (atr * 1.5)
        t2 = current_price - (atr * 3.0)

    # --- POSITION SIZING CALCULATOR (₹1,00,000 Cap, Max ₹1,000 Risk) ---
    stop_loss_dist = abs(current_price - stop_loss)
    capital_limit_qty = int(100000 / current_price)
    risk_qty = int(1000 / (stop_loss_dist + 1e-9))
    suggested_equity_qty = max(1, min(capital_limit_qty, risk_qty))

    return {
        "Symbol": asset_name,
        "Current Price": round(current_price, 2),
        "Action Signal": bias_direction,
        "RSI (14)": round(current_rsi, 1),
        "Entry Min": round(entry_min, 2),
        "Entry Max": round(entry_max, 2),
        "Stop Loss": round(stop_loss, 2),
        "Target 1": round(t1, 2),
        "Target 2": round(t2, 2),
        "Suggested Qty": suggested_equity_qty
    }

# =====================================================================
# UI LAYOUT FRAMEWORK
# =====================================================================
st.title("🛡️ Institutional Derivatives Workstation")

# Global variable scope extraction outside layout UI expanders
batch_data = download_all_market_data()
nifty_raw = extract_ticker_dataframe(batch_data, "^NSEI") if batch_data is not None else None
bn_raw = extract_ticker_dataframe(batch_data, "^NSEBANK") if batch_data is not None else None

# --- INFRASTRUCTURE HEALTH MONITOR PANEL ---
with st.expander("🩺 System Infrastructure Diagnostics", expanded=False):
    yfin_health = "🟢 OPERATIONAL" if batch_data is not None and not batch_data.empty else "🔴 FETCH ERROR"
    st.write(f"**Data Provider Integration:** {yfin_health}")
    
    nifty_status = "🟢 CONNECTED" if nifty_raw is not None else "🔴 DISCONNECTED"
    bn_status = "🟢 CONNECTED" if bn_raw is not None else "🔴 DISCONNECTED"
    
    st.write(f"**Nifty 50 Buffer Status:** {nifty_status}")
    st.write(f"**Bank Nifty Buffer Status:** {bn_status}")

if batch_data is not None and nifty_raw is not None and bn_raw is not None:
    nifty_last = nifty_raw['CLOSE'].iloc[-1]
    nifty_ema = nifty_raw['CLOSE'].ewm(span=20, adjust=False).mean().iloc[-1]
    
    st.markdown("### 🌐 Market Status Regime")
    with st.container(border=True):
        regime_status = "🟢 BULLISH CONTINUATION" if nifty_last > nifty_ema else "🔴 BEARISH STRUCTURAL DOWNGRADE"
        st.write(f"**Market Regime (Nifty 50):** {regime_status}")
        st.write(f"• **Risk Limit Profile:** Capital Ceiling = ₹1,00,000 | Max Risk Limit Per Trade = 1% (₹1,000)")

    st.markdown("---")
    
    # --- SCAN TRIGGER WITH ALGORITHMIC SIGNAL MATCHING ---
    if st.button("🚀 Run Market Scanner (Filter Active Signals Only)", use_container_width=True):
        with st.spinner("Analyzing data frameworks for indicator requirements..."):
            processed_setups = []
            for stock in DYNAMIC_MARKET_BASKET:
                stock_raw = extract_ticker_dataframe(batch_data, f"{stock}.NS")
                if stock_raw is not None and len(stock_raw) >= 50:
                    setup_data = process_trading_workstation_logic(stock_raw, stock)
                    if setup_data is not None:  # Only append if indicators met requirements
                        processed_setups.append(setup_data)

            # Save calculations directly to session state
            if processed_setups:
                st.session_state.scan_results = pd.DataFrame(processed_setups)
            else:
                st.session_state.scan_results = pd.DataFrame()  # Empty dataframe if no setups pass
            st.session_state.last_scan_time = datetime.now()

    # --- PERSISTENT RENDERING MATRIX ---
    if st.session_state.scan_results is not None:
        if st.session_state.scan_results.empty:
            st.warning("⚠️ No stocks currently meet the indicator requirements for a high-probability Buy or Sell setup.")
        else:
            col_title, col_time = st.columns([3, 1])
            with col_title:
                st.markdown("### 📊 Active Tactical Execution Matrix")
            with col_time:
                if st.session_state.last_scan_time:
                    formatted_time = st.session_state.last_scan_time.strftime('%d-%b-%Y %H:%M:%S')
                    st.caption(f"⏱️ Last Scan: {formatted_time}")
            
            st.dataframe(
                st.session_state.scan_results,
                column_config={
                    "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
                    "RSI (14)": st.column_config.NumberColumn(format="%.1f"),
                    "Entry Min": st.column_config.NumberColumn(format="₹%.2f"),
                    "Entry Max": st.column_config.NumberColumn(format="₹%.2f"),
                    "Stop Loss": st.column_config.NumberColumn(format="₹%.2f"),
                    "Target 1": st.column_config.NumberColumn(format="₹%.2f"),
                    "Target 2": st.column_config.NumberColumn(format="₹%.2f"),
                    "Suggested Qty": st.column_config.NumberColumn(format="%d Shares")
                },
                hide_index=True,
                use_container_width=True
            )
else:
    st.error("Pipeline Failure: Unable to clear market matrices.")

apply_sebi_footer()
