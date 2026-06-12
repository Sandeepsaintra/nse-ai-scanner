import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Enforce professional wide layout configuration
st.set_page_config(layout="wide", page_title="Institutional Derivatives Workstation")

# =====================================================================
# EXPANDED INDEX CONSTITUENTS (NIFTY 50 REPRESENTATIVE BASKET)
# =====================================================================
EXPANDED_MARKET_UNIVERSE = [
    "RELIANCE", "SBIN", "TCS", "INFY", "TATAMOTORS", "ITC", "HDFCBANK", "ICICIBANK",
    "BHARTIARTL", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "MARUTI", "BAJFINANCE",
    "M&M", "SUNPHARMA", "ULTRACEMCO", "POWERGRID", "NTPC", "TITAN", "ADANIENT", 
    "JSWSTEEL", "TATASTEEL", "HINDALCO", "COALINDIA", "BPCL", "ONGC", "GRASIM",
    "TECHM", "WIPRO", "HCLTECH", "APOLLOHOSP", "DIVISLAB", "CIPLA", "DRREDDY",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "INDUSINDBK", "BAJAJFINSV", "BRITANNIA",
    "NESTLEIND", "ASIANPAINT", "ADANIPORTS", "HDFCLIFE", "SBILIFE", "BEL", "HAL"
]
TICKERS = ["^NSEI", "^NSEBANK"] + [f"{stock}.NS" for stock in EXPANDED_MARKET_UNIVERSE]

# Initialize Session State Buffers
if "long_setups" not in st.session_state:
    st.session_state.long_setups = None
if "short_setups" not in st.session_state:
    st.session_state.short_setups = None
if "index_bias" not in st.session_state:
    st.session_state.index_bias = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "telemetry" not in st.session_state:
    st.session_state.telemetry = {"scanned": 0, "breadth": 50.0}

# =====================================================================
# SIDEBAR CONTROL RADAR
# =====================================================================
st.sidebar.title("🛠️ Momentum Control Panel")
debug_mode = st.sidebar.checkbox("🪲 Enable Diagnostic Debug Mode", value=False)
trigger_mode = st.sidebar.selectbox(
    "🎯 Stock Signal Isolation Mode",
    ["Fresh Crossovers Only", "Persistent Strong Trends"],
    index=0
)
score_threshold = 45.0 if debug_mode else 70.0

# =====================================================================
# MATHEMATICAL HELPER LAYER
# =====================================================================
def calculate_wilder_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

@st.cache_data(ttl=120)
def download_all_market_data():
    try:
        df = yf.download(tickers=TICKERS, period="1y", group_by="ticker", progress=False, threads=True)
        return df
    except Exception:
        return None

def extract_ticker_dataframe(batch_df, ticker_symbol):
    try:
        if batch_df is None or batch_df.empty: return None
        df = batch_df[ticker_symbol].copy()
        df.columns = [str(col).strip().upper() for col in df.columns]
        df = df.dropna(subset=['CLOSE']).reset_index()
        df.rename(columns={df.columns[0]: 'DATE'}, inplace=True)
        return df
    except Exception:
        return None

# =====================================================================
# NIFTY DIRECTION & OPTIONS STRATEGY ENGINE
# =====================================================================
def compute_nifty_options_bias(nifty_df, breadth_pct):
    close = nifty_df['CLOSE']
    high = nifty_df['HIGH']
    low = nifty_df['LOW']
    
    current_price = float(close.iloc[-1])
    today_high = float(high.iloc[-1])
    today_low = float(low.iloc[-1])
    
    # Technical Vectors
    ema20_series = close.ewm(span=20, adjust=False).mean()
    ema20 = ema20_series.iloc[-1]
    ema20_prev = ema20_series.iloc[-2]
    ema_slope = "RISING" if ema20 > ema20_prev else "FALLING"
    
    rsi_series = calculate_wilder_rsi(close, 14)
    current_rsi = rsi_series.iloc[-1]
    
    # Wilder Smoothed ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/14, adjust=False).mean().iloc[-1]

    # Weighted Confidence Matrix Calculations (0-100 pts)
    conf_score = 0.0
    
    # 1. Trend Factor (35%)
    if current_price > ema20 and ema_slope == "RISING": conf_score += 35
    elif current_price < ema20 and ema_slope == "FALLING": conf_score += 35
    else: conf_score += 15
    
    # 2. RSI Momentum Factor (25%)
    if (current_price > ema20 and current_rsi > 55) or (current_price < ema20 and current_rsi < 45): conf_score += 25
    else: conf_score += 10
    
    # 3. Market Breadth Factor (40%)
    if current_price > ema20:
        conf_score += (breadth_pct / 100.0) * 40
    else:
        conf_score += ((100.0 - breadth_pct) / 100.0) * 40

    # Operational Logic Boundaries
    if current_price > ema20 and current_rsi > 52 and breadth_pct >= 55.0:
        signal = "🟢 BULLISH"
        strategy = "🎯 BUY ATM / 1-STRIKE OTM CE (Call Option)"
        risk_profile = "Low" if breadth_pct > 70.0 else "Moderate (Awaiting Breadth Expansion)"
        entry_zone = f"Above Today's High (₹{round(today_high + 5, 2)})"
        invalidation = f"Below Daily EMA20 (₹{round(ema20, 2)})"
        targets = f"1R: ₹{round(current_price + (atr*1.5), 2)} | 2R: ₹{round(current_price + (atr*3.0), 2)}"
    elif current_price < ema20 and current_rsi < 48 and breadth_pct <= 45.0:
        signal = "🔴 BEARISH"
        strategy = "🎯 BUY ATM / 1-STRIKE OTM PE (Put Option)"
        risk_profile = "Low" if breadth_pct < 30.0 else "Moderate (Awaiting Structural Decay)"
        entry_zone = f"Below Today's Low (₹{round(today_low - 5, 2)})"
        invalidation = f"Above Daily EMA20 (₹{round(ema20, 2)})"
        targets = f"1R: ₹{round(current_price - (atr*1.5), 2)} | 2R: ₹{round(current_price - (atr*3.0), 2)}"
    else:
        signal = "🟡 NEUTRAL / CHOPPY"
        strategy = "🛑 CASH / SPREADS ONLY (High Intraday Theta Decay Risk)"
        risk_profile = "EXTREMELY HIGH FOR OPTION BUYERS"
        entry_zone = "No Optimal Long/Short Directional Entry Trigger"
        invalidation = "N/A"
        targets = "N/A"

    return {
        "Signal": signal,
        "Confidence": f"{round(conf_score, 1)} / 100",
        "Preferred Strategy": strategy,
        "Risk Profile": risk_profile,
        "Entry Zone Trigger": entry_zone,
        "Invalidation Floor": invalidation,
        "Target Matrices": targets
    }

# =====================================================================
# STOCK SCORING MACHINE ENGINE
# =====================================================================
def process_stock_scoring_logic(stock_df, nifty_df, asset_name, threshold, mode, market_regime):
    close = stock_df['CLOSE']
    high = stock_df['HIGH']
    low = stock_df['LOW']
    vol = stock_df['VOLUME']
    current_price = float(close.iloc[-1])
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/14, adjust=False).mean().iloc[-1]

    ema20_series = close.ewm(span=20, adjust=False).mean()
    ema20 = ema20_series.iloc[-1]
    rsi_series = calculate_wilder_rsi(close, 14)
    current_rsi = rsi_series.iloc[-1]
    prev_rsi = rsi_series.iloc[-2]
    
    avg_volume = vol.rolling(20).mean().iloc[-1]
    current_volume = vol.iloc[-1]

    # Signal Isolation Mapping
    if mode == "Fresh Crossovers Only":
        bullish_trigger = (close.iloc[-2] <= ema20_series.iloc[-2]) and (close.iloc[-1] > ema20_series.iloc[-1])
        bearish_trigger = (close.iloc[-2] >= ema20_series.iloc[-2]) and (close.iloc[-1] < ema20_series.iloc[-1])
    else:
        bullish_trigger = (current_price > ema20)
        bearish_trigger = (current_price < ema20)

    if not (bullish_trigger or bearish_trigger): return None
    asset_bias = "BULLISH" if bullish_trigger else "BEARISH"

    # Multi-Factor Score Assembly
    f_trend = 30.0 if mode == "Fresh Crossovers Only" else min(30.0, (abs(current_price - ema20)/ema20)*300)
    f_rsi = 25.0 if ((asset_bias == "BULLISH" and 45 < current_rsi < 70 and current_rsi > prev_rsi) or 
                     (asset_bias == "BEARISH" and 30 < current_rsi < 55 and current_rsi < prev_rsi)) else 5.0
    f_vol = min(20.0, (current_volume/(avg_volume+1e-9))*10.0) if current_volume > avg_volume else 0.0
    
    # Relative Strength Space Calculator
    stock_perf = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
    nifty_perf = (nifty_df['CLOSE'].iloc[-1] - nifty_df['CLOSE'].iloc[-5]) / nifty_df['CLOSE'].iloc[-5]
    f_rs = 15.0 if (asset_bias == "BULLISH" and stock_perf > nifty_perf) or (asset_bias == "BEARISH" and stock_perf < nifty_perf) else 0.0
    
    composite_score = f_trend + f_rsi + f_vol + f_rs
    if composite_score < threshold: return None

    # Risk Target Calculations
    is_bull = (asset_bias == "BULLISH")
    action = "🟢 BUY / LONG" if is_bull else "🔴 SELL / SHORT"
    stop_loss = current_price - (atr * 1.5) if is_bull else current_price + (atr * 1.5)
    t1 = current_price + (atr * 1.5) if is_bull else current_price - (atr * 1.5)
    
    qty = max(1, min(int(100000/current_price), int(1000/(abs(current_price - stop_loss)+1e-9))))

    return {
        "Symbol": asset_name, "Signal Score": round(composite_score, 1), "Action Signal": action,
        "Current Price": round(current_price, 2), "RSI (14)": round(current_rsi, 1),
        "Vol Surge": f"{round((current_volume / avg_volume), 2)}x", "Stop Loss": round(stop_loss, 2),
        "Target 1": round(t1, 2), "Suggested Qty": qty, "Bias": asset_bias
    }

# =====================================================================
# UI LAYOUT FRAMEWORK
# =====================================================================
st.title("🛡️ Institutional Derivatives Workstation")

batch_data = download_all_market_data()
nifty_raw = extract_ticker_dataframe(batch_data, "^NSEI") if batch_data is not None else None
bn_raw = extract_ticker_dataframe(batch_data, "^NSEBANK") if batch_data is not None else None

if batch_data is not None and nifty_raw is not None and bn_raw is not None:
    
    # --- AUTOMATED WORKFLOW SCAN EXECUTION TRIGGER ---
    if st.button("🚀 Run Integrated Index & Broad Market Quant Scan", use_container_width=True):
        with st.spinner("Analyzing broad market frameworks and multi-factor models..."):
            stock_setups = []
            total_scanned = 0
            bullish_breadth_count = 0
            
            # Phase 1: Scan Broad Market Universe to Compile Breadth Telemetry
            for stock in EXPANDED_MARKET_UNIVERSE:
                stock_raw = extract_ticker_dataframe(batch_data, f"{stock}.NS")
                if stock_raw is not None and len(stock_raw) >= 50:
                    total_scanned += 1
                    s_close = stock_raw['CLOSE'].iloc[-1]
                    s_ema = stock_raw['CLOSE'].ewm(span=20, adjust=False).mean().iloc[-1]
                    if s_close > s_ema:
                        bullish_breadth_count += 1
                        
                    # Evaluate individual alpha scoring metrics
                    nifty_regime = "BULLISH" if nifty_raw['CLOSE'].iloc[-1] > nifty_raw['CLOSE'].ewm(span=20, adjust=False).mean().iloc[-1] else "BEARISH"
                    setup = process_stock_scoring_logic(stock_raw, nifty_raw, stock, score_threshold, trigger_mode, nifty_regime)
                    if setup is not None: stock_setups.append(setup)

            # Calculate True Structural Index Breadth
            breadth_percentage = (bullish_breadth_count / total_scanned) * 100 if total_scanned > 0 else 50.0
            st.session_state.telemetry = {"scanned": total_scanned, "breadth": breadth_percentage}
            
            # Phase 2: Compute Core Nifty Option Decision Vectors
            st.session_state.index_bias = compute_nifty_options_bias(nifty_raw, breadth_percentage)
            
            # Phase 3: Slice and Rank Stock Data Blocks
            if stock_setups:
                df_universe = pd.DataFrame(stock_setups)
                st.session_state.long_setups = df_universe[df_universe["Bias"] == "BULLISH"].sort_values(by="Signal Score", ascending=False).head(5)
                st.session_state.short_setups = df_universe[df_universe["Bias"] == "BEARISH"].sort_values(by="Signal Score", ascending=False).head(5)
            else:
                st.session_state.long_setups, st.session_state.short_setups = pd.DataFrame(), pd.DataFrame()
            
            st.session_state.last_scan_time = datetime.now()

    # =====================================================================
    # DASHBOARD RENDERING PLATFORM
    # =====================================================================
    if st.session_state.index_bias is not None:
        bias_block = st.session_state.index_bias
        
        st.markdown("### 🌐 Nifty 50 Options Strategy Matrix")
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        with col_metric1:
            st.metric("NIFTY DIRECTION BIAS", bias_block["Signal"])
        with col_metric2:
            st.metric("STRATEGY CONFIDENCE", bias_block["Confidence"])
        with col_metric3:
            st.metric("INDEX MARKET BREADTH", f"{round(st.session_state.telemetry['breadth'], 1)}% Bullish")
        with col_metric4:
            # Native Progress Bar Gauge for Market Breadth Visualizations
            st.progress(st.session_state.telemetry['breadth'] / 100.0)
            
        with st.container(border=True):
            st.write(f"**⚡ Preferred Options Play:** `{bias_block['Preferred Strategy']}`")
            st.write(f"• **System Entry Trigger Zone:** {bias_block['Entry Zone Trigger']}")
            st.write(f"• **Risk Horizon Profile:** {bias_block['Risk Profile']}")
            st.write(f"• **Defensive Invalidation Floor:** {bias_block['Invalidation Floor']}")
            st.write(f"• **Calculated Invalidation Targets:** {bias_block['Target Matrices']}")

        st.markdown("---")
        
        # Bottom Multi-Column Stock Ranking Layout Matrix
        col_long, col_short = st.columns(2)
        column_formatting = {
            "Signal Score": st.column_config.NumberColumn(format="%.1f pts"),
            "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Stop Loss": st.column_config.NumberColumn(format="₹%.2f"),
            "Target 1": st.column_config.NumberColumn(format="₹%.2f"),
            "Suggested Qty": st.column_config.NumberColumn(format="%d S")
        }
        
        with col_long:
            st.markdown("### 📈 Top 5 Alpha Long Breakouts")
            if st.session_state.long_setups.empty: st.info("No stocks passed high-conviction long criteria.")
            else: st.dataframe(st.session_state.long_setups.drop(columns=["Bias"], errors="ignore"), column_config=column_formatting, hide_index=True, use_container_width=True)
                
        with col_short:
            st.markdown("### 📉 Top 5 Alpha Short Breakouts")
            if st.session_state.short_setups.empty: st.info("No stocks passed high-conviction short criteria.")
            else: st.dataframe(st.session_state.short_setups.drop(columns=["Bias"], errors="ignore"), column_config=column_formatting, hide_index=True, use_container_width=True)
            
        if st.session_state.last_scan_time:
            st.caption(f"⏱️ Matrix Telemetry Stream Cached At: {st.session_state.last_scan_time.strftime('%d-%b-%Y %H:%M:%S')}")
else:
    st.error("Pipeline Failure: Unable to parse baseline indices.")

apply_sebi_footer()
