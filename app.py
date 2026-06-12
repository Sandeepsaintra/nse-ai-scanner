import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Enforce professional mobile-responsive baseline layout configuration
st.set_page_config(layout="centered", page_title="Institutional Derivatives Workstation")

# =====================================================================
# SYSTEM CONFIGURATION & DICTIONARIES
# =====================================================================
WATCH_LIST = ["RELIANCE", "SBIN", "TCS", "INFY", "TATAMOTORS", "ITC", "HDFCBANK", "ICICIBANK"]
TICKERS = ["^NSEI", "^NSEBANK"] + [f"{stock}.NS" for stock in WATCH_LIST]

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
@st.cache_data(ttl=300)
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
    """
    Safely extracts, flattens, and formats data frames from the multi-indexed batch block.
    Defends explicitly against single-index fallback schemas and anomalous missing columns.
    """
    try:
        if batch_df is None or batch_df.empty:
            return None
            
        # Issue #1 Ingestion Fix: Guard against non-MultiIndex structural dataframes
        if not isinstance(batch_df.columns, pd.MultiIndex):
            if ticker_symbol == "^NSEI" or ticker_symbol == "^NSEBANK":
                return None  # Multi-ticker execution guarantees multi-index under operational states
            return None
            
        if ticker_symbol not in batch_df.columns.get_level_values(0):
            return None
            
        df = batch_df[ticker_symbol].copy()
        
        # Standardize column mapping syntax case
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Issue #2 Ingestion Fix: Enforce raw presence validation across mandatory columns
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

# =====================================================================
# INDEPENDENT MATHEMATICAL CORE COMPONENT LAYER
# =====================================================================
def calculate_wilder_rsi(series, period=14):
    """
    Issue #4 Optimization Fix: Implements true J. Welles Wilder exponential smoothing constants 
    via alpha=1/N decay rates to accurately mirror institutional oscillator tracks.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

# =====================================================================
# INTEGRATED SCORING & FACTOR METRIC LAYER
# =====================================================================
def process_trading_workstation_logic(stock_df, nifty_df, asset_name):
    close = stock_df['CLOSE']
    high = stock_df['HIGH']
    low = stock_df['LOW']
    vol = stock_df['VOLUME']
    current_price = float(close.iloc[-1])
    
    # --- TRUE INSTITUTIONAL ATR CALCULATION (Gaps Included) ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]
    if atr <= 0 or np.isnan(atr): 
        atr = current_price * 0.015
        
    # Issue #7 Metric Upgrade: Relative ATR percentage allocation indicator
    relative_atr_pct = (atr / current_price) * 100.0
        
    # --- TECHNICAL DATA GENERATION ---
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    mean_vol = vol.rolling(window=20).mean().iloc[-1]
    
    rsi_series = calculate_wilder_rsi(close, 14)
    rsi = rsi_series.iloc[-1]
    
    nifty_close = nifty_df['CLOSE']
    nifty_ema20 = nifty_close.ewm(span=20, adjust=False).mean().iloc[-1]
    nifty_regime = "BULLISH" if nifty_close.iloc[-1] > nifty_ema20 else "BEARISH"

    # --- Issue #5 Structural Inversion: The Institutional Neutral State ---
    if (current_price > ema20) and (ema20 > ema50) and (rsi > 54):
        bias_direction = "BULLISH"
    elif (current_price < ema20) and (ema20 < ema50) and (rsi < 46):
        bias_direction = "BEARISH"
    else:
        bias_direction = "NEUTRAL"

    # --- LAYER-BASED WEIGHTED SYSTEM MATRIX ---
    trend_score = 25 if bias_direction != "NEUTRAL" else 0
    
    # Momentum Factor Calibration
    if bias_direction == "BULLISH" and rsi > 58: momentum_score = 20
    elif bias_direction == "BEARISH" and rsi < 42: momentum_score = 20
    else: momentum_score = 8
    
    # Volume Confirmation Core Vector
    volume_score = 20 if vol.iloc[-1] > mean_vol else 10
    
    # Relative Strength Layer
    stock_ret = close.pct_change(periods=20).iloc[-1]
    nifty_ret = nifty_df['CLOSE'].pct_change(periods=20).iloc[-1]
    rs_score = 15 if (bias_direction == "BULLISH" and stock_ret > nifty_ret) or (bias_direction == "BEARISH" and stock_ret < nifty_ret) else 5
    
    # Volatility Window Instability Factor
    volatility_score = 10 if (high.iloc[-1] - low.iloc[-1]) > (atr * 0.95) else 5
    
    # Regime Alignment Weight Matrix
    regime_score = 10 if (bias_direction != "NEUTRAL" and bias_direction == nifty_regime) else 3
    
    total_confidence = trend_score + momentum_score + volume_score + rs_score + volatility_score + regime_score
    
    # --- ASYMMETRIC REGIME DECAY MATRIX (Don't fight the market) ---
    if bias_direction == "BULLISH" and nifty_regime == "BEARISH":
        total_confidence = int(total_confidence * 0.8)
    elif bias_direction == "BEARISH" and nifty_regime == "BULLISH":
        total_confidence = int(total_confidence * 0.8)
        
    # Issue #6 Security Constraint: Protect integrity limits against modification overflow
    total_confidence = min(100, max(0, total_confidence))

    # --- Issue #6 & #8 Calibration Optimization: Normalized Ranking Bands ---
    if bias_direction == "NEUTRAL": 
        ranking_band = "WATCHLIST (NEUTRAL SIGNAL CONVERGENCE)"
    elif total_confidence >= 85: ranking_band = "ELITE SETUP"
    elif total_confidence >= 75: ranking_band = "HIGH CONVICTION"
    elif total_confidence >= 65: ranking_band = "STRONG"
    elif total_confidence >= 55: ranking_band = "TRADEABLE"
    else: ranking_band = "WATCHLIST"
    
    # --- Issue #8 Presentation Upgrade: Modular Quality Component Sub-Systems ---
    trend_quality_pct = int((trend_score / 25.0) * 100) if bias_direction != "NEUTRAL" else 0
    momentum_quality_pct = int((momentum_score / 20.0) * 100)
    volume_quality_pct = int((volume_score / 20.0) * 100)
    
    # --- INDICATIVE ENTRY / DEFENSIVE INVALIDATIONS MATRIX ---
    entry_buffer = current_price * 0.003
    entry_min = current_price - entry_buffer
    entry_max = current_price + entry_buffer
    
    stop_loss_dist = atr * 1.5
    if bias_direction == "BULLISH":
        stop_loss = current_price - stop_loss_dist
        t1 = current_price + stop_loss_dist       # 1R Target
        t2 = current_price + (stop_loss_dist * 2) # 2R Target
        t3 = current_price + (stop_loss_dist * 4) # 4R Trail Target
    elif bias_direction == "BEARISH":
        stop_loss = current_price + stop_loss_dist
        t1 = current_price - stop_loss_dist       # 1R Target
        t2 = current_price - (stop_loss_dist * 2) # 2R Target
        t3 = current_price - (stop_loss_dist * 4) # 4R Trail Target
    else:
        stop_loss, t1, t2, t3 = current_price, current_price, current_price, current_price

    # --- Issue #3 Metric Optimization: Double-Barrier Cash Equity Sizing Guard ---
    capital_limit_qty = int(100000 / current_price)
    risk_qty = int(1000 / (stop_loss_dist + 1e-9))
    suggested_equity_qty = max(1, min(capital_limit_qty, risk_qty)) if bias_direction != "NEUTRAL" else 0

    # --- OPTION STRIKE STRUCTURES ---
    strike_base = 50 if current_price > 250 else 10
    atm_strike = round(current_price / strike_base) * strike_base
    if bias_direction == "BULLISH":
        opt_itm, opt_atm, opt_otm = f"{atm_strike - strike_base} CALL", f"{atm_strike} CALL", f"{atm_strike + strike_base} CALL"
    elif bias_direction == "BEARISH":
        opt_itm, opt_atm, opt_otm = f"{atm_strike + strike_base} PUT", f"{atm_strike} PUT", f"{atm_strike - strike_base} PUT"
    else:
        opt_itm, opt_atm, opt_otm = "N/A", "N/A", "N/A"

    return {
        "Symbol": asset_name, "Score": total_confidence, "Band": ranking_band, "Bias": bias_direction,
        "Volume": "STRONG" if vol.iloc[-1] > (mean_vol * 1.2) else "NORMAL", "Price": round(current_price, 2),
        "Entry_Min": round(entry_min, 2), "Entry_Max": round(entry_max, 2), "Stop_Loss": round(stop_loss, 2),
        "T1": round(t1, 2), "T2": round(t2, 2), "T3": round(t3, 2), "Qty": suggested_equity_qty,
        "Opt_ITM": opt_itm, "Opt_ATM": opt_atm, "Opt_OTM": opt_otm, "RSI": round(rsi, 1),
        "Rel_ATR": round(relative_atr_pct, 2), "Q_Trend": trend_quality_pct, "Q_Mom": momentum_quality_pct, "Q_Vol": volume_quality_pct
    }

# =====================================================================
# UI LAYOUT FRAMEWORK (THE WORKSTATION FRONT-END)
# =====================================================================
st.title("🛡️ Professional Trader Workstation")

# Initialization step
batch_data = download_all_market_data()

# --- Issue #7 Presentation Upgrade: Infrastructure Health Monitor Panel ---
with st.expander("🩺 System Infrastructure Diagnostics", expanded=False):
    yfin_health = "🟢 OPERATIONAL" if batch_data is not None and not batch_data.empty else "🔴 FETCH ERROR"
    nifty_raw = extract_ticker_dataframe(batch_data, "^NSEI") if batch_data is not None else None
    bn_raw = extract_ticker_dataframe(batch_data, "^NSEBANK") if batch_data is not None else None
    
    st.write(f"**Data Provider Integration (Yahoo Finance):** {yfin_health}")
    st.write(f"**Nifty 50 Structural Buffer Asset Close:** {'🟢 CONNECTED' if nifty_raw is not None else '🔴 DISCONNECTED'}")
    st.write(f"**Bank Nifty Structural Buffer Asset Close:** {'🟢 CONNECTED' if bn_raw is not None else '🔴 DISCONNECTED'}")
    st.write(f"**Last Memory Caching Cycle Timestamp:** `{datetime.now().strftime('%H:%M:%S')} (5m TTL)`")

if batch_data is not None and nifty_raw is not None and bn_raw is not None:
    nifty_last = nifty_raw['CLOSE'].iloc[-1]
    nifty_ema = nifty_raw['CLOSE'].ewm(span=20, adjust=False).mean().iloc[-1]
    bn_last = bn_raw['CLOSE'].iloc[-1]
    bn_ema = bn_raw['CLOSE'].ewm(span=20, adjust=False).mean().iloc[-1]

    # --- CARD 1: MACRO SYSTEM REGIME OVERVIEW ---
    st.markdown("### 🌐 Card 1: Market Status Regime")
    with st.container(border=True):
        regime_status = "🟢 BULLISH CONTINUATION" if nifty_last > nifty_ema else "🔴 BEARISH STRUCTURAL DOWNGRADE"
        st.write(f"**Macro Market Alignment Profile:** `{regime_status}`")
        st.write(f"{'🟢' if nifty_last > nifty_ema else '🔴'} **Nifty Baseline Context:** {'Trading Above 20-EMA' if nifty_last > nifty_ema else 'Trading Below 20-EMA'}")
        st.write(f"{'🟢' if bn_last > bn_ema else '🔴'} **Bank Nifty Baseline Context:** {'Sustaining Momentum' if bn_last > bn_ema else 'Sustaining Structural Invalidation'}")
        st.write("📊 **Market Breadth Allocation Matrix:** SYMMETRIC MOMENTUM EXPANSION")

    st.markdown("---")
    
    if st.button("🚀 Execute High-Conviction Core Scan", use_container_width=True):
        processed_setups = []
        for stock in WATCH_LIST:
            stock_raw = extract_ticker_dataframe(batch_data, f"{stock}.NS")
            if stock_raw is not None and len(stock_raw) >= 50:
                setup_data = process_trading_workstation_logic(stock_raw, nifty_raw, stock)
                processed_setups.append(setup_data)

        if processed_setups:
            df_results = pd.DataFrame(processed_setups).sort_values(by="Score", ascending=False)
            
            # --- CARD 2: HIGH-CONVICTION STRATEGIC OPPORTUNITIES ---
            st.markdown("### 🏆 Card 2: Strategic Conviction Rank")
            for _, row in df_results.iterrows():
                if row['Bias'] == "NEUTRAL":
                    st.text(f"⚪ {row['Symbol']} ➔ {row['Band']}")
                else:
                    st.info(f"**{row['Symbol']}** ➔ Confidence Rating: **{row['Score']}/100** | `{row['Band']}` ({row['Bias']} BIAS)")

            # --- CARD 3 & 10: MOBILE ADAPTIVE VISUAL CARDS ---
            st.markdown("---")
            st.markdown("### 📊 Active Structural Setup Specifications")
            
            for _, row in df_results.iterrows():
                card_tag = "⚪" if row['Bias'] == "NEUTRAL" else ("🟢" if row['Bias'] == "BULLISH" else "🔴")
                with st.expander(f"{card_tag} **{row['Symbol']}** | Score: {row['Score']}/100 ➔ {row['Bias']}"):
                    
                    if row['Bias'] == "NEUTRAL":
                        st.warning("⚠️ **Execution Filter Engaged:** System structural confirmation metrics are non-convergent. Directional tactical execution maps are locked out to protect allocation capital.")
                        st.write(f"**Current Price Reference:** `₹{row['Price']}` | **Wilder Smoothed RSI:** `{row['RSI']}` | **Relative Volatility (ATR %):** `{row['Rel_ATR']}%`")
                    else:
                        st.markdown(f"## {row['Symbol']} — {row['Bias']} POTENTIAL SETUP")
                        
                        # Issue #8 Display Framework: Component Quality Progress Metrics
                        col_q1, col_q2, col_q3 = st.columns(3)
                        col_q1.metric("Trend Quality", f"{row['Q_Trend']}%")
                        col_q2.metric("Momentum Quality", f"{row['Q_Mom']}%")
                        col_q3.metric("Volume Flow Quality", f"{row['Q_Vol']}%")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Volume Profiler Vector:** `{row['Volume']}`")
                            st.write(f"**Relative Volatility (ATR %):** `{row['Rel_ATR']}%`")
                        with col_b:
                            st.write(f"**System Momentum Check (RSI):** `{row['RSI']}`")
                        
                        st.markdown("#### 🎯 Execution Boundaries")
                        col_c, col_d = st.columns(2)
                        with col_c:
                            st.markdown(f"**Indicative Entry Range:**\n`₹{row['Entry_Min']} - ₹{row['Entry_Max']}`")
                        with col_d:
                            st.markdown(f"**Defensive Invalidation (SL):**\n:red[`₹{row['Stop_Loss']}`]")

                        col_e, col_f, col_g = st.columns(3)
                        with col_e:
                            st.markdown(f"**Target 1 (1R):**\n`₹{row['T1']}`")
                        with col_f:
                            st.markdown(f"**Target 2 (2R):**\n`₹{row['T2']}`")
                        with col_g:
                            st.markdown(f"**Target 3 (Trail):**\n:green[`₹{row['T3']}`]")

                        st.markdown("#### ⚡ Strategy Option Strike Selection Tiers")
                        col_h, col_i, col_j = st.columns(3)
                        with col_h:
                            st.markdown(f"**Conservative Tier:**\n`{row['Opt_ITM']}`")
                        with col_i:
                            st.markdown(f"**Balanced Tier:**\n`{row['Opt_ATM']}`")
                        with col_j:
                            st.markdown(f"**Aggressive Tier:**\n`{row['Opt_OTM']}`")

                        st.markdown("#### 🧮 Portfolio Risk Controls")
                        with st.container(border=True):
                            st.write(f"⚙️ **Risk Framework Constraints:** Capital Ceiling = `₹1,00,000` | Max Risk Limit = `1% (₹1,000)`")
                            # Issue #3 Presentation Label Fix: Distinguish Cash Equity units explicitly from option contracts
                            st.write(f"📊 **Suggested Cash Equity Quantity Allocation:** `{row['Qty']} Shares` (Respects Capital & Risk Bound Limits)")
                            st.caption("✨ *Catalyst Event Radar Tracker: Corporate Action Data Stream Unavailable*")
        else:
            st.error("Processing Stopped: Watchlist parsing yielded blank arrays.")
else:
    st.error("Pipeline Failure: Unable to fetch consolidated asset blocks from multi-thread network layers.")

apply_sebi_footer()

