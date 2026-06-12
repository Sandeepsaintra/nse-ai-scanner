import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", page_title="Professional Option Scanner")

# =====================================================================
# LAYER 1: MARKET BIAS ENGINE
# =====================================================================
def get_market_bias(nifty_df):
    """Determines macro trend alignment using the Nifty 50 Index."""
    try:
        close = nifty_df["Close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        
        latest_close = float(close.iloc[-1])
        latest_ema20 = float(ema20.iloc[-1])
        latest_ema50 = float(ema50.iloc[-1])
        
        if latest_close > latest_ema20 and latest_close > latest_ema50:
            return "BULLISH", 85
        elif latest_close < latest_ema20 and latest_close < latest_ema50:
            return "BEARISH", 85
        else:
            return "SIDEWAYS", 50
    except Exception:
        return "UNKNOWN", 0

# =====================================================================
# LAYER 2: SUB-COMPONENT SCORING FUNCTIONS
# =====================================================================
def calculate_rs_score(stock_df, nifty_df):
    """Calculates 0-30 points for 5D and 21D Relative Strength vs Nifty."""
    stock_close = stock_df['Close']
    nifty_close = nifty_df['Close']
    
    # Calculate percentage changes
    s_ret_5d = ((stock_close.iloc[-1] - stock_close.iloc[-6]) / stock_close.iloc[-6]) * 100
    n_ret_5d = ((nifty_close.iloc[-1] - nifty_close.iloc[-6]) / nifty_close.iloc[-6]) * 100
    rs_5d = s_ret_5d - n_ret_5d
    
    s_ret_21d = ((stock_close.iloc[-1] - stock_close.iloc[-22]) / stock_close.iloc[-22]) * 100
    n_ret_21d = ((nifty_close.iloc[-1] - nifty_close.iloc[-22]) / nifty_close.iloc[-22]) * 100
    rs_21d = s_ret_21d - n_ret_21d
    
    # Normalized Scoring
    rs_5d_score = 10 if rs_5d > 2 else (5 if rs_5d > 0 else 0)
    rs_21d_score = 20 if rs_21d > 5 else (10 if rs_21d > 0 else 0)
    
    return rs_5d_score + rs_21d_score

def calculate_trend_score(stock_df):
    """Calculates 0-30 points using EMA geometry."""
    close = float(stock_df["Close"].iloc[-1])
    ema20 = float(stock_df["EMA20"].iloc[-1])
    ema50 = float(stock_df["EMA50"].iloc[-1])
    ema200 = float(stock_df["EMA200"].iloc[-1])
    
    score = 0
    if close > ema20: score += 10
    if ema20 > ema50: score += 10
    if ema50 > ema200: score += 10
    return score

def calculate_momentum_score(stock_df, signal_type):
    """Calculates 0-20 points combining RSI and MACD logic."""
    rsi_today = float(stock_df["RSI"].iloc[-1])
    rsi_yesterday = float(stock_df["RSI"].iloc[-2])
    macd_today = float(stock_df["MACD"].iloc[-1])
    macd_yesterday = float(stock_df["MACD"].iloc[-2])
    sig_today = float(stock_df["Signal_Line"].iloc[-1])
    sig_yesterday = float(stock_df["Signal_Line"].iloc[-2])
    
    rsi_score = 0
    macd_score = 0
    
    if signal_type == "CALL":
        if rsi_today > 55: rsi_score = 10
        elif rsi_today > 50 and rsi_today > rsi_yesterday: rsi_score = 7
        elif rsi_today > rsi_yesterday: rsi_score = 3
        
        if macd_yesterday <= sig_yesterday and macd_today > sig_today: macd_score = 10
        elif macd_today > sig_today: macd_score = 7
        elif macd_today > macd_yesterday: macd_score = 3
    else:
        if rsi_today < 45: rsi_score = 10
        elif rsi_today < 50 and rsi_today < rsi_yesterday: rsi_score = 7
        elif rsi_today < rsi_yesterday: rsi_score = 3
        
        if macd_yesterday >= sig_yesterday and macd_today < sig_today: macd_score = 10
        elif macd_today < sig_today: macd_score = 7
        elif macd_today < macd_yesterday: macd_score = 3
        
    return rsi_score + macd_score

def calculate_volume_score(stock_df):
    """Calculates 0-10 points based on trading volume spikes."""
    vol_today = float(stock_df["Volume"].iloc[-1])
    vol_avg = float(stock_df["Volume"].tail(20).mean())
    ratio = vol_today / vol_avg if vol_avg > 0 else 0
    
    if ratio > 2.0: return 10
    elif ratio > 1.5: return 5
    return 0

def calculate_price_action_score(stock_df, signal_type):
    """Calculates 0-10 points based on confirmed daily resistance breakouts."""
    close_today = float(stock_df["Close"].iloc[-1])
    
    if signal_type == "CALL":
        highest_10 = float(stock_df["High"].iloc[-11:-1].max())
        highest_20 = float(stock_df["High"].iloc[-21:-1].max())
        if close_today > highest_20: return 10
        elif close_today > highest_10: return 5
    else:
        lowest_10 = float(stock_df["Low"].iloc[-11:-1].min())
        lowest_20 = float(stock_df["Low"].iloc[-21:-1].min())
        if close_today < lowest_20: return 10
        elif close_today < lowest_10: return 5
    return 0

# =====================================================================
# MASTER SCORING ENGINE (LAYERS 2 - 5)
# =====================================================================
def score_stock(stock_df, nifty_df, market_bias, atr_period):
    """Evaluates stock data through the complete multi-layer system."""
    if len(stock_df) < 250 or len(nifty_df) < 250:
        return None
        
    # Clean up yfinance columns if multi-indexed
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)

    close_series = stock_df['Close']
    
    # Core technical computations
    stock_df['EMA20'] = close_series.ewm(span=20, adjust=False).mean()
    stock_df['EMA50'] = close_series.ewm(span=50, adjust=False).mean()
    stock_df['EMA200'] = close_series.ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    stock_df['MACD'] = ema12 - ema26
    stock_df['Signal_Line'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR Volatility Model
    high = stock_df['High']
    low = stock_df['Low']
    tr1 = high - low
    tr2 = (high - close_series.shift()).abs()
    tr3 = (low - close_series.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean().iloc[-1]
    
    # Evaluate Pathways
    results = {}
    for choice in ["CALL", "PUT"]:
        rs_pts = calculate_rs_score(stock_df, nifty_df)
        trend_pts = calculate_trend_score(stock_df) if choice == "CALL" else (30 - calculate_trend_score(stock_df))
        mom_pts = calculate_momentum_score(stock_df, choice)
        vol_pts = calculate_volume_score(stock_df)
        pa_pts = calculate_price_action_score(stock_df, choice)
        
        base_score = rs_pts + trend_pts + mom_pts + vol_pts + pa_pts
        
        # Layer 3: Market Alignment
        if market_bias == "BULLISH":
            adjustment = 15 if choice == "CALL" else -20
        elif market_bias == "BEARISH":
            adjustment = 15 if choice == "PUT" else -20
        else:
            adjustment = 0
            
        final_score = max(0, min(100, base_score + adjustment))
        results[choice] = (final_score, rs_pts, trend_pts, mom_pts, vol_pts, pa_pts, atr)
        
    # Determine the winning strategy direction
    winning_signal = "CALL" if results["CALL"][0] >= results["PUT"][0] else "PUT"
    f_score, rs_s, trend_s, mom_s, vol_s, pa_s, final_atr = results[winning_signal]
    
    # Layer 4: News/Catalyst Placeholder Values
    news_status = "Neutral"
    event_risk = "None"
    
    # Layer 4 Trade Engine Classification
    if f_score >= 90: action = f"STRONG {winning_signal}"
    elif f_score >= 75: action = winning_signal
    elif f_score >= 60: action = "WATCHLIST"
    else: action = "NO TRADE"
    
    current_price = float(close_series.iloc[-1])
    
    # Layer 5: Risk Level Generation
    if winning_signal == "CALL":
        sl = current_price - (1.5 * final_atr)
        t1 = current_price + (2.0 * final_atr)
        t2 = current_price + (4.0 * final_atr)
    else:
        sl = current_price + (1.5 * final_atr)
        t1 = current_price - (2.0 * final_atr)
        t2 = current_price - (4.0 * final_atr)
        
    return {
        "Score": f_score, "Action": action, "RS": rs_s, "Trend": trend_s,
        "Momentum": mom_s, "Volume": vol_s, "PA": pa_s, "News": news_status,
        "Event": event_risk, "Entry": round(current_price, 2), "Stoploss": round(sl, 2),
        "Target 1": round(t1, 2), "Target 2": round(t2, 2)
    }

# =====================================================================
# DASHBOARD INTERFACE
# =====================================================================
st.title("🛡️ Institutional 5-Layer Options Decision Engine")

# Sidebar configurations
st.sidebar.header("⚙️ Scanner Configurations")
mode_toggle = st.sidebar.radio("Risk Horizon Mode", ["Aggressive Mode (ATR 14)", "Conservative Mode (ATR 20)"])
atr_len = 14 if "ATR 14" in mode_toggle else 20

watch_list = ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "TATAMOTORS.NS", "ITC.NS", "HCLTECH.NS"]

if st.button("🚀 Run Multi-Factor Matrix Scan"):
    with st.spinner("Downloading structural index data and analyzing layers..."):
        # Download Nifty 50 Index reference data
        nifty_raw = yf.download("^NSEI", period="2y", interval="1d", progress=False)
        if isinstance(nifty_raw.columns, pd.MultiIndex):
            nifty_raw.columns = nifty_raw.columns.get_level_values(0)
            
        m_bias, m_conf = get_market_bias(nifty_raw)
        
        # Display Layer 1 Status Card
        st.subheader("🌍 Layer 1: Market Macro Regime")
        if m_bias == "BULLISH":
            st.success(f"**MARKET BIAS:** {m_bias} ({m_conf}% Confidence) — Long setups preferred.")
        elif m_bias == "BEARISH":
            st.error(f"**MARKET BIAS:** {m_bias} ({m_conf}% Confidence) — Short setups preferred.")
        else:
            st.warning(f"**MARKET BIAS:** {m_bias} ({m_conf}% Confidence) — High index option volatility risk.")
            
        # Scan processing loops
        compiled_data = []
        for asset in watch_list:
            raw_stock = yf.download(asset, period="2y", interval="1d", progress=False)
            metrics = score_stock(raw_stock, nifty_raw, m_bias, atr_len)
            if metrics:
                metrics["Symbol"] = asset.split('.')[0]
                compiled_data.append(metrics)
                
        # Format and display dashboard results matrix
        if compiled_data:
            df_display = pd.DataFrame(compiled_data)
            column_order = ["Symbol", "Action", "Score", "RS", "Trend", "Momentum", "Volume", "PA", "News", "Event", "Entry", "Stoploss", "Target 1", "Target 2"]
            df_display = df_display[column_order]
            
            st.subheader("📊 Layer 2-5: Strategy Selection Matrix")
            st.dataframe(df_display.style.background_gradient(cmap="RdYlGn", subset=["Score"]))
