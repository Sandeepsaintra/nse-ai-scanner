import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ... [Keep your helper functions calculate_rs_score, etc., here] ...

def score_stock(stock_df, nifty_df, market_bias, atr_period):
    # 1. Force a copy and handle MultiIndex early
    stock_df = stock_df.copy()
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
        
    # 2. Length Guard
    if len(stock_df) < 250 or len(nifty_df) < 250:
        return None

    # 3. Extract close_series AFTER flattening
    close_series = stock_df['Close']
    
    # Core indicators
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
    
    # ATR
    high, low = stock_df['High'], stock_df['Low']
    tr = pd.concat([high-low, (high-close_series.shift()).abs(), (low-close_series.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean().iloc[-1]
    
    # Scoring Logic (unchanged from your functional requirement)
    results = {}
    for choice in ["CALL", "PUT"]:
        rs_pts = calculate_rs_score(stock_df, nifty_df)
        trend_pts = calculate_trend_score(stock_df) if choice == "CALL" else (30 - calculate_trend_score(stock_df))
        mom_pts = calculate_momentum_score(stock_df, choice)
        vol_pts = calculate_volume_score(stock_df)
        pa_pts = calculate_price_action_score(stock_df, choice)
        
        base_score = rs_pts + trend_pts + mom_pts + vol_pts + pa_pts
        adjustment = 15 if (market_bias == "BULLISH" and choice == "CALL") or (market_bias == "BEARISH" and choice == "PUT") else (-20 if (market_bias == "BULLISH" and choice == "PUT") or (market_bias == "BEARISH" and choice == "CALL") else 0)
        
        results[choice] = (max(0, min(100, base_score + adjustment)), rs_pts, trend_pts, mom_pts, vol_pts, pa_pts, atr)
        
    winning_signal = "CALL" if results["CALL"][0] >= results["PUT"][0] else "PUT"
    f_score, rs_s, trend_s, mom_s, vol_s, pa_s, final_atr = results[winning_signal]
    
    # Action Logic
    action = "STRONG" if f_score >= 90 else (winning_signal if f_score >= 75 else ("WATCHLIST" if f_score >= 60 else "NO TRADE"))
    
    return {
        "Score": f_score, "Action": action, "Direction": winning_signal,
        "RS": rs_s, "Trend": trend_s, "Momentum": mom_s, "Volume": vol_s, "PA": pa_s,
        "Entry": round(float(close_series.iloc[-1]), 2),
        "Stoploss": round(float(close_series.iloc[-1] - (1.5 * final_atr if winning_signal == "CALL" else -1.5 * final_atr)), 2),
        "Target 1": round(float(close_series.iloc[-1] + (2.0 * final_atr if winning_signal == "CALL" else -2.0 * final_atr)), 2)
    }

# --- Update your Main execution block ---
if st.button("🚀 Run Multi-Factor Matrix Scan"):
    nifty_raw = yf.download("^NSEI", period="2y", interval="1d", progress=False).copy()
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    # ... rest of the loop logic ...
