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
    
    s_ret_5d = ((stock_close.iloc[-1] - stock_close.iloc[-6]) / stock_close.iloc[-6]) * 100
    n_ret_5d = ((nifty_close.iloc[-1] - nifty_close.iloc[-6]) / nifty_close.iloc[-6]) * 100
    rs_5d = s_ret_5d - n_ret_5d
    
    s_ret_21d = ((stock_close.iloc[-1] - stock_close.iloc[-22]) / stock_close.iloc[-22]) * 100
    n_ret_21d = ((nifty_close.iloc[-1] - nifty_close.iloc[-22]) / nifty_close.iloc[-22]) * 100
    rs_21d = s_ret_21d - n_ret_21d
    
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
    """Calculates 0-10 points based on confirmed breakouts/breakdowns."""
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
        
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)

    close_series = stock_df['Close']
    
    stock_df['EMA20'] = close_series.ewm(span=20, adjust=False).mean()
    stock_df['EMA50'] = close_series.ewm(span=50, adjust=False).mean()
    stock_df['EMA200'] = close_series.ewm(span=200, adjust=False).mean()
    
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_df['RSI'] = 10
