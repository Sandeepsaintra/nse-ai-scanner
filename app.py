def engine_technical(df, ticker):
    """Calculates Trend, Volume Confirmation, SL, and Targets"""
    # Ensure data is numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    close = df['CLOSE'].iloc[-1]
    ema20 = df['CLOSE'].ewm(span=20).mean().iloc[-1]
    ema50 = df['CLOSE'].ewm(span=50).mean().iloc[-1]
    
    # 1. Volume Confirmation Logic
    # Calculate 20-period Moving Average of Volume
    df['VOL_SMA_20'] = df['VOLUME'].rolling(window=20).mean()
    current_vol = df['VOLUME'].iloc[-1]
    avg_vol = df['VOL_SMA_20'].iloc[-1]
    vol_confirm = current_vol > avg_vol
    
    # 2. Risk Management (ATR)
    atr = get_atr(df)
    
    # 3. Decision Logic (Bias + Volume Confirmation)
    if close > ema20 and ema20 > ema50 and vol_confirm:
        bias = "BUY (CALL)"
        sl = round(close - (1.5 * atr), 2)
        target = round(close + (2.5 * atr), 2)
    elif close < ema20 and ema20 < ema50 and vol_confirm:
        bias = "SELL (PUT)"
        sl = round(close + (1.5 * atr), 2)
        target = round(close - (2.5 * atr), 2)
    else:
        bias = "HOLD"
        sl, target = 0, 0
        
    return {
        "Symbol": ticker.replace(".NS", ""), 
        "Bias": bias,
        "Entry": round(close, 2), 
        "SL": sl, 
        "Target": target,
        "Vol_Confirmed": vol_confirm
    }
