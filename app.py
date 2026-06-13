import os
import time
from datetime import datetime, timedelta
import requests
from io import StringIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pandas_ta as ta
import yfinance as yf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "BAJFINANCE.NS"
]

def get_fii_sentiment():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        
        for days_back in range(0, 5):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime('%d%m%Y')
            display_date = (datetime.now() - timedelta(days=days_back)).strftime('%d-%b-%Y')
            
            url = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{date_str}.csv"
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), skiprows=1, thousands=',')
                df.columns = df.columns.str.strip()
                
                fii_data = df[df['Client Type'] == 'FII']
                if not fii_data.empty:
                    fii_long = int(fii_data['Future Index Long'].iloc[0])
                    fii_short = int(fii_data['Future Index Short'].iloc[0])
                    
                    total_positions = fii_long + fii_short
                    if total_positions == 0:
                        continue
                        
                    long_pct = (fii_long / total_positions) * 100
                    short_pct = (fii_short / total_positions) * 100
                    
                    sentiment = "BEARISH" if short_pct > 60 else "BULLISH" if long_pct > 60 else "NEUTRAL"
                    
                    return {
                        "date": display_date,
                        "fiiLong": round(long_pct, 1),
                        "fiiShort": round(short_pct, 1),
                        "sentiment": sentiment
                    }
                    
        return {"error": "Could not fetch FII data from NSE archives."}
        
    except Exception as e:
        print(f"Error fetching FII data: {e}")
        return {"error": "Macro pipeline failed."}

def analyze_stock(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="100d", interval="1d")
        
        if df.empty or len(df) < 50:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        
        df['EMA9'] = ta.ema(df['close'], length=9)
        df['EMA21'] = ta.ema(df['close'], length=21)
        df['EMA50'] = ta.ema(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd_df['MACD_12_26_9']
        df['MACDs'] = macd_df['MACDs_12_26_9']
        
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        bb_df = ta.bbands(df['close'], length=20, std=2)
        df['BBU'] = bb_df['BBU_20_2.0']
        df['BBL'] = bb_df['BBL_20_2.0']

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        lp = float(latest['close'])
        chg_pct = ((lp - float(prev['close'])) / float(prev['close'])) * 100
        rsi = float(latest['RSI'])
        macd = float(latest['MACD'])
        macd_sig = float(latest['MACDs'])
        atr = float(latest['ATR'])
        
        score = 0
        if lp > latest['EMA9']: score += 1
        if lp > latest['EMA21']: score += 1
        if lp > latest['EMA50']: score += 1
        if latest['EMA9'] > latest['EMA21']: score += 1
        if macd > macd_sig: score += 1
        if 50 < rsi < 70: score += 1
        if rsi < 30: score += 2  
        if rsi > 75: score -= 2  
        if lp < latest['BBL']: score += 1
        if lp > latest['BBU']: score -= 1

        if score >= 6: signal = "STRONG BUY"
        elif score >= 3: signal = "BUY"
        elif score <= -2: signal = "STRONG SELL"
        elif score <= -1: signal = "SELL"
        else: signal = "NEUTRAL"
        
        sl = (lp - (1.5 * atr)) if "BUY" in signal else (lp + (1.5 * atr))
        t1 = (lp + (1.5 * (lp - sl))) if "BUY" in signal else (lp - (1.5 * (sl - lp)))
        t2 = (lp + (3.0 * (lp - sl))) if "BUY" in signal else (lp - (3.0 * (sl - lp)))

        return {
            "sym": ticker_symbol.replace(".NS", ""),
            "price": round(lp, 2),
            "chgPct": round(chg_pct, 2),
            "signal": signal,
            "confidence": int(min(95, max(30, 50 + (score * 5)))),
            "entry": round(lp, 2),
            "sl": round(sl, 2),
            "t1": round(t1, 2),
            "t2": round(t2, 2),
            "rsi": round(rsi, 1),
            "macd": round(macd, 2),
            "atr": round(atr, 2),
            "e9": round(latest['EMA9'], 2),
            "e21": round(latest['EMA21'], 2),
            "e50": round(latest['EMA50'], 2),
            "volRatio": 1.1,
            "vol": int(latest['volume']),
            "superBull": bool(lp > latest['EMA21']),
            "score": score
        }
    except Exception as e:
        print(f"Error processing {ticker_symbol}: {e}")
        return None

@app.get("/api/macro-sentiment")
def read_macro_sentiment():
    return get_fii_sentiment()

@app.get("/api/live-scan")
def read_live_market_scan():
    results = []
    for symbol in TICKERS:
        data = analyze_stock(symbol)
        if data:
            results.append(data)
        time.sleep(0.1) 
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
