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

# Allow your React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. NIFTY HEAVYWEIGHTS LIST ---
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "BAJFINANCE.NS"
]

# --- 2. FII MACRO SENTIMENT ENGINE (NSE SCRAPER) ---
def get_fii_sentiment():
    try:
        # NSE requires standard browser headers to prevent blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        
        # Look back up to 5 days to find the most recent trading day's file
        for days_back in range(0, 5):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime('%d%m%Y')
            display_date = (datetime.now() - timedelta(days=days_back)).strftime('%d-%b-%Y')
            
            url = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{date_str}.csv"
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                # Successfully found the file. Use thousands=',' to parse numbers cleanly.
                df = pd.read_csv(StringIO(response.text), skiprows=1, thousands=',')
                df.columns = df.columns.str.strip()
