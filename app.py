import streamlit as st
import yfinance as yf
import pandas as pd

# -------------------------------------------------------------------
# 1. CORE DATA FUNCTIONS
# -------------------------------------------------------------------
def fetch_stock_data(ticker, period="3mo", interval="1d"):
    """Downloads historical data from Yahoo Finance."""
    stock_data = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    return stock_data

def calculate_rsi(data, window=14):
    """Calculates the 14-period Relative Strength Index."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).dropna()
    loss = (-delta.where(delta < 0, 0)).dropna()
    
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -------------------------------------------------------------------
# 2. HYBRID SCORING ENGINE
# -------------------------------------------------------------------
def bullish_reversal_score(rsi_today, rsi_yesterday, volume, avg_volume, close, ema_5, previous_high):
    """Applies strict mandatory filters and calculates confirmation bonuses."""
    # Step 1: Mandatory Filters
    mandatory_pass = (
        rsi_today < 30
        and rsi_today > rsi_yesterday
        and volume > avg_volume * 1.5
    )
    if not mandatory_pass:
        return 0, "IGNORE"
        
    # Step 2: Baseline Score
    score = 75
    
    # Step 3: Bonus Points
    if volume > avg_volume * 2:
        score += 10
    if close > ema_5:
        score += 20
    if close > previous_high:
        score += 25
        
    # Step 4: Final Signal Classification
    if score >= 110:
        signal = "STRONG BUY"
    elif score >= 90:
        signal = "BUY"
    else:
        signal = "WATCHLIST"
        
    return score, signal
# -------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE LAYOUT
# -------------------------------------------------------------------
st.set_page_config(page_title="NSE AI Scanner", layout="wide")
st.title("🏹 Nifty Bullish Reversal AI Scanner")
st.markdown("Scans Indian equities for high-probability oversold turning points using volume and price context.")

# Sidebar Configuration
st.sidebar.header("Scan Settings")
scan_mode = st.sidebar.selectbox(
    "Select Scan Universe",
    ["Nifty 50 Snippet", "Bank Nifty", "Custom"]
)

# Determine Tickers based on selection
if scan_mode == "Custom":
    ticker_input = st.sidebar.text_area(
        "Enter Tickers (comma separated)",
        value="RELIANCE.NS, TCS.NS, INFY.NS"
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
elif scan_mode == "Bank Nifty":
    tickers = ["SBIN.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
else:
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "ITC.NS"]

# -------------------------------------------------------------------
# 4. EXECUTION PIPELINE
# -------------------------------------------------------------------
if st.button("🚀 Run Market Scan"):
    results = []
    
    with st.spinner(f"Processing {len(tickers)} assets..."):
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                # Fetch & process
                df = fetch_stock_data(ticker)
                if len(df) < 20:
                    continue
                    
                df['RSI'] = calculate_rsi(df)
                
                # Extract pipeline metrics via integer positions
                volume = float(df['Volume'].iloc[-1])
                avg_volume = float(df['Volume'].tail(20).mean())
                close = float(df['Close'].iloc[-1])
                ema_5 = float(df['Close'].ewm(span=5, adjust=False).mean().iloc[-1])
                rsi_today = float(df['RSI'].iloc[-1])
                rsi_yesterday = float(df['RSI'].iloc[-2])
                previous_high = float(df['High'].iloc[-2])
                
                # Compute scoring
                score, signal = bullish_reversal_score(
                    rsi_today, rsi_yesterday, volume, avg_volume, 
                    close, ema_5, previous_high
                )
                
                # Filter out the noise
                if signal != "IGNORE":
                    results.append({
                        "Symbol": ticker,
                        "RSI": round(rsi_today, 2),
                        "Volume Ratio": round(volume / avg_volume, 2),
                        "Score": score,
                        "Signal": signal
                    })
            except Exception as e:
                st.sidebar.error(f"Skipped {ticker}: Structural error.")
                
            progress_bar.progress((i + 1) / len(tickers))
        progress_bar.empty()
        
    # -------------------------------------------------------------------
    # 5. RENDER RESULTS WITH COLOR CODING
    # -------------------------------------------------------------------
    if results:
        dashboard_df = pd.DataFrame(results)
        # Rank by highest score
        dashboard_df = dashboard_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        
        # Color function for visual polish 🎨
        def color_signal(val):
            if val == "STRONG BUY":
                return "background-color: #1ed760; color: black; font-weight: bold;"  # Vibrant Green
            elif val == "BUY":
                return "background-color: #a6e22e; color: black;"                    # Light Green
            elif val == "WATCHLIST":
                return "background-color: #ffe600; color: black;"                    # Bright Yellow
            return ""

        # Apply styling to the Signal column
        styled_df = dashboard_df.style.map(color_signal, subset=["Signal"])
        
        # Timestamp for clarity 📅
        st.success(f"Scan complete! Identified {len(dashboard_df)} trading setups.")
        st.caption(f"Scan executed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
        
        # Render the polished table safely
        try:
            st.dataframe(styled_df, use_container_width=True)
        except Exception:
            st.write(styled_df)
    else:
        st.info("Scan complete. No setups met the mandatory bullish reversal criteria today.")
        # Apply styling to the Signal column
        styled_df = dashboard_df.style.map(color_signal, subset=["Signal"])
        
        # Timestamp for clarity 📅
        st.success(f"Scan complete! Identified {len(dashboard_df)} trading setups.")
        st.caption(f"Scan executed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
        
        # Render the polished table safely
        try:
            st.dataframe(styled_df, use_container_width=True)
        except Exception:
            st.write(styled_df)
    else:
        st.info("Scan complete. No setups met the mandatory bullish reversal criteria today.")
