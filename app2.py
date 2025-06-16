import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Predictor Dashboard", layout="wide")

st.title("📊 AI-Powered Stock Prediction Dashboard")

# Sidebar
st.sidebar.title("Select Stock and Date Range")
ticker = st.sidebar.selectbox("Choose Stock", ["AAPL", "MSFT", "GOOGL", "AMZN"])
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Load data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # RSI calculation
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

df = load_data(ticker, start_date, end_date)

# Live Market Data
st.subheader(f"📈 Live Market Metrics for {ticker}")
try:
    ticker_obj = yf.Ticker(ticker)
    df_live = ticker_obj.history(period="1d", interval="1m")
    if not df_live.empty:
        latest_price = float(df_live["Close"].iloc[-1])
        day_high = float(df_live["High"].max())
        day_low = float(df_live["Low"].min())
        volume = int(df_live["Volume"].sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💵 Current Price", f"${latest_price:.2f}")
        col2.metric("📊 Day High", f"${day_high:.2f}")
        col3.metric("📉 Day Low", f"${day_low:.2f}")
        col4.metric("🔁 Volume", f"{volume:,}")
    else:
        st.warning("⚠️ Could not fetch live market info.")
except Exception as e:
    st.warning("⚠️ Could not fetch live market info due to rate limits.")
st.caption(f"⏱ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# MA20 Section
st.markdown("## 📈 Price with MA20")
st.markdown("**MA20 (20-Day Moving Average)** helps smooth price action and identify trends.")
fig_ma20 = go.Figure()
fig_ma20.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price"))
fig_ma20.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color='orange')))
fig_ma20.update_layout(title="Stock Price vs 20-Day Moving Average", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_ma20, use_container_width=True)

# RSI Section
st.markdown("## 📉 RSI (Relative Strength Index)")
st.markdown("**RSI** indicates overbought (>70) or oversold (<30) conditions in the market.")
if "RSI" in df.columns:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color='green')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
    fig_rsi.update_layout(title="RSI (Relative Strength Index)", xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)
else:
    st.error("❌ RSI data not available. Try changing the date range.")

# Footer
st.markdown("---")
st.caption("🚀 Built with ❤️ using Streamlit | Data by Yahoo Finance")
