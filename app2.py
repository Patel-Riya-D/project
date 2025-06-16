import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import time
import os

st.set_page_config(page_title="📈 Stock Predictor Dashboard", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&family=Raleway:wght@400;800&display=swap');
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e4ec 100%);
        font-family: 'Quicksand', sans-serif;
    }
    h1, h2, h3, .stTextInput > label, .stSelectbox > label, .stSlider > label {
        color: #1e272e;
        font-family: 'Raleway', sans-serif;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5em 1.2em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e55039;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
selected = option_menu(
    menu_title="📈 MARKETMINDS",
    options=["Dashboard", "Prediction", "About"],
    icons=["bar-chart", "graph-up-arrow", "info-circle"],
    default_index=0,
    orientation="horizontal"
)

# Helper Functions
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_data(ticker, period="3y"):
    df = yf.download(ticker, period=period, auto_adjust=True)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

def prepare_data(df, feature_col='Close', look_back=60):
    data = df[[feature_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X = [scaled_data[i-look_back:i, 0] for i in range(look_back, len(scaled_data))]
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

def predict_next(model, df, scaler, look_back=60):
    last_data = df[['Close']].values[-look_back:]
    scaled_last = scaler.transform(last_data)
    X_last = np.reshape(scaled_last, (1, look_back, 1))
    predicted = model.predict(X_last)
    return scaler.inverse_transform(predicted)

def plot_price_prediction(df, model, look_back):
    X, scaler = prepare_data(df, look_back=look_back)
    y_pred = model.predict(X)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    df_plot = df.copy()
    df_plot = df_plot.iloc[look_back:]
    df_plot['Prediction'] = y_pred_inv

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Prediction'], name="Predicted", line=dict(color="green", dash='dot')))
    fig.update_layout(title="Actual vs Predicted Stock Price", xaxis_title="Date", yaxis_title="Price", template="plotly")
    st.plotly_chart(fig, use_container_width=True)

# Dashboard Tab
if selected == "Dashboard":
    st.title("📊 Stock Trend Dashboard")
    ticker = st.selectbox("Choose Stock Ticker", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    period = st.selectbox("Select Time Period", ["1y", "3y", "5y"])
    df = fetch_data(ticker, period)

    try:
        info = yf.Ticker(ticker).info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💵 Current Price", f"${info.get('currentPrice', 0):.2f}")
        col2.metric("📊 Day High", f"${info.get('dayHigh', 0):.2f}")
        col3.metric("📉 Day Low", f"${info.get('dayLow', 0):.2f}")
        col4.metric("🔁 Volume", f"{info.get('volume', 0):,}")
    except Exception:
        st.error("⚠️ Could not fetch live market info due to rate limits.")

    st.caption(f"⏱ Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.subheader("📈 Price with MA20")
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color="blue")))
    fig_live.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color="orange", dash="dot")))
    fig_live.update_layout(title="Live Stock Price Chart", xaxis_title="Date", yaxis_title="Price", template="plotly")
    st.plotly_chart(fig_live, use_container_width=True)

    st.subheader("📉 RSI")
    st.area_chart(df[['RSI']])

# Prediction Tab
elif selected == "Prediction":
    st.title("🤖 Predict Future Stock Price")
    ticker = st.selectbox("📍 Select Stock", ["AAPL", "MSFT", "GOOGL", "AMZN"])
    look_back = st.slider("🔁 Look-back Window", min_value=30, max_value=100, value=60)

    model_path = f"{ticker.lower()}_model.h5"
    if not os.path.exists(model_path):
        st.warning(f"Model file `{model_path}` not found. Please upload it.")
    else:
        if st.button("✨ Predict & Visualize"):
            df = fetch_data(ticker, "3y")
            model = load_model(model_path)
            next_price = predict_next(model, df, prepare_data(df, look_back=look_back)[1], look_back=look_back)
            st.metric(label="🔮 Predicted Next Close", value=f"${next_price[0][0]:.2f}")
            st.subheader("📊 Prediction Chart")
            plot_price_prediction(df, model, look_back)

# About Tab
elif selected == "About":
    st.title("📘 About This App")
    st.markdown("""
    - Uses **LSTM** models to forecast stock prices.
    - Indicators: **MA20** (20-day moving average), **RSI** (momentum).
    - Interactive visualization using **Plotly**.
    - Models used: `aapl_model.h5`, `msft_model.h5`, `googl_model.h5`, `amzn_model.h5`.
    - Developed by **Riya Patel** 💖
    """)
