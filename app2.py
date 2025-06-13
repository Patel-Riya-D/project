import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from streamlit_autorefresh import st_autorefresh

# Set page config
st.set_page_config(page_title="📈 Stock Predictor Dashboard", layout="wide")

# Set background

def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #0e1117;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Price Chart", "Prediction"],
        icons=["house", "graph-up", "cpu"],
        menu_icon="cast",
        default_index=0,
    )

# Function to fetch stock data
def fetch_data(ticker, period="2y"):
    df = yf.download(ticker, period=period)
    df.dropna(inplace=True)
    return df

# Function to plot closing price with MA

def plot_data(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name="MA50", line=dict(color='lightgreen')))

    fig.update_layout(title="Stock Closing Price with Moving Averages",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Function to prepare data for LSTM
def prepare_data(df, look_back=60):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i - look_back:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to predict next price
def predict_next(model, df, scaler, look_back=60):
    last_60 = df['Close'].values[-look_back:].reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60)
    X_test = np.reshape(last_60_scaled, (1, look_back, 1))
    pred = model.predict(X_test)
    return scaler.inverse_transform(pred)

# Function to plot predictions
def plot_interactive(df, train_pred, test_pred, scaler, look_back):
    data = df['Close'].values.reshape(-1, 1)
    data_scaled = scaler.fit_transform(data)

    train_pred_plot = np.empty_like(data_scaled)
    train_pred_plot[:, :] = np.nan
    train_pred_plot[look_back:len(train_pred) + look_back, 0] = train_pred[:, 0]

    test_pred_plot = np.empty_like(data_scaled)
    test_pred_plot[:, :] = np.nan
    test_pred_plot[len(train_pred) + look_back:, 0] = test_pred[:, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scaler.inverse_transform(data_scaled)[:, 0], name="Actual", line=dict(color='white')))
    fig.add_trace(go.Scatter(y=scaler.inverse_transform(train_pred_plot)[:, 0], name="Train Prediction", line=dict(color='deepskyblue')))
    fig.add_trace(go.Scatter(y=scaler.inverse_transform(test_pred_plot)[:, 0], name="Test Prediction", line=dict(color='magenta')))

    fig.update_layout(title="Actual vs Predicted Stock Price",
                      xaxis_title="Days",
                      yaxis_title="Price",
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Home
if selected == "Home":
    st.title("📊 Welcome to Stock Predictor Dashboard")
    st.markdown("""
        This application uses an LSTM model to forecast future stock prices. 
        - Visualize historical trends
        - Predict future prices
        - Enjoy smooth animations and interactive charts
    """)

# Price Chart
elif selected == "Price Chart":
    st.title("📈 Price with MA20 & MA50")
    ticker = st.text_input("Enter Ticker", value="AAPL")
    if ticker:
        df = fetch_data(ticker)
        plot_data(df)

# Prediction
elif selected == "Prediction":
    st.title("🤖 Predict Future Stock Price")
    look_back = st.slider("🔁 Look-back Window", min_value=30, max_value=100, value=60)
    if st.button("✨ Predict & Visualize"):
        with st.spinner("⏳ Loading model and preparing data..."):
            df = fetch_data("AAPL", "3y")
            X, y, scaler = prepare_data(df, look_back=look_back)
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            # Load pre-trained model
            model = load_model("lstm_model.h5")

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            next_price = predict_next(model, df, scaler, look_back=look_back)

        st.subheader("🎯 Prediction Plot")
        plot_interactive(df, train_pred, test_pred, scaler, look_back)
        st.metric(label="🔮 Predicted Next Close", value=f"${next_price[0][0]:.2f}")
