import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from streamlit_autorefresh import st_autorefresh

# Set page config
st.set_page_config(page_title="📈 Stock Predictor Dashboard", layout="wide")

# Custom style
st.markdown("""
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
    .css-1avcm0n.e1nzilvr5 {
        font-size: 28px !important;
        font-weight: 800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
selected = option_menu(
    menu_title="📈 MARKETMINDS",
    options=["Dashboard", "Prediction", "About"],
    icons=["bar-chart", "graph-up-arrow", "info-circle"],
    menu_icon="",
    default_index=0,
    orientation="horizontal"
)

# Compute RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fetch stock data
def fetch_data(ticker, period="3y"):
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.columns = df.columns.map(str)
    close_col = [col for col in df.columns if "close" in col.lower() or col.lower() == "close"]
    if not close_col:
        raise KeyError("Close column not found")
    close_col = close_col[0]
    df.rename(columns={close_col: "Close"}, inplace=True)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

# Prepare LSTM input
def prepare_data(df, feature_col='Close', look_back=60):
    data = df[[feature_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Predict next close
def predict_next(model, df, scaler, look_back=60):
    last_data = df[['Close']].values[-look_back:]
    scaled_last = scaler.transform(last_data)
    X_last = np.reshape(scaled_last, (1, look_back, 1))
    predicted = model.predict(X_last)
    return scaler.inverse_transform(predicted)

# Build model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 🔄 Animated Prediction Plot
def plot_interactive(df, train_pred, test_pred, scaler, look_back):
    train_pred_inv = scaler.inverse_transform(train_pred)
    test_pred_inv = scaler.inverse_transform(test_pred)
    df_plot = df[['Close']].copy()
    df_plot['Train Prediction'] = np.nan
    df_plot['Test Prediction'] = np.nan
    df_plot.iloc[look_back:look_back+len(train_pred_inv), df_plot.columns.get_loc('Train Prediction')] = train_pred_inv.flatten()
    df_plot.iloc[look_back+len(train_pred_inv):look_back+len(train_pred_inv)+len(test_pred_inv), df_plot.columns.get_loc('Test Prediction')] = test_pred_inv.flatten()

    frames = []
    for i in range(look_back, len(df_plot)):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=df_plot.index[:i], y=df_plot['Close'][:i], name='Actual Close', line=dict(color='#222f3e', width=3)),
                go.Scatter(x=df_plot.index[:i], y=df_plot['Train Prediction'][:i], name='Train Prediction', line=dict(color='#ff9f43', width=2, dash='dot')),
                go.Scatter(x=df_plot.index[:i], y=df_plot['Test Prediction'][:i], name='Test Prediction', line=dict(color='#0abde3', width=2, dash='dash')),
            ],
            name=str(i)
        ))

    fig = go.Figure(
        data=[
            go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Actual Close', line=dict(color='#222f3e', width=3)),
            go.Scatter(x=df_plot.index, y=df_plot['Train Prediction'], name='Train Prediction', line=dict(color='#ff9f43', width=2, dash='dot')),
            go.Scatter(x=df_plot.index, y=df_plot['Test Prediction'], name='Test Prediction', line=dict(color='#0abde3', width=2, dash='dash')),
        ],
        layout=go.Layout(
            title='📊 Prediction: Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            hovermode='x unified',
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {"label": "▶️ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]},
                        {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                    ],
                }
            ]
        ),
        frames=frames
    )

    st.plotly_chart(fig, use_container_width=True)

# Dashboard tab
if selected == "Dashboard":
    st_autorefresh(interval=30 * 1000, key="refresh_dashboard")

    st.title("📊 Stock Trend Dashboard")
    ticker = st.selectbox("Choose Stock Ticker", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    period = st.selectbox("Select Time Period", ["1y", "3y", "5y"])
    df = fetch_data(ticker, period)

    info = yf.Ticker(ticker).info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Current Price", f"${info.get('currentPrice', 0):.2f}")
    col2.metric("📊 Day High", f"${info.get('dayHigh', 0):.2f}")
    col3.metric("📉 Day Low", f"${info.get('dayLow', 0):.2f}")
    col4.metric("🔁 Volume", f"{info.get('volume', 0):,}")
    st.caption(f"⏱ Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.subheader("📈 Price with MA20")
    st.markdown("""
    *MA20 (20-Day Moving Average)* smooths price trends over 20 days,  
    helping you identify upward/downward movements and trend reversals.
    """)
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color="blue")))
    fig_live.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color="orange", dash="dot")))
    fig_live.update_layout(title="Live Stock Price Chart", xaxis_title="Date", yaxis_title="Price", template="plotly")
    st.plotly_chart(fig_live, use_container_width=True)

    st.subheader("📉 RSI")
    st.markdown("""
    *RSI* measures momentum.  
    - Above 70 = *Overbought* (may fall)  
    - Below 30 = *Oversold* (may rise)
    """)
    st.area_chart(df[['RSI']])

# Prediction tab
elif selected == "Prediction":
    st.title("🤖 Predict Future Stock Price")
    look_back = st.slider("🔁 Look-back Window", min_value=30, max_value=100, value=60)
    if st.button("✨ Predict & Visualize"):
        with st.spinner("⏳ Fetching and training model..."):
            df = fetch_data("AAPL", "3y")
            X, y, scaler = prepare_data(df, look_back=look_back)
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
            model = build_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            next_price = predict_next(model, df, scaler, look_back=look_back)
        st.subheader("🎯 Prediction Plot")
        plot_interactive(df, train_pred, test_pred, scaler, look_back)
        st.metric(label="🔮 Predicted Next Close", value=f"${next_price[0][0]:.2f}")

# About tab
elif selected == "About":
    st.title("📘 About This App")
    st.markdown("""
    - Built with *LSTM* to forecast stock trends  
    - Includes *MA20* and *RSI* indicators  
    - Interactive dashboard with Plotly graphs  
    - Created by *Riya Patel* 💖
    """)
