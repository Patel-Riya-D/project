import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import time

# Set page config
st.set_page_config(page_title="📈 Stock Predictor Dashboard", layout="wide")

# Custom CSS
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&family=Raleway:wght@400;800&display=swap');
    .stApp {{
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e4ec 100%);
        font-family: 'Quicksand', sans-serif;
    }}
    h1, h2, h3, .stTextInput > label, .stSelectbox > label, .stSlider > label {{
        color: #1e272e;
        font-family: 'Raleway', sans-serif;
    }}
    .stButton>button {{
        background-color: #ff6f61;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5em 1.2em;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #e55039;
        transform: scale(1.05);
    }}
    .css-1avcm0n.e1nzilvr5 {{
        font-size: 28px !important;
        font-weight: 800 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
selected = option_menu(
    menu_title="📈 MARKETMINDS",
    options=["Dashboard", "Prediction", "About"],
    icons=["bar-chart", "graph-up-arrow", "info-circle"],
    menu_icon="",
    default_index=0,
    orientation="horizontal"
)

# RSI Function
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
    close_col = [col for col in df.columns if "close" in col.lower()]
    if not close_col:
        raise KeyError("Close column not found")
    df.rename(columns={close_col[0]: "Close"}, inplace=True)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

# Prepare LSTM data
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

# Predict next value
def predict_next(model, df, scaler, look_back=60):
    last_data = df[['Close']].values[-look_back:]
    scaled_last = scaler.transform(last_data)
    X_last = np.reshape(scaled_last, (1, look_back, 1))
    predicted = model.predict(X_last)
    return scaler.inverse_transform(predicted)

# LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Animated Plot
def plot_interactive(df, train_pred, test_pred, scaler, look_back):
    train_pred_inv = scaler.inverse_transform(train_pred)
    test_pred_inv = scaler.inverse_transform(test_pred)
    df_plot = df[['Close']].copy()
    df_plot['Train Prediction'] = np.nan
    df_plot['Test Prediction'] = np.nan
    df_plot.iloc[look_back:look_back+len(train_pred_inv), df_plot.columns.get_loc('Train Prediction')] = train_pred_inv.flatten()
    df_plot.iloc[look_back+len(train_pred_inv):look_back+len(train_pred_inv)+len(test_pred_inv), df_plot.columns.get_loc('Test Prediction')] = test_pred_inv.flatten()

    frames = [
        go.Frame(
            data=[
                go.Scatter(x=df_plot.index[:k], y=df_plot['Close'].iloc[:k], mode='lines', name='Actual Close'),
                go.Scatter(x=df_plot.index[:k], y=df_plot['Train Prediction'].iloc[:k], mode='lines', name='Train Prediction'),
                go.Scatter(x=df_plot.index[:k], y=df_plot['Test Prediction'].iloc[:k], mode='lines', name='Test Prediction')
            ]
        ) for k in range(look_back+1, len(df_plot))
    ]

    fig = go.Figure(
        data=[
            go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Actual Close'),
            go.Scatter(x=df_plot.index, y=df_plot['Train Prediction'], name='Train Prediction'),
            go.Scatter(x=df_plot.index, y=df_plot['Test Prediction'], name='Test Prediction')
        ],
        layout=go.Layout(
            title='📊 Animated Prediction: Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            hovermode='x unified',
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20}, 'fromcurrent': True}]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }]
        ),
        frames=frames
    )
    st.plotly_chart(fig, use_container_width=True)

# Main App Logic
if selected == "Dashboard":
    st.title("📊 Stock Trend Dashboard")
    ticker = st.selectbox("Choose Stock Ticker", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    period = st.selectbox("Select Time Period", ["1y", "3y", "5y"])

    try:
        df_live = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not df_live.empty:
            latest_price = float(df_live["Close"].iloc[-1])
            day_high = float(df_live["High"].max())
            day_low = float(df_live["Low"].min())
            volume = int(df_live["Volume"].sum())
        else:
            latest_price = day_high = day_low = volume = 0.0
    except:
        latest_price = day_high = day_low = volume = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Current Price", f"${latest_price:.2f}" if latest_price else "N/A")
    col2.metric("📊 Day High", f"${day_high:.2f}" if day_high else "N/A")
    col3.metric("📉 Day Low", f"${day_low:.2f}" if day_low else "N/A")
    col4.metric("🔁 Volume", f"{volume:,}" if volume else "N/A")

    st.caption(f"⏱ Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = fetch_data(ticker, period)

    st.subheader("📈 MA20 - 20-Day Moving Average")
    st.markdown("The **MA20** smooths short-term fluctuations to highlight longer-term trends in stock price. It is the average of the closing prices over the last 20 days.")
    fig_ma20 = go.Figure()
    fig_ma20.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color="blue")))
    fig_ma20.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color="orange", dash="dot")))
    fig_ma20.update_layout(title="Price with MA20", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_ma20, use_container_width=True)

    st.subheader("📉 RSI - Relative Strength Index")
    st.markdown("**RSI** indicates whether a stock is overbought or oversold, using price momentum. Values above 70 may indicate overbought, while below 30 suggest oversold.")
    st.area_chart(df[['RSI']])

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

elif selected == "About":
    st.title("📘 About This App")
    st.markdown("""
    - Built with **LSTM** to forecast stock trends  
    - Includes **MA20** and **RSI** indicators  
    - Interactive dashboard with Plotly animations  
    - Developed by Riya Patel 💖
    """)
