import yfinance as yf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

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
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), verbose=1)

# Predict next close price
def predict_next(model, df, scaler, look_back=60):
    last_data = df[['Close']].values[-look_back:]
    scaled_last = scaler.transform(last_data)
    X_last = np.reshape(scaled_last, (1, look_back, 1))
    predicted = model.predict(X_last)
    return scaler.inverse_transform(predicted)

# Main logic
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    os.makedirs("models", exist_ok=True)

    for ticker in tickers:
        print(f"\n🚀 Training model for {ticker}")
        model_filename = os.path.join("models", f"{ticker}_lstm_model.h5")

        if os.path.exists(model_filename):
            print(f"✅ Model for {ticker} already exists at {model_filename}, skipping training.")
            continue

        df = fetch_data(ticker)
        X, y, scaler = prepare_data(df)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = build_model((X_train.shape[1], 1))
        train_model(model, X_train, y_train, X_test, y_test)

        model.save(model_filename)
        print(f"✅ Saved model to {model_filename}")

        next_price = predict_next(model, df, scaler)
        print(f"🔮 Predicted next close for {ticker}: ${next_price[0][0]:.2f}")
