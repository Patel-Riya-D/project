import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_data(ticker, period="3y"):
    df = yf.download(ticker, period=period, interval='1d', progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    # flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]

    close_cols = [col for col in df.columns if col.lower().startswith('close')]
    if not close_cols:
        raise ValueError(f"'Close' column not found for ticker '{ticker}'")

    close_col = close_cols[0]

    df_clean = pd.DataFrame()
    df_clean['Close'] = df[close_col]
    df_clean['MA20'] = df_clean['Close'].rolling(window=20).mean()
    df_clean['RSI'] = compute_rsi(df_clean['Close'])
    df_clean.dropna(inplace=True)
    return df_clean

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

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next(model, df, scaler, look_back=60):
    last_data = df[['Close']].values[-look_back:]
    scaled_last = scaler.transform(last_data)
    X_last = np.reshape(scaled_last, (1, look_back, 1))
    predicted = model.predict(X_last)
    return scaler.inverse_transform(predicted)

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    model.save("lstm_model.h5")
    print("✅ Model saved as lstm_model.h5")

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Add more tickers here

    for ticker in tickers:
        print(f"🚀 Training model for {ticker}")
        df = fetch_data(ticker)
        X, y, scaler = prepare_data(df)

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = build_model((X_train.shape[1], 1))
        train_model(model, X_train, y_train, X_test, y_test)

        model_filename = f"{ticker}_lstm_model.h5"
        model.save(model_filename)
        print(f"✅ Saved {model_filename}")

