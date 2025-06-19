# project
# 📈 Stock Price Trend Prediction with LSTM

This project utilizes a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical market data. It features an interactive Streamlit dashboard, technical indicators like Moving Average and RSI, and visual comparisons between actual and predicted prices.

---

## 🎯 Objective

To develop a predictive system that analyzes past stock trends to forecast future prices using LSTM and visualize the results using charts and a user-friendly dashboard.

---

## 🚀 Features

- 🔮 Predict future stock prices using pre-trained **LSTM models**
- 📊 Live price charts with **MA20 (20-day Moving Average)**
- 💹 **RSI (Relative Strength Index)** indicator with buy/sell signals
- 🎨 Interactive charts using **Plotly**
- 🌐 Deployed using **Render**

---

## 🛠️ Tools & Technologies

| Tool/Library            | Purpose                                |
|-------------------------|----------------------------------------|
| Python 3.10+            | Core programming language              |
| Keras / TensorFlow      | Deep learning framework (LSTM model)   |
| Pandas, NumPy           | Data handling and manipulation         |
| Matplotlib / Plotly     | Data visualization                     |
| yfinance                | Fetch historical stock data            |
| Streamlit               | Dashboard and web app interface        |
| streamlit-option-menu   | Sidebar navigation for Streamlit       |

---

## 📘 Methodology

# ✅ Step 1: Data Collection
* Download historical stock data using the yfinance API.

* Example: AAPL (Apple), MSFT (Microsoft), AMZN (Amazon), GOOGL (Alphabet).

# ✅ Step 2: Data Preprocessing
* Normalize close prices using MinMaxScaler.

* Create sequences of 60 days of prices as input, and the 61st as target.

* Split into training and testing sets.

# ✅ Step 3: Build LSTM Model
* A Sequential model with:

* 2 LSTM layers

* 1 Dense layer to output the predicted price

# ✅ Step 4: Model Training & Evaluation
* Compile and train the model on training data.

* Predict on test data.

* Evaluate using visual plots and metrics.

# ✅ Step 5: Add Technical Indicators
* Moving Average for trend smoothing.

* RSI (Relative Strength Index) for identifying overbought/oversold zones.

# ✅ Step 6: Streamlit Dashboard
* A clean interactive interface with:

* Sidebar for stock selection

* Tabs for viewing predictions, MA, RSI

* Live graphs using Plotly

## 📊 Visual Results

# 📍 Predicted vs Actual Prices

* Shows how accurately the LSTM model predicts stock prices based on test data.

![Animated Prediction Plot](https://github.com/Patel-Riya-D/project/blob/main/prediction_plot.png)

# 📍 Moving Average Plot (MA20)

* Displays the 50-day and 100-day moving averages to analyze long-term trends.

![MA20 Plot](https://github.com/Patel-Riya-D/project/blob/main/MA20.png)

# 📍 RSI (Relative Strength Index) Plot

* Helps visualize overbought (above 70) and oversold (below 30) conditions.

![RSI Plot](https://github.com/Patel-Riya-D/project/blob/main/RSI_plot.png)

## 📁 Project Structure

```

📦 stock-price-lstm/
├── app2.py                  # Streamlit dashboard app
├── stock_predictor.ipynb    # Jupyter notebook for LSTM model
├── model.h5                 # Trained LSTM model weights
├── requirements.txt         # Python dependencies
├── screenshots/             # Images of graphs and output
│   ├── prediction_plot.png
│   ├── MA20.png
│   └── RSI_plot.png
└── README.md                # This documentation file

```

## 🌍 Live Demo

🟢 Deployed at: https://project-snfk.onrender.com/
