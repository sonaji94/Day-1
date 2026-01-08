import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.trend import MACD

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="📈 Stock Price Prediction",
    layout="wide"
)

st.title("📈 Stock Price Prediction App")
st.markdown(
    """
    Predict **next-day closing prices** using  
    **Machine Learning + Technical Indicators**  
    _(For educational purposes only)_
    """
)

# =========================
# User Inputs
# =========================
with st.sidebar:
    st.header("🔧 User Inputs")

    ticker = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Examples: AAPL, TSLA, INFY.NS"
    )

    start_date = st.date_input(
        "Start Date",
        pd.to_datetime("2018-01-01")
    )

    end_date = st.date_input(
        "End Date",
        pd.to_datetime("today")
    )

    load_button = st.button("📥 Load Data")


# =========================
# Data Loading
# =========================
@st.cache_data
def load_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)


if load_button:
    with st.spinner("Fetching stock data..."):
        data = load_stock_data(ticker, start_date, end_date)

    if data.empty:
        st.error("❌ No data found. Check the stock symbol.")
        st.stop()

    st.subheader(f"📄 Latest Data for {ticker}")
    st.dataframe(data.tail())

    # =========================
    # Technical Indicators
    # =========================
    close_series = data["Close"].squeeze()

    data["RSI"] = RSIIndicator(close_series, window=14).rsi()

    macd = MACD(close_series)
    data["MACD"] = macd.macd()
    data["Signal_Line"] = macd.macd_signal()

    data["MA20"] = close_series.rolling(window=20).mean()
    data["MA50"] = close_series.rolling(window=50).mean()

    # =========================
    # Price Chart
    # =========================
    st.subheader("📉 Closing Price with Moving Averages")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["Close"], label="Close Price", linewidth=2)
    ax.plot(data["MA20"], label="MA 20", linestyle="--")
    ax.plot(data["MA50"], label="MA 50", linestyle="--")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price Trend")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # =========================
    # Machine Learning Dataset
    # =========================
    data["Tomorrow"] = data["Close"].shift(-1)
    data.dropna(inplace=True)

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "RSI", "MACD", "Signal_Line", "MA20", "MA50"
    ]

    X = data[features]
    y = data["Tomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # Model Training
    # =========================
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    # =========================
    # Model Evaluation
    # =========================
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    col1, col2 = st.columns(2)
    col1.metric("📉 Mean Squared Error", f"{mse:.2f}")
    col2.metric("📊 R² Score", f"{r2:.2f}")

    # =========================
    # Actual vs Predicted Plot
    # =========================
    st.subheader("🔮 Actual vs Predicted Prices")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(y_test.values, label="Actual Price", color="blue")
    ax2.plot(predictions, label="Predicted Price", color="red", linestyle="--")

    ax2.set_title("Model Prediction Performance")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.grid(True)
    ax2.legend()

    st.pyplot(fig2)

    # =========================
    # Next Day Prediction
    # =========================
    last_row_scaled = scaler.transform(X.iloc[-1].values.reshape(1, -1))
    next_day_price = model.predict(last_row_scaled)[0]

    st.success(
        f"📅 **Predicted Next-Day Closing Price:** **${next_day_price:.2f}**"
    )

    # =========================
    # Disclaimer
    # =========================
    st.markdown("---")
    st.info(
        "⚠️ This project is for **educational purposes only**. "
        "Stock markets are volatile and predictions may be inaccurate."
    )
