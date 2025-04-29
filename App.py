import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Crypto Analyzer", layout="wide")
st.title("Crypto Analyzer - Top 10 Coins with Prediction")
st.markdown("Menampilkan 10 koin teratas, status bullish/bearish, dan prediksi jangka pendek.")

@st.cache_data(show_spinner=False)
def fetch_top_coins(limit=10):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1}
    return requests.get(url, params=params).json()

@st.cache_data(show_spinner=False)
def fetch_price_history(coin_id, days=200, vs_currency="usd"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    res = requests.get(url, params=params).json()
    if "prices" not in res:
        return pd.DataFrame()
    prices = pd.DataFrame(res["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    prices = prices.set_index("timestamp")
    prices = prices.rename(columns={"price": "Close"})
    return prices

@st.cache_data(show_spinner=False)
def predict_next_days(prices, days_ahead=3):
    if len(prices) < days_ahead:
        return None
    df = prices.reset_index()
    df["t"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["t"]], df["Close"])
    future_t = np.array([[len(df) + i] for i in range(days_ahead)])
    preds = model.predict(future_t)
    return round(preds[-1], 4)

# Ambil data top coins
coins = fetch_top_coins(10)

rows = []
for coin in coins:
    coin_id = coin["id"]
    symbol = coin["symbol"].upper()
    name = coin["name"]
    current = coin["current_price"]
    pct24 = coin["price_change_percentage_24h"]
    status = "Bullish" if pct24 > 0 else "Bearish"
    
    hist = fetch_price_history(coin_id)
    if hist.empty:
        pred = "N/A"
    else:
        pred = predict_next_days(hist)

    rows.append({
        "Coin": f"{name} ({symbol})",
        "Harga Sekarang": f"${current:,.4f}",
        "24h %": f"{pct24:.2f}%",
        "Status": status,
        "Prediksi 3 Hari Kedepan": f"${pred}" if pred != "N/A" else "Data tidak cukup"
    })

# Tampilkan hasil
df_result = pd.DataFrame(rows)
st.dataframe(df_result.style.applymap(
    lambda x: "color:green" if "Bullish" in str(x) else ("color:red" if "Bearish" in str(x) else ""),
    subset=["Status"]
))
