import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

st.set_page_config(page_title="Crypto Predictor", layout="wide", initial_sidebar_state="expanded")

# DARK MODE
dark_mode_css = """
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-18e3th9 {
        background-color: #0e1117;
    }
    table {
        color: #ffffff !important;
    }
    </style>
"""
st.markdown(dark_mode_css, unsafe_allow_html=True)

# === Helper Functions ===

@st.cache_data(ttl=3600)
def fetch_top_coins(n=50):
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": n, "page": 1}
    res = requests.get(url, params=params)
    return res.json()

@st.cache_data(ttl=3600)
def fetch_price_history(coin_id, vs_currency="usd", days=200):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    res = requests.get(url, params=params).json()
    if "prices" not in res:
        return None
    prices = pd.DataFrame(res["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    prices = prices.set_index("timestamp")
    prices = prices.rename(columns={"price": "Close"})
    return prices

def predict_price_rf(prices, days_forward=3):
    df = prices.copy()
    df['Target'] = df['Close'].shift(-days_forward)
    df.dropna(inplace=True)

    X = []
    y = []

    for i in range(30, len(df)):
        X.append(df['Close'].iloc[i-30:i].values)
        y.append(df['Target'].iloc[i])

    X, y = np.array(X), np.array(y)
    if len(X) < 10:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    last_30 = df['Close'].iloc[-30:].values.reshape(1, -1)
    prediction = model.predict(last_30)[0]

    mse = mean_squared_error(y_test, model.predict(X_test))
    return prediction, mse

# === UI ===

st.title("Crypto Price Predictor (Random Forest + Top 50 Coins)")
st.markdown("Prediksi harga 3 hari ke depan untuk 50 koin teratas menggunakan model Random Forest Regressor.")

# === Proses koin ===

coins = fetch_top_coins(50)
data_rows = []

for coin in coins:
    coin_id = coin["id"]
    name = coin["name"]
    symbol = coin["symbol"].upper()
    price = coin["current_price"]
    change_24h = coin["price_change_percentage_24h"]

    status = "Bullish" if change_24h and change_24h > 0 else "Bearish"

    hist = fetch_price_history(coin_id, days=200)
    if hist is None or len(hist) < 60:
        continue

    pred_price, mse = predict_price_rf(hist, days_forward=3)
    if pred_price is None:
        continue

    last_close = hist["Close"].iloc[-1]
    diff_pct = ((pred_price - last_close) / last_close) * 100

    trend = "Bullish" if pred_price > last_close else "Bearish"

    data_rows.append({
        "Coin": name,
        "Symbol": symbol,
        "Current Price": f"${price:,.4f}",
        "Predicted Price (3d)": f"${pred_price:,.4f}",
        "Change (3d)": f"{diff_pct:.2f}%",
        "Status": status,
        "Predicted Trend": trend
    })

# === Tampilkan tabel ===

df_result = pd.DataFrame(data_rows)
if not df_result.empty:
    st.dataframe(df_result.style.applymap(
        lambda v: "color: green" if isinstance(v, str) and "Bullish" in v else "color: red"
    ))
else:
    st.warning("Tidak ada data yang berhasil diproses. Mungkin karena API sedang down atau data historis tidak lengkap.")
