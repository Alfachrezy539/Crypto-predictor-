import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from babel.numbers import format_currency

st.set_page_config(layout="wide", page_title="Crypto Price Predictor", page_icon=":chart_with_upwards_trend:", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    .css-1d391kg {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Crypto Price Predictor (Random Forest + Top 50 Coins)")
st.caption("Prediksi harga 3 hari ke depan untuk 50 koin teratas menggunakan model Random Forest Regressor")

@st.cache_data(ttl=3600)
def get_top_coins(limit=50):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": False
    }
    res = requests.get(url, params=params).json()
    return res

@st.cache_data(ttl=3600)
def get_coin_history(coin_id, days=200):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    res = requests.get(url, params=params).json()
    if "prices" not in res:
        return None
    df = pd.DataFrame(res["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df.resample("1D").mean().dropna()
    return df

def predict_price(df):
    df = df.copy()
    df["Target"] = df["price"].shift(-3)
    df = df.dropna()

    df["Day"] = np.arange(len(df)).reshape(-1, 1)
    X = df[["Day"]]
    y = df["Target"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    next_day = np.array([[len(df)]])
    predicted = model.predict(next_day)[0]
    return predicted

@st.cache_data(ttl=3600)
def get_usd_to_idr():
    res = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=IDR").json()
    return res["rates"]["IDR"]

coins = get_top_coins(limit=50)
@st.cache_data
def get_usd_to_idr():
    try:
        url = "https://api.exchangerate.host/latest?base=USD"
        res = requests.get(url).json()
        if "rates" in res and "IDR" in res["rates"]:
            return res["rates"]["IDR"]
        else:
            st.warning("Gagal mendapatkan kurs USD ke IDR. Menggunakan default 16000.")
            return 16000
    except:
        st.warning("Terjadi kesalahan saat mengambil data kurs.")
        return 16000
usd_to_idr = get_usd_to_idr()

results = []
for coin in coins:
    coin_id = coin["id"]
    symbol = coin["symbol"].upper()
    name = coin["name"]
    price_usd = coin["current_price"]
    price_idr = price_usd * usd_to_idr

    hist = get_coin_history(coin_id)
    if hist is None or len(hist) < 30:
        continue

    predicted_usd = predict_price(hist)
    predicted_idr = predicted_usd * usd_to_idr

    results.append({
        "Coin": name,
        "Symbol": symbol,
        "Current Price (USD)": format_currency(price_usd, 'USD', locale='en_US'),
        "Current Price (IDR)": format_currency(price_idr, 'IDR', locale='id_ID'),
        "Predicted Price (USD, +3d)": format_currency(predicted_usd, 'USD', locale='en_US'),
        "Predicted Price (IDR, +3d)": format_currency(predicted_idr, 'IDR', locale='id_ID')
    })

df_result = pd.DataFrame(results)
st.dataframe(df_result, use_container_width=True)
