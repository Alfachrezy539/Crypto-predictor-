import streamlit as st
import requests
import pandas as pd
import numpy as np

# 1) Halaman & Dark-mode CSS
st.set_page_config(
    page_title="CryptoPredictor Forecast",
    layout="wide",
)
st.markdown("""
    <style>
    body {background-color: #0E1117; color: #ECEDEF;}
    .stApp {background-color: #0E1117;}
    .css-1avcm0n, .css-1inwz65 {background-color: #1A1C23;}
    table, th, td {color: #ECEDEF !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("CryptoPredictor Forecast (Top 50 + 3-Day Prediction)")

# 2) Ambil data Top 50 koin dari CoinGecko
@st.cache_data(ttl=300)
def fetch_top_coins(vs_currency="idr", per_page=50):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    r = requests.get(url, params=params)
    return pd.DataFrame(r.json())

df = fetch_top_coins()

# 3) Status Bullish / Bearish
df["Status"] = df["price_change_percentage_24h"].apply(
    lambda x: "Bullish" if x and x > 0 else "Bearish"
)

# 4) Prediksi 3 hari ke depan
#    asumsi: perubahan harian rata-rata ≈ perubahan 24h terakhir
#    Prediksi price_tomorrow = current_price * (1 + pct24h/100)
df["pct24"] = df["price_change_percentage_24h"] / 100.0
for i in range(1, 4):
    df[f"Prediksi +{i} Hari (IDR)"] = (
        df["current_price"] * (1 + df["pct24"] * i)
    ).round(2)

# 5) Seleksi kolom & rename
display_cols = [
    "symbol", "current_price", "price_change_percentage_24h", "Status",
    "Prediksi +1 Hari (IDR)", "Prediksi +2 Hari (IDR)", "Prediksi +3 Hari (IDR)"
]
df_display = df[display_cols].rename(columns={
    "symbol": "Koin",
    "current_price": "Harga (IDR)",
    "price_change_percentage_24h": "24 h (%)",
})

# 6) Tampilkan
st.dataframe(df_display, use_container_width=True)

st.caption("Data: CoinGecko • Forecast: simple linear extrapolation of 24 h change")
