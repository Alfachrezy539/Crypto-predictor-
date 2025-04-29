import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ——————————————————————————————
# 1) CONFIG STREAMLIT & DARK MODE
# ——————————————————————————————
st.set_page_config(
    page_title="CryptoPredictor Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
    <style>
      body {background-color: #0E1117; color: #ECEDEF;}
      .stApp {background-color: #0E1117;}
      .css-1avcm0n, .css-1inwz65 {background-color: #1A1C23;}
      table, th, td {color: #ECEDEF !important;}
    </style>
""", unsafe_allow_html=True)
st.title("🚀 CryptoPredictor Pro — Top 20 + Short & Mid-Term Forecast")

# ——————————————————————————————
# 2) HELPERS: API & DATAFRAME
# ——————————————————————————————
@st.cache_data(ttl=600)
def fetch_top_coins(vs_currency="idr", per_page=20):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    res = requests.get(url, params=params).json()
    df = pd.DataFrame(res)
    return df

@st.cache_data(ttl=600)
def fetch_price_history(coin_id: str, vs_currency="idr", days=200):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    res = requests.get(url, params=params).json()
    prices = pd.DataFrame(res["prices"], columns=["timestamp","price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    prices = prices.set_index("timestamp")
    prices = prices.rename(columns={"price":"Close"})
    return prices

def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    # SMA & EMA
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    # MACD
    df["MACD"]    = df["EMA12"] - df["EMA26"]
    df["Signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df["BB_mid"]  = df["Close"].rolling(20).mean()
    df["BB_std"]  = df["Close"].rolling(20).std()
    df["BB_up"]   = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_dn"]   = df["BB_mid"] - 2 * df["BB_std"]
    # RSI14
    delta = df["Close"].diff()
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100/(1+rs))
    df.dropna(inplace=True)
    return df

def predict_linear(df: pd.DataFrame, days=3):
    # gunakan index ordinal sebagai X
    X = np.arange(len(df)).reshape(-1,1)
    y = df["Close"].values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(df), len(df)+days).reshape(-1,1)
    preds = model.predict(future_X)
    return preds

# ——————————————————————————————
# 3) BUILD TABEL UTAMA
# ——————————————————————————————
top = fetch_top_coins()
rows = []
for _, row in top.iterrows():
    coin_id = row["id"]
    symbol  = row["symbol"].upper()
    price   = row["current_price"]
    pct24   = row["price_change_percentage_24h"]
    status  = "Bullish" if pct24>0 else "Bearish"

    # tarik historis & indikator
    hist = fetch_price_history(coin_id, days=200)
    ind  = compute_indicators(hist)

    # short-term prediksi linear regression 3 hari
    preds = predict_linear(ind, days=3)
    p1,p2,p3 = [round(x,2) for x in preds]

    # medium-term sinyal
    macd_signal = "MT Buy" if ind["MACD"].iloc[-1] > ind["Signal"].iloc[-1] else "MT Sell"
    rsi_signal  = "Oversold" if ind["RSI14"].iloc[-1]<30 else ("Overbought" if ind["RSI14"].iloc[-1]>70 else "Neutral")
    bb_signal   = ("Touch Lower" if price < ind["BB_dn"].iloc[-1]
                   else "Touch Upper" if price > ind["BB_up"].iloc[-1]
                   else "Mid Band")

    # gabung jadi rekomendasi
    rec = []
    # ST recommendation
    if preds[0]/price -1 > 0.015: rec.append("ST Buy")
    elif preds[0]/price -1 < -0.015: rec.append("ST Sell")
    else: rec.append("ST Wait")
    # MT recommendation
    if macd_signal=="MT Buy" and rsi_signal!="Overbought": rec.append("MT Buy")
    elif macd_signal=="MT Sell": rec.append("MT Sell")
    else: rec.append("MT Wait")

    rows.append({
        "Koin": symbol,
        "Harga (IDR)": price,
        "24 h (%)": pct24,
        "Status": status,
        "Pred +1 hri": p1,
        "Pred +2 hri": p2,
        "Pred +3 hri": p3,
        "SMA50": round(ind["SMA50"].iloc[-1],2),
        "SMA200": round(ind["SMA200"].iloc[-1],2),
        "MACD": round(ind["MACD"].iloc[-1],4),
        "Signal": round(ind["Signal"].iloc[-1],4),
        "RSI14": round(ind["RSI14"].iloc[-1],2),
        "BB Lower": round(ind["BB_dn"].iloc[-1],2),
        "BB Upper": round(ind["BB_up"].iloc[-1],2),
        "Reco": " & ".join(rec)
    })

df_out = pd.DataFrame(rows)

# ——————————————————————————————
# 4) RENDER & STYLING
# ——————————————————————————————
def highlight_reco(val):
    if "ST Buy" in val: return "background-color: #00FF0044;"
    if "MT Buy" in val: return "background-color: #00770044;"
    if "ST Sell" in val: return "background-color: #FF000044;"
    return ""

st.dataframe(
    df_out.style
          .applymap(highlight_reco, subset=["Reco"])
          .format({
              "Harga (IDR)": "{:,.0f}",
              "24 h (%)": "{:+.2f}",
              "Pred +1 hri": "{:,.0f}",
              "Pred +2 hri": "{:,.0f}",
              "Pred +3 hri": "{:,.0f}"
          })
    , use_container_width=True
)

st.caption("Data: CoinGecko | ST: short-term regression | MT: MACD/RSI/Bollinger")
