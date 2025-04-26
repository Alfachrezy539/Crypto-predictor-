import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# Setting tampilan dark mode
st.set_page_config(page_title="Crypto Analysis", page_icon=":chart_with_upwards_trend:", layout="wide", initial_sidebar_state="expanded")

# Judul
st.title("Crypto Realtime Analysis with Buy/Sell Signal")
st.markdown("By ChatGPT for you")

# Daftar koin yang dianalisis
coins = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "PEPE-USD", "HBAR-USD"]

# Range waktu data
end = datetime.datetime.now()
start = end - datetime.timedelta(days=14)  # 14 hari terakhir

# Fungsi untuk mengambil data dan menghitung MA
@st.cache_data(ttl=300)  # cache data selama 5 menit
def load_data(ticker):
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df['MA_3'] = df['Close'].rolling(window=3).mean()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    return df

# Membuat tabel utama
all_data = []
for coin in coins:
    df = load_data(coin)
    if not df.empty:
        price_now = df['Close'].iloc[-1]
        ma_3 = df['MA_3'].iloc[-1]
        ma_7 = df['MA_7'].iloc[-1]
        # Menentukan sinyal
        if ma_3 > ma_7:
            status = "BULLISH (BUY)"
        elif ma_3 < ma_7:
            status = "BEARISH (SELL)"
        else:
            status = "WAIT"
        all_data.append({
            "Coin": coin.replace("-USD", ""),
            "Current Price (USD)": round(price_now, 4),
            "MA 3 Hari": round(ma_3, 4),
            "MA 7 Hari": round(ma_7, 4),
            "Status": status
        })

# Tampilkan tabel
st.dataframe(pd.DataFrame(all_data).set_index("Coin"), height=450)

# Update note
st.caption("Update otomatis setiap 5 menit. Gunakan sinyal sebagai referensi awal, bukan keputusan akhir.")
