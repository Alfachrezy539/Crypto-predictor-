import streamlit as st
import pandas as pd
import yfinance as yf

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Crypto Analyzer", page_icon=":money_with_wings:", layout="wide", initial_sidebar_state="expanded")

# Title
st.title('Crypto Analyzer :money_with_wings:')
st.markdown("Cek sinyal Bullish/Bearish berdasarkan Moving Average sederhana.")

# Ambil data harga
def get_data(symbol):
    data = yf.download(symbol, period="7d", interval="1d")
    return data

# Hitung MA3 dan MA7
def calculate_ma(data):
    data['MA3'] = data['Close'].rolling(window=3).mean()
    data['MA7'] = data['Close'].rolling(window=7).mean()
    return data

# Tentukan status Bullish/Bearish
def get_status(row):
    if pd.isna(row['MA3']) or pd.isna(row['MA7']):
        return "WAIT"
    elif row['MA3'] > row['MA7']:
        return "BULLISH (BUY)"
    else:
        return "BEARISH (SELL)"

# Daftar koin
coins = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'PEPE-USD': 'Pepe',
    'XRP-USD': 'XRP',
    'HBAR-USD': 'Hedera'
}

# Load data semua koin
results = []
for symbol, name in coins.items():
    data = get_data(symbol)
    if not data.empty:
        data = calculate_ma(data)
        latest = data.iloc[-1]
        status = get_status(latest)
        results.append({
            'Coin': name,
            'Current Price (USD)': round(latest['Close'], 6),
            'MA 3 Hari': round(latest['MA3'], 6) if not pd.isna(latest['MA3']) else None,
            'MA 7 Hari': round(latest['MA7'], 6) if not pd.isna(latest['MA7']) else None,
            'Status': status
        })

# Tampilkan tabel
df = pd.DataFrame(results)

def highlight_status(val):
    if val == "BULLISH (BUY)":
        return 'background-color: #00FF0044; font-weight: bold;'  # hijau transparan
    elif val == "BEARISH (SELL)":
        return 'background-color: #FF000044; font-weight: bold;'  # merah transparan
    elif val == "WAIT":
        return 'background-color: #FFFF0044; font-weight: bold;'  # kuning transparan
    else:
        return ''

st.dataframe(df.style.applymap(highlight_status, subset=['Status']), use_container_width=True)

# Footer
st.caption("Data dari Yahoo Finance | App by Your Assistant")
