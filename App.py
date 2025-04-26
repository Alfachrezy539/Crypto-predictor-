import streamlit as st
import pandas as pd
import yfinance as yf

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Crypto Analyzer", page_icon=":money_with_wings:", layout="wide", initial_sidebar_state="expanded")

# Title
st.title('Crypto Analyzer :money_with_wings:')
st.markdown("Cek sinyal Bullish/Bearish berdasarkan Moving Average sederhana.")

# Fungsi untuk ambil data harga
def get_data(symbol):
    try:
        data = yf.download(symbol, period="7d", interval="1d", progress=False, threads=False)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

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

# Daftar koin yang dipantau
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
        if not data.empty and len(data) >= 1:
            latest = data.iloc[-1]
            status = get_status(latest)
            results.append({
                'Coin': name,
                'Current Price (USD)': round(latest['Close'], 6) if not pd.isna(latest['Close']) else None,
                'MA 3 Hari': round(latest['MA3'], 6) if not pd.isna(latest['MA3']) else None,
                'MA 7 Hari': round(latest['MA7'], 6) if not pd.isna(latest['MA7']) else None,
                'Status': status
            })
        else:
            st.warning(f"Data tidak cukup untuk {name}.")
    else:
        st.warning(f"Tidak bisa mengambil data untuk {name}.")

# Tampilkan tabel hasil
if results:
    df = pd.DataFrame(results)

    def highlight_status(val):
        if val == "BULLISH (BUY)":
            return 'background-color: #00FF0044; font-weight: bold;'
        elif val == "BEARISH (SELL)":
            return 'background-color: #FF000044; font-weight: bold;'
        elif val == "WAIT":
            return 'background-color: #FFFF0044; font-weight: bold;'
        else:
            return ''

    st.dataframe(df.style.applymap(highlight_status, subset=['Status']), use_container_width=True)
else:
    st.info("Belum ada data yang bisa ditampilkan.")

# Footer
st.caption("Data dari Yahoo Finance | App by Your Assistant")
