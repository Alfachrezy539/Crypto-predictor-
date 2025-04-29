import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

def calculate_moving_averages(data):
    data['MA3'] = data['Close'].rolling(window=3).mean()
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data = data.dropna()
    return data

def get_status(row):
    if 'MA3' not in row or 'MA7' not in row:
        return 'No data'
    if pd.isna(row['MA3']) or pd.isna(row['MA7']):
        return 'No data'
    return 'Bullish' if row['MA3'] > row['MA7'] else 'Bearish'

def predict_future_prices(data, days=3):
    # Gunakan rata-rata selisih harga penutupan harian untuk prediksi sederhana
    data['diff'] = data['Close'].diff()
    avg_change = data['diff'].mean()
    last_price = data['Close'].iloc[-1]

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
    predictions = [last_price + avg_change * (i + 1) for i in range(days)]

    pred_df = pd.DataFrame({'Predicted Close': predictions}, index=future_dates)
    return pred_df

def main():
    st.set_page_config(page_title="Crypto Analyzer + Prediksi", layout="wide")
    st.title('Crypto Analyzer + Prediksi Harga Sederhana')

    symbol = st.text_input('Masukkan symbol crypto (contoh: BTC-USD, ETH-USD, XRP-USD)', 'BTC-USD')
    period = st.selectbox('Pilih periode data', ['7d', '14d', '30d', '60d', '90d', '180d', '1y'], index=2)
    interval = st.selectbox('Pilih interval', ['1h', '2h', '4h', '1d'], index=3)

    if st.button('Analisa & Prediksi'):
        with st.spinner('Mengambil data...'):
            try:
                data = yf.download(symbol, period=period, interval=interval)
                if data.empty:
                    st.error('Data tidak ditemukan.')
                    return

                data = calculate_moving_averages(data)
                data['Status'] = data.apply(get_status, axis=1)

                st.subheader('Data Harga + MA + Status')
                st.dataframe(data[['Close', 'MA3', 'MA7', 'Status']])

                st.subheader('Status Terakhir:')
                st.success(f"{symbol} sedang: {data.iloc[-1]['Status']}")

                st.line_chart(data[['Close', 'MA3', 'MA7']])

                # Prediksi
                pred_df = predict_future_prices(data)
                st.subheader('Prediksi Harga 3 Hari ke Depan:')
                st.dataframe(pred_df)
                st.line_chart(pred_df)

            except Exception as e:
                st.error(f"Terjadi error: {e}")

if __name__ == "__main__":
    main()
