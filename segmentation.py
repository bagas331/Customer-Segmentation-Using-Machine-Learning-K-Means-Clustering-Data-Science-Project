import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Judul aplikasi
st.title("Aplikasi Segmentasi Pelanggan")
st.write("Masukkan data pelanggan untuk memprediksi segmen")

# Form input pengguna
umur = st.number_input("Umur", min_value=18, max_value=100, value=35)
pendapatan = st.number_input("Pendapatan ($)", min_value=0, max_value=200000, value=50000)
total_belanja = st.number_input("Total Belanja (jumlah seluruh pembelian)($)", min_value=0, max_value=2000, value=1000)
jumlah_pembelian_web = st.number_input("Jumlah Pembelian via Web", min_value=0, max_value=100, value=10)
jumlah_pembelian_toko = st.number_input("Jumlah Pembelian di Toko", min_value=0, max_value=100, value=10)
jumlah_kunjungan_web = st.number_input("Jumlah Kunjungan ke Web", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (hari sejak pembelian terakhir)", min_value=0, max_value=365, value=30)

# Masukkan data ke dalam DataFrame
input_data = pd.DataFrame({ 
    "Age": [umur],
    "Income": [pendapatan],
    "Total_Spending": [total_belanja],
    "NumWebPurchases": [jumlah_pembelian_web],
    "NumStorePurchases": [jumlah_pembelian_toko],
    "NumWebVisitsMonth": [jumlah_kunjungan_web],
    "Recency": [recency]
})

# Scaling input
input_scaled = scaler.transform(input_data)

# Prediksi segmen
if st.button("Prediksi Segmen"):
    cluster = kmeans.predict(input_scaled)
    st.write(f"Segmen yang Diprediksi: {cluster[0]}")

    # Keterangan berdasarkan cluster
    keterangan_segmen = {
    0: "Pelanggan Premium — Pendapatan tinggi & Belanja tinggi",
    1: "Pelanggan Umum — Aktivitas rata-rata di berbagai kanal, perilaku belanja moderat",
    2: "Pembeli Digital — Sering belanja online, jarang belanja di toko fisik",
    3: "Pelanggan Fisik — Lebih sering belanja di toko dibandingkan online",
    4: "Pelanggan Baru — Baru melakukan pembelian, potensi untuk dilibatkan lebih jauh",
    5: "Pelanggan Tidak Aktif — Sudah lama tidak melakukan pembelian",
    6: "Pelanggan Hemat — Pendapatan rendah & Belanja rendah"
}
    st.subheader("Keterangan Segmen:")

