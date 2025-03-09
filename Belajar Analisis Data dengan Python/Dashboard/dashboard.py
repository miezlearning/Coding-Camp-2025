import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Judul Dashboard
st.title("Dashboard Analisis Kualitas Udara di Aotizhongxin")
st.markdown("""
- **Nama:** Muhammad Alif  
- **Email:** m.alif7890@gmail.com  
- **ID Dicoding:** miezlearning  
""")

# Sidebar untuk Informasi
st.sidebar.header("Tentang Dashboard")
st.sidebar.markdown("""
Dashboard ini menganalisis kualitas udara di Aotizhongxin (2013-2017) berdasarkan dataset PRSA.  
**Pertanyaan Bisnis:**  
1. Tren PM2.5 tahunan.  
2. Hubungan suhu dan PM2.5.  
**Analisis Lanjutan:** Distribusi kategori PM2.5.
""")

# Memuat Dataset
st.header("Memuat Dataset")
st.markdown("Dataset diunggah dari file CSV lokal.")
uploaded_file = st.file_uploader("Unggah file `PRSA_Data_Aotizhongxin_20130301-20170228.csv`", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Cleaning Data
    numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP']
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df_aoti = df[df['station'] == 'Aotizhongxin'].copy()

    st.write("Data berhasil dimuat dan dibersihkan. Berikut 5 baris pertama:")
    st.dataframe(df_aoti.head())

    # Pertanyaan 1: Tren PM2.5
    st.header("Pertanyaan 1: Tren Kualitas Udara (PM2.5)")
    yearly_pm25 = df_aoti.groupby('year')['PM2.5'].mean()
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    yearly_pm25.plot(kind='line', marker='o', color='teal', linewidth=2, ax=ax1)
    ax1.set_title('Tren Rata-rata Tahunan PM2.5 di Aotizhongxin (2013-2017)', fontsize=14, pad=10)
    ax1.set_xlabel('Tahun', fontsize=12)
    ax1.set_ylabel('Konsentrasi PM2.5 (μg/m³)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(yearly_pm25.index)
    st.pyplot(fig1)
    st.markdown("**Insight:** Tren menunjukkan fluktuasi, dengan beberapa tahun memiliki PM2.5 lebih tinggi dari ambang batas aman WHO (25 μg/m³).")

    # Pertanyaan 2: Hubungan Suhu dan PM2.5
    st.header("Pertanyaan 2: Hubungan Suhu dan PM2.5")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_aoti, x='TEMP', y='PM2.5', alpha=0.3, color='purple', ax=ax2)
    ax2.set_title('Hubungan antara Suhu dan PM2.5', fontsize=14, pad=10)
    ax2.set_xlabel('Suhu (°C)', fontsize=12)
    ax2.set_ylabel('Konsentrasi PM2.5 (μg/m³)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)
    
    correlation = df_aoti['TEMP'].corr(df_aoti['PM2.5'])
    st.write(f"**Korelasi antara TEMP dan PM2.5:** {correlation:.2f}")
    st.markdown("**Insight:** Korelasi negatif lemah. Suhu rendah cenderung berhubungan dengan PM2.5 tinggi, tetapi hubungan ini tidak kuat.")

    # Analisis Lanjutan: Binning PM2.5
    st.header("Analisis Lanjutan: Distribusi Kategori PM2.5")
    bins = [0, 35, 75, float('inf')]
    labels = ['Rendah', 'Sedang', 'Tinggi']
    df_aoti['PM25_Category'] = pd.cut(df_aoti['PM2.5'], bins=bins, labels=labels, right=False)
    category_counts = df_aoti['PM25_Category'].value_counts()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    category_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax3)
    ax3.set_title('Distribusi Kategori PM2.5', fontsize=14, pad=10)
    ax3.set_xlabel('Kategori', fontsize=12)
    ax3.set_ylabel('Jumlah Pengukuran', fontsize=12)
    ax3.set_xticklabels(labels, rotation=0)
    st.pyplot(fig3)
    st.markdown("**Insight:** Sebagian besar pengukuran PM2.5 berada pada kategori sedang hingga tinggi, menunjukkan tantangan kualitas udara yang konsisten.")

    # Conclusion
    st.header("Kesimpulan")
    st.markdown("""
    - **Pertanyaan 1**: Konsentrasi PM2.5 di Aotizhongxin menunjukkan fluktuasi tahunan dengan beberapa tahun memiliki polusi yang lebih tinggi dibandingkan tahun lainnya.
    - **Pertanyaan 2**: Terdapat korelasi negatif lemah antara suhu dan PM2.5, yang menunjukkan bahwa suhu yang lebih rendah cenderung berkorelasi dengan konsentrasi PM2.5 yang lebih tinggi, meskipun hubungannya tidak terlalu kuat.
    - **Analisis Lanjutan**: Binning menunjukkan distribusi polusi didominasi oleh kategori sedang dan tinggi, mengindikasikan masalah polusi yang konsisten.
    """)
else:
    st.warning("Silakan unggah file CSV untuk memulai analisis.")