import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from streamlit_folium import folium_static
import os

os.system("pip install -r requirements.txt")


# Judul Dashboard
st.title("Dashboard Analisis Kualitas Udara di Aotizhongxin")

# Sidebar
st.sidebar.image("../Assets/udaralogo.png", caption="Analisis Kualitas Udara", use_container_width =True)  # Ganti "your_logo.png" dengan nama file logo Anda
st.sidebar.header("Tentang Dashboard")
st.sidebar.markdown("""
- **Nama:** Muhammad Alif  
- **Email:** m.alif7890@gmail.com  
- **ID Dicoding:** miezlearning  
""")
st.sidebar.markdown("""
Analisis mendalam kualitas udara Aotizhongxin (2013-2017):  
- Pola musiman PM2.5.  
- Korelasi dengan cuaca.  
- Binning & analisis geospatial berbasis arah angin.
""")

# Dropdown untuk memilih tahun
tahun_dipilih = st.sidebar.selectbox("Pilih Tahun", ["Semua"] + list(range(2013, 2018)))

# Memuat Dataset
st.header("Memuat Dataset")
uploaded_file = st.file_uploader("Unggah `PRSA_Data_Aotizhongxin_20130301-20170228.csv`", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Cleaning Data
    numeric_columns = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())
    df['wd'] = df['wd'].fillna(df['wd'].mode()[0])
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Musim Dingin'
        elif month in [3, 4, 5]:
            return 'Musim Semi'
        elif month in [6, 7, 8]:
            return 'Musim Panas'
        else:
            return 'Musim Gugur'
    df['season'] = df['month'].apply(get_season)
    df_aoti = df[df['station'] == 'Aotizhongxin'].copy()

    # Filter berdasarkan tahun yang dipilih
    if tahun_dipilih != "Semua":
        df_aoti = df_aoti[df_aoti['year'] == int(tahun_dipilih)]
        st.write(f"Data difilter untuk tahun {tahun_dipilih}:")
    else:
        st.write("Data berhasil dimuat (semua tahun):")
    st.dataframe(df_aoti.head())

    # Pertanyaan 1: Pola Musiman PM2.5
    st.header("Pertanyaan 1: Pola Musiman PM2.5")
    monthly_pm25 = df_aoti.groupby('month')['PM2.5'].mean()
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    monthly_pm25.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title(f'Rata-rata PM2.5 per Bulan (Tahun: {tahun_dipilih})', fontsize=14)
    ax1.set_xlabel('Bulan', fontsize=12)
    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'], rotation=0)
    ax1.axhline(y=25, color='red', linestyle='--', label='Batas Aman WHO')
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='season', y='PM2.5', data=df_aoti, palette='Set2', ax=ax2)
    ax2.set_title(f'Distribusi PM2.5 per Musim (Tahun: {tahun_dipilih})', fontsize=14)
    st.pyplot(fig2)
    st.markdown("**Insight:** PM2.5 tertinggi di musim dingin (Feb), terendah di musim panas.")

    # Pertanyaan 2: Korelasi Faktor Cuaca
    st.header("Pertanyaan 2: Korelasi Faktor Cuaca")
    weather_vars = ['TEMP', 'PRES', 'DEWP', 'WSPM', 'PM2.5']
    corr_matrix = df_aoti[weather_vars].corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax3)
    ax3.set_title(f'Matriks Korelasi (Tahun: {tahun_dipilih})', fontsize=14)
    st.pyplot(fig3)

    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.scatterplot(x='TEMP', y='PM2.5', data=df_aoti, alpha=0.3, ax=axes[0, 0])
    axes[0, 0].set_title('Suhu vs PM2.5')
    sns.scatterplot(x='PRES', y='PM2.5', data=df_aoti, alpha=0.3, ax=axes[0, 1])
    axes[0, 1].set_title('Tekanan Udara vs PM2.5')
    sns.scatterplot(x='DEWP', y='PM2.5', data=df_aoti, alpha=0.3, ax=axes[1, 0])
    axes[1, 0].set_title('Kelembapan vs PM2.5')
    sns.scatterplot(x='WSPM', y='PM2.5', data=df_aoti, alpha=0.3, ax=axes[1, 1])
    axes[1, 1].set_title('Kecepatan Angin vs PM2.5')
    plt.tight_layout()
    st.pyplot(fig4)
    st.markdown("**Insight:** DEWP berkorelasi positif kuat, TEMP dan PRES negatif.")

    # Analisis Lanjutan
    st.header("Analisis Lanjutan")

    # Binning PM2.5
    st.subheader("1. Binning PM2.5")
    bins = [0, 35, 75, float('inf')]
    labels = ['Rendah', 'Sedang', 'Tinggi']
    df_aoti['PM25_Category'] = pd.cut(df_aoti['PM2.5'], bins=bins, labels=labels, right=False)
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    df_aoti['PM25_Category'].value_counts().plot(kind='bar', color=['green', 'orange', 'red'], ax=ax5)
    ax5.set_title(f'Distribusi Kategori PM2.5 (Tahun: {tahun_dipilih})', fontsize=14)
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='PM25_Category', y='TEMP', data=df_aoti, palette='Set3', ax=ax6)
    ax6.set_title(f'Suhu berdasarkan Kategori PM2.5 (Tahun: {tahun_dipilih})', fontsize=14)
    st.pyplot(fig6)

    # Geospatial: PM2.5 berdasarkan Arah Angin
    st.subheader("2. Geospatial: PM2.5 berdasarkan Arah Angin")
    wind_pm25 = df_aoti.groupby('wd')['PM2.5'].mean().sort_values(ascending=False)
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    wind_pm25.plot(kind='bar', color='teal', ax=ax7)
    ax7.set_title(f'Rata-rata PM2.5 berdasarkan Arah Angin (Tahun: {tahun_dipilih})', fontsize=14)
    ax7.set_xlabel('Arah Angin', fontsize=12)
    ax7.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
    ax7.set_xticklabels(wind_pm25.index, rotation=45)
    st.pyplot(fig7)
    st.markdown("**Insight:** Kategori tinggi dominan, PM2.5 tinggi dari arah angin tertentu (misalnya NE).")

    # Conclusion
    st.header("Kesimpulan")
    st.markdown(f"""
    - **Pertanyaan 1**: Untuk tahun {tahun_dipilih}, PM2.5 memuncak di musim dingin (Feb), terendah di musim panas.
    - **Pertanyaan 2**: Kelembapan (DEWP) paling berkorelasi positif, suhu dan tekanan negatif.
    - **Analisis Lanjutan**: Polusi tinggi dominan, terkait suhu rendah dan arah angin tertentu.
    """)
else:
    st.warning("Unggah file CSV untuk memulai.")