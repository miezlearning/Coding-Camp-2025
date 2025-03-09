# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul dashboard
st.title("🌬️ Dashboard Kualitas Udara Aotizhongxin")

# Membaca data
df = pd.read_csv('../Dataset/PRSA_Data_Aotizhongxin_20130301-20170228.csv')  # Ganti dengan nama file CSV Anda

# Preprocessing
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df_aoti = df[df['station'] == 'Aotizhongxin'].copy()  # Gunakan .copy() untuk menghindari SettingWithCopyWarning
numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP']
for column in numeric_columns:
    df_aoti[column] = df_aoti[column].fillna(df_aoti[column].median())

# Sidebar untuk filter tahun
st.sidebar.header("Filter Data")
years = df_aoti['year'].unique()
selected_year = st.sidebar.multiselect("Pilih Tahun", options=years, default=years)

# Filter data berdasarkan tahun
filtered_df = df_aoti[df_aoti['year'].isin(selected_year)]

# Visualisasi 1: Tren PM2.5
st.subheader("Tren Rata-rata Tahunan PM2.5")
yearly_pm25 = filtered_df.groupby('year')['PM2.5'].mean()
fig1, ax1 = plt.subplots(figsize=(10, 6))
yearly_pm25.plot(kind='line', marker='o', ax=ax1)
ax1.set_title('Tren Rata-rata Tahunan PM2.5')
ax1.set_xlabel('Tahun')
ax1.set_ylabel('Konsentrasi PM2.5 (μg/m³)')
ax1.grid(True)
st.pyplot(fig1)

# Visualisasi 2: Scatter Plot Suhu vs PM2.5
st.subheader("Hubungan Suhu dan PM2.5")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(filtered_df['TEMP'], filtered_df['PM2.5'], alpha=0.1)
ax2.set_title('Hubungan antara Suhu dan PM2.5')
ax2.set_xlabel('Suhu (°C)')
ax2.set_ylabel('Konsentrasi PM2.5 (μg/m³)')
ax2.grid(True)
st.pyplot(fig2)

# Menampilkan korelasi
correlation = filtered_df['TEMP'].corr(filtered_df['PM2.5'])
st.write(f"**Korelasi antara Suhu dan PM2.5:** {correlation:.2f}")

# Menampilkan data jika dipilih
if st.checkbox("Tampilkan Data Mentah"):
    st.write("### Data Terfilter")
    st.dataframe(filtered_df)

# Tambahan: Statistik sederhana
st.subheader("Statistik Singkat")
st.write("Rata-rata PM2.5: {:.2f} μg/m³".format(filtered_df['PM2.5'].mean()))
st.write("Suhu Minimum: {:.2f} °C".format(filtered_df['TEMP'].min()))
st.write("Suhu Maksimum: {:.2f} °C".format(filtered_df['TEMP'].max()))