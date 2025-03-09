# -*- coding: utf-8 -*-
"""Proyek Analisis Data: Air Quality Dataset

# Proyek Analisis Data: Air Quality Dataset
- **Nama:** Muhammad Alif
- **Email:** m.alif7890@gmail.com
- **ID Dicoding:** miezlearning
"""

# Import Semua Packages/Library yang Digunakan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""## Menentukan Pertanyaan Bisnis

1. Bagaimana tren kualitas udara (PM2.5) di Aotizhongxin dari tahun 2013 hingga 2017?
2. Apa hubungan antara suhu (TEMP) dan konsentrasi PM2.5 di Aotizhongxin?
"""

"""## Data Wrangling"""

"""### Gathering Data"""
# Membaca data CSV
df = pd.read_csv('../Dataset/PRSA_Data_Aotizhongxin_20130301-20170228.csv')  # Ganti dengan nama file CSV Anda

# Menampilkan 5 baris pertama
print("5 baris pertama dataset:")
print(df.head())

"""**Insight:**
- Dataset berisi pengukuran kualitas udara per jam
- Terdapat berbagai parameter seperti PM2.5, PM10, SO2, NO2, CO, O3, dan faktor cuaca
"""

"""### Assessing Data"""
# Memeriksa info dataset
print("\nInfo dataset:")
print(df.info())

# Memeriksa missing values
print("\nMissing values:")
print(df.isnull().sum())

"""**Insight:**
- Terdapat beberapa missing values pada kolom seperti SO2, NO2, dll
- Tipe data perlu diperiksa dan mungkin perlu konversi
"""

"""### Cleaning Data"""
# Mengisi missing values dengan median untuk kolom numerik
numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP']
for column in numeric_columns:
    df[column].fillna(df[column].median(), inplace=True)

# Mengubah kolom tanggal menjadi datetime
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# Memfilter hanya data dari Aotizhongxin
df_aoti = df[df['station'] == 'Aotizhongxin']

"""**Insight:**
- Missing values telah diisi dengan median
- Data telah diformat menjadi datetime untuk analisis waktu
"""

"""## Exploratory Data Analysis (EDA)"""

# Statistik deskriptif
print("\nStatistik Deskriptif:")
print(df_aoti[['PM2.5', 'TEMP']].describe())

# Rata-rata tahunan PM2.5
yearly_pm25 = df_aoti.groupby('year')['PM2.5'].mean()

"""**Insight:**
- Rata-rata PM2.5 menunjukkan variasi antar tahun
- Suhu memiliki rentang dari sangat dingin hingga hangat
"""

"""## Visualization & Explanatory Analysis"""

"""### Pertanyaan 1: Tren Kualitas Udara (PM2.5)"""
plt.figure(figsize=(10, 6))
yearly_pm25.plot(kind='line', marker='o')
plt.title('Tren Rata-rata Tahunan PM2.5 di Aotizhongxin (2013-2017)')
plt.xlabel('Tahun')
plt.ylabel('Konsentrasi PM2.5 (μg/m³)')
plt.grid(True)
plt.show()

"""### Pertanyaan 2: Hubungan Suhu dan PM2.5"""
plt.figure(figsize=(10, 6))
plt.scatter(df_aoti['TEMP'], df_aoti['PM2.5'], alpha=0.1)
plt.title('Hubungan antara Suhu dan PM2.5')
plt.xlabel('Suhu (°C)')
plt.ylabel('Konsentrasi PM2.5 (μg/m³)')
plt.grid(True)
plt.show()

# Menghitung korelasi
correlation = df_aoti['TEMP'].corr(df_aoti['PM2.5'])
print(f"Korelasi antara TEMP dan PM2.5: {correlation:.2f}")

"""**Insight:**
- Pertanyaan 1: Terdapat tren fluktuasi PM2.5 dari tahun ke tahun
- Pertanyaan 2: Terdapat korelasi negatif lemah antara suhu dan PM2.5
"""

"""## Conclusion

- Conclusion pertanyaan 1: Konsentrasi PM2.5 di Aotizhongxin menunjukkan fluktuasi tahunan dengan beberapa tahun memiliki polusi yang lebih tinggi dibandingkan tahun lainnya.
- Conclusion pertanyaan 2: Terdapat korelasi negatif lemah antara suhu dan PM2.5, yang menunjukkan bahwa suhu yang lebih rendah cenderung berkorelasi dengan konsentrasi PM2.5 yang lebih tinggi, meskipun hubungannya tidak terlalu kuat.
"""