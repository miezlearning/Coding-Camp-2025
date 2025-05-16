import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transform import (
    transformasi_harga, transformasi_peringkat, transformasi_warna, 
    transformasi_kolom_teks, transformasi_data, KURS_USD_KE_IDR
)

@pytest.fixture
def df_mentah_sampel():
    data = {
        'Judul': ['Kaos A', 'Unknown Product', 'Hoodie B'], 'Harga': ['$10.00', '$5.00', '$49.88'],
        'Peringkat': ['Rating: ⭐ 4.5 / 5', 'Rating: ⭐ Invalid Rating / 5', 'Rating: ⭐ 4.8 / 5'],
        'Warna': ['3 Colors', '5 Colors', '3 Colors'], 'Ukuran': ['Size: M', 'Size: S', 'Size: L'],
        'Gender': ['Gender: Men', 'Gender: Women', 'Gender: Unisex'],
        'timestamp': [datetime.now().isoformat()] * 3
    }
    return pd.DataFrame(data)

def test_transformasi_harga_valid():
    assert transformasi_harga("$10.00") == 10.00 * KURS_USD_KE_IDR

def test_transformasi_harga_tidak_tersedia():
    assert pd.isna(transformasi_harga("Price Unavailable"))

def test_transformasi_peringkat_valid():
    assert transformasi_peringkat("Rating: ⭐ 4.5 / 5") == 4.5

def test_transformasi_peringkat_tidak_valid():
    assert pd.isna(transformasi_peringkat("Rating: ⭐ Invalid Rating / 5"))

def test_transformasi_warna_valid():
    assert transformasi_warna("3 Colors") == 3

def test_transformasi_kolom_teks():
    assert transformasi_kolom_teks("Size: M", "Size: ") == "M"

def test_transformasi_data_pipeline_lengkap(df_mentah_sampel):
    df_hasil = transformasi_data(df_mentah_sampel.copy())
    # Setelah dropna dan drop_duplicates, hanya Hoodie B yang valid
    assert len(df_hasil) == 1 
    assert df_hasil.iloc[0]['Judul'] == 'Hoodie B'
    assert df_hasil['Harga'].dtype == float and df_hasil['Peringkat'].dtype == float

def test_transformasi_data_input_kosong():
    df_kosong = pd.DataFrame(columns=['Judul', 'Harga'])
    assert transformasi_data(df_kosong).empty

def test_transformasi_data_kolom_hilang(df_mentah_sampel, caplog):
    df_tanpa_harga = df_mentah_sampel.drop(columns=['Harga'])
    transformasi_data(df_tanpa_harga.copy())
    assert "Kolom 'Harga' tidak ditemukan" in caplog.text