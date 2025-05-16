import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
BASE_DIR_TRANSF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR_TRANSF)

from utils.transform import (
    transformasi_harga, transformasi_peringkat, transformasi_warna,
    transformasi_kolom_teks, transformasi_data, KURS_USD_KE_IDR
)

@pytest.fixture
def df_mentah_sampel():
    data = {
        'Judul': ['Kaos A', 'Unknown Product', 'Hoodie B'],
        'Harga': ['$10.00', '$5.00', '$49.88'],
        'Peringkat': ['Rating: ⭐ 4.5 / 5', 'Rating: ⭐ Invalid Rating / 5', 'Rating: ⭐ 4.8 / 5'],
        'Warna': ['3 Colors', '5 Colors', '3 Colors'],
        'Ukuran': ['Size: M', 'Size: S', 'Size: L'],
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
    # Baris 'Unknown Product' / 'Invalid Rating' akan di-drop.
    # Menyisakan 'Kaos A' dan 'Hoodie B'.
    assert len(df_hasil) == 2  # <-- Diperbaiki dari 1 menjadi 2
    
    judul_hasil = df_hasil['Judul'].tolist()
    assert 'Kaos A' in judul_hasil
    assert 'Hoodie B' in judul_hasil
    
    assert df_hasil['Harga'].dtype == float
    assert df_hasil['Peringkat'].dtype == float


def test_transformasi_data_input_kosong():
    df_kosong = pd.DataFrame(columns=['Judul', 'Harga']) # Minimal kolom untuk menghindari error di transformasi
    hasil_kosong = transformasi_data(df_kosong)
    assert hasil_kosong.empty

def test_transformasi_data_kolom_hilang(df_mentah_sampel, caplog):
    df_tanpa_harga = df_mentah_sampel.drop(columns=['Harga'])
    transformasi_data(df_tanpa_harga.copy()) # Panggil fungsi untuk memicu log
    assert "Kolom 'Harga' tidak ditemukan" in caplog.text


