import pytest
import pandas as pd
from unittest.mock import patch, mock_open
# import os # Tidak digunakan lagi
from datetime import datetime

import sys
import os as uos # Hindari konflik dengan os modul jika ada
sys.path.insert(0, uos.path.abspath(uos.path.join(uos.path.dirname(__file__), '..')))

from utils.load import muat_ke_csv # Hanya impor muat_ke_csv

@pytest.fixture
def df_transformasi_sampel():
    data = {
        'Judul': ['Kaos Bersih', 'Hoodie Super'], 'Harga': [160000.0, 800000.0],
        'Peringkat': [4.5, 4.9], 'Warna': [3, 5], 'Ukuran': ['M', 'XL'],
        'Gender': ['Pria', 'Unisex'], 'timestamp': [datetime.now().isoformat()] * 2
    }
    return pd.DataFrame(data)

def test_muat_ke_csv_sukses(df_transformasi_sampel):
    m = mock_open()
    with patch('builtins.open', m):
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            hasil = muat_ke_csv(df_transformasi_sampel, "dummy_path.csv")
            assert hasil is True
            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            assert kwargs['index'] is False
    m.assert_called_once_with("dummy_path.csv", 'w', encoding='utf-8', newline='')

def test_muat_ke_csv_df_kosong(caplog):
    df_kosong = pd.DataFrame()
    path_file_tes_kosong = "tes_kosong.csv"
    hasil = muat_ke_csv(df_kosong, path_file_tes_kosong)
    assert hasil is False
    assert "DataFrame kosong." in caplog.text
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        muat_ke_csv(df_kosong, path_file_tes_kosong)
        mock_to_csv.assert_not_called()

@patch('pandas.DataFrame.to_csv', side_effect=IOError("Disk penuh"))
def test_muat_ke_csv_io_error(mock_to_csv_error, df_transformasi_sampel, caplog):
    m = mock_open()
    with patch('builtins.open', m):
        hasil = muat_ke_csv(df_transformasi_sampel, "error_path.csv")
    assert hasil is False and "Gagal menulis ke file CSV" in caplog.text

@patch('pandas.DataFrame.to_csv', side_effect=Exception("Kesalahan pandas"))
def test_muat_ke_csv_kesalahan_tak_terduga(mock_to_csv_unexpected, df_transformasi_sampel, caplog):
    m = mock_open()
    with patch('builtins.open', m):
        hasil = muat_ke_csv(df_transformasi_sampel, "unexpected_error.csv")
    assert hasil is False and "Kesalahan tak terduga saat menulis ke CSV" in caplog.text