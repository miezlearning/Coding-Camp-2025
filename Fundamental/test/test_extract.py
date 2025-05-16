import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from bs4 import BeautifulSoup
import requests

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.extract import ambil_detail_produk, ekstrak_data, URL_DASAR

CONTOH_HTML_KARTU_PRODUK_VALID = """<div class="collection-card"><div class="product-details"><h3 class="product-title">Kaos Keren</h3><div class="price-container"><span class="price">$120.50</span></div><p style="font-size: 14px; color: #777;">Rating: ⭐ 4.5 / 5</p><p style="font-size: 14px; color: #777;">3 Colors</p><p style="font-size: 14px; color: #777;">Size: L</p><p style="font-size: 14px; color: #777;">Gender: Men</p></div></div>"""
CONTOH_HTML_KARTU_PRODUK_HARGA_TIDAK_TERSEDIA = """<div class="collection-card"><div class="product-details"><h3 class="product-title">Celana Spesial</h3><p class="price">Price Unavailable</p><p style="font-size: 14px; color: #777;">Rating: ⭐ 3.0 / 5</p><p style="font-size: 14px; color: #777;">2 Colors</p><p style="font-size: 14px; color: #777;">Size: M</p><p style="font-size: 14px; color: #777;">Gender: Unisex</p></div></div>"""
CONTOH_HTML_KARTU_PRODUK_PERINGKAT_TIDAK_VALID = """<div class="collection-card"><div class="product-details"><h3 class="product-title">Jaket Lama</h3><div class="price-container"><span class="price">$75.00</span></div><p style="font-size: 14px; color: #777;">Rating: ⭐ Invalid Rating / 5</p><p style="font-size: 14px; color: #777;">1 Color</p><p style="font-size: 14px; color: #777;">Size: S</p><p style="font-size: 14px; color: #777;">Gender: Women</p></div></div>"""
CONTOH_HTML_KARTU_PRODUK_TIDAK_ADA_PERINGKAT = """<div class="collection-card"><div class="product-details"><h3 class="product-title">Barang Misterius</h3><div class="price-container"><span class="price">$99.00</span></div><p style="font-size: 14px; color: #777;">Rating: Not Rated</p><p style="font-size: 14px; color: #777;">5 Colors</p><p style="font-size: 14px; color: #777;">Size: XL</p><p style="font-size: 14px; color: #777;">Gender: Men</p></div></div>"""

def test_ambil_detail_produk_valid():
    sup = BeautifulSoup(CONTOH_HTML_KARTU_PRODUK_VALID, "html.parser")
    detail = ambil_detail_produk(sup.find("div", class_="collection-card"))
    assert detail["Judul"] == "Kaos Keren" and detail["Harga"] == "$120.50"

def test_ambil_detail_produk_harga_tidak_tersedia():
    sup = BeautifulSoup(CONTOH_HTML_KARTU_PRODUK_HARGA_TIDAK_TERSEDIA, "html.parser")
    detail = ambil_detail_produk(sup.find("div", class_="collection-card"))
    assert detail["Harga"] == "Price Unavailable"

def test_ambil_detail_produk_peringkat_tidak_valid():
    sup = BeautifulSoup(CONTOH_HTML_KARTU_PRODUK_PERINGKAT_TIDAK_VALID, "html.parser")
    detail = ambil_detail_produk(sup.find("div", class_="collection-card"))
    assert detail["Peringkat"] == "Rating: ⭐ Invalid Rating / 5"

def test_ambil_detail_produk_tidak_ada_peringkat():
    sup = BeautifulSoup(CONTOH_HTML_KARTU_PRODUK_TIDAK_ADA_PERINGKAT, "html.parser")
    detail = ambil_detail_produk(sup.find("div", class_="collection-card"))
    assert detail["Peringkat"] == "Rating: Not Rated"

@patch('utils.ekstrak.requests.get')
def test_ekstrak_data_sukses(mock_get):
    mock_respons = MagicMock()
    mock_respons.status_code = 200
    mock_respons.content = f"<html><body><div id='collectionList'>{CONTOH_HTML_KARTU_PRODUK_VALID}</div></body></html>".encode('utf-8')
    mock_get.return_value = mock_respons
    with patch('utils.ekstrak.HALAMAN_MAKSIMUM', 1): df = ekstrak_data()
    assert not df.empty and len(df) == 1 and "timestamp" in df.columns

@patch('utils.ekstrak.requests.get')
def test_ekstrak_data_exception_request(mock_get, caplog):
    mock_get.side_effect = requests.exceptions.RequestException("Kesalahan jaringan tes")
    with patch('utils.ekstrak.HALAMAN_MAKSIMUM', 1): df = ekstrak_data()
    assert df.empty and "Gagal mengambil halaman" in caplog.text

@patch('utils.ekstrak.requests.get')
def test_ekstrak_data_tidak_ada_grid_produk(mock_get, caplog):
    mock_respons = MagicMock(); mock_respons.status_code = 200
    mock_respons.content = "<html><body>No grid</body></html>".encode('utf-8')
    mock_get.return_value = mock_respons
    with patch('utils.ekstrak.HALAMAN_MAKSIMUM', 1): df = ekstrak_data()
    assert df.empty and "Tidak ada grid produk" in caplog.text

@patch('utils.ekstrak.requests.get')
def test_ekstrak_data_iterasi_halaman(mock_get):
    mock_res1 = MagicMock(); mock_res1.status_code = 200; mock_res1.content = f"<html><body><div id='collectionList'>{CONTOH_HTML_KARTU_PRODUK_VALID}</div></body></html>".encode('utf-8')
    mock_res2 = MagicMock(); mock_res2.status_code = 200; mock_res2.content = f"<html><body><div id='collectionList'>{CONTOH_HTML_KARTU_PRODUK_HARGA_TIDAK_TERSEDIA}</div></body></html>".encode('utf-8')
    mock_get.side_effect = [mock_res1, mock_res2]
    with patch('utils.ekstrak.HALAMAN_MAKSIMUM', 2): df = ekstrak_data()
    assert len(df) == 2 and mock_get.call_count == 2