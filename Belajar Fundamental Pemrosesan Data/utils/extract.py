import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

URL_DASAR = "https://fashion-studio.dicoding.dev"
HALAMAN_MAKSIMUM = 50

def ambil_detail_produk(kartu_produk):
    detail = {
        "Judul": None, "Harga": None, "Peringkat": None,
        "Warna": None, "Ukuran": None, "Gender": None
    }
    try:
        tag_judul = kartu_produk.find("h3", class_="product-title")
        if tag_judul: detail["Judul"] = tag_judul.get_text(strip=True)

        wadah_harga = kartu_produk.find("div", class_="price-container")
        if wadah_harga:
            span_harga = wadah_harga.find("span", class_="price")
            if span_harga: detail["Harga"] = span_harga.get_text(strip=True)
        else:
            p_harga = kartu_produk.find("p", class_="price")
            if p_harga and "Price Unavailable" in p_harga.get_text(strip=True):
                detail["Harga"] = "Price Unavailable"

        tag_p_all = kartu_produk.find_all("p", style="font-size: 14px; color: #777;")
        for tag_p in tag_p_all:
            teks = tag_p.get_text(strip=True)
            if teks.startswith("Rating:"): detail["Peringkat"] = teks
            elif "Colors" in teks and not teks.startswith("Rating:"): detail["Warna"] = teks
            elif teks.startswith("Size:"): detail["Ukuran"] = teks
            elif teks.startswith("Gender:"): detail["Gender"] = teks
        
        if not detail["Peringkat"]:
            p_peringkat_kosong = kartu_produk.find("p", string=lambda t: t and "Not Rated" in t)
            if p_peringkat_kosong: detail["Peringkat"] = p_peringkat_kosong.get_text(strip=True)

    except AttributeError as e:
        logging.error(f"Kesalahan parsing kartu produk: {e} - Kartu: {kartu_produk.prettify()[:200]}")
    return detail

def ekstrak_data():
    semua_produk = []
    timestamp_sekarang = datetime.now().isoformat()

    for nomor_halaman in range(1, HALAMAN_MAKSIMUM + 1):
        url_halaman = f"{URL_DASAR}/page{nomor_halaman}" if nomor_halaman > 1 else URL_DASAR
        logging.info(f"Melakukan scraping halaman: {url_halaman}")
        
        try:
            respons = requests.get(url_halaman, timeout=10)
            respons.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Gagal mengambil halaman {url_halaman}: {e}")
            continue

        sup = BeautifulSoup(respons.content, "html.parser")
        grid_produk = sup.find("div", id="collectionList")
        if not grid_produk:
            logging.warning(f"Tidak ada grid produk ditemukan di halaman {url_halaman}")
            continue

        kartu_produk_all = grid_produk.find_all("div", class_="collection-card")
        if not kartu_produk_all:
            logging.warning(f"Tidak ada kartu produk ditemukan di halaman {url_halaman}")
            continue

        for kartu in kartu_produk_all:
            info_produk = ambil_detail_produk(kartu)
            if info_produk.get("Judul"):
                info_produk["timestamp"] = timestamp_sekarang
                semua_produk.append(info_produk)
            else:
                logging.warning(f"Melewati kartu tanpa judul di {url_halaman}")
        
        logging.info(f"Selesai scraping halaman {nomor_halaman}. Total produk sejauh ini: {len(semua_produk)}")

    if not semua_produk:
        logging.error("Tidak ada produk yang diekstrak. Pipeline ETL mungkin gagal.")
    
    df = pd.DataFrame(semua_produk)
    return df

if __name__ == '__main__':
    logging.info("Memulai proses ekstraksi (tes modul)...")
    df_mentah = ekstrak_data()
    logging.info(f"Ekstraksi selesai (tes modul). Total produk diekstrak: {len(df_mentah)}")
    if not df_mentah.empty:
        logging.info("Contoh data yang diekstrak (tes modul):")
        logging.info(df_mentah.head())
        logging.info("\nTipe data dari data yang diekstrak (tes modul):")
        df_mentah.info(buf=logging.getLogger().handlers[0].stream)
    else:
        logging.warning("Tidak ada data yang diekstrak (tes modul).")