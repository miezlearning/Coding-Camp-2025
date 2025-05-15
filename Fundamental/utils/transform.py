import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KURS_USD_KE_IDR = 16000

def transformasi_harga(teks_harga):
    try:
        if pd.isna(teks_harga) or teks_harga == "Price Unavailable": return np.nan
        harga_usd = float(str(teks_harga).replace('$', '').replace(',', ''))
        return harga_usd * KURS_USD_KE_IDR
    except ValueError as e:
        logging.warning(f"Tidak dapat mengkonversi harga: {teks_harga}. Kesalahan: {e}. Mengembalikan NaN.")
        return np.nan
    except Exception as e:
        logging.error(f"Kesalahan tak terduga saat mengkonversi harga: {teks_harga}. Kesalahan: {e}. Mengembalikan NaN.")
        return np.nan

def transformasi_peringkat(teks_peringkat):
    try:
        if pd.isna(teks_peringkat) or "Invalid Rating" in str(teks_peringkat) or "Not Rated" in str(teks_peringkat):
            return np.nan
        cocok = re.search(r'(\d+\.?\d*)', str(teks_peringkat))
        if cocok: return float(cocok.group(1))
        return np.nan
    except ValueError as e:
        logging.warning(f"Tidak dapat mengkonversi peringkat: {teks_peringkat}. Kesalahan: {e}. Mengembalikan NaN.")
        return np.nan
    except Exception as e:
        logging.error(f"Kesalahan tak terduga saat mengkonversi peringkat: {teks_peringkat}. Kesalahan: {e}. Mengembalikan NaN.")
        return np.nan

def transformasi_warna(teks_warna):
    try:
        if pd.isna(teks_warna): return np.nan
        cocok = re.search(r'(\d+)\s*Colors', str(teks_warna))
        if cocok: return int(cocok.group(1))
        return np.nan
    except ValueError as e:
        logging.warning(f"Tidak dapat mengkonversi warna: {teks_warna}. Kesalahan: {e}. Mengembalikan NaN.")
        return np.nan
    except Exception as e:
        logging.error(f"Kesalahan tak terduga saat mengkonversi warna: {teks_warna}. Kesalahan: {e}. Mengembalikan NaN.")
        return np.nan

def transformasi_kolom_teks(teks, awalan_dihapus):
    try:
        if pd.isna(teks): return np.nan
        teks_bersih = str(teks).replace(awalan_dihapus, '').strip()
        return teks_bersih if teks_bersih else np.nan
    except Exception as e:
        logging.error(f"Kesalahan tak terduga saat transformasi kolom teks '{teks}' dengan awalan '{awalan_dihapus}'. Kesalahan: {e}. Mengembalikan NaN.")
        return np.nan

def transformasi_data(df_input):
    if df_input.empty:
        logging.warning("DataFrame input untuk transformasi kosong. Mengembalikan DataFrame kosong.")
        return df_input
        
    logging.info("Memulai transformasi data...")
    df_hasil = df_input.copy()

    try:
        if 'Judul' in df_hasil.columns:
             df_hasil['Judul'] = df_hasil['Judul'].replace('Unknown Product', np.nan)
        else: logging.warning("Kolom 'Judul' tidak ditemukan untuk transformasi.")
    except Exception as e: logging.error(f"Kesalahan saat transformasi 'Judul': {e}")

    kolom_transformasi = {
        'Harga': transformasi_harga,
        'Peringkat': transformasi_peringkat,
        'Warna': transformasi_warna,
        'Ukuran': lambda x: transformasi_kolom_teks(x, "Size: "),
        'Gender': lambda x: transformasi_kolom_teks(x, "Gender: ")
    }

    for kolom, fungsi_transformasi in kolom_transformasi.items():
        if kolom in df_hasil.columns:
            df_hasil[kolom] = df_hasil[kolom].apply(fungsi_transformasi)
        else:
            logging.warning(f"Kolom '{kolom}' tidak ditemukan untuk transformasi.")

    baris_awal = len(df_hasil)
    df_hasil.dropna(inplace=True)
    logging.info(f"Menghapus {baris_awal - len(df_hasil)} baris dengan nilai NaN.")

    baris_sebelum_duplikat = len(df_hasil)
    df_hasil.drop_duplicates(inplace=True)
    logging.info(f"Menghapus {baris_sebelum_duplikat - len(df_hasil)} baris duplikat.")

    try:
        if not df_hasil.empty:
            tipe_data_kolom = {
                'Harga': float, 'Peringkat': float, 'Warna': int,
                'Judul': str, 'Ukuran': str, 'Gender': str,
                'timestamp': str # Setelah konversi pd.to_datetime
            }
            for kolom, tipe in tipe_data_kolom.items():
                if kolom in df_hasil.columns:
                    if kolom == 'timestamp':
                        df_hasil[kolom] = pd.to_datetime(df_hasil[kolom]).astype(str)
                    else:
                        df_hasil[kolom] = df_hasil[kolom].astype(tipe)
        else:
            logging.warning("DataFrame kosong setelah penghapusan NaN/duplikat.")
    except KeyError as e:
        logging.error(f"KeyError saat konversi tipe: {e}.")
    except Exception as e:
        logging.error(f"Kesalahan tak terduga saat konversi tipe: {e}")

    logging.info("Transformasi data selesai.")
    return df_hasil

if __name__ == '__main__':
    data_sampel = {
        'Judul': ['Kaos Merah', 'Unknown Product', 'Hoodie Biru'], 'Harga': ['$10.00', '$5.00', '$49.88'],
        'Peringkat': ['Rating: ⭐ 4.5 / 5', 'Rating: ⭐ Invalid Rating / 5', 'Rating: ⭐ 4.8 / 5'],
        'Warna': ['3 Colors', '5 Colors', '3 Colors'], 'Ukuran': ['Size: M', 'Size: S', 'Size: L'],
        'Gender': ['Gender: Men', 'Gender: Women', 'Gender: Unisex'],
        'timestamp': [datetime.now().isoformat()] * 3
    }
    df_mentah_sampel = pd.DataFrame(data_sampel)
    logging.info("Data sampel mentah (tes modul):\n%s", df_mentah_sampel)
    df_transformasi_sampel = transformasi_data(df_mentah_sampel.copy())
    logging.info("\nData sampel setelah transformasi (tes modul):\n%s", df_transformasi_sampel)
    if not df_transformasi_sampel.empty:
        logging.info("\nTipe data sampel setelah transformasi (tes modul):")
        df_transformasi_sampel.info(buf=logging.getLogger().handlers[0].stream)