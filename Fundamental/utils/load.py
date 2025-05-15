import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PATH_CSV_OUTPUT = "produk.csv"

def muat_ke_csv(df_input, path_file=PATH_CSV_OUTPUT):
    if df_input.empty:
        logging.warning(f"DataFrame kosong. Tidak ada data yang akan ditulis ke {path_file}.")
        return False

    try:
        df_input.to_csv(path_file, index=False, encoding='utf-8')
        logging.info(f"Data berhasil dimuat ke {path_file}")
        return True
    except IOError as e:
        logging.error(f"Gagal menulis ke file CSV {path_file}: {e}")
        return False
    except Exception as e:
        logging.error(f"Kesalahan tak terduga saat menulis ke CSV {path_file}: {e}")
        return False

if __name__ == '__main__':
    data_transformasi_sampel = {
        'Judul': ['Kaos Alpha', 'Hoodie Beta'], 'Harga': [160000.0, 794008.0],
        'Peringkat': [4.5, 4.8], 'Warna': [3, 3], 'Ukuran': ['M', 'L'],
        'Gender': ['Pria', 'Unisex'], 'timestamp': [datetime.now().isoformat()] * 2
    }
    df_untuk_dimuat = pd.DataFrame(data_transformasi_sampel)
    logging.info("Data sampel hasil transformasi untuk dimuat (tes modul):\n%s", df_untuk_dimuat)
    
    path_csv_tes = "tes_produk_output.csv" 
    if muat_ke_csv(df_untuk_dimuat, path_file=path_csv_tes):
        logging.info(f"Contoh pemuatan CSV berhasil. Data disimpan ke {path_csv_tes} (tes modul).")
    else:
        logging.error("Contoh pemuatan CSV gagal (tes modul).")