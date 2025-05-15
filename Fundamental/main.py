import logging
from utils.extract import ekstrak_data
from utils.transform import transformasi_data
from utils.load import muat_ke_csv 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def pipeline_etl_utama():
    logging.info("Memulai Pipeline ETL...")

    # 1. Ekstrak
    logging.info("--- Tahap 1: Mengekstrak Data ---")
    try:
        df_mentah = ekstrak_data()
        if df_mentah.empty:
            logging.error("Ekstraksi menghasilkan DataFrame kosong. Menghentikan pipeline.")
            return
        logging.info(f"Berhasil mengekstrak {len(df_mentah)} catatan mentah.")
    except Exception as e:
        logging.critical(f"Kesalahan kritis saat ekstraksi data: {e}", exc_info=True)
        return

    # 2. Transformasi
    logging.info("--- Tahap 2: Mentransformasi Data ---")
    try:
        df_transformasi = transformasi_data(df_mentah)
        if df_transformasi.empty:
            logging.warning("Transformasi menghasilkan DataFrame kosong. Tidak ada data untuk dimuat.")
        else:
            logging.info(f"Berhasil mentransformasi data. DataFrame hasil memiliki {len(df_transformasi)} catatan.")
            logging.info("Contoh data hasil transformasi:")
            logging.info(df_transformasi.head())
            logging.info("\nTipe data dari data hasil transformasi:")
            df_transformasi.info(buf=logging.getLogger().handlers[0].stream)

    except Exception as e:
        logging.critical(f"Kesalahan kritis saat transformasi data: {e}", exc_info=True)
        return

    # 3. Muat
    logging.info("--- Tahap 3: Memuat Data ---")
    if not df_transformasi.empty:
        try:
            if muat_ke_csv(df_transformasi): # Menggunakan path default dari modul muat.py
                logging.info("Berhasil memuat data ke CSV.")
            else:
                logging.error("Gagal memuat data ke CSV.")
        except Exception as e:
            logging.error(f"Kesalahan saat pemuatan CSV: {e}", exc_info=True)
    else:
        logging.info("Tidak ada data hasil transformasi untuk dimuat.")

    logging.info("Pipeline ETL selesai.")

if __name__ == '__main__':
    pipeline_etl_utama()