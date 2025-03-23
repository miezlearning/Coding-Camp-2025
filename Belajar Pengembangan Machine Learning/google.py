from google_play_scraper import reviews
result, _ = reviews(
    'com.whatsapp',  # ID aplikasi WhatsApp
    lang='id',       # Bahasa Indonesia
    country='id',    # Negara Indonesia
    count=3000       # Jumlah ulasan
)
# Simpan ke file CSV
import pandas as pd
df = pd.DataFrame(result)
df.to_csv('whatsapp_reviews.csv', index=False)