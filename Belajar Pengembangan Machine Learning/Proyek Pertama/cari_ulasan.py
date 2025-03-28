from google_play_scraper import reviews
import pandas as pd
from pathlib import Path

app_id = 'com.miHoYo.GenshinImpact'  # Genshin Impact app ID

def scrape_reviews(app_id, count=1000):
    result, _ = reviews(
        app_id,
        lang='id',  
        country='id',  
        count=count  
    )
    return result

reviews_data = scrape_reviews(app_id, count=15000)  

df = pd.DataFrame(reviews_data)
base_dir = Path.home() / 'Documents' / 'Dicoding' / 'Belajar Pengembangan Machine Learning' / 'data'
base_dir.mkdir(parents=True, exist_ok=True)  # Membuat folder jika belum ada
csv_path = base_dir / 'hasil_review_genshin_impact.csv'
df.to_csv(csv_path, index=False)

print(f"Data ulasan telah disimpan di: {csv_path}")
