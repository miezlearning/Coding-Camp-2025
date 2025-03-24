from google_play_scraper import reviews
import pandas as pd
from pathlib import Path

app_ids = ['com.mobile.legends', 'com.tencent.ig', 'com.bandainamcoent.opbrww']  

def scrape_reviews(app_id, count=1000):
    result, _ = reviews(
        app_id,
        lang='id',  
        country='id',  
        count=count  
    )
    return result

all_reviews = []
for app_id in app_ids:
    reviews_data = scrape_reviews(app_id, count=6500)  
    all_reviews.extend(reviews_data)

df = pd.DataFrame(all_reviews)
base_dir = Path.home() / 'Documents' / 'Dicoding' / 'Belajar Pengembangan Machine Learning' / 'data'
base_dir.mkdir(parents=True, exist_ok=True)  # Membuat folder jika belum ad
csv_path = base_dir / 'hasil_review.csv'
df.to_csv(csv_path, index=False)

print(f"Data ulasan telah disimpan di: {csv_path}")