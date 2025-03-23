from google_play_scraper import reviews
import pandas as pd
from pathlib import Path

result, _ = reviews(
    'com.mobile.legends',  
    lang='id',       
    country='id',    
    count=50000       
)

df = pd.DataFrame(result)
base_dir = Path.home() / 'Documents' / 'Dicoding' / 'Belajar Pengembangan Machine Learning' / 'data'
base_dir.mkdir(parents=True, exist_ok=True)
csv_path = base_dir / 'mobellejen.csv'
df.to_csv(csv_path, index=False)