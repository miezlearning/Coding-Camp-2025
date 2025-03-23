from google_play_scraper import reviews
import pandas as pd
import os
result, _ = reviews(
    'com.whatsapp',  
    lang='id',       
    country='id',    
    count=12000       
)

df = pd.DataFrame(result)
base_dir = '/C:/Users/Miez/Documents/Dicoding/Belajar Pengembangan Machine Learning/data'
os.makedirs(base_dir, exist_ok=True)
csv_path = os.path.join(base_dir, 'whatsapp_reviews.csv')
df.to_csv(csv_path, index=False)