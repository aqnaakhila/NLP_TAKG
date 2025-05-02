import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Set Environment variable untuk Kaggle API key
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['kaggle.json'] = current_dir # Tempat kaggle.json disimpan

# Inisialisasi API
api = KaggleApi()
api.authenticate()

# Unduh dataset
dataset = 'rmisra/news-category-dataset'
api.dataset_download_files(dataset, path='content/data', unzip=True)

print("Download dan Ekstraksi selesai")