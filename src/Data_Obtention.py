import pandas as pd
import os

def download_taxi_data(url, output_filename):
    taxi = pd.read_parquet(url)

    # Ruta base del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', output_filename)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    taxi.to_parquet(DATA_PATH, index=False)

    print(f"✅ Archivo guardado en: {DATA_PATH}")
    return DATA_PATH