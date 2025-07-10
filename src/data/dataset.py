import pandas as pd
import os
from utils.path_utils import get_project_root

def download_data(url, output_filename):
    """
    Descarga el conjunto de datos de taxi desde una URL y lo guarda en formato Parquet.
    // Downloads the taxi dataset from a URL and saves it in Parquet format.
    """
    taxi = pd.read_parquet(url)
    
    # Ruta base del proyecto // Base project directory
    BASE_DIR = get_project_root()
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', output_filename)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    taxi.to_parquet(DATA_PATH, index=False)
    print(f"Archivo guardado en: {DATA_PATH}")
    return DATA_PATH

def load_and_process_dataset(input_path, output_path=None):
    """
    Carga y limpia el conjunto de datos, y crea la variable objetivo y features básicas.
    // Loads and cleans the dataset, creating the target variable and basic features.
    """

    df = pd.read_parquet(input_path)
    df = df[df['fare_amount'] > 0].reset_index(drop=True)

    # Crear variable objetivo // Create target variable
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df['target'] = (df['tip_fraction'] > 0.2).astype("int32")

    # Fechas a datetime // Convert dates to datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # Creación de features // Create features
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = ((df['pickup_weekday'].between(0, 4)) &
                        (df['pickup_hour'].between(8, 18))).astype("int32")
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    EPS = 1e-7
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Archivo procesado guardado en: {output_path}")

    return df.reset_index(drop=True)
