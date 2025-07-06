import pandas as pd
import os

def load_and_process_dataset(input_path, output_path=None):
    df = pd.read_parquet(input_path)

    # Limpieza básica
    df = df[df['fare_amount'] > 0]
    df = df[df['tip_amount'] > 0]

    # Crear variable objetivo
    df['tip_ratio'] = df['tip_amount'] / df['fare_amount']
    df['target'] = (df['tip_ratio'] > 0.2).astype(int)

    # Crear columnas adicionales
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df['work_hours'] = df['pickup_hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + 1e-7)

    # Guardar el DataFrame procesado
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"✅ Archivo procesado guardado en: {output_path}")

    return df
