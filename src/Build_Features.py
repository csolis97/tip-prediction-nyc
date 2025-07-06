import pandas as pd
from datetime import datetime

def build_features(df):
    # Asegúrate de que las columnas datetime estén en formato datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    # Crear features numéricas adicionales según el notebook original
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute



    # trip_speed (distancia / tiempo)
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + 1e-7)  # para evitar división por cero

    # Definir las features numéricas y categóricas según el notebook
    numeric_feat = [
        "pickup_weekday",
        "pickup_hour",
        "work_hours",
        "pickup_minute",
        "passenger_count",
        "trip_distance",
        "trip_time",
        "trip_speed"
    ]
    categorical_feat = [
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
    ]
    
    features = numeric_feat + categorical_feat
    
    # Seleccionar X (features) y y (target)
    X = df[features]
    y = df['target']
    
    return X, y
