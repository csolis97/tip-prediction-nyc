import pandas as pd
from datetime import datetime

def build_features(df):
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
