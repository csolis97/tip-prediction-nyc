import pandas as pd
from datetime import datetime

def build_features(df):
    """    
    Construye las features a partir del DataFrame de taxi.
    // Builds features from the taxi DataFrame.
    """
    # Definir las features numéricas y categóricas //  Define numeric and categorical features
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
    
    # Seleccionar X (features) ; Y (target) // Select X (features) ; Y (target)
    X = df[features]
    y = df['target']
    
    return X, y
