from sklearn.ensemble import RandomForestClassifier
import joblib
import os
def train_model(X, y, model_path="model.joblib"):
    """ Entrena un modelo de Random Forest y lo guarda en el path especificado."""
    # Asegúrate de que la carpeta del modelo existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Entrenar el modelo
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
    rfc.fit(X, y)
    # Guardar modelo
    joblib.dump(rfc, model_path)
    print(f"Modelo guardado en {model_path}")
    return rfc