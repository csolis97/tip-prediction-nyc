from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from utils.path_utils import get_project_root


def train_model(X, y, model_path=None):
    """ 
    Entrena un modelo de Random Forest y lo guarda en el path especificado.
    // Trains a Random Forest model and saves it to the specified path.
    """

    # Si no se especifica un path, usa la carpeta models del proyecto // If no path is specified, use the models folder in the project root
    if model_path is None:
        root = get_project_root()
        model_path = os.path.join(root, "models", "random_forest_model.joblib")

    dir_path = os.path.dirname(model_path)
    if dir_path:
        print(f"Creando directorio: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

    # Entrenar el modelo // Train the model
    print("Entrenando el modelo...")
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rfc.fit(X, y)

    # Guardar modelo // Save the model
    joblib.dump(rfc, model_path)
    print(f"Modelo guardado en: {model_path}")
    return rfc
