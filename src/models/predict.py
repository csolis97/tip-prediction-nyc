from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

def predict_with_threshold(model, X, threshold=0.5):
    """
    Genera predicciones binarias usando un umbral dado.
    // Generates binary predictions using a given threshold.
    """
    probs = model.predict_proba(X)[:, 1]
    return (probs > threshold).astype(int)

def evaluate_f1_score(y_true, y_pred):
    """
    Calcula el F1-score dado y_true y y_pred.
    // Calculates the F1-score given y_true and y_pred.
    """
    f1 = f1_score(y_true, y_pred)
    print(f"F1-score: {f1:.4f}")
    return f1


def evaluate_f2(model, X, y, threshold=None):
    """
    Calcula el F1-score del modelo con un umbral de clasificación dado.
    // Calculates the F1-score of the model with a given classification threshold.
    """

    if threshold is None:
        try:
            threshold = float(input("Ingrese el umbral de clasificación (entre 0 y 1): "))
        except ValueError:
            print("Entrada inválida. Usando umbral por defecto = 0.5")
            threshold = 0.5

    probs = model.predict_proba(X)[:, 1]
    y_pred = (probs > threshold).astype(int)
    f1 = f1_score(y, y_pred)

    print(f"F1-score con threshold={threshold:.2f}: {f1:.4f}")
    return f1, y_pred