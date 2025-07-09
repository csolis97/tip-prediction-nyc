from sklearn.metrics import f1_score

def evaluate_f1(model, X, y, threshold=None):
    if threshold is None:
        try:
            threshold = float(input("Ingrese el umbral de clasificación (entre 0 y 1): "))
        except ValueError:
            print("Entrada inválida. Usando umbral por defecto = 0.5")
            threshold = 0.5

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)
    f1 = f1_score(y, preds)
    print(f"F1-score con threshold {threshold:.1f}:")
    return f1
