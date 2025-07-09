import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """Genera una matriz de confusión con Seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_probability_histogram(model, X, y_true):
    """Muestra histogramas de probabilidad por clase verdadera."""
    probas = model.predict_proba(X)[:, 1]
    
    plt.figure(figsize=(6, 4))
    sns.histplot(probas[y_true == 1], color="green", label="Clase 1 (Alta propina)", kde=True, stat="density", bins=30)
    sns.histplot(probas[y_true == 0], color="red", label="Clase 0 (Baja propina)", kde=True, stat="density", bins=30)
    plt.title("Distribución de Probabilidades por Clase")
    plt.xlabel("Probabilidad de Alta propina")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names):
    """Muestra un gráfico de barras con las importancias de las características."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importancias de Características")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlabel("Características")
    plt.ylabel("Importancia")
    plt.tight_layout()
    plt.show()
    