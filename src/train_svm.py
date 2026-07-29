"""Treina um classificador SVM de gêneros musicais com busca de hiperparâmetros.

Pipeline: StandardScaler + SVC, ajustado via GridSearchCV (5-fold estratificado)
sobre o conjunto de treino. O melhor modelo é avaliado no conjunto de teste e
salvo em models/svm_model.joblib.

Uso:
    python src/train_svm.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT_DIR / "features" / "svm"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    X = np.load(FEATURES_DIR / "X.npy")
    y = np.load(FEATURES_DIR / "y.npy")
    classes = np.load(FEATURES_DIR / "classes.npy")
    return X, y, classes


def main():
    X, y, classes = load_data()
    print(f"Dados: X {X.shape}, y {y.shape}, {len(classes)} gêneros")
    print(f"Gêneros: {list(classes)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras\n")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(random_state=RANDOM_STATE)),
    ])

    # gamma só se aplica ao kernel rbf, por isso dois grids separados
    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10, 100],
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.01, 0.001],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    print("Executando grid search...")
    grid_search.fit(X_train, y_train)

    print(f"\nMelhores hiperparâmetros: {grid_search.best_params_}")
    print(f"Melhor acurácia (CV no treino): {grid_search.best_score_:.4f}\n")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Acurácia no teste: {test_accuracy:.4f}\n")
    print("Relatório por gênero:")
    print(classification_report(y_test, y_pred, target_names=classes))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"SVM - Matriz de Confusão (acurácia teste: {test_accuracy:.2%})")
    fig.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"Matriz de confusão salva em {cm_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "svm_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Modelo salvo em {model_path}")


if __name__ == "__main__":
    main()
