"""Entrenamiento reproducible y evaluación sin fuga entre eventos."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["magnitud", "profundidad_km", "distancia_km"]
CLASSES = ["I-IV", "V-VI", "VII-XII"]
DATA_PATH = Path(__file__).parent / "data" / "sismos_sinteticos.csv"


def build_pipeline(k: int) -> Pipeline:
    """Crea un escalador y clasificador KNN ajustados como una sola unidad."""
    return Pipeline(
        [
            ("escalador", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]
    )


def train_and_evaluate(data: pd.DataFrame | None = None) -> dict:
    """Selecciona K solo en entrenamiento y evalua una prueba por evento."""
    frame = data if data is not None else pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["intensidad", "evento_id"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(missing))}")
    if frame[list(required)].isnull().any().any():
        raise ValueError("El dataset contiene valores vacios en campos requeridos.")
    if frame["evento_id"].nunique() < 6:
        raise ValueError("Se requieren al menos 6 eventos para la evaluacion agrupada.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["evento_id"]))
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    if set(train["evento_id"]) & set(test["evento_id"]):
        raise RuntimeError("Fuga de eventos entre entrenamiento y prueba.")

    folds = min(5, train["evento_id"].nunique())
    cv = GroupKFold(n_splits=folds)
    scores = {}
    for k in (3, 5, 7, 9):
        fold_scores = cross_val_score(
            build_pipeline(k),
            train[FEATURES],
            train["intensidad"],
            groups=train["evento_id"],
            cv=cv,
            scoring="f1_macro",
            error_score="raise",
        )
        scores[k] = float(fold_scores.mean())
    best_k = max(scores, key=scores.get)
    model = build_pipeline(best_k).fit(train[FEATURES], train["intensidad"])
    predicted = model.predict(test[FEATURES])
    labels = [label for label in CLASSES if label in frame["intensidad"].unique()]
    metrics = {
        "exactitud": float(accuracy_score(test["intensidad"], predicted)),
        "precision_macro": float(
            precision_score(test["intensidad"], predicted, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(test["intensidad"], predicted, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(test["intensidad"], predicted, labels=labels, average="macro", zero_division=0)
        ),
    }
    matrix = confusion_matrix(test["intensidad"], predicted, labels=labels)
    return {
        "data": frame,
        "model": model,
        "best_k": best_k,
        "cv_scores": scores,
        "metrics": metrics,
        "labels": labels,
        "confusion": matrix,
        "train": train,
        "test": test,
        "predicted": predicted,
    }


def classify(result: dict, magnitud: float, profundidad_km: float, distancia_km: float, k: int | None = None) -> dict:
    """Predice una clase y resume la votacion de los vecinos mas cercanos."""
    values = [magnitud, profundidad_km, distancia_km]
    limits = [(0, 10), (0, 700), (0, 1500)]
    for value, (low, high), name in zip(values, limits, FEATURES):
        if not low <= float(value) <= high:
            raise ValueError(f"{name} debe estar entre {low} y {high}.")
    selected_k = k or result["best_k"]
    if selected_k not in (3, 5, 7, 9):
        raise ValueError("K debe ser 3, 5, 7 o 9.")
    model = result["model"]
    sample = pd.DataFrame([values], columns=FEATURES)
    scaled = model.named_steps["escalador"].transform(sample)
    distances, indices = model.named_steps["knn"].kneighbors(scaled, n_neighbors=selected_k)
    neighbors = result["train"].iloc[indices[0]]
    counts = neighbors["intensidad"].value_counts().reindex(CLASSES, fill_value=0)
    return {
        "prediccion": str(model.predict(sample)[0]),
        "votos": counts.to_dict(),
        "vecinos": neighbors,
        "distancias": distances[0],
        "k": selected_k,
    }
