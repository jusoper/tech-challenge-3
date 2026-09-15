"""Treinamento TF-IDF + RandomForest e persistência do artefato."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

# Evita SIGFPE intermitente do Accelerate/OpenBLAS no macOS ao importar NumPy.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_text_pipeline(seed: int = 42) -> Pipeline:
    """Monta pipeline TF-IDF + Logistic Regression (leve e exportável para ONNX)."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_text_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[Pipeline, dict[str, float | str]]:
    """
    Treina o classificador e retorna o pipeline + métricas de holdout.

    Args:
        texts: Laudos de entrada.
        labels: Rótulos de urgência.
        seed: Seed para split e RandomForest.
        test_size: Fração de holdout.

    Returns:
        Pipeline treinado e dicionário de métricas.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=seed,
        stratify=list(labels),
    )
    pipeline = build_text_pipeline(seed=seed)
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, zero_division=0)
    metrics: dict[str, float | str] = {
        "accuracy": accuracy,
        "n_train": float(len(x_train)),
        "n_test": float(len(x_test)),
        "report": report,
    }
    logger.info("train_ok", extra={"accuracy": accuracy, "n_train": len(x_train)})
    return pipeline, metrics


def save_pipeline(pipeline: Pipeline, path: Path) -> Path:
    """Salva o pipeline sklearn em disco via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("model_saved", extra={"path": str(path)})
    return path


def save_metrics(metrics: dict[str, float | str], path: Path) -> Path:
    """Persiste métricas de treino em JSON (sem o report textual longo no mesmo nível)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accuracy": metrics["accuracy"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = path.with_name("classification_report.txt")
    report_path.write_text(str(metrics.get("report", "")), encoding="utf-8")
    return path


def load_pipeline(path: Path) -> Pipeline:
    """Carrega pipeline sklearn persistido."""
    if not path.is_file():
        raise FileNotFoundError(f"artefato não encontrado: {path}")
    return joblib.load(path)
