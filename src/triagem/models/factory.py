"""Factory de classificadores (Design Pattern — Factory)."""

from __future__ import annotations

from pathlib import Path

from triagem.models.base import (
    HeuristicUrgencyClassifier,
    SklearnUrgencyClassifier,
    UrgencyClassifier,
)
from triagem.pipeline.paths import model_artifact_path
from triagem.pipeline.train import load_pipeline


def create_classifier(
    kind: str = "heuristic",
    artifact_path: Path | None = None,
) -> UrgencyClassifier:
    """
    Cria o classificador conforme `MODEL_KIND`.

    Args:
        kind: Identificador do backend (`heuristic` ou `sklearn`).
        artifact_path: Caminho do joblib quando `kind=sklearn`.

    Returns:
        Instância pronta para inferência.

    Raises:
        ValueError: Se o kind não for suportado.
        FileNotFoundError: Se o artefato sklearn não existir.
    """
    normalized = (kind or "heuristic").strip().lower()
    if normalized == "heuristic":
        return HeuristicUrgencyClassifier()
    if normalized == "sklearn":
        path = artifact_path or model_artifact_path()
        pipeline = load_pipeline(path)
        return SklearnUrgencyClassifier(pipeline)
    raise ValueError(
        f"MODEL_KIND desconhecido: {kind!r}. Suportados: heuristic, sklearn (onnx na Etapa 4)."
    )
