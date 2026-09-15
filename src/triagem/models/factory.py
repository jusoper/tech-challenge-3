"""Factory de classificadores (Design Pattern — Factory)."""

from __future__ import annotations

from triagem.models.base import HeuristicUrgencyClassifier, UrgencyClassifier


def create_classifier(kind: str = "heuristic") -> UrgencyClassifier:
    """
    Cria o classificador conforme `MODEL_KIND`.

    Args:
        kind: Identificador do backend (`heuristic` por enquanto).

    Returns:
        Instância pronta para inferência.

    Raises:
        ValueError: Se o kind não for suportado.
    """
    normalized = (kind or "heuristic").strip().lower()
    if normalized == "heuristic":
        return HeuristicUrgencyClassifier()
    raise ValueError(
        f"MODEL_KIND desconhecido: {kind!r}. "
        "Suportados na Etapa 1: heuristic (sklearn/onnx nas etapas 2–4)."
    )
