"""Factory de classificadores (Design Pattern — Factory)."""

from __future__ import annotations

import json
from pathlib import Path

from triagem.models.base import (
    HeuristicUrgencyClassifier,
    OnnxUrgencyClassifier,
    SklearnUrgencyClassifier,
    UrgencyClassifier,
)
from triagem.pipeline.paths import model_artifact_path, onnx_artifact_path, onnx_labels_path
from triagem.pipeline.train import load_pipeline


def create_classifier(
    kind: str = "heuristic",
    artifact_path: Path | None = None,
) -> UrgencyClassifier:
    """
    Cria o classificador conforme `MODEL_KIND`.

    Args:
        kind: Identificador do backend (`heuristic`, `sklearn` ou `onnx`).
        artifact_path: Caminho do artefato (joblib ou onnx).

    Returns:
        Instância pronta para inferência.

    Raises:
        ValueError: Se o kind não for suportado.
        FileNotFoundError: Se o artefato não existir.
    """
    normalized = (kind or "heuristic").strip().lower()
    if normalized == "heuristic":
        return HeuristicUrgencyClassifier()
    if normalized == "sklearn":
        path = artifact_path or model_artifact_path()
        pipeline = load_pipeline(path)
        return SklearnUrgencyClassifier(pipeline)
    if normalized == "onnx":
        path = artifact_path or onnx_artifact_path()
        if not path.is_file():
            raise FileNotFoundError(f"artefato ONNX não encontrado: {path}")
        labels_file = onnx_labels_path()
        if not labels_file.is_file():
            raise FileNotFoundError(f"labels ONNX não encontrados: {labels_file}")
        classes = json.loads(labels_file.read_text(encoding="utf-8"))["classes"]
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return OnnxUrgencyClassifier(session, classes)
    raise ValueError(f"MODEL_KIND desconhecido: {kind!r}. Suportados: heuristic, sklearn, onnx.")
