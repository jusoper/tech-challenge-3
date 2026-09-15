"""Exportação do pipeline sklearn para ONNX Runtime."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def export_pipeline_to_onnx(
    pipeline: Pipeline,
    output_path: Path,
    labels_path: Path | None = None,
    target_opset: int = 15,
) -> tuple[Path, Path]:
    """
    Converte o pipeline TF-IDF + classificador para ONNX e salva o mapa de classes.

    Args:
        pipeline: Pipeline sklearn treinado (espera texto em `str`).
        output_path: Destino do arquivo `.onnx`.
        labels_path: Destino do JSON com `classes_` (default ao lado do ONNX).
        target_opset: Opset ONNX alvo.

    Returns:
        Tupla `(onnx_path, labels_path)`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path = labels_path or output_path.with_name("onnx_labels.json")

    classifier = pipeline.named_steps["clf"]
    initial_types = [("input", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=target_opset,
        options={id(classifier): {"zipmap": False}},
    )
    output_path.write_bytes(onnx_model.SerializeToString())

    classes = [str(c) for c in classifier.classes_]
    labels_path.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    logger.info("onnx_exported", extra={"path": str(output_path), "classes": classes})
    return output_path, labels_path
