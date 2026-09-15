"""Orquestra ingest → train → save → export ONNX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from triagem.pipeline.export_onnx import export_pipeline_to_onnx
from triagem.pipeline.ingest import load_laudos_csv, persist_processed
from triagem.pipeline.paths import (
    model_artifact_path,
    onnx_artifact_path,
    onnx_labels_path,
    processed_data_path,
    raw_data_path,
)
from triagem.pipeline.train import load_pipeline, save_metrics, save_pipeline, train_text_classifier

logger = logging.getLogger(__name__)


def run_training_pipeline(
    raw_csv: Path | None = None,
    processed_csv: Path | None = None,
    artifact_path: Path | None = None,
    seed: int = 42,
    export_onnx: bool = True,
) -> dict[str, Any]:
    """
    Executa o pipeline completo de treino (+ export ONNX por padrão).

    Returns:
        Dicionário com paths e métricas principais.
    """
    raw_csv = raw_csv or raw_data_path()
    processed_csv = processed_csv or processed_data_path()
    artifact_path = artifact_path or model_artifact_path()

    texts, labels = load_laudos_csv(raw_csv)
    persist_processed(texts, labels, processed_csv)
    pipeline, metrics = train_text_classifier(texts, labels, seed=seed)
    save_pipeline(pipeline, artifact_path)
    metrics_path = artifact_path.with_name("metrics.json")
    save_metrics(metrics, metrics_path)

    result: dict[str, Any] = {
        "raw_csv": str(raw_csv),
        "processed_csv": str(processed_csv),
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "accuracy": metrics["accuracy"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
    }

    if export_onnx:
        onnx_path, labels_path = export_pipeline_to_onnx(
            load_pipeline(artifact_path),
            onnx_artifact_path(),
            labels_path=onnx_labels_path(),
        )
        result["onnx_path"] = str(onnx_path)
        result["onnx_labels_path"] = str(labels_path)

    logger.info("pipeline_done", extra=result)
    return result
