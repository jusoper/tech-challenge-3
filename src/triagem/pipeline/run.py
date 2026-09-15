"""Orquestra ingest → train → save (usado pela CLI e pela DAG Airflow)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from triagem.pipeline.ingest import load_laudos_csv, persist_processed
from triagem.pipeline.paths import model_artifact_path, processed_data_path, raw_data_path
from triagem.pipeline.train import save_metrics, save_pipeline, train_text_classifier

logger = logging.getLogger(__name__)


def run_training_pipeline(
    raw_csv: Path | None = None,
    processed_csv: Path | None = None,
    artifact_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Executa o pipeline completo de treino.

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

    result = {
        "raw_csv": str(raw_csv),
        "processed_csv": str(processed_csv),
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "accuracy": metrics["accuracy"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
    }
    logger.info("pipeline_done", extra=result)
    return result
