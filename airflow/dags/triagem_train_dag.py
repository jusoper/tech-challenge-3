"""
DAG Airflow: ingestão → treino → salvamento do modelo de triagem.

Uso típico (local):
  export AIRFLOW_HOME="$(pwd)/airflow"
  export PYTHONPATH="$(pwd)/src"
  airflow dags list
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _task_ingest(**_context) -> str:
    """Lê o CSV bruto e persiste a versão validada em data/processed/."""
    from triagem.pipeline.ingest import load_laudos_csv, persist_processed
    from triagem.pipeline.paths import processed_data_path, raw_data_path

    texts, labels = load_laudos_csv(raw_data_path())
    out = persist_processed(texts, labels, processed_data_path())
    LOGGER.info("ingest finished: %s (%s rows)", out, len(texts))
    return str(out)


def _task_train(**_context) -> str:
    """Treina TF-IDF + RandomForest a partir do CSV processado."""
    from triagem.pipeline.ingest import load_laudos_csv
    from triagem.pipeline.paths import processed_data_path
    from triagem.pipeline.train import train_text_classifier

    texts, labels = load_laudos_csv(processed_data_path())
    pipeline, metrics = train_text_classifier(texts, labels, seed=42)
    # Passa métricas via XCom; o artefato é gravado na task seguinte.
    _context["ti"].xcom_push(key="accuracy", value=float(metrics["accuracy"]))
    _context["ti"].xcom_push(key="n_train", value=float(metrics["n_train"]))
    _context["ti"].xcom_push(key="n_test", value=float(metrics["n_test"]))
    _context["ti"].xcom_push(key="report", value=str(metrics["report"]))
    # Mantém pipeline em arquivo temporário do worker via save imediato parcial
    from triagem.pipeline.paths import model_artifact_path
    from triagem.pipeline.train import save_pipeline

    artifact = save_pipeline(pipeline, model_artifact_path())
    LOGGER.info("train finished accuracy=%s path=%s", metrics["accuracy"], artifact)
    return str(artifact)


def _task_save_model(**_context) -> str:
    """Persiste métricas e confirma que o artefato do modelo existe."""
    from triagem.pipeline.paths import model_artifact_path
    from triagem.pipeline.train import save_metrics

    artifact = model_artifact_path()
    if not artifact.is_file():
        raise FileNotFoundError(f"modelo não encontrado: {artifact}")

    ti = _context["ti"]
    metrics = {
        "accuracy": float(ti.xcom_pull(task_ids="train_model", key="accuracy")),
        "n_train": float(ti.xcom_pull(task_ids="train_model", key="n_train")),
        "n_test": float(ti.xcom_pull(task_ids="train_model", key="n_test")),
        "report": str(ti.xcom_pull(task_ids="train_model", key="report") or ""),
    }
    metrics_path = save_metrics(metrics, artifact.with_name("metrics.json"))
    LOGGER.info("model saved at %s ; metrics at %s", artifact, metrics_path)
    return str(artifact)


with DAG(
    dag_id="triagem_train_pipeline",
    description="Pipeline de treino/retreino do classificador de laudos",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["triagem", "ml", "tech-challenge-3"],
) as dag:
    ingest = PythonOperator(task_id="ingest_laudos", python_callable=_task_ingest)
    train = PythonOperator(task_id="train_model", python_callable=_task_train)
    save = PythonOperator(task_id="save_model", python_callable=_task_save_model)

    ingest >> train >> save
