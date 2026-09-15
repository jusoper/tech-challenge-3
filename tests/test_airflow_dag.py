"""Garante a presença da DAG Airflow sem importar o pacote airflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAG_FILE = ROOT / "airflow" / "dags" / "triagem_train_dag.py"


def test_airflow_dag_file_exists_and_defines_tasks() -> None:
    assert DAG_FILE.is_file()
    content = DAG_FILE.read_text(encoding="utf-8")
    assert 'dag_id="triagem_train_pipeline"' in content
    assert 'task_id="ingest_laudos"' in content
    assert 'task_id="train_model"' in content
    assert 'task_id="save_model"' in content
    assert "ingest >> train >> save" in content
