"""Smoke tests da estrutura do repositório."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_paths_exist() -> None:
    expected = [
        "src/triagem/api/main.py",
        "src/triagem/models/factory.py",
        "src/triagem/pipeline/run.py",
        "airflow/dags/triagem_train_dag.py",
        ".github/workflows/ci.yml",
        "docker-compose.yml",
        "monitoring/prometheus.yml",
        "Dockerfile",
        "pyproject.toml",
        "README.md",
        ".env.example",
        ".gitignore",
        ".dockerignore",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert missing == [], f"faltando: {missing}"
