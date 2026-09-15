"""Caminhos padrão do pipeline de treino."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]


def raw_data_path(filename: str = "laudos_sinteticos.csv") -> Path:
    """Retorna o caminho do CSV bruto em `data/raw/`."""
    return ROOT_DIR / "data" / "raw" / filename


def processed_data_path(filename: str = "laudos_processed.csv") -> Path:
    """Retorna o caminho do CSV processado em `data/processed/`."""
    return ROOT_DIR / "data" / "processed" / filename


def model_artifact_path(filename: str = "sklearn_pipeline.joblib") -> Path:
    """Retorna o caminho do artefato do modelo em `models/artifacts/`."""
    return ROOT_DIR / "models" / "artifacts" / filename
