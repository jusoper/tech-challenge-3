"""Configurações via variáveis de ambiente (Pydantic Settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Parâmetros de runtime da API e do modelo."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    seed: int = Field(default=42, description="Seed global para reprodutibilidade.")
    model_kind: str = Field(
        default="heuristic",
        description="Backend do classificador (heuristic; sklearn/onnx nas próximas etapas).",
    )
    artifacts_dir: Path = Field(default=ROOT_DIR / "models" / "artifacts")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")


def get_settings() -> Settings:
    """Carrega settings a partir do ambiente / `.env`."""
    return Settings()
