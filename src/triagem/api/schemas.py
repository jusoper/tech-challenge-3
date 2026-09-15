"""Contratos HTTP Pydantic para a API de triagem."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UrgencyLabel = Literal["normal", "atencao", "urgente"]


class TriageRequest(BaseModel):
    """Corpo de entrada com o texto do laudo médico."""

    text: str = Field(
        min_length=1,
        max_length=20_000,
        description="Texto livre do laudo / sintoma / exame.",
        examples=[
            "Paciente com dor torácica intensa e suspeita de infarto agudo.",
        ],
    )


class TriageResponse(BaseModel):
    """Classificação de urgência retornada pela API."""

    label: UrgencyLabel
    confidence: float = Field(ge=0.0, le=1.0)
    model_kind: str
    latency_ms: float = Field(ge=0.0, description="Tempo de inferência em milissegundos.")


class HealthResponse(BaseModel):
    """Liveness / readiness básico."""

    status: str
    model_kind: str
