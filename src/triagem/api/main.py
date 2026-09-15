"""Aplicação FastAPI: `/health` e `/predict` (Etapa 1)."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from triagem.api.schemas import HealthResponse, TriageRequest, TriageResponse
from triagem.config import get_settings
from triagem.models import UrgencyClassifier, create_classifier

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    classifier = create_classifier(settings.model_kind)
    app.state.classifier = classifier
    app.state.model_kind = classifier.kind
    logger.info("classifier_ready", extra={"model_kind": classifier.kind})
    yield
    app.state.classifier = None


app = FastAPI(
    title="Triagem de Laudos Médicos",
    description="API de classificação de urgência (normal / atenção / urgente).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Confirma que o processo está no ar e o classificador foi carregado."""
    classifier = getattr(request.app.state, "classifier", None)
    kind = getattr(request.app.state, "model_kind", "unknown")
    if classifier is None:
        raise HTTPException(status_code=503, detail="classificador não inicializado")
    return HealthResponse(status="ok", model_kind=kind)


@app.post("/predict", response_model=TriageResponse, tags=["inference"])
def predict(payload: TriageRequest, request: Request) -> TriageResponse:
    """Recebe o texto do laudo e retorna a classificação de urgência."""
    classifier: UrgencyClassifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="classificador não inicializado")

    started = time.perf_counter()
    try:
        result = classifier.predict(payload.text)
    except Exception as exc:
        logger.exception("predict_failed")
        raise HTTPException(status_code=400, detail=f"erro na inferência: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "predict_ok",
        extra={"label": result.label, "latency_ms": round(latency_ms, 3)},
    )
    return TriageResponse(
        label=result.label,
        confidence=result.confidence,
        model_kind=result.model_kind,
        latency_ms=round(latency_ms, 3),
    )
