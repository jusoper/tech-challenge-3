"""Middleware ASGI para instrumentação Prometheus."""

from __future__ import annotations

import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from triagem.api.metrics import PREDICT_ERRORS, REQUEST_COUNT, REQUEST_LATENCY


def _endpoint_label(path: str) -> str:
    """Normaliza path para label de baixa cardinalidade."""
    if path.startswith("/predict"):
        return "/predict"
    if path.startswith("/health"):
        return "/health"
    if path.startswith("/metrics"):
        return "/metrics"
    return "other"


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Conta requests, mede latência e erros de /predict."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path.startswith("/metrics"):
            return await call_next(request)

        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - started
            endpoint = _endpoint_label(request.url.path)
            labels = {
                "method": request.method,
                "endpoint": endpoint,
                "status": str(status_code),
            }
            REQUEST_COUNT.labels(**labels).inc()
            REQUEST_LATENCY.labels(**labels).observe(elapsed)
            if endpoint == "/predict" and status_code >= 400:
                PREDICT_ERRORS.inc()
