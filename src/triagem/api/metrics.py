"""Métricas Prometheus da API de triagem."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "triagem_http_requests_total",
    "Total de requisições HTTP recebidas pela API de triagem.",
    labelnames=("method", "endpoint", "status"),
)

REQUEST_LATENCY = Histogram(
    "triagem_http_request_duration_seconds",
    "Latência das requisições HTTP em segundos.",
    labelnames=("method", "endpoint", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICT_ERRORS = Counter(
    "triagem_predict_errors_total",
    "Erros na inferência do endpoint /predict.",
)
