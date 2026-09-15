"""Testes de instrumentação Prometheus (Etapa 3)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from triagem.api.main import app


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    with TestClient(app) as client:
        client.get("/health")
        client.post("/predict", json={"text": "exame dentro da normalidade"})
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "triagem_http_requests_total" in body
    assert "triagem_http_request_duration_seconds" in body


def test_monitoring_files_exist() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / "docker-compose.yml").is_file()
    assert (root / "monitoring" / "prometheus.yml").is_file()
    dashboard = root / "monitoring" / "grafana" / "dashboards" / "triagem-observability.json"
    assert dashboard.is_file()
    content = dashboard.read_text(encoding="utf-8")
    assert "Throughput" in content
    assert "Latência" in content
    assert "Taxa de erro" in content
