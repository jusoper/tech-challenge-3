"""Testes da API de triagem (Etapa 1)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from triagem.api.main import app
from triagem.models import create_classifier


def test_health_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_kind"] == "heuristic"


def test_predict_urgente() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Paciente com suspeita de infarto e emergência imediata."},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "urgente"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0.0


def test_predict_normal() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Exame dentro dos limites da normalidade."},
        )
    assert response.status_code == 200
    assert response.json()["label"] == "normal"


def test_predict_rejects_empty() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_factory_heuristic() -> None:
    clf = create_classifier("heuristic")
    assert clf.kind == "heuristic"
    assert clf.predict("hemorragia intensa").label == "urgente"
