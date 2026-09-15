"""Testes do pipeline de treino (Etapa 2)."""

from __future__ import annotations

from pathlib import Path

from triagem.models import create_classifier
from triagem.pipeline.ingest import load_laudos_csv
from triagem.pipeline.run import run_training_pipeline


def test_run_training_pipeline(tmp_path: Path) -> None:
    raw = tmp_path / "laudos.csv"
    raw.write_text(
        "text,label\n"
        + "\n".join(
            [
                "exame normal sem alteracoes,normal",
                "exame dentro da normalidade,normal",
                "laudo sem alteracoes relevantes,normal",
                "achados inespecificos acompanhar,atencao",
                "alteracao leve reavaliar,atencao",
                "febre e dor moderada acompanhar,atencao",
                "suspeita de infarto emergencia,urgente",
                "hemorragia e choque urgente,urgente",
                "avc em janela terapeutica urgente,urgente",
                "trauma grave parada emergencia,urgente",
                "controle de rotina normal,normal",
                "resultado indefinido atencao,atencao",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    processed = tmp_path / "processed.csv"
    artifact = tmp_path / "model.joblib"
    result = run_training_pipeline(
        raw_csv=raw,
        processed_csv=processed,
        artifact_path=artifact,
        seed=42,
    )
    assert artifact.is_file()
    assert processed.is_file()
    assert float(result["accuracy"]) >= 0.0

    clf = create_classifier("sklearn", artifact_path=artifact)
    pred = clf.predict("suspeita de infarto emergencia")
    assert pred.model_kind == "sklearn"
    assert pred.label in {"normal", "atencao", "urgente"}


def test_load_laudos_rejects_bad_label(tmp_path: Path) -> None:
    raw = tmp_path / "bad.csv"
    raw.write_text("text,label\nola,invalido\n" + ("ok,normal\n" * 12), encoding="utf-8")
    try:
        load_laudos_csv(raw)
        raised = False
    except ValueError:
        raised = True
    assert raised
