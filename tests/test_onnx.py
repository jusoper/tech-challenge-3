"""Testes de export ONNX e classificador otimizado (Etapa 4)."""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime as ort

from triagem.models.base import OnnxUrgencyClassifier
from triagem.pipeline.export_onnx import export_pipeline_to_onnx
from triagem.pipeline.run import run_training_pipeline
from triagem.pipeline.train import load_pipeline


def test_onnx_export_and_predict(tmp_path: Path) -> None:
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
    artifact = tmp_path / "model.joblib"
    result = run_training_pipeline(
        raw_csv=raw,
        processed_csv=tmp_path / "processed.csv",
        artifact_path=artifact,
        seed=42,
        export_onnx=False,
    )
    assert artifact.is_file()
    assert float(result["accuracy"]) >= 0.0

    onnx_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "onnx_labels.json"
    export_pipeline_to_onnx(load_pipeline(artifact), onnx_path, labels_path=labels_path)
    assert onnx_path.is_file()
    assert labels_path.is_file()

    classes = json.loads(labels_path.read_text(encoding="utf-8"))["classes"]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    clf = OnnxUrgencyClassifier(session, classes)
    pred = clf.predict("suspeita de infarto emergencia")
    assert pred.model_kind == "onnx"
    assert pred.label in {"normal", "atencao", "urgente"}
