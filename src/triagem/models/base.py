"""Contrato comum e implementações de classificadores de urgência."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

UrgencyLabel = Literal["normal", "atencao", "urgente"]

_URGENTE_PATTERNS = (
    r"\burgente\b",
    r"\bemerge?ncia\b",
    r"\binfarto\b",
    r"\bavc\b",
    r"\bchoque\b",
    r"\bparada\b",
    r"\bhemorragia\b",
    r"\binsufici[eê]ncia\s+respirat",
    r"\bsepsis\b",
    r"\bsepticemia\b",
    r"\btrauma\s+grave\b",
)

_ATENCAO_PATTERNS = (
    r"\batenc[aã]o\b",
    r"\baltera[cç][aã]o\b",
    r"\bsuspei[tc]",
    r"\bindefinid",
    r"\bacompanhar\b",
    r"\breavali",
    r"\bdolor",
    r"\bfebre\b",
    r"\binflam",
)


@dataclass(frozen=True)
class Prediction:
    """Resultado de uma classificação de urgência."""

    label: UrgencyLabel
    confidence: float
    model_kind: str


class UrgencyClassifier(ABC):
    """Interface Strategy para classificadores de triagem."""

    @property
    @abstractmethod
    def kind(self) -> str:
        """Identificador do backend (heuristic, sklearn, onnx...)."""

    @abstractmethod
    def predict(self, text: str) -> Prediction:
        """Classifica o texto do laudo."""


class HeuristicUrgencyClassifier(UrgencyClassifier):
    """
    Baseline por palavras-chave para a Etapa 1 (API + Docker).

    Será substituído/complementado por sklearn e ONNX nas etapas seguintes.
    """

    def __init__(self) -> None:
        self._urgente = [re.compile(p, re.IGNORECASE) for p in _URGENTE_PATTERNS]
        self._atencao = [re.compile(p, re.IGNORECASE) for p in _ATENCAO_PATTERNS]

    @property
    def kind(self) -> str:
        return "heuristic"

    def predict(self, text: str) -> Prediction:
        cleaned = (text or "").strip()
        if not cleaned:
            return Prediction(label="normal", confidence=0.5, model_kind=self.kind)

        if any(p.search(cleaned) for p in self._urgente):
            return Prediction(label="urgente", confidence=0.9, model_kind=self.kind)
        if any(p.search(cleaned) for p in self._atencao):
            return Prediction(label="atencao", confidence=0.75, model_kind=self.kind)
        return Prediction(label="normal", confidence=0.7, model_kind=self.kind)


class SklearnUrgencyClassifier(UrgencyClassifier):
    """Classificador sklearn (TF-IDF + Logistic Regression) via joblib."""

    def __init__(self, pipeline: object) -> None:
        self._pipeline = pipeline

    @property
    def kind(self) -> str:
        return "sklearn"

    def predict(self, text: str) -> Prediction:
        cleaned = (text or "").strip()
        if not cleaned:
            return Prediction(label="normal", confidence=0.5, model_kind=self.kind)

        label = str(self._pipeline.predict([cleaned])[0])
        confidence = 0.8
        if hasattr(self._pipeline, "predict_proba"):
            proba = self._pipeline.predict_proba([cleaned])[0]
            confidence = float(max(proba))
        if label not in {"normal", "atencao", "urgente"}:
            label = "normal"
        return Prediction(label=label, confidence=confidence, model_kind=self.kind)  # type: ignore[arg-type]


class OnnxUrgencyClassifier(UrgencyClassifier):
    """Classificador ONNX Runtime (mesmo pipeline TF-IDF + LR exportado)."""

    def __init__(self, session: object, classes: list[str]) -> None:
        self._session = session
        self._classes = classes
        self._input_name = session.get_inputs()[0].name
        self._output_names = [o.name for o in session.get_outputs()]

    @property
    def kind(self) -> str:
        return "onnx"

    def predict(self, text: str) -> Prediction:
        import numpy as np

        cleaned = (text or "").strip()
        if not cleaned:
            return Prediction(label="normal", confidence=0.5, model_kind=self.kind)

        # skl2onnx espera shape (N, 1) de strings
        payload = np.array([[cleaned]], dtype=object)
        outputs = self._session.run(self._output_names, {self._input_name: payload})

        label = "normal"
        confidence = 0.5
        # Saídas típicas com zipmap=False: label (str/int) + probabilities
        if len(outputs) >= 1:
            raw_label = outputs[0][0]
            if isinstance(raw_label, (int, np.integer)) and self._classes:
                idx = int(raw_label)
                label = self._classes[idx] if 0 <= idx < len(self._classes) else "normal"
            else:
                label = str(raw_label)
        if len(outputs) >= 2:
            proba = np.asarray(outputs[1][0]).reshape(-1)
            confidence = float(proba.max()) if proba.size else 0.5
        if label not in {"normal", "atencao", "urgente"}:
            label = "normal"
        return Prediction(label=label, confidence=confidence, model_kind=self.kind)  # type: ignore[arg-type]
