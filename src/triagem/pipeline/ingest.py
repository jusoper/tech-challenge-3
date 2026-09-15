"""Ingestão de laudos a partir de CSV (text, label)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("text", "label")
VALID_LABELS = frozenset({"normal", "atencao", "urgente"})


def load_laudos_csv(path: Path) -> tuple[list[str], list[str]]:
    """
    Lê um CSV com colunas `text` e `label`.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        Tupla `(texts, labels)`.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se colunas ou labels forem inválidos.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV sem cabeçalho")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

        for row_num, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text:
                raise ValueError(f"linha {row_num}: text vazio")
            if label not in VALID_LABELS:
                raise ValueError(f"linha {row_num}: label inválido {label!r}")
            texts.append(text)
            labels.append(label)

    if len(texts) < 10:
        raise ValueError(f"dataset muito pequeno: {len(texts)} linhas (mínimo 10)")

    logger.info("ingest_ok", extra={"rows": len(texts), "path": str(path)})
    return texts, labels


def persist_processed(
    texts: list[str],
    labels: list[str],
    output_path: Path,
) -> Path:
    """Grava o dataset validado em `data/processed/`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        for text, label in zip(texts, labels):
            writer.writerow({"text": text, "label": label})
    logger.info("processed_saved", extra={"path": str(output_path), "rows": len(texts)})
    return output_path
