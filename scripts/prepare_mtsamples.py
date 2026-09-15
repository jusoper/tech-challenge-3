#!/usr/bin/env python3
"""
Converte MTSamples em CSV de triagem (`text`,`label`).

Fonte: mtsamples.csv (transcrições clínicas públicas).
Como o dataset original não traz urgência ESI, derivamos rótulos por
regras clínicas (palavras-chave + specialty) alinhadas ao desafio
normal / atencao / urgente. O mapeamento fica documentado abaixo e no README.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

LABELS = ("normal", "atencao", "urgente")

# Specialty → prior de urgência (pode ser sobrescrito por keywords no texto).
SPECIALTY_PRIOR = {
    "Emergency Room Reports": "urgente",
    "Hospice - Palliative Care": "atencao",
    "Discharge Summary": "atencao",
    "Surgery": "atencao",
    "Neurosurgery": "urgente",
    "Cardiovascular / Pulmonary": "atencao",
    "Neurology": "atencao",
    "Hematology - Oncology": "atencao",
    "General Medicine": "normal",
    "Consult - History and Phy.": "normal",
    "SOAP / Chart / Progress Notes": "normal",
    "Radiology": "normal",
    "Pediatrics - Neonatal": "atencao",
}

URGENTE_RE = re.compile(
    r"\b("
    r"emergency|emergent|urgent|critical|unstable|sepsis|septic|"
    r"myocardial infarction|infarct|stemi|nstemi|cardiac arrest|"
    r"stroke|cva|hemorrhage|haemorrhage|trauma|gunshot|overdose|"
    r"respiratory failure|intubat|icu|code blue|hypotensive shock|"
    r"anaphylaxis|status epilepticus"
    r")\b",
    re.IGNORECASE,
)

ATENCAO_RE = re.compile(
    r"\b("
    r"abnormal|elevated|follow[- ]?up|monitor|suspicion|suspected|"
    r"concerning|moderate|worsening|rule out|possible|evaluate|"
    r"complaint of|presents with|pain|fever|infection"
    r")\b",
    re.IGNORECASE,
)

NORMAL_RE = re.compile(
    r"\b("
    r"within normal limits|unremarkable|routine|elective|stable|"
    r"no acute|normal examination|well[- ]controlled|annual|"
    r"healthy|negative for"
    r")\b",
    re.IGNORECASE,
)


def derive_label(text: str, specialty: str) -> str:
    """Deriva urgência a partir do texto e da specialty."""
    urgente_hits = len(URGENTE_RE.findall(text))
    atencao_hits = len(ATENCAO_RE.findall(text))
    normal_hits = len(NORMAL_RE.findall(text))

    if urgente_hits >= 1 or specialty.strip() == "Emergency Room Reports":
        return "urgente"
    if specialty.strip() in {"Neurosurgery", "Cardiovascular / Pulmonary"} and atencao_hits >= 2:
        return "urgente"
    if atencao_hits >= 2 and atencao_hits > normal_hits:
        return "atencao"
    if normal_hits >= 1 and normal_hits >= atencao_hits:
        return "normal"
    return SPECIALTY_PRIOR.get(specialty.strip(), "atencao")


def clean_text(description: str, transcription: str, max_chars: int) -> str:
    """Junta description + início da transcription e normaliza espaços."""
    parts = [description.strip(), transcription.strip()]
    joined = " ".join(p for p in parts if p)
    joined = re.sub(r"\s+", " ", joined).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars].rsplit(" ", 1)[0]
    return joined


def prepare(
    source: Path,
    output: Path,
    max_chars: int,
    min_chars: int,
    seed: int,
    max_rows: int | None,
) -> Counter:
    """Lê MTSamples e grava CSV padronizado."""
    rng = random.Random(seed)
    rows_out: list[dict[str, str]] = []

    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            transcription = (row.get("transcription") or "").strip()
            description = (row.get("description") or "").strip()
            specialty = (row.get("medical_specialty") or "").strip()
            if not transcription and not description:
                continue
            text = clean_text(description, transcription, max_chars=max_chars)
            if len(text) < min_chars:
                continue
            label = derive_label(text, specialty)
            rows_out.append({"text": text, "label": label})

    rng.shuffle(rows_out)
    if max_rows is not None:
        rows_out = rows_out[:max_rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows_out)

    return Counter(r["label"] for r in rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/external/mtsamples.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/laudos_medicos.csv"))
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=4000)
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"fonte não encontrada: {args.source}. Rode scripts/download_mtsamples.py")

    counts = prepare(
        source=args.source,
        output=args.output,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    total = sum(counts.values())
    print(f"wrote {total} rows → {args.output}")
    for label in LABELS:
        print(f"  {label}: {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
