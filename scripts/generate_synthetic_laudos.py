#!/usr/bin/env python3
"""Gera CSV sintético de laudos para treino local / CI."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

TEMPLATES = {
    "normal": [
        "Exame dentro dos limites da normalidade. Sem alterações significativas.",
        "Achados compatíveis com a faixa etária. Nenhuma anormalidade detectada.",
        "Laudo sem alterações relevantes. Paciente estável e assintomático.",
        "Resultados laboratoriais dentro da referência. Evolução favorável.",
        "Imagem sem evidência de lesão aguda. Controle de rotina.",
    ],
    "atencao": [
        "Achados inespecíficos; recomenda-se reavaliar e acompanhar evolução.",
        "Alteração leve detectada. Manter observação clínica e retorno em breve.",
        "Suspeita de processo inflamatório. Indica-se acompanhamento.",
        "Quadro de febre baixa e dor moderada. Reavaliar em 48 horas.",
        "Resultado indefinido. Necessário correlacionar com clínica do paciente.",
    ],
    "urgente": [
        "Paciente com dor torácica intensa e suspeita de infarto. Encaminhar à emergência.",
        "Sinais de hemorragia ativa e choque. Prioridade máxima de atendimento.",
        "Quadro compatível com AVC em janela terapêutica. Urgente.",
        "Insuficiência respiratória aguda. Internação imediata recomendada.",
        "Trauma grave com risco de parada. Acionar protocolo de emergência.",
    ],
}


def generate_rows(n: int, seed: int) -> list[dict[str, str]]:
    """Gera `n` linhas equilibradas entre as 3 classes."""
    rng = random.Random(seed)
    labels = list(TEMPLATES.keys())
    rows: list[dict[str, str]] = []
    for i in range(n):
        label = labels[i % len(labels)]
        base = rng.choice(TEMPLATES[label])
        suffix = f" Caso #{i + 1}."
        rows.append({"text": base + suffix, "label": label})
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/laudos_sinteticos.csv"),
    )
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(args.n, args.seed)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
