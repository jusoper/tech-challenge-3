#!/usr/bin/env python3
"""Compara latência in-process: sklearn (joblib) vs ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from triagem.models import create_classifier
from triagem.pipeline.ingest import load_laudos_csv
from triagem.pipeline.paths import raw_data_path

SAMPLES_FALLBACK = [
    "Exame dentro dos limites da normalidade. Sem alterações significativas.",
    "Achados inespecíficos; recomenda-se reavaliar e acompanhar evolução.",
    "Paciente com dor torácica intensa e suspeita de infarto. Encaminhar à emergência.",
]


def _load_texts(n: int) -> list[str]:
    raw = raw_data_path()
    if raw.is_file():
        texts, _labels = load_laudos_csv(raw)
        if texts:
            return [texts[i % len(texts)] for i in range(n)]
    return [SAMPLES_FALLBACK[i % len(SAMPLES_FALLBACK)] for i in range(n)]


def _bench(kind: str, texts: list[str], warmup: int = 10) -> dict[str, float]:
    clf = create_classifier(kind)
    for text in texts[:warmup]:
        clf.predict(text)
    latencies: list[float] = []
    for text in texts:
        started = time.perf_counter()
        clf.predict(text)
        latencies.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(latencies)
    p95_idx = max(0, int(0.95 * len(ordered)) - 1)
    return {
        "n": float(len(latencies)),
        "p50_ms": float(statistics.median(latencies)),
        "mean_ms": float(statistics.mean(latencies)),
        "p95_ms": float(ordered[p95_idx]),
        "min_ms": float(min(latencies)),
        "max_ms": float(max(latencies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/artifacts/latency_comparison.json"),
    )
    args = parser.parse_args()

    texts = _load_texts(args.n)
    sklearn_stats = _bench("sklearn", texts)
    onnx_stats = _bench("onnx", texts)

    speedup = sklearn_stats["p50_ms"] / onnx_stats["p50_ms"] if onnx_stats["p50_ms"] else 0.0
    payload = {
        "sklearn": sklearn_stats,
        "onnx": onnx_stats,
        "speedup_p50": round(speedup, 3),
        "improvement_pct_p50": round((1 - (1 / speedup)) * 100, 2) if speedup else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
