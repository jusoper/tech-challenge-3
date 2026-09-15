#!/usr/bin/env python3
"""Mede baseline de latência local da API `/predict`."""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

SAMPLES = [
    "Exame dentro dos limites da normalidade. Sem alterações significativas.",
    "Achados inespecíficos; recomenda-se reavaliar e acompanhar evolução.",
    "Paciente com dor torácica intensa e suspeita de infarto. Encaminhar à emergência.",
]


def measure(base_url: str, n: int) -> list[float]:
    """Dispara N requisições e retorna latências em ms (lado cliente)."""
    url = f"{base_url.rstrip('/')}/predict"
    latencies: list[float] = []
    with httpx.Client(timeout=10.0) as client:
        for i in range(n):
            text = SAMPLES[i % len(SAMPLES)]
            started = time.perf_counter()
            response = client.post(url, json={"text": text})
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            latencies.append(elapsed_ms)
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=30, help="Número de requisições")
    args = parser.parse_args()

    # warm-up
    measure(args.base_url, n=3)
    latencies = measure(args.base_url, n=args.n)

    print(f"base_url={args.base_url}")
    print(f"n={len(latencies)}")
    print(f"p50_ms={statistics.median(latencies):.2f}")
    print(f"mean_ms={statistics.mean(latencies):.2f}")
    print(f"p95_ms={sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]:.2f}")
    print(f"min_ms={min(latencies):.2f}")
    print(f"max_ms={max(latencies):.2f}")


if __name__ == "__main__":
    main()
