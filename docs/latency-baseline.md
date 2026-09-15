# Baseline de latência — Etapa 1

Medição local da API `/predict` (classificador heurístico) para referência antes das otimizações da Etapa 4.

## Como medir

```bash
# terminal 1
poetry run uvicorn triagem.api.main:app --port 8000

# terminal 2
poetry run python scripts/measure_latency.py --base-url http://127.0.0.1:8000 --n 30
```

Via Docker:

```bash
docker build -t triagem-api:etapa1 .
docker run --rm -p 8000:8000 triagem-api:etapa1
poetry run python scripts/measure_latency.py --n 30
```

## Resultados

| Ambiente | n | p50 (ms) | mean (ms) | p95 (ms) | Data |
|----------|---|----------|-----------|----------|------|
| Local (`uvicorn` + heuristic) | 30 | 3.99 | 4.07 | 5.02 | 2026-09-15 |
| Docker (`triagem-api:etapa1`) | — | — | — | — | _pendente_ |

Observação: o baseline da Etapa 1 usa o classificador `heuristic`. Na Etapa 4 a mesma tabela recebe a linha do modelo sklearn e a do modelo otimizado (ex.: ONNX).
