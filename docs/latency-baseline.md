# Latência — baselines e otimização ONNX

## Como medir (API HTTP)

```bash
poetry run uvicorn triagem.api.main:app --port 8000
poetry run python scripts/measure_latency.py --base-url http://127.0.0.1:8000 --n 30
```

## Como comparar sklearn vs ONNX (in-process)

```bash
poetry run python scripts/run_train_pipeline.py --seed 42
poetry run python scripts/benchmark_latency.py --n 200
```

Artefato: `models/artifacts/latency_comparison.json`

## Resultados

### API HTTP (Etapa 1 — heuristic)

| Ambiente | n | p50 (ms) | mean (ms) | p95 (ms) | Data |
|----------|---|----------|-----------|----------|------|
| Local (`uvicorn` + heuristic) | 30 | 3.99 | 4.07 | 5.02 | 2026-09-15 |

### Inferência do modelo (Etapa 4 — TF-IDF + Logistic Regression)

Medição in-process (sem HTTP), n=200 textos do dataset médico.

| Backend | p50 (ms) | mean (ms) | p95 (ms) | Speedup p50 |
|---------|----------|-----------|----------|-------------|
| sklearn (joblib) | 6.65 | 6.59 | 8.72 | 1.0× |
| **ONNX Runtime** | **0.53** | **0.55** | **0.94** | **~12.6×** |

- Melhoria no p50: **~92%** mais rápido com ONNX.
- Accuracy holdout do modelo: **0.759** (3200 train / 800 test).
- Técnica: exportação do pipeline sklearn com `skl2onnx` + inferência `onnxruntime` (CPU).

## Interpretação

Para triagem em tempo real, a latência de inferência entra no orçamento total da API (rede + pré-processamento + modelo). ONNX reduz o custo do modelo em ~11× no p50, liberando margem para picos de throughput monitorados no Grafana (Etapa 3) sem estourar o p95 clínico.
