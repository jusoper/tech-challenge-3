# Tech Challenge FIAP — Fase 3

Triagem automática de **laudos médicos** (NLP leve) servida via **FastAPI + Docker**, com pipeline CI/CD, Airflow, monitoramento (Prometheus/Grafana) e otimização de latência.

> Status atual: **Etapa 2** — CI (GitHub Actions) + pipeline de treino + DAG Airflow.

## Decisão arquitetural (Deploy em Nuvem)

### Problema e requisito de latência

O hospital precisa classificar urgência (`normal` / `atencao` / `urgente`) a partir do texto do laudo. A decisão de triagem afeta o fluxo clínico **na hora** — portanto o padrão principal é **inferência em tempo real** (API HTTP), não batch.

Comparativo alinhado às aulas (batch × tempo real × serverless):

| Padrão | Adequação ao cenário | Motivo |
|--------|----------------------|--------|
| **Tempo real (API)** | **Escolhido** | Latência baixa e determinística para triagem. |
| Batch | Complementar | Retreino / reprocessamento noturno de laudos históricos. |
| Serverless puro | Possível, com ressalvas | Cold start pode atrasar triagem em horários de baixo tráfego. |

### Provedor escolhido: **GCP (Cloud Run + Artifact Registry)**

Critérios: container-first (mesmo Dockerfile local → nuvem), escala automática, URL HTTPS gerenciada e boa integração com jobs de retreino.

| Camada | Serviço GCP | Papel |
|--------|-------------|-------|
| Imagem | **Artifact Registry** | Versionar a imagem da API. |
| Inferência real-time | **Cloud Run** | Servir FastAPI com autoscaling. |
| Retreino / batch | **Cloud Run Jobs** (+ Cloud Storage) | Pipeline de treino agendado (Airflow dispara / job executa). |
| ML gerenciado (opcional) | **Vertex AI** | Registry/endpoints se o time escalar o ciclo de vida do modelo. |

Alternativas consideradas:

- **AWS** (ECR + ECS/Fargate ou Lambda + API Gateway + SageMaker): maduro, porém mais peças para o mesmo resultado containerizado.
- **Azure** (ACR + Container Apps / Azure ML): viável; o time já tem mais familiaridade prática com o fluxo GCP do curso.

Arquitetura híbrida final: **Cloud Run (real-time)** + **Jobs batch para retreino** — exatamente o padrão “coexistência” visto na Disciplina 01.

### FinOps

- Preferir **Cloud Run com escala a zero** (ou `min instances=1` só se a latência de cold start for inaceitável para o hospital).
- Cobrança por request/CPU-segundo alinha custo ao volume real de laudos.
- Jobs de retreino sob demanda evitam VM ociosa 24/7.
- Tags/labels por ambiente (`dev`/`prod`) para visibilidade de custo (FinOps).
- Revisar periodicamente: endpoints sem tráfego, imagens antigas no registry, jobs sem necessidade.

### Segurança

- Laudos são **dados sensíveis de saúde**: TLS obrigatório; nunca logar o texto completo do laudo em claro em produção.
- IAM com **menor privilégio** (service account só para Artifact Registry + Cloud Run + bucket do job).
- API atrás de autenticação (API key / IAP / Identity) antes de exposição pública.
- Rate limiting e validação de payload (já iniciada com Pydantic `max_length`).
- Segredos apenas em Secret Manager / `.env` local (nunca commitados).

## Setup local

```bash
cp .env.example .env
poetry install
poetry run uvicorn triagem.api.main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Inferência:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Paciente com dor torácica e suspeita de infarto."}'
```

## Docker

```bash
docker build -t triagem-api:etapa1 .
docker run --rm -p 8000:8000 triagem-api:etapa1
```

## Baseline de latência

Com a API no ar (local ou Docker):

```bash
poetry run python scripts/measure_latency.py --base-url http://127.0.0.1:8000 --n 30
```

Os números ficam registrados em `docs/latency-baseline.md` após a medição.

## Dataset sintético + treino (Etapa 2)

```bash
poetry run python scripts/generate_synthetic_laudos.py --n 300 --seed 42
poetry run python scripts/run_train_pipeline.py --seed 42
```

Artefatos gerados:
- `data/raw/laudos_sinteticos.csv`
- `data/processed/laudos_processed.csv`
- `models/artifacts/sklearn_pipeline.joblib`
- `models/artifacts/metrics.json`

Para servir a API com o modelo treinado:

```bash
MODEL_KIND=sklearn poetry run uvicorn triagem.api.main:app --port 8000
```

## CI/CD (GitHub Actions)

Workflow em `.github/workflows/ci.yml` (lint + format check + pytest + smoke de treino) nos eventos `push`/`pull_request`.

## Airflow DAG

Arquivo: `airflow/dags/triagem_train_dag.py`  
Tasks: `ingest_laudos` → `train_model` → `save_model`.

```bash
export AIRFLOW_HOME="$(pwd)/airflow"
export PYTHONPATH="$(pwd)/src"
# requer Apache Airflow instalado no ambiente de orquestração
airflow dags list
airflow dags test triagem_train_pipeline
```

## Testes

```bash
poetry run pytest
poetry run ruff check src scripts tests
```

## Estrutura

```
src/triagem/
  api/          # FastAPI (/health, /predict)
  config/       # Pydantic Settings
  models/       # Strategy + Factory do classificador
  pipeline/     # ingest → train → save
scripts/        # generate data, train, latency
airflow/dags/   # DAG de treino/retreino
.github/workflows/ci.yml
monitoring/     # Prometheus/Grafana (Etapa 3)
```

## Roadmap das etapas

1. **Etapa 1:** arquitetura + API + Docker + baseline
2. **Etapa 2 (atual):** GitHub Actions + DAG Airflow + pipeline de treino
3. **Etapa 3:** Prometheus + Grafana no Compose
4. **Etapa 4:** otimização ONNX + comparação de latência + vídeo STAR
