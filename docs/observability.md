# Observabilidade — Etapa 3

Stack local: **API FastAPI** + **Prometheus** + **Grafana**.

## Subir a stack

```bash
docker compose up --build -d
```

URLs:
- API: http://localhost:8000/health e http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (user/senha `admin`/`admin`)

Dashboard provisionado: **Triagem API — Latência × Throughput × Erros**
(`monitoring/grafana/dashboards/triagem-observability.json`)

## Gerar tráfego para ver os gráficos

```bash
for i in $(seq 1 40); do
  curl -s -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"text":"Paciente com dor torácica e suspeita de infarto."}' > /dev/null
done
```

## Latência × Throughput (o que monitorar)

| Métrica | Painel | Por quê importa |
|---------|--------|-----------------|
| Throughput (`rate(..._total[1m])`) | Painel 1 | Volume de triagens/segundo — capacidade do serviço. |
| Latência p50/p95 (histogram) | Painel 2 | Tempo de resposta percebido; p95 captura cauda lenta. |
| Taxa de erro | Painel 3 | Qualidade sob carga; erros sobem quando o sistema satura. |

Em produção hospitalar, o objetivo é manter **latência baixa e estável** mesmo com picos de throughput (plantão, plantão noturno, lotação de PS). Se o throughput sobe e o p95 explode, é sinal de saturação — hora de escalar o Cloud Run ou otimizar o modelo (Etapa 4 / ONNX).

## PromQL úteis

```promql
# req/s em /predict
sum(rate(triagem_http_requests_total{endpoint="/predict"}[1m]))

# latência p95
histogram_quantile(0.95, sum(rate(triagem_http_request_duration_seconds_bucket{endpoint="/predict"}[5m])) by (le))

# erros de inferência
sum(rate(triagem_predict_errors_total[1m]))
```

## Export do dashboard

O JSON versionado em `monitoring/grafana/dashboards/triagem-observability.json` é o entregável de configuração do dashboard (≥ 3 painéis).
