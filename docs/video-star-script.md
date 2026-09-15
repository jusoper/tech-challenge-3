# Roteiro do vídeo STAR (≤ 5 minutos) — Tech Challenge Fase 3

Fale em tom natural. Se precisar enxugar, corte os trechos entre colchetes.

---

Olá. Neste Tech Challenge da FIAP, fase 3, a gente atacou um problema clínico realista: um hospital precisa de triagem automática de laudos em texto — classificar urgência em normal, atenção ou urgente — para acelerar o fluxo no pronto-socorro. O modelo é um classificador NLP leve, servido por API REST em Docker. O foco desta fase não era só treinar: era fechar o ciclo de MLOps com CI/CD, orquestração de retreino, monitoramento e otimização de latência.

A nossa tarefa técnica: API FastAPI containerizada; pipeline CI/CD no GitHub Actions com lint e testes; DAG Airflow para ingestão, treino e salvamento do modelo; stack Prometheus + Grafana; e pelo menos uma otimização de inferência — no nosso caso, exportação para ONNX Runtime — com comparação de latência documentada. Dataset público: MTSamples, transcrições clínicas, padronizadas em text/label. Entrega: repositório GitHub mais este vídeo STAR.

Na ação: [mostrar README / arquitetura] escolhemos GCP Cloud Run para inferência em tempo real, com Cloud Run Jobs para retreino batch — alinhado a FinOps e segurança de dados sensíveis. [mostrar árvore src/triagem] O código usa Factory e Strategy para trocar heuristic, sklearn e onnx sem mudar a API. [mostrar Actions] No push, o CI roda ruff, pytest, prepara o dataset e treina. [mostrar DAG] A DAG Airflow tem ingest_laudos → train_model → save_model. [mostrar Compose / Grafana] Com `docker compose up` sobem API, Prometheus e Grafana; o dashboard mostra throughput, latência p50/p95 e taxa de erro — o trade-off clássico latência versus throughput em produção. [mostrar benchmark] Treinamos TF-IDF + Logistic Regression, exportamos para ONNX e comparamos a latência in-process; o ONNX fica mais rápido no p50, com a mesma interface `/predict`.

O resultado: um serviço de triagem observável e reproduzível, com modelo otimizado e métricas claras. Lições: instrumentar cedo; manter o contrato da API estável enquanto o backend do modelo muda; e validar otimização com números, não só com feeling. Próximos passos: autenticação na API, dataset com rótulos ESI clínicos oficiais, e deploy do container no Cloud Run.

Obrigado.
