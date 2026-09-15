# Multi-stage image for the inference API (Deploy em Nuvem — Aulas 01–05).
# Builder installs deps; runtime copies only the venv + app code.

FROM python:3.11-slim AS builder

ENV POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv /opt/venv \
    && pip install "poetry==${POETRY_VERSION}" poetry-plugin-export

COPY pyproject.toml poetry.lock* ./
RUN poetry export --without-hashes --only main -o requirements.txt \
    && pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MODEL_KIND=heuristic

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml ./
COPY src ./src

RUN mkdir -p models/artifacts data/raw data/processed \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "triagem.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
