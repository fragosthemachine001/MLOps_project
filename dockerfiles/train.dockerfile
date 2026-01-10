FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock requirements.txt README.md ./

#Cache dependencies to avoid re-downloading
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

RUN mkdir -p /app/models /app/reports/figures
COPY src/ src/
COPY data/ data/

ENTRYPOINT ["uv", "run", "src/credit_card_fraud_analysis/train.py"]
