FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

ENV PYTHONPATH=/app:/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt --no-cache-dir

COPY src/ src/
COPY models/ models/

CMD ["sh", "-c", "uvicorn src.credit_card_fraud_analysis.api:app --host 0.0.0.0 --port ${PORT}"]