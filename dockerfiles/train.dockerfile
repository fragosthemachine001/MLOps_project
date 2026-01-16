FROM python:3.12-slim

# System dependencies (minimal)
RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (no cache to avoid disk issues)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ src/
COPY data/ data/

# Create output folders
RUN mkdir -p /app/models /app/reports/figures

# Run training
ENTRYPOINT ["python", "src/credit_card_fraud_analysis/train.py"]

