FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set a default port in case you run it locally
ENV PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt --no-cache-dir

# Copy the entire src directory and any model files needed
COPY src/ src/
COPY models/ models/

# Expose is documentation only; Cloud Run uses the environment variable
EXPOSE $PORT

# Use 'sh -c' to ensure $PORT is evaluated at runtime
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]