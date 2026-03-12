FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV LLM_PROVIDER=openai
ENV LLM_MODEL=gpt-4o-mini
ENV EMBEDDING_MODEL=text-embedding-3-large
ENV EMBEDDING_DIM=3072
ENV LOG_LEVEL=INFO
ENV RAG_SERVICE_PORT=8001
ENV RAG_SERVICE_HOST=0.0.0.0

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

CMD ["python", "app.py"]
