FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Giá trị mặc định — ghi đè qua environment trong docker-compose
ENV REDIS_URL=redis://redis:6379/0
ENV REDIS_QUEUE_KEY=livestream:comments
ENV POP_TIMEOUT=5
ENV RAG_SERVICE_URL=http://rag-service:8000
ENV LIVESTREAM_SESSION_ID=livestream_session
ENV RAG_REQUEST_TIMEOUT=60
ENV MAX_RAG_RETRIES=2
ENV RETRY_DELAY_SEC=2.0
ENV LOG_LEVEL=INFO

CMD ["python", "-u", "main.py"]
