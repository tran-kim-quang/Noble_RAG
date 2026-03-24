FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies trước (cache layer)
COPY service/RAG/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ package service/RAG vào /app
# (tất cả subpackage: api, core, sales, memory, models, utils, rag, tools, llm, storage)
COPY service/RAG/ /app/

# Copy data/curated (system prompt, playbooks, project data)
COPY data/curated/ /app/data/curated/

# Copy scripts nếu cần (init SQL, utilities)
COPY scripts/ /app/scripts/

# PYTHONPATH = /app để các import tuyệt đối như
# "from core.config import ..." hoạt động đúng
ENV PYTHONPATH=/app

# Default env (bị override bởi docker-compose environment block)
ENV LLM_PROVIDER=openai
ENV LLM_MODEL=gpt-4o-mini
ENV EMBEDDING_PROVIDER=ollama
ENV EMBEDDING_MODEL=embeddinggemma:300m
ENV EMBEDDING_DIM=768
ENV LOG_LEVEL=INFO
ENV RAG_SERVICE_PORT=8000
ENV RAG_SERVICE_HOST=0.0.0.0

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
