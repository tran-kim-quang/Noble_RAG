FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY service/vision/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY service/vision/ /app/
COPY common/ /app/common/

ENV PYTHONPATH=/app
ENV VISION_SERVICE_PORT=8020
ENV VISION_SERVICE_HOST=0.0.0.0
ENV LOG_LEVEL=INFO

EXPOSE 8020

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8020/docs >/dev/null || exit 1

CMD ["python", "app.py"]
