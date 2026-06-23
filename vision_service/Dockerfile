FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY vision_service/requirements.txt /app/vision_requirements.txt
RUN pip install --no-cache-dir --default-timeout=600 --retries=10 -r /app/vision_requirements.txt

COPY vision_service /app/vision_service

EXPOSE 8031

CMD ["uvicorn", "vision_service.app:app", "--host", "0.0.0.0", "--port", "8031"]
