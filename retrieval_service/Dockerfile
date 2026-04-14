FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY retrieval_service/requirements.txt /app/retrieval_requirements.txt
RUN pip install --no-cache-dir -r /app/retrieval_requirements.txt

COPY retrieval_service /app/retrieval_service
COPY data /app/data

EXPOSE 8011

CMD ["uvicorn", "retrieval_service.app:app", "--host", "0.0.0.0", "--port", "8011"]
