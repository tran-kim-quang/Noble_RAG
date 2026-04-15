FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY orchestrator_service/requirements.txt /app/orchestrator_requirements.txt
RUN pip install --no-cache-dir -r /app/orchestrator_requirements.txt

COPY orchestrator_service /app/orchestrator_service

EXPOSE 8021

CMD ["uvicorn", "orchestrator_service.app:app", "--host", "0.0.0.0", "--port", "8021"]
    