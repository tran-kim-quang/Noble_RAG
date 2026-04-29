FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY vision_service/requirements.txt /app/vision_requirements.txt
RUN pip install --no-cache-dir -r /app/vision_requirements.txt

COPY vision_service /app/vision_service

EXPOSE 8031

CMD ["uvicorn", "vision_service.app:app", "--host", "0.0.0.0", "--port", "8031"]
