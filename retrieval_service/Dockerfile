FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG EMBEDDING_PROFILE=remote

WORKDIR /app

COPY retrieval_service/requirements.remote.txt /app/retrieval_requirements.remote.txt
COPY retrieval_service/requirements.local.txt /app/retrieval_requirements.local.txt
RUN set -eux; \
    case "${EMBEDDING_PROFILE}" in \
      remote) REQUIREMENTS_FILE="/app/retrieval_requirements.remote.txt" ;; \
      local) REQUIREMENTS_FILE="/app/retrieval_requirements.local.txt" ;; \
      *) echo "Unsupported EMBEDDING_PROFILE='${EMBEDDING_PROFILE}' (use: remote|local)" && exit 1 ;; \
    esac; \
    pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"

COPY retrieval_service /app/retrieval_service
COPY data /app/data
COPY scripts /app/scripts

EXPOSE 8011

CMD ["uvicorn", "retrieval_service.app:app", "--host", "0.0.0.0", "--port", "8011"]
