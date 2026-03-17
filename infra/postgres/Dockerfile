FROM pgvector/pgvector:pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-age \
    && rm -rf /var/lib/apt/lists/*
