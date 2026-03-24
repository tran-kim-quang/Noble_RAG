#!/bin/bash

set -e

IMAGE_NAME="noble-rag-service"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image: ${FULL_IMAGE}"

docker build -t "${FULL_IMAGE}" .

echo "Build successful!"
echo "To run the service:"
echo "  docker run -p 8001:8001 -e LLM_API_KEY=your_key ${FULL_IMAGE}"
echo ""
echo "Or with docker-compose:"
echo "  docker-compose up rag-service"
