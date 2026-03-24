#!/usr/bin/env python3
"""
Initialize Qdrant collections for the Noble RAG system.

This script creates the necessary vector collections in Qdrant with proper
configuration for embedding search.

Usage:
    python scripts/init_qdrant.py
"""

import os
import sys
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'noble_rag_embeddings')

# Embedding configuration
EMBEDDING_DIM = 3072  # OpenAI text-embedding-3-large dimension (or 1536 for text-embedding-3-small)
DISTANCE_METRIC = Distance.COSINE  # Cosine similarity for semantic search


class QdrantInitializer:
    """Initialize and manage Qdrant collections for RAG system."""

    def __init__(self, url: str, api_key: Optional[str] = None):
        """Initialize Qdrant client."""
        kwargs = {'url': url}
        if api_key:
            kwargs['api_key'] = api_key

        try:
            self.client = QdrantClient(**kwargs)
            logger.info(f"Connected to Qdrant at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            sys.exit(1)

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def create_embeddings_collection(
        self,
        collection_name: str,
        vector_size: int = EMBEDDING_DIM,
        distance: Distance = DISTANCE_METRIC,
    ) -> bool:
        """
        Create a embeddings collection for RAG vectors.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of embedding vectors
            distance: Distance metric (Cosine, Euclidean, etc.)

        Returns:
            True if created successfully, False if already exists
        """
        if self.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            return False

        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logger.info(
                f"Created collection '{collection_name}' "
                f"(vectors: {vector_size}D, metric: {distance})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise

    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection."""
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': str(collection_info.config.params.vectors.distance),
                'point_count': collection_info.points_count,
                'vector_count': collection_info.vectors_count,
            }
        except Exception as e:
            logger.error(f"Failed to get info for '{collection_name}': {e}")
            return {}

    def health_check(self) -> bool:
        """Check if Qdrant server is healthy."""
        try:
            self.client.get_collections()
            logger.info("Qdrant health check passed")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def initialize_all(self) -> bool:
        """Initialize all required collections."""
        logger.info("Starting Qdrant initialization...")

        if not self.health_check():
            logger.error("Qdrant is not accessible")
            return False

        # Create main embeddings collection
        created = self.create_embeddings_collection(QDRANT_COLLECTION)

        # Get collection info
        info = self.get_collection_info(QDRANT_COLLECTION)
        if info:
            logger.info(f"Collection info: {info}")

        logger.info("Qdrant initialization completed successfully!")
        return True


def main():
    """Main initialization routine."""
    logger.info(f"Initializing Qdrant at {QDRANT_URL}")
    logger.info(f"Collection name: {QDRANT_COLLECTION}")
    logger.info(f"Vector dimension: {EMBEDDING_DIM}")
    logger.info(f"Distance metric: {DISTANCE_METRIC}")

    initializer = QdrantInitializer(QDRANT_URL, QDRANT_API_KEY)

    try:
        if initializer.initialize_all():
            logger.info("✓ Qdrant initialization successful")
            return 0
        else:
            logger.error("✗ Qdrant initialization failed")
            return 1
    except Exception as e:
        logger.error(f"✗ Initialization error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
