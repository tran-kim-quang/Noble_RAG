from __future__ import annotations

import logging
from uuid import uuid4

from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions

from retrieval_service.config import Settings
from retrieval_service.schemas import IngestDocument

log = logging.getLogger("retrieval-service")


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.document_store = QdrantDocumentStore(
            url=settings.qdrant_url,
            index=settings.collection_name,
            embedding_dim=settings.embedding_dim,
            api_key=settings.qdrant_api_key or None,
            recreate_index=False,
        )

        self.doc_embedder = SentenceTransformersDocumentEmbedder(model=settings.embedding_model)
        self.text_embedder = SentenceTransformersTextEmbedder(model=settings.embedding_model)

        self.doc_embedder.warm_up()
        self.text_embedder.warm_up()
        self._validate_embedding_dim()
        self._validate_qdrant_collection_dim()

        self.indexing = Pipeline()
        self.indexing.add_component("embedder", self.doc_embedder)
        self.indexing.add_component(
            "writer",
            DocumentWriter(document_store=self.document_store, policy=DuplicatePolicy.OVERWRITE),
        )
        self.indexing.connect("embedder.documents", "writer.documents")

        self.retrieval = Pipeline()
        self.retrieval.add_component("embedder", self.text_embedder)
        self.retrieval.add_component("retriever", QdrantEmbeddingRetriever(document_store=self.document_store))
        self.retrieval.connect("embedder.embedding", "retriever.query_embedding")
        log.info(
            "startup complete collection=%s model=%s dim=%s qdrant=%s",
            settings.collection_name,
            settings.embedding_model,
            settings.embedding_dim,
            settings.qdrant_url,
        )

    def health(self) -> dict:
        try:
            self.document_store.count_documents()
            status = "ok"
        except Exception:
            status = "degraded"
        return {"status": status, "collection": self.settings.collection_name}

    def ingest(self, documents: list[IngestDocument]) -> int:
        haystack_docs = [
            Document(
                id=item.doc_id or str(uuid4()),
                content=item.text,
                meta={
                    "source": item.source or "unknown",
                    **(item.metadata or {}),
                },
            )
            for item in documents
        ]

        out = self.indexing.run({"embedder": {"documents": haystack_docs}})
        written = int(out["writer"]["documents_written"])
        log.info("ingest documents=%s written=%s", len(documents), written)
        return written

    def retrieve(self, query: str, top_k: int | None = None) -> dict:
        query = (query or "").strip()
        if not query:
            return {"results": [], "confidence": 0.0, "low_confidence": True}
        k = top_k or self.settings.default_top_k
        k = max(1, min(k, self.settings.max_top_k))
        out = self.retrieval.run({"embedder": {"text": query}, "retriever": {"top_k": k}})
        docs = out.get("retriever", {}).get("documents", []) or []
        confidence = float(docs[0].score or 0.0) if docs else 0.0
        low_confidence = confidence < self.settings.min_retrieve_score
        log.info(
            "retrieve query_len=%s top_k=%s results=%s confidence=%.4f threshold=%.4f low_confidence=%s",
            len(query),
            k,
            len(docs),
            confidence,
            self.settings.min_retrieve_score,
            low_confidence,
        )

        results = [
            {
                "text": doc.content or "",
                "score": float(doc.score or 0.0),
                "source": str((doc.meta or {}).get("source", "unknown")),
                "doc_id": str(doc.id),
            }
            for doc in docs
        ]
        if low_confidence and self.settings.low_confidence_empty_results:
            results = []
        return {
            "results": results,
            "confidence": confidence,
            "low_confidence": low_confidence,
        }

    def _validate_embedding_dim(self) -> None:
        embedding = self.text_embedder.run("dimension probe")["embedding"]
        model_dim = len(embedding)
        if model_dim != self.settings.embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"EMBEDDING_DIM={self.settings.embedding_dim} but model '{self.settings.embedding_model}' returns {model_dim}. "
                "Please update EMBEDDING_DIM or choose a compatible model."
            )

    def _validate_qdrant_collection_dim(self) -> None:
        client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
            timeout=self.settings.qdrant_timeout_sec,
        )
        try:
            collection = client.get_collection(self.settings.collection_name)
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code == 404:
                return
            raise
        vectors_config = collection.config.params.vectors
        if isinstance(vectors_config, dict):
            named = next(iter(vectors_config.values()), None)
            current_size = getattr(named, "size", None)
        else:
            current_size = getattr(vectors_config, "size", None)
        if current_size is None:
            return
        if int(current_size) != int(self.settings.embedding_dim):
            raise ValueError(
                "Qdrant collection dimension mismatch: "
                f"collection='{self.settings.collection_name}' has dim={current_size}, "
                f"but EMBEDDING_DIM={self.settings.embedding_dim}. "
                "Use a new collection name or align EMBEDDING_DIM/model."
            )
