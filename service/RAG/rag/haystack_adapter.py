"""Haystack-based RAG adapter with LightRAG-compatible method names.

This class keeps existing call-sites stable (`ainsert`, `aquery`, `adelete_by_doc_id`)
while replacing LightRAG internals with a Haystack-style ingestion and retrieval flow.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

try:
    from haystack import Document  # type: ignore
    from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter  # type: ignore
except Exception:  # pragma: no cover
    Document = None
    DocumentCleaner = None
    DocumentSplitter = None


@dataclass
class HaystackRAGSettings:
    postgres_url: str
    qdrant_url: str
    collection_name: str
    workspace: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int


class HaystackRAGAdapter:
    def __init__(
        self,
        *,
        settings: HaystackRAGSettings,
        llm_model_func: Callable[..., Awaitable[str]],
        embedding_func: Callable[[List[str]], Awaitable[Any]],
    ) -> None:
        self.settings = settings
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self._qdrant: Optional[QdrantClient] = None
        self._schema_ready = False

    @property
    def qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.settings.qdrant_url)
        return self._qdrant

    async def initialize_storages(self) -> None:
        await self._ensure_schema()
        self._ensure_qdrant_collection()

    async def ainsert(self, text: str, file_paths: Optional[str] = None) -> str:
        await self.initialize_storages()
        content = (text or "").strip()
        if not content:
            raise ValueError("cannot ingest empty text")

        document_id = f"doc_{uuid.uuid4().hex[:16]}"
        track_id = document_id
        file_path = (file_paths or f"{document_id}.txt").strip()
        metadata = {"file_path": file_path}

        chunks = self._chunk_document(content=content, document_id=document_id, file_path=file_path)
        vectors = await self._embed_texts([chunk["content"] for chunk in chunks])
        points = self._build_qdrant_points(chunks=chunks, vectors=vectors)
        if points:
            self.qdrant.upsert(collection_name=self.settings.collection_name, points=points)

        conn = await asyncpg.connect(self.settings.postgres_url)
        try:
            await conn.execute(
                """
                INSERT INTO sales.haystack_documents (
                    id, workspace, status, file_path, content_length, chunks_count,
                    track_id, metadata, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, NOW(), NOW())
                """,
                document_id,
                self.settings.workspace,
                "indexed",
                file_path,
                len(content),
                len(chunks),
                track_id,
                json.dumps(metadata, ensure_ascii=False),
            )

            for chunk in chunks:
                await conn.execute(
                    """
                    INSERT INTO sales.haystack_doc_chunks (
                        chunk_id, workspace, full_doc_id, file_path, chunk_order_index, content, metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW())
                    """,
                    chunk["chunk_id"],
                    self.settings.workspace,
                    document_id,
                    file_path,
                    chunk["chunk_order_index"],
                    chunk["content"],
                    json.dumps(chunk.get("metadata") or {}, ensure_ascii=False),
                )
        finally:
            await conn.close()

        return document_id

    async def aquery(
        self,
        query: str,
        *,
        top_k: int = 5,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        await self.initialize_storages()
        search_query = (query or "").strip()
        if not search_query:
            return ""

        vector = (await self._embed_texts([search_query]))[0]
        results = self.qdrant.search(
            collection_name=self.settings.collection_name,
            query_vector=vector,
            limit=max(1, top_k),
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="workspace",
                        match=MatchValue(value=self.settings.workspace),
                    )
                ]
            ),
        )

        context_blocks: List[str] = []
        for point in results:
            payload = dict(point.payload or {})
            content = str(payload.get("content") or "").strip()
            source = str(payload.get("file_path") or payload.get("document_id") or "unknown")
            if not content:
                continue
            context_blocks.append(f"[source={source}]\n{content}")

        if not context_blocks:
            return ""

        prompt = (
            "Bạn là trợ lý tư vấn bất động sản của Noble. "
            "Dựa trên ngữ cảnh truy xuất dưới đây để trả lời ngắn gọn, đúng trọng tâm. "
            "Nếu thiếu dữ liệu thì nói rõ là chưa đủ thông tin.\n\n"
            f"Question:\n{search_query}\n\n"
            "Retrieved Context:\n"
            + "\n\n---\n\n".join(context_blocks[:top_k])
        )
        history = conversation_history or []
        response = await self.llm_model_func(
            prompt,
            history_messages=history[-6:],
            enable_cot=False,
        )
        return str(response or "")

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        await self.initialize_storages()
        doc_id = (doc_id or "").strip()
        if not doc_id:
            return

        self.qdrant.delete(
            collection_name=self.settings.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="workspace", match=MatchValue(value=self.settings.workspace)),
                    FieldCondition(key="document_id", match=MatchValue(value=doc_id)),
                ]
            ),
        )

        conn = await asyncpg.connect(self.settings.postgres_url)
        try:
            await conn.execute(
                "DELETE FROM sales.haystack_doc_chunks WHERE workspace = $1 AND full_doc_id = $2",
                self.settings.workspace,
                doc_id,
            )
            await conn.execute(
                "DELETE FROM sales.haystack_documents WHERE workspace = $1 AND id = $2",
                self.settings.workspace,
                doc_id,
            )
        finally:
            await conn.close()

    def _ensure_qdrant_collection(self) -> None:
        collections = self.qdrant.get_collections().collections
        names = {c.name for c in collections}
        if self.settings.collection_name in names:
            return
        self.qdrant.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=VectorParams(size=self.settings.embedding_dim, distance=Distance.COSINE),
        )

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        conn = await asyncpg.connect(self.settings.postgres_url)
        try:
            await conn.execute(
                """
                CREATE SCHEMA IF NOT EXISTS sales;

                CREATE TABLE IF NOT EXISTS sales.haystack_documents (
                    id              TEXT PRIMARY KEY,
                    workspace       TEXT NOT NULL,
                    status          TEXT DEFAULT 'indexed',
                    file_path       TEXT,
                    content_length  INT DEFAULT 0,
                    chunks_count    INT DEFAULT 0,
                    track_id        TEXT,
                    metadata        JSONB DEFAULT '{}'::jsonb,
                    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_haystack_documents_workspace_updated
                    ON sales.haystack_documents(workspace, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_haystack_documents_track
                    ON sales.haystack_documents(track_id);

                CREATE TABLE IF NOT EXISTS sales.haystack_doc_chunks (
                    chunk_id          TEXT PRIMARY KEY,
                    workspace         TEXT NOT NULL,
                    full_doc_id       TEXT NOT NULL,
                    file_path         TEXT,
                    chunk_order_index INT DEFAULT 0,
                    content           TEXT NOT NULL,
                    metadata          JSONB DEFAULT '{}'::jsonb,
                    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_haystack_doc_chunks_workspace_doc
                    ON sales.haystack_doc_chunks(workspace, full_doc_id, chunk_order_index);
                """
            )
        finally:
            await conn.close()
        self._schema_ready = True

    def _chunk_document(self, *, content: str, document_id: str, file_path: str) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        if Document is not None and DocumentCleaner is not None and DocumentSplitter is not None:
            haystack_docs = [Document(content=content, meta={"document_id": document_id, "file_path": file_path})]
            cleaner = DocumentCleaner()
            cleaned = cleaner.run(documents=haystack_docs)["documents"]
            splitter = DocumentSplitter(
                split_by="word",
                split_length=max(100, self.settings.chunk_size),
                split_overlap=max(0, self.settings.chunk_overlap),
            )
            split_docs = splitter.run(documents=cleaned)["documents"]
            for idx, doc in enumerate(split_docs):
                text = str(doc.content or "").strip()
                if not text:
                    continue
                docs.append(
                    {
                        "chunk_id": f"chk_{uuid.uuid4().hex[:18]}",
                        "document_id": document_id,
                        "chunk_order_index": idx,
                        "content": text,
                        "metadata": dict(doc.meta or {}),
                        "file_path": file_path,
                    }
                )
            if docs:
                return docs

        # Fallback chunking if Haystack preprocessors are unavailable.
        words = content.split()
        size = max(100, self.settings.chunk_size)
        overlap = min(max(0, self.settings.chunk_overlap), max(0, size - 1))
        step = max(1, size - overlap)
        for idx, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + size]
            text = " ".join(chunk_words).strip()
            if not text:
                continue
            docs.append(
                {
                    "chunk_id": f"chk_{uuid.uuid4().hex[:18]}",
                    "document_id": document_id,
                    "chunk_order_index": idx,
                    "content": text,
                    "metadata": {"document_id": document_id, "file_path": file_path},
                    "file_path": file_path,
                }
            )
        return docs

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        arr = await self.embedding_func(texts)
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
        if not isinstance(arr, list):
            raise TypeError("embedding_func must return list-like vectors")
        vectors: List[List[float]] = []
        for item in arr:
            if hasattr(item, "tolist"):
                item = item.tolist()
            vec = [float(x) for x in item]
            if len(vec) != self.settings.embedding_dim:
                # Robustness: pad/truncate to configured dim.
                if len(vec) < self.settings.embedding_dim:
                    vec.extend([0.0] * (self.settings.embedding_dim - len(vec)))
                else:
                    vec = vec[: self.settings.embedding_dim]
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors

    def _build_qdrant_points(
        self,
        *,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> List[PointStruct]:
        points: List[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "workspace": self.settings.workspace,
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_order_index": chunk["chunk_order_index"],
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path"),
                        "metadata": chunk.get("metadata") or {},
                        "ingested_at": datetime.now(tz=timezone.utc).isoformat(),
                    },
                )
            )
        return points
