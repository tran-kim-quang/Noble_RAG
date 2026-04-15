from __future__ import annotations

from collections import Counter, defaultdict
import logging
import re
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

_TOPIC_TO_STRENGTH = {
    "location": "kết nối giao thông thuận tiện",
    "amenities": "tiện ích nội khu đa dạng cho sinh hoạt hàng ngày",
    "legal": "pháp lý có hồ sơ hiện hữu để đối chiếu",
    "bank": "có hỗ trợ tài chính từ ngân hàng liên kết",
    "timeline": "thông tin tiến độ và bàn giao tương đối rõ",
    "price": "khung giá và sản phẩm được công bố rõ ràng",
}

_TOPIC_TO_TRADEOFF = {
    "location": "cần kiểm tra thời gian di chuyển giờ cao điểm theo lịch trình thực tế",
    "amenities": "nên kiểm tra thêm mức phí vận hành đi kèm hệ tiện ích",
    "price": "cần cân đối ngân sách theo tầng, view và tiến độ thanh toán",
    "timeline": "mốc bàn giao luôn có thể thay đổi theo điều kiện thị trường và pháp lý",
}

_TOPIC_TO_TRAITS = {
    "location": ["near_key_arteries", "central_commute_heavy"],
    "amenities": ["family_friendly"],
    "legal": ["legal_safe"],
    "bank": ["investor_fit"],
    "price": ["premium_pricing"],
    "timeline": ["long_term_liveability"],
}

_TRAIT_REASON = {
    "family_friendly": "Dữ liệu tiện ích cho thấy ưu tiên sinh hoạt gia đình.",
    "investor_fit": "Có thông tin hỗ trợ tài chính và sản phẩm đầu tư.",
    "near_key_arteries": "Thông tin vị trí nằm trên trục giao thông chính.",
    "central_commute_heavy": "Nên kiểm tra thêm quãng đường đi vào các điểm trung tâm.",
    "legal_safe": "Có dữ liệu pháp lý để đối chiếu tính an toàn pháp lý.",
    "premium_pricing": "Khung giá thể hiện phân khúc giá trị cao hơn mức phổ thông.",
    "long_term_liveability": "Thông tin tiến độ và tiện ích phù hợp nhu cầu ở lâu dài.",
}


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

    def retrieve_project_grounded(
        self,
        query: str,
        retrieval_intent: str | None = None,
        top_k: int | None = None,
    ) -> dict:
        """
        Build a project-grounded payload with 3 retrieval layers:
        1) project cards
        2) trait tags
        3) evidence chunks
        """
        combined_query = self._build_grounded_query(query=query, retrieval_intent=retrieval_intent)
        raw = self.retrieve(combined_query, top_k=top_k)
        evidence_chunks = self._build_evidence_chunks(raw.get("results", []))
        project_cards = self._build_project_cards(evidence_chunks)
        trait_tags = self._build_trait_tags(evidence_chunks, combined_query)
        return {
            "route": "project_grounded",
            "project_cards": project_cards,
            "trait_tags": trait_tags,
            "evidence_chunks": evidence_chunks,
            "confidence": raw.get("confidence"),
            "low_confidence": bool(raw.get("low_confidence", False)),
        }

    def _build_grounded_query(self, query: str, retrieval_intent: str | None) -> str:
        query = (query or "").strip()
        retrieval_intent = (retrieval_intent or "").strip()
        if not retrieval_intent:
            return query
        return f"{query}\nintent: {retrieval_intent}"

    def _build_evidence_chunks(self, results: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        for item in results:
            source = str(item.get("source", "unknown"))
            topic = self._infer_topic(source=source, text=str(item.get("text", "")))
            project_id = self._infer_project_id(source=source, text=str(item.get("text", "")))
            chunks.append(
                {
                    "text": str(item.get("text", "")),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "source": source,
                    "doc_id": str(item.get("doc_id", "")),
                    "project_id": project_id,
                    "topic": topic,
                }
            )
        return chunks

    def _build_project_cards(self, evidence_chunks: list[dict]) -> list[dict]:
        if not evidence_chunks:
            return []

        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in evidence_chunks:
            grouped[str(chunk.get("project_id") or "unknown_project")].append(chunk)

        cards: list[dict] = []
        for project_id, chunks in grouped.items():
            topic_counter = Counter([c.get("topic") for c in chunks if c.get("topic")])
            top_topics = [topic for topic, _ in topic_counter.most_common(3)]
            strengths = self._dedupe_keep_order([_TOPIC_TO_STRENGTH.get(topic, "") for topic in top_topics])
            tradeoffs = [t for t in self._dedupe_keep_order([_TOPIC_TO_TRADEOFF.get(topic, "") for topic in top_topics]) if t]
            if not tradeoffs:
                tradeoffs = ["cần đối chiếu thêm ngân sách, nhu cầu ở thực và kỳ vọng đầu tư trước khi chốt."]

            fit_personas = self._infer_fit_personas(top_topics)
            primary_chunk = max(chunks, key=lambda c: float(c.get("score", 0.0) or 0.0))
            summary = self._build_project_summary(project_id=project_id, primary_chunk=primary_chunk, top_topics=top_topics)
            score = sum(float(c.get("score", 0.0) or 0.0) for c in chunks) / max(len(chunks), 1)

            cards.append(
                {
                    "project_id": project_id,
                    "summary": summary,
                    "strengths": strengths or ["có dữ liệu nền tảng để phân tích theo nhu cầu cụ thể."],
                    "tradeoffs": tradeoffs,
                    "fit_personas": fit_personas,
                    "score": float(score),
                }
            )

        cards.sort(key=lambda c: c["score"], reverse=True)
        return cards[:3]

    def _build_trait_tags(self, evidence_chunks: list[dict], query: str) -> list[dict]:
        if not evidence_chunks:
            return []

        weights: dict[tuple[str, str | None], float] = defaultdict(float)
        query_lower = query.lower()
        for chunk in evidence_chunks:
            topic = chunk.get("topic")
            project_id = chunk.get("project_id")
            score = float(chunk.get("score", 0.0) or 0.0)
            traits = _TOPIC_TO_TRAITS.get(str(topic), [])
            for tag in traits:
                boost = 0.08 if self._query_mentions_trait(query_lower, tag) else 0.0
                weights[(tag, project_id)] = max(weights[(tag, project_id)], min(1.0, score + boost))

        trait_tags = [
            {
                "tag": tag,
                "weight": float(round(weight, 4)),
                "reason": _TRAIT_REASON.get(tag, "Trait được suy diễn từ dữ liệu bằng chứng hiện có."),
                "project_id": project_id,
            }
            for (tag, project_id), weight in weights.items()
        ]
        trait_tags.sort(key=lambda item: item["weight"], reverse=True)
        return trait_tags[:8]

    def _infer_topic(self, source: str, text: str) -> str:
        source_lower = source.lower()
        text_lower = text.lower()
        if "price" in source_lower or "gia" in text_lower:
            return "price"
        if "legal" in source_lower or "phap ly" in text_lower:
            return "legal"
        if "amenit" in source_lower or "tien ich" in text_lower:
            return "amenities"
        if "bank" in source_lower or "lai suat" in text_lower:
            return "bank"
        if "timeline" in source_lower or "ban giao" in text_lower:
            return "timeline"
        if "location" in source_lower or "ho tay" in text_lower or "vi tri" in text_lower:
            return "location"
        return "general"

    def _infer_project_id(self, source: str, text: str) -> str:
        source_lower = source.lower()
        text_lower = text.lower()
        match = re.search(r"project[_\s-]?id[:=]\s*([a-z0-9_-]+)", text_lower)
        if match:
            return match.group(1)
        if "noble palace" in text_lower or "tay thang long" in text_lower:
            return "noble_palace_tay_thang_long"
        if source_lower:
            return "noble_palace_tay_thang_long"
        return "unknown_project"

    def _build_project_summary(self, project_id: str, primary_chunk: dict, top_topics: list[str]) -> str:
        topic_hint = ", ".join(top_topics) if top_topics else "nhu cầu tổng quan"
        content = str(primary_chunk.get("text", "")).strip()
        if content:
            return (
                f"{project_id}: dữ liệu hiện tại cho thấy dự án có thể phù hợp khi ưu tiên {topic_hint}. "
                f"Bằng chứng nổi bật: {content}"
            )
        return f"{project_id}: có dữ liệu nền để tư vấn theo nhóm nhu cầu {topic_hint}."

    def _infer_fit_personas(self, top_topics: list[str]) -> list[str]:
        personas: list[str] = []
        if "amenities" in top_topics or "location" in top_topics:
            personas.append("gia đình có con nhỏ")
        if "legal" in top_topics or "bank" in top_topics:
            personas.append("nhà đầu tư ưu tiên an toàn")
        if "timeline" in top_topics or "price" in top_topics:
            personas.append("người mua ở lâu dài cần kế hoạch tài chính rõ")
        if not personas:
            personas.append("khách cần shortlist theo tiêu chí cụ thể")
        return self._dedupe_keep_order(personas)[:3]

    def _query_mentions_trait(self, query_lower: str, tag: str) -> bool:
        if tag == "family_friendly":
            return any(token in query_lower for token in ("gia đình", "con nhỏ", "trường học", "bệnh viện"))
        if tag == "investor_fit":
            return any(token in query_lower for token in ("đầu tư", "thanh khoản", "giữ giá"))
        if tag == "legal_safe":
            return "pháp lý" in query_lower or "phap ly" in query_lower
        if tag == "premium_pricing":
            return "giá" in query_lower or "gia" in query_lower
        if tag == "near_key_arteries":
            return any(token in query_lower for token in ("khu", "vị trí", "vi tri", "di chuyển", "di chuyen"))
        return False

    def _dedupe_keep_order(self, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = (value or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
        return out

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
