from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
import re
import socket
from typing import TYPE_CHECKING
from typing import Any
import urllib.error
import urllib.request
from uuid import uuid4

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions

from retrieval_service.config import Settings
from retrieval_service.schemas import IngestDocument

if TYPE_CHECKING:
    from haystack.components.embedders import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
    )

log = logging.getLogger("retrieval-service")

_TOPIC_TO_STRENGTH = {
    "location": "kết nối giao thông thuận tiện",
    "amenities": "tiện ích nội khu đa dạng cho sinh hoạt hàng ngày",
    "legal": "pháp lý có hồ sơ hiện hữu để đối chiếu",
    "bank": "có hỗ trợ tài chính từ ngân hàng liên kết",
    "timeline": "thông tin tiến độ và bàn giao tương đối rõ",
    "price": "khung giá và sản phẩm được công bố rõ ràng",
    "hospital_access": "tiếp cận bệnh viện và dịch vụ y tế thuận tiện",
    "school_access": "tiếp cận trường học thuận tiện cho gia đình",
    "medical_access": "có bằng chứng gần dịch vụ y tế",
    "education_access": "có bằng chứng gần hạ tầng giáo dục",
    "family_living": "môi trường sống phù hợp gia đình có con nhỏ",
    "green_space": "không gian xanh hỗ trợ chất lượng sống",
    "daily_convenience": "tiện ích sinh hoạt hằng ngày ở cự ly phù hợp",
    "commute": "khả năng di chuyển tới trục chính tương đối thuận tiện",
}

_TOPIC_TO_TRADEOFF = {
    "location": "cần kiểm tra thời gian di chuyển giờ cao điểm theo lịch trình thực tế",
    "amenities": "nên kiểm tra thêm mức phí vận hành đi kèm hệ tiện ích",
    "price": "cần cân đối ngân sách theo tầng, view và tiến độ thanh toán",
    "timeline": "mốc bàn giao luôn có thể thay đổi theo điều kiện thị trường và pháp lý",
    "hospital_access": "nên xác minh cự ly và tuyến đường ở giờ cao điểm",
    "school_access": "nên xác minh lựa chọn trường theo tuyến và cấp học",
    "family_living": "cần đối chiếu thêm nhịp sống khu vực theo khung giờ gia đình",
    "commute": "nên kiểm tra thời gian đi làm theo lịch trình thực tế",
}

_TOPIC_TO_TRAITS = {
    "location": ["near_key_arteries", "central_commute_heavy"],
    "amenities": ["family_friendly"],
    "legal": ["legal_safe"],
    "bank": ["investor_fit"],
    "price": ["premium_pricing"],
    "timeline": ["long_term_liveability"],
    "hospital_access": ["near_hospital", "family_friendly"],
    "school_access": ["near_school", "family_friendly"],
    "medical_access": ["near_hospital"],
    "education_access": ["near_school", "family_friendly"],
    "family_living": ["family_friendly"],
    "green_space": ["family_friendly"],
    "daily_convenience": ["daily_convenience"],
    "commute": ["near_key_arteries"],
}

_TRAIT_REASON = {
    "family_friendly": "Dữ liệu tiện ích cho thấy ưu tiên sinh hoạt gia đình.",
    "investor_fit": "Có thông tin hỗ trợ tài chính và sản phẩm đầu tư.",
    "near_key_arteries": "Thông tin vị trí nằm trên trục giao thông chính.",
    "central_commute_heavy": "Nên kiểm tra thêm quãng đường đi vào các điểm trung tâm.",
    "legal_safe": "Có dữ liệu pháp lý để đối chiếu tính an toàn pháp lý.",
    "premium_pricing": "Khung giá thể hiện phân khúc giá trị cao hơn mức phổ thông.",
    "long_term_liveability": "Thông tin tiến độ và tiện ích phù hợp nhu cầu ở lâu dài.",
    "near_hospital": "Có bằng chứng dự án gần hạ tầng y tế.",
    "near_school": "Có bằng chứng dự án gần hạ tầng giáo dục.",
    "daily_convenience": "Có bằng chứng đáp ứng nhu cầu sinh hoạt hằng ngày.",
}

_POI_KEYWORDS = {
    "hospital": ("benh vien", "bệnh viện", "hospital", "medical"),
    "school": ("truong hoc", "trường học", "school", "giao duc", "giáo dục"),
    "park": ("cong vien", "công viên", "park", "green"),
    "mall": ("mall", "tttm", "trung tam thuong mai", "trung tâm thương mại"),
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

        self.indexing: Pipeline | None = None
        self.retrieval: Pipeline | None = None
        self.writer: DocumentWriter | None = None
        self.embedding_retriever: QdrantEmbeddingRetriever | None = None
        self.doc_embedder: SentenceTransformersDocumentEmbedder | None = None
        self.text_embedder: SentenceTransformersTextEmbedder | None = None

        if settings.embedding_backend == "local":
            try:
                from haystack.components.embedders import (
                    SentenceTransformersDocumentEmbedder,
                    SentenceTransformersTextEmbedder,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "EMBEDDING_BACKEND=local requires sentence-transformers dependencies. "
                    "Build image with EMBEDDING_PROFILE=local or install local embedding deps."
                ) from exc
            self.doc_embedder = SentenceTransformersDocumentEmbedder(model=settings.embedding_model)
            self.text_embedder = SentenceTransformersTextEmbedder(model=settings.embedding_model)
            self.doc_embedder.warm_up()
            self.text_embedder.warm_up()
            self._validate_embedding_dim()

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
        elif settings.embedding_backend == "remote":
            if not settings.embedding_api_url:
                raise ValueError("EMBEDDING_API_URL is required when EMBEDDING_BACKEND=remote")
            self._validate_remote_embedding_dim()
            self.writer = DocumentWriter(
                document_store=self.document_store,
                policy=DuplicatePolicy.OVERWRITE,
            )
            self.embedding_retriever = QdrantEmbeddingRetriever(document_store=self.document_store)
        else:
            raise ValueError(
                f"Unsupported EMBEDDING_BACKEND='{settings.embedding_backend}'. Use 'local' or 'remote'."
            )

        self._validate_qdrant_collection_dim()
        log.info(
            "startup complete collection=%s backend=%s model=%s dim=%s qdrant=%s",
            settings.collection_name,
            settings.embedding_backend,
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
        haystack_docs = []
        for item in documents:
            doc = Document(
                id=item.doc_id or str(uuid4()),
                content=item.text,
                meta={
                    "source": item.source or "unknown",
                    **(item.metadata or {}),
                },
            )
            haystack_docs.append(doc)

        if self.settings.embedding_backend == "local":
            if self.indexing is None:
                raise RuntimeError("indexing pipeline is not initialized")
            out = self.indexing.run({"embedder": {"documents": haystack_docs}})
            written = int(out["writer"]["documents_written"])
        else:
            if self.writer is None:
                raise RuntimeError("document writer is not initialized")
            texts = [doc.content or "" for doc in haystack_docs]
            embeddings = self._embed_remote_batch(texts)
            docs_with_embeddings = [
                Document(id=doc.id, content=doc.content, meta=doc.meta, embedding=embedding)
                for doc, embedding in zip(haystack_docs, embeddings, strict=True)
            ]
            out = self.writer.run(documents=docs_with_embeddings)
            written = int(out.get("documents_written", len(docs_with_embeddings)))
        log.info("ingest documents=%s written=%s", len(documents), written)
        return written

    def retrieve(self, query: str, top_k: int | None = None) -> dict:
        query = (query or "").strip()
        if not query:
            return {"results": [], "confidence": 0.0, "low_confidence": True}
        k = top_k or self.settings.default_top_k
        k = max(1, min(k, self.settings.max_top_k))
        if self.settings.embedding_backend == "local":
            if self.retrieval is None:
                raise RuntimeError("retrieval pipeline is not initialized")
            out = self.retrieval.run({"embedder": {"text": query}, "retriever": {"top_k": k}})
            docs = out.get("retriever", {}).get("documents", []) or []
        else:
            if self.embedding_retriever is None:
                raise RuntimeError("embedding retriever is not initialized")
            query_embedding = self._embed_remote_text(query)
            out = self.embedding_retriever.run(query_embedding=query_embedding, top_k=k)
            docs = out.get("documents", []) or []
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
                "metadata": doc.meta or {},
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
        retrieval_intent: dict[str, Any] | str | None = None,
        top_k: int | None = None,
    ) -> dict:
        """
        Multi-pass grounded retrieval:
        1) candidate project discovery
        2) proximity fact extraction (POI/proximity)
        3) evidence chunk support
        """
        intent = self._normalize_retrieval_intent(retrieval_intent)
        k = top_k or self.settings.default_top_k
        k = max(1, min(k, self.settings.max_top_k))

        candidate_query = self._build_candidate_query(query=query, retrieval_intent=intent)
        candidate_raw = self.retrieve(candidate_query, top_k=max(k, 6))
        candidate_chunks = self._build_evidence_chunks(candidate_raw.get("results", []))
        candidate_project_ids = self._dedupe_keep_order(
            [str(chunk.get("project_id") or "") for chunk in candidate_chunks if chunk.get("project_id")],
            max_items=6,
        )

        proximity_query = self._build_proximity_query(query=query, retrieval_intent=intent)
        proximity_raw = self.retrieve(proximity_query, top_k=max(k * 2, 8))
        proximity_chunks = self._build_evidence_chunks(proximity_raw.get("results", []))
        proximity_facts = self._build_proximity_facts(
            chunks=proximity_chunks,
            retrieval_intent=intent,
            candidate_project_ids=candidate_project_ids,
        )

        evidence_query = self._build_evidence_query(query=query, retrieval_intent=intent)
        evidence_raw = self.retrieve(evidence_query, top_k=max(k, 5))
        evidence_chunks = self._build_evidence_chunks(evidence_raw.get("results", []))
        if not evidence_chunks:
            evidence_chunks = candidate_chunks

        if candidate_project_ids:
            evidence_chunks.sort(
                key=lambda item: (
                    0 if str(item.get("project_id") or "") in set(candidate_project_ids) else 1,
                    -float(item.get("score", 0.0) or 0.0),
                )
            )

        project_cards = self._build_project_cards(
            evidence_chunks=evidence_chunks,
            proximity_facts=proximity_facts,
            retrieval_intent=intent,
        )
        if not project_cards:
            project_cards = self._build_project_cards(
                evidence_chunks=candidate_chunks,
                proximity_facts=proximity_facts,
                retrieval_intent=intent,
            )

        trait_tags = self._build_trait_tags(
            evidence_chunks=evidence_chunks,
            query=evidence_query,
            retrieval_intent=intent,
        )
        confidence = max(
            float(candidate_raw.get("confidence", 0.0) or 0.0),
            float(evidence_raw.get("confidence", 0.0) or 0.0),
            float(proximity_raw.get("confidence", 0.0) or 0.0),
        )
        low_confidence = (
            bool(candidate_raw.get("low_confidence", False))
            and bool(evidence_raw.get("low_confidence", False))
            and not project_cards
        )
        return {
            "route": "project_grounded",
            "project_cards": project_cards,
            "trait_tags": trait_tags,
            "proximity_facts": proximity_facts,
            "evidence_chunks": evidence_chunks[: max(k * 2, 8)],
            "confidence": confidence,
            "low_confidence": low_confidence,
        }

    def _normalize_retrieval_intent(self, retrieval_intent: dict[str, Any] | str | None) -> dict[str, Any]:
        base: dict[str, Any] = {
            "goal": "project_matching",
            "semantic_focus": [],
            "persona_hint": [],
            "poi_types": [],
            "filters": {},
        }
        if retrieval_intent is None:
            return base
        if isinstance(retrieval_intent, str):
            raw = retrieval_intent.strip()
            if not raw:
                return base
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    retrieval_intent = parsed
                else:
                    base["semantic_focus"] = self._dedupe_keep_order([raw], max_items=6)
                    return base
            except json.JSONDecodeError:
                base["semantic_focus"] = self._dedupe_keep_order([raw], max_items=6)
                return base
        if not isinstance(retrieval_intent, dict):
            return base

        base["goal"] = str(retrieval_intent.get("goal") or base["goal"]).strip() or "project_matching"
        base["semantic_focus"] = self._dedupe_keep_order(
            [str(item).strip() for item in retrieval_intent.get("semantic_focus", []) if str(item).strip()],
            max_items=8,
        )
        base["persona_hint"] = self._dedupe_keep_order(
            [str(item).strip() for item in retrieval_intent.get("persona_hint", []) if str(item).strip()],
            max_items=5,
        )
        base["poi_types"] = self._dedupe_keep_order(
            [str(item).strip().lower() for item in retrieval_intent.get("poi_types", []) if str(item).strip()],
            max_items=4,
        )
        filters = retrieval_intent.get("filters", {})
        base["filters"] = filters if isinstance(filters, dict) else {}
        return base

    def _build_grounded_query(self, query: str, retrieval_intent: dict[str, Any] | str | None) -> str:
        query = (query or "").strip()
        if isinstance(retrieval_intent, dict):
            retrieval_intent_str = json.dumps(retrieval_intent, ensure_ascii=False)
        else:
            retrieval_intent_str = str(retrieval_intent or "").strip()
        if not retrieval_intent_str:
            return query
        return f"{query}\nintent: {retrieval_intent_str}"

    def _build_candidate_query(self, query: str, retrieval_intent: dict[str, Any]) -> str:
        focus_terms = self._dedupe_keep_order(
            [query.strip(), retrieval_intent.get("goal", "project_matching")]
            + [str(item) for item in retrieval_intent.get("semantic_focus", [])],
            max_items=10,
        )
        return " | ".join(item for item in focus_terms if item)

    def _build_proximity_query(self, query: str, retrieval_intent: dict[str, Any]) -> str:
        poi_types = retrieval_intent.get("poi_types", []) or []
        if not poi_types:
            poi_types = [
                poi_type
                for poi_type, keywords in _POI_KEYWORDS.items()
                if any(token in query.lower() for token in keywords)
            ]
        tags = [f"near_{poi_type}" for poi_type in poi_types]
        parts = [query.strip(), "proximity", "poi"] + [str(item) for item in poi_types] + tags
        return " | ".join(self._dedupe_keep_order(parts, max_items=10))

    def _build_evidence_query(self, query: str, retrieval_intent: dict[str, Any]) -> str:
        return self._build_grounded_query(query=query, retrieval_intent=retrieval_intent)

    def _build_evidence_chunks(self, results: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        for item in results:
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            source = str(item.get("source", metadata.get("source", "unknown")))
            text = str(item.get("text", ""))
            topic = self._infer_topic(source=source, text=text, metadata=metadata)
            project_id = self._infer_project_id(source=source, text=text, metadata=metadata)
            chunks.append(
                {
                    "text": text,
                    "score": float(item.get("score", 0.0) or 0.0),
                    "source": source,
                    "doc_id": str(item.get("doc_id", "")),
                    "project_id": project_id,
                    "topic": topic,
                }
            )
        return chunks

    def _build_proximity_facts(
        self,
        chunks: list[dict],
        retrieval_intent: dict[str, Any],
        candidate_project_ids: list[str],
    ) -> list[dict]:
        facts: list[dict] = []
        allowed_poi_types = set(str(item) for item in retrieval_intent.get("poi_types", []) if item)
        candidate_set = set(candidate_project_ids)
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            poi_type = self._extract_poi_type(text)
            if allowed_poi_types and poi_type != "unknown" and poi_type not in allowed_poi_types:
                continue
            if poi_type == "unknown":
                continue
            project_id = str(chunk.get("project_id") or "unknown_project")
            if candidate_set and project_id not in candidate_set:
                continue
            facts.append(
                {
                    "project_id": project_id,
                    "poi_type": poi_type,
                    "poi_name": self._extract_poi_name(text, poi_type),
                    "proximity_text": text[:260],
                    "distance_text": self._extract_distance_text(text),
                    "travel_mode": self._infer_travel_mode(text),
                    "evidence_source": str(chunk.get("source", "unknown")),
                    "semantic_tags": self._build_semantic_tags(text=text, poi_type=poi_type),
                }
            )
        deduped: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for fact in facts:
            key = (
                str(fact.get("project_id", "")),
                str(fact.get("poi_type", "")),
                str(fact.get("poi_name", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
            if len(deduped) >= 12:
                break
        return deduped

    def _build_project_cards(
        self,
        evidence_chunks: list[dict],
        proximity_facts: list[dict] | None = None,
        retrieval_intent: dict[str, Any] | None = None,
    ) -> list[dict]:
        if not evidence_chunks:
            return []

        proximity_facts = proximity_facts or []
        retrieval_intent = retrieval_intent or {}
        facts_by_project: dict[str, list[dict]] = defaultdict(list)
        for fact in proximity_facts:
            facts_by_project[str(fact.get("project_id") or "unknown_project")].append(fact)

        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in evidence_chunks:
            grouped[str(chunk.get("project_id") or "unknown_project")].append(chunk)

        cards: list[dict] = []
        for project_id, chunks in grouped.items():
            topic_counter = Counter([c.get("topic") for c in chunks if c.get("topic")])
            top_topics = [topic for topic, _ in topic_counter.most_common(4)]
            strengths = self._dedupe_keep_order([_TOPIC_TO_STRENGTH.get(topic, "") for topic in top_topics])
            tradeoffs = [t for t in self._dedupe_keep_order([_TOPIC_TO_TRADEOFF.get(topic, "") for topic in top_topics]) if t]
            if not tradeoffs:
                tradeoffs = ["cần đối chiếu thêm ngân sách, nhu cầu ở thực và kỳ vọng đầu tư trước khi chốt."]

            project_facts = facts_by_project.get(project_id, [])
            proximity_tags = self._dedupe_keep_order(
                [f"near_{str(fact.get('poi_type', '')).strip()}" for fact in project_facts if fact.get("poi_type")],
                max_items=6,
            )
            key_pois = self._dedupe_keep_order(
                [str(fact.get("poi_name", "")).strip() for fact in project_facts if str(fact.get("poi_name", "")).strip()],
                max_items=6,
            )
            area, product_types = self._infer_area_and_product_types(chunks)
            fit_personas = self._infer_fit_personas(top_topics, retrieval_intent)
            family_fit_score = self._estimate_family_fit_score(top_topics, proximity_tags, retrieval_intent)
            investor_fit_score = self._estimate_investor_fit_score(top_topics, retrieval_intent)
            primary_chunk = max(chunks, key=lambda c: float(c.get("score", 0.0) or 0.0))
            summary = self._build_project_summary(
                project_id=project_id,
                primary_chunk=primary_chunk,
                top_topics=top_topics,
                proximity_tags=proximity_tags,
            )
            score = sum(float(c.get("score", 0.0) or 0.0) for c in chunks) / max(len(chunks), 1)
            score += 0.02 * min(len(project_facts), 3)

            cards.append(
                {
                    "project_id": project_id,
                    "summary": summary,
                    "strengths": strengths or ["có dữ liệu nền tảng để phân tích theo nhu cầu cụ thể."],
                    "tradeoffs": tradeoffs,
                    "area": area,
                    "product_types": product_types,
                    "fit_personas": fit_personas,
                    "family_fit_score": round(family_fit_score, 4),
                    "investor_fit_score": round(investor_fit_score, 4),
                    "proximity_tags": proximity_tags,
                    "key_pois": key_pois,
                    "score": float(round(score, 4)),
                }
            )

        cards.sort(key=lambda c: c["score"], reverse=True)
        return cards[:3]

    def _build_trait_tags(self, evidence_chunks: list[dict], query: str, retrieval_intent: dict[str, Any]) -> list[dict]:
        if not evidence_chunks:
            return []

        weights: dict[tuple[str, str | None], float] = defaultdict(float)
        query_lower = query.lower()
        persona_hint = [str(item).lower() for item in retrieval_intent.get("persona_hint", []) if item]
        for chunk in evidence_chunks:
            topic = chunk.get("topic")
            project_id = chunk.get("project_id")
            score = float(chunk.get("score", 0.0) or 0.0)
            traits = _TOPIC_TO_TRAITS.get(str(topic), [])
            for tag in traits:
                boost = 0.08 if self._query_mentions_trait(query_lower, tag) else 0.0
                if tag == "family_friendly" and any("family" in hint for hint in persona_hint):
                    boost += 0.05
                if tag == "investor_fit" and any("investor" in hint or "safe" in hint for hint in persona_hint):
                    boost += 0.05
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
        return trait_tags[:10]

    def _infer_topic(self, source: str, text: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        topic_from_meta = str(metadata.get("topic", "")).strip().lower()
        if topic_from_meta:
            return topic_from_meta

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
        if any(token in text_lower for token in _POI_KEYWORDS["hospital"]):
            return "hospital_access"
        if any(token in text_lower for token in _POI_KEYWORDS["school"]):
            return "school_access"
        if any(token in text_lower for token in _POI_KEYWORDS["park"]):
            return "green_space"
        if any(token in text_lower for token in _POI_KEYWORDS["mall"]):
            return "daily_convenience"
        if "location" in source_lower or "ho tay" in text_lower or "vi tri" in text_lower:
            return "location"
        if any(token in text_lower for token in ("di chuyen", "thoi gian di", "commute")):
            return "commute"
        return "general"

    def _infer_project_id(self, source: str, text: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        if metadata.get("project_id"):
            return str(metadata["project_id"])

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

    def _build_project_summary(
        self,
        project_id: str,
        primary_chunk: dict,
        top_topics: list[str],
        proximity_tags: list[str],
    ) -> str:
        topic_hint = ", ".join(top_topics) if top_topics else "nhu cầu tổng quan"
        proximity_hint = ", ".join(proximity_tags[:2]) if proximity_tags else "không gian sống tổng hợp"
        content = str(primary_chunk.get("text", "")).strip()
        if content:
            return (
                f"{project_id}: dữ liệu hiện tại cho thấy dự án có thể phù hợp khi ưu tiên {topic_hint} "
                f"va lens {proximity_hint}. Bằng chứng nổi bật: {content}"
            )
        return f"{project_id}: có dữ liệu nền để tư vấn theo nhóm nhu cầu {topic_hint}."

    def _infer_fit_personas(self, top_topics: list[str], retrieval_intent: dict[str, Any]) -> list[str]:
        personas: list[str] = []
        persona_hint = [str(item).lower() for item in retrieval_intent.get("persona_hint", []) if item]
        if "amenities" in top_topics or "location" in top_topics or "school_access" in top_topics:
            personas.append("gia đình có con nhỏ")
        if "legal" in top_topics or "bank" in top_topics:
            personas.append("nhà đầu tư ưu tiên an toàn")
        if "timeline" in top_topics or "price" in top_topics:
            personas.append("người mua ở lâu dài cần kế hoạch tài chính rõ")
        if any("family" in hint for hint in persona_hint):
            personas.append("gia đình ưu tiên ổn định sinh hoạt")
        if any("investor" in hint or "safe" in hint for hint in persona_hint):
            personas.append("nhà đầu tư chú trọng an toàn")
        if not personas:
            personas.append("khách cần shortlist theo tiêu chí cụ thể")
        return self._dedupe_keep_order(personas)[:4]

    def _estimate_family_fit_score(
        self,
        top_topics: list[str],
        proximity_tags: list[str],
        retrieval_intent: dict[str, Any],
    ) -> float:
        score = 0.45
        if any(topic in top_topics for topic in ("amenities", "school_access", "family_living", "green_space")):
            score += 0.25
        if any(tag in proximity_tags for tag in ("near_school", "near_hospital", "near_park")):
            score += 0.15
        if any("family" in str(item).lower() for item in retrieval_intent.get("persona_hint", [])):
            score += 0.1
        return max(0.0, min(1.0, score))

    def _estimate_investor_fit_score(self, top_topics: list[str], retrieval_intent: dict[str, Any]) -> float:
        score = 0.4
        if any(topic in top_topics for topic in ("legal", "bank", "price", "timeline")):
            score += 0.3
        if any("investor" in str(item).lower() or "safe" in str(item).lower() for item in retrieval_intent.get("persona_hint", [])):
            score += 0.15
        return max(0.0, min(1.0, score))

    def _infer_area_and_product_types(self, chunks: list[dict]) -> tuple[str | None, list[str]]:
        joined = " ".join(str(chunk.get("text", "")) for chunk in chunks).lower()
        area = None
        if "tay ho tay" in joined or "tây hồ tây" in joined:
            area = "tay_ho_tay"
        elif "tay thang long" in joined:
            area = "tay_thang_long"

        product_types: list[str] = []
        if "can ho" in joined or "căn hộ" in joined:
            product_types.append("apartment")
        if "shophouse" in joined:
            product_types.append("shophouse")
        if "biet thu" in joined or "biệt thự" in joined:
            product_types.append("villa")
        if not product_types:
            product_types.append("unspecified")
        return area, self._dedupe_keep_order(product_types)

    def _extract_poi_type(self, text: str) -> str:
        lowered = text.lower()
        for poi_type, keywords in _POI_KEYWORDS.items():
            if any(token in lowered for token in keywords):
                return poi_type
        return "unknown"

    def _extract_poi_name(self, text: str, poi_type: str) -> str | None:
        lowered = text.lower()
        keywords = _POI_KEYWORDS.get(poi_type, ())
        for token in keywords:
            idx = lowered.find(token)
            if idx == -1:
                continue
            snippet = text[idx : idx + 70].strip(" .,:;-")
            if snippet:
                return snippet
        return None

    def _extract_distance_text(self, text: str) -> str | None:
        match = re.search(r"(\d+(?:[.,]\d+)?\s*(?:m|km|phut|phút|min))", text.lower())
        if match:
            return match.group(1)
        return None

    def _infer_travel_mode(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("di bo", "đi bộ", "walk")):
            return "walk"
        if any(token in lowered for token in ("o to", "ô tô", "lai xe", "drive")):
            return "drive"
        return "unspecified"

    def _build_semantic_tags(self, text: str, poi_type: str) -> list[str]:
        tags = [f"near_{poi_type}"]
        lowered = text.lower()
        if any(token in lowered for token in ("gia dinh", "gia đình", "con nho", "con nhỏ")):
            tags.append("family_friendly")
        if poi_type in {"hospital", "school", "park"}:
            tags.append("family_living")
        return self._dedupe_keep_order(tags, max_items=5)

    def _query_mentions_trait(self, query_lower: str, tag: str) -> bool:
        if tag == "family_friendly":
            return any(token in query_lower for token in ("gia đình", "con nhỏ", "trường học", "bệnh viện", "gia dinh"))
        if tag == "investor_fit":
            return any(token in query_lower for token in ("đầu tư", "thanh khoản", "giữ giá", "dau tu"))
        if tag == "legal_safe":
            return "pháp lý" in query_lower or "phap ly" in query_lower
        if tag == "premium_pricing":
            return "giá" in query_lower or "gia" in query_lower
        if tag == "near_key_arteries":
            return any(token in query_lower for token in ("khu", "vị trí", "vi tri", "di chuyển", "di chuyen"))
        if tag == "near_hospital":
            return any(token in query_lower for token in ("bệnh viện", "benh vien", "medical"))
        if tag == "near_school":
            return any(token in query_lower for token in ("trường học", "truong hoc", "education"))
        if tag == "daily_convenience":
            return any(token in query_lower for token in ("sinh hoạt", "sinh hoat", "mall", "tiện"))
        return False

    def _dedupe_keep_order(self, values: list[str], max_items: int | None = None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = (value or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
            if max_items is not None and len(out) >= max_items:
                break
        return out

    def _validate_embedding_dim(self) -> None:
        if self.text_embedder is None:
            raise RuntimeError("text embedder is not initialized")
        embedding = self.text_embedder.run("dimension probe")["embedding"]
        model_dim = len(embedding)
        if model_dim != self.settings.embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"EMBEDDING_DIM={self.settings.embedding_dim} but model '{self.settings.embedding_model}' returns {model_dim}. "
                "Please update EMBEDDING_DIM or choose a compatible model."
            )

    def _validate_remote_embedding_dim(self) -> None:
        embedding = self._embed_remote_text("dimension probe")
        model_dim = len(embedding)
        if model_dim != self.settings.embedding_dim:
            raise ValueError(
                "Remote embedding dimension mismatch: "
                f"EMBEDDING_DIM={self.settings.embedding_dim} but endpoint '{self.settings.embedding_api_url}' "
                f"returned dim={model_dim}. Please align endpoint model or EMBEDDING_DIM."
            )

    def _embed_remote_text(self, text: str) -> list[float]:
        embeddings = self._embed_remote_batch([text])
        return embeddings[0]

    def _embed_remote_batch(self, texts: list[str]) -> list[list[float]]:
        fmt = self.settings.embedding_api_format
        if fmt == "ollama":
            return [self._embed_remote_ollama(text) for text in texts]
        if fmt == "openai":
            return self._embed_remote_openai(texts)
        if fmt in {"host_model", "noble_api"}:
            return [self._embed_remote_host_model(text) for text in texts]
        raise ValueError(
            f"Unsupported EMBEDDING_API_FORMAT='{fmt}'. Use 'ollama', 'openai', or 'host_model'."
        )

    def _embed_remote_ollama(self, text: str) -> list[float]:
        payload = {
            "model": self.settings.embedding_model,
            "prompt": text,
        }
        parsed = self._post_embedding_json(payload)
        embedding = parsed.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Remote embedding endpoint returned invalid 'embedding' payload for ollama format.")
        return [float(item) for item in embedding]

    def _embed_remote_openai(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        parsed = self._post_embedding_json(payload)
        data = parsed.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("Remote embedding endpoint returned invalid 'data' payload for openai format.")

        sorted_items = sorted(data, key=lambda item: int(item.get("index", 0)))
        embeddings: list[list[float]] = []
        for item in sorted_items:
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("Remote embedding endpoint returned invalid embedding item in 'data'.")
            embeddings.append([float(value) for value in embedding])
        return embeddings

    def _embed_remote_host_model(self, text: str) -> list[float]:
        payload = {
            "text": text,
            "model": self.settings.embedding_model,
        }
        parsed = self._post_embedding_json(payload)
        embedding = parsed.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Remote embedding endpoint returned invalid 'embedding' payload for host_model format.")
        return [float(item) for item in embedding]

    def _post_embedding_json(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.settings.embedding_api_key:
            key_header = self.settings.embedding_api_key_header or "X-API-Key"
            headers[key_header] = self.settings.embedding_api_key
        if self.settings.embedding_api_auth_token:
            headers["Authorization"] = f"Bearer {self.settings.embedding_api_auth_token}"

        req = urllib.request.Request(
            self.settings.embedding_api_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.settings.embedding_api_timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"embedding endpoint HTTP {exc.code}: {raw[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"embedding endpoint unreachable: {exc.reason}") from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"embedding endpoint timeout after {self.settings.embedding_api_timeout_sec}s"
            ) from exc

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
