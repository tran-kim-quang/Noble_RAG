"""Confidence evaluation for KB retrieval and search fallback decisions."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, TypedDict


class KnowledgeDecision(TypedDict, total=False):
    top_score: float
    avg_top_k_score: float
    should_search: bool
    decision_reason: str
    matched_documents: List[Dict[str, Any]]
    low_confidence: bool
    special_case_triggered: bool


def _fold_vn(text: str) -> str:
    raw = (text or "").strip().lower()
    folded = unicodedata.normalize("NFD", raw)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d")


def is_realtime_query(text: str) -> bool:
    folded = _fold_vn(text)
    if not folded:
        return False
    keywords = (
        "weather",
        "thoi tiet",
        "time",
        "gio ",
        "gio o ",
        "current hour",
        "latest",
        "moi nhat",
        "hom nay",
        "bay gio",
        "hien tai",
        "currently",
        "now",
        "news",
        "tin tuc",
        "ty gia",
        "exchange rate",
        "gia vang",
        "gold price",
        "stock price",
        "gia co phieu",
    )
    return any(keyword in folded for keyword in keywords)


def is_project_specific_query(text: str, history: Optional[List[Dict[str, Any]]] = None) -> bool:
    folded = _fold_vn(text)
    history_folded = " ".join(_fold_vn(str(item.get("content") or "")) for item in (history or [])[-6:])
    corpus = f"{folded} {history_folded}".strip()
    if not corpus:
        return False

    if any(keyword in corpus for keyword in ("thi truong", "market", "ben ngoai", "du lieu ngoai")):
        return False

    project_markers = (
        "noble place",
        "tay thang long",
        "du an",
    )
    internal_fact_markers = (
        "phap ly",
        "chinh sach",
        "bang gia",
        "inventory",
        "mat bang",
        "tien do",
        "gio hang",
    )
    project_type_markers = ("can ho", "biet thu", "lien ke")

    has_project_marker = any(keyword in corpus for keyword in project_markers)
    has_internal_fact_marker = any(keyword in corpus for keyword in internal_fact_markers)
    has_project_type_marker = any(keyword in corpus for keyword in project_type_markers)

    if has_project_marker and (has_internal_fact_marker or has_project_type_marker):
        return True
    if "noble place" in corpus or "tay thang long" in corpus:
        return True
    return False


def is_short_ambiguous_query(text: str, history: Optional[List[Dict[str, Any]]] = None) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return False
    if len(normalized) > 40:
        return False

    folded = _fold_vn(normalized)
    if is_project_specific_query(normalized, history):
        return False

    vague_patterns = (
        "gia?",
        "price?",
        "cai nao tot hon",
        "which one is better",
        "du an do",
        "that project",
        "sao roi",
        "the nao",
        "how about",
    )
    token_count = len(re.findall(r"[a-z0-9]+", folded))
    return token_count <= 4 or any(pattern in folded for pattern in vague_patterns)


def evaluate_kb_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.5,
) -> KnowledgeDecision:
    top_score = 0.0
    avg_top_k_score = 0.0
    scores: List[float] = []
    matched_documents: List[Dict[str, Any]] = []

    for item in candidates[:3]:
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
        matched_documents.append(
            {
                "score": score,
                "source": item.get("source"),
                "document_id": item.get("document_id"),
                "metadata": item.get("metadata") or {},
            }
        )

    if scores:
        top_score = scores[0]
        avg_top_k_score = sum(scores) / len(scores)

    special_case_triggered = False
    decision_reason = "kb_confident"
    should_search = False

    if is_realtime_query(query):
        special_case_triggered = True
        should_search = True
        decision_reason = "realtime_query"
    elif is_short_ambiguous_query(query, history):
        decision_reason = "short_ambiguous_query"
        should_search = False
    elif is_project_specific_query(query, history):
        decision_reason = "project_specific_kb_first"
        should_search = False
    elif top_score < threshold:
        should_search = True
        decision_reason = "low_kb_confidence"

    return {
        "top_score": top_score,
        "avg_top_k_score": avg_top_k_score,
        "should_search": should_search,
        "decision_reason": decision_reason,
        "matched_documents": matched_documents,
        "low_confidence": top_score < threshold,
        "special_case_triggered": special_case_triggered,
    }
