"""Helpers for the Vision -> Session Context -> Sales contract."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from memory.session_store import ensure_session_context_shape
from memory.session_store import load_session_context, save_session_context


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False


def normalize_customer_profile(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(profile or {})

    sales_stage = str(raw.get("sales_stage") or raw.get("current_state") or "new")
    normalized: Dict[str, Any] = {
        "sales_stage": sales_stage,
        "family_size": raw.get("family_size", raw.get("family_member_count")),
        "children_count": raw.get("children_count"),
        "purpose": raw.get("purpose"),
        "location_preference": raw.get("location_preference"),
        "budget_min": raw.get("budget_min"),
        "budget_max": raw.get("budget_max"),
        "budget_text": raw.get("budget_text"),
        "project_interest": raw.get("project_interest"),
        "interest_summary": raw.get("interest_summary"),
    }

    return {k: v for k, v in normalized.items() if not _is_empty(v)}


def _merge_dict_preserve_non_empty(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if _is_empty(value):
            continue
        merged[key] = value
    return merged


def is_session_context_ready(session_context: Optional[Dict[str, Any]], customer_id: Optional[str] = None) -> bool:
    if not isinstance(session_context, dict):
        return False
    if not bool(session_context.get("context_ready")):
        return False
    if customer_id:
        stored_customer = str(session_context.get("customer_id") or "").strip()
        if stored_customer and stored_customer != customer_id.strip():
            return False
    context_json = session_context.get("context_json")
    return isinstance(context_json, dict) and "conversation_context" in context_json


def merge_runtime_conversation_context(
    session_context: Dict[str, Any],
    *,
    last_intent: Optional[str],
    missing_slots: Optional[list[str]],
    last_question: Optional[str],
) -> Dict[str, Any]:
    payload = ensure_session_context_shape(str(session_context.get("session_id") or "default"), session_context)
    context_json = dict(payload.get("context_json") or {})
    conversation = dict(context_json.get("conversation_context") or {})
    conversation["last_intent"] = last_intent
    conversation["missing_slots"] = list(missing_slots or [])
    conversation["last_question"] = last_question
    context_json["conversation_context"] = conversation
    payload["context_json"] = context_json
    return payload


async def build_and_save_session_context(
    *,
    session_id: str,
    customer_id: str,
    customer_profile: Optional[Dict[str, Any]],
    vision_summary: Optional[Dict[str, Any]],
    existing_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current = existing_context if isinstance(existing_context, dict) else await load_session_context(session_id)
    payload = ensure_session_context_shape(session_id, current)
    payload["session_id"] = session_id
    payload["customer_id"] = customer_id
    payload["context_ready"] = True
    payload["context_ready_at"] = time.time()

    context_json = dict(payload.get("context_json") or {})
    existing_profile = dict(context_json.get("customer_profile") or {})
    normalized_profile = normalize_customer_profile(customer_profile)
    context_json["customer_profile"] = _merge_dict_preserve_non_empty(existing_profile, normalized_profile)

    if isinstance(vision_summary, dict):
        context_json["vision_context"] = vision_summary

    conversation = dict(context_json.get("conversation_context") or {})
    conversation.setdefault("last_intent", None)
    conversation.setdefault("missing_slots", [])
    conversation.setdefault("last_question", None)
    context_json["conversation_context"] = conversation

    payload["context_json"] = context_json
    await save_session_context(session_id, payload)
    return payload
