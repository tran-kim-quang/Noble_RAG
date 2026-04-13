"""Node 6: Retrieve knowledge from Haystack-backed RAG based on current sales state."""

import json
import logging
import os
import re
import unicodedata
import asyncio
from typing import Any, Dict, List

import asyncpg

from core.config import get_settings
from core.dependencies import llm_model_func
from sales.graph_state import SalesAgentState
from rag.retriever import query_rag
from utils.json_extract import extract_first_json_object
from utils.text import split_into_sentences

log = logging.getLogger("rag-service")
settings = get_settings()
_CATALOG_PROJECTS_CACHE: List[Dict[str, str]] | None = None
_PROJECT_FACTS_CACHE: List[Dict[str, Any]] | None = None

_STATE_QUERY_TEMPLATES: Dict[str, str] = {
    "need_discovery": (
        "Dựa trên kho tri thức dự án Noble, trích ra các dữ kiện và định hướng tư vấn liên quan trực tiếp đến nhu cầu hiện có của khách. "
        "Không chốt dự án cụ thể nếu còn thiếu tiêu chí cốt lõi. "
        "Ưu tiên thông tin thực tế về tiện ích, phong cách sống, nhóm sản phẩm, vị trí, kết nối, hoặc điểm phù hợp với nhu cầu sau: "
        "mục đích {purpose}, loại hình {property_type}, khu vực {location}, gia đình {family_size} người, "
        "có {children_count} con nhỏ. "
        "Nếu khách chỉ mở lời tư vấn chung, chưa nêu tiêu chí: trả lời tối đa 2–3 câu, không liệt kê brochure dài. "
        "Nếu dữ liệu chưa đủ để kết luận, hãy nêu 2-4 dữ kiện/định hướng hữu ích nhất để sales dùng trả lời ngắn gọn rồi hỏi tiếp. "
        "Câu khách: {user_text}"
    ),
    "project_qa": (
        "Trả lời trực tiếp câu hỏi sau về dự án/sản phẩm/chính sách của Noble bằng dữ kiện có trong kho tri thức. "
        "Nếu dữ liệu thiếu thì nêu rõ phần chưa đủ. Câu hỏi: {user_text}"
    ),
    "product_matching": (
        "Tư vấn dự án bất động sản Noble phù hợp với mục đích {purpose}, "
        "loại hình ưu tiên {property_type}, ngân sách tham khảo {budget}, "
        "ưu tiên gần khu vực {location}, gia đình {family_size} người, "
        "có {children_count} con nhỏ. {matching_guidance} "
        "Chỉ nêu phương án phù hợp với các tiêu chí này "
        "và các dữ kiện có trong kho tri thức."
    ),
    "comparison": (
        "So sánh các dự án bất động sản Noble về: giá, vị trí, pháp lý, tiến độ, "
        "chính sách thanh toán, tiềm năng cho thuê. Khách quan tâm: {user_text}"
    ),
    "objection_handling": (
        "Playbook xử lý phản đối bất động sản: {objection_type}. "
        "FAQ và chính sách liên quan. Khách nói: {user_text}"
    ),
    "closing_next_step": (
        "CTA playbook và bước tiếp theo phù hợp cho khách quan tâm: {user_text}. "
        "Shortlist, bảng giá, lịch xem dự án."
    ),
}


def _clean_project_name(text: str) -> str:
    clean = re.sub(r"[*_`#>\"]+", "", text or "")
    clean = re.sub(r"\s+", " ", clean).strip(" .,!?:;-'\"")
    clean = re.sub(
        r"\b(thế nào|the nao|là gì|la gi|ra sao|so sánh|so sanh|giá bao nhiêu|gia bao nhieu)\b.*$",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip(" .,!?:;-'\"")
    return clean


def _project_lookup_key(text: str) -> str:
    clean = _clean_project_name(text)
    clean = unicodedata.normalize("NFD", clean)
    clean = "".join(ch for ch in clean if unicodedata.category(ch) != "Mn")
    clean = clean.lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _project_facts_cache_path() -> str:
    return os.path.join(settings.rag_working_dir, f"project_facts_{settings.rag_workspace}.json")


def _load_project_facts_from_disk() -> List[Dict[str, Any]]:
    path = _project_facts_cache_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception as e:
        log.warning("load project facts cache failed: %s", e)
    return []


def _save_project_facts_to_disk(items: List[Dict[str, Any]]) -> None:
    path = _project_facts_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False)
    except Exception as e:
        log.warning("save project facts cache failed: %s", e)


def _build_retrieval_query(state: Dict[str, Any]) -> str:
    next_state: str = state.get("next_sales_state") or "product_matching"
    template = _STATE_QUERY_TEMPLATES.get(next_state)
    if not template:
        return ""

    lead = state.get("lead_profile") or {}
    query = template.format(
        family_size=lead.get("family_member_count") or "không rõ",
        children_count=lead.get("children_count") or "không rõ",
        purpose=lead.get("purpose") or "không xác định",
        budget=lead.get("budget_text") or "không xác định",
        location=", ".join(lead.get("location_preference") or []) or "linh hoạt",
        property_type=lead.get("property_type") or "không nêu rõ",
        matching_guidance=_build_matching_guidance(lead),
        objection_type=state.get("objection_type") or "chung",
        user_text=state.get("user_text") or "",
    )
    return query


def _build_retrieval_query_lite(state: Dict[str, Any]) -> str:
    user_text = (state.get("user_text") or "").strip()
    next_state = state.get("next_sales_state") or "project_qa"
    project_name = state.get("resolved_project_name")

    if next_state == "project_qa" and project_name:
        return f"Thông tin chính xác và ngắn gọn về {project_name}: {user_text}"
    if next_state == "comparison":
        return f"So sánh ngắn gọn theo câu hỏi: {user_text}"
    if next_state == "objection_handling":
        return f"Xử lý băn khoăn phổ biến bất động sản cho câu: {user_text}"
    if next_state == "closing_next_step":
        return f"Gợi ý bước tiếp theo phù hợp cho khách với yêu cầu: {user_text}"
    return user_text


async def _resolve_project_name_from_user_context(state: Dict[str, Any]) -> str:
    history = state.get("chat_history") or []
    history_text = "\n".join(
        f"{(item.get('role') or 'unknown')}: {(item.get('content') or '').strip()}"
        for item in history[-8:]
        if isinstance(item, dict)
    ) or "(chưa có)"
    user_text = (state.get("user_text") or "").strip()
    prompt = f"""Bạn là bộ xác định tên dự án Noble theo lịch sử hội thoại.
Chỉ trả về JSON hợp lệ duy nhất:
{{
  "project_name": null | "<tên dự án Noble cụ thể đang được nhắc tới>"
}}

Quy tắc:
- Chỉ trả project_name khi khách đang nói tới một dự án cụ thể.
- Nếu khách chỉ hỏi chung như "dự án nào", "noble nào", "cho biết sơ qua về dự án", phải trả null.
- Không tự đoán tên dự án nếu lịch sử chat chưa đủ rõ.
- Giữ nguyên tên dự án đúng như có thể suy ra từ hội thoại.

Lịch sử gần đây:
{history_text}

Tin nhắn hiện tại:
\"\"\"{user_text}\"\"\""""
    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
        )
        payload = extract_first_json_object(str(raw))
        if not payload:
            return ""
        data = json.loads(payload)
    except Exception as e:
        log.warning("resolve project name from context failed: %s", e)
        return ""

    return str(data.get("project_name") or "").strip()


def _build_matching_guidance(lead: Dict[str, Any]) -> str:
    purpose = lead.get("purpose")
    property_type = lead.get("property_type")
    family_size = lead.get("family_member_count")
    children_count = lead.get("children_count")

    if property_type:
        return f"Ưu tiên đúng loại hình khách đã nêu: {property_type}."

    if purpose == "mua_o" and (family_size or children_count):
        return (
            "Khách đang mua để ở cho gia đình nhưng chưa chốt loại hình. "
            "Ưu tiên mô tả phương án theo tiêu chí sống phù hợp cho gia đình, "
            "không ép sang một loại hình cụ thể nếu dữ liệu chưa đủ."
        )

    return "Nếu khách chưa chốt loại hình, ưu tiên phương án phù hợp với nhu cầu thực tế thay vì áp sẵn một loại hình."


def _extract_project_name_from_doc(doc_text: str) -> str:
    text = doc_text or ""
    patterns = [
        r"Tên dự án\s*[\r\n:]+\s*(Noble[^\r\n]+)",
        r"^#\s*(Noble[^\r\n]+)",
        r"^##\s*(Noble[^\r\n]+)",
        r"(Noble(?:\s+[A-Za-zÀ-ỹ0-9]+){1,5})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        name = _clean_project_name(match.group(1))
        if name and "openclaw" not in name.lower():
            return name
    return ""


def _extract_project_summary_from_doc(doc_text: str, project_name: str) -> str:
    sentences = split_into_sentences(doc_text)
    preferred: List[str] = []
    fallback: List[str] = []
    for sentence in sentences:
        clean = " ".join(sentence.split()).strip(" .")
        if len(clean) < 40:
            continue
        lower = clean.lower()
        if any(bad in lower for bad in ("openclaw", "hospital", "nội bộ")):
            continue
        if project_name and project_name.lower() in lower:
            preferred.append(clean)
        else:
            fallback.append(clean)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return ""


def _extract_catalog_projects(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        full_doc_id = str(row.get("full_doc_id") or "").strip()
        if not full_doc_id:
            continue
        entry = grouped.setdefault(
            full_doc_id,
            {"file_path": str(row.get("file_path") or ""), "chunks": []},
        )
        entry["chunks"].append(
            (
                int(row.get("chunk_order_index") or 0),
                str(row.get("content") or ""),
            )
        )

    projects: List[Dict[str, str]] = []
    seen_names = set()
    for full_doc_id, payload in grouped.items():
        ordered_chunks = [content for _, content in sorted(payload["chunks"], key=lambda item: item[0])]
        doc_text = "\n".join(ordered_chunks).strip()
        project_name = _extract_project_name_from_doc(doc_text)
        if not project_name:
            continue
        lower_name = project_name.lower()
        if lower_name in seen_names:
            continue
        summary = _extract_project_summary_from_doc(doc_text, project_name)
        if not summary:
            continue
        seen_names.add(lower_name)
        projects.append(
            {
                "project_name": project_name,
                "summary": summary,
                "source": str(payload.get("file_path") or ""),
                "doc_id": full_doc_id,
            }
        )

    return projects[:5]


async def _load_catalog_projects() -> List[Dict[str, str]]:
    global _CATALOG_PROJECTS_CACHE
    if _CATALOG_PROJECTS_CACHE is not None:
        return list(_CATALOG_PROJECTS_CACHE)

    conn = await asyncpg.connect(settings.postgres_url)
    try:
        rows = await conn.fetch(
            """
            SELECT full_doc_id, file_path, chunk_order_index, content
            FROM sales.haystack_doc_chunks
            WHERE workspace = $1
              AND content ILIKE '%Noble%'
            ORDER BY full_doc_id, chunk_order_index
            LIMIT 80
            """,
            settings.rag_workspace,
        )
    finally:
        await conn.close()
    _CATALOG_PROJECTS_CACHE = _extract_catalog_projects([dict(row) for row in rows])
    return list(_CATALOG_PROJECTS_CACHE)


async def _extract_project_fact_from_doc(doc_text: str, project_name: str) -> Dict[str, Any]:
    prompt = f"""Bạn là bộ trích xuất dữ kiện dự án bất động sản Noble từ tài liệu nội bộ.
Chỉ trả về JSON hợp lệ duy nhất theo schema:
{{
  "project_name": "{project_name}",
  "location": "<khu vực/quận hoặc null>",
  "product_type": "<loại hình chính hoặc null>",
  "lifestyle_fit": ["<phù hợp phong cách sống nào>"],
  "family_fit": ["<phù hợp kiểu gia đình nào>"],
  "child_friendly_features": ["<tiện ích/liên quan trẻ nhỏ>"],
  "key_strengths": ["<điểm mạnh chính 1>", "<điểm mạnh chính 2>", "<điểm mạnh chính 3>"],
  "cautions": ["<điểm cần lưu ý hoặc thiếu dữ liệu>"],
  "source_summary": "<1 câu tóm tắt dự án bằng facts>"
}}

Quy tắc:
- Chỉ dùng thông tin có trong tài liệu.
- Nếu không chắc thì để null hoặc mảng rỗng.
- Không bịa giá, pháp lý, ưu đãi nếu tài liệu không nêu.
- key_strengths phải cụ thể, không chung chung.
- family_fit và child_friendly_features chỉ điền khi tài liệu có căn cứ.

Tài liệu:
\"\"\"{doc_text[:5000]}\"\"\""""
    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
        )
        payload = extract_first_json_object(str(raw))
        if not payload:
            raise ValueError("No JSON in project fact extraction")
        data = json.loads(payload)
    except Exception as e:
        log.warning("extract project facts failed for %s: %s", project_name, e)
        return {
            "project_name": project_name,
            "location": None,
            "product_type": None,
            "lifestyle_fit": [],
            "family_fit": [],
            "child_friendly_features": [],
            "key_strengths": [],
            "cautions": [],
            "source_summary": _extract_project_summary_from_doc(doc_text, project_name),
        }

    data["project_name"] = project_name
    return data


async def _load_project_facts() -> List[Dict[str, Any]]:
    global _PROJECT_FACTS_CACHE
    if _PROJECT_FACTS_CACHE is not None:
        return list(_PROJECT_FACTS_CACHE)
    cached = _load_project_facts_from_disk()
    if cached:
        _PROJECT_FACTS_CACHE = cached
        return list(_PROJECT_FACTS_CACHE)

    conn = await asyncpg.connect(settings.postgres_url)
    try:
        rows = await conn.fetch(
            """
            SELECT full_doc_id, file_path, chunk_order_index, content
            FROM sales.haystack_doc_chunks
            WHERE workspace = $1
              AND content ILIKE '%Noble%'
            ORDER BY full_doc_id, chunk_order_index
            LIMIT 120
            """,
            settings.rag_workspace,
        )
    finally:
        await conn.close()

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        full_doc_id = str(row.get("full_doc_id") or "").strip()
        if not full_doc_id:
            continue
        entry = grouped.setdefault(
            full_doc_id,
            {"file_path": str(row.get("file_path") or ""), "chunks": []},
        )
        entry["chunks"].append((int(row.get("chunk_order_index") or 0), str(row.get("content") or "")))

    facts: List[Dict[str, Any]] = []
    seen = set()
    for full_doc_id, payload in grouped.items():
        doc_text = "\n".join(
            content for _, content in sorted(payload["chunks"], key=lambda item: item[0])
        ).strip()
        project_name = _extract_project_name_from_doc(doc_text)
        if not project_name or project_name.lower() in seen:
            continue
        seen.add(project_name.lower())
        item = await _extract_project_fact_from_doc(doc_text, project_name)
        item["doc_id"] = full_doc_id
        item["source"] = str(payload.get("file_path") or "")
        facts.append(item)
    _PROJECT_FACTS_CACHE = facts[:5]
    _save_project_facts_to_disk(_PROJECT_FACTS_CACHE)
    return list(_PROJECT_FACTS_CACHE)


async def warm_retrieval_caches(
    *,
    include_catalog: bool = True,
    include_project_facts: bool = True,
    start_delay_sec: float = 0.0,
) -> None:
    try:
        delay = max(0.0, float(start_delay_sec))
        if delay > 0:
            await asyncio.sleep(delay)
        if include_catalog:
            await _load_catalog_projects()
        if include_project_facts:
            await _load_project_facts()
        log.info(
            "warm_retrieval_caches: ready (catalog=%s, project_facts=%s)",
            include_catalog,
            include_project_facts,
        )
    except Exception as e:
        log.warning("warm_retrieval_caches failed: %s", e)


async def refresh_retrieval_caches() -> None:
    global _CATALOG_PROJECTS_CACHE, _PROJECT_FACTS_CACHE
    _CATALOG_PROJECTS_CACHE = None
    _PROJECT_FACTS_CACHE = None
    try:
        await _load_catalog_projects()
        await _load_project_facts()
        log.info("refresh_retrieval_caches: rebuilt")
    except Exception as e:
        log.warning("refresh_retrieval_caches failed: %s", e)


def _project_fact_to_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    summary = str(item.get("source_summary") or "").strip()
    strengths = [str(x).strip() for x in item.get("key_strengths") or [] if str(x).strip()]
    return {
        "project_name": str(item.get("project_name") or "").strip(),
        "fit_reasons": strengths[:2],
        "risk_notes": [str(x).strip() for x in item.get("cautions") or [] if str(x).strip()][:1],
        "content": summary,
    }


def _filter_project_facts_for_lead(project_facts: List[Dict[str, Any]], lead: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not project_facts:
        return []

    preferred_locations = [str(item).strip().lower() for item in lead.get("location_preference") or [] if str(item).strip()]
    if not preferred_locations:
        return project_facts[:3]

    exact: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []
    for item in project_facts:
        haystack = " ".join(
            [
                str(item.get("project_name") or ""),
                str(item.get("location") or ""),
                str(item.get("source_summary") or ""),
                " ".join(str(x) for x in item.get("key_strengths") or []),
            ]
        ).lower()
        if any(location in haystack for location in preferred_locations):
            exact.append(item)
        else:
            fallback.append(item)

    ranked = exact + fallback
    return ranked[:3]


def _same_known_project(left: str, right: str) -> bool:
    left_key = _project_lookup_key(left)
    right_key = _project_lookup_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_tokens = left_key.split()
    right_tokens = right_key.split()
    if len(left_tokens) >= 2 and len(right_tokens) >= 2 and left_tokens[:2] == right_tokens[:2]:
        return True
    return False


def _project_overlap_score(query: str, project_name: str) -> int:
    query_key = _project_lookup_key(query)
    project_key = _project_lookup_key(project_name)
    if not query_key or not project_key:
        return 0
    if query_key == project_key:
        return 100
    if project_key in query_key:
        return 95
    query_tokens = query_key.split()
    project_tokens = project_key.split()
    if len(query_tokens) >= 2 and len(project_tokens) >= 2 and query_tokens[:2] == project_tokens[:2]:
        return 80
    overlap = len(set(query_tokens) & set(project_tokens))
    return overlap * 10


def _select_known_projects_from_text(
    text: str,
    project_facts: List[Dict[str, Any]],
    max_items: int = 2,
) -> List[str]:
    text_lower = (text or "").lower()
    names: List[str] = []
    for item in project_facts:
        name = str(item.get("project_name") or "").strip()
        if not name:
            continue
        if _project_overlap_score(text_lower, name) >= 20 and name.lower() not in {x.lower() for x in names}:
            names.append(name)
    return names[:max_items]


def _resolve_known_project_names(
    state: Dict[str, Any],
    project_facts: List[Dict[str, Any]],
    max_items: int = 2,
) -> List[str]:
    candidates = [str(item.get("project_name") or "").strip() for item in project_facts if str(item.get("project_name") or "").strip()]
    if not candidates:
        return []

    evidence_texts = [
        str(state.get("resolved_project_name") or "").strip(),
        str(state.get("user_text") or "").strip(),
    ]
    history = state.get("chat_history") or []
    evidence_texts.extend(
        str(item.get("content") or "").strip()
        for item in history[-4:]
        if isinstance(item, dict) and str(item.get("content") or "").strip()
    )

    scores: Dict[str, int] = {name: 0 for name in candidates}
    for text in evidence_texts:
        if not text:
            continue
        for name in candidates:
            score = _project_overlap_score(text, name)
            if score > scores[name]:
                scores[name] = score

    ranked = [name for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score >= 20]
    deduped: List[str] = []
    for name in ranked:
        if any(_same_known_project(name, existing) for existing in deduped):
            continue
        deduped.append(name)
    return deduped[:max_items]


async def _load_chunks_for_doc_ids(doc_ids: List[str], max_chunks_per_doc: int = 4) -> List[Dict[str, Any]]:
    if not doc_ids:
        return []
    conn = await asyncpg.connect(settings.postgres_url)
    try:
        rows = await conn.fetch(
            """
            SELECT full_doc_id, chunk_order_index, content
            FROM sales.haystack_doc_chunks
            WHERE workspace = $1
              AND full_doc_id = ANY($2::text[])
            ORDER BY full_doc_id, chunk_order_index
            """,
            settings.rag_workspace,
            doc_ids,
        )
    finally:
        await conn.close()

    grouped: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        doc_id = str(row.get("full_doc_id") or "").strip()
        if not doc_id:
            continue
        count = counts.get(doc_id, 0)
        if count >= max_chunks_per_doc:
            continue
        counts[doc_id] = count + 1
        grouped.setdefault(doc_id, []).append(str(row.get("content") or "").strip())

    items: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        text = "\n".join(grouped.get(doc_id) or []).strip()
        if text:
            items.append({"content": text, "source": doc_id})
    return items


async def retrieve_context(state: SalesAgentState) -> Dict[str, Any]:
    next_st = (state.get("next_sales_state") or "").strip()
    action = (state.get("response_action") or "").strip()
    if next_st == "greeting" and action not in {"catalog_overview", "project_qa"}:
        log.info("retrieve_context: skip retrieval for greeting")
        return {
            "retrieved_candidates": [],
            "retrieved_context": [],
            "has_retrieved_context": False,
            "project_qa_blocked": False,
            "resolved_project_name": state.get("resolved_project_name"),
            "retrieval_mode": "greeting_skip",
        }

    if (state.get("response_action") or "") == "catalog_overview":
        projects = await _load_catalog_projects()
        project_facts = await _load_project_facts()
        return {
            "catalog_projects": projects,
            "project_facts": project_facts,
            "retrieved_candidates": [
                {"project_name": item.get("project_name"), "content": item.get("summary")}
                for item in projects
            ],
            "retrieved_context": [],
            "has_retrieved_context": bool(projects or project_facts),
            "project_qa_blocked": False,
            "resolved_project_name": None,
            "retrieval_mode": "catalog",
        }

    if (state.get("response_action") or "") == "project_qa":
        resolved_project_name = await _resolve_project_name_from_user_context(state)
        if not resolved_project_name:
            log.info("retrieve_context: project_qa blocked because project name is unresolved")
            return {
                "retrieved_candidates": [],
                "retrieved_context": [],
                "has_retrieved_context": False,
                "project_qa_blocked": True,
                "resolved_project_name": None,
            }
        state = dict(state)
        state["resolved_project_name"] = resolved_project_name

    if (state.get("next_sales_state") or "") == "product_matching":
        project_facts = await _load_project_facts()
        filtered_facts = _filter_project_facts_for_lead(project_facts, state.get("lead_profile") or {})
        candidates = [_project_fact_to_candidate(item) for item in filtered_facts if item.get("project_name")]
        context_items = [
            {"content": str(item.get("source_summary") or "").strip(), "source": str(item.get("source") or "project_facts")}
            for item in filtered_facts
            if str(item.get("source_summary") or "").strip()
        ]
        if filtered_facts:
            log.info("retrieve_context: mode=project_facts candidates=%d", len(filtered_facts))
            return {
                "retrieved_candidates": candidates,
                "retrieved_context": context_items,
                "project_facts": filtered_facts,
                "has_retrieved_context": bool(context_items or filtered_facts),
                "project_qa_blocked": False,
                "resolved_project_name": state.get("resolved_project_name"),
                "retrieval_mode": "project_facts",
            }

    if (state.get("next_sales_state") or "") in {"project_qa", "comparison"}:
        project_facts = await _load_project_facts()
        target_names = _resolve_known_project_names(
            state,
            project_facts,
            max_items=1 if (state.get("next_sales_state") or "") == "project_qa" else 2,
        )

        matched_facts = [
            item for item in project_facts
            if any(_same_known_project(str(item.get("project_name") or ""), name) for name in target_names)
        ]
        if matched_facts:
            context_items = await _load_chunks_for_doc_ids(
                [str(item.get("doc_id") or "") for item in matched_facts if str(item.get("doc_id") or "").strip()]
            )
            candidates = [_project_fact_to_candidate(item) for item in matched_facts if item.get("project_name")]
            log.info(
                "retrieve_context: mode=project_chunks candidates=%d state=%s",
                len(matched_facts),
                state.get("next_sales_state"),
            )
            return {
                "retrieved_candidates": candidates,
                "retrieved_context": context_items,
                "project_facts": matched_facts,
                "has_retrieved_context": bool(context_items or matched_facts),
                "project_qa_blocked": False,
                "resolved_project_name": state.get("resolved_project_name"),
                "retrieval_mode": "project_chunks",
            }

    retrieval_mode = (state.get("retrieval_mode") or "full").lower()
    if retrieval_mode == "lite":
        query = _build_retrieval_query_lite(state)
        top_k = 3
        history = (state.get("chat_history") or [])[-2:]
    else:
        query = _build_retrieval_query(state)
        top_k = 5
        history = state.get("chat_history") or []

    if not query:
        log.info("retrieve_context: no query needed for state=%s", state.get("next_sales_state"))
        return {"retrieved_context": []}

    log.info("retrieve_context: mode=%s query='%s...'", retrieval_mode, query[:80])
    raw_answer = await query_rag(
        query,
        top_k=top_k,
        history=history,
    )

    context_items: List[Dict[str, Any]] = []
    if raw_answer and raw_answer.strip():
        context_items.append({"content": raw_answer, "source": "haystack_rag"})
    candidates = _extract_candidates(raw_answer)
    project_facts: List[Dict[str, Any]] = []
    if (state.get("next_sales_state") or "") == "product_matching":
        project_facts = await _load_project_facts()

    log.info("retrieve_context: got %d items %d candidates", len(context_items), len(candidates))
    return {
        "retrieved_candidates": candidates,
        "retrieved_context": context_items,
        "project_facts": project_facts,
        "has_retrieved_context": bool(context_items),
        "project_qa_blocked": False,
        "resolved_project_name": state.get("resolved_project_name"),
        "retrieval_mode": retrieval_mode,
    }


def _extract_candidates(raw_answer: str) -> List[Dict[str, Any]]:
    text = (raw_answer or "").strip()
    if not text:
        return []

    candidates = _extract_candidates_from_numbered_blocks(text)
    if candidates:
        return candidates[:2]
    return _extract_candidates_from_project_mentions(text)[:2]


def _extract_candidates_from_numbered_blocks(text: str) -> List[Dict[str, Any]]:
    matches = list(re.finditer(r"(?m)^\s*(\d+)[\.\)]\s+(.+)$", text))
    if not matches:
        return []

    candidates: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        first_line = match.group(2).strip()
        project_name = _clean_project_name(re.split(r"\s*[-:]\s*", first_line, maxsplit=1)[0].strip())
        fit_reasons: List[str] = []
        risk_notes: List[str] = []
        for line in block.splitlines()[1:]:
            clean = line.strip(" -\t")
            if not clean:
                continue
            lower = clean.lower()
            if lower.startswith(("lưu ý", "rủi ro", "hạn chế", "cân nhắc")):
                risk_notes.append(clean)
            elif not lower.startswith("lý do phù hợp"):
                fit_reasons.append(clean.rstrip("."))
        candidates.append(
            {
                "project_name": project_name or f"Phương án {idx + 1}",
                "fit_reasons": fit_reasons[:2],
                "risk_notes": risk_notes[:1],
                "content": block,
            }
        )
    return candidates


def _extract_candidates_from_project_mentions(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"(Noble[^\n,.;:]+)", re.IGNORECASE)
    names: List[str] = []
    for match in pattern.findall(text):
        clean = match.strip()
        clean = _clean_project_name(clean)
        if clean and clean.lower() not in {item.lower() for item in names}:
            names.append(clean)

    sentences = split_into_sentences(text)
    candidates: List[Dict[str, Any]] = []
    for name in names[:2]:
        related = [sentence for sentence in sentences if name.lower() in sentence.lower()]
        candidates.append(
            {
                "project_name": name,
                "fit_reasons": [sentence.rstrip(".") for sentence in related[:2]],
                "risk_notes": [],
                "content": " ".join(related[:3]).strip(),
            }
        )
    return candidates
