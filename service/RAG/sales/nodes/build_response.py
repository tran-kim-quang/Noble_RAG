"""Node 7: Build and generate the response using LLM."""

import json
import logging
import os
import re
from typing import Any, Dict, List

from sales.graph_state import SalesAgentState
from sales.prompt_builder import build_prompt
from sales.response_templates import render_match_options_from_candidates
from core.dependencies import llm_model_func
from utils.json_extract import extract_first_json_object
from utils.text import normalize_whitespace, split_into_sentences

log = logging.getLogger("rag-service")

_GROUNDED_STATES = {"project_qa", "product_matching", "comparison", "objection_handling", "closing_next_step"}
_DISCOVERY_STATES = {"greeting", "need_discovery"}


def _has_retrieved_context(state: SalesAgentState) -> bool:
    if state.get("project_facts"):
        return True
    if state.get("catalog_projects"):
        return True
    context = state.get("retrieved_context") or []
    return any(
        (item.get("content") or item.get("text") or "").strip()
        for item in context
        if isinstance(item, dict)
    )


def _question_count(text: str) -> int:
    return text.count("?")


def _enforce_short_form(text: str, max_sentences: int) -> str:
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


def _discovery_max_sentences(state: SalesAgentState) -> int:
    return 3 if _has_retrieved_context(state) else 2


def _llm_generation_available() -> bool:
    return bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("LLM_API_KEY") or "").strip()
    )


def _sanitize_project_qa_text(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    clean = re.sub(r"(?<=\d)\.\s+(?=\d{3}\b)", ".", clean)
    clean = clean.replace("**", "")
    clean = clean.replace("__", "")
    clean = re.sub(r"\*\s+", "", clean)
    clean = re.sub(r"\s+\)", ")", clean)
    clean = re.sub(r"\(\s+", "(", clean)
    return normalize_whitespace(clean)


def _same_project_name(left: str, right: str) -> bool:
    left_norm = normalize_whitespace(left).lower()
    right_norm = normalize_whitespace(right).lower()
    return bool(left_norm and right_norm and left_norm == right_norm)


def _same_project_family(left: str, right: str) -> bool:
    left_tokens = normalize_whitespace(left).lower().split()
    right_tokens = normalize_whitespace(right).lower().split()
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    return left_tokens[:2] == right_tokens[:2]


def _related_projects_text(projects: List[str]) -> str:
    if not projects:
        return ""
    if len(projects) == 1:
        return projects[0]
    return ", ".join(projects[:-1]) + f" và {projects[-1]}"


def _clean_display_text(text: str) -> str:
    clean = (text or "")
    clean = normalize_whitespace(clean)
    clean = clean.strip(" .,!?:;-'\"")
    return clean


async def _resolve_requested_projects_from_context(
    state: SalesAgentState,
    candidate_names: List[str],
    max_items: int = 2,
) -> List[str]:
    history = state.get("chat_history") or []
    history_text = "\n".join(
        f"{(item.get('role') or 'unknown')}: {(item.get('content') or '').strip()}"
        for item in history[-8:]
        if isinstance(item, dict)
    ) or "(chưa có)"
    prompt = f"""Bạn là bộ xác định khách đang nói tới dự án Noble nào dựa trên lịch sử chat.
Chỉ trả về JSON hợp lệ duy nhất:
{{
  "projects": []
}}

Quy tắc:
- Chỉ chọn tên dự án thật sự được khách nhắc tới hoặc ngữ cảnh đã xác nhận rõ.
- Không tự bịa tên dự án mới.
- Nếu khách hỏi chung chung hoặc chưa rõ dự án cụ thể, trả mảng rỗng.
- Nếu có danh sách candidate, chỉ chọn trong danh sách đó khi phù hợp.
- Tối đa {max_items} phần tử.

Candidate names:
{candidate_names}

Lịch sử gần đây:
{history_text}

Tin nhắn hiện tại:
\"\"\"{state.get("user_text") or ""}\"\"\""""
    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
        )
        payload = extract_first_json_object(str(raw))
        if not payload:
            return []
        data = json.loads(payload)
    except Exception as e:
        log.warning("resolve requested projects failed: %s", e)
        return []

    projects = []
    for item in data.get("projects") or []:
        text = _clean_display_text(str(item))
        if text and text.lower() not in {name.lower() for name in projects}:
            projects.append(text)
    return projects[:max_items]


async def _project_qa_response_from_context(state: SalesAgentState) -> str:
    candidate_names = [
        _clean_display_text(str(candidate.get("project_name") or ""))
        for candidate in state.get("retrieved_candidates") or []
        if str(candidate.get("project_name") or "").strip()
    ]
    requested_projects = await _resolve_requested_projects_from_context(state, candidate_names, max_items=1)
    requested = requested_projects[0] if requested_projects else (state.get("resolved_project_name") or "").strip()

    if requested:
        exact_matches = [name for name in candidate_names if _same_project_name(name, requested)]
        if exact_matches:
            context_items = state.get("retrieved_context") or []
            raw_text = "\n".join(
                str(item.get("content") or item.get("text") or "").strip()
                for item in context_items
                if isinstance(item, dict)
            ).strip()
            cleaned = _sanitize_project_qa_text(raw_text)
            if cleaned:
                return _enforce_short_form(cleaned, max_sentences=4)

        related = [name for name in candidate_names if _same_project_family(name, requested)]
        if related:
            return (
                f"Em kiểm tra trong kho thông tin hiện tại thì chưa thấy dữ liệu khớp chính xác cho {requested}. "
                f"Hiện hệ thống mới có dữ liệu gần nhất về {_related_projects_text(related)}. "
                "Nếu Anh/Chị muốn, em có thể tư vấn theo dự án này hoặc Anh/Chị xác nhận lại đúng tên dự án để em kiểm tra sát hơn ạ."
            )

        return (
            f"Hiện trong kho thông tin em chưa thấy dữ liệu khớp chính xác cho {requested}, nên em chưa dám khẳng định chi tiết để tránh tư vấn sai cho Anh/Chị. "
            "Nếu Anh/Chị muốn, em sẽ kiểm tra lại đúng tên dự án hoặc hỗ trợ đối chiếu với dự án Noble gần nhất trong hệ thống ạ."
        )

    return _fallback_without_context(state)


async def _comparison_response_from_context(state: SalesAgentState) -> str:
    candidate_names = [
        _clean_display_text(str(candidate.get("project_name") or ""))
        for candidate in state.get("retrieved_candidates") or []
        if str(candidate.get("project_name") or "").strip()
    ]
    requested_projects = await _resolve_requested_projects_from_context(state, candidate_names, max_items=2)

    if requested_projects:
        available: List[str] = []
        missing: List[str] = []
        for requested in requested_projects:
            if any(_same_project_name(candidate, requested) or _same_project_family(candidate, requested) for candidate in candidate_names):
                available.append(requested)
            else:
                missing.append(requested)

        if missing and available:
            return (
                f"Hiện em mới đối chiếu được dữ liệu gần với {_related_projects_text(available)} trong hệ thống, "
                f"còn chưa thấy dữ liệu đủ rõ cho {_related_projects_text(missing)} nên chưa thể so sánh công bằng cho Anh/Chị. "
                "Anh/Chị xác nhận lại tên dự án còn thiếu giúp em, hoặc em sẽ so sánh ngay trên những dự án đang có dữ liệu ạ."
            )
        if missing and not available:
            return (
                f"Hiện em chưa thấy dữ liệu đủ rõ trong hệ thống cho {_related_projects_text(missing)}, nên nếu so sánh lúc này sẽ dễ sai cho Anh/Chị. "
                "Anh/Chị xác nhận lại đúng tên dự án giúp em để em kiểm tra chính xác hơn ạ."
            )

    if len(candidate_names) >= 2:
        top_projects = _related_projects_text(candidate_names[:2])
        return (
            f"Hiện trong kho thông tin em có dữ liệu liên quan tới {top_projects}, nhưng chưa đủ cấu trúc rõ theo cùng một bộ tiêu chí để đưa ra bảng so sánh gọn và công bằng ngay cho Anh/Chị. "
            "Nếu Anh/Chị muốn, em sẽ giúp chốt lại đúng 2 dự án cần so sánh và đối chiếu lần lượt theo các tiêu chí như vị trí, pháp lý, tiến độ và mức giá ạ."
        )

    return _fallback_without_context(state)


def _fallback_without_context(state: SalesAgentState) -> str:
    missing_slots: List[str] = state.get("missing_slots") or []
    next_state = state.get("next_sales_state") or "need_discovery"
    project_name = (state.get("resolved_project_name") or "").strip()
    if next_state == "project_qa":
        if project_name:
            return (
                f"Em đã hiểu Anh/Chị đang hỏi về {project_name}, nhưng hiện em chưa truy xuất được dữ liệu dự án để trả lời chính xác ngay. "
                "Anh/Chị chờ em kiểm tra lại kho thông tin rồi em phản hồi đúng trọng tâm giúp mình nhé."
            )
        return (
            "Em cần đúng tên dự án Noble mà Anh/Chị đang quan tâm để kiểm tra thông tin chính xác hơn. "
            "Anh/Chị nhắn lại giúp em tên dự án nhé."
        )
    if next_state == "closing_next_step":
        return (
            "Em chưa có đủ dữ liệu từ kho dự án để chốt bước tiếp theo thật chính xác. "
            "Nếu Anh/Chị muốn, em sẽ kiểm tra lại thông tin phù hợp rồi quay lại ngay."
        )
    if next_state == "objection_handling":
        return (
            "Em chưa có đủ dữ liệu từ kho dự án để phản hồi chắc chắn cho băn khoăn này. "
            "Anh/Chị cho em kiểm tra lại thông tin chính xác rồi em phản hồi ngay nhé."
        )
    if next_state == "comparison":
        return (
            "Em đã hiểu mình cần so sánh các phương án này, nhưng hiện em chưa truy xuất được dữ liệu dự án để đối chiếu chính xác. "
            "Anh/Chị cho em kiểm tra lại thông tin rồi em so sánh ngắn gọn theo đúng tiêu chí mình quan tâm nhé."
        )
    if missing_slots:
        return (
            "Em chưa có đủ dữ liệu từ kho dự án để đề xuất chính xác. "
            f"Anh/Chị chia sẻ thêm giúp em các mục còn thiếu: {', '.join(missing_slots)} nhé?"
        )
    if next_state == "product_matching":
        return (
            "Em đã nắm được nhu cầu cơ bản của Anh/Chị, nhưng hiện em chưa truy xuất được dữ liệu dự án để lọc shortlist thật sát. "
            "Anh/Chị chờ em kiểm tra lại kho thông tin rồi em gửi lại phương án phù hợp nhất nhé."
        )
    return (
        "Em chưa có đủ dữ liệu từ kho dự án để đề xuất chính xác lúc này. "
        "Anh/Chị cho em kiểm tra thêm rồi em gửi lại phương án phù hợp nhất nhé."
    )


def _catalog_overview_response(state: SalesAgentState) -> str:
    projects = [item for item in state.get("catalog_projects") or [] if isinstance(item, dict)]
    if not projects:
        return (
            "Hiện em chưa đọc được danh sách dự án Noble từ kho thông tin nội bộ để trả lời chắc chắn cho Anh/Chị. "
            "Anh/Chị cho em kiểm tra lại dữ liệu rồi em gửi ngay danh sách ngắn gọn nhé."
        )

    top_projects = projects[:3]
    lines = ["Dạ, hiện tại em đang thấy một vài hướng dự án nổi bật của Noble để Anh/Chị tham khảo nhanh ạ:"]
    for idx, item in enumerate(top_projects, start=1):
        name = str(item.get("project_name") or f"Dự án {idx}").strip()
        summary = normalize_whitespace(str(item.get("summary") or "").strip())
        summary = re.sub(r"^[*\-•#\s]+", "", summary).rstrip(".")
        if summary:
            lines.append(f"{idx}. {name}: {summary}.")
        else:
            lines.append(f"{idx}. {name}.")
    lines.append(
        "Nếu Anh/Chị đang nghiêng về nhu cầu ở thực hay đầu tư, em sẽ lọc ngay giúp mình phương án đáng xem nhất thay vì phải tự so từng dự án ạ."
    )
    return "\n".join(lines)


def _dedupe_candidate_names(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = _clean_display_text(str(candidate.get("project_name") or "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        clone = dict(candidate)
        clone["project_name"] = name
        deduped.append(clone)
    return deduped


def _default_sales_follow_up(lead: Dict[str, Any]) -> str:
    if lead.get("budget_text") or lead.get("budget_min") or lead.get("budget_max"):
        return "Nếu mình đi tiếp, Anh/Chị muốn em đi sâu trước về loại căn phù hợp với nhu cầu sử dụng hay mức tài chính đang dễ chốt hơn ạ?"
    if lead.get("location_preference"):
        return "Nếu mình đi tiếp, Anh/Chị muốn em bóc tách sâu hơn về điểm mạnh sống thực của dự án này hay loại căn phù hợp nhất với nhu cầu của gia đình mình ạ?"
    return "Nếu mình đi tiếp, Anh/Chị muốn em đi sâu trước về điểm mạnh nổi bật nhất của dự án hay loại căn phù hợp nhất với nhu cầu của mình ạ?"


def _pre_score_project_candidate(lead: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    matched_needs: List[str] = []
    location_pref = [str(x).strip().lower() for x in lead.get("location_preference") or [] if str(x).strip()]
    haystack = " ".join(
        [
            str(item.get("project_name") or ""),
            str(item.get("location") or ""),
            str(item.get("product_type") or ""),
            str(item.get("source_summary") or ""),
            " ".join(str(x) for x in item.get("key_strengths") or []),
            " ".join(str(x) for x in item.get("family_fit") or []),
            " ".join(str(x) for x in item.get("child_friendly_features") or []),
        ]
    ).lower()

    if location_pref and any(pref in haystack for pref in location_pref):
        score += 40
        matched_needs.append("khớp khu vực ưu tiên")

    purpose = str(lead.get("purpose") or "").lower()
    if purpose in {"mua_o", "de_o", "ở", "owner_occupancy", "mua để ở"}:
        if any(token in haystack for token in ("căn hộ", "can ho", "apartment", "ở thực", "ở lâu dài")):
            score += 20
            matched_needs.append("phù hợp nhu cầu ở thực")

    children_count = int(lead.get("children_count") or 0)
    if children_count > 0 and item.get("child_friendly_features"):
        score += 20
        matched_needs.append("có tiện ích liên quan trẻ nhỏ")

    family_size = int(lead.get("family_member_count") or 0)
    if family_size >= 3 and item.get("family_fit"):
        score += 10
        matched_needs.append("có dấu hiệu phù hợp hộ gia đình")

    if item.get("key_strengths"):
        score += 5

    clone = dict(item)
    clone["pre_score"] = score
    clone["matched_needs"] = matched_needs
    return clone


def _adaptive_recommendation_count(scored_items: List[Dict[str, Any]]) -> int:
    if not scored_items:
        return 0
    if len(scored_items) == 1:
        return 1

    scores = [int(item.get("fit_score") or 0) for item in scored_items]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0
    top3 = scores[2] if len(scores) > 2 else 0

    if top1 - top2 >= 15:
        return 1
    if len(scores) == 2 or top2 - top3 >= 10:
        return 2
    return min(3, len(scores))


def _render_ranked_product_matching(
    lead: Dict[str, Any],
    scored_items: List[Dict[str, Any]],
    next_question: str,
    why_top_choice: str,
) -> str:
    top_k = _adaptive_recommendation_count(scored_items)
    if top_k <= 0:
        return _default_sales_follow_up(lead)

    selected = scored_items[:top_k]
    top_choice = selected[0]
    lines: List[str] = []

    if top_k == 1:
        lines.append(
            f"Với nhu cầu hiện tại, phương án em thấy nổi bật nhất là {top_choice['project_name']}."
        )
        if why_top_choice:
            lines.append(why_top_choice)
        elif top_choice.get("fit_summary"):
            lines.append(str(top_choice["fit_summary"]))
    else:
        lines.append(
            f"Hiện tại em thấy có {top_k} phương án đáng cân nhắc, nhưng nổi bật nhất vẫn là {top_choice['project_name']}."
        )
        if why_top_choice:
            lines.append(why_top_choice)

    top_strengths = top_choice.get("strengths") or []
    if top_strengths:
        lines.append("Điểm mạnh nổi bật nhất của phương án này là:")
        for item in top_strengths[:3]:
            lines.append(f"- {item}")

    top_benefits = top_choice.get("benefits_for_customer") or []
    if top_benefits:
        lines.append("Nếu đặt vào đúng nhu cầu hiện tại của gia đình mình, lợi ích rõ nhất là:")
        for item in top_benefits[:2]:
            lines.append(f"- {item}")

    if top_k > 1:
        lines.append("Ngoài phương án nổi bật nhất, em cũng giữ lại thêm các lựa chọn thay thế để mình dễ so hơn:")
        for item in selected[1:]:
            bullet = f"- {item['project_name']}: {item.get('fit_summary') or 'Phù hợp ở một số tiêu chí gần với nhu cầu hiện tại.'}"
            tradeoff = str(item.get("tradeoff") or "").strip()
            if tradeoff:
                bullet += f" Lưu ý: {tradeoff}"
            lines.append(bullet)

    lines.append(next_question or _default_sales_follow_up(lead))
    return "\n".join(lines).strip()


async def _product_matching_response_with_reasoning(state: SalesAgentState) -> str:
    lead = state.get("lead_profile") or {}
    project_facts = [item for item in state.get("project_facts") or [] if isinstance(item, dict)]
    candidate_blocks = []
    prescored_facts = [_pre_score_project_candidate(lead, item) for item in project_facts[:5]]
    prescored_facts.sort(key=lambda item: int(item.get("pre_score") or 0), reverse=True)
    for item in prescored_facts[:3]:
        strengths = [
            normalize_whitespace(str(x)).strip().rstrip(".")
            for x in item.get("key_strengths") or []
            if str(x).strip()
        ][:3]
        if not strengths and str(item.get("source_summary") or "").strip():
            strengths = [normalize_whitespace(str(item.get("source_summary") or "").strip()).rstrip(".")]
        candidate_blocks.append(
            {
                "project_name": _clean_display_text(str(item.get("project_name") or "")),
                "location": item.get("location"),
                "product_type": item.get("product_type"),
                "family_fit": item.get("family_fit") or [],
                "child_friendly_features": item.get("child_friendly_features") or [],
                "lifestyle_fit": item.get("lifestyle_fit") or [],
                "key_strengths": strengths,
                "cautions": item.get("cautions") or [],
                "source_summary": item.get("source_summary"),
                "pre_score": item.get("pre_score") or 0,
                "matched_needs": item.get("matched_needs") or [],
            }
        )

    if not candidate_blocks:
        candidates = _dedupe_candidate_names(state.get("retrieved_candidates") or [])
        if not candidates:
            return render_match_options_from_candidates(state, state.get("retrieved_candidates") or [])
        for candidate in candidates[:3]:
            facts = [str(item).strip() for item in candidate.get("fit_reasons") or [] if str(item).strip()]
            if not facts:
                facts = split_into_sentences(str(candidate.get("content") or ""))[:3]
            facts = [normalize_whitespace(str(item)).strip().rstrip(".") for item in facts if str(item).strip()]
            if not facts:
                continue
            candidate_blocks.append(
                {
                    "project_name": candidate.get("project_name"),
                    "key_strengths": facts[:3],
                    "cautions": [normalize_whitespace(str(item)).strip() for item in candidate.get("risk_notes") or [] if str(item).strip()][:1],
                    "source_summary": "",
                }
            )
        if not candidate_blocks:
            return render_match_options_from_candidates(state, state.get("retrieved_candidates") or [])

    reasoning_prompt = f"""Bạn là chuyên gia sales bất động sản của Noble.
Nhiệm vụ: dựa trên hồ sơ khách và dữ kiện dự án đã retrieve, chốt rất nhanh phương án phù hợp nhất rồi diễn giải bằng lợi ích gắn trực tiếp với nhu cầu khách.

Chỉ trả về JSON hợp lệ duy nhất theo schema:
{{
  "recommendations": [
    {{
      "project_name": "<tên dự án>",
      "fit_score": 0,
      "fit_summary": "<1 câu tóm tắt vì sao dự án này hợp với khách>",
      "strengths": ["<điểm mạnh 1>", "<điểm mạnh 2>", "<điểm mạnh 3>"],
      "benefits_for_customer": ["<lợi ích cụ thể cho khách 1>", "<lợi ích cụ thể cho khách 2>"],
      "tradeoff": "<điểm cần lưu ý hoặc thiếu dữ liệu>"
    }}
  ],
  "why_top_choice": "<1-2 câu giải thích vì sao phương án đứng đầu nổi bật hơn các phương án còn lại>",
  "next_question": "<1 câu hỏi sales tiếp theo để kéo khách đi sâu hơn>"
}}

Quy tắc:
- Chỉ dùng facts đã cho, không bịa thông tin mới.
- Xếp hạng tất cả phương án có liên quan, nhưng chỉ trả tối đa 5 recommendations.
- Nếu chỉ có 1 dự án thực sự phù hợp thì chỉ trả 1 recommendation, không tạo 2 phương án giả.
- strengths phải là điểm mạnh riêng của dự án.
- benefits_for_customer phải gắn trực tiếp với hồ sơ khách theo logic: nhu cầu hoặc ưu tiên nào của khách -> điểm mạnh nào của dự án -> lợi ích thực tế khách nhận được.
- Không được chỉ liệt kê facts đẹp của dự án rồi đổi cách diễn đạt.
- Nếu hồ sơ khách không nhắc tới con nhỏ, người lớn tuổi, đầu tư hoặc ngân sách thì không được tự gượng ép lợi ích theo các chiều đó.
- fit_summary và why_top_choice phải cho thấy vì sao phương án đứng đầu hợp hơn các phương án còn lại trong đúng bối cảnh khách hiện tại.
- tradeoff chỉ nêu điểm cần lưu ý thật sự, không viết cho có.
- next_question phải giúp tiến gần chốt hơn, ví dụ đi sâu về loại căn, ngân sách, ưu tiên ở thực, nhu cầu cho con nhỏ.
- Ưu tiên bám vào pre_score và matched_needs đã cho để kết luận nhanh, không suy diễn vòng vo lại từ đầu.
- Viết ngắn, quyết đoán, không dùng văn brochure.
- Giữ giọng sales tự nhiên, nhưng output vẫn phải là JSON.

Hồ sơ khách:
{lead}

Dữ kiện dự án đã retrieve:
{candidate_blocks}
"""

    try:
        raw = await llm_model_func(
            reasoning_prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
            max_tokens=700,
        )
        payload = extract_first_json_object(str(raw))
        if not payload:
            raise ValueError("No JSON in product matching reasoning response")
        data = json.loads(payload)
    except Exception as e:
        log.warning("product_matching reasoning failed: %s", e)
        return render_match_options_from_candidates(state, state.get("retrieved_candidates") or [])

    recommendations: List[Dict[str, Any]] = []
    for raw_item in data.get("recommendations") or []:
        project_name = _clean_display_text(str(raw_item.get("project_name") or "")).strip()
        if not project_name:
            continue
        strengths = [
            normalize_whitespace(str(item)).strip().rstrip(".")
            for item in raw_item.get("strengths") or []
            if str(item).strip()
        ][:3]
        benefits = [
            normalize_whitespace(str(item)).strip().rstrip(".")
            for item in raw_item.get("benefits_for_customer") or []
            if str(item).strip()
        ][:2]
        recommendations.append(
            {
                "project_name": project_name,
                "fit_score": int(raw_item.get("fit_score") or 0),
                "fit_summary": normalize_whitespace(str(raw_item.get("fit_summary") or "").strip()),
                "strengths": strengths,
                "benefits_for_customer": benefits,
                "tradeoff": normalize_whitespace(str(raw_item.get("tradeoff") or "").strip()),
            }
        )

    recommendations.sort(key=lambda item: item.get("fit_score", 0), reverse=True)
    why_top_choice = normalize_whitespace(str(data.get("why_top_choice") or "").strip())
    next_question = normalize_whitespace(str(data.get("next_question") or "").strip())

    if not recommendations:
        return render_match_options_from_candidates(state, state.get("retrieved_candidates") or [])
    return _render_ranked_product_matching(lead, recommendations, next_question, why_top_choice)


async def _repair_response(state: SalesAgentState, invalid_response: str) -> str:
    next_state = state.get("next_sales_state") or "need_discovery"
    missing_slots = ", ".join(state.get("missing_slots") or []) or "không có"
    has_context = _has_retrieved_context(state)
    repair_prompt = (
        "Hãy viết lại câu trả lời sales dưới đây để tuân thủ đúng policy.\n\n"
        f"STATE: {next_state}\n"
        f"MISSING_SLOTS: {missing_slots}\n"
        f"HAS_CONTEXT: {'yes' if has_context else 'no'}\n"
        "RULES:\n"
        "- Nếu state là greeting: tối đa 2 câu, chỉ 1 câu hỏi.\n"
        "- Nếu state là need_discovery và HAS_CONTEXT=no: tối đa 2 câu, chỉ 1 câu hỏi.\n"
        "- Nếu state là need_discovery và HAS_CONTEXT=yes: tối đa 3 câu; 1-2 câu đầu tóm tắt định hướng dựa trên context, câu cuối cùng là tối đa 1 câu hỏi.\n"
        "- Chỉ hỏi về các slot còn thiếu.\n"
        "- Không nêu tên dự án hoặc địa danh ví dụ nếu chưa có context.\n"
        "- Nếu khách chưa rõ nhu cầu ở, có thể gợi ý tiêu chí sống trung lập dựa trên gia đình, con cái, mục đích mua.\n"
        "- Không ép khách sang một loại hình sản phẩm nếu khách chưa tự nêu.\n"
        "- Giữ giọng điệu tự nhiên, xưng em, gọi Anh/Chị.\n\n"
        f"Câu trả lời cần viết lại:\n{invalid_response}"
    )
    repaired = await llm_model_func(repair_prompt, history_messages=[])
    return _enforce_short_form(str(repaired).strip(), max_sentences=3 if has_context else 2)


async def build_response(state: SalesAgentState) -> Dict[str, Any]:
    next_state = state.get("next_sales_state") or "need_discovery"
    response_action = state.get("response_action") or ""
    if response_action == "catalog_overview":
        deterministic = _catalog_overview_response(state)
        return {"draft_response": deterministic, "final_response": deterministic}
    if response_action == "project_qa" and state.get("project_qa_blocked"):
        project_prompt = (
            "Để em trả lời chính xác về số lượng căn, phân khu, pháp lý hay tiện ích, "
            "Anh/Chị cho em xin đúng tên dự án Noble mình đang quan tâm nhé?"
        )
        return {"draft_response": project_prompt, "final_response": project_prompt}
    if next_state in _GROUNDED_STATES and not _has_retrieved_context(state):
        fallback = _fallback_without_context(state)
        return {"draft_response": fallback, "final_response": fallback}
    if response_action == "match_options":
        deterministic = await _product_matching_response_with_reasoning(state)
        return {"draft_response": deterministic, "final_response": deterministic}
    if response_action == "project_qa":
        deterministic = await _project_qa_response_from_context(state)
        return {"draft_response": deterministic, "final_response": deterministic}
    if next_state == "comparison":
        deterministic = await _comparison_response_from_context(state)
        return {"draft_response": deterministic, "final_response": deterministic}
    if next_state in _GROUNDED_STATES and not _llm_generation_available():
        fallback = _fallback_without_context(state)
        return {"draft_response": fallback, "final_response": fallback}

    prompt = build_prompt(state)
    history = state.get("chat_history") or []

    try:
        raw = await llm_model_func(prompt, history_messages=history)
        response_text = str(raw).strip()
        if next_state in _DISCOVERY_STATES:
            response_text = _enforce_short_form(response_text, max_sentences=_discovery_max_sentences(state))
            if _question_count(response_text) > 1:
                response_text = await _repair_response(state, response_text)
        if response_action == "project_qa":
            response_text = _sanitize_project_qa_text(response_text)
        if response_action in {"match_options", "explain_option_detail"}:
            if _question_count(response_text) > 0:
                response_text = _enforce_short_form(response_text, max_sentences=4)
        log.info(
            "build_response: state=%s action=%s response_len=%d",
            next_state,
            response_action,
            len(response_text),
        )
        return {"draft_response": response_text, "final_response": response_text}
    except Exception as e:
        log.error("build_response error: %s", e)
        fallback = "Xin lỗi, em gặp lỗi khi xử lý. Anh/Chị thử lại giúp em nhé."
        return {"draft_response": fallback, "final_response": fallback, "errors": [str(e)]}
