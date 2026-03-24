"""Streaming query endpoint — preserves the existing /query/stream behaviour."""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import get_settings
from core.dependencies import llm_model_func, rag
from models.api_models import QueryRequest
from rag.retriever import (
    decompose_subqueries,
    route_query,
    summarize_search_answer,
    query_rag,
)
from memory.chat_history_store import load_chat_history, save_chat_history
from sales.prompt_builder import SYSTEM_PROMPT as SYSTEM_PERSONA
from utils.time import timed_await, now_vietnam_str
from lightrag import QueryParam

router = APIRouter(tags=["query"])
settings = get_settings()
log = logging.getLogger("rag-service")


@router.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def generate():
        t_total = time.perf_counter()
        session_id = request.session_id or "default_session"
        history = await timed_await(
            f"stream.history_load[{session_id}]", load_chat_history(session_id)
        )

        # Skip decomposition for very short queries
        if len(request.query.strip().split()) < 5:
            subqueries = [request.query]
        else:
            subqueries = await timed_await(
                "stream.decompose",
                decompose_subqueries(request.query),
            )

        if len(subqueries) > 1:
            for chunk in await _handle_multi_intent(subqueries, request, history, session_id, t_total):
                yield chunk
            return

        # Single query path
        category, refined_query = await timed_await(
            "stream.route", route_query(request.query, history)
        )
        log.info("Stream router: %s", category)
        full_answer = ""

        if category == "SEARCH":
            yield json.dumps(
                {"chunk": "  \n*(Em đang tìm kiếm thông tin mới nhất trên mạng...)*", "done": False},
                ensure_ascii=False,
            ) + "\n"
            search_q = refined_query or request.query
            full_answer = await timed_await(
                "stream.search",
                summarize_search_answer(request.query, search_q, history, SYSTEM_PERSONA),
            )
            yield json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False) + "\n"

        elif category == "OTHER":
            prompt = (
                f"{SYSTEM_PERSONA}\n\nThời gian hiện tại (VN): {now_vietnam_str()}\n\n"
                f"Câu hỏi/lời nhắn: {request.query}\n\nTrả lời ngắn gọn (≤3 câu):"
            )
            raw = await timed_await(
                "stream.other", llm_model_func(prompt, history_messages=history)
            )
            full_answer = str(raw)
            yield json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False) + "\n"

        else:  # RAG path with streaming
            yield json.dumps(
                {"chunk": "  \n*(Em đang truy xuất thông tin từ kho tài liệu...)*", "done": False},
                ensure_ascii=False,
            ) + "\n"
            rag_query = (
                f"{SYSTEM_PERSONA}\n\n"
                f"Câu hỏi của khách hàng: {request.query}\n\n"
                "Hãy trả lời dựa trên thông tin retrieve. Viết bằng tiếng Việt, ngắn gọn."
            )
            try:
                from lightrag import QueryParam
                response = await asyncio.wait_for(
                    rag.aquery(
                        rag_query,
                        param=QueryParam(
                            top_k=request.top_k,
                            mode="naive",
                            conversation_history=history,
                            stream=True,
                        ),
                    ),
                    timeout=settings.query_timeout_sec,
                )

                buffer = ""
                if response is None:
                    full_answer = "Xin lỗi, em chưa tìm thấy thông tin phù hợp."
                    yield json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False) + "\n"
                elif isinstance(response, str):
                    full_answer = response or "Xin lỗi, em chưa tìm thấy thông tin phù hợp."
                    for sent in re.split(r"(?<=[.!?\n])\s*", full_answer):
                        if sent.strip():
                            yield json.dumps({"chunk": sent.strip(), "done": False}, ensure_ascii=False) + "\n"
                            await asyncio.sleep(0)
                else:
                    async for token in response:
                        if not token:
                            continue
                        buffer += token
                        parts = re.split(r"(?<=[.!?\n])\s*", buffer)
                        if len(parts) > 1:
                            for sent in parts[:-1]:
                                if sent.strip():
                                    full_answer += sent.strip() + " "
                                    yield json.dumps({"chunk": sent.strip(), "done": False}, ensure_ascii=False) + "\n"
                            buffer = parts[-1]
                    if buffer.strip():
                        full_answer += buffer.strip()
                        yield json.dumps({"chunk": buffer.strip(), "done": False}, ensure_ascii=False) + "\n"

            except asyncio.TimeoutError:
                full_answer = "Xin lỗi, hệ thống đang bận. Bạn thử lại nhé."
                yield json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False) + "\n"
            except Exception as e:
                log.error("Stream RAG error: %s", e)
                full_answer = "Xin lỗi, em gặp lỗi khi truy xuất dữ liệu."
                yield json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False) + "\n"

        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": full_answer.strip()})
        await timed_await(
            f"stream.history_save[{session_id}]", save_chat_history(session_id, history)
        )
        yield json.dumps({"chunk": "", "done": True, "model": settings.llm_model}, ensure_ascii=False) + "\n"
        log.info("Timing[stream.total]: %.2fs", time.perf_counter() - t_total)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def _handle_multi_intent(
    subqueries: List[str],
    request: QueryRequest,
    history: List[Dict[str, Any]],
    session_id: str,
    t_total: float,
):
    """Run subqueries in parallel and yield results as they complete."""
    import asyncio
    completed: Dict[int, str] = {}
    semaphore = asyncio.Semaphore(settings.subquery_max_concurrency)

    async def _run(idx: int, sub_q: str):
        async with semaphore:
            cat, refined = await route_query(sub_q, history)
            if cat == "SEARCH":
                ans = await summarize_search_answer(sub_q, refined or sub_q, history, SYSTEM_PERSONA, 2)
            elif cat == "OTHER":
                p = f"{SYSTEM_PERSONA}\n\nTiểu câu hỏi: {sub_q}\nTrả lời ngắn gọn:"
                raw = await llm_model_func(p, history_messages=history)
                ans = str(raw)
            else:
                ans = await query_rag(sub_q, top_k=request.top_k, history=history, system_context=SYSTEM_PERSONA)
                if not ans:
                    ans = "Xin lỗi, em chưa tìm thấy thông tin phù hợp."
            return idx, ans

    tasks = [asyncio.create_task(_run(i, q)) for i, q in enumerate(subqueries, 1)]
    results = []
    for task in asyncio.as_completed(tasks):
        try:
            idx, ans = await task
            completed[idx] = ans
            results.append(json.dumps({"chunk": f"{idx}) {ans}", "done": False}, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error("Subquery task error: %s", e)

    full_answer = "\n\n".join(
        completed.get(i, f"{i}) Lỗi xử lý.") for i in range(1, len(subqueries) + 1)
    )
    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": full_answer})
    await save_chat_history(session_id, history)
    results.append(
        json.dumps({"chunk": "", "done": True, "model": "multi-intent-composed"}, ensure_ascii=False) + "\n"
    )
    log.info("Timing[stream.multi.total]: %.2fs", time.perf_counter() - t_total)
    return results
