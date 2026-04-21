import json
import os
import time
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger


def _truthy_env(name: str, default: str = "false") -> bool:
    value = (os.getenv(name) or default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _append_chat_candidates(urls: list[str], value: str) -> None:
    base = (value or "").strip().rstrip("/")
    if not base:
        return
    if (
        base.endswith("/sales/query")
        or base.endswith("/api/v1/sales/query")
        or base.endswith("/sales/chat")
        or base.endswith("/api/v1/sales/chat")
    ):
        urls.append(base)
        return
    if base.endswith("/sales/query/stream"):
        urls.append(base[: -len("/stream")])
        return
    if base.endswith("/api/v1/sales/query/stream"):
        urls.append(base[: -len("/stream")])
        return
    if base.endswith("/sales/chat/stream"):
        urls.append(base[: -len("/stream")])
        return
    if base.endswith("/api/v1/sales/chat/stream"):
        urls.append(base[: -len("/stream")])
        return
    urls.append(f"{base}/sales/query")
    urls.append(f"{base}/api/v1/sales/query")
    urls.append(f"{base}/sales/chat")
    urls.append(f"{base}/api/v1/sales/chat")


def _candidate_rag_chat_urls() -> list[str]:
    urls = []
    direct_url = (os.getenv("NOBLE_RAG_CHAT_URL") or "").strip()
    if direct_url:
        _append_chat_candidates(urls, direct_url)

    bases = []
    env_base = (os.getenv("NOBLE_RAG_BASE_URL") or "").strip()
    if env_base:
        bases.append(env_base.rstrip("/"))

    env_port = (os.getenv("RAG_SERVICE_PORT") or "").strip()
    if env_port:
        bases.append(f"http://127.0.0.1:{env_port}")

    bases.extend(
        [
            "http://127.0.0.1:8021",
            "http://127.0.0.1:8010",
            "http://127.0.0.1:18081",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8001",
        ]
    )

    seen = set()
    unique_bases = []
    for b in bases:
        if b and b not in seen:
            seen.add(b)
            unique_bases.append(b)

    for b in unique_bases:
        _append_chat_candidates(urls, b)

    deduped = []
    seen_urls = set()
    for url in urls:
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(url)
    return deduped


def _candidate_rag_stream_urls() -> list[str]:
    stream_urls = []
    direct_stream = (os.getenv("NOBLE_RAG_CHAT_STREAM_URL") or "").strip()
    if direct_stream:
        ds = direct_stream.rstrip("/")
        if ds.endswith("/stream"):
            stream_urls.append(ds)
        elif ds.endswith("/sales/query") or ds.endswith("/api/v1/sales/query"):
            stream_urls.append(ds + "/stream")
        elif ds.endswith("/sales/chat") or ds.endswith("/api/v1/sales/chat"):
            stream_urls.append(ds + "/stream")
        else:
            stream_urls.append(ds + "/stream")

    direct_chat = (os.getenv("NOBLE_RAG_CHAT_URL") or "").strip()
    if direct_chat:
        if direct_chat.endswith("/sales/query"):
            stream_urls.append(direct_chat + "/stream")
        elif direct_chat.endswith("/api/v1/sales/query"):
            stream_urls.append(direct_chat + "/stream")
        elif direct_chat.endswith("/sales/chat"):
            stream_urls.append(direct_chat + "/stream")
        elif direct_chat.endswith("/api/v1/sales/chat"):
            stream_urls.append(direct_chat + "/stream")
        else:
            stream_urls.append(direct_chat.rstrip("/") + "/stream")

    for url in _candidate_rag_chat_urls():
        if url.endswith("/sales/query"):
            stream_urls.append(url + "/stream")
        elif url.endswith("/api/v1/sales/query"):
            stream_urls.append(url + "/stream")
        elif url.endswith("/sales/chat"):
            stream_urls.append(url + "/stream")
        elif url.endswith("/api/v1/sales/chat"):
            stream_urls.append(url + "/stream")

    seen = set()
    unique_urls = []
    for u in stream_urls:
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def _resolve_rag_session_id(avatar_session: "BaseAvatar", datainfo: dict) -> str:
    sid = datainfo.get("rag_session_id") or datainfo.get("session_id")
    if sid:
        return str(sid)

    avatar_sid = getattr(avatar_session, "sessionid", None)
    if avatar_sid is None:
        return "livetalking-default"
    return f"livetalking-{avatar_sid}"


def _extract_rag_text(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("assistant_reply", "response", "final_response", "answer", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    out = []
    start = 0
    for i, char in enumerate(buffer):
        # Keep phrase chunks larger for TTS latency: avoid splitting on commas/colons.
        if char in ".!?\n":
            seg = buffer[start : i + 1].strip()
            start = i + 1
            if seg:
                out.append(seg)
    remain = buffer[start:]
    return out, remain


def _stream_partial_chars() -> int:
    raw = (os.getenv("NOBLE_RAG_STREAM_PARTIAL_CHARS") or "48").strip()
    try:
        return max(16, int(raw))
    except Exception:
        return 48


def _coalesce_window_sec() -> float:
    raw = (os.getenv("NOBLE_RAG_STREAM_COALESCE_WINDOW_SEC") or "0.9").strip()
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.9


def _coalesce_enabled() -> bool:
    return _truthy_env("NOBLE_RAG_STREAM_COALESCE_ENABLED", "true")


def _emit_segment(text: str, avatar_session: "BaseAvatar", datainfo: dict) -> None:
    if not text:
        return
    logger.info(text)
    avatar_session.put_msg_txt(text, datainfo)


def _stream_rag_to_audio(
    payload: dict,
    avatar_session: "BaseAvatar",
    datainfo: dict,
    timeout_sec: float,
) -> bool:
    attempted = []
    last_error = None

    for rag_url in _candidate_rag_stream_urls():
        attempted.append(rag_url)
        req = Request(
            rag_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            logger.info("llm RAG stream request start: url=%s, session_id=%s", rag_url, payload.get("session_id"))
            req_start = time.perf_counter()
            first_chunk_logged = False
            emitted_any = False
            pending = ""
            coalesce_enabled = _coalesce_enabled()
            coalesce_window = _coalesce_window_sec()

            with urlopen(req, timeout=timeout_sec) as resp:
                while True:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    phase = str(item.get("phase") or "")
                    chunk = str(item.get("chunk") or "")
                    done = bool(item.get("done"))

                    if chunk and phase in ("thinking_ack", "response", "error"):
                        if not first_chunk_logged:
                            first_chunk_logged = True
                            logger.info(
                                "llm RAG stream first chunk received: %.3fs",
                                time.perf_counter() - req_start,
                            )
                        if phase == "thinking_ack":
                            if _truthy_env("NOBLE_RAG_SPEAK_THINKING_ACK", "true"):
                                _emit_segment(chunk, avatar_session, datainfo)
                                emitted_any = True
                        else:
                            pending += chunk
                            if done:
                                final_text = pending.strip()
                                if final_text:
                                    _emit_segment(final_text, avatar_session, datainfo)
                                    pending = ""
                                    emitted_any = True
                                continue
                            # Coalesce short stream replies into one TTS request to avoid
                            # rapid start/end state toggles that can cause visual stutter.
                            if coalesce_enabled and not done:
                                if (time.perf_counter() - req_start) < coalesce_window:
                                    continue
                            segments, pending = _split_ready_sentences(pending)
                            for seg in segments:
                                if len(seg) >= 4:
                                    _emit_segment(seg, avatar_session, datainfo)
                                    emitted_any = True
                            # Emit partial chunk to reduce perceived silence for mic mode.
                            if len(pending.strip()) >= _stream_partial_chars():
                                _emit_segment(pending.strip(), avatar_session, datainfo)
                                pending = ""
                                emitted_any = True

                    if done:
                        route_category = item.get("route_category")
                        latency_sec = item.get("latency_sec")
                        if route_category is not None or latency_sec is not None:
                            logger.info(
                                "llm RAG stream complete meta: route_category=%s latency_sec=%s",
                                route_category,
                                latency_sec,
                            )
                        break

            if pending.strip():
                _emit_segment(pending.strip(), avatar_session, datainfo)
                emitted_any = True

            if emitted_any:
                return True

            last_error = RuntimeError("Stream endpoint returned no speakable chunks")

        except HTTPError as e:
            last_error = e
            if e.code == 404:
                continue
            if e.code >= 500:
                continue
            raise
        except (URLError, RemoteDisconnected, TimeoutError) as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    logger.warning("llm RAG stream unavailable. Tried=%s last_error=%s", attempted, last_error)
    return False


def _chat_rag_once(payload: dict, timeout_sec: float) -> str:
    body = ""
    last_error = None
    attempted = []

    for rag_url in _candidate_rag_chat_urls():
        attempted.append(rag_url)
        req = Request(
            rag_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            logger.info("llm RAG request start: url=%s, session_id=%s", rag_url, payload.get("session_id"))
            with urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            break
        except HTTPError as e:
            last_error = e
            if e.code == 404:
                continue
            raise
        except (URLError, RemoteDisconnected, TimeoutError) as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    if not body:
        raise RuntimeError(f"No reachable Noble RAG chat endpoint. Tried={attempted}, Last error={last_error}")

    data = json.loads(body) if body else {}
    return _extract_rag_text(data)


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    try:
        start = time.perf_counter()
        session_id = _resolve_rag_session_id(avatar_session, datainfo)
        timeout_sec = float((os.getenv("NOBLE_RAG_TIMEOUT_SEC") or "35").strip())
        stream_only = _truthy_env("NOBLE_RAG_STREAM_ONLY", "false")
        stream_error_reply = (
            os.getenv("NOBLE_RAG_STREAM_ERROR_REPLY")
            or "Sunny xin lỗi, kết nối luồng phản hồi đang bận. Bạn thử lại ngay giúp Sunny nhé."
        ).strip()

        payload = {
            "session_id": session_id,
            "message": message,
        }
        if datainfo.get("raw_transcript"):
            payload["raw_transcript"] = datainfo["raw_transcript"]

        streamed = _stream_rag_to_audio(payload, avatar_session, datainfo, timeout_sec)
        if not streamed:
            if stream_only:
                logger.warning("llm stream-only mode enabled; skip non-stream fallback /sales/query")
                if stream_error_reply:
                    _emit_segment(stream_error_reply, avatar_session, datainfo)
                return
            text = _chat_rag_once(payload, timeout_sec)
            if not text:
                logger.warning("llm RAG returned empty text in non-stream mode")
                return
            _emit_segment(text, avatar_session, datainfo)

        logger.info("llm total time: %.3fs", time.perf_counter() - start)

    except HTTPError as e:
        logger.exception("llm RAG HTTPError: %s %s", e.code, e.reason)
    except URLError as e:
        logger.exception("llm RAG URLError: %s", e)
    except Exception:
        logger.exception("llm exception:")
    return
