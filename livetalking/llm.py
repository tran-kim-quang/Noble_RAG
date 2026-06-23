import json
import os
import re
import time
import base64
import binascii
import hashlib
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from utils.logger import logger

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

DEFAULT_RAG_CHAT_URL = "http://127.0.0.1:8021/sales/query"
DEFAULT_RAG_LIVETALKING_QUERY_URL = "http://127.0.0.1:8021/integrations/livetalking/query-stream"
DEFAULT_RAG_START_URL = "http://127.0.0.1:8021/integrations/livetalking/start"
DEFAULT_RAG_STOP_URL = "http://127.0.0.1:8021/integrations/livetalking/stop"
_ENV_LOADED = False
_RAG_OPENER = build_opener(ProxyHandler({}))
_FALLBACK_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8B9Z0AAAAASUVORK5CYII="
_LLM_QUEUE_MAXSIZE = 8
_llm_job_queue: "queue.Queue[tuple[str, Any, dict[str, Any]]]" = queue.Queue(maxsize=_LLM_QUEUE_MAXSIZE)
_llm_workers_started = False
_llm_workers_lock = threading.Lock()


def _load_env_file():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or key.startswith(("NOBLE_RAG_", "AGENT_RAG_"))):
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    _load_env_file()
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using %.1f", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using %d", name, value, default)
        return default


def _resolve_rag_url() -> str:
    rag_url = _env("NOBLE_RAG_CHAT_URL") or _env("AGENT_RAG_ENDPOINT")
    if rag_url:
        return rag_url

    logger.warning(
        "NOBLE_RAG_CHAT_URL/AGENT_RAG_ENDPOINT is empty; using default endpoint: %s",
        DEFAULT_RAG_CHAT_URL,
    )
    return DEFAULT_RAG_CHAT_URL


def _resolve_livetalking_query_url() -> str:
    rag_url = _env("NOBLE_RAG_LIVETALKING_QUERY_STREAM_URL") or _env("NOBLE_RAG_LIVETALKING_QUERY_URL")
    return rag_url or DEFAULT_RAG_LIVETALKING_QUERY_URL


def _resolve_rag_start_url() -> str:
    rag_url = _env("NOBLE_RAG_LIVETALKING_START_URL") or _env("NOBLE_RAG_START_URL")
    return rag_url or DEFAULT_RAG_START_URL


def _resolve_rag_stop_url() -> str:
    rag_url = _env("NOBLE_RAG_LIVETALKING_STOP_URL") or _env("NOBLE_RAG_STOP_URL")
    return rag_url or DEFAULT_RAG_STOP_URL


def _build_rag_session_id(avatar_session: "BaseAvatar") -> str:
    return f"livetalking-{avatar_session.sessionid}-{int(time.time() * 1000)}"


def _extract_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        for item in payload:
            text = _extract_text(item)
            if text:
                return text
        return ""
    if isinstance(payload, dict):
        preferred_keys = (
            "assistant_reply",
            "answer",
            "response",
            "output",
            "text",
            "message",
            "content",
            "result",
            "data",
        )
        for key in preferred_keys:
            if key in payload:
                text = _extract_text(payload[key])
                if text:
                    return text
        for value in payload.values():
            text = _extract_text(value)
            if text:
                return text
    return ""


def _call_rag(message: str, avatar_session: "BaseAvatar", timeout_sec: float | None = None) -> tuple[str, str]:
    rag_url = _resolve_rag_url()
    timeout_sec = timeout_sec or _env_float("NOBLE_RAG_TIMEOUT_SEC", 45.0)
    retries = max(_env_int("NOBLE_RAG_RETRIES", 2), 0)
    retry_delay_sec = max(_env_float("NOBLE_RAG_RETRY_DELAY_SEC", 1.0), 0.0)
    rag_session_id = _build_rag_session_id(avatar_session)
    payload = {
        "message": message,
        "query": message,
        "question": message,
        "text": message,
        "session_id": rag_session_id,
        "sessionId": rag_session_id,
    }
    req = Request(
        rag_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info("llm RAG request start: url=%s, session_id=%s", rag_url, rag_session_id)

    body = ""
    for attempt in range(1, retries + 2):
        try:
            with _RAG_OPENER.open(req, timeout=timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            break
        except HTTPError:
            raise
        except (TimeoutError, URLError) as exc:
            if attempt > retries:
                raise
            logger.warning(
                "llm RAG request failed attempt=%d/%d url=%s err=%s",
                attempt,
                retries + 1,
                rag_url,
                exc,
            )
            if retry_delay_sec:
                time.sleep(retry_delay_sec * attempt)

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = body
    extracted = _extract_text(parsed)
    if not extracted:
        logger.warning(
            "llm RAG response has no extractable text; payload_type=%s",
            type(parsed).__name__,
        )
    return extracted, rag_session_id


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float | None = None,
    retries: int | None = None,
    retry_delay_sec: float | None = None,
) -> dict[str, Any]:
    timeout_sec = timeout_sec or _env_float("NOBLE_RAG_TIMEOUT_SEC", 45.0)
    if retries is None:
        retries = max(_env_int("NOBLE_RAG_RETRIES", 2), 0)
    else:
        retries = max(int(retries), 0)
    if retry_delay_sec is None:
        retry_delay_sec = max(_env_float("NOBLE_RAG_RETRY_DELAY_SEC", 1.0), 0.0)
    else:
        retry_delay_sec = max(float(retry_delay_sec), 0.0)
    raw_body = json.dumps(payload).encode("utf-8")

    body = ""
    for attempt in range(1, retries + 2):
        req = Request(
            url,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _RAG_OPENER.open(req, timeout=timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            break
        except HTTPError:
            raise
        except (TimeoutError, URLError) as exc:
            if attempt > retries:
                raise
            logger.warning(
                "llm RAG request failed attempt=%d/%d url=%s err=%s",
                attempt,
                retries + 1,
                url,
                exc,
            )
            if retry_delay_sec:
                time.sleep(retry_delay_sec * attempt)

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"assistant_reply": str(body).strip()}
    if isinstance(parsed, dict):
        return parsed
    return {"assistant_reply": _extract_text(parsed)}


def _post_sse_json(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float | None = None,
    retries: int | None = None,
    retry_delay_sec: float | None = None,
) -> dict[str, Any]:
    timeout_sec = timeout_sec or _env_float("NOBLE_RAG_TIMEOUT_SEC", 45.0)
    if retries is None:
        retries = max(_env_int("NOBLE_RAG_RETRIES", 2), 0)
    else:
        retries = max(int(retries), 0)
    if retry_delay_sec is None:
        retry_delay_sec = max(_env_float("NOBLE_RAG_RETRY_DELAY_SEC", 1.0), 0.0)
    else:
        retry_delay_sec = max(float(retry_delay_sec), 0.0)
    raw_body = json.dumps(payload).encode("utf-8")

    for attempt in range(1, retries + 2):
        req = Request(
            url,
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            done_payload: dict[str, Any] | None = None
            delta_parts: list[str] = []
            with _RAG_OPENER.open(req, timeout=timeout_sec) as resp:
                for line_bytes in resp:
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict) and "assistant_reply" in parsed and "session" in parsed:
                        done_payload = parsed
                        break
                    if isinstance(parsed, dict):
                        text_chunk = str(parsed.get("text") or "").strip()
                        if text_chunk:
                            delta_parts.append(text_chunk)
            if done_payload is not None:
                return done_payload
            if delta_parts:
                return {"assistant_reply": " ".join(delta_parts).strip()}
            return {"assistant_reply": ""}
        except HTTPError:
            raise
        except (TimeoutError, URLError) as exc:
            if attempt > retries:
                raise
            logger.warning(
                "llm RAG SSE request failed attempt=%d/%d url=%s err=%s",
                attempt,
                retries + 1,
                url,
                exc,
            )
            if retry_delay_sec:
                time.sleep(retry_delay_sec * attempt)


def _resolve_snapshot_payload(datainfo: dict[str, Any] | None = None) -> dict[str, Any]:
    info = datainfo or {}
    image_base64 = str(info.get("image_base64") or "").strip()
    image_filename = str(info.get("image_filename") or "").strip() or "snapshot.jpg"
    image_content_type = str(info.get("image_content_type") or "").strip() or "image/jpeg"
    if not image_base64:
        image_base64 = _FALLBACK_IMAGE_BASE64
        image_filename = "fallback.png"
        image_content_type = "image/png"
    return {
        "image_base64": image_base64,
        "image_filename": image_filename,
        "image_content_type": image_content_type,
    }


def _build_request_id(prefix: str, avatar_session: "BaseAvatar") -> str:
    sessionid = str(getattr(avatar_session, "sessionid", "") or "").strip() or "unknown"
    return f"{prefix}-{sessionid}-{int(time.time() * 1000)}"


def _image_payload_meta(snapshot: dict[str, Any]) -> dict[str, Any]:
    image_base64 = str(snapshot.get("image_base64") or "").strip()
    image_filename = str(snapshot.get("image_filename") or "").strip()
    image_content_type = str(snapshot.get("image_content_type") or "").strip()
    cleaned = image_base64
    if cleaned.startswith("data:"):
        _, _, cleaned = cleaned.partition(",")

    byte_size = None
    image_sha1 = None
    decode_error = None
    try:
        raw = base64.b64decode(cleaned, validate=True)
        byte_size = len(raw)
        image_sha1 = hashlib.sha1(raw).hexdigest()[:16]
    except (binascii.Error, ValueError) as exc:
        decode_error = str(exc)

    return {
        "b64_len": len(cleaned),
        "byte_size": byte_size,
        "image_sha1": image_sha1,
        "decode_error": decode_error,
        "image_filename": image_filename or None,
        "image_content_type": image_content_type or None,
    }


def _normalize_text_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _should_skip_duplicate_speech(avatar_session: "BaseAvatar", text: str) -> bool:
    normalized = _normalize_text_for_dedupe(text)
    if not normalized:
        return False
    now = time.time()
    window_sec = max(1.0, _env_float("NOBLE_RAG_REPLY_DEDUPE_SEC", 8.0))
    last_text = str(getattr(avatar_session, "_noble_last_spoken_text", "") or "")
    last_ts = float(getattr(avatar_session, "_noble_last_spoken_ts", 0.0) or 0.0)
    if normalized == last_text and (now - last_ts) <= window_sec:
        return True
    setattr(avatar_session, "_noble_last_spoken_text", normalized)
    setattr(avatar_session, "_noble_last_spoken_ts", now)
    return False


def _set_noble_session_state(
    avatar_session: "BaseAvatar",
    session_id: str | None,
    face_session_key: str | None,
) -> None:
    setattr(avatar_session, "_noble_rag_session_id", str(session_id or "").strip() or None)
    setattr(avatar_session, "_noble_rag_face_session_key", str(face_session_key or "").strip() or None)


def _clear_noble_session_state(avatar_session: "BaseAvatar") -> None:
    _set_noble_session_state(avatar_session, None, None)
    setattr(avatar_session, "_noble_rag_vision_context", None)


def _update_noble_session_state_from_payload(avatar_session: "BaseAvatar", payload: dict[str, Any]) -> None:
    session_obj = payload.get("session")
    if not isinstance(session_obj, dict):
        return
    _set_noble_session_state(
        avatar_session,
        session_obj.get("session_id"),
        session_obj.get("face_session_key"),
    )
    vision_obj = payload.get("vision_context")
    if isinstance(vision_obj, dict):
        setattr(avatar_session, "_noble_rag_vision_context", vision_obj)


def _log_livetalking_payload(tag: str, payload: dict[str, Any]) -> None:
    session_obj = payload.get("session") if isinstance(payload, dict) else None
    vision_obj = payload.get("vision_context") if isinstance(payload, dict) else None
    if not isinstance(session_obj, dict) and not isinstance(vision_obj, dict):
        logger.info("llm RAG %s response summary: payload_keys=%s", tag, list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__)
        return

    session_id = session_obj.get("session_id") if isinstance(session_obj, dict) else None
    face_session_key = session_obj.get("face_session_key") if isinstance(session_obj, dict) else None
    customer_kind = session_obj.get("customer_kind") if isinstance(session_obj, dict) else None
    resumed = session_obj.get("resumed") if isinstance(session_obj, dict) else None
    should_greet = session_obj.get("should_greet") if isinstance(session_obj, dict) else None

    recognized = vision_obj.get("recognized") if isinstance(vision_obj, dict) else None
    name = vision_obj.get("name") if isinstance(vision_obj, dict) else None
    age = vision_obj.get("age") if isinstance(vision_obj, dict) else None
    gender = vision_obj.get("gender") if isinstance(vision_obj, dict) else None
    source = vision_obj.get("source") if isinstance(vision_obj, dict) else None
    reason = vision_obj.get("reason") if isinstance(vision_obj, dict) else None
    face_id = vision_obj.get("face_id") if isinstance(vision_obj, dict) else None
    confidence = vision_obj.get("confidence") if isinstance(vision_obj, dict) else None
    face_count = vision_obj.get("face_count") if isinstance(vision_obj, dict) else None

    logger.info(
        (
            "llm RAG %s response: session_id=%s face_session_key=%s customer_kind=%s resumed=%s should_greet=%s "
            "recognized=%s name=%s age=%s gender=%s source=%s reason=%s face_id=%s confidence=%s face_count=%s"
        ),
        tag,
        session_id,
        face_session_key,
        customer_kind,
        resumed,
        should_greet,
        recognized,
        name,
        age,
        gender,
        source,
        reason,
        face_id,
        confidence,
        face_count,
    )


def _build_start_scan_failed_payload(reason: str) -> dict[str, Any]:
    greeting = (
        "Em chưa quét được khuôn mặt ở thời điểm bắt đầu phiên. "
        "Anh/chị vui lòng nhìn rõ vào camera để em nhận diện lại ạ."
    )
    return {
        "session": {
            "session_id": None,
            "face_session_key": None,
            "customer_kind": "guest",
            "resumed": False,
            "ttl_sec": 0,
            "should_greet": True,
            "greeting": greeting,
        },
        "vision_context": {
            "recognized": False,
            "name": None,
            "age": None,
            "gender": None,
            "source": "vision_start_gate",
            "reason": reason,
            "face_id": None,
            "confidence": 0.0,
            "face_count": 0,
            "bbox": None,
        },
    }


def _call_livetalking_start(
    avatar_session: "BaseAvatar",
    datainfo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = _resolve_rag_start_url()
    start_timeout_sec = max(_env_float("NOBLE_RAG_LIVETALKING_START_TIMEOUT_SEC", 6.0), 0.5)
    start_retries = max(_env_int("NOBLE_RAG_LIVETALKING_START_RETRIES", 0), 0)
    snapshot = _resolve_snapshot_payload(datainfo)
    request_id = _build_request_id("ltstart", avatar_session)
    image_meta = _image_payload_meta(snapshot)
    payload = {
        "request_id": request_id,
        "session_id": None,
        **snapshot,
    }
    logger.info(
        (
            "llm RAG livetalking start: url=%s request_id=%s avatar_session=%s "
            "timeout_sec=%.2f retries=%s image_b64_len=%s image_bytes=%s image_sha1=%s decode_error=%s file=%s content_type=%s"
        ),
        url,
        request_id,
        getattr(avatar_session, "sessionid", None),
        start_timeout_sec,
        start_retries,
        image_meta["b64_len"],
        image_meta["byte_size"],
        image_meta["image_sha1"],
        image_meta["decode_error"],
        image_meta["image_filename"],
        image_meta["image_content_type"],
    )
    try:
        parsed = _post_json(
            url,
            payload,
            timeout_sec=start_timeout_sec,
            retries=start_retries,
            retry_delay_sec=0.4,
        )
    except (TimeoutError, URLError) as exc:
        logger.error(
            "llm RAG livetalking start unavailable request_id=%s avatar_session=%s err=%s",
            request_id,
            getattr(avatar_session, "sessionid", None),
            exc,
        )
        _clear_noble_session_state(avatar_session)
        parsed = _build_start_scan_failed_payload("vision_start_timeout_or_unavailable")
        _log_livetalking_payload("livetalking/start-fallback", parsed)
        return parsed
    except HTTPError as exc:
        logger.error(
            "llm RAG livetalking start http-error request_id=%s avatar_session=%s code=%s",
            request_id,
            getattr(avatar_session, "sessionid", None),
            exc.code,
        )
        _clear_noble_session_state(avatar_session)
        parsed = _build_start_scan_failed_payload("vision_start_http_error")
        _log_livetalking_payload("livetalking/start-fallback", parsed)
        return parsed
    except Exception:
        logger.exception(
            "llm RAG livetalking start unexpected request_id=%s avatar_session=%s",
            request_id,
            getattr(avatar_session, "sessionid", None),
        )
        _clear_noble_session_state(avatar_session)
        parsed = _build_start_scan_failed_payload("vision_start_exception")
        _log_livetalking_payload("livetalking/start-fallback", parsed)
        return parsed
    _update_noble_session_state_from_payload(avatar_session, parsed)
    _log_livetalking_payload("livetalking/start", parsed)
    vision_reason = ""
    if isinstance(parsed.get("vision_context"), dict):
        vision_reason = str(parsed["vision_context"].get("reason") or "")
    if "Cannot decode image" in vision_reason or "Invalid base64 image payload" in vision_reason:
        logger.error(
            (
                "llm RAG livetalking start decode-failed request_id=%s avatar_session=%s "
                "image_b64_len=%s image_bytes=%s image_sha1=%s reason=%s"
            ),
            request_id,
            getattr(avatar_session, "sessionid", None),
            image_meta["b64_len"],
            image_meta["byte_size"],
            image_meta["image_sha1"],
            vision_reason,
        )
    return parsed


def _call_livetalking_query(
    message: str,
    avatar_session: "BaseAvatar",
    datainfo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = _resolve_livetalking_query_url()
    request_id = _build_request_id("ltquery", avatar_session)
    payload = {
        "request_id": request_id,
        "message": message,
        "session_id": str(getattr(avatar_session, "_noble_rag_session_id", "") or "").strip() or None,
    }
    logger.info(
        "llm RAG livetalking query start: url=%s request_id=%s session_id=%s avatar_session=%s",
        url,
        request_id,
        payload["session_id"],
        getattr(avatar_session, "sessionid", None),
    )
    parsed = _post_sse_json(url, payload)
    _update_noble_session_state_from_payload(avatar_session, parsed)
    _log_livetalking_payload("livetalking/query", parsed)
    return parsed


def _call_livetalking_stop(avatar_session: "BaseAvatar") -> dict[str, Any]:
    url = _resolve_rag_stop_url()
    payload = {
        "session_id": str(getattr(avatar_session, "_noble_rag_session_id", "") or "").strip() or None,
        "face_session_key": str(getattr(avatar_session, "_noble_rag_face_session_key", "") or "").strip() or None,
    }
    logger.info(
        "llm RAG livetalking stop: url=%s, session_id=%s, face_session_key=%s",
        url,
        payload["session_id"],
        payload["face_session_key"],
    )
    parsed = _post_json(url, payload)
    _clear_noble_session_state(avatar_session)
    return parsed


def llm_livetalking_start(
    avatar_session: "BaseAvatar",
    datainfo: dict[str, Any] | None = None,
    speak_greeting: bool = True,
) -> dict[str, Any]:
    payload = _call_livetalking_start(avatar_session, datainfo=datainfo)
    session_obj = payload.get("session") if isinstance(payload, dict) else None
    if speak_greeting and isinstance(session_obj, dict):
        should_greet = bool(session_obj.get("should_greet", False))
        greeting = str(session_obj.get("greeting") or "").strip()
        if should_greet and greeting:
            if _should_skip_duplicate_speech(avatar_session, greeting):
                logger.warning(
                    "skip duplicate greeting spam guard: avatar_session=%s session_id=%s",
                    getattr(avatar_session, "sessionid", None),
                    str(getattr(avatar_session, "_noble_rag_session_id", "") or "").strip() or None,
                )
                return payload
            eventinfo = dict(datainfo or {})
            avatar_session.put_msg_txt_chunked(greeting, eventinfo)
    return payload


def llm_livetalking_stop(avatar_session: "BaseAvatar") -> dict[str, Any]:
    return _call_livetalking_stop(avatar_session)


def _call_dashscope(message: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个知识助手，尽量以简短、口语化的方式输出"},
            {"role": "user", "content": message},
        ],
        stream=False,
    )
    return _extract_text(completion.model_dump())

def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    try:
        start = time.perf_counter()
        reply = ""
        rag_session_id = str(getattr(avatar_session, "_noble_rag_session_id", "") or "").strip()

        try:
            if not rag_session_id:
                llm_livetalking_start(avatar_session, datainfo=datainfo, speak_greeting=False)
            payload = _call_livetalking_query(message, avatar_session, datainfo=datainfo)
            reply = _extract_text(payload)
            rag_session_id = str(getattr(avatar_session, "_noble_rag_session_id", "") or "").strip()
            if not reply:
                reply, rag_session_id = _call_rag(message, avatar_session)
        except HTTPError as exc:
            logger.exception("llm RAG HTTPError: %s", exc)
            try:
                reply, rag_session_id = _call_rag(message, avatar_session)
            except Exception:
                logger.exception("llm fallback sales/query exception:")
        except URLError as exc:
            logger.exception("llm RAG URLError: %s", exc)
            try:
                reply, rag_session_id = _call_rag(message, avatar_session)
            except Exception:
                logger.exception("llm fallback sales/query exception:")
        except Exception:
            logger.exception("llm RAG exception:")
            try:
                reply, rag_session_id = _call_rag(message, avatar_session)
            except Exception:
                logger.exception("llm fallback sales/query exception:")

        if not reply:
            logger.warning("llm no reply from RAG, fallback disabled.")
            return

        eventinfo = dict(datainfo or {})
        eventinfo["raw_transcript"] = message
        if rag_session_id:
            eventinfo["rag_session_id"] = rag_session_id
        vision_context = getattr(avatar_session, "_noble_rag_vision_context", None)
        if isinstance(vision_context, dict):
            eventinfo["vision_context"] = json.dumps(vision_context, ensure_ascii=False)

        if _should_skip_duplicate_speech(avatar_session, reply):
            logger.warning(
                "skip duplicate assistant reply spam guard: avatar_session=%s session_id=%s",
                getattr(avatar_session, "sessionid", None),
                rag_session_id or None,
            )
            return

        logger.info(reply)
        avatar_session.put_msg_txt_chunked(reply, eventinfo)
        logger.info("llm total time: %.3fs", time.perf_counter() - start)
    except Exception:
        logger.exception("llm exception:")
        return


def _llm_worker_loop(worker_id: int):
    while True:
        try:
            message, avatar_session, datainfo = _llm_job_queue.get()
            try:
                llm_response(message, avatar_session, datainfo)
            finally:
                _llm_job_queue.task_done()
        except Exception:
            logger.exception("llm worker[%s] unexpected exception", worker_id)


def _ensure_llm_workers_started():
    global _llm_workers_started
    if _llm_workers_started:
        return
    with _llm_workers_lock:
        if _llm_workers_started:
            return
        worker_count = max(_env_int("NOBLE_RAG_LLM_WORKERS", 1), 1)
        for idx in range(worker_count):
            th = threading.Thread(
                target=_llm_worker_loop,
                args=(idx,),
                name=f"noble-llm-worker-{idx}",
                daemon=True,
            )
            th.start()
        _llm_workers_started = True
        logger.info("llm async queue started: workers=%s maxsize=%s", worker_count, _LLM_QUEUE_MAXSIZE)


def enqueue_llm_response(message, avatar_session: "BaseAvatar", datainfo: dict | None = None) -> bool:
    _ensure_llm_workers_started()
    sessionid = str(getattr(avatar_session, "sessionid", "") or "").strip()
    payload = (str(message or ""), avatar_session, dict(datainfo or {}))
    try:
        _llm_job_queue.put_nowait(payload)
        logger.info(
            "llm queued: session=%s queue_size=%s",
            sessionid or None,
            _llm_job_queue.qsize(),
        )
        return True
    except queue.Full:
        logger.warning("llm queue full, drop message: session=%s", sessionid or None)
        return False
