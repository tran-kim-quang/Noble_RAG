import json
import os
import re
import time
from typing import Any, Dict, Optional

import aiohttp

from basereal import BaseReal
from logger import logger


def _service_base_url(env_name: str, default_port: int) -> str:
    env_value = (os.getenv(env_name) or "").strip()
    if env_value:
        return env_value.rstrip("/")

    scheme = (os.getenv("NOBLE_API_SCHEME") or "http").strip() or "http"
    host = (os.getenv("NOBLE_API_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    return f"{scheme}://{host}:{default_port}"


def _normalize_endpoint(env_name: str, fallback: str) -> str:
    endpoint = (os.getenv(env_name) or fallback).strip() or fallback
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return endpoint


def _rag_stream_method() -> str:
    method = (os.getenv("RAG_STREAM_METHOD") or "POST").strip().upper()
    if method not in {"POST", "GET"}:
        logger.warning("Unsupported RAG_STREAM_METHOD=%s, fallback to POST", method)
        return "POST"
    return method


def _rag_chat_mode() -> str:
    mode = (os.getenv("RAG_CHAT_MODE") or "stream").strip().lower()
    if mode not in {"query", "stream"}:
        logger.warning("Unsupported RAG_CHAT_MODE=%s, fallback to stream", mode)
        return "stream"
    return mode


def _clean_text_for_tts(text: str) -> str:
    """Loại bỏ Markdown và các ký tự đặc biệt gây khó cho TTS."""
    if not text:
        return ""
    # Chuyển % thành chữ để tránh lỗi ElevenLabs loop
    text = text.replace("%", " phần trăm")
    # Loại bỏ Bold, Italic, Header
    text = re.sub(r"[\*#_~`>]", " ", text)
    # Loại bỏ các khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_into_sentences(text: str) -> list[str]:
    """Chia văn bản thành danh sách câu để phát TTS từng phần."""
    if not text:
        return []
    # Tách theo các dấu kết thúc câu phổ biến
    parts = re.split(r"([.!?;:\n]+)", text)
    sentences = []
    current = ""
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i+1]).strip()
        if sentence:
            cleaned = _clean_text_for_tts(sentence)
            if cleaned:
                sentences.append(cleaned)
    # Phần còn dư nếu có
    if len(parts) % 2 == 1:
        last = parts[-1].strip()
        if last:
            cleaned = _clean_text_for_tts(last)
            if cleaned:
                sentences.append(cleaned)
    return sentences


def get_noble_runtime_config() -> Dict[str, str]:
    return {
        "rag_base_url": _service_base_url("NOBLE_RAG_API_URL", 8010),
        "rag_chat_mode": _rag_chat_mode(),
        "rag_chat_endpoint": _normalize_endpoint("RAG_CHAT_ENDPOINT", "/query/stream"),
        "rag_stream_endpoint": _normalize_endpoint("RAG_STREAM_ENDPOINT", "/query/stream"),
        "rag_camera_chat_endpoint": _normalize_endpoint(
            "RAG_CAMERA_CHAT_ENDPOINT",
            "/api/v1/sales/chat-with-camera",
        ),
        "rag_stream_method": _rag_stream_method(),
        "vision_base_url": _service_base_url("NOBLE_VISION_API_URL", 8020),
        "whisper_base_url": _service_base_url("NOBLE_WHISPER_API_URL", 8001),
        "vision_source": (os.getenv("NOBLE_VISION_SOURCE") or "browser").strip() or "browser",
    }


_TTS_PUNCT_PATTERN = re.compile(r"[.!?;:,]\s*$")


def _stream_tts_config() -> Dict[str, float]:
    min_chars = max(12, int((os.getenv("RAG_STREAM_TTS_MIN_CHARS") or "45").strip() or "45"))
    max_chars = max(min_chars, int((os.getenv("RAG_STREAM_TTS_MAX_CHARS") or "140").strip() or "140"))
    flush_sec = max(0.05, float((os.getenv("RAG_STREAM_TTS_FLUSH_SEC") or "0.45").strip() or "0.45"))
    return {"min_chars": float(min_chars), "max_chars": float(max_chars), "flush_sec": flush_sec}


def _should_flush_buffer(buffer_text: str, elapsed_sec: float, cfg: Dict[str, float]) -> bool:
    text = buffer_text.strip()
    if not text:
        return False

    text_len = len(text)
    if text_len >= int(cfg["max_chars"]):
        return True
    if text_len >= int(cfg["min_chars"]) and _TTS_PUNCT_PATTERN.search(text):
        return True
    if elapsed_sec >= cfg["flush_sec"] and text_len >= max(8, int(cfg["min_chars"] // 2)):
        return True
    return False


async def _relay_stream_response_to_avatar(
    *,
    response: aiohttp.ClientResponse,
    rag_session_id: str,
    nerfreal: BaseReal,
) -> Dict[str, Any]:
    final_event: Dict[str, Any] = {"session_id": rag_session_id}
    buffer = ""
    tts_cfg = _stream_tts_config()
    speech_buffer = ""
    last_flush_ts = time.perf_counter()

    def _flush_speech_buffer(force: bool = False) -> None:
        nonlocal speech_buffer, last_flush_ts
        candidate = speech_buffer.strip()
        if not candidate:
            speech_buffer = ""
            last_flush_ts = time.perf_counter()
            return
        if not force:
            elapsed = time.perf_counter() - last_flush_ts
            if not _should_flush_buffer(candidate, elapsed, tts_cfg):
                return
        
        # Làm sạch trước khi đưa vào TTS
        cleaned = _clean_text_for_tts(candidate)
        if cleaned:
            nerfreal.put_msg_txt(cleaned)
        
        speech_buffer = ""
        last_flush_ts = time.perf_counter()

    async for raw_chunk in response.content.iter_chunked(1024):
        buffer += raw_chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON RAG stream line: %s", line)
                continue

            phase = (event.get("phase") or "").strip()
            chunk = str(event.get("chunk") or "")

            if phase == "complete":
                _flush_speech_buffer(force=True)
                final_event = event
                continue

            if phase not in {"response", "error"} or not chunk:
                continue

            speech_buffer += chunk
            _flush_speech_buffer(force=False)

    if buffer.strip():
        try:
            event = json.loads(buffer.strip())
        except json.JSONDecodeError:
            logger.warning("Skipping trailing non-JSON RAG stream payload: %s", buffer.strip())
        else:
            if (event.get("phase") or "").strip() == "complete":
                _flush_speech_buffer(force=True)
                final_event = event

    _flush_speech_buffer(force=True)
    return final_event


async def relay_rag_chat_to_avatar(
    message: str,
    rag_session_id: str,
    nerfreal: BaseReal,
    raw_transcript: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = get_noble_runtime_config()
    payload = {
        "session_id": rag_session_id,
        "message": message,
    }
    if raw_transcript:
        payload["raw_transcript"] = raw_transcript

    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None)
    base_url = cfg["rag_base_url"]

    async with aiohttp.ClientSession(timeout=timeout) as session:
        if cfg["rag_chat_mode"] == "stream":
            stream_method = cfg["rag_stream_method"]
            stream_url = f"{base_url}{cfg['rag_stream_endpoint']}"
            request_kwargs: Dict[str, Any] = {}
            if stream_method == "GET":
                request_kwargs["params"] = payload
            else:
                request_kwargs["json"] = payload

            logger.info("Forwarding chat to Noble RAG stream endpoint: %s %s", stream_method, stream_url)
            async with session.request(stream_method, stream_url, **request_kwargs) as response:
                response_body = await response.text() if response.status >= 400 else None
                if response.status >= 400:
                    raise RuntimeError(
                        f"Noble RAG returned {response.status}: {response_body or 'unknown error'}"
                    )
                return await _relay_stream_response_to_avatar(
                    response=response,
                    rag_session_id=rag_session_id,
                    nerfreal=nerfreal,
                )

        query_url = f"{base_url}{cfg['rag_chat_endpoint']}"
        logger.info("Forwarding chat to Noble RAG query endpoint: POST %s", query_url)
        async with session.post(query_url, json=payload) as response:
            response_text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"Noble RAG returned {response.status}: {response_text or 'unknown error'}")

            try:
                result = json.loads(response_text) if response_text else {}
            except json.JSONDecodeError:
                logger.warning("RAG query response is not JSON, using raw text fallback.")
                result = {"response": response_text}

        answer = str(result.get("response") or "").strip()
        if answer:
            sentences = _split_into_sentences(answer)
            for sentence in sentences:
                nerfreal.put_msg_txt(sentence)

        return result


async def relay_rag_chat_with_camera_to_avatar(
    *,
    message: str,
    nerfreal: BaseReal,
    image_bytes: bytes,
    image_filename: str = "capture.jpg",
    image_content_type: str = "image/jpeg",
    rag_session_id: Optional[str] = None,
    channel: str = "kiosk",
    source: str = "livetalking_auto_camera",
    customer_id_hint: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = get_noble_runtime_config()
    base_url = cfg["rag_base_url"]
    url = f"{base_url}{cfg['rag_camera_chat_endpoint']}"

    form = aiohttp.FormData()
    form.add_field(
        "file",
        image_bytes,
        filename=image_filename,
        content_type=image_content_type or "image/jpeg",
    )
    form.add_field("message", message)
    form.add_field("channel", channel or "kiosk")
    form.add_field("source", source or "livetalking_auto_camera")
    form.add_field("allow_resume", "true")
    if rag_session_id:
        form.add_field("session_id", rag_session_id)
    if customer_id_hint:
        form.add_field("customer_id_hint", customer_id_hint)

    timeout = aiohttp.ClientTimeout(total=90, connect=10, sock_connect=10, sock_read=90)
    logger.info("Forwarding chat to Noble RAG camera endpoint: POST %s", url)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, data=form) as response:
            response_text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"Noble RAG camera chat returned {response.status}: {response_text or 'unknown error'}")
            try:
                result = json.loads(response_text) if response_text else {}
            except json.JSONDecodeError:
                logger.warning("RAG camera chat response is not JSON, using raw text fallback.")
                result = {"response": response_text}

    fp = result.get("face_payload") if isinstance(result, dict) else None
    if isinstance(fp, dict):
        mk = list(fp.get("meta", {}).keys()) if isinstance(fp.get("meta"), dict) else []
        logger.info(
            "Noble RAG camera: Máy A đã gọi Máy B (qua chat-with-camera); response customer_id=%s session_id=%s face_payload_keys=%s meta_keys=%s",
            result.get("customer_id"),
            result.get("session_id"),
            sorted(fp.keys()),
            mk,
        )
    else:
        logger.warning(
            "Noble RAG camera: response không có face_payload (kiểm tra Máy A / MACHINE_B_BASE_URL trong container rag-service)"
        )

    answer = str(result.get("response") or "").strip()
    if answer:
        sentences = _split_into_sentences(answer)
        for sentence in sentences:
            nerfreal.put_msg_txt(sentence)
    return result
