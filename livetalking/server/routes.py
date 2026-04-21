###############################################################################
#  æœåŠ¡å™¨è·¯ç”± â€” ç»Ÿä¸€å¼‚å¸¸å¤„ç†çš„ API è·¯ç”±
###############################################################################

import json
import numpy as np
import asyncio
import os
import time
import aiohttp
from aiohttp import web
from http.client import RemoteDisconnected

from utils.logger import logger

STT_API_URL = (os.getenv("STT_API_URL") or "").strip()
STT_TIMEOUT_SEC = float((os.getenv("STT_TIMEOUT_SEC") or "30").strip())
STT_CONNECT_TIMEOUT_SEC = float((os.getenv("STT_CONNECT_TIMEOUT_SEC") or "8").strip())
STT_SOCK_READ_TIMEOUT_SEC = float((os.getenv("STT_SOCK_READ_TIMEOUT_SEC") or str(STT_TIMEOUT_SEC)).strip())
STT_DEFAULT_LANGUAGE = (os.getenv("STT_LANGUAGE") or "vi").strip()
STT_REQUIRE_DEEPGRAM = (os.getenv("STT_REQUIRE_DEEPGRAM") or "true").strip().lower() in {"1", "true", "yes", "on"}
RAG_TIMEOUT_SEC = float((os.getenv("NOBLE_RAG_TIMEOUT_SEC") or "35").strip())


def _truthy_env(name: str, default: str = "false") -> bool:
    value = (os.getenv(name) or default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _candidate_stt_urls() -> list[str]:
    urls = []
    direct = (os.getenv("STT_API_URL") or STT_API_URL or "").strip().rstrip("/")
    if direct:
        if direct.endswith("/transcribe"):
            urls.append(direct)
        else:
            urls.append(direct + "/transcribe")
    # Host runtime usually exposes whisper/deepgram proxy on 18001; 8001 is common in-container.
    urls.extend(
        [
            "http://127.0.0.1:18001/transcribe",
            "http://127.0.0.1:8001/transcribe",
        ]
    )
    out = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


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
    deduped = []
    for b in bases:
        if b and b not in seen:
            seen.add(b)
            _append_chat_candidates(deduped, b)

    out = []
    seen2 = set()
    for url in deduped:
        if url and url not in seen2:
            seen2.add(url)
            out.append(url)
    return out


def _candidate_rag_stream_urls() -> list[str]:
    stream_urls = []
    direct_stream = (os.getenv("NOBLE_RAG_CHAT_STREAM_URL") or "").strip()
    if direct_stream:
        ds = direct_stream.rstrip("/")
        if ds.endswith("/stream"):
            stream_urls.append(ds)
        else:
            stream_urls.append(ds + "/stream")

    direct_chat = (os.getenv("NOBLE_RAG_CHAT_URL") or "").strip()
    if direct_chat:
        stream_urls.append(direct_chat.rstrip("/") + "/stream")

    for url in _candidate_rag_chat_urls():
        if (
            url.endswith("/sales/query")
            or url.endswith("/api/v1/sales/query")
            or url.endswith("/sales/chat")
            or url.endswith("/api/v1/sales/chat")
        ):
            stream_urls.append(url + "/stream")

    out = []
    seen = set()
    for u in stream_urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _resolve_rag_session_id(params: dict, sessionid: int) -> str:
    sid = params.get("rag_session_id") or params.get("session_id")
    if sid:
        return str(sid)
    return f"livetalking-{sessionid}"


def _split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    out = []
    start = 0
    for i, ch in enumerate(buffer):
        if ch in ".,!?;:\n":
            seg = buffer[start : i + 1].strip()
            start = i + 1
            if seg:
                out.append(seg)
    return out, buffer[start:]


# â”€â”€â”€ è·¯ç”±å·¥å…·å‡½æ•° â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def json_ok(data=None):
    """è¿”å›žæˆåŠŸ JSON å“åº”"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        if isinstance(data, dict):
            body.update(data)
        else:
            body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """è¿”å›žé”™è¯¯ JSON å“åº”"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager

def get_session(request, sessionid: int):
    """ä»Ž app ä¸­èŽ·å– session å®žä¾‹"""
    return session_manager.get_session(sessionid)


# â”€â”€â”€ è·¯ç”±å¤„ç†å‡½æ•° â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def human(request):
    """æ–‡æœ¬è¾“å…¥ï¼ˆecho/chat æ¨¡å¼ï¼‰ï¼Œæ”¯æŒ voice/emotion å‚æ•°"""
    try:
        params: dict = await request.json()

        sessionid: int = params.get('sessionid', 0)
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts å‚æ•°é€ä¼ ï¼ˆvoice, emotion ç­‰ï¼‰
            datainfo['tts'] = params.get('tts')
        if params.get('raw_transcript'):
            datainfo['raw_transcript'] = params.get('raw_transcript')
        if params.get('rag_session_id'):
            datainfo['rag_session_id'] = params.get('rag_session_id')
        if params.get('session_id'):
            datainfo['session_id'] = params.get('session_id')

        if params['type'] == 'echo':
            session_manager.append_chat_message(sessionid, "assistant", params['text'], datainfo)
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            logger.info(
                "human chat request: session=%s rag_session=%s text_len=%s raw_len=%s",
                sessionid,
                datainfo.get('rag_session_id') or datainfo.get('session_id') or f"livetalking-{sessionid}",
                len(str(params.get('text', '') or '')),
                len(str(datainfo.get('raw_transcript', '') or '')),
            )
            session_manager.append_chat_message(sessionid, "user", params['text'], datainfo)
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )
                logger.info("human chat queued llm stream: session=%s", sessionid)
            else:
                logger.warning("human chat skipped: llm_response handler missing")

        response_data = {}
        if datainfo.get('rag_session_id'):
            response_data['rag_session_id'] = datainfo.get('rag_session_id')
        return json_ok(response_data if response_data else None)
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """æ‰“æ–­å½“å‰è¯´è¯"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """ä¸Šä¼ éŸ³é¢‘æ–‡ä»¶"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def stt_transcribe(request):
    """Proxy microphone audio to STT backend (Deepgram/Whisper service)."""
    try:
        stt_start = time.perf_counter()
        form = await request.post()
        fileobj = form.get("file")
        if fileobj is None:
            return json_error("file is required")

        language = str(form.get("language", STT_DEFAULT_LANGUAGE)).strip() or STT_DEFAULT_LANGUAGE
        filebytes = fileobj.file.read()
        if not filebytes:
            return json_error("empty audio")

        timeout = aiohttp.ClientTimeout(
            total=STT_TIMEOUT_SEC,
            connect=STT_CONNECT_TIMEOUT_SEC,
            sock_read=STT_SOCK_READ_TIMEOUT_SEC,
        )
        body = ""
        used_url = ""
        attempted = []
        last_error = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for stt_url in _candidate_stt_urls():
                attempted.append(stt_url)
                payload = aiohttp.FormData()
                payload.add_field(
                    "file",
                    filebytes,
                    filename=getattr(fileobj, "filename", "audio.webm"),
                    content_type=getattr(fileobj, "content_type", "application/octet-stream"),
                )
                if language:
                    payload.add_field("language", language)
                try:
                    async with session.post(stt_url, data=payload) as resp:
                        body = await resp.text()
                        if resp.status >= 400:
                            last_error = RuntimeError(f"status={resp.status}")
                            logger.warning(
                                "stt proxy backend failed: url=%s status=%s body=%s",
                                stt_url,
                                resp.status,
                                body[:500],
                            )
                            continue
                        used_url = stt_url
                        break
                except asyncio.TimeoutError as e:
                    last_error = e
                    logger.warning(
                        "stt_transcribe timeout: api=%s timeout=%.1fs connect=%.1fs read=%.1fs",
                        stt_url,
                        STT_TIMEOUT_SEC,
                        STT_CONNECT_TIMEOUT_SEC,
                        STT_SOCK_READ_TIMEOUT_SEC,
                    )
                    continue
                except aiohttp.ClientError as e:
                    last_error = e
                    logger.warning("stt_transcribe network error: url=%s err=%s", stt_url, str(e))
                    continue
        if not used_url:
            logger.warning("stt backend unavailable. tried=%s last_error=%s", attempted, last_error)
            return json_error("stt network error, vui lÃ²ng thá»­ láº¡i")

        try:
            stt_json = json.loads(body)
        except Exception:
            logger.error("stt proxy got invalid json: %s", body[:300])
            return json_error("stt backend returned invalid json")

        model_name = str(stt_json.get("model", "")).strip()
        model_l = model_name.lower()
        if STT_REQUIRE_DEEPGRAM and "deepgram" not in model_l:
            logger.error("stt backend provider mismatch, require=deepgram got model=%s", model_name)
            return json_error("stt backend is not deepgram, request blocked")
        transcript_text = str(stt_json.get("text", "")).strip()
        logger.info(
            "stt_transcribe ok: url=%s provider=%s model=%s latency=%.3fs text_len=%s text='%s'",
            used_url,
            "deepgram" if "deepgram" in model_l else "stt",
            model_name,
            time.perf_counter() - stt_start,
            len(transcript_text),
            transcript_text[:80],
        )

        return json_ok(
            data={
                "text": transcript_text,
                "language": stt_json.get("language", language),
                "provider": "deepgram" if "deepgram" in model_l else "stt",
                "model": model_name,
            }
        )
    except asyncio.CancelledError:
        logger.warning("stt_transcribe cancelled (client disconnected).")
        return json_error("stt request cancelled")
    except Exception as e:
        logger.exception("stt_transcribe exception:")
        return json_error(str(e))


async def set_audiotype(request):
    """è®¾ç½®è‡ªå®šä¹‰çŠ¶æ€ï¼ˆåŠ¨ä½œç¼–æŽ’ï¼‰"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """å½•åˆ¶æŽ§åˆ¶"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """æŸ¥è¯¢æ˜¯å¦æ­£åœ¨è¯´è¯"""
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


# â”€â”€â”€ è·¯ç”±æ³¨å†Œ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def setup_routes(app):
    """æ³¨å†Œæ‰€æœ‰è·¯ç”±åˆ° aiohttp app"""
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/stt/transcribe", stt_transcribe)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_static('/', path='web')
