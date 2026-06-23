import re
###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import numpy as np
import asyncio
import os
from aiohttp import web

from utils.logger import logger
from server.chat_history import chat_history_store


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)

def _log_noble_start_payload(payload: dict):
    if not isinstance(payload, dict):
        logger.info("noble_livetalking_start response: invalid payload type=%s", type(payload).__name__)
        return
    session_obj = payload.get("session")
    vision_obj = payload.get("vision_context")
    if not isinstance(session_obj, dict):
        logger.info("noble_livetalking_start response: missing session payload_keys=%s", list(payload.keys()))
        return

    logger.info(
        (
            "noble_livetalking_start resolved: session_id=%s face_session_key=%s customer_kind=%s resumed=%s ttl_sec=%s "
            "should_greet=%s greeting=%s"
        ),
        session_obj.get("session_id"),
        session_obj.get("face_session_key"),
        session_obj.get("customer_kind"),
        session_obj.get("resumed"),
        session_obj.get("ttl_sec"),
        session_obj.get("should_greet"),
        str(session_obj.get("greeting") or "")[:120],
    )

    if isinstance(vision_obj, dict):
        logger.info(
            (
                "noble_livetalking_start vision: recognized=%s name=%s age=%s gender=%s source=%s reason=%s "
                "face_id=%s confidence=%s face_count=%s bbox=%s"
            ),
            vision_obj.get("recognized"),
            vision_obj.get("name"),
            vision_obj.get("age"),
            vision_obj.get("gender"),
            vision_obj.get("source"),
            vision_obj.get("reason"),
            vision_obj.get("face_id"),
            vision_obj.get("confidence"),
            vision_obj.get("face_count"),
            vision_obj.get("bbox"),
        )


def _build_start_scan_failed_payload(reason: str) -> dict:
    return {
        "session": {
            "session_id": None,
            "face_session_key": None,
            "customer_kind": "guest",
            "resumed": False,
            "ttl_sec": 0,
            "should_greet": True,
            "greeting": (
                "Em chưa quét được khuôn mặt ở thời điểm bắt đầu phiên. "
                "Anh/chị vui lòng nhìn rõ vào camera để em nhận diện lại ạ."
            ),
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


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            chat_history_store.add_user_turn(sessionid, params['text'], opt=avatar_session.opt, params=params)
            avatar_session.put_msg_txt_chunked(
                params['text'],
                datainfo,
                max_chunk_chars=params.get('max_chunk_chars'),
            )
        elif params['type'] == 'chat':
            chat_history_store.add_user_turn(sessionid, params['text'], opt=avatar_session.opt, params=params)
            llm_enqueue = request.app.get("llm_enqueue_response")
            if llm_enqueue:
                accepted = bool(llm_enqueue(params['text'], avatar_session, datainfo))
                if not accepted:
                    logger.warning("llm enqueue rejected: session=%s", sessionid)
            else:
                llm_response = request.app.get("llm_response")
                if llm_response:
                    asyncio.get_event_loop().run_in_executor(
                        None, llm_response, params['text'], avatar_session, datainfo
                    )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
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


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
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


async def noble_livetalking_start(request):
    """Create/reset Noble_RAG LiveTalking session and optionally emit greeting."""
    try:
        params: dict = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        image_base64 = str(params.get("image_base64") or "").strip()
        if not image_base64:
            return json_error("image_base64 is required for start session vision scan")

        llm_start = request.app.get("llm_livetalking_start")
        if llm_start is None:
            return json_error("llm_livetalking_start is not configured")

        datainfo = {
            "image_base64": image_base64,
            "image_filename": params.get("image_filename"),
            "image_content_type": params.get("image_content_type"),
        }
        try:
            start_gate_timeout_sec = float(os.getenv("NOBLE_LIVETALKING_START_GATE_TIMEOUT_SEC", "8").strip() or "8")
        except ValueError:
            start_gate_timeout_sec = 8.0
        start_gate_timeout_sec = max(0.5, start_gate_timeout_sec)

        payload = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                llm_start,
                avatar_session,
                datainfo,
                True,
            ),
            timeout=start_gate_timeout_sec,
        )
        _log_noble_start_payload(payload)
        return json_ok(data=payload)
    except asyncio.TimeoutError:
        logger.error("noble_livetalking_start timed out at start-gate")
        payload = _build_start_scan_failed_payload("vision_start_gate_timeout")
        _log_noble_start_payload(payload)
        return json_ok(data=payload)
    except Exception as e:
        logger.exception("noble_livetalking_start exception:")
        return json_error(str(e))


async def noble_livetalking_stop(request):
    """Stop Noble_RAG LiveTalking session and clear session state."""
    try:
        params: dict = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        llm_stop = request.app.get("llm_livetalking_stop")
        if llm_stop is None:
            return json_error("llm_livetalking_stop is not configured")

        payload = await asyncio.get_event_loop().run_in_executor(
            None,
            llm_stop,
            avatar_session,
        )
        return json_ok(data=payload)
    except Exception as e:
        logger.exception("noble_livetalking_stop exception:")
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())



async def play_script(request):
    """Receive a script text and enqueue lines sequentially for the avatar to speak."""
    try:
        params: dict = await request.json()
        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        raw_script: str = params.get('script', '')
        split_by: str = params.get('split_by', 'sentence')
        if not raw_script:
            return json_error("script is empty")

        if split_by == 'line':
            lines = [line.strip() for line in raw_script.splitlines() if line.strip()]
        elif split_by == 'sentence':
            # Split by sentence-ending punctuation (supports Vietnamese/Chinese/English)
            chunks = re.split(r'(?<=[.!?。！？])\s+', raw_script)
            lines = [c.strip() for c in chunks if c.strip()]
        else:
            lines = [raw_script.strip()]

        total = len(lines)
        logger.info('play_script session=%s lines=%d split_by=%s', sessionid, total, split_by)

        for text in lines:
            avatar_session.put_msg_txt_chunked(text, {})

        return json_ok(data={"lines": total, "split_by": split_by})
    except Exception as e:
        logger.exception('play_script exception:')
        return json_error(str(e))

# ─── 路由注册 ──────────────────────────────────────────────────────────────

def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/noble/livetalking/start", noble_livetalking_start)
    app.router.add_post("/noble/livetalking/stop", noble_livetalking_stop)
    app.router.add_post("/interrupt_talk", interrupt_talk)
        app.router.add_post("/is_speaking", is_speaking)
    app.router.add_post("/play_script", play_script)
    app.router.add_static('/', path='web')

