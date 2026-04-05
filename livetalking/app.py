###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from rag_chat_client import (
    get_noble_runtime_config,
    relay_rag_chat_to_avatar,
    relay_rag_chat_with_camera_to_avatar,
)

import argparse
import random
import shutil
import asyncio
import os
from contextlib import suppress
from typing import Dict, Optional, Tuple
from uuid import uuid4
from logger import logger
import gc


app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
chat_tasks:Dict[int, asyncio.Task] = {}
rag_session_ids:Dict[int, str] = {}
rag_session_confirmed:Dict[int, bool] = {}
camera_customer_hints:Dict[int, str] = {}
camera_frame_cache:Dict[int, Tuple[bytes, str]] = {}
opt = None
model = None
avatar = None
camera_capture_lock: Optional[asyncio.Lock] = None
        

#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal


def build_rag_session_id() -> str:
    return f"livetalking_{uuid4().hex}"


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _get_camera_capture_lock() -> asyncio.Lock:
    global camera_capture_lock
    if camera_capture_lock is None:
        camera_capture_lock = asyncio.Lock()
    return camera_capture_lock


def _capture_camera_frame_sync() -> bytes:
    import cv2

    camera_index = int((os.getenv("NOBLE_CAMERA_INDEX") or "0").strip() or "0")
    camera_indexes_raw = (os.getenv("NOBLE_CAMERA_INDEXES") or "").strip()
    camera_indexes = []
    if camera_indexes_raw:
        for token in camera_indexes_raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                camera_indexes.append(int(token))
            except ValueError:
                continue
    if camera_index not in camera_indexes:
        camera_indexes.insert(0, camera_index)
    if not camera_indexes:
        camera_indexes = [0]

    warmup_frames = max(1, int((os.getenv("NOBLE_CAMERA_WARMUP_FRAMES") or "4").strip() or "4"))
    jpeg_quality = max(30, min(100, int((os.getenv("NOBLE_CAMERA_JPEG_QUALITY") or "90").strip() or "90")))
    width = int((os.getenv("NOBLE_CAMERA_WIDTH") or "0").strip() or "0")
    height = int((os.getenv("NOBLE_CAMERA_HEIGHT") or "0").strip() or "0")

    backend_names = (os.getenv("NOBLE_CAMERA_BACKENDS") or "").strip()
    if backend_names:
        backend_order = [name.strip().upper() for name in backend_names.split(",") if name.strip()]
    elif os.name == "nt":
        backend_order = ["CAP_MSMF", "CAP_DSHOW", "CAP_ANY"]
    else:
        backend_order = ["CAP_ANY"]

    backend_values = []
    for name in backend_order:
        value = getattr(cv2, name, None)
        if value is None:
            continue
        backend_values.append((name, value))
    if not backend_values:
        backend_values = [("CAP_ANY", cv2.CAP_ANY)]

    open_errors = []
    read_errors = []
    for idx in camera_indexes:
        for backend_name, backend in backend_values:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                open_errors.append(f"index={idx}/{backend_name}")
                cap.release()
                continue
            try:
                if width > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height > 0:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                ok = False
                frame = None
                for _ in range(warmup_frames):
                    ok, frame = cap.read()
                if not ok or frame is None:
                    read_errors.append(f"index={idx}/{backend_name}: camera read failed")
                    continue

                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                if not encoded_ok:
                    read_errors.append(f"index={idx}/{backend_name}: jpeg encode failed")
                    continue
                return bytes(encoded.tobytes())
            finally:
                cap.release()

    detail = "; ".join((open_errors + read_errors)[:6]) or "unknown camera open/read failure"
    raise RuntimeError(f"cannot capture camera frame ({detail})")


def _capture_for_vision_smoke_sync() -> bytes:
    """Giảm spam WARN từ OpenCV khi quét MSMF/DSHOW cho smoke test."""
    try:
        import cv2

        logging_mod = getattr(getattr(cv2, "utils", None), "logging", None)
        if logging_mod is not None and hasattr(logging_mod, "setLogLevel"):
            level = getattr(logging_mod, "LOG_LEVEL_SILENT", None)
            if level is None:
                level = getattr(logging_mod, "LOG_LEVEL_ERROR", None)
            if level is not None:
                logging_mod.setLogLevel(level)
    except Exception:
        pass
    return _capture_camera_frame_sync()


async def maybe_capture_camera_frame() -> Optional[bytes]:
    if not _env_flag("NOBLE_AUTO_CAMERA_ON_CHAT", True):
        return None
    lock = _get_camera_capture_lock()
    loop = asyncio.get_event_loop()
    async with lock:
        return await loop.run_in_executor(None, _capture_camera_frame_sync)


def _decode_image_data_url(raw_value: str) -> Tuple[bytes, str]:
    text = (raw_value or "").strip()
    if not text:
        raise ValueError("image_base64 is empty")

    content_type = "image/jpeg"
    payload = text
    if text.startswith("data:"):
        match = re.match(r"^data:([^;]+);base64,(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError("invalid image data URL")
        content_type = (match.group(1) or "image/jpeg").strip() or "image/jpeg"
        payload = match.group(2)

    try:
        return base64.b64decode(payload, validate=True), content_type
    except Exception as exc:
        raise ValueError("invalid image_base64 payload") from exc


async def build_nerfreal_async(sessionid: int) -> BaseReal:
    """
    Windows can fail to initialize multiprocessing resources when avatar
    creation happens in a threadpool worker. Fallback to direct init.
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, build_nerfreal, sessionid)
    except PermissionError:
        logger.warning(
            'Threadpool init failed for session %s; fallback to direct init.',
            sessionid,
        )
        return build_nerfreal(sessionid)


async def cancel_chat_task(sessionid: int) -> None:
    task = chat_tasks.pop(sessionid, None)
    if not task or task.done():
        return

    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def run_rag_chat(
    sessionid: int,
    rag_session_id: str,
    message: str,
    raw_transcript: str = None,
    image_bytes: Optional[bytes] = None,
    image_content_type: str = "image/jpeg",
):
    try:
        nerfreal = nerfreals[sessionid]
        use_camera_route = _env_flag("NOBLE_USE_CAMERA_CHAT_ENDPOINT", True)
        strict_camera_route = _env_flag("NOBLE_CAMERA_CHAT_STRICT", False)
        if use_camera_route:
            if image_bytes is None:
                cached = camera_frame_cache.get(sessionid)
                if cached and cached[0]:
                    image_bytes, image_content_type = cached
                    logger.info(
                        "Using cached browser frame for session=%s bytes=%s",
                        sessionid,
                        len(image_bytes or b""),
                    )
            if image_bytes is None and not strict_camera_route:
                try:
                    image_bytes = await maybe_capture_camera_frame()
                    logger.info(
                        "Using local OpenCV capture for session=%s bytes=%s",
                        sessionid,
                        len(image_bytes or b""),
                    )
                except Exception as capture_exc:
                    logger.warning("auto camera capture failed, fallback to standard chat: %s", capture_exc)
                    image_bytes = None

            if image_bytes:
                source = (os.getenv("NOBLE_CAMERA_CHAT_SOURCE") or "livetalking_auto_camera").strip()
                channel = (os.getenv("NOBLE_CAMERA_CHAT_CHANNEL") or "kiosk").strip()
                current_hint = camera_customer_hints.get(sessionid)
                # By default, let backend bind/open session by customer_id from vision.
                prefer_customer_session = _env_flag("NOBLE_CAMERA_SESSION_BY_CUSTOMER_ID", True)
                session_id_for_camera = None
                if not prefer_customer_session and rag_session_confirmed.get(sessionid):
                    session_id_for_camera = rag_session_id
                result = await relay_rag_chat_with_camera_to_avatar(
                    message=message,
                    nerfreal=nerfreal,
                    image_bytes=image_bytes,
                    image_filename=f"livetalking_session_{sessionid}.jpg",
                    image_content_type=image_content_type or "image/jpeg",
                    rag_session_id=session_id_for_camera,
                    channel=channel or "kiosk",
                    source=source or "livetalking_auto_camera",
                    customer_id_hint=current_hint,
                )
                logger.info(
                    "Camera route delivered for session=%s session_id=%s customer_id=%s",
                    sessionid,
                    str(result.get("session_id") or ""),
                    str(result.get("customer_id") or ""),
                )
                resolved_session_id = str(result.get("session_id") or "").strip()
                if resolved_session_id:
                    rag_session_ids[sessionid] = resolved_session_id
                    rag_session_confirmed[sessionid] = True
                resolved_customer_id = str(result.get("customer_id") or "").strip()
                if resolved_customer_id:
                    camera_customer_hints[sessionid] = resolved_customer_id
                return
            if strict_camera_route:
                logger.warning("camera route strict mode: no image available for session=%s", sessionid)
                nerfreal.put_msg_txt(
                    "Em chưa chụp được ảnh camera cho lượt này. Anh/Chị bật camera browser rồi thử lại giúp em nhé."
                )
                return

        await relay_rag_chat_to_avatar(
            message=message,
            rag_session_id=rag_session_id,
            raw_transcript=raw_transcript,
            nerfreal=nerfreal,
        )
    except asyncio.CancelledError:
        logger.info('Cancelled Noble RAG chat relay for LiveTalking session %s', sessionid)
        raise
    except Exception:
        logger.exception('Failed to relay Noble RAG response')
        nerfreals[sessionid].put_msg_txt(
            'Xin loi, em gap su co khi ket noi sang Noble RAG. Anh/Chi thu lai giup em nhe.'
        )
    finally:
        current_task = asyncio.current_task()
        if chat_tasks.get(sessionid) is current_task:
            chat_tasks.pop(sessionid, None)

#@app.route('/offer', methods=['POST'])
async def offer(request):
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # if len(nerfreals) >= opt.max_session:
        #     logger.info('reach max session')
        #     return web.Response(
        #         content_type="application/json",
        #         text=json.dumps(
        #             {"code": -1, "msg": "reach max session"}
        #         ),
        #     )
        sessionid = randN(6) #len(nerfreals)
        nerfreals[sessionid] = None
        logger.info('sessionid=%d, session num=%d',sessionid,len(nerfreals))
        nerfreal = await build_nerfreal_async(sessionid)
        nerfreals[sessionid] = nerfreal
        
        #ice_server = RTCIceServer(urls='stun:stun.l.google.com:19302')
        ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await cancel_chat_task(sessionid)
                await pc.close()
                pcs.discard(pc)
                rag_session_ids.pop(sessionid, None)
                rag_session_confirmed.pop(sessionid, None)
                camera_customer_hints.pop(sessionid, None)
                camera_frame_cache.pop(sessionid, None)
                del nerfreals[sessionid]
            if pc.connectionState == "closed":
                await cancel_chat_task(sessionid)
                pcs.discard(pc)
                rag_session_ids.pop(sessionid, None)
                rag_session_confirmed.pop(sessionid, None)
                camera_customer_hints.pop(sessionid, None)
                camera_frame_cache.pop(sessionid, None)
                del nerfreals[sessionid]
                # gc.collect()

        player = HumanPlayer(nerfreals[sessionid])
        audio_sender = pc.addTrack(player.audio)
        video_sender = pc.addTrack(player.video)
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
            ),
        )
    except Exception as e:
        logger.exception('offer exception:')
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def human(request):
    try:
        params = await request.json()

        sessionid = int(params.get('sessionid',0))
        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            raise ValueError('LiveTalking session is not ready. Please start the avatar connection first.')
        incoming_rag_session_id = (params.get('rag_session_id') or '').strip()
        rag_session_id = incoming_rag_session_id
        image_bytes: Optional[bytes] = None
        image_content_type = "image/jpeg"
        image_base64 = (params.get("image_base64") or "").strip()
        referer = (request.headers.get("Referer") or "").strip()
        user_agent = (request.headers.get("User-Agent") or "").strip()
        if image_base64:
            try:
                image_bytes, image_content_type = _decode_image_data_url(image_base64)
                camera_frame_cache[sessionid] = (image_bytes, image_content_type)
                logger.info(
                    "UI image_base64 received session=%s bytes=%s content_type=%s",
                    sessionid,
                    len(image_bytes or b""),
                    image_content_type,
                )
            except Exception as decode_exc:
                logger.warning("invalid image_base64 from UI, fallback to local capture: %s", decode_exc)
        else:
            logger.warning(
                "UI image_base64 missing for session=%s referer=%s ua=%s; will use cache/local capture",
                sessionid,
                referer or "unknown",
                user_agent[:120] if user_agent else "unknown",
            )

        if not rag_session_id:
            rag_session_id = rag_session_ids.get(sessionid) or build_rag_session_id()
        if incoming_rag_session_id:
            rag_session_confirmed[sessionid] = True
        elif sessionid not in rag_session_confirmed:
            rag_session_confirmed[sessionid] = False
        rag_session_ids[sessionid] = rag_session_id
        raw_transcript = (params.get('raw_transcript') or params.get('text') or '').strip() or None
        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()
            await cancel_chat_task(sessionid)

        if params['type']=='echo':
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type']=='chat':
            chat_task = asyncio.create_task(
                run_rag_chat(
                    sessionid=sessionid,
                    rag_session_id=rag_session_id,
                    message=params['text'],
                    raw_transcript=raw_transcript,
                    image_bytes=image_bytes,
                    image_content_type=image_content_type,
                )
            )
            chat_tasks[sessionid] = chat_task

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok", "rag_session_id": rag_session_id}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def interrupt_talk(request):
    try:
        params = await request.json()

        sessionid = int(params.get('sessionid',0))
        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            raise ValueError('LiveTalking session is not ready.')
        nerfreals[sessionid].flush_talk()
        await cancel_chat_task(sessionid)
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def noble_config(request):
    return web.json_response(get_noble_runtime_config())


async def noble_session(request):
    return web.json_response({"session_id": build_rag_session_id()})


async def local_whisper_transcribe(request):
    try:
        form = await request.post()
        fileobj = form.get("audio")
        if fileobj is None:
            raise ValueError("audio is required")

        filebytes = fileobj.file.read()
        if not filebytes:
            raise ValueError("audio is empty")

        language = (form.get("language") or os.getenv("NOBLE_WHISPER_LANGUAGE") or "vi").strip()
        filename = fileobj.filename or "mic.webm"

        cfg = get_noble_runtime_config()
        whisper_base_url = (cfg.get("whisper_base_url") or "").rstrip("/")
        if not whisper_base_url:
            raise ValueError("NOBLE_WHISPER_API_URL is not configured")

        if len(filebytes) < 512:
            raise ValueError("audio is too short")

        timeout = aiohttp.ClientTimeout(total=180)
        retry_eof = max(0, int((os.getenv("NOBLE_WHISPER_EOF_RETRY") or "1").strip() or "1"))

        def _build_payload() -> aiohttp.FormData:
            payload = aiohttp.FormData()
            payload.add_field(
                "file",
                filebytes,
                filename=filename,
                content_type=fileobj.content_type or "audio/webm",
            )
            payload.add_field("language", language)
            payload.add_field("word_timestamps", "false")
            return payload

        last_error = ""
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(retry_eof + 1):
                async with session.post(f"{whisper_base_url}/transcribe", data=_build_payload()) as response:
                    body = await response.text()
                    if response.status >= 400:
                        last_error = f"whisper-service returned {response.status}: {body[:300]}"
                        eof_error = "end of file" in body.lower() or "eof" in body.lower()
                        if eof_error and attempt < retry_eof:
                            await asyncio.sleep(0.2)
                            continue
                        raise RuntimeError(last_error)

                    try:
                        data = json.loads(body) if body else {}
                    except Exception:
                        data = {}

                    text = str(
                        data.get("text")
                        or data.get("transcript")
                        or data.get("full_text")
                        or ""
                    ).strip()
                    if not text and body and not body.lstrip().startswith("{"):
                        text = body.strip()
                    return web.json_response({"text": text, "language": language})
        raise RuntimeError(last_error or "whisper-service returned empty response")
    except Exception as e:
        logger.exception("whisper-service proxy transcribe failed")
        return web.Response(
            status=502,
            content_type="application/json",
            text=json.dumps({"detail": str(e)}),
        )


async def on_startup_vision_smoke(_app):
    """Chụp 1 frame (OpenCV) rồi POST tới vision — chỉ khi bạn chủ động bật.

    - VISION_STARTUP_SMOKE=1 và NOBLE_VISION_SOURCE=camera: chạy smoke OpenCV.
    - NOBLE_VISION_SOURCE=browser (mặc định Noble): bỏ qua — ảnh lấy từ trình duyệt khi chat, không có webcam OpenCV lúc start.
    - Muốn vẫn thử OpenCV khi đang browser: VISION_STARTUP_SMOKE=1 và VISION_STARTUP_SMOKE_FORCE=1.
    """
    if not _env_flag("VISION_STARTUP_SMOKE", False):
        return
    vision_source = (os.getenv("NOBLE_VISION_SOURCE") or "browser").strip().lower()
    if vision_source == "browser" and not _env_flag("VISION_STARTUP_SMOKE_FORCE", False):
        logger.info(
            "VISION_STARTUP_SMOKE: skipped (NOBLE_VISION_SOURCE=browser). "
            "Set NOBLE_VISION_SOURCE=camera or VISION_STARTUP_SMOKE_FORCE=1 to run OpenCV smoke."
        )
        return
    cfg = get_noble_runtime_config()
    base = (cfg.get("vision_base_url") or "").strip().rstrip("/")
    if not base:
        logger.warning("VISION_STARTUP_SMOKE: skip — NOBLE_VISION_API_URL / vision_base_url empty")
        return
    rel = (os.getenv("VISION_STARTUP_PATH") or "vision/identify-and-context").strip().strip("/")
    url = f"{base}/{rel}"
    loop = asyncio.get_event_loop()
    try:
        jpeg = await loop.run_in_executor(None, _capture_for_vision_smoke_sync)
    except Exception as e:
        logger.warning("VISION_STARTUP_SMOKE: camera capture failed: %s", e)
        return
    session_id = f"livetalking_smoke_{uuid4().hex[:12]}"
    payload = aiohttp.FormData()
    payload.add_field("session_id", session_id)
    payload.add_field("source", "livetalking_startup")
    payload.add_field("channel", "camera-kiosk")
    payload.add_field("allow_create_unknown", "true")
    payload.add_field(
        "image",
        jpeg,
        filename="smoke.jpg",
        content_type="image/jpeg",
    )
    try:
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=payload) as resp:
                status = resp.status
                response_body = await resp.read()
        try:
            data = json.loads(response_body.decode("utf-8"))
            if isinstance(data, dict):
                vs = data.get("vision_summary")
                logger.info(
                    "VISION_STARTUP_SMOKE: POST %s http_status=%s session_id=%s top_keys=%s vision_summary_keys=%s",
                    url,
                    status,
                    session_id,
                    sorted(data.keys()),
                    sorted(vs.keys()) if isinstance(vs, dict) else None,
                )
            else:
                logger.info(
                    "VISION_STARTUP_SMOKE: POST %s http_status=%s session_id=%s non-object JSON",
                    url,
                    status,
                    session_id,
                )
        except Exception:
            logger.info(
                "VISION_STARTUP_SMOKE: POST %s http_status=%s session_id=%s body_prefix=%r",
                url,
                status,
                session_id,
                response_body[:400],
            )
    except Exception as e:
        logger.warning("VISION_STARTUP_SMOKE: POST %s failed: %s", url, e)


async def proxy_vision_identify(request):
    cfg = get_noble_runtime_config()
    try:
        form = await request.post()
        session_id = (form.get('session_id') or '').strip()
        source = (form.get('source') or cfg['vision_source']).strip() or cfg['vision_source']
        fileobj = form.get("image")

        if not session_id:
            raise ValueError("session_id is required")
        if fileobj is None:
            raise ValueError("image is required")

        filebytes = fileobj.file.read()
        if not filebytes:
            raise ValueError("image is empty")

        payload = aiohttp.FormData()
        payload.add_field("session_id", session_id)
        payload.add_field("source", source)
        payload.add_field(
            "image",
            filebytes,
            filename=fileobj.filename or "capture.jpg",
            content_type=fileobj.content_type or "image/jpeg",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{cfg['vision_base_url']}/vision/identify", data=payload) as response:
                response_body = await response.read()
                return web.Response(
                    status=response.status,
                    body=response_body,
                    content_type=response.content_type,
                    charset=response.charset,
                )
    except Exception as e:
        logger.exception('vision proxy exception:')
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"detail": str(e)}),
        )


async def proxy_vision_session(request):
    cfg = get_noble_runtime_config()
    session_id = request.match_info["session_id"].strip()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{cfg['vision_base_url']}/vision/session/{session_id}") as response:
                response_body = await response.read()
                return web.Response(
                    status=response.status,
                    body=response_body,
                    content_type=response.content_type,
                    charset=response.charset,
                )
    except Exception as e:
        logger.exception('vision session proxy exception:')
        return web.Response(
            status=502,
            content_type="application/json",
            text=json.dumps({"detail": str(e)}),
        )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def set_audiotype(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)    
        nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def record(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        if params['type']=='start_record':
            # nerfreals[sessionid].put_msg_txt(params['text'])
            nerfreals[sessionid].start_recording()
        elif params['type']=='end_record':
            nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def is_speaking(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    for sessionid in list(chat_tasks.keys()):
        await cancel_chat_task(sessionid)
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    rag_session_ids.clear()
    rag_session_confirmed.clear()
    camera_customer_hints.clear()
    camera_frame_cache.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                if response.status >= 400:
                    logger.error('Push endpoint returned HTTP %s: %s', response.status, await response.text())
                    return None
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')
        return None

async def run(push_url,sessionid):
    nerfreal = await build_nerfreal_async(sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    if not answer:
        logger.error('RTCPush unavailable for session %s at %s. Skip this push session.', sessionid, push_url)
        await pc.close()
        pcs.discard(pc)
        nerfreals.pop(sessionid, None)
        return False

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
    except Exception:
        logger.exception('Invalid RTCPush SDP answer for session %s at %s', sessionid, push_url)
        await pc.close()
        pcs.discard(pc)
        nerfreals.pop(sessionid, None)
        return False
    return True
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='half-avatar', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 azuretts
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    parser.add_argument('--idle_audiotype', type=int, default=0, help="custom audiotype used for idle playback")
    parser.add_argument('--idle_delay', type=float, default=0.0, help="seconds to wait before triggering idle playback")
    parser.add_argument('--enable_transition', action='store_true', help="enable silent/speaking transition blend")
    parser.add_argument('--transition_duration', type=float, default=0.1, help="transition blend duration in seconds")
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk') #musetalk wav2lip ultralight

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':       
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt) 
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    @web.middleware
    async def no_cache_frontend_middleware(request, handler):
        response = await handler(request)
        path = request.path.lower()
        if path.endswith(".html") or path.endswith(".js"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    appasync = web.Application(
        client_max_size=1024**2*100,
        middlewares=[no_cache_frontend_middleware],
    )
    appasync.on_startup.append(on_startup_vision_smoke)
    appasync.on_shutdown.append(on_shutdown)

    default_page = 'dashboard.html'
    if opt.transport == 'rtmp':
        default_page = 'echoapi.html'
    elif opt.transport == 'rtcpush':
        default_page = 'rtcpushapi.html'

    async def root_index(_request):
        raise web.HTTPFound('/' + default_page)

    appasync.router.add_get("/", root_index)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_get("/noble/config", noble_config)
    appasync.router.add_post("/noble/session", noble_session)
    appasync.router.add_post("/noble/whisper/transcribe", local_whisper_transcribe)
    appasync.router.add_post("/noble/vision/identify", proxy_vision_identify)
    appasync.router.add_get("/noble/vision/session/{session_id}", proxy_vision_session)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename = default_page
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        try:
            loop.run_until_complete(site.start())
        except OSError as e:
            winerr = getattr(e, "winerror", None)
            errno = getattr(e, "errno", None)
            if winerr == 10048 or errno == 98:
                logger.error(
                    "Port %s already in use — close the other LiveTalking (or any app on this port) "
                    "or set env LIVETALKING_PORT to another value.",
                    opt.listenport,
                )
            raise
        if opt.transport=='rtcpush':
            push_success = 0
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                ok = loop.run_until_complete(run(push_url,k))
                if ok:
                    push_success += 1
            if push_success == 0:
                logger.warning(
                    'No RTCPush session established. Keep server running for API/WebRTC; '
                    'check SRS/WHIP endpoint or switch --transport webrtc.'
                )
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    
