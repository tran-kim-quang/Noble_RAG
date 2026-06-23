###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import os
import ipaddress
from typing import Dict, Optional

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager


def _request_remote_ip(request: web.Request) -> str:
    forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return (request.remote or "").strip()


def _is_loopback_ip(value: str) -> bool:
    if not value:
        return False
    if value in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _is_literal_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _guess_localhost_peer_ip() -> str:
    env_ip = (os.getenv("LIVETALKING_LOCALHOST_PEER_IP") or "").strip()
    if env_ip:
        return env_ip
    # In WSL, resolv.conf nameserver usually points to Windows host IP.
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("nameserver "):
                    continue
                ip = line.split(" ", 1)[1].strip()
                if ip and not _is_loopback_ip(ip):
                    return ip
    except OSError:
        pass
    return "127.0.0.1"


def _candidate_summary(sdp: str) -> dict:
    summary = {
        "count": 0,
        "host": 0,
        "srflx": 0,
        "relay": 0,
        "prflx": 0,
        "other": 0,
        "mdns": 0,
        "hosts": set(),
    }
    for raw_line in (sdp or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("a=candidate:"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        summary["count"] += 1
        addr = parts[4]
        if addr.endswith(".local"):
            summary["mdns"] += 1
        summary["hosts"].add(addr)
        try:
            typ_idx = parts.index("typ")
            cand_type = parts[typ_idx + 1]
        except (ValueError, IndexError):
            cand_type = "other"
        if cand_type in ("host", "srflx", "relay", "prflx"):
            summary[cand_type] += 1
        else:
            summary["other"] += 1
    summary["hosts"] = sorted(summary["hosts"])
    return summary


def _candidate_summary_text(summary: dict) -> str:
    return (
        f"count={summary['count']} host={summary['host']} srflx={summary['srflx']} "
        f"relay={summary['relay']} prflx={summary['prflx']} other={summary['other']} "
        f"mdns={summary['mdns']} hosts=[{','.join(summary['hosts'][:8])}]"
    )


def _rewrite_localhost_mdns_candidates(sdp: str, replacement_ip: str) -> tuple[str, int]:
    sep = "\r\n" if "\r\n" in sdp else "\n"
    lines = []
    replaced = 0
    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if line.startswith("a=candidate:"):
            parts = line.split()
            if len(parts) >= 8:
                try:
                    typ_idx = parts.index("typ")
                    cand_type = parts[typ_idx + 1]
                except (ValueError, IndexError):
                    cand_type = ""
                addr = parts[4]
                if cand_type == "host" and addr.endswith(".local"):
                    parts[4] = replacement_ip
                    raw_line = " ".join(parts)
                    replaced += 1
        lines.append(raw_line)
    rewritten = sep.join(lines)
    if sdp.endswith("\r\n") and not rewritten.endswith("\r\n"):
        rewritten += "\r\n"
    if sdp.endswith("\n") and not sdp.endswith("\r\n") and not rewritten.endswith("\n"):
        rewritten += "\n"
    return rewritten, replaced


def _parse_stun_servers() -> list[str]:
    raw = (os.getenv("LIVETALKING_STUN_SERVERS") or "").strip()
    if not raw:
        raw = "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302,stun:stun.freeswitch.org:3478"
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_turn_urls() -> list[str]:
    raw = (os.getenv("LIVETALKING_TURN_URLS") or "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_ice_servers(host_only: bool) -> tuple[list[RTCIceServer], str]:
    if host_only:
        return [], "host candidates only"

    ice_servers: list[RTCIceServer] = []
    desc_parts: list[str] = []

    stun_servers = _parse_stun_servers()
    if stun_servers:
        ice_servers.append(RTCIceServer(urls=stun_servers))
        desc_parts.append(f"stun[{','.join(stun_servers)}]")

    turn_urls = _parse_turn_urls()
    turn_user = (os.getenv("LIVETALKING_TURN_USERNAME") or "").strip()
    turn_cred = (os.getenv("LIVETALKING_TURN_CREDENTIAL") or "").strip()
    if turn_urls:
        if turn_user and turn_cred:
            ice_servers.append(RTCIceServer(urls=turn_urls, username=turn_user, credential=turn_cred))
            desc_parts.append(f"turn[{','.join(turn_urls)}] user={turn_user}")
        else:
            ice_servers.append(RTCIceServer(urls=turn_urls))
            desc_parts.append(f"turn[{','.join(turn_urls)}] (no-credentials)")

    if not ice_servers:
        return [], "host candidates only"
    return ice_servers, " ".join(desc_parts)


class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        params = await request.json()
        remote_ip = _request_remote_ip(request)
        remote_sdp = params["sdp"]

        remote_summary = _candidate_summary(remote_sdp)
        logger.info("Remote offer candidate summary: %s", _candidate_summary_text(remote_summary))

        patch_env_raw = (os.getenv("LIVETALKING_PATCH_LOCALHOST_MDNS") or "1").strip().lower()
        enable_patch = patch_env_raw not in ("0", "false", "no")
        if (not enable_patch) and _is_loopback_ip(remote_ip) and remote_summary["mdns"] > 0 and remote_summary["srflx"] == 0 and remote_summary["relay"] == 0:
            enable_patch = True
            logger.warning(
                "LIVETALKING_PATCH_LOCALHOST_MDNS is disabled, but remote offer is localhost+mDNS-only. "
                "Enabling rewrite automatically for this session."
            )
        if enable_patch and remote_summary["mdns"] > 0:
            rewrite_ip = ""
            if _is_loopback_ip(remote_ip):
                rewrite_ip = _guess_localhost_peer_ip()
            elif _is_literal_ip(remote_ip):
                rewrite_ip = remote_ip
            if rewrite_ip:
                patched_sdp, replaced = _rewrite_localhost_mdns_candidates(remote_sdp, rewrite_ip)
                if replaced > 0:
                    logger.info("Rewrote %d localhost mDNS host candidates in remote offer to %s.", replaced, rewrite_ip)
                    remote_sdp = patched_sdp
                    patched_summary = _candidate_summary(remote_sdp)
                    logger.info("Patched remote offer candidate summary: %s", _candidate_summary_text(patched_summary))
                    remote_summary = patched_summary

        force_host_ice = (os.getenv("LIVETALKING_FORCE_HOST_ICE") or "auto").strip().lower()
        host_only = False
        ice_mode = "auto"
        if force_host_ice in ("1", "true", "yes"):
            host_only = True
            ice_mode = "forced"
        elif force_host_ice in ("0", "false", "no"):
            host_only = False
            ice_mode = "disabled-by-request"
        elif _is_loopback_ip(remote_ip) and remote_summary["mdns"] > 0 and remote_summary["srflx"] == 0 and remote_summary["relay"] == 0:
            host_only = True
            ice_mode = "auto"

        ice_servers, ice_desc = _build_ice_servers(host_only=host_only)
        logger.info("ICE servers for remote=%s (%s): %s", remote_ip or "unknown", ice_mode, ice_desc)

        if remote_summary["mdns"] > 0 and remote_summary["srflx"] == 0 and remote_summary["relay"] == 0:
            logger.warning(
                "Remote offer has mDNS host candidates only (mdns=%d). "
                "This environment often fails unless mDNS candidates are rewritten or browser mDNS is disabled.",
                remote_summary["mdns"],
            )

        offer = RTCSessionDescription(sdp=remote_sdp, type=params["type"])

        if False: # 不再由 RTCManager 控制 max_session，让业务逻辑或SessionManager 控制
            logger.info('reach max session')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "reach max session"}),
            )

        #sessionid = _rand_session_id()

        # 通过 SessionManager 构建
        sessionid = await session_manager.create_session(params)
        logger.info('offer sessionid=%s', sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 创建 PeerConnection
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        self.pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info("ICE connection state is %s", pc.iceConnectionState)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid, reason=pc.connectionState)

        # 添加发送轨道
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 设置编解码器偏好
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        local_summary = _candidate_summary(pc.localDescription.sdp or "")
        logger.info("Local answer candidate summary: %s", _candidate_summary_text(local_summary))

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
