###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
import os
import ipaddress
import re
import socket
import struct
import fcntl
from functools import lru_cache
from typing import Any, Dict, List, Optional
import queue
import aioice.ice as aioice_ice

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager

DEFAULT_STUN_URLS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
    "stun:stun.freeswitch.org:3478",
]

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

    def _bool_from_value(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def _is_local_or_private_client(self, remote_addr: Optional[str]) -> bool:
        host = (remote_addr or "").strip()
        if not host:
            return False
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if host.startswith("::ffff:"):
            host = host.split("::ffff:", 1)[1]
        if "%" in host:
            host = host.split("%", 1)[0]
        if host.lower() == "localhost":
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return bool(ip.is_loopback or ip.is_private or ip.is_link_local)

    def _truthy_env(self, name: str, default: str = "false") -> bool:
        return self._bool_from_value(os.getenv(name, default))

    def _env_present(self, name: str) -> bool:
        return os.getenv(name) is not None

    @lru_cache(maxsize=1)
    def _running_in_wsl(self) -> bool:
        try:
            with open("/proc/version", "r", encoding="utf-8", errors="ignore") as fh:
                return "microsoft" in fh.read().lower()
        except Exception:
            return False

    @lru_cache(maxsize=1)
    def _resolve_localhost_peer_ip(self) -> Optional[str]:
        override = (os.getenv("LIVETALKING_LOCALHOST_PEER_IP") or "").strip()
        candidates = [override] if override else []
        if self._running_in_wsl():
            gateway_ip, _ = self._wsl_default_route()
            if gateway_ip:
                candidates.append(gateway_ip)
        candidates.append("127.0.0.1")

        for value in candidates:
            if not value:
                continue
            try:
                ipaddress.ip_address(value)
                return value
            except ValueError:
                continue
        return None

    def _prefer_host_local(self, remote_addr: Optional[str]) -> bool:
        if self._env_present("LIVETALKING_PREFER_HOST_LOCAL"):
            return self._truthy_env("LIVETALKING_PREFER_HOST_LOCAL", "false")
        return self._running_in_wsl() and self._is_local_or_private_client(remote_addr)

    @lru_cache(maxsize=1)
    def _wsl_default_route(self) -> tuple[Optional[str], Optional[str]]:
        if not self._running_in_wsl():
            return None, None
        try:
            with open("/proc/net/route", "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            return None, None

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            iface, destination_hex, gateway_hex = parts[:3]
            if destination_hex != "00000000":
                continue
            try:
                gateway_ip = socket.inet_ntoa(struct.pack("<L", int(gateway_hex, 16)))
                return gateway_ip, iface
            except Exception:
                continue
        return None, None

    @lru_cache(maxsize=1)
    def _wsl_primary_ipv4(self) -> Optional[str]:
        override = (os.getenv("LIVETALKING_LOCALHOST_SERVER_IP") or "").strip()
        if override:
            try:
                ipaddress.ip_address(override)
                return override
            except ValueError:
                return None
        _, dev = self._wsl_default_route()
        if not dev:
            return None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                request = struct.pack("256s", dev[:15].encode("utf-8"))
                response = fcntl.ioctl(sock.fileno(), 0x8915, request)
            return socket.inet_ntoa(response[20:24])
        except Exception:
            return None

    def _split_csv(self, raw: Optional[str]) -> List[str]:
        if raw is None:
            return []
        return [part.strip() for part in str(raw).split(",") if part.strip()]

    def _normalize_ice_server_entry(self, entry: Any) -> Optional[dict]:
        if isinstance(entry, str):
            urls = self._split_csv(entry)
            if not urls:
                return None
            return {"urls": urls}

        if not isinstance(entry, dict):
            raise ValueError(f"Unsupported ICE server entry: {entry!r}")

        urls_value = entry.get("urls")
        if isinstance(urls_value, str):
            urls = self._split_csv(urls_value)
        elif isinstance(urls_value, list):
            urls = [str(url).strip() for url in urls_value if str(url).strip()]
        else:
            urls = []

        if not urls:
            return None

        normalized = {"urls": urls}
        for key in ("username", "credential", "credentialType"):
            value = entry.get(key)
            if value not in (None, ""):
                normalized[key] = str(value)
        return normalized

    def _normalize_ice_server_entries(self, raw_entries: Any) -> List[dict]:
        if raw_entries is None:
            return []
        if isinstance(raw_entries, dict) and "iceServers" in raw_entries:
            raw_entries = raw_entries["iceServers"]
        elif isinstance(raw_entries, dict):
            raw_entries = [raw_entries]
        elif isinstance(raw_entries, str):
            raw_entries = [raw_entries]

        if not isinstance(raw_entries, list):
            raise ValueError("ICE server config must be a list or object")

        entries = []
        for entry in raw_entries:
            normalized = self._normalize_ice_server_entry(entry)
            if normalized:
                entries.append(normalized)
        return entries

    def _ice_entry_has_turn(self, entry: dict) -> bool:
        return any(
            str(url).startswith(("turn:", "turns:"))
            for url in entry.get("urls", [])
        )

    def _log_ice_server_entries(self, remote_addr: Optional[str], entries: List[dict], source: str) -> None:
        if not entries:
            logger.info("ICE servers for remote=%s (%s): host candidates only", remote_addr, source)
            return

        parts = []
        for entry in entries:
            kind = "turn" if self._ice_entry_has_turn(entry) else "stun"
            urls = ",".join(entry.get("urls", []))
            auth = " auth" if entry.get("username") or entry.get("credential") else ""
            parts.append(f"{kind}[{urls}]{auth}")
        logger.info("ICE servers for remote=%s (%s): %s", remote_addr, source, "; ".join(parts))

    def _resolve_ice_server_entries(
        self,
        remote_addr: Optional[str] = None,
        enabled: bool = True,
    ) -> tuple[List[dict], str]:
        if not enabled:
            entries = []
            source = "disabled-by-request"
            self._log_ice_server_entries(remote_addr, entries, source)
            return entries, source

        raw_json = (os.getenv("LIVETALKING_ICE_SERVERS_JSON") or "").strip()
        if raw_json:
            try:
                entries = self._normalize_ice_server_entries(json.loads(raw_json))
                source = "LIVETALKING_ICE_SERVERS_JSON"
                self._log_ice_server_entries(remote_addr, entries, source)
                return entries, source
            except Exception as exc:
                logger.warning("Invalid LIVETALKING_ICE_SERVERS_JSON: %s", exc)

        raw_urls = (os.getenv("LIVETALKING_ICE_SERVERS") or "").strip()
        if raw_urls:
            if raw_urls.lower() in {"0", "false", "off", "none"}:
                entries = []
                source = "LIVETALKING_ICE_SERVERS=disabled"
                logger.info("ICE servers disabled by LIVETALKING_ICE_SERVERS=%s", raw_urls)
                self._log_ice_server_entries(remote_addr, entries, source)
                return entries, source
            entries = [{"urls": self._split_csv(raw_urls)}]
            source = "LIVETALKING_ICE_SERVERS"
            self._log_ice_server_entries(remote_addr, entries, source)
            return entries, source

        entries: List[dict] = []
        turn_urls = self._split_csv(os.getenv("LIVETALKING_TURN_URLS"))
        turn_username = (os.getenv("LIVETALKING_TURN_USERNAME") or "").strip()
        turn_password = (os.getenv("LIVETALKING_TURN_PASSWORD") or "").strip()
        if turn_urls:
            turn_entry = {"urls": turn_urls}
            if turn_username:
                turn_entry["username"] = turn_username
            if turn_password:
                turn_entry["credential"] = turn_password
            if not (turn_username and turn_password):
                logger.warning(
                    "TURN urls configured without both LIVETALKING_TURN_USERNAME and "
                    "LIVETALKING_TURN_PASSWORD. Relay authentication will likely fail."
                )
            entries.append(turn_entry)

        disable_stun = self._truthy_env("LIVETALKING_DISABLE_STUN", "false")
        prefer_host_local = self._prefer_host_local(remote_addr)
        stun_urls = self._split_csv(os.getenv("LIVETALKING_STUN_URLS"))
        if stun_urls:
            entries.append({"urls": stun_urls})
        elif not disable_stun and not (
            prefer_host_local and self._is_local_or_private_client(remote_addr)
        ):
            entries.append({"urls": list(DEFAULT_STUN_URLS)})

        source = "auto"
        self._log_ice_server_entries(remote_addr, entries, source)
        return entries, source

    def _entries_to_rtc_ice_servers(self, entries: List[dict]) -> List[RTCIceServer]:
        servers = []
        for entry in entries:
            kwargs = {"urls": entry.get("urls", [])}
            if entry.get("username"):
                kwargs["username"] = entry["username"]
            if entry.get("credential"):
                kwargs["credential"] = entry["credential"]
            servers.append(RTCIceServer(**kwargs))
        return servers

    def _sdp_candidate_summary(self, sdp: str) -> str:
        lines = [ln.strip() for ln in (sdp or "").splitlines() if ln.strip().startswith("a=candidate:")]
        if not lines:
            return "no-candidate"

        type_counts = {"host": 0, "srflx": 0, "relay": 0, "prflx": 0, "other": 0}
        mdns_count = 0
        hosts = []

        for ln in lines:
            m = re.search(r"\styp\s+(\w+)", ln)
            ctyp = m.group(1).lower() if m else "other"
            if ctyp in type_counts:
                type_counts[ctyp] += 1
            else:
                type_counts["other"] += 1

            parts = ln.split()
            # a=candidate:<foundation> <component> <transport> <priority> <ip> <port> typ ...
            if len(parts) >= 6:
                host = parts[4]
                hosts.append(host)
                if host.endswith(".local"):
                    mdns_count += 1

        uniq_hosts = []
        for h in hosts:
            if h not in uniq_hosts:
                uniq_hosts.append(h)
        preview_hosts = ",".join(uniq_hosts[:6])
        return (
            f"count={len(lines)} host={type_counts['host']} srflx={type_counts['srflx']} "
            f"relay={type_counts['relay']} prflx={type_counts['prflx']} other={type_counts['other']} "
            f"mdns={mdns_count} hosts=[{preview_hosts}]"
        )

    def _sdp_candidate_counts(self, sdp: str) -> dict:
        lines = [ln.strip() for ln in (sdp or "").splitlines() if ln.strip().startswith("a=candidate:")]
        counts = {"total": len(lines), "host": 0, "srflx": 0, "relay": 0, "prflx": 0, "other": 0, "mdns": 0}
        for ln in lines:
            m = re.search(r"\styp\s+(\w+)", ln)
            ctyp = m.group(1).lower() if m else "other"
            if ctyp in counts:
                counts[ctyp] += 1
            else:
                counts["other"] += 1
            parts = ln.split()
            if len(parts) >= 6 and parts[4].endswith(".local"):
                counts["mdns"] += 1
        return counts

    def _maybe_enable_loopback_candidate(self, remote_addr: Optional[str]) -> None:
        enable = self._truthy_env("LIVETALKING_ENABLE_LOOPBACK_CANDIDATE", "true")
        if not enable or not self._is_local_or_private_client(remote_addr):
            return
        if not (remote_addr or "").strip().startswith(("127.", "::1", "[::1]")):
            return
        if self._running_in_wsl():
            logger.info(
                "Skipping loopback ICE candidate patch for localhost client in WSL; "
                "WSL uses primary interface candidates instead."
            )
            return
        if getattr(aioice_ice, "_livetalking_loopback_patch", False):
            return

        orig_get_host_addresses = aioice_ice.get_host_addresses

        def _patched_get_host_addresses(use_ipv4: bool, use_ipv6: bool):
            addrs = orig_get_host_addresses(use_ipv4, use_ipv6)
            if use_ipv4 and "127.0.0.1" not in addrs:
                addrs.append("127.0.0.1")
            return addrs

        aioice_ice.get_host_addresses = _patched_get_host_addresses
        aioice_ice._livetalking_loopback_patch = True
        logger.info("Enabled loopback ICE candidate patch for localhost clients.")

    def _maybe_rewrite_localhost_mdns_offer(self, sdp: str, remote_addr: Optional[str]) -> tuple[str, int]:
        enable = self._truthy_env("LIVETALKING_REWRITE_LOCAL_MDNS_OFFER", "true")
        if not enable:
            return sdp, 0
        host = (remote_addr or "").strip()
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if host.startswith("::ffff:"):
            host = host.split("::ffff:", 1)[1]
        if host not in {"127.0.0.1", "::1", "[::1]", "localhost"}:
            return sdp, 0
        rewrite_ip = self._resolve_localhost_peer_ip()
        if not rewrite_ip:
            return sdp, 0

        rewritten = []
        replacements = 0
        for raw_line in (sdp or "").splitlines():
            line = raw_line.rstrip("\r")
            if not line.startswith("a=candidate:"):
                rewritten.append(line)
                continue

            parts = line.split()
            if len(parts) < 8:
                rewritten.append(line)
                continue

            candidate_host = parts[4]
            candidate_type = ""
            for idx in range(6, len(parts) - 1):
                if parts[idx] == "typ":
                    candidate_type = parts[idx + 1].lower()
                    break

            if candidate_type == "host" and candidate_host.endswith(".local"):
                parts[4] = rewrite_ip
                line = " ".join(parts)
                replacements += 1
            rewritten.append(line)

        if replacements:
            logger.info(
                "Rewrote %s localhost mDNS host candidates in remote offer to %s.",
                replacements,
                rewrite_ip,
            )
        return "\r\n".join(rewritten) + "\r\n", replacements

    def _maybe_filter_wsl_local_answer(self, sdp: str, remote_addr: Optional[str]) -> tuple[str, int]:
        if not self._running_in_wsl():
            return sdp, 0
        host = (remote_addr or "").strip()
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if host.startswith("::ffff:"):
            host = host.split("::ffff:", 1)[1]
        if host not in {"127.0.0.1", "::1", "[::1]", "localhost"}:
            return sdp, 0

        primary_ip = self._wsl_primary_ipv4()
        if not primary_ip:
            return sdp, 0

        filtered = []
        removed = 0
        for raw_line in (sdp or "").splitlines():
            line = raw_line.rstrip("\r")
            if line.startswith("c=IN IP4 "):
                filtered.append(f"c=IN IP4 {primary_ip}")
                continue
            if not line.startswith("a=candidate:"):
                filtered.append(line)
                continue

            parts = line.split()
            if len(parts) < 8:
                filtered.append(line)
                continue

            candidate_host = parts[4]
            candidate_type = ""
            for idx in range(6, len(parts) - 1):
                if parts[idx] == "typ":
                    candidate_type = parts[idx + 1].lower()
                    break

            if candidate_type == "host" and candidate_host != primary_ip:
                removed += 1
                continue

            filtered.append(line)

        if removed:
            logger.info(
                "Filtered %s WSL localhost host candidates from local answer; keeping primary interface %s.",
                removed,
                primary_ip,
            )
        return "\r\n".join(filtered) + "\r\n", removed

    async def handle_ice_config(self, request):
        enabled = self._bool_from_value(request.query.get("enabled"), True)
        entries, source = self._resolve_ice_server_entries(request.remote, enabled=enabled)
        return web.json_response(
            {
                "enabled": enabled,
                "source": source,
                "hasTurn": any(self._ice_entry_has_turn(entry) for entry in entries),
                "iceServers": entries,
            }
        )

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        params = await request.json()
        raw_offer_sdp = params["sdp"]
        use_ice_servers = self._bool_from_value(params.get("useIceServers"), True)
        self._maybe_enable_loopback_candidate(request.remote)
        logger.info("Remote offer candidate summary: %s", self._sdp_candidate_summary(raw_offer_sdp))
        raw_offer_counts = self._sdp_candidate_counts(raw_offer_sdp)
        patched_offer_sdp, mdns_rewrites = self._maybe_rewrite_localhost_mdns_offer(raw_offer_sdp, request.remote)
        if mdns_rewrites:
            logger.info("Patched remote offer candidate summary: %s", self._sdp_candidate_summary(patched_offer_sdp))
        offer = RTCSessionDescription(sdp=patched_offer_sdp, type=params["type"])
        offer_counts = self._sdp_candidate_counts(offer.sdp)
        ice_entries, _ = self._resolve_ice_server_entries(request.remote, enabled=use_ice_servers)
        if (
            raw_offer_counts["mdns"] > 0
            and raw_offer_counts["srflx"] == 0
            and raw_offer_counts["relay"] == 0
            and not any(self._ice_entry_has_turn(entry) for entry in ice_entries)
        ):
            logger.warning(
                "Remote offer has mDNS host candidates only (mdns=%s). "
                "Applied localhost rewrite=%s. This environment often fails without relay candidates "
                "unless the browser mDNS candidates are rewritten or browser mDNS is disabled.",
                raw_offer_counts["mdns"],
                "yes" if mdns_rewrites else "no",
            )

        if False: # 不再由 RTCManager 控制 max_session，让业务逻辑或SessionManager 控制
            logger.info('reach max session')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "reach max session"}),
            )

        #sessionid = _rand_session_id()

        # 通过 SessionManager 构建
        sessionid = await session_manager.create_session(params)
        logger.info('offer sessionid=%d', sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 创建 PeerConnection
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=self._entries_to_rtc_ice_servers(ice_entries)
            )
        )
        self.pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info("ICE connection state is %s", pc.iceConnectionState)

        connect_timeout_sec = float((os.getenv("LIVETALKING_CONNECT_TIMEOUT_SEC") or "0").strip())
        session_cleaned = False
        connect_guard_task = None

        async def _cleanup_session(reason: str):
            nonlocal session_cleaned, connect_guard_task
            if session_cleaned:
                return
            session_cleaned = True
            if connect_guard_task and not connect_guard_task.done():
                connect_guard_task.cancel()
            try:
                await pc.close()
            except Exception:
                pass
            self.pcs.discard(pc)
            session_manager.remove_session(sessionid, reason=reason)
            logger.info("WebRTC session %s cleaned, reason=%s", sessionid, reason)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            nonlocal connect_guard_task
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "connected":
                if connect_guard_task and not connect_guard_task.done():
                    connect_guard_task.cancel()
            if pc.connectionState in ("failed", "closed"):
                await _cleanup_session(pc.connectionState)

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
        # Select the video transceiver explicitly to avoid index/order mismatch.
        for transceiver in pc.getTransceivers():
            sender = getattr(transceiver, "sender", None)
            track = getattr(sender, "track", None) if sender else None
            if track and track.kind == "video":
                transceiver.setCodecPreferences(preferences)
                break

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        logger.info("Local answer candidate summary: %s", self._sdp_candidate_summary(pc.localDescription.sdp))
        response_sdp, filtered_local_candidates = self._maybe_filter_wsl_local_answer(
            pc.localDescription.sdp,
            request.remote,
        )
        if filtered_local_candidates:
            logger.info("Filtered local answer candidate summary: %s", self._sdp_candidate_summary(response_sdp))
        else:
            response_sdp = pc.localDescription.sdp

        async def _connect_guard():
            try:
                await asyncio.sleep(connect_timeout_sec)
                if pc.connectionState != "connected":
                    logger.warning(
                        "WebRTC connect timeout: sessionid=%s state=%s timeout=%.1fs",
                        sessionid,
                        pc.connectionState,
                        connect_timeout_sec,
                    )
                    await _cleanup_session("connect-timeout")
            except asyncio.CancelledError:
                return

        if connect_timeout_sec > 0:
            connect_guard_task = asyncio.create_task(_connect_guard())

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": response_sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid):
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
        session_manager.flush_all_sessions(reason="shutdown")
