from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any


class VisionClient:
    def __init__(self, base_url: str, timeout_sec: float = 12.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def identify(
        self,
        image_base64: str,
        image_filename: str | None = None,
        image_content_type: str | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "image_base64": image_base64,
            "image_filename": image_filename,
            "image_content_type": image_content_type,
        }
        return self._post_json(
            "/vision/identify",
            payload,
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
        )

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if trace_id:
            headers["X-Langfuse-Trace-Id"] = str(trace_id)
        if session_id:
            headers["X-Langfuse-Session-Id"] = str(session_id)
        if user_id:
            headers["X-Langfuse-User-Id"] = str(user_id)
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"vision HTTP {exc.code}: {body[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"vision unreachable: {exc.reason}"
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"vision timeout after {self.timeout_sec}s"
            ) from exc
