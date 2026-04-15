from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any


class RetrievalClient:
    def __init__(self, base_url: str, timeout_sec: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def retrieve(self, query: str, top_k: int) -> dict[str, Any]:
        payload = {"query": query, "top_k": top_k}
        parsed = self._post_json("/retrieve", payload)
        return {
            "results": parsed.get("results", []) or [],
            "confidence": parsed.get("confidence", 0.0),
            "low_confidence": bool(parsed.get("low_confidence", False)),
        }

    def retrieve_project_grounded(self, query: str, retrieval_intent: str | None, top_k: int) -> dict[str, Any]:
        payload = {
            "query": query,
            "retrieval_intent": retrieval_intent,
            "top_k": top_k,
        }
        parsed = self._post_json("/retrieve/project-grounded", payload)
        return {
            "project_cards": parsed.get("project_cards", []) or [],
            "trait_tags": parsed.get("trait_tags", []) or [],
            "evidence_chunks": parsed.get("evidence_chunks", []) or [],
            "confidence": parsed.get("confidence", 0.0),
            "low_confidence": bool(parsed.get("low_confidence", False)),
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"retrieval HTTP {exc.code}: {body[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"retrieval unreachable: {exc.reason}"
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"retrieval timeout after {self.timeout_sec}s"
            ) from exc
