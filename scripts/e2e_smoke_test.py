#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = int(resp.status)
            parsed = json.loads(resp.read().decode("utf-8"))
            return code, parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body}
        return int(exc.code), parsed


def print_case(label: str, status: int, payload: dict[str, Any]) -> None:
    route = payload.get("route")
    action = payload.get("action")
    reason = payload.get("decision_reason")
    print(f"[{label}] status={status} route={route} action={action} reason={reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal E2E smoke test for orchestrator flow")
    parser.add_argument("--base-url", default="http://127.0.0.1:8021", help="Orchestrator base URL")
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/sales/query"

    cases = [
        ("NO_RETRIEVAL", {"message": "xin chao", "need_retrieval": False}),
        ("CONFIDENT", {"message": "gia can 2 phong ngu", "need_retrieval": True, "top_k": 5}),
        ("LOW_CONF", {"message": "du an co san truot tuyet trong nha khong", "need_retrieval": True}),
        ("AUTO_HEUR", {"message": "phap ly du an hien tai"}),
        ("BLANK", {"message": "   "}),
    ]

    failed = 0
    for label, payload in cases:
        status, data = post_json(url, payload)
        print_case(label, status, data)

        if label == "BLANK":
            if status != 422:
                failed += 1
        else:
            if status != 200:
                failed += 1

    print("-" * 72)
    if failed:
        print(f"SMOKE FAIL failed_cases={failed}/{len(cases)}")
        return 2
    print("SMOKE PASS all cases returned expected HTTP status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
