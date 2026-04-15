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
    reason = payload.get("decision_reason")
    has_state = bool(payload.get("lead_state"))
    print(f"[{label}] status={status} route={route} has_state={has_state} reason={reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E smoke test for 2-route orchestrator flow")
    parser.add_argument("--base-url", default="http://127.0.0.1:8021", help="Orchestrator base URL")
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/sales/query"
    base_state = {
        "name": None,
        "phone_contact": None,
        "need": {"summary": "", "topics": [], "evidence": [], "last_updated_at": None},
        "painpoint": {"summary": "", "topics": [], "evidence": [], "last_updated_at": None},
    }
    cases = [
        ("CONSULT", {"message": "Mua de dau tu thi nen bat dau tu dau?", "lead_state": base_state}),
        ("PROJECT", {"message": "Co can nao gan truong hoc va benh vien?", "lead_state": base_state}),
        ("FORCE_PROJECT", {"message": "Cho minh thong tin phap ly", "force_route": "project_grounded", "lead_state": base_state}),
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
            elif data.get("route") not in {"consult_discovery", "project_grounded"}:
                failed += 1

    print("-" * 72)
    if failed:
        print(f"SMOKE FAIL failed_cases={failed}/{len(cases)}")
        return 2
    print("SMOKE PASS all cases returned expected contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
