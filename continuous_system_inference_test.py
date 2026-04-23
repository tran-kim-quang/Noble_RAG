#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _configure_stdio_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _default_lead_state() -> dict[str, Any]:
    return {
        "name": None,
        "phone_contact": None,
        "need": {"summary": "", "topics": [], "evidence": [], "last_updated_at": None},
        "painpoint": {"summary": "", "topics": [], "evidence": [], "last_updated_at": None},
    }


def _http_json(url: str, payload: dict[str, Any] | None, timeout_sec: float) -> tuple[int, dict[str, Any], str | None]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return int(resp.status), parsed, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        return int(exc.code), parsed, f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return 0, {}, f"url_error:{exc.reason}"
    except TimeoutError:
        return 0, {}, "timeout"
    except Exception as exc:
        return 0, {}, f"error:{exc}"


def _load_cases(cases_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases file must contain non-empty `cases` list")
    normalized: list[dict[str, Any]] = []
    for item in cases:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        normalized.append(item)
    if not normalized:
        raise ValueError("cases list has no valid `query` values")
    return normalized


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    rank = (len(values) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


@dataclass
class SessionState:
    lead_state: dict[str, Any]
    recent_history: list[dict[str, str]]
    session_id: str


class RollingMetrics:
    def __init__(self) -> None:
        self.total = 0
        self.success = 0
        self.errors = 0
        self.http_non_200 = 0
        self.timeouts = 0
        self.latencies_ms: list[float] = []
        self.route_counts: dict[str, int] = {}
        self.low_conf_project = 0
        self.project_total = 0
        self.contract_pass = 0
        self.contract_fail = 0
        self.error_samples: list[str] = []

    def add_result(
        self,
        status: int,
        latency_ms: float,
        route: str | None,
        error: str | None,
        response: dict[str, Any],
        contract_ok: bool,
    ) -> None:
        self.total += 1
        self.latencies_ms.append(latency_ms)
        if status == 200 and error is None:
            self.success += 1
        else:
            self.errors += 1
        if status not in (0, 200):
            self.http_non_200 += 1
        if error == "timeout":
            self.timeouts += 1
        if error and len(self.error_samples) < 10:
            self.error_samples.append(error)

        route_key = (route or "unknown").strip() or "unknown"
        self.route_counts[route_key] = self.route_counts.get(route_key, 0) + 1
        if route_key == "project_grounded":
            self.project_total += 1
            payload = response.get("project_grounded_payload") or {}
            if bool(payload.get("low_confidence", False)):
                self.low_conf_project += 1

        if contract_ok:
            self.contract_pass += 1
        else:
            self.contract_fail += 1

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)
        avg = statistics.mean(lat) if lat else 0.0
        return {
            "total": self.total,
            "success": self.success,
            "errors": self.errors,
            "http_non_200": self.http_non_200,
            "timeouts": self.timeouts,
            "success_rate": (self.success / self.total) if self.total else 0.0,
            "latency_ms": {
                "avg": round(avg, 2),
                "p50": round(_percentile(lat, 50), 2),
                "p95": round(_percentile(lat, 95), 2),
                "p99": round(_percentile(lat, 99), 2),
                "max": round(max(lat), 2) if lat else 0.0,
            },
            "route_counts": self.route_counts,
            "project_low_conf_rate": (self.low_conf_project / self.project_total) if self.project_total else 0.0,
            "contract_pass": self.contract_pass,
            "contract_fail": self.contract_fail,
            "error_samples": self.error_samples,
        }


def _contract_check(response: dict[str, Any]) -> tuple[bool, str]:
    route = response.get("route")
    if route not in {"consult_discovery", "project_grounded"}:
        return False, "invalid_route"
    if not str(response.get("assistant_reply", "")).strip():
        return False, "empty_assistant_reply"
    lead_state = response.get("lead_state")
    if not isinstance(lead_state, dict):
        return False, "missing_lead_state"
    if not isinstance(response.get("need_update"), dict):
        return False, "missing_need_update"
    if not isinstance(response.get("painpoint_update"), dict):
        return False, "missing_painpoint_update"
    if route == "project_grounded":
        payload = response.get("project_grounded_payload")
        if not isinstance(payload, dict):
            return False, "missing_project_payload"
    return True, "ok"


def run_continuous_inference_test(
    orchestrator_base_url: str,
    retrieval_base_url: str,
    cases_path: Path,
    report_jsonl_path: Path,
    max_requests: int | None,
    interval_sec: float,
    timeout_sec: float,
    force_project_every: int,
    session_count: int,
    summary_every: int,
    check_retrieval_health: bool,
) -> int:
    cases = _load_cases(cases_path)
    random.shuffle(cases)

    orch_health_url = orchestrator_base_url.rstrip("/") + "/health"
    orch_query_url = orchestrator_base_url.rstrip("/") + "/sales/query"
    retrieval_health_url = retrieval_base_url.rstrip("/") + "/health"

    sessions = [
        SessionState(
            lead_state=_default_lead_state(),
            recent_history=[],
            session_id=f"continuous-session-{idx + 1}",
        )
        for idx in range(max(1, session_count))
    ]
    metrics = RollingMetrics()
    sent = 0

    report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with report_jsonl_path.open("a", encoding="utf-8") as report_file:
        print(f"[{_now_iso()}] start continuous inference test")
        print(f"orchestrator={orchestrator_base_url} retrieval={retrieval_base_url} cases={len(cases)}")
        print(f"report={report_jsonl_path} max_requests={max_requests} interval_sec={interval_sec}")

        try:
            while max_requests is None or sent < max_requests:
                idx = sent % len(cases)
                case = cases[idx]
                message = str(case.get("query", "")).strip()
                sess = sessions[sent % len(sessions)]

                payload: dict[str, Any] = {
                    "message": message,
                    "lead_state": sess.lead_state,
                    "recent_history": sess.recent_history[-8:],
                    "session_id": sess.session_id,
                }
                if force_project_every > 0 and (sent + 1) % force_project_every == 0:
                    payload["force_route"] = "project_grounded"

                orch_health_status, orch_health, orch_health_err = _http_json(orch_health_url, None, timeout_sec)
                ret_health_status, ret_health, ret_health_err = _http_json(retrieval_health_url, None, timeout_sec)
                orch_health_ok = orch_health_status == 200 and str(orch_health.get("status", "")).lower() == "ok"
                if check_retrieval_health:
                    ret_health_ok = ret_health_status == 200 and str(ret_health.get("status", "")).lower() == "ok"
                    health_ok = orch_health_ok and ret_health_ok
                else:
                    health_ok = orch_health_ok

                t0 = time.perf_counter()
                status, response, error = _http_json(orch_query_url, payload, timeout_sec)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                route = response.get("route") if isinstance(response, dict) else None
                contract_ok, contract_reason = _contract_check(response if isinstance(response, dict) else {})
                if not contract_ok and error is None:
                    error = f"contract:{contract_reason}"

                metrics.add_result(
                    status=status,
                    latency_ms=latency_ms,
                    route=str(route) if route is not None else None,
                    error=error,
                    response=response if isinstance(response, dict) else {},
                    contract_ok=contract_ok,
                )

                if status == 200 and isinstance(response, dict):
                    sess.lead_state = response.get("lead_state") or sess.lead_state
                    sess.recent_history.append({"role": "user", "message": message})
                    reply = str(response.get("assistant_reply", "")).strip()
                    if reply:
                        sess.recent_history.append({"role": "assistant", "message": reply})
                    sess.recent_history = sess.recent_history[-12:]

                record = {
                    "ts": _now_iso(),
                    "idx": sent + 1,
                    "query": message,
                    "status": status,
                    "latency_ms": round(latency_ms, 2),
                    "route": route,
                    "error": error,
                    "contract_ok": contract_ok,
                    "contract_reason": contract_reason,
                    "health_ok": health_ok,
                    "orchestrator_health": {"status": orch_health_status, "payload": orch_health, "error": orch_health_err},
                    "retrieval_health": {"status": ret_health_status, "payload": ret_health, "error": ret_health_err},
                }
                report_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                report_file.flush()

                print(
                    f"[{sent + 1:05d}] status={status} route={route} latency={latency_ms:.1f}ms "
                    f"health_ok={health_ok} contract_ok={contract_ok} error={error or '-'} q='{message[:70]}'"
                )

                sent += 1
                if summary_every > 0 and sent % summary_every == 0:
                    summary = metrics.summary()
                    print("-" * 90)
                    print(f"[{_now_iso()}] rolling_summary {json.dumps(summary, ensure_ascii=False)}")
                    print("-" * 90)

                if interval_sec > 0:
                    time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Writing final summary...")

    final = metrics.summary()
    print("=" * 90)
    print(f"[{_now_iso()}] final_summary {json.dumps(final, ensure_ascii=False)}")
    print(f"jsonl_report={report_jsonl_path}")
    print("=" * 90)
    return 0 if metrics.total > 0 else 2


def run_interactive_chat_test(
    orchestrator_base_url: str,
    retrieval_base_url: str,
    report_jsonl_path: Path,
    timeout_sec: float,
    check_retrieval_health: bool,
    session_id: str,
    show_raw_response: bool,
) -> int:
    orch_health_url = orchestrator_base_url.rstrip("/") + "/health"
    orch_query_url = orchestrator_base_url.rstrip("/") + "/sales/query"
    retrieval_health_url = retrieval_base_url.rstrip("/") + "/health"

    sess = SessionState(
        lead_state=_default_lead_state(),
        recent_history=[],
        session_id=str(session_id).strip() or "manual-chat-01",
    )
    metrics = RollingMetrics()
    sent = 0

    report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with report_jsonl_path.open("a", encoding="utf-8") as report_file:
        print(f"[{_now_iso()}] start interactive chat test")
        print(f"session_id={sess.session_id} orchestrator={orchestrator_base_url} retrieval={retrieval_base_url}")
        print(f"report={report_jsonl_path}")
        print("commands: /exit | /summary | /state | /project <message>")
        print("-" * 90)

        while True:
            try:
                raw = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nInterrupted by user. Finishing interactive session...")
                break

            if not raw:
                continue
            if raw in {"/exit", "/quit"}:
                break
            if raw == "/summary":
                print(json.dumps(metrics.summary(), ensure_ascii=False, indent=2))
                continue
            if raw == "/state":
                print(json.dumps(sess.lead_state, ensure_ascii=False, indent=2))
                continue

            force_project = False
            message = raw
            if raw.startswith("/project "):
                force_project = True
                message = raw[len("/project ") :].strip()
                if not message:
                    print("Please provide a message after /project.")
                    continue

            payload: dict[str, Any] = {
                "message": message,
                "lead_state": sess.lead_state,
                "recent_history": sess.recent_history[-8:],
                "session_id": sess.session_id,
            }
            if force_project:
                payload["force_route"] = "project_grounded"

            orch_health_status, orch_health, orch_health_err = _http_json(orch_health_url, None, timeout_sec)
            ret_health_status, ret_health, ret_health_err = _http_json(retrieval_health_url, None, timeout_sec)
            orch_health_ok = orch_health_status == 200 and str(orch_health.get("status", "")).lower() == "ok"
            if check_retrieval_health:
                ret_health_ok = ret_health_status == 200 and str(ret_health.get("status", "")).lower() == "ok"
                health_ok = orch_health_ok and ret_health_ok
            else:
                health_ok = orch_health_ok

            t0 = time.perf_counter()
            status, response, error = _http_json(orch_query_url, payload, timeout_sec)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            route = response.get("route") if isinstance(response, dict) else None
            contract_ok, contract_reason = _contract_check(response if isinstance(response, dict) else {})
            if not contract_ok and error is None:
                error = f"contract:{contract_reason}"

            metrics.add_result(
                status=status,
                latency_ms=latency_ms,
                route=str(route) if route is not None else None,
                error=error,
                response=response if isinstance(response, dict) else {},
                contract_ok=contract_ok,
            )

            assistant_reply = ""
            if status == 200 and isinstance(response, dict):
                sess.lead_state = response.get("lead_state") or sess.lead_state
                assistant_reply = str(response.get("assistant_reply", "")).strip()
                sess.recent_history.append({"role": "user", "message": message})
                if assistant_reply:
                    sess.recent_history.append({"role": "assistant", "message": assistant_reply})
                sess.recent_history = sess.recent_history[-12:]

            sent += 1
            record = {
                "ts": _now_iso(),
                "mode": "interactive",
                "idx": sent,
                "session_id": sess.session_id,
                "query": message,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "route": route,
                "error": error,
                "contract_ok": contract_ok,
                "contract_reason": contract_reason,
                "health_ok": health_ok,
                "orchestrator_health": {"status": orch_health_status, "payload": orch_health, "error": orch_health_err},
                "retrieval_health": {"status": ret_health_status, "payload": ret_health, "error": ret_health_err},
                "assistant_reply": assistant_reply,
                "lead_state": sess.lead_state,
            }
            if show_raw_response:
                record["response"] = response

            report_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            report_file.flush()

            print(
                f"[{sent:05d}] status={status} route={route} latency={latency_ms:.1f}ms "
                f"health_ok={health_ok} contract_ok={contract_ok} error={error or '-'}"
            )
            if assistant_reply:
                print(f"Assistant> {assistant_reply}")
            else:
                print("Assistant> <empty response>")
            print("-" * 90)

    final = metrics.summary()
    print("=" * 90)
    print(f"[{_now_iso()}] final_summary {json.dumps(final, ensure_ascii=False)}")
    print(f"jsonl_report={report_jsonl_path}")
    print("=" * 90)
    return 0 if metrics.total > 0 else 2


def main() -> int:
    _configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Inference test for Noble_RAG (orchestrator + retrieval + qdrant path). "
            "Supports interactive chat mode and continuous load mode."
        )
    )
    parser.add_argument("--mode", choices=["interactive", "continuous"], default="interactive")
    parser.add_argument("--orchestrator-base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--retrieval-base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--cases", default="data/sample_queries.json", help="Query cases JSON file.")
    parser.add_argument("--report-jsonl", default="inference_continuous_report.jsonl", help="Output JSONL report file.")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Total requests to run. Use 0 for infinite loop until Ctrl+C.",
    )
    parser.add_argument("--interval-sec", type=float, default=1.5, help="Sleep between requests.")
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="HTTP timeout per request.")
    parser.add_argument(
        "--force-project-every",
        type=int,
        default=5,
        help="Every N requests force project route to stress retrieval path. Use 0 to disable.",
    )
    parser.add_argument("--session-count", type=int, default=3, help="Number of rolling conversation sessions.")
    parser.add_argument("--summary-every", type=int, default=20, help="Print rolling summary every N requests.")
    parser.add_argument("--session-id", default="manual-chat-01", help="Session id for interactive mode.")
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Include raw response payload in JSONL record (interactive mode).",
    )
    parser.add_argument(
        "--check-retrieval-health",
        action="store_true",
        help="Require retrieval /health to pass in health_ok (disabled by default for internal-only retrieval service).",
    )
    args = parser.parse_args()

    max_requests = None if args.max_requests == 0 else max(1, int(args.max_requests))
    if args.mode == "interactive":
        return run_interactive_chat_test(
            orchestrator_base_url=args.orchestrator_base_url,
            retrieval_base_url=args.retrieval_base_url,
            report_jsonl_path=Path(args.report_jsonl),
            timeout_sec=max(1.0, args.timeout_sec),
            check_retrieval_health=bool(args.check_retrieval_health),
            session_id=str(args.session_id).strip() or "manual-chat-01",
            show_raw_response=bool(args.show_raw_response),
        )

    return run_continuous_inference_test(
        orchestrator_base_url=args.orchestrator_base_url,
        retrieval_base_url=args.retrieval_base_url,
        cases_path=Path(args.cases),
        report_jsonl_path=Path(args.report_jsonl),
        max_requests=max_requests,
        interval_sec=max(0.0, args.interval_sec),
        timeout_sec=max(1.0, args.timeout_sec),
        force_project_every=max(0, args.force_project_every),
        session_count=max(1, args.session_count),
        summary_every=max(1, args.summary_every),
        check_retrieval_health=bool(args.check_retrieval_health),
    )


if __name__ == "__main__":
    sys.exit(main())
