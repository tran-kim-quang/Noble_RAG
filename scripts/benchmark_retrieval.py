#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def eval_case(
    results: list[dict[str, Any]], expected_doc_ids: list[str], expected_source: list[str], low_confidence: bool = False
) -> tuple[bool, bool, bool, str, list[str], list[str]]:
    got_ids = [str(item.get("doc_id", "")) for item in results]
    got_sources = [str(item.get("source", "")) for item in results]

    if expected_doc_ids:
        hit_at_1 = len(got_ids) > 0 and got_ids[0] in expected_doc_ids
        hit_at_k = any(doc_id in got_ids for doc_id in expected_doc_ids)
        return (
            hit_at_k,
            hit_at_1,
            hit_at_k,
            f"expected_doc_ids={expected_doc_ids}",
            got_ids,
            got_sources,
        )

    if expected_source:
        hit_at_1 = len(got_sources) > 0 and got_sources[0] in expected_source
        hit_at_k = any(src in got_sources for src in expected_source)
        return (
            hit_at_k,
            hit_at_1,
            hit_at_k,
            f"expected_source={expected_source}",
            got_ids,
            got_sources,
        )

    # no-answer case
    hit = len(results) == 0 or low_confidence
    return (
        hit,
        False,
        False,
        f"expected_empty_or_low_confidence got={len(results)} low_confidence={low_confidence}",
        got_ids,
        got_sources,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal retrieval benchmark script")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011", help="Retrieval service base URL")
    parser.add_argument("--cases", required=True, help="Path to benchmark cases JSON")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k for all cases")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"[error] cases file not found: {cases_path}")
        return 1

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        print("[error] cases must be a non-empty list")
        return 1

    default_top_k = args.top_k if args.top_k is not None else int(payload.get("top_k", 5))
    retrieve_url = args.base_url.rstrip("/") + "/retrieve"

    total = len(cases)
    passed = 0
    answerable_total = 0
    hit_at_1_count = 0
    hit_at_k_count = 0
    answerable_pass_count = 0
    no_answer_total = 0
    no_answer_pass_count = 0
    no_answer_false_positive_count = 0
    fail_cases: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        query = str(case.get("query", "")).strip()
        expected_doc_ids = [str(x) for x in case.get("expected_doc_ids", [])]
        expected_source = [str(x) for x in case.get("expected_source", [])]
        case_top_k = int(case.get("top_k", default_top_k))

        if not query:
            print(f"[{idx:02d}] FAIL query is empty")
            continue

        req = {"query": query, "top_k": case_top_k}
        try:
            res = post_json(retrieve_url, req)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"[{idx:02d}] FAIL HTTP {exc.code} query='{query}' body={body[:200]}")
            continue
        except Exception as exc:
            print(f"[{idx:02d}] FAIL query='{query}' error={exc}")
            continue

        results = res.get("results", []) or []
        low_confidence = bool(res.get("low_confidence", False))
        ok, hit_at_1, hit_at_k, reason, got_ids, got_sources = eval_case(
            results, expected_doc_ids, expected_source, low_confidence=low_confidence
        )
        top = results[0] if results else {}
        top_doc = top.get("doc_id", "-")
        top_source = top.get("source", "-")
        top_score = top.get("score", "-")

        is_answerable = bool(expected_doc_ids or expected_source)
        if is_answerable:
            answerable_total += 1
            if hit_at_1:
                hit_at_1_count += 1
            if hit_at_k:
                hit_at_k_count += 1
            if ok:
                answerable_pass_count += 1
        else:
            no_answer_total += 1
            if ok:
                no_answer_pass_count += 1
            if results and not low_confidence:
                no_answer_false_positive_count += 1

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            fail_cases.append(
                {
                    "index": idx,
                    "query": query,
                    "reason": reason,
                    "top3_doc_ids": got_ids[:3],
                    "top3_sources": got_sources[:3],
                }
            )

        print(
            f"[{idx:02d}] {status} query='{query}' top_doc={top_doc} top_source={top_source} top_score={top_score} low_confidence={low_confidence} {reason}"
        )

    recall = passed / total if total else 0.0
    hit_at_1 = hit_at_1_count / answerable_total if answerable_total else 0.0
    hit_at_k = hit_at_k_count / answerable_total if answerable_total else 0.0
    answerable_pass_rate = answerable_pass_count / answerable_total if answerable_total else 0.0
    no_answer_pass_rate = no_answer_pass_count / no_answer_total if no_answer_total else 0.0
    print("-" * 80)
    print(f"summary passed={passed}/{total} recall={recall:.3f}")
    print(
        f"summary answerable={answerable_total} answerable_pass_rate={answerable_pass_rate:.3f} "
        f"hit@1={hit_at_1:.3f} hit@k={hit_at_k:.3f}"
    )
    print(
        f"summary no_answer={no_answer_total} no_answer_pass_rate={no_answer_pass_rate:.3f} "
        f"false_positive_no_answer={no_answer_false_positive_count}"
    )
    if fail_cases:
        print("-" * 80)
        print("fail_cases:")
        for item in fail_cases:
            print(
                f"  - [{item['index']:02d}] query='{item['query']}' reason={item['reason']} "
                f"top3_doc_ids={item['top3_doc_ids']} top3_sources={item['top3_sources']}"
            )
    return 0 if passed == total else 2


if __name__ == "__main__":
    sys.exit(main())
