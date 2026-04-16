from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any
from typing import Iterator

log = logging.getLogger("sales-orchestrator.observability")

try:
    from langfuse import get_client
    from langfuse import propagate_attributes
except Exception:  # pragma: no cover - optional dependency at runtime
    get_client = None
    propagate_attributes = None


class _NoopObservation:
    trace_id: str | None = None

    def update(self, **kwargs: Any) -> None:
        _ = kwargs


@contextmanager
def _noop_observation() -> Iterator[_NoopObservation]:
    yield _NoopObservation()


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        out[str(key)] = value
    return out or None


def start_observation(enabled: bool, name: str, *, as_type: str = "span", **kwargs: Any):
    if not enabled or get_client is None:
        return _noop_observation()
    try:
        client = get_client()
        payload = {key: value for key, value in dict(kwargs).items() if value is not None}
        payload["name"] = name
        payload["as_type"] = as_type
        return client.start_as_current_observation(**payload)
    except TypeError:
        payload = {key: value for key, value in payload.items() if key != "trace_context"}
        try:
            return client.start_as_current_observation(**payload)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("langfuse start observation failed: %s", exc)
            return _noop_observation()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("langfuse start observation failed: %s", exc)
        return _noop_observation()


@contextmanager
def propagate_context(
    enabled: bool,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    if not enabled or propagate_attributes is None:
        yield
        return

    kwargs: dict[str, Any] = {}
    sid = str(session_id or "").strip()
    uid = str(user_id or "").strip()
    md = _clean_metadata(metadata)
    if sid:
        kwargs["session_id"] = sid[:128]
    if uid:
        kwargs["user_id"] = uid[:128]
    if md:
        kwargs["metadata"] = md

    if not kwargs:
        yield
        return

    try:
        context = propagate_attributes(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("langfuse propagate attributes failed: %s", exc)
        yield
        return

    with context:
        yield


def flush_observability(enabled: bool) -> None:
    if not enabled or get_client is None:
        return
    try:
        get_client().flush()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("langfuse flush failed: %s", exc)
