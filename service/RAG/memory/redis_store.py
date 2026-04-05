"""Low-level async Redis helper — opens a new connection per call to avoid
shared-state issues in async contexts."""

import json
import logging
from typing import Any, AsyncIterator, Optional

log = logging.getLogger("rag-service")


async def redis_get(redis_url: str, key: str) -> Optional[str]:
    import redis.asyncio as redis
    client = None
    try:
        client = redis.Redis.from_url(
            redis_url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5
        )
        return await client.get(key)
    except Exception as e:
        log.error("Redis GET error key=%s: %s", key, e)
        return None
    finally:
        if client:
            await client.aclose()


async def redis_set(redis_url: str, key: str, value: str, ttl: int = 86400) -> bool:
    import redis.asyncio as redis
    client = None
    try:
        client = redis.Redis.from_url(
            redis_url, socket_timeout=5, socket_connect_timeout=5
        )
        await client.setex(key, ttl, value)
        return True
    except Exception as e:
        log.error("Redis SET error key=%s: %s", key, e)
        return False
    finally:
        if client:
            await client.aclose()


async def redis_delete(redis_url: str, key: str) -> bool:
    import redis.asyncio as redis
    client = None
    try:
        client = redis.Redis.from_url(redis_url, socket_timeout=5, socket_connect_timeout=5)
        await client.delete(key)
        return True
    except Exception as e:
        log.error("Redis DELETE error key=%s: %s", key, e)
        return False
    finally:
        if client:
            await client.aclose()


async def redis_scan_keys_matching(redis_url: str, pattern: str) -> AsyncIterator[str]:
    """Duyệt key khớp pattern (SCAN)."""
    import redis.asyncio as redis

    client = None
    try:
        client = redis.Redis.from_url(
            redis_url, decode_responses=True, socket_timeout=30, socket_connect_timeout=5
        )
        async for key in client.scan_iter(match=pattern, count=500):
            yield key
    finally:
        if client:
            await client.aclose()


async def redis_delete_keys_matching_pattern(redis_url: str, pattern: str) -> int:
    """Xóa mọi key khớp pattern (SCAN, không dùng KEYS *). Trả về số key đã xóa."""
    import redis.asyncio as redis

    client = None
    deleted = 0
    try:
        client = redis.Redis.from_url(
            redis_url, decode_responses=True, socket_timeout=30, socket_connect_timeout=5
        )
        async for key in client.scan_iter(match=pattern, count=500):
            await client.delete(key)
            deleted += 1
        return deleted
    except Exception as e:
        log.error("Redis SCAN+DELETE error pattern=%s: %s", pattern, e)
        return deleted
    finally:
        if client:
            await client.aclose()


async def redis_get_json(redis_url: str, key: str) -> Optional[Any]:
    raw = await redis_get(redis_url, key)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


async def redis_set_json(redis_url: str, key: str, value: Any, ttl: int = 86400) -> bool:
    return await redis_set(redis_url, key, json.dumps(value, ensure_ascii=False), ttl)
