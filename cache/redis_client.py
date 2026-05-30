"""
cache/redis_client.py — Redis cache with dedup and query result layers.

Responsibility:
    Provides two logical caches backed by a single Redis instance:
    1. Image dedup  (``img:{phash_hex}`` → ``message_id``, TTL 30 days)
       Checked *before* hitting HF Spaces — highest-leverage optimisation.
    2. Query result  (``search:{normalised_query}`` → serialised top-K,
       TTL 5 min).  Only for text/string search hits — vector queries
       are too varied to cache.

    When Redis is not configured (REDIS_URL is empty), the NoOpCache
    drop-in replacement is used instead — same interface, every call
    returns "cache miss" or is a no-op.  This lets the pipeline and
    search router work identically without touching their code.

Blast radius on failure:
    LOW–MEDIUM.  Redis is optional.  If Redis is configured and goes
    down at runtime:
    - Dedup checks fail → duplicate memes get re-processed (wasted HF
      calls but no data corruption).
    - Query cache misses → every search hits Atlas/FAISS (slower but
      correct).
    The bot continues to function, just less efficiently.
    If Redis is not configured at all, the bot runs fine — just without
    caching.
"""

from __future__ import annotations

import json
from typing import Optional

from core.logging import get_logger
from core.models import SearchResult

log = get_logger("cache")


class NoOpCache:
    """
    NoOpCache() -> NoOpCache

    Drop-in replacement for RedisCache that does nothing.  Every read
    returns None (cache miss), every write is silently dropped.  Used
    when REDIS_URL is not configured.

    On failure: never fails — all methods are no-ops.
    """

    async def connect(self) -> None:
        """
        connect() -> None

        No-op.  Logs that caching is disabled.

        On failure: never fails.
        """
        log.info("cache_disabled", detail="No Redis configured — running without cache")

    async def close(self) -> None:
        """
        close() -> None

        No-op.

        On failure: never fails.
        """

    async def check_duplicate(self, phash: str) -> Optional[str]:
        """
        check_duplicate(phash: str) -> Optional[str]

        Always returns None (not a duplicate).

        On failure: never fails.
        """
        return None

    async def mark_seen(self, phash: str, message_id: str) -> None:
        """
        mark_seen(phash: str, message_id: str) -> None

        No-op — silently drops the write.

        On failure: never fails.
        """

    async def get_search_results(self, query: str) -> Optional[list[SearchResult]]:
        """
        get_search_results(query: str) -> Optional[list[SearchResult]]

        Always returns None (cache miss).

        On failure: never fails.
        """
        return None

    async def set_search_results(self, query: str, results: list[SearchResult]) -> None:
        """
        set_search_results(query: str, results: list[SearchResult]) -> None

        No-op — silently drops the write.

        On failure: never fails.
        """


class RedisCache:
    """
    RedisCache(url, dedup_ttl_days, query_ttl) -> RedisCache

    Async Redis wrapper with dedup and query caching.  Must call
    connect() before using any cache methods, and close() on shutdown.

    On failure: construction never fails (just stores config).
    Methods raise AssertionError if called before connect().
    """

    def __init__(self, url: str, *, dedup_ttl_days: int = 30, query_ttl: int = 300) -> None:
        self._url = url
        self._dedup_ttl = dedup_ttl_days * 86_400  # convert to seconds
        self._query_ttl = query_ttl
        self._redis = None  # aioredis.Redis — imported lazily

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        connect() -> None

        Open the Redis connection pool and verify connectivity with a
        PING command.

        On failure: raises redis.ConnectionError if the server is
        unreachable or the URL is malformed.
        """
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(
            self._url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        # Quick health check
        await self._redis.ping()
        log.info("redis_connected")

    async def close(self) -> None:
        """
        close() -> None

        Drain and close the Redis connection pool.  Safe to call
        multiple times.

        On failure: logs a warning but never raises — shutdown must
        not be blocked by Redis cleanup issues.
        """
        if self._redis:
            await self._redis.aclose()
            log.info("redis_closed")

    @property
    def _r(self):
        """
        _r -> aioredis.Redis

        Convenience accessor that asserts we're connected.

        On failure: raises AssertionError if connect() was never called.
        """
        assert self._redis is not None, "RedisCache not connected — call connect() first"
        return self._redis

    # ── Layer 1: Image Dedup ─────────────────────────────────────────────

    async def check_duplicate(self, phash: str) -> Optional[str]:
        """
        check_duplicate(phash: str) -> Optional[str]

        Return the message_id if this pHash was already indexed, else
        None.  Used by the ingest pipeline to skip re-processing of
        images that have already been seen.

        On failure: raises redis.RedisError on connection issues.
        Callers should treat failures as "not a duplicate" and proceed
        with ingest to avoid silent data loss.
        """
        return await self._r.get(f"img:{phash}")

    async def mark_seen(self, phash: str, message_id: str) -> None:
        """
        mark_seen(phash: str, message_id: str) -> None

        Record that this pHash has been ingested.  Sets a Redis key
        with the configured dedup TTL (default 30 days).

        On failure: raises redis.RedisError on connection issues.
        If this fails after a successful ingest, the same image may be
        re-processed on next encounter — wasteful but not harmful.
        """
        await self._r.set(f"img:{phash}", message_id, ex=self._dedup_ttl)

    # ── Layer 2: Query Result Cache ──────────────────────────────────────

    @staticmethod
    def _normalise_query(query: str) -> str:
        """
        _normalise_query(query: str) -> str

        Lowercase + strip whitespace so ``/find Doge`` and
        ``/find  doge`` hit the same cache key.

        On failure: never fails — operates on pure strings.
        """
        return query.strip().lower()

    async def get_search_results(self, query: str) -> Optional[list[SearchResult]]:
        """
        get_search_results(query: str) -> Optional[list[SearchResult]]

        Return cached search results for the given query, or None on
        cache miss.  Deserialises JSON back into SearchResult objects.

        On failure: returns None if the cached JSON is corrupt (logs a
        warning).  Raises redis.RedisError on connection issues.
        """
        raw = await self._r.get(f"search:{self._normalise_query(query)}")
        if raw is None:
            return None
        try:
            items = json.loads(raw)
            return [
                SearchResult(
                    message_id=r["message_id"],
                    image_url=r["image_url"],
                    searchable_text=r["searchable_text"],
                    score=r.get("score", 0.0),
                )
                for r in items
            ]
        except (json.JSONDecodeError, KeyError):
            log.warning("query_cache_corrupt", query=query)
            return None

    async def set_search_results(self, query: str, results: list[SearchResult]) -> None:
        """
        set_search_results(query: str, results: list[SearchResult]) -> None

        Cache search results with a short TTL (default 5 minutes).
        Serialises SearchResult list to JSON.

        On failure: raises redis.RedisError on connection issues.
        If caching fails the search still returns results to the user —
        just the next identical query won't be cached.
        """
        payload = json.dumps(
            [
                {
                    "message_id": r.message_id,
                    "image_url": r.image_url,
                    "searchable_text": r.searchable_text,
                    "score": r.score,
                }
                for r in results
            ]
        )
        await self._r.set(
            f"search:{self._normalise_query(query)}",
            payload,
            ex=self._query_ttl,
        )
