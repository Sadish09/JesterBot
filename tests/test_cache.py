"""
tests/test_cache.py — NoOpCache interface and RedisCache serialisation.

Catches:
- NoOpCache always returns cache-miss (never leaks state)
- RedisCache serialisation round-trip preserves SearchResult data
- Corrupt JSON in cache is handled gracefully
- Query normalisation (case, whitespace)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from cache.redis_client import NoOpCache, RedisCache
from core.models import SearchResult


class TestNoOpCache:
    """NoOpCache must be a safe, silent drop-in for RedisCache."""

    @pytest.mark.asyncio
    async def test_connect_close(self) -> None:
        cache = NoOpCache()
        await cache.connect()
        await cache.close()

    @pytest.mark.asyncio
    async def test_check_duplicate_always_none(self) -> None:
        cache = NoOpCache()
        assert await cache.check_duplicate("abc123") is None
        assert await cache.check_duplicate("") is None

    @pytest.mark.asyncio
    async def test_mark_seen_noop(self) -> None:
        cache = NoOpCache()
        await cache.mark_seen("abc123", "msg_1")
        assert await cache.check_duplicate("abc123") is None

    @pytest.mark.asyncio
    async def test_get_search_results_always_none(self) -> None:
        cache = NoOpCache()
        assert await cache.get_search_results("doge") is None

    @pytest.mark.asyncio
    async def test_set_then_get_still_none(self) -> None:
        cache = NoOpCache()
        results = [SearchResult(message_id="1", image_url="x", searchable_text="t")]
        await cache.set_search_results("query", results)
        assert await cache.get_search_results("query") is None


class TestRedisCacheSerialisation:
    """JSON round-trip with mocked Redis."""

    @pytest.fixture
    def cache_with_mock(self) -> tuple[RedisCache, AsyncMock]:
        mock_r = AsyncMock()
        cache = RedisCache("redis://fake:6379/0")
        cache._redis = mock_r
        return cache, mock_r

    @pytest.mark.asyncio
    async def test_roundtrip(self, cache_with_mock: tuple[RedisCache, AsyncMock]) -> None:
        cache, mock_r = cache_with_mock
        stored: dict[str, str] = {}
        mock_r.set = AsyncMock(side_effect=lambda k, v, ex=0: stored.update({k: v}))
        mock_r.get = AsyncMock(side_effect=lambda k: stored.get(k))

        results = [
            SearchResult(message_id="111", image_url="https://x/1.png",
                         searchable_text="hello", score=0.95),
        ]
        await cache.set_search_results("test", results)
        got = await cache.get_search_results("test")
        assert got is not None
        assert got[0].message_id == "111"
        assert got[0].score == 0.95

    @pytest.mark.asyncio
    async def test_corrupt_json_returns_none(self, cache_with_mock: tuple[RedisCache, AsyncMock]) -> None:
        cache, mock_r = cache_with_mock
        mock_r.get = AsyncMock(return_value="not json{{{")
        assert await cache.get_search_results("bad") is None

    @pytest.mark.asyncio
    async def test_missing_fields_returns_none(self, cache_with_mock: tuple[RedisCache, AsyncMock]) -> None:
        cache, mock_r = cache_with_mock
        mock_r.get = AsyncMock(return_value=json.dumps([{"message_id": "1"}]))
        assert await cache.get_search_results("partial") is None


class TestQueryNormalisation:
    def test_lowercase(self) -> None:
        assert RedisCache._normalise_query("DOGE") == "doge"

    def test_strip(self) -> None:
        assert RedisCache._normalise_query("  hello  ") == "hello"

    def test_combined(self) -> None:
        assert RedisCache._normalise_query("  MeMe SeArCh  ") == "meme search"
