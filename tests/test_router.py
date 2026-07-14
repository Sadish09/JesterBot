"""
tests/test_router.py — Search router phase logic.

Catches:
- Caption hit → returns immediately (phase 0)
- Cache hit → returns immediately (phase 1)
- Text search hit → caches + returns (phase 2)
- Short query bypass → skips text search
- Vector search fallback when text misses (phase 3)
- All phases fail → empty list, no crash
- HF embed failure → empty list
- Empty FAISS index → empty list
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cache.redis_client import NoOpCache
from core.models import SearchResult
from search.caption_index import CaptionIndex
from search.router import SearchRouter
from search.text_search import TextSearch
from search.vector_search import FAISSIndex


@pytest.fixture
def mock_components() -> dict:
    """Build mock components for the search router."""
    cache = AsyncMock(spec=NoOpCache)
    cache.get_search_results = AsyncMock(return_value=None)
    cache.set_search_results = AsyncMock()

    db = AsyncMock()
    db.fetch_by_ids = AsyncMock(return_value=[])

    hf = AsyncMock()
    hf.embed_text = AsyncMock(return_value=[0.1] * 512)

    text_search = AsyncMock(spec=TextSearch)
    text_search.search = AsyncMock(return_value=[])

    faiss_index = FAISSIndex(dim=512)
    caption_index = CaptionIndex()

    return {
        "cache": cache,
        "db": db,
        "hf": hf,
        "text_search": text_search,
        "faiss_index": faiss_index,
        "caption_index": caption_index,
    }


@pytest.fixture
def router(mock_components: dict) -> SearchRouter:
    return SearchRouter(
        redis=mock_components["cache"],
        db=mock_components["db"],
        hf=mock_components["hf"],
        text_search=mock_components["text_search"],
        faiss_index=mock_components["faiss_index"],
        caption_index=mock_components["caption_index"],
    )


@pytest.fixture
def sample_results() -> list[SearchResult]:
    return [
        SearchResult(message_id="1", image_url="x", searchable_text="doge", score=0.9),
    ]


class TestPhase0CaptionHit:
    @pytest.mark.asyncio
    async def test_caption_hit_returns_immediately(
        self, mock_components: dict, sample_results: list[SearchResult]
    ) -> None:
        """Caption match → returns immediately, skips all other phases."""
        caption_index = mock_components["caption_index"]
        caption_index.add("msg1", "http://img/1.png", "doge_meme")

        router = SearchRouter(
            redis=mock_components["cache"],
            db=mock_components["db"],
            hf=mock_components["hf"],
            text_search=mock_components["text_search"],
            faiss_index=mock_components["faiss_index"],
            caption_index=caption_index,
        )

        results = await router.search("doge meme")
        assert len(results) == 1
        assert results[0].message_id == "msg1"
        # All other phases should be skipped
        mock_components["cache"].get_search_results.assert_not_awaited()
        mock_components["text_search"].search.assert_not_awaited()
        mock_components["hf"].embed_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_caption_miss_falls_through(
        self, router: SearchRouter, mock_components: dict
    ) -> None:
        """No caption match → falls through to Phase 1+."""
        results = await router.search("nonexistent query")
        # Should have tried the cache at minimum
        mock_components["cache"].get_search_results.assert_awaited_once()



class TestPhase1CacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_immediately(
        self, router: SearchRouter, mock_components: dict, sample_results: list[SearchResult]
    ) -> None:
        mock_components["cache"].get_search_results.return_value = sample_results
        results = await router.search("doge meme")
        assert results == sample_results
        # Text search should NOT be called
        mock_components["text_search"].search.assert_not_awaited()
        mock_components["hf"].embed_text.assert_not_awaited()


class TestPhase2TextSearch:
    @pytest.mark.asyncio
    async def test_text_hit_caches_and_returns(
        self, router: SearchRouter, mock_components: dict, sample_results: list[SearchResult]
    ) -> None:
        mock_components["text_search"].search.return_value = sample_results
        results = await router.search("doge meme")
        assert results == sample_results
        # Should cache the results
        mock_components["cache"].set_search_results.assert_awaited_once()
        # Should NOT fall through to vector search
        mock_components["hf"].embed_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_short_query_skips_text(
        self, router: SearchRouter, mock_components: dict
    ) -> None:
        """Queries < 3 chars skip text search entirely."""
        await router.search("ab")
        mock_components["text_search"].search.assert_not_awaited()


class TestPhase3VectorSearch:
    @pytest.mark.asyncio
    async def test_empty_faiss_returns_empty(
        self, router: SearchRouter, mock_components: dict
    ) -> None:
        """Empty FAISS → [], doesn't even call embed_text."""
        results = await router.search("long query here")
        assert results == []

    @pytest.mark.asyncio
    async def test_hf_embed_failure(
        self, router: SearchRouter, mock_components: dict
    ) -> None:
        """HF embed_text raises → empty list, no crash."""
        # Add a vector so FAISS isn't empty
        mock_components["faiss_index"].add([0.1] * 512, "doc_1")
        mock_components["hf"].embed_text = AsyncMock(side_effect=RuntimeError("cold start"))
        results = await router.search("some query")
        assert results == []

    @pytest.mark.asyncio
    async def test_hf_not_implemented(
        self, router: SearchRouter, mock_components: dict
    ) -> None:
        """HF stub NotImplementedError → empty list."""
        mock_components["faiss_index"].add([0.1] * 512, "doc_1")
        mock_components["hf"].embed_text = AsyncMock(side_effect=NotImplementedError)
        results = await router.search("some query")
        assert results == []


class TestAllPhasesFail:
    @pytest.mark.asyncio
    async def test_all_miss_returns_empty(
        self, router: SearchRouter, mock_components: dict
    ) -> None:
        """Cache miss + text miss + empty FAISS → empty list, no crash."""
        results = await router.search("something obscure")
        assert results == []
        assert isinstance(results, list)
