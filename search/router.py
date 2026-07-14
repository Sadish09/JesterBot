"""
search/router.py — Multi-phase search router with caption-first fast path.

Responsibility:
    Unified search entry-point used by the /find slash command.
    Implements a four-phase search strategy with early exit:

    Phase 0 — Caption index      (instant, in-memory substring match)
    Phase 1 — Redis query cache   (instant on hit)
    Phase 2 — Atlas text search   (< 200 ms target)
    Phase 3 — FAISS vector search (< 1.5 s target, includes HF embed round-trip)

    Phase 0 catches snake_case caption matches like "doge_meme" before
    touching any external service.  Short queries (< 3 chars) skip
    Phase 2 — string matching on 1-2 char queries returns noise.

    Vector search results are NOT cached — queries are too varied for
    a meaningful hit rate.

Blast radius on failure:
    MEDIUM.  If the entire router fails (unexpected exception), the
    /find command shows an error message to the user.  If individual
    phases fail:
    - Caption index miss → falls through to Redis cache
    - Redis cache miss → falls through to text search (slightly slower)
    - Text search fails → falls through to vector search
    - Vector search fails → returns empty results
    The user always gets a response; it just may be less relevant.
"""

from __future__ import annotations

import time

from cache.redis_client import NoOpCache, RedisCache
from core.config import get_settings
from core.logging import get_logger
from core.models import SearchResult
from integrations.db import MongoDB
from integrations.hf import HFSpacesClient
from search.caption_index import CaptionIndex
from search.text_search import TextSearch
from search.vector_search import FAISSIndex

log = get_logger("search.router")

_SHORT_QUERY_THRESHOLD = 3  # skip text search for queries shorter than this


class SearchRouter:
    """
    SearchRouter(redis, db, hf, text_search, faiss_index, caption_index) -> SearchRouter

    Unified search entry-point.  All dependencies are injected via
    constructor — no global state.

    On failure: construction never fails (just stores references).
    """

    def __init__(
        self,
        redis: RedisCache | NoOpCache,
        db: MongoDB,
        hf: HFSpacesClient,
        text_search: TextSearch,
        faiss_index: FAISSIndex,
        caption_index: CaptionIndex | None = None,
    ) -> None:
        self._redis = redis
        self._db = db
        self._hf = hf
        self._text = text_search
        self._faiss = faiss_index
        self._captions = caption_index

    async def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        """
        search(query: str, k: int = None) -> list[SearchResult]

        Run the multi-phase search pipeline.  Returns up to k results
        (defaults to SEARCH_TOP_K from config), falling through phases
        until we get hits.

        Phase 0: Caption index (in-memory) → return immediately on hit
        Phase 1: Check Redis query cache → return immediately on hit
        Phase 2: Atlas text search (skip if query < 3 chars) → cache and return on hit
        Phase 3: FAISS vector search → embed query via HF, search index, fetch docs

        On failure:
        - Returns empty list if all phases fail or produce no results.
        - Never raises — the /find command depends on this.
        - Logs timing and phase information for every search.
        """
        settings = get_settings()
        k = k or settings.SEARCH_TOP_K
        q = query.strip()
        t0 = time.monotonic()

        # ── Phase 0: Caption index (in-memory, instant) ──────────────────
        if self._captions and self._captions.count > 0:
            t_cap = time.monotonic()
            caption_hits = self._captions.search(q, limit=k)
            if caption_hits:
                log.info(
                    "search_caption_hit",
                    query=q,
                    count=len(caption_hits),
                    ms=_ms(t_cap),
                )
                return caption_hits
            log.debug("search_caption_miss", query=q, ms=_ms(t_cap))

        # ── Phase 1: Redis query cache ───────────────────────────────────
        cached = await self._redis.get_search_results(q)
        if cached:
            log.info("search_cache_hit", query=q, ms=_ms(t0))
            return cached

        # ── Phase 2: Atlas text search (skip for short queries) ──────────
        if len(q) >= _SHORT_QUERY_THRESHOLD:
            t_text = time.monotonic()
            text_results = await self._text.search(q, limit=k)
            if text_results:
                await self._redis.set_search_results(q, text_results)
                log.info(
                    "search_text_hit",
                    query=q,
                    count=len(text_results),
                    ms=_ms(t_text),
                )
                return text_results
            log.info("search_text_miss", query=q, ms=_ms(t_text))

        # ── Phase 3: Vector search (FAISS) ───────────────────────────────
        if self._faiss.is_empty:
            log.info("search_faiss_empty", query=q)
            return []

        t_vec = time.monotonic()
        try:
            query_embedding = await self._hf.embed_text(q)
        except NotImplementedError:
            # Safety net — HF embed_text is implemented but this catch
            # remains in case the endpoint is temporarily misconfigured.
            log.warning(
                "embed_text_not_implemented",
                detail="HF embed_text raised NotImplementedError — cannot do vector search",
            )
            return []
        except Exception:
            log.exception("embed_text_failed", query=q)
            return []

        faiss_hits = await self._faiss.search(query_embedding, k=k)
        if not faiss_hits:
            log.info("search_faiss_miss", query=q, ms=_ms(t_vec))
            return []

        # Fetch full documents from MongoDB for the FAISS hits
        hit_ids = [doc_id for doc_id, _ in faiss_hits]
        score_map = {doc_id: score for doc_id, score in faiss_hits}

        try:
            docs = await self._db.fetch_by_ids(hit_ids)
        except NotImplementedError:
            log.warning("db_fetch_stub", detail="MongoDB fetch_by_ids not implemented")
            # Return minimal results from FAISS IDs alone
            return [
                SearchResult(
                    message_id=doc_id,
                    image_url="",
                    searchable_text="",
                    score=score,
                )
                for doc_id, score in faiss_hits
            ]

        results = [
            SearchResult(
                message_id=d.message_id,
                image_url=d.image_url,
                searchable_text=d.searchable_text,
                score=score_map.get(d.message_id, score_map.get(d.mongo_id or "", 0.0)),
            )
            for d in docs
        ]

        log.info("search_faiss_hit", query=q, count=len(results), ms=_ms(t_vec))
        return results


def _ms(start: float) -> float:
    """
    _ms(start: float) -> float

    Return milliseconds elapsed since the given monotonic timestamp.

    On failure: never fails — pure arithmetic.
    """
    return round((time.monotonic() - start) * 1000, 1)
