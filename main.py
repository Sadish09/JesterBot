"""
main.py — Jester bot entry point and startup orchestrator.

Responsibility:
    Wires together all subsystems and runs the ordered startup sequence:

    1. Load config + configure logging
    2. Connect Redis (optional — uses NoOpCache if REDIS_URL is empty)
    3. Connect MongoDB (Atlas, ~100 ms)  [STUB — friend implements]
    4. Fetch embeddings → build FAISS index  [STUB until MongoDB is live]
    5. Start HF keep-warm ping
    6. Build ingest pipeline + worker pool
    7. Register slash commands + attach event handlers
    8. Start Discord bot + workers
    9. Log "ready"

    Also handles graceful shutdown on SIGINT / SIGTERM: cancels workers,
    closes Redis, MongoDB, HF client, and httpx.

Blast radius on failure:
    TOTAL.  If this module fails, the bot doesn't start at all.
    Startup failures in steps 2-5 are handled gracefully — stubs
    that raise NotImplementedError are caught and logged, the bot
    continues with degraded functionality.  Only step 1 (config) and
    step 8 (Discord token) are hard requirements.
"""

from __future__ import annotations

import asyncio
import signal
import time
from typing import Union

from bot.client import JesterBot
from bot.commands import register_commands
from bot.events import setup_events
from cache.redis_client import NoOpCache, RedisCache
from core.config import get_settings
from core.logging import get_logger, setup_logging
from ingest.downloader import close_client as close_httpx
from ingest.pipeline import IngestPipeline
from ingest.queue import IngestQueue
from integrations.db import MongoDB
from integrations.hf import HFSpacesClient
from search.caption_index import CaptionIndex
from search.router import SearchRouter
from search.text_search import TextSearch
from search.vector_search import FAISSIndex

log = get_logger("main")

# Union type for the cache — either Redis-backed or no-op
Cache = Union[RedisCache, NoOpCache]


async def main() -> None:
    """
    main() -> None

    Async entry point.  Runs the full startup sequence, starts the
    Discord bot, and waits for either the bot to disconnect or a
    shutdown signal.  Then runs the cleanup sequence.

    On failure:
    - Config validation errors crash immediately (intentional).
    - Redis connection failure logs a warning and falls back to NoOpCache.
    - MongoDB / HF stub failures are caught — bot starts degraded.
    - Discord login failure (bad token) crashes after cleanup.
    """
    settings = get_settings()
    setup_logging(env=settings.ENV)
    log.info("starting", env=settings.ENV)
    t0 = time.monotonic()

    # ── 1. Cache (Redis or NoOp) ─────────────────────────────────────────
    cache: Cache
    if settings.REDIS_URL:
        try:
            redis_cache = RedisCache(
                settings.REDIS_URL,
                dedup_ttl_days=settings.DEDUP_TTL_DAYS,
                query_ttl=settings.QUERY_CACHE_TTL,
            )
            await redis_cache.connect()
            cache = redis_cache
        except Exception:
            log.warning(
                "redis_connect_failed",
                detail="Could not connect to Redis — falling back to NoOpCache",
                exc_info=True,
            )
            cache = NoOpCache()
            await cache.connect()
    else:
        cache = NoOpCache()
        await cache.connect()

    # ── 2. MongoDB (stub) ────────────────────────────────────────────────
    db = MongoDB(settings.MONGODB_URI)
    try:
        await db.connect()
    except NotImplementedError:
        log.warning("mongodb_stub", detail="MongoDB connect() not implemented — continuing")

    # ── 3. HF Spaces client ──────────────────────────────────────────────
    hf = HFSpacesClient(settings.HF_SPACES_URL, api_token=settings.HF_API_TOKEN)

    # ── 4. FAISS index — rebuild from MongoDB ────────────────────────────
    faiss_index = FAISSIndex(dim=settings.EMBEDDING_DIM)
    try:
        t_faiss = time.monotonic()
        docs = await db.fetch_all_embeddings()
        faiss_index.build_from_documents(docs)
        log.info(
            "faiss_rebuilt",
            count=faiss_index.count,
            ms=round((time.monotonic() - t_faiss) * 1000, 1),
        )
    except NotImplementedError:
        log.warning(
            "faiss_stub",
            detail="MongoDB fetch_all_embeddings() not implemented — starting with empty index",
        )

    # ── 5. HF keep-warm ping ─────────────────────────────────────────────
    await hf.start_keepalive(settings.HF_WARMUP_INTERVAL)

    # ── 6. Caption index (in-memory) ──────────────────────────────────
    caption_index = CaptionIndex()

    # ── 7. Ingest pipeline + worker pool ─────────────────────────────────
    pipeline = IngestPipeline(
        redis=cache, db=db, hf=hf, faiss_index=faiss_index,
        caption_index=caption_index,
    )
    queue = IngestQueue(pipeline)

    # ── 8. Build the Discord bot + register commands/events ──────────────
    text_search = TextSearch(db)
    search_router = SearchRouter(
        redis=cache,
        db=db,
        hf=hf,
        text_search=text_search,
        faiss_index=faiss_index,
        caption_index=caption_index,
    )

    bot = JesterBot(
        ingest_queue=queue,
        search_router=search_router,
        meme_channel_id=settings.DISCORD_MEME_CHANNEL_ID,
    )

    # Register commands and events BEFORE starting the bot.
    # This avoids circular imports within the bot/ package — client.py
    # no longer needs to import commands.py or events.py.
    register_commands(bot)
    setup_events(bot)

    # ── 8. Start workers + run bot ───────────────────────────────────────
    async def runner() -> None:
        """
        runner() -> None

        Inner coroutine that starts the bot within its async context
        manager, kicks off ingest workers, and sets the ready flag.

        On failure: propagates discord.LoginFailure on bad token.
        """
        async with bot:
            queue.start_workers()
            bot.ready_flag = True
            log.info(
                "ready",
                startup_ms=round((time.monotonic() - t0) * 1000, 1),
                workers=settings.INGEST_WORKER_COUNT,
                faiss_count=faiss_index.count,
                cache=type(cache).__name__,
            )
            await bot.start(settings.DISCORD_TOKEN)

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        """
        _signal_handler() -> None

        Set the shutdown event when a termination signal is received.

        On failure: never fails — just sets an asyncio Event.
        """
        log.info("shutdown_signal_received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    bot_task = asyncio.create_task(runner())

    # Wait for either the bot to finish or a shutdown signal
    done, _ = await asyncio.wait(
        [bot_task, asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # ── Cleanup ──────────────────────────────────────────────────────────
    log.info("shutting_down")
    await queue.shutdown()
    await hf.close()
    try:
        await db.close()
    except NotImplementedError:
        pass
    await cache.close()
    await close_httpx()

    if not bot.is_closed():
        await bot.close()

    log.info("shutdown_complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
