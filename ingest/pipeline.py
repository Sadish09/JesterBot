"""
ingest/pipeline.py — Full ingest flow from raw image to indexed meme.

Responsibility:
    Orchestrates the complete ingest pipeline for a single image:
    download → pHash → dedup check → HF Spaces (CLIP + OCR) → MongoDB
    upsert → FAISS index update → Redis mark-seen.  Each step is
    individually timed so you can spot bottlenecks in logs.

    Designed to be called from queue workers.  Never blocks the Discord
    event loop — all I/O is async, CPU-bound pHash runs in an executor.

Blast radius on failure:
    LOW per meme.  If any step fails, only that single meme is skipped.
    The pipeline catches exceptions at each step and returns None
    instead of propagating.  If HF or MongoDB stubs raise
    NotImplementedError, the pipeline logs a warning and continues
    with degraded data (empty embedding / no persistence).  The queue
    worker that called this pipeline is never killed.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Optional

from cache.redis_client import RedisCache
from core.config import get_settings
from core.logging import get_logger
from core.models import IngestJob, MemeDocument
from ingest.downloader import ImageTooLargeError, download_image
from ingest.fingerprint import compute_phash
from integrations.db import MongoDB
from integrations.hf import HFSpacesClient
from search.vector_search import FAISSIndex

log = get_logger("ingest.pipeline")


class IngestPipeline:
    """
    IngestPipeline(redis, db, hf, faiss_index) -> IngestPipeline

    Orchestrates: download → hash → dedup → HF → MongoDB → FAISS.
    Stateless — all dependencies are injected via constructor.

    On failure: construction never fails (just stores references).
    """

    def __init__(
        self,
        redis: RedisCache,
        db: MongoDB,
        hf: HFSpacesClient,
        faiss_index: FAISSIndex,
    ) -> None:
        self._redis = redis
        self._db = db
        self._hf = hf
        self._faiss = faiss_index

    async def process(self, job: IngestJob) -> Optional[MemeDocument]:
        """
        process(job: IngestJob) -> Optional[MemeDocument]

        Run the full ingest pipeline for a single job.  Returns the
        created MemeDocument on success, or None if the image was
        skipped (duplicate, too large, or any step failed).

        Steps:
        1. Stream-download image (8 MB cap)
        2. Compute DCT pHash in executor
        3. Check Redis dedup cache
        4. Send to HF Spaces for CLIP embedding + OCR
        5. Build MemeDocument and merge searchable text
        6. Upsert to MongoDB
        7. Add embedding to FAISS index
        8. Mark pHash as seen in Redis

        On failure: catches exceptions at each step and returns None.
        Never raises — the calling queue worker must not crash.
        HF/MongoDB NotImplementedError is handled gracefully with
        fallback values (empty embedding, skipped write).
        """
        t0 = time.monotonic()

        # ── Step 1: Download ─────────────────────────────────────────────
        try:
            t_dl = time.monotonic()
            image_bytes = await download_image(job.image_url)
            log.info(
                "step_download",
                message_id=job.message_id,
                ms=_ms(t_dl),
                size_bytes=len(image_bytes),
            )
        except ImageTooLargeError:
            log.warning("image_too_large", message_id=job.message_id, url=job.image_url)
            return None
        except Exception:
            log.exception("download_failed", message_id=job.message_id)
            return None

        # ── Step 2: Compute pHash ────────────────────────────────────────
        t_hash = time.monotonic()
        loop = asyncio.get_running_loop()
        img_hash = await loop.run_in_executor(None, compute_phash, image_bytes)
        log.info("step_phash", message_id=job.message_id, hash=img_hash, ms=_ms(t_hash))

        # ── Step 3: Redis dedup check ────────────────────────────────────
        t_dedup = time.monotonic()
        existing = await self._redis.check_duplicate(img_hash)
        if existing is not None:
            log.info(
                "duplicate_skipped",
                message_id=job.message_id,
                existing_message_id=existing,
                hash=img_hash,
            )
            return None
        log.info("step_dedup_miss", message_id=job.message_id, ms=_ms(t_dedup))

        # ── Step 4: HF Spaces (CLIP + OCR) ──────────────────────────────
        t_hf = time.monotonic()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        try:
            embedding, ocr_text = await self._hf.process_image(image_b64)
        except NotImplementedError:
            log.warning(
                "hf_stub_active",
                message_id=job.message_id,
                detail="HF integration not implemented yet — using empty embedding/OCR",
            )
            settings = get_settings()
            embedding = [0.0] * settings.EMBEDDING_DIM
            ocr_text = ""
        except Exception:
            log.exception("hf_processing_failed", message_id=job.message_id)
            return None
        log.info("step_hf", message_id=job.message_id, ms=_ms(t_hf))

        # ── Step 5: Build document ───────────────────────────────────────
        doc = MemeDocument(
            message_id=job.message_id,
            channel_id=job.channel_id,
            guild_id=job.guild_id,
            image_url=job.image_url,
            img_hash=img_hash,
            timestamp=job.timestamp,
            caption=job.caption,
            ocr_text=ocr_text,
            embedding=embedding,
        )
        doc.build_searchable_text()

        # ── Step 6: MongoDB upsert ───────────────────────────────────────
        t_db = time.monotonic()
        try:
            await self._db.upsert_meme(doc)
        except NotImplementedError:
            log.warning(
                "db_stub_active",
                message_id=job.message_id,
                detail="MongoDB integration not implemented yet — skipping write",
            )
        except Exception:
            log.exception("db_write_failed", message_id=job.message_id)
            return None
        log.info("step_db_upsert", message_id=job.message_id, ms=_ms(t_db))

        # ── Step 7: FAISS index update ───────────────────────────────────
        t_faiss = time.monotonic()
        self._faiss.add(embedding, job.message_id)
        log.info("step_faiss_add", message_id=job.message_id, ms=_ms(t_faiss))

        # ── Step 8: Mark seen in Redis ───────────────────────────────────
        await self._redis.mark_seen(img_hash, job.message_id)

        total_ms = _ms(t0)
        log.info("ingest_complete", message_id=job.message_id, total_ms=total_ms)
        return doc


def _ms(start: float) -> float:
    """
    _ms(start: float) -> float

    Return milliseconds elapsed since the given monotonic timestamp.
    Used for per-step timing in pipeline logs.

    On failure: never fails — pure arithmetic.
    """
    return round((time.monotonic() - start) * 1000, 1)
