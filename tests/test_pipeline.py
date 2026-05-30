"""
tests/test_pipeline.py — Ingest pipeline with mocked dependencies.

Catches:
- Happy path: download → hash → dedup miss → HF → DB → FAISS → mark seen
- Dedup hit: pipeline returns None early, skips HF/DB/FAISS
- HF failure: falls back to zero embedding + empty OCR
- DB failure: pipeline continues (FAISS still gets the vector)
- Download failure: returns None, no side effects
- Image too large: returns None cleanly
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from cache.redis_client import NoOpCache
from core.models import IngestJob
from ingest.pipeline import IngestPipeline
from integrations.db import MongoDB
from integrations.hf import HFSpacesClient
from search.vector_search import FAISSIndex


def _make_png(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (100, 150, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_deps() -> tuple[NoOpCache, AsyncMock, AsyncMock, FAISSIndex]:
    """Return (cache, mock_db, mock_hf, real_faiss)."""
    cache = NoOpCache()

    db = AsyncMock(spec=MongoDB)
    db.upsert_meme = AsyncMock()

    hf = AsyncMock(spec=HFSpacesClient)
    hf.process_image = AsyncMock(return_value=([0.5] * 512, "ocr text"))

    faiss_index = FAISSIndex(dim=512)

    return cache, db, hf, faiss_index


@pytest.fixture
def pipeline(mock_deps: tuple) -> IngestPipeline:
    cache, db, hf, faiss_index = mock_deps
    return IngestPipeline(redis=cache, db=db, hf=hf, faiss_index=faiss_index)


@pytest.fixture
def job() -> IngestJob:
    return IngestJob(
        message_id="12345",
        channel_id="99999",
        guild_id="88888",
        image_url="https://cdn.discord.com/test.png",
        caption="funny meme",
    )


class TestPipelineHappyPath:
    @pytest.mark.asyncio
    async def test_successful_ingest(
        self, pipeline: IngestPipeline, job: IngestJob, mock_deps: tuple
    ) -> None:
        """Full pipeline → returns MemeDocument, FAISS gets a vector."""
        cache, db, hf, faiss_index = mock_deps
        png_bytes = _make_png()

        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.return_value = png_bytes
            doc = await pipeline.process(job)

        assert doc is not None
        assert doc.message_id == "12345"
        assert doc.caption == "funny meme"
        assert doc.ocr_text == "ocr text"
        assert doc.searchable_text == "funny meme ocr text"
        assert len(doc.embedding) == 512
        assert faiss_index.count == 1
        db.upsert_meme.assert_awaited_once()
        hf.process_image.assert_awaited_once()


class TestPipelineDedup:
    @pytest.mark.asyncio
    async def test_duplicate_skipped(self, job: IngestJob, mock_deps: tuple) -> None:
        """If Redis says it's a dup, pipeline returns None immediately."""
        cache_mock = AsyncMock(spec=NoOpCache)
        cache_mock.check_duplicate = AsyncMock(return_value="existing_msg_id")
        _, db, hf, faiss_index = mock_deps

        pipeline = IngestPipeline(redis=cache_mock, db=db, hf=hf, faiss_index=faiss_index)
        png_bytes = _make_png()

        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.return_value = png_bytes
            doc = await pipeline.process(job)

        assert doc is None
        hf.process_image.assert_not_awaited()
        db.upsert_meme.assert_not_awaited()
        assert faiss_index.count == 0


class TestPipelineFailures:
    @pytest.mark.asyncio
    async def test_download_failure(self, pipeline: IngestPipeline, job: IngestJob) -> None:
        """Download exception → returns None, no side effects."""
        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.side_effect = Exception("CDN timeout")
            doc = await pipeline.process(job)
        assert doc is None

    @pytest.mark.asyncio
    async def test_hf_failure_returns_none(self, job: IngestJob, mock_deps: tuple) -> None:
        """HF raising a non-NotImplementedError → pipeline returns None."""
        cache, db, hf, faiss_index = mock_deps
        hf.process_image = AsyncMock(side_effect=RuntimeError("HF cold start"))
        pipeline = IngestPipeline(redis=cache, db=db, hf=hf, faiss_index=faiss_index)

        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.return_value = _make_png()
            doc = await pipeline.process(job)
        assert doc is None

    @pytest.mark.asyncio
    async def test_hf_not_implemented_fallback(self, job: IngestJob, mock_deps: tuple) -> None:
        """HF NotImplementedError → zero embedding, empty OCR, pipeline continues."""
        cache, db, hf, faiss_index = mock_deps
        hf.process_image = AsyncMock(side_effect=NotImplementedError)
        pipeline = IngestPipeline(redis=cache, db=db, hf=hf, faiss_index=faiss_index)

        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.return_value = _make_png()
            doc = await pipeline.process(job)

        assert doc is not None
        assert doc.ocr_text == ""
        assert all(v == 0.0 for v in doc.embedding)
        assert faiss_index.count == 1

    @pytest.mark.asyncio
    async def test_db_failure_continues(self, job: IngestJob, mock_deps: tuple) -> None:
        """DB write fails → pipeline returns None but doesn't crash workers."""
        cache, db, hf, faiss_index = mock_deps
        db.upsert_meme = AsyncMock(side_effect=RuntimeError("Mongo down"))
        pipeline = IngestPipeline(redis=cache, db=db, hf=hf, faiss_index=faiss_index)

        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.return_value = _make_png()
            doc = await pipeline.process(job)
        # DB failure returns None (line 170 in pipeline.py)
        assert doc is None

    @pytest.mark.asyncio
    async def test_db_not_implemented_continues(self, job: IngestJob, mock_deps: tuple) -> None:
        """DB stub NotImplementedError → pipeline skips write but finishes."""
        cache, db, hf, faiss_index = mock_deps
        db.upsert_meme = AsyncMock(side_effect=NotImplementedError)
        pipeline = IngestPipeline(redis=cache, db=db, hf=hf, faiss_index=faiss_index)

        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.return_value = _make_png()
            doc = await pipeline.process(job)

        assert doc is not None
        assert faiss_index.count == 1  # FAISS still got the vector


class TestPipelineImageTooLarge:
    @pytest.mark.asyncio
    async def test_oversized_image(self, pipeline: IngestPipeline, job: IngestJob) -> None:
        from ingest.downloader import ImageTooLargeError
        with patch("ingest.pipeline.download_image", new_callable=AsyncMock) as dl:
            dl.side_effect = ImageTooLargeError("too big")
            doc = await pipeline.process(job)
        assert doc is None
