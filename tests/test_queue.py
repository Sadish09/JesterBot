"""
tests/test_queue.py — Ingest queue worker behaviour.

Catches:
- Enqueue + worker picks up the job
- Stale jobs are dropped
- Worker survives pipeline exceptions
- Shutdown cancels workers cleanly
- Queue depth tracking
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import IngestJob
from ingest.queue import IngestQueue


@pytest.fixture
def mock_pipeline() -> AsyncMock:
    p = AsyncMock()
    p.process = AsyncMock(return_value=None)
    return p


@pytest.fixture
def queue(mock_pipeline: AsyncMock) -> IngestQueue:
    return IngestQueue(mock_pipeline, worker_count=1)


def _make_job(age_seconds: float = 0) -> IngestJob:
    """Create a job with a timestamp `age_seconds` in the past."""
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return IngestJob(
        message_id="123",
        channel_id="456",
        guild_id="789",
        image_url="https://cdn.discord.com/test.png",
        caption="test",
        timestamp=ts,
    )


class TestEnqueue:
    def test_depth_increases(self, queue: IngestQueue) -> None:
        assert queue.depth == 0
        queue.enqueue(_make_job())
        assert queue.depth == 1

    def test_multiple_enqueue(self, queue: IngestQueue) -> None:
        for _ in range(5):
            queue.enqueue(_make_job())
        assert queue.depth == 5


class TestWorkerProcessing:
    @pytest.mark.asyncio
    async def test_worker_processes_job(
        self, queue: IngestQueue, mock_pipeline: AsyncMock
    ) -> None:
        """Worker picks up and processes an enqueued job."""
        job = _make_job()
        queue.enqueue(job)
        queue.start_workers()

        # Give worker time to pick up the job
        await asyncio.sleep(0.1)

        mock_pipeline.process.assert_awaited_once()
        called_job = mock_pipeline.process.call_args[0][0]
        assert called_job.message_id == "123"

        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_stale_job_dropped(
        self, queue: IngestQueue, mock_pipeline: AsyncMock
    ) -> None:
        """Job older than max_age is dropped without processing."""
        stale_job = _make_job(age_seconds=120)  # 2 min old, max_age is 60
        queue.enqueue(stale_job)
        queue.start_workers()

        await asyncio.sleep(0.1)

        mock_pipeline.process.assert_not_awaited()
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_fresh_job_not_dropped(
        self, queue: IngestQueue, mock_pipeline: AsyncMock
    ) -> None:
        """Job within max_age is processed normally."""
        fresh_job = _make_job(age_seconds=5)  # 5 sec old, max_age is 60
        queue.enqueue(fresh_job)
        queue.start_workers()

        await asyncio.sleep(0.1)

        mock_pipeline.process.assert_awaited_once()
        await queue.shutdown()


class TestWorkerResilience:
    @pytest.mark.asyncio
    async def test_worker_survives_exception(self, mock_pipeline: AsyncMock) -> None:
        """Pipeline exception doesn't kill the worker — it processes the next job."""
        mock_pipeline.process = AsyncMock(
            side_effect=[RuntimeError("boom"), None]
        )
        queue = IngestQueue(mock_pipeline, worker_count=1)

        queue.enqueue(_make_job())  # will fail
        queue.enqueue(_make_job())  # should still be processed
        queue.start_workers()

        await asyncio.sleep(0.2)

        assert mock_pipeline.process.await_count == 2
        await queue.shutdown()


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_clears_workers(self, queue: IngestQueue) -> None:
        queue.start_workers()
        assert len(queue._workers) == 1
        await queue.shutdown()
        assert len(queue._workers) == 0

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, queue: IngestQueue) -> None:
        """Calling shutdown twice doesn't crash."""
        queue.start_workers()
        await queue.shutdown()
        await queue.shutdown()  # should be safe
