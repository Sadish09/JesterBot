"""
ingest/queue.py — Asyncio-based ingest queue with a configurable worker pool.

Responsibility:
    Wraps an asyncio.Queue with a pool of N long-lived worker coroutines
    (default N=3, tunable via INGEST_WORKER_COUNT).  The on_message event
    handler pushes IngestJobs into this queue and returns immediately
    (fire-and-forget), so Discord never sees lag from the bot.

    Workers dequeue jobs, run them through the IngestPipeline, and catch
    all exceptions — a single bad image never kills a worker.  Stale jobs
    (older than INGEST_JOB_MAX_AGE seconds) are dropped with a warning.

Blast radius on failure:
    MEDIUM.  If the queue itself breaks (shouldn't — it's stdlib), no
    new memes are ingested.  If a single worker crashes (shouldn't —
    exceptions are caught), that worker stops processing but the
    remaining N-1 workers continue.  The bot and search remain
    functional; only new ingest is degraded.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from core.config import get_settings
from core.logging import get_logger
from core.models import IngestJob

if TYPE_CHECKING:
    from ingest.pipeline import IngestPipeline

log = get_logger("ingest.queue")


class IngestQueue:
    """
    IngestQueue(pipeline, worker_count=None) -> IngestQueue

    Fire-and-forget ingest queue backed by asyncio.Queue.  Call
    start_workers() after the event loop is running, and shutdown()
    during teardown.

    On failure: construction never fails.
    """

    def __init__(self, pipeline: IngestPipeline, *, worker_count: int | None = None) -> None:
        settings = get_settings()
        self._queue: asyncio.Queue[IngestJob] = asyncio.Queue()
        self._pipeline = pipeline
        self._worker_count = worker_count or settings.INGEST_WORKER_COUNT
        self._max_age = settings.INGEST_JOB_MAX_AGE
        self._workers: list[asyncio.Task[None]] = []

    # ── Public API ───────────────────────────────────────────────────────

    def enqueue(self, job: IngestJob) -> None:
        """
        enqueue(job: IngestJob) -> None

        Push a job onto the queue.  Non-blocking, never raises (uses
        put_nowait on an unbounded queue).  Called from the on_message
        event handler on the Discord event loop.

        On failure: never fails for unbounded queues.  Logs the current
        queue depth for monitoring.
        """
        self._queue.put_nowait(job)
        log.debug(
            "job_enqueued",
            message_id=job.message_id,
            queue_depth=self._queue.qsize(),
        )

    def start_workers(self) -> None:
        """
        start_workers() -> None

        Spawn the worker coroutine pool.  Call once after the event
        loop is running (typically inside the bot's async context).

        On failure: never fails — asyncio.create_task() is infallible
        for valid coroutines.
        """
        for i in range(self._worker_count):
            task = asyncio.create_task(self._worker(i), name=f"ingest-worker-{i}")
            self._workers.append(task)
        log.info("workers_started", count=self._worker_count)

    async def shutdown(self) -> None:
        """
        shutdown() -> None

        Cancel all workers and drain the queue.  Waits for workers to
        acknowledge cancellation.

        On failure: catches CancelledError from workers.  Always
        completes — shutdown must not hang.
        """
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        log.info("workers_stopped")

    @property
    def depth(self) -> int:
        """
        depth -> int

        Current number of pending jobs in the queue.  Used by the
        /status command for health monitoring.

        On failure: never fails.
        """
        return self._queue.qsize()

    # ── Internal ─────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """
        _worker(worker_id: int) -> None

        Long-lived coroutine that dequeues and processes jobs until
        cancelled.  Each iteration:
        1. Await next job from the queue
        2. Check job staleness (drop if older than max_age)
        3. Run the ingest pipeline
        4. Catch all exceptions — log and continue

        On failure: catches all exceptions per-job and logs them.
        Only exits on asyncio.CancelledError (shutdown).
        """
        log.info("worker_started", worker_id=worker_id)
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                log.info("worker_cancelled", worker_id=worker_id)
                return

            try:
                # ── Staleness guard ──────────────────────────────────────
                age = time.time() - job.timestamp.timestamp()
                if age > self._max_age:
                    log.warning(
                        "job_stale_dropped",
                        worker_id=worker_id,
                        message_id=job.message_id,
                        age_s=round(age, 1),
                    )
                    self._queue.task_done()
                    continue

                await self._pipeline.process(job)
            except Exception:
                log.exception(
                    "worker_job_failed",
                    worker_id=worker_id,
                    message_id=job.message_id,
                )
            finally:
                self._queue.task_done()
