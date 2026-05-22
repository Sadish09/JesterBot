"""
integrations/hf.py — HF Spaces client — STUB.

Responsibility:
    Async client for the combined CLIP + OCR HF Space.  Sends images to
    a single /process endpoint and returns both the 512-dim CLIP
    embedding and OCR-extracted text in one round-trip.  Also provides
    embed_text() for converting search queries to vectors, and a
    keep-warm ping to prevent free-tier Spaces from sleeping.

    YOUR FRIEND IMPLEMENTS THIS.  All methods currently raise
    NotImplementedError.  The contracts below define the exact
    signatures and expected behaviour.

Blast radius on failure:
    HIGH.  If the HF Space is down or cold-starting:
    - Ingest pipeline cannot produce embeddings or OCR text.  The
      pipeline falls back to empty embeddings (zero vector) and empty
      OCR text, so memes are still stored but unsearchable by content.
    - Vector search (/find) cannot embed the query text, so the FAISS
      fallback path is disabled.  Text search still works.
    The bot remains functional but loses its "smart" search capability.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from core.logging import get_logger

log = get_logger("integrations.hf")


class HFSpacesClient:
    """
    HFSpacesClient(base_url, api_token) -> HFSpacesClient

    Async client for the combined CLIP + OCR HF Space.
    All methods raise NotImplementedError until wired up.

    On failure: construction never fails (just stores config).
    """

    def __init__(self, base_url: str, api_token: Optional[str] = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._keepalive_task: Optional[asyncio.Task[None]] = None
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        if self._api_token:
            return {"Authorization": f"Bearer {self._api_token}"}
        return {}

    # ── Core API ─────────────────────────────────────────────────────────

    async def process_image(self, image_b64: str) -> tuple[list[float], str]:
        """
        process_image(image_b64: str) -> tuple[list[float], str]

        Send a base64-encoded image to the HF Space's /process endpoint.
        Returns a tuple of (512-dim CLIP embedding, OCR-extracted text).

        Contract:
        - POST to {base_url}/process with JSON {"image_b64": image_b64}
        - Set Authorization: Bearer {api_token} if token is provided
        - Timeout after 30s

        On failure: should raise on HF cold-start timeouts or network
        errors.  The ingest pipeline catches NotImplementedError and
        falls back to empty embedding + empty OCR text.
        """
        client = self._get_client()
        url = f"{self._base_url}/process"

        response = await client.post(
            url,
            json={"image_b64": image_b64},
            headers=self._headers(),
        )
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        embedding = data.get("embedding")
        ocr_text = data.get("ocr_text", "")

        if not isinstance(embedding, list):
            raise ValueError("HF /process response missing embedding")

        return embedding, str(ocr_text)

    async def embed_text(self, query: str) -> list[float]:
        """
        embed_text(query: str) -> list[float]

        Get a CLIP text embedding for a search query string.
        Returns a 512-dim L2-normalised vector suitable for FAISS
        IndexFlatIP cosine similarity search.

        Contract:
        - POST to {base_url}/embed_text with JSON {"text": query}
        - Must return a normalised vector (L2 norm = 1)

        On failure: should raise on network/timeout errors.  The search
        router catches NotImplementedError and skips vector search
        entirely, falling back to text-only results.
        """
        client = self._get_client()
        url = f"{self._base_url}/embed_text"

        response = await client.post(
            url,
            json={"text": query},
            headers=self._headers(),
        )
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        embedding = data.get("embedding")

        if not isinstance(embedding, list):
            raise ValueError("HF /embed_text response missing embedding")

        return embedding

    # ── Keep-warm ────────────────────────────────────────────────────────

    async def start_keepalive(self, interval: int = 240) -> None:
        """
        start_keepalive(interval: int = 240) -> None

        Start a background coroutine that pings the HF Space every
        *interval* seconds to prevent free-tier sleep (cold starts
        add 10+ seconds to ingest latency).

        Contract:
        - GET {base_url}/ every interval seconds in a loop
        - Log warnings on ping failure but never crash

        On failure: should catch all exceptions internally and log
        them.  If the keep-alive task itself crashes, the Space may
        go to sleep — ingest latency increases but nothing breaks.
        """
        if self._keepalive_task and not self._keepalive_task.done():
            return

        client = self._get_client()
        url = f"{self._base_url}/"

        async def _loop() -> None:
            while True:
                try:
                    await client.get(url, headers=self._headers())
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.warning("hf_keepalive_failed", exc_info=True)
                await asyncio.sleep(interval)

        self._keepalive_task = asyncio.create_task(_loop())
        log.info("hf_keepalive_started", interval=interval)

    async def stop_keepalive(self) -> None:
        """
        stop_keepalive() -> None

        Cancel the keep-warm background task if it's running.

        On failure: never fails — cancellation is best-effort.
        """
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            log.info("hf_keepalive_stopped")

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def close(self) -> None:
        """
        close() -> None

        Shut down the HTTP client and stop background tasks.
        Safe to call multiple times.

        On failure: logs but never raises — shutdown must complete.
        """
        await self.stop_keepalive()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("hf_client_closed")
