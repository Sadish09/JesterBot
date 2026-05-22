"""
ingest/downloader.py — Streaming image downloader with size cap.

Responsibility:
    Downloads images from Discord CDN URLs in 8 KiB chunks, enforcing
    a configurable size limit (default 8 MB).  Uses a shared httpx
    AsyncClient with connection pooling for efficiency across multiple
    concurrent downloads.  Never loads a full image into memory before
    knowing its size.

Blast radius on failure:
    LOW.  If a single download fails (network error, 4xx, size limit),
    only that one meme is skipped — the ingest pipeline logs a warning
    and the queue worker moves on.  If the httpx client itself is broken
    (constructor failure), all downloads fail and no new memes are
    indexed, but the bot stays online and search still works on
    existing data.
"""

from __future__ import annotations

import httpx

from core.config import get_settings
from core.logging import get_logger

log = get_logger("ingest.downloader")

_CHUNK_SIZE = 8192  # 8 KiB


class ImageTooLargeError(Exception):
    """Raised when an image exceeds the configured size limit."""


# Module-level client — reused across the process lifetime for connection
# pooling.  Created lazily on first call; closed from main.py shutdown.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """
    _get_client() -> httpx.AsyncClient

    Return the shared httpx client, creating it lazily on first call.
    Configured with redirect following and a 30s timeout (10s connect).

    On failure: raises if httpx.AsyncClient construction fails (e.g.
    invalid SSL config).  In practice this never happens.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    return _client


async def close_client() -> None:
    """
    close_client() -> None

    Shut down the shared httpx client.  Called from main.py shutdown
    sequence.  Safe to call even if the client was never created.

    On failure: logs but does not raise — shutdown must complete.
    """
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        log.info("httpx_client_closed")


async def download_image(url: str) -> bytes:
    """
    download_image(url: str) -> bytes

    Stream-download an image from the given URL, enforcing the size
    cap from IMAGE_SIZE_LIMIT_MB in config.  Downloads in 8 KiB chunks
    and bails the moment accumulated size exceeds the limit.

    Returns the raw image bytes on success.

    On failure:
    - Raises ImageTooLargeError if the payload exceeds the size limit.
    - Raises httpx.HTTPStatusError on 4xx / 5xx responses.
    - Raises httpx.ConnectError / httpx.TimeoutException on network
      issues.
    The ingest pipeline catches all of these and skips the meme.
    """
    settings = get_settings()
    max_bytes = settings.IMAGE_SIZE_LIMIT_MB * 1024 * 1024
    client = _get_client()

    async with client.stream("GET", url) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes(_CHUNK_SIZE):
            size += len(chunk)
            if size > max_bytes:
                raise ImageTooLargeError(
                    f"Image exceeds {settings.IMAGE_SIZE_LIMIT_MB} MB limit "
                    f"(got {size} bytes so far)"
                )
            chunks.append(chunk)

    image_bytes = b"".join(chunks)
    log.debug("image_downloaded", url=url, size_bytes=len(image_bytes))
    return image_bytes
