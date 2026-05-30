"""
tests/conftest.py — Shared fixtures for the entire test suite.

Sets up environment variables before any module imports get_settings(),
provides synthetic test images via Pillow, and common mock factories.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

# ── Environment bootstrap ────────────────────────────────────────────────
# Must happen before any code imports core.config.get_settings()
os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MEME_CHANNEL_ID", "999888777")
os.environ.setdefault("HF_SPACES_URL", "https://test.hf.space")
os.environ.setdefault("HF_API_TOKEN", "hf_test_token")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/jester_test")
os.environ.setdefault("ENV", "development")
os.environ.setdefault("EMBEDDING_DIM", "512")

from core.config import get_settings  # noqa: E402
from core.models import IngestJob, MemeDocument, SearchResult  # noqa: E402


# ── Image fixtures ───────────────────────────────────────────────────────

def _make_image(width: int, height: int, color: tuple[int, ...] = (255, 0, 0)) -> bytes:
    """Create a solid-colour PNG image and return its raw bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def red_image_bytes() -> bytes:
    """100×100 solid red PNG — baseline for pHash tests."""
    return _make_image(100, 100, (255, 0, 0))


@pytest.fixture
def red_image_bytes_resized() -> bytes:
    """200×200 solid red PNG — same content, different resolution."""
    return _make_image(200, 200, (255, 0, 0))


@pytest.fixture
def blue_image_bytes() -> bytes:
    """100×100 solid blue PNG — visually distinct from red."""
    return _make_image(100, 100, (0, 0, 255))


@pytest.fixture
def gradient_image_bytes() -> bytes:
    """100×100 gradient image — more realistic content for pHash."""
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        for y in range(100):
            img.putpixel((x, y), (x * 2, y * 2, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiny_image_bytes() -> bytes:
    """4×4 PNG — extremely small but valid."""
    return _make_image(4, 4, (128, 128, 128))


@pytest.fixture
def corrupt_bytes() -> bytes:
    """Not a valid image — should trigger UnidentifiedImageError."""
    return b"this is not an image at all"


# ── Model fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_job() -> IngestJob:
    """A realistic IngestJob for pipeline tests."""
    return IngestJob(
        message_id="111222333",
        channel_id="999888777",
        guild_id="444555666",
        image_url="https://cdn.discordapp.com/attachments/test/image.png",
        caption="when the code compiles",
    )


@pytest.fixture
def sample_document() -> MemeDocument:
    """A realistic MemeDocument with all fields populated."""
    return MemeDocument(
        message_id="111222333",
        channel_id="999888777",
        guild_id="444555666",
        image_url="https://cdn.discordapp.com/attachments/test/image.png",
        img_hash="a1b2c3d4e5f6a7b8",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        caption="when the code compiles",
        ocr_text="laughing face",
        embedding=[0.1] * 512,
    )


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """A list of search results for cache/router tests."""
    return [
        SearchResult(
            message_id="111",
            image_url="https://example.com/1.png",
            searchable_text="doge meme",
            score=0.95,
        ),
        SearchResult(
            message_id="222",
            image_url="https://example.com/2.png",
            searchable_text="cat meme",
            score=0.82,
        ),
    ]


# ── Embedding helpers ────────────────────────────────────────────────────

@pytest.fixture
def random_embedding() -> list[float]:
    """A random 512-dim embedding (not normalised)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(512).tolist()


@pytest.fixture
def zero_embedding() -> list[float]:
    """The fallback zero embedding used when HF is unavailable."""
    return [0.0] * 512
