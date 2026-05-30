"""
tests/test_models.py — Domain model correctness.

Catches:
- build_searchable_text merging logic edge cases
- Default timestamp population
- Dataclass construction with missing/extra fields
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.models import IngestJob, MemeDocument, SearchResult


class TestMemeDocument:
    """MemeDocument data integrity."""

    def test_build_searchable_text_both(self) -> None:
        """Caption + OCR → joined with space."""
        doc = MemeDocument(
            message_id="1",
            channel_id="2",
            guild_id="3",
            image_url="https://x.com/img.png",
            img_hash="abcdef0123456789",
            timestamp=datetime.now(timezone.utc),
            caption="when the code works",
            ocr_text="laughing face emoji",
        )
        doc.build_searchable_text()
        assert doc.searchable_text == "when the code works laughing face emoji"

    def test_build_searchable_text_caption_only(self) -> None:
        """Only caption, no OCR → just caption."""
        doc = MemeDocument(
            message_id="1",
            channel_id="2",
            guild_id="3",
            image_url="https://x.com/img.png",
            img_hash="abcdef0123456789",
            timestamp=datetime.now(timezone.utc),
            caption="hello",
            ocr_text="",
        )
        doc.build_searchable_text()
        assert doc.searchable_text == "hello"

    def test_build_searchable_text_ocr_only(self) -> None:
        """Only OCR, no caption → just OCR text."""
        doc = MemeDocument(
            message_id="1",
            channel_id="2",
            guild_id="3",
            image_url="https://x.com/img.png",
            img_hash="abcdef0123456789",
            timestamp=datetime.now(timezone.utc),
            caption="",
            ocr_text="some text in image",
        )
        doc.build_searchable_text()
        assert doc.searchable_text == "some text in image"

    def test_build_searchable_text_both_empty(self) -> None:
        """No caption, no OCR → empty string (not None, not whitespace)."""
        doc = MemeDocument(
            message_id="1",
            channel_id="2",
            guild_id="3",
            image_url="https://x.com/img.png",
            img_hash="abcdef0123456789",
            timestamp=datetime.now(timezone.utc),
        )
        doc.build_searchable_text()
        assert doc.searchable_text == ""

    def test_default_embedding_empty(self) -> None:
        """Default embedding is an empty list, not None."""
        doc = MemeDocument(
            message_id="1",
            channel_id="2",
            guild_id="3",
            image_url="https://x.com/img.png",
            img_hash="abcdef0123456789",
            timestamp=datetime.now(timezone.utc),
        )
        assert doc.embedding == []
        assert isinstance(doc.embedding, list)

    def test_embedding_not_shared_between_instances(self) -> None:
        """Each instance gets its own list (no mutable default sharing bug)."""
        doc_a = MemeDocument(
            message_id="1", channel_id="2", guild_id="3",
            image_url="x", img_hash="h", timestamp=datetime.now(timezone.utc),
        )
        doc_b = MemeDocument(
            message_id="2", channel_id="2", guild_id="3",
            image_url="x", img_hash="h", timestamp=datetime.now(timezone.utc),
        )
        doc_a.embedding.append(42.0)
        assert doc_b.embedding == []  # Must not leak


class TestIngestJob:
    """IngestJob construction."""

    def test_timestamp_auto_populated(self) -> None:
        """Omitting timestamp → defaults to roughly now."""
        job = IngestJob(
            message_id="1",
            channel_id="2",
            guild_id="3",
            image_url="https://x.com/img.png",
            caption="test",
        )
        assert job.timestamp is not None
        assert job.timestamp.tzinfo is not None  # must be tz-aware
        delta = (datetime.now(timezone.utc) - job.timestamp).total_seconds()
        assert abs(delta) < 5  # within 5 seconds of now

    def test_missing_required_field_raises(self) -> None:
        """Omitting a required field raises TypeError."""
        with pytest.raises(TypeError):
            IngestJob(  # type: ignore[call-arg]
                message_id="1",
                # missing channel_id, guild_id, image_url, caption
            )


class TestSearchResult:
    """SearchResult construction."""

    def test_default_score_zero(self) -> None:
        """Score defaults to 0.0 if not provided."""
        r = SearchResult(
            message_id="1",
            image_url="https://x.com/img.png",
            searchable_text="test",
        )
        assert r.score == 0.0

    def test_custom_score(self) -> None:
        """Score can be set to a custom value."""
        r = SearchResult(
            message_id="1",
            image_url="https://x.com/img.png",
            searchable_text="test",
            score=0.95,
        )
        assert r.score == 0.95
