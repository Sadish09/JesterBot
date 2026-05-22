"""
core/models.py — Domain models shared across the codebase.

Responsibility:
    Defines the three data structures that travel between packages:
    IngestJob (queue ↔ pipeline), MemeDocument (pipeline ↔ DB ↔ FAISS),
    and SearchResult (search ↔ bot commands).  Plain dataclasses — no
    ORM coupling.  The MongoDB layer converts to/from these when
    reading or writing documents.

Blast radius on failure:
    TOTAL.  Every package imports at least one model from here.
    A broken dataclass definition (e.g. wrong field type, missing
    import) prevents the entire process from starting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(slots=True)
class IngestJob:
    """
    IngestJob(message_id, channel_id, guild_id, image_url, caption, timestamp) -> IngestJob

    Lightweight job that travels through the asyncio ingest queue.
    Created by the on_message event handler, consumed by queue workers.
    Contains only the metadata needed to kick off the ingest pipeline —
    no image bytes, no embeddings.

    On failure: dataclass construction raises TypeError if required
    fields are missing.  timestamp defaults to utcnow if omitted.
    """

    message_id: str          # Discord snowflake
    channel_id: str
    guild_id: str
    image_url: str
    caption: str             # user message text (may be empty)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class MemeDocument:
    """
    MemeDocument(message_id, channel_id, guild_id, image_url, img_hash,
                 timestamp, caption, ocr_text, searchable_text,
                 embedding, mongo_id) -> MemeDocument

    Mirrors the MongoDB document schema from the plan.  Used as the
    transfer object between the ingest pipeline and the DB layer.
    ``_id`` is assigned by Mongo on insert; we use ``message_id`` as
    the unique business key.

    On failure: construction raises TypeError on missing required fields.
    ``build_searchable_text()`` never fails — it gracefully handles
    empty caption / ocr_text.
    """

    message_id: str
    channel_id: str
    guild_id: str
    image_url: str
    img_hash: str                          # DCT pHash hex
    timestamp: datetime
    caption: str = ""
    ocr_text: str = ""
    searchable_text: str = ""              # caption + " " + ocr_text
    embedding: list[float] = field(default_factory=list)
    mongo_id: Optional[str] = None         # _id from MongoDB

    def build_searchable_text(self) -> None:
        """
        build_searchable_text() -> None

        Merge caption and OCR text into the searchable_text field.
        This field is what Atlas Search indexes for full-text queries.
        Filters out empty strings so you don't get leading/trailing spaces.

        On failure: never fails.  If both caption and ocr_text are empty,
        searchable_text becomes an empty string.
        """
        parts = [p for p in (self.caption, self.ocr_text) if p]
        self.searchable_text = " ".join(parts)


@dataclass(slots=True)
class SearchResult:
    """
    SearchResult(message_id, image_url, searchable_text, score) -> SearchResult

    A single hit returned to the Discord user via the /find command.
    Produced by both the text search and vector search paths.

    On failure: construction raises TypeError on missing required fields.
    score defaults to 0.0 if omitted.
    """

    message_id: str
    image_url: str
    searchable_text: str
    score: float = 0.0
