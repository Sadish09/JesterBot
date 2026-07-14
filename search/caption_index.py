"""
search/caption_index.py — In-memory caption store for fast snake_case matching.

Responsibility:
    Maintains a normalised caption index in memory for instant substring
    matching.  Fed by the ingest pipeline on each new meme, queried by
    the search router as Phase 0 (before Redis, Atlas, or FAISS).

    Normalisation collapses underscores, hyphens, and whitespace into
    single spaces and lowercases everything:
        "doge_meme"     → "doge meme"
        "Surprised Cat" → "surprised cat"
        "yeet--yolo"    → "yeet yolo"

    Matching is substring-based: query "doge" matches caption
    "doge meme lol" because "doge" is contained in the normalised form.

Blast radius on failure:
    LOW.  If the caption index fails, the search router simply skips
    Phase 0 and falls through to the existing phases.  No memes are
    lost — the index is purely a fast-path optimisation.  The index
    is ephemeral (in-memory only, lost on restart).
"""

from __future__ import annotations

import re

from core.logging import get_logger
from core.models import SearchResult

log = get_logger("search.caption")

# Regex that matches one or more underscores, hyphens, or whitespace characters
_SEPARATOR_RE = re.compile(r"[\s_-]+")


def _normalise(text: str) -> str:
    """
    _normalise(text: str) -> str

    Collapse separators (underscores, hyphens, whitespace) to single
    spaces and lowercase.  Used to make "doge_meme" match "doge meme".

    On failure: never fails — pure string manipulation.
    """
    return _SEPARATOR_RE.sub(" ", text.strip()).lower()


class CaptionIndex:
    """
    CaptionIndex() -> CaptionIndex

    In-memory caption → meme lookup for fast snake_case matching.
    Thread-safe for single-writer (ingest pipeline) + multi-reader
    (search router) on the asyncio event loop.

    On failure: construction never fails (empty list).
    """

    def __init__(self) -> None:
        # Each entry: (normalised_caption, message_id, image_url)
        self._entries: list[tuple[str, str, str]] = []
        # Quick dedup to avoid storing the same message_id twice
        self._seen_ids: set[str] = set()

    def add(self, message_id: str, image_url: str, caption: str) -> None:
        """
        add(message_id: str, image_url: str, caption: str) -> None

        Index a meme's caption for future search.  Skips entries with
        empty captions or duplicate message_ids.  The caption is
        normalised before storage.

        On failure: never fails.  Silently skips empty/duplicate entries.
        """
        if not caption or not caption.strip():
            return
        if message_id in self._seen_ids:
            return

        normalised = _normalise(caption)
        if not normalised:
            return

        self._entries.append((normalised, message_id, image_url))
        self._seen_ids.add(message_id)
        log.debug("caption_indexed", message_id=message_id, caption=normalised)

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        """
        search(query: str, limit: int = 3) -> list[SearchResult]

        Find memes whose normalised caption contains the normalised query
        as a substring.  Returns up to `limit` results, scored by how
        closely the caption length matches the query length (shorter
        captions = tighter match = higher score).

        On failure: never fails.  Returns empty list on empty index or
        empty query.
        """
        q = _normalise(query)
        if not q:
            return []

        hits: list[tuple[float, str, str, str]] = []
        for caption, message_id, image_url in self._entries:
            if q in caption:
                # Score: ratio of query length to caption length
                # Exact match = 1.0, partial match < 1.0
                score = len(q) / len(caption) if caption else 0.0
                hits.append((score, message_id, image_url, caption))

        # Sort by score descending (tightest matches first)
        hits.sort(key=lambda h: h[0], reverse=True)

        return [
            SearchResult(
                message_id=message_id,
                image_url=image_url,
                searchable_text=caption,
                score=round(score, 3),
            )
            for score, message_id, image_url, caption in hits[:limit]
        ]

    @property
    def count(self) -> int:
        """
        count -> int

        Number of captions currently indexed.

        On failure: never fails.
        """
        return len(self._entries)
