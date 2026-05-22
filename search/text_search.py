"""
search/text_search.py — Text search wrapper over MongoDB Atlas Search.

Responsibility:
    Thin wrapper that delegates to MongoDB.text_search().  Exists as a
    separate module so the search router has a uniform interface and so
    text-search-specific logic (e.g. query sanitisation) can live here
    without polluting the DB layer.

Blast radius on failure:
    LOW.  If text search fails (MongoDB stub, Atlas index missing,
    network error), the search router falls through to vector search.
    The user still gets results, just from FAISS instead of Atlas.
    If both text and vector search fail, the user gets an empty result
    set with a friendly message.
"""

from __future__ import annotations

from core.logging import get_logger
from core.models import SearchResult
from integrations.db import MongoDB

log = get_logger("search.text")


class TextSearch:
    """
    TextSearch(db: MongoDB) -> TextSearch

    Atlas full-text search on the searchable_text field.
    Delegates entirely to the MongoDB.text_search() stub.

    On failure: construction never fails (just stores the db reference).
    """

    def __init__(self, db: MongoDB) -> None:
        self._db = db

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """
        search(query: str, limit: int = 5) -> list[SearchResult]

        Run Atlas full-text search and return ranked results.  Returns
        an empty list (never None) when there are no matches or when
        the DB stub hasn't been implemented yet.

        On failure:
        - Returns [] if MongoDB.text_search() raises NotImplementedError
          (stub not implemented yet).
        - Returns [] and logs exception on any other error (network,
          Atlas Search index missing, etc.).
        - Never raises — the search router depends on this.
        """
        try:
            return await self._db.text_search(query, limit=limit)
        except NotImplementedError:
            log.warning(
                "text_search_stub",
                detail="MongoDB text_search not implemented — returning empty",
            )
            return []
        except Exception:
            log.exception("text_search_failed", query=query)
            return []
