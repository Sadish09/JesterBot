"""
search/vector_search.py — In-process FAISS vector index.

Responsibility:
    Wraps a FAISS IndexFlatIP (inner product = cosine similarity on
    L2-normalised vectors) with an ID mapping layer.  Provides bulk
    build on startup (from MongoDB embeddings), incremental O(1) add
    on each ingest, and async search that runs in a thread executor
    to keep the event loop free during numpy/BLAS work.

    Switch to IndexHNSWFlat when the corpus exceeds ~50k memes and
    query latency starts to degrade.

    Memory estimates:
        10k  memes × 512 dims × 4 bytes ≈  20 MB
        100k memes × 512 dims × 4 bytes ≈ 200 MB

Blast radius on failure:
    MEDIUM.  If FAISS is broken (import error, OOM):
    - The bot cannot do vector/semantic search.  /find falls back to
      text-only search via Atlas (still useful).
    - Ingest pipeline's FAISS add step fails silently — memes are
      still persisted in MongoDB but not vector-indexed.
    If FAISS search returns bad results (wrong IDs), users get
    irrelevant memes — annoying but not dangerous.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import faiss
import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("search.vector")


class FAISSIndex:
    """
    FAISSIndex(dim: int = None) -> FAISSIndex

    Thin wrapper around a FAISS flat inner-product index with an
    int-to-string ID mapping.  Uses EMBEDDING_DIM from config if
    dim is not provided.

    On failure: construction raises if faiss.IndexFlatIP fails
    (invalid dimension).  In practice this never happens with a
    valid config.
    """

    def __init__(self, dim: int | None = None) -> None:
        settings = get_settings()
        self._dim = dim or settings.EMBEDDING_DIM
        self._index = faiss.IndexFlatIP(self._dim)
        # FAISS uses sequential int IDs internally — we map them to Mongo _id / message_id strings
        self._id_map: list[str] = []

    # ── Bulk load (startup) ──────────────────────────────────────────────

    def build_from_documents(self, docs: list[tuple[str, list[float]]]) -> None:
        """
        build_from_documents(docs: list[tuple[str, list[float]]]) -> None

        Rebuild the entire index from (doc_id, embedding) tuples.
        Called once on startup with data from
        MongoDB.fetch_all_embeddings().  Resets any existing index
        contents.  All vectors are L2-normalised before adding so
        inner product computes cosine similarity.

        On failure: raises ValueError if embeddings have wrong
        dimensionality.  If docs is empty, logs and returns — the
        index stays empty but usable.
        """
        if not docs:
            log.info("faiss_build_empty", detail="No documents to index")
            return

        ids, embeddings = zip(*docs)
        matrix = np.array(embeddings, dtype=np.float32)

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(matrix)

        self._index.reset()
        self._index.add(matrix)
        self._id_map = list(ids)

        log.info("faiss_built", count=len(ids), dim=self._dim)

    # ── Incremental update (on ingest) ───────────────────────────────────

    def add(self, embedding: list[float], doc_id: str) -> None:
        """
        add(embedding: list[float], doc_id: str) -> None

        Add a single vector to the index.  O(1) for IndexFlatIP.
        The vector is L2-normalised before adding.  The doc_id is
        appended to the ID map for later retrieval.

        On failure: raises ValueError if embedding dimensionality
        doesn't match the index.  The ingest pipeline does not catch
        this — it would indicate a bug in the HF Space response.
        """
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._id_map.append(doc_id)
        log.debug("faiss_added", doc_id=doc_id, total=self._index.ntotal)

    # ── Search ───────────────────────────────────────────────────────────

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        search(query_embedding: list[float], k: int = 5) -> list[tuple[str, float]]

        Find the k nearest vectors and return (doc_id, score) tuples,
        sorted by descending similarity.  The query vector is
        L2-normalised before searching.

        The actual FAISS search runs in run_in_executor() to avoid
        blocking the asyncio event loop during numpy/BLAS work.

        Returns an empty list if the index is empty.  Clamps k to
        the actual number of indexed vectors.

        On failure: raises if the query embedding has wrong
        dimensionality.  Returns empty list on empty index (not an
        error).
        """
        if self._index.ntotal == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        k_clamped = min(k, self._index.ntotal)

        loop = asyncio.get_running_loop()
        distances, indices = await loop.run_in_executor(
            None, self._index.search, vec, k_clamped
        )

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._id_map[idx], float(dist)))

        return results

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """
        count -> int

        Number of vectors currently in the index.

        On failure: never fails.
        """
        return self._index.ntotal

    @property
    def is_empty(self) -> bool:
        """
        is_empty -> bool

        True if the index contains zero vectors.

        On failure: never fails.
        """
        return self._index.ntotal == 0
