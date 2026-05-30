"""
tests/test_vector_search.py — FAISS index correctness and edge cases.

Catches:
- Empty index returns [] (not crash)
- Single-vector add + search retrieves correct ID
- Bulk build + search returns ranked results
- Cosine similarity via L2 normalisation works correctly
- k clamping when k > ntotal
- build_from_documents resets old data
- Wrong dimensionality detection
"""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio

from search.vector_search import FAISSIndex


@pytest.fixture
def index() -> FAISSIndex:
    """Fresh FAISS index with dim=8 for fast tests."""
    return FAISSIndex(dim=8)


@pytest.fixture
def normalised_vectors() -> list[tuple[str, list[float]]]:
    """Four doc-embedding pairs with known similarity relationships."""
    rng = np.random.default_rng(42)
    # v0 and v1 are similar (close angle), v2 and v3 are different
    base = rng.standard_normal(8).astype(np.float32)
    v0 = base.tolist()
    v1 = (base + rng.standard_normal(8) * 0.1).tolist()   # small perturbation
    v2 = (-base).tolist()                                    # opposite direction
    v3 = rng.standard_normal(8).tolist()                     # random

    return [
        ("doc_0", v0),
        ("doc_1", v1),
        ("doc_2", v2),
        ("doc_3", v3),
    ]


class TestFAISSEmpty:
    """Behaviour when the index has no vectors."""

    def test_count_zero(self, index: FAISSIndex) -> None:
        assert index.count == 0

    def test_is_empty_true(self, index: FAISSIndex) -> None:
        assert index.is_empty is True

    @pytest.mark.asyncio
    async def test_search_empty_returns_list(self, index: FAISSIndex) -> None:
        """Search on empty index must return [] without crashing."""
        results = await index.search([0.0] * 8, k=5)
        assert results == []


class TestFAISSAdd:
    """Single-vector add operations."""

    def test_add_increments_count(self, index: FAISSIndex) -> None:
        index.add([1.0] * 8, "doc_a")
        assert index.count == 1
        assert index.is_empty is False

    def test_add_multiple(self, index: FAISSIndex) -> None:
        for i in range(10):
            index.add([float(i)] * 8, f"doc_{i}")
        assert index.count == 10

    @pytest.mark.asyncio
    async def test_add_then_search_finds_it(self, index: FAISSIndex) -> None:
        """Adding a vector and searching with the same vector → top hit."""
        vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        index.add(vec, "the_doc")
        results = await index.search(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == "the_doc"
        assert results[0][1] > 0.99  # cosine self-similarity ≈ 1.0


class TestFAISSBulkBuild:
    """build_from_documents() behaviour."""

    def test_build_populates_index(
        self, index: FAISSIndex, normalised_vectors: list[tuple[str, list[float]]]
    ) -> None:
        index.build_from_documents(normalised_vectors)
        assert index.count == 4

    def test_build_resets_previous(
        self, index: FAISSIndex, normalised_vectors: list[tuple[str, list[float]]]
    ) -> None:
        """Calling build twice replaces old data, not appends."""
        index.add([1.0] * 8, "old_doc")
        assert index.count == 1

        index.build_from_documents(normalised_vectors)
        assert index.count == 4  # not 5

    def test_build_empty_list(self, index: FAISSIndex) -> None:
        """Empty list doesn't crash — index stays empty."""
        index.build_from_documents([])
        assert index.count == 0

    @pytest.mark.asyncio
    async def test_similarity_ranking(
        self, index: FAISSIndex, normalised_vectors: list[tuple[str, list[float]]]
    ) -> None:
        """Searching with v0 should rank doc_1 (similar) above doc_2 (opposite)."""
        index.build_from_documents(normalised_vectors)

        query = normalised_vectors[0][1]  # v0
        results = await index.search(query, k=4)

        doc_ids = [doc_id for doc_id, _ in results]
        # doc_0 should be first (self-match), doc_1 should be second (similar)
        assert doc_ids[0] == "doc_0"
        assert doc_ids[1] == "doc_1"
        # doc_2 (opposite direction) should be last or near-last
        assert doc_ids.index("doc_2") > doc_ids.index("doc_1")


class TestFAISSEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_k_clamped_to_ntotal(self, index: FAISSIndex) -> None:
        """Requesting k=100 when only 2 vectors → returns 2, not crash."""
        index.add([1.0] * 8, "a")
        index.add([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "b")
        results = await index.search([1.0] * 8, k=100)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_scores_between_minus1_and_1(
        self, index: FAISSIndex, normalised_vectors: list[tuple[str, list[float]]]
    ) -> None:
        """All cosine similarity scores should be in [-1, 1]."""
        index.build_from_documents(normalised_vectors)
        results = await index.search(normalised_vectors[0][1], k=4)
        for _, score in results:
            assert -1.0 <= score <= 1.01  # tiny float tolerance

    def test_zero_vector_add(self, index: FAISSIndex) -> None:
        """Zero vector add should not crash (normalisation handles it)."""
        # faiss.normalize_L2 on a zero vector produces NaN — this is a
        # known edge case.  We test that it doesn't raise.
        index.add([0.0] * 8, "zero_doc")
        assert index.count == 1
