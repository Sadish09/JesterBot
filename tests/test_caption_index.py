"""
tests/test_caption_index.py — Caption index normalisation and search.

Catches:
- Snake_case normalisation ("doge_meme" → "doge meme")
- Substring matching ("doge" matches "doge meme lol")
- Empty query / empty index → empty results
- Score ranking (tighter match = higher score)
- Dedup (same message_id not indexed twice)
- Empty caption skipped
"""

from __future__ import annotations

import pytest

from search.caption_index import CaptionIndex, _normalise


class TestNormalise:
    def test_snake_case(self) -> None:
        assert _normalise("doge_meme") == "doge meme"

    def test_mixed_case(self) -> None:
        assert _normalise("Surprised_Pikachu") == "surprised pikachu"

    def test_hyphens(self) -> None:
        assert _normalise("yeet-yolo") == "yeet yolo"

    def test_multiple_separators(self) -> None:
        assert _normalise("doge__meme--lol  wow") == "doge meme lol wow"

    def test_whitespace_strip(self) -> None:
        assert _normalise("  doge_meme  ") == "doge meme"

    def test_empty(self) -> None:
        assert _normalise("") == ""

    def test_already_normalised(self) -> None:
        assert _normalise("doge meme") == "doge meme"


class TestCaptionSearch:
    def test_exact_match(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "doge_meme")
        results = idx.search("doge_meme")
        assert len(results) == 1
        assert results[0].message_id == "msg1"
        assert results[0].image_url == "http://img/1.png"
        assert results[0].score == 1.0  # exact match

    def test_partial_match(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "doge_meme_lol")
        results = idx.search("doge")
        assert len(results) == 1
        assert results[0].score < 1.0  # partial match

    def test_query_normalisation(self) -> None:
        """Query "doge meme" should match caption "doge_meme"."""
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "doge_meme")
        results = idx.search("doge meme")
        assert len(results) == 1

    def test_case_insensitive(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "Doge_Meme")
        results = idx.search("doge meme")
        assert len(results) == 1

    def test_no_match(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "doge_meme")
        results = idx.search("cat")
        assert results == []

    def test_empty_index(self) -> None:
        idx = CaptionIndex()
        results = idx.search("doge")
        assert results == []

    def test_empty_query(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "doge_meme")
        results = idx.search("")
        assert results == []

    def test_limit(self) -> None:
        idx = CaptionIndex()
        for i in range(10):
            idx.add(f"msg{i}", f"http://img/{i}.png", f"doge_variant_{i}")
        results = idx.search("doge", limit=3)
        assert len(results) == 3

    def test_score_ranking(self) -> None:
        """Tighter match (shorter caption) scores higher."""
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "doge")  # exact match for "doge"
        idx.add("msg2", "http://img/2.png", "doge_meme_collection")  # long caption
        results = idx.search("doge")
        assert results[0].message_id == "msg1"  # exact match first
        assert results[0].score > results[1].score


class TestCaptionEdgeCases:
    def test_empty_caption_skipped(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "")
        assert idx.count == 0

    def test_whitespace_only_caption_skipped(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "   ")
        assert idx.count == 0

    def test_dedup_by_message_id(self) -> None:
        idx = CaptionIndex()
        idx.add("msg1", "http://img/1.png", "doge_meme")
        idx.add("msg1", "http://img/1.png", "doge_meme")  # duplicate
        assert idx.count == 1

    def test_count(self) -> None:
        idx = CaptionIndex()
        assert idx.count == 0
        idx.add("msg1", "http://img/1.png", "doge")
        idx.add("msg2", "http://img/2.png", "cat")
        assert idx.count == 2
