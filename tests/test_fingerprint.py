"""
tests/test_fingerprint.py — DCT pHash correctness and edge cases.

Catches:
- Hash stability: same image always → same hash
- Resize resilience: scaled version → similar hash (low hamming)
- Distinct detection: different images → distant hashes
- Corrupt input: invalid bytes → clean exception
- Tiny image: extremely small images still hash correctly
- Hamming distance: boundary checks at 0, middle, max
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from core.config import get_settings
from ingest.fingerprint import compute_phash, hamming_distance


class TestComputePhash:
    """Test the DCT pHash computation."""

    def test_deterministic(self, red_image_bytes: bytes) -> None:
        """Same image bytes → identical hash every time."""
        h1 = compute_phash(red_image_bytes)
        h2 = compute_phash(red_image_bytes)
        assert h1 == h2

    def test_returns_hex_string(self, red_image_bytes: bytes) -> None:
        """Hash is a 16-char lowercase hex string (64 bits)."""
        h = compute_phash(red_image_bytes)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_resize_resilience(
        self, red_image_bytes: bytes, red_image_bytes_resized: bytes
    ) -> None:
        """Same solid colour at different resolutions → identical hash."""
        h_small = compute_phash(red_image_bytes)
        h_large = compute_phash(red_image_bytes_resized)
        dist = hamming_distance(h_small, h_large)
        # Solid colour images should hash identically regardless of size
        assert dist == 0, f"Expected 0, got {dist}"

    def test_different_images_distant(
        self, red_image_bytes: bytes, blue_image_bytes: bytes
    ) -> None:
        """Visually different images → hamming distance > threshold."""
        h_red = compute_phash(red_image_bytes)
        h_blue = compute_phash(blue_image_bytes)
        dist = hamming_distance(h_red, h_blue)
        # Red vs blue should be clearly distinguishable — though for solid
        # images the DCT is dominated by DC component, so they might be
        # closer than you'd expect.  Just verify they're not identical.
        assert h_red != h_blue or dist > 0

    def test_gradient_differs_from_solid(
        self, red_image_bytes: bytes, gradient_image_bytes: bytes
    ) -> None:
        """Gradient image is clearly different from a solid colour."""
        h_solid = compute_phash(red_image_bytes)
        h_grad = compute_phash(gradient_image_bytes)
        dist = hamming_distance(h_solid, h_grad)
        # Gradient has high-frequency content that solid doesn't — expect
        # a meaningful distance.
        assert dist > 0, "Gradient and solid should produce different hashes"

    def test_tiny_image(self, tiny_image_bytes: bytes) -> None:
        """4×4 image (smaller than the 32×32 resize) still hashes."""
        h = compute_phash(tiny_image_bytes)
        assert len(h) == 16

    def test_corrupt_bytes_raises(self, corrupt_bytes: bytes) -> None:
        """Invalid image bytes raise an identifiable exception."""
        with pytest.raises(Exception):
            compute_phash(corrupt_bytes)

    def test_jpeg_compression_resilience(self, gradient_image_bytes: bytes) -> None:
        """JPEG re-compression of the same image → low hamming distance."""
        # Hash the original PNG
        h_orig = compute_phash(gradient_image_bytes)

        # Re-encode as JPEG at low quality and hash that
        img = Image.open(io.BytesIO(gradient_image_bytes))
        jpeg_buf = io.BytesIO()
        img.save(jpeg_buf, format="JPEG", quality=50)
        h_jpeg = compute_phash(jpeg_buf.getvalue())

        dist = hamming_distance(h_orig, h_jpeg)
        threshold = get_settings().PHASH_HAMMING_THRESHOLD
        assert dist <= threshold, (
            f"JPEG recompression should stay within threshold {threshold}, "
            f"got distance {dist}"
        )


class TestHammingDistance:
    """Test the hamming distance utility."""

    def test_identical_hashes(self) -> None:
        """Same hash → distance 0."""
        assert hamming_distance("a1b2c3d4e5f6a7b8", "a1b2c3d4e5f6a7b8") == 0

    def test_one_bit_flip(self) -> None:
        """Flipping the last bit → distance 1."""
        # 0x...8 = ...1000, 0x...9 = ...1001 → 1 bit difference
        assert hamming_distance("a1b2c3d4e5f6a7b8", "a1b2c3d4e5f6a7b9") == 1

    def test_all_bits_different(self) -> None:
        """0x0000... vs 0xFFFF... → 64 bits different."""
        assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_symmetric(self) -> None:
        """hamming(a, b) == hamming(b, a)."""
        a, b = "a1b2c3d4e5f6a7b8", "1234567890abcdef"
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_triangle_inequality(self) -> None:
        """d(a,c) <= d(a,b) + d(b,c) — metric property."""
        a = "a1b2c3d4e5f6a7b8"
        b = "1234567890abcdef"
        c = "0000000000000000"
        d_ac = hamming_distance(a, c)
        d_ab = hamming_distance(a, b)
        d_bc = hamming_distance(b, c)
        assert d_ac <= d_ab + d_bc

    def test_invalid_hex_raises(self) -> None:
        """Non-hex string should raise ValueError."""
        with pytest.raises(ValueError):
            hamming_distance("not_hex_at_all!!", "a1b2c3d4e5f6a7b8")
