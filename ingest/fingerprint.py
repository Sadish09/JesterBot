"""
ingest/fingerprint.py — DCT-based perceptual hash (pHash) for image dedup.

Responsibility:
    Computes a 64-bit perceptual hash for an image using the standard
    pHash algorithm: 32×32 greyscale resize → 2D DCT → top-left 8×8
    low-frequency coefficients → median threshold → 64-bit hash.
    This hash is resilient to JPEG re-compression, minor crops, and
    colour shifts — exactly what you want for meme reposts.

    Also provides hamming_distance() to compare two hashes.  The
    PHASH_HAMMING_THRESHOLD config knob controls how close two hashes
    must be to be considered duplicates.

Blast radius on failure:
    LOW.  If hashing fails for a single image (corrupt bytes, truncated
    download), the ingest pipeline skips only that meme.  If the entire
    module is broken (scipy missing, import error), all new ingests
    fail but the bot stays online and search works on existing data.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.fft import dctn

from core.logging import get_logger

log = get_logger("ingest.fingerprint")

_HASH_SIZE = 8       # 8×8 = 64-bit hash
_RESIZE_DIM = 32     # resize image to 32×32 before DCT


def compute_phash(image_bytes: bytes) -> str:
    """
    compute_phash(image_bytes: bytes) -> str

    Compute a 64-bit DCT perceptual hash and return it as a 16-char
    hex string.

    On failure:
    - Raises PIL.UnidentifiedImageError if the bytes are not a valid
      image.
    - Raises ValueError if the image is too small to resize.
    The ingest pipeline runs this in an executor and catches exceptions
    to skip corrupt images.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # greyscale
    img = img.resize((_RESIZE_DIM, _RESIZE_DIM), Image.Resampling.LANCZOS)

    pixels = np.asarray(img, dtype=np.float64)

    # 2D DCT
    dct_full = dctn(pixels, type=2, norm="ortho")

    # Keep only the top-left 8×8 (low-frequency coefficients)
    dct_low = dct_full[:_HASH_SIZE, :_HASH_SIZE]

    # Threshold at the median (excluding DC coefficient)
    median = np.median(dct_low[1:, 1:])  # skip [0,0] — DC is always huge
    bits = (dct_low > median).flatten()

    # Pack into a 64-bit integer → hex string
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)

    hex_str = f"{hash_int:016x}"
    log.debug("phash_computed", hash=hex_str)
    return hex_str


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """
    hamming_distance(hash_a: str, hash_b: str) -> int

    Compute the Hamming distance between two hex pHash strings.
    Returns the number of differing bits (0 = identical, 64 = totally
    different).  Compare against PHASH_HAMMING_THRESHOLD from config
    to decide if two images are duplicates.

    On failure: raises ValueError if either hash is not a valid hex
    string.  This should never happen if both hashes came from
    compute_phash().
    """
    int_a = int(hash_a, 16)
    int_b = int(hash_b, 16)
    xor = int_a ^ int_b
    return bin(xor).count("1")
