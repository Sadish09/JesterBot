"""
core/config.py — Centralised configuration loaded from environment variables.

Responsibility:
    Provides a single validated Settings object that every other module
    reads for connection strings, feature flags, and tunable knobs.
    Uses pydantic-settings so invalid / missing values blow up at import
    time, not at 3 AM when the bot first touches a key.

Blast radius on failure:
    TOTAL.  If this module fails to load (missing required env var,
    pydantic validation error), the process crashes on startup before
    any connections are opened.  Every module depends on get_settings().
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Settings(**env) -> Settings

    All env-driven configuration lives here.  Instantiated once via
    get_settings() and cached for the process lifetime.

    On failure: raises pydantic ValidationError at construction if any
    required field is missing or has an invalid type.
    """

    # ── Discord ──────────────────────────────────────────────────────────
    DISCORD_TOKEN: str
    DISCORD_MEME_CHANNEL_ID: int  # snowflake of the sendables channel
    DISCORD_GUILD_ID: int = 0     # set for instant dev sync, 0 = global sync

    # ── MongoDB (friend implements the client) ───────────────────────────
    MONGODB_URI: str = "mongodb://localhost:27017/jester"

    # ── Redis (optional — set to "" to disable caching) ────────────────
    REDIS_URL: str = ""

    # ── HF Spaces (friend implements the client) ─────────────────────────
    HF_SPACES_URL: str
    HF_API_TOKEN: str

    # ── Tunable knobs ────────────────────────────────────────────────────
    INGEST_WORKER_COUNT: int = 3
    SEARCH_TOP_K: int = 5
    IMAGE_SIZE_LIMIT_MB: int = 8
    QUERY_CACHE_TTL: int = 300         # seconds
    HF_WARMUP_INTERVAL: int = 240      # seconds between keep-warm pings
    DEDUP_TTL_DAYS: int = 30           # Redis dedup key expiry
    PHASH_HAMMING_THRESHOLD: int = 10  # max hamming distance to consider duplicate
    INGEST_JOB_MAX_AGE: int = 60       # seconds — drop stale jobs

    # ── FAISS ────────────────────────────────────────────────────────────
    EMBEDDING_DIM: int = 512           # CLIP ViT-B/32 output dimensionality

    # ── Environment flag ─────────────────────────────────────────────────
    ENV: str = "development"           # "production" on Railway

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    get_settings() -> Settings

    Return the singleton Settings instance, cached after first call.
    Every module calls this instead of constructing Settings directly.

    On failure: raises pydantic ValidationError on first call if
    required env vars (DISCORD_TOKEN, DISCORD_MEME_CHANNEL_ID) are
    missing.  Subsequent calls return the cached instance.
    """
    return Settings()  # type: ignore[call-arg]
