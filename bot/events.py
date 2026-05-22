"""
bot/events.py — Discord message event handler (ingest hot-path entry point).

Responsibility:
    Attaches the on_message handler to the bot.  When a message arrives
    in the configured meme channel with image attachments, it creates
    an IngestJob and pushes it into the asyncio queue.  Returns
    immediately — Discord sees zero lag from the bot.  All heavy work
    (download, hash, HF, DB, FAISS) happens in the queue workers.

Blast radius on failure:
    LOW.  If on_message itself throws (should never happen — it's pure
    validation + queue push), discord.py catches it and logs the error.
    The bot stays online.  If the queue push fails (shouldn't — it's
    put_nowait on an unbounded queue), that single meme is lost but
    nothing else is affected.
"""

from __future__ import annotations

from datetime import timezone
from typing import TYPE_CHECKING

import discord

from core.logging import get_logger
from core.models import IngestJob

if TYPE_CHECKING:
    from bot.client import JesterBot

log = get_logger("bot.events")

# Image MIME types we accept for ingest
_IMAGE_CONTENT_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})


def setup_events(bot: discord.Client) -> None:
    """
    setup_events(bot: discord.Client) -> None

    Attach the on_message handler to the bot instance.  Called from
    main.py after constructing JesterBot.  Uses duck typing to access
    bot.meme_channel_id, bot.ready_flag, and bot.ingest_queue — these
    are JesterBot-specific attributes.

    On failure: never fails at registration time.  If the bot doesn't
    have the expected attributes, on_message will raise AttributeError
    at runtime (indicates a wiring bug in main.py).
    """

    @bot.event
    async def on_message(message: discord.Message) -> None:
        """
        on_message(message: discord.Message) -> None

        Fire-and-forget ingest entry point.  For each image attachment
        in a meme-channel message, creates an IngestJob and pushes it
        to the queue.  Returns immediately — never blocks.

        Filters applied:
        - Ignores own messages
        - Ignores messages outside the configured meme channel
        - Ignores messages received before the bot is ready
        - Ignores messages without image attachments

        On failure: discord.py catches any unhandled exception and logs
        it.  The bot stays online.  Individual attachment failures
        are impossible here since we only read metadata, not content.
        """
        # Ignore own messages
        if message.author == bot.user:
            return

        # Only listen to the configured meme channel
        if message.channel.id != bot.meme_channel_id:  # type: ignore[attr-defined]
            return

        # Must be ready (FAISS built, workers running)
        if not bot.ready_flag:  # type: ignore[attr-defined]
            return

        # Filter for image attachments
        image_attachments = [
            att for att in message.attachments
            if att.content_type and att.content_type in _IMAGE_CONTENT_TYPES
        ]

        if not image_attachments:
            return

        # Build the message caption (text the user typed alongside the image)
        caption = message.content.strip() if message.content else ""

        # Enqueue one job per image attachment
        for att in image_attachments:
            job = IngestJob(
                message_id=str(message.id),
                channel_id=str(message.channel.id),
                guild_id=str(message.guild.id) if message.guild else "",
                image_url=att.url,
                caption=caption,
                timestamp=message.created_at.replace(tzinfo=timezone.utc),
            )
            bot.ingest_queue.enqueue(job)  # type: ignore[attr-defined]
            log.info(
                "job_dispatched",
                message_id=job.message_id,
                filename=att.filename,
                size=att.size,
            )
