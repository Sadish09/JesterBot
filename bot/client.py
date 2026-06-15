"""
bot/client.py: Discord bot client.

Responsibility:
    Defines JesterBot, a discord.Client subclass that holds references
    to all shared resources (ingest queue, search router, ready flag).
    Provides the command tree for slash commands and the setup_hook
    for syncing them with Discord's API.

    This module does NOT contain event handlers or command definitions
    those live in events.py and commands.py.  Registration of commands
    and events is done from main.py

Blast radius on failure:
    NUCLEAR. If this module fails to load, the bot cannot start.
    If JesterBot construction fails (bad intents config), the process
    crashes on startup.  If setup_hook fails (command sync error),
    slash commands won't be available but the bot still connects and
    can listen for messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
from discord import app_commands

from core.logging import get_logger

if TYPE_CHECKING:
    from ingest.queue import IngestQueue
    from search.router import SearchRouter

log = get_logger("bot")


class JesterBot(discord.Client):
    """
    JesterBot(ingest_queue, search_router, meme_channel_id) -> JesterBot

    Meme search bot. Listens to a single channel, indexes
    images, and exposes a /find command.

    intents.message_content and intents.guild_messages are enabled so
    the bot can read message text and attachments in the meme channel.

    On failure: raises discord.LoginFailure if the token is invalid
    when start() is called.  Construction itself only fails if intents
    are somehow invalid (shouldn't happen with our static config).
    """

    def __init__(
        self,
        *,
        ingest_queue: IngestQueue,
        search_router: SearchRouter,
        meme_channel_id: int,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True       # required to read message text
        intents.guild_messages = True
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self.ingest_queue = ingest_queue
        self.search_router = search_router
        self.meme_channel_id = meme_channel_id

        # Flipped to True once FAISS is built and workers are running
        self.ready_flag = False

    async def setup_hook(self) -> None:
        """
        setup_hook() -> None

        Called once by discord.py before the bot connects to the
        gateway.  Syncs the command tree to a specific guild (instant)
        when DISCORD_GUILD_ID is set, or globally (up to 1 hour
        propagation) when it's 0.

        On failure: logs a warning but does NOT crash the bot.  Slash
        commands won't be available until the next successful sync,
        but the bot still connects and ingests memes.
        """
        from core.config import get_settings

        try:
            settings = get_settings()
            if settings.DISCORD_GUILD_ID:
                guild = discord.Object(id=settings.DISCORD_GUILD_ID)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                log.info("slash_commands_synced", mode="guild", guild_id=settings.DISCORD_GUILD_ID)
            else:
                await self.tree.sync()
                log.info("slash_commands_synced", mode="global")
        except Exception:
            log.exception(
                "slash_command_sync_failed",
                detail="Bot will start but commands may not be available — retry on next restart",
            )

    async def on_ready(self) -> None:
        """
        on_ready() -> None

        Called when the bot has successfully connected to Discord and
        received the READY event.  Logs the bot user and guild count.

        On failure: never fails.
        """
        log.info(
            "discord_ready",
            user=str(self.user),
            guilds=len(self.guilds),
        )
