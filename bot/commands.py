"""
bot/commands.py — Slash commands (/find, /status).

Responsibility:
    Registers slash commands on the bot's command tree.  /find defers
    the response ephemerally (only visible to the invoker), runs the
    search router, and shows results with interactive "Send" buttons.
    The user picks a meme and the bot sends it publicly.  /status is a
    quick health check showing ready state and queue depth.

    Command registration is called from main.py before bot.start() to
    avoid circular imports within the bot/ package.

Blast radius on failure:
    LOW.  If register_commands() fails (decorator error, type mismatch),
    slash commands won't be available but the bot still connects and
    still ingests memes from the channel.  If a command handler throws
    at runtime, discord.py catches it — the user sees an error message
    but the bot stays online.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import discord
from discord import app_commands

from bot.views import MemePickerView, build_picker_embeds
from core.logging import get_logger

if TYPE_CHECKING:
    from bot.client import JesterBot

log = get_logger("bot.commands")


def register_commands(bot: discord.Client) -> None:
    """
    register_commands(bot: discord.Client) -> None

    Register all slash commands on the bot's command tree.  Called from
    main.py after constructing JesterBot, before bot.start().  The
    commands are synced with Discord's API in setup_hook().

    Uses duck typing to access bot.ready_flag, bot.search_router, and
    bot.ingest_queue — these are JesterBot-specific attributes.

    On failure: raises if the @tree.command decorator fails (invalid
    command name, duplicate name).  In practice this never happens
    with static definitions.
    """

    @bot.tree.command(name="find", description="Search the meme archive")  # type: ignore[attr-defined]
    @app_commands.describe(query="What meme are you looking for?")
    async def find(interaction: discord.Interaction, query: str) -> None:
        """
        find(interaction, query: str) -> None

        Search the meme archive using the multi-phase search router.
        Results are shown ephemerally (only visible to the invoker)
        with interactive "Send" buttons.  The user picks a meme and
        the bot sends it publicly to the channel.

        On failure:
        - Sends an ephemeral error message to the user if the search
          router raises an unexpected exception.
        - Sends a "not ready" message if the bot hasn't finished
          startup.
        - Never crashes the bot — all exceptions are caught.
        """
        # Defer ephemerally — only the invoker sees the "thinking..." state
        await interaction.response.defer(thinking=True, ephemeral=True)

        if not bot.ready_flag:  # type: ignore[attr-defined]
            await interaction.followup.send(
                "⏳ Bot is still starting up — try again in a moment.",
                ephemeral=True,
            )
            return

        log.info("find_invoked", user=str(interaction.user), query=query)

        try:
            results = await bot.search_router.search(query, k=3)  # type: ignore[attr-defined]
        except Exception:
            log.exception("find_search_error", query=query)
            await interaction.followup.send(
                "❌ Something went wrong during search. Try again?",
                ephemeral=True,
            )
            return

        if not results:
            await interaction.followup.send(
                f"🔍 No memes found for **{query}**. Try a different search?",
                ephemeral=True,
            )
            return

        # Build ephemeral picker with embeds and buttons
        embeds = build_picker_embeds(results, query)
        view = MemePickerView(
            results=results,
            query=query,
            invoker_id=interaction.user.id,
        )

        await interaction.followup.send(
            content=f"🔍 **Results for: {query}** — pick one to send!",
            embeds=embeds,
            view=view,
            ephemeral=True,
        )
        log.info("find_picker_shown", query=query, count=len(results))

    @bot.tree.command(name="status", description="Bot health check")  # type: ignore[attr-defined]
    async def status(interaction: discord.Interaction) -> None:
        """
        status(interaction) -> None

        Quick health check — shows whether the bot is ready and the
        current ingest queue depth.

        On failure: never fails in practice — only reads attributes
        and sends a simple embed.
        """
        embed = discord.Embed(
            title="🃏 Jester Status",
            color=discord.Color.green() if bot.ready_flag else discord.Color.orange(),  # type: ignore[attr-defined]
        )
        embed.add_field(name="Ready", value="✅" if bot.ready_flag else "⏳", inline=True)  # type: ignore[attr-defined]
        embed.add_field(
            name="Queue Depth",
            value=str(bot.ingest_queue.depth),  # type: ignore[attr-defined]
            inline=True,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
