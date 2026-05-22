"""
bot/commands.py — Slash commands (/find, /status).

Responsibility:
    Registers slash commands on the bot's command tree.  /find defers
    the response immediately (Discord gives 3 seconds before the
    interaction token expires), runs the search router, and edits the
    deferred message with a rich embed of results.  /status is a
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

from core.logging import get_logger
from core.models import SearchResult

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
        Defers the response immediately to avoid Discord's 3-second
        interaction timeout, then edits with results.

        On failure:
        - Sends an ephemeral error message to the user if the search
          router raises an unexpected exception.
        - Sends a "not ready" message if the bot hasn't finished
          startup.
        - Never crashes the bot — all exceptions are caught.
        """
        # Defer immediately — we might need > 3 s for vector search
        await interaction.response.defer(thinking=True)

        if not bot.ready_flag:  # type: ignore[attr-defined]
            await interaction.followup.send(
                "⏳ Bot is still starting up — try again in a moment.",
                ephemeral=True,
            )
            return

        log.info("find_invoked", user=str(interaction.user), query=query)

        try:
            results = await bot.search_router.search(query)  # type: ignore[attr-defined]
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
            )
            return

        # Build a rich embed with the top results
        embed = _build_results_embed(query, results)
        await interaction.followup.send(embed=embed)
        log.info("find_responded", query=query, count=len(results))

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


def _build_results_embed(query: str, results: list[SearchResult]) -> discord.Embed:
    """
    _build_results_embed(query: str, results: list[SearchResult]) -> discord.Embed

    Build a Discord embed showing search results.  Displays up to 5
    results with score, text preview, and the first result's image as
    the embed image.

    On failure: never fails — operates on pure data.  If a result has
    an empty image_url, the embed simply has no image.
    """
    embed = discord.Embed(
        title=f"🔍 Results for: {query}",
        color=discord.Color.blurple(),
    )

    for i, r in enumerate(results[:5], start=1):
        # Truncate searchable text for the embed field
        text_preview = (r.searchable_text[:80] + "…") if len(r.searchable_text) > 80 else r.searchable_text
        score_str = f" (score: {r.score:.2f})" if r.score > 0 else ""

        embed.add_field(
            name=f"#{i}{score_str}",
            value=text_preview or "_no text_",
            inline=False,
        )

    # Show the first result's image as the embed thumbnail
    if results and results[0].image_url:
        embed.set_image(url=results[0].image_url)

    embed.set_footer(text=f"Showing top {min(len(results), 5)} results")
    return embed
