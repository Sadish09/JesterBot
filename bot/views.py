"""
bot/views.py — Discord UI components (interactive buttons, views).

Responsibility:
    Provides the MemePickerView: an ephemeral interactive view that
    shows search results with "Send" buttons.  Only the invoking user
    can click the buttons.  On click, the selected meme is sent publicly
    to the channel and all buttons are disabled.

    The view has a 120-second timeout — after that, buttons stop working
    and the ephemeral message fades.

Blast radius on failure:
    LOW.  If the view fails to construct (shouldn't happen — it's pure
    data), the /find command falls back to a plain embed.  If a button
    callback fails, discord.py catches it and the user sees an error —
    the bot stays online.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import discord

from core.logging import get_logger
from core.models import SearchResult

if TYPE_CHECKING:
    pass

log = get_logger("bot.views")


class MemePickerView(discord.ui.View):
    """
    MemePickerView(results, query, invoker_id) -> MemePickerView

    Ephemeral view with up to 3 "Send" buttons — one per search result.
    Only the user who invoked /find can click the buttons (enforced by
    invoker_id check in each callback).

    On click: sends the selected meme publicly to the channel, then
    disables all buttons on the ephemeral message.

    On failure: button callbacks catch all exceptions and send an
    ephemeral error.  Timeout (120s) disables buttons automatically.
    """

    def __init__(
        self,
        results: list[SearchResult],
        query: str,
        invoker_id: int,
        *,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self._results = results[:3]
        self._query = query
        self._invoker_id = invoker_id
        self._sent = False  # only allow one send

        # Dynamically add buttons for each result
        for i, result in enumerate(self._results):
            button = _SendButton(
                index=i,
                result=result,
                query=query,
                invoker_id=invoker_id,
                view_ref=self,
            )
            self.add_item(button)

    async def on_timeout(self) -> None:
        """
        on_timeout() -> None

        Called when the view times out (120s).  Disables all buttons.

        On failure: never fails — just clears items.
        """
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True
        # We can't edit the message from here without a reference,
        # but the buttons will stop responding to clicks.


class _SendButton(discord.ui.Button["MemePickerView"]):
    """
    Internal button that sends a specific meme publicly when clicked.
    """

    def __init__(
        self,
        index: int,
        result: SearchResult,
        query: str,
        invoker_id: int,
        view_ref: MemePickerView,
    ) -> None:
        # Emoji labels for visual clarity
        labels = ["1️⃣ Send", "2️⃣ Send", "3️⃣ Send"]
        super().__init__(
            style=discord.ButtonStyle.primary,
            label=labels[index] if index < len(labels) else f"Send #{index + 1}",
        )
        self._index = index
        self._result = result
        self._query = query
        self._invoker_id = invoker_id
        self._view_ref = view_ref

    async def callback(self, interaction: discord.Interaction) -> None:
        """
        callback(interaction) -> None

        Send the selected meme publicly to the channel.  Only the
        invoking user can click this button.

        On failure: sends an ephemeral error message and logs the exception.
        """
        # ── Permission check: only the invoker can send ──────────────────
        if interaction.user.id != self._invoker_id:
            await interaction.response.send_message(
                "❌ Only the person who ran `/find` can pick a meme.",
                ephemeral=True,
            )
            return

        # ── Prevent double-send ──────────────────────────────────────────
        if self._view_ref._sent:
            await interaction.response.send_message(
                "Already sent a meme from this search.",
                ephemeral=True,
            )
            return
        self._view_ref._sent = True

        # ── Send the meme publicly ───────────────────────────────────────
        try:
            embed = discord.Embed(
                title=f"🃏 {self._query}",
                color=discord.Color.blurple(),
            )
            if self._result.image_url:
                embed.set_image(url=self._result.image_url)
            if self._result.searchable_text:
                text = self._result.searchable_text
                if len(text) > 100:
                    text = text[:100] + "…"
                embed.description = text

            # Send publicly to the channel
            assert interaction.channel is not None
            await interaction.channel.send(embed=embed)  # type: ignore[union-attr]

            # Update the ephemeral message — disable all buttons, show confirmation
            for item in self._view_ref.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True

            await interaction.response.edit_message(
                content=f"✅ Sent result #{self._index + 1}!",
                view=self._view_ref,
            )

            log.info(
                "meme_picked",
                user=str(interaction.user),
                query=self._query,
                result_index=self._index,
                message_id=self._result.message_id,
            )

        except Exception:
            log.exception("meme_pick_failed", query=self._query, index=self._index)
            try:
                await interaction.response.send_message(
                    "❌ Something went wrong sending the meme.",
                    ephemeral=True,
                )
            except discord.InteractionResponded:
                pass


def build_picker_embeds(results: list[SearchResult], query: str) -> list[discord.Embed]:
    """
    build_picker_embeds(results: list[SearchResult], query: str) -> list[discord.Embed]

    Build up to 3 embeds, one per search result, for the ephemeral picker.
    Each embed shows the result number, score, text preview, and image.

    On failure: never fails — operates on pure data.
    """
    embeds: list[discord.Embed] = []

    for i, r in enumerate(results[:3], start=1):
        embed = discord.Embed(
            color=discord.Color.blurple(),
        )

        # Title with result number and score
        score_str = f" — score: {r.score:.2f}" if r.score > 0 else ""
        embed.title = f"#{i}{score_str}"

        # Text preview
        if r.searchable_text:
            text = r.searchable_text
            if len(text) > 120:
                text = text[:120] + "…"
            embed.description = text
        else:
            embed.description = "_no caption_"

        # Image
        if r.image_url:
            embed.set_image(url=r.image_url)

        embeds.append(embed)

    return embeds
