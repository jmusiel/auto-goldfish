"""Moxfield API integration for loading decklists."""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Tuple

import requests

from . import rate_limiter
from .card_resolver import resolve_cards

_API_BASE = "https://api2.moxfield.com/v3/decks/all"

# Matches Moxfield URLs like:
#   https://www.moxfield.com/decks/AbCdEf123
#   https://moxfield.com/decks/AbCdEf123
_URL_RE = re.compile(r"moxfield\.com/decks/([A-Za-z0-9_-]+)")


class MoxfieldAPIError(Exception):
    """Raised when the Moxfield API returns an error."""


def _package_version() -> str:
    try:
        return version("auto_goldfish")
    except PackageNotFoundError:
        return "dev"


def _get_user_agent() -> str:
    """Return the User-Agent for Moxfield requests.

    Defaults to identifying this project by name + GitHub URL so Moxfield
    admins can reach us. The MOXFIELD_USER_AGENT env var overrides this for
    deployments with a partner credential of their own.
    """
    override = os.environ.get("MOXFIELD_USER_AGENT", "").strip()
    if override:
        return override
    return f"auto-goldfish/{_package_version()} (+https://github.com/jmusiel/auto-goldfish)"


def _slugify(name: str) -> str:
    """Lowercase + collapse non-alphanumerics into single underscores.

    Used to turn a free-form Moxfield deck title into something safe for
    URL paths and filesystem entries (the deck name is used as both).
    Returns "" if nothing alphanumeric remains.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _extract_deck_id(deck_url: str) -> str:
    """Extract the deck ID from a Moxfield URL."""
    match = _URL_RE.search(deck_url)
    if not match:
        raise ValueError(
            f"Invalid Moxfield URL: {deck_url!r}. "
            "Expected format: https://www.moxfield.com/decks/<deck_id>"
        )
    return match.group(1)


def fetch_decklist(deck_url: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Fetch a decklist from the Moxfield API and resolve via Scryfall.

    Returns
    -------
    (suggested_name, cards)
        ``suggested_name`` is the Moxfield deck title slugified to a safe
        URL/filesystem identifier (empty string if the title has no
        alphanumeric characters). Callers may use it as a default when the
        user did not provide one.
    """
    user_agent = _get_user_agent()
    deck_id = _extract_deck_id(deck_url)

    rate_limiter.wait("moxfield")

    resp = requests.get(
        f"{_API_BASE}/{deck_id}",
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    if resp.status_code == 404:
        raise MoxfieldAPIError(f"Deck not found: {deck_id}")
    resp.raise_for_status()

    data = resp.json()

    entries: list[tuple[int, str, bool]] = []

    commanders_board = data.get("boards", {}).get("commanders", {})
    for card_key, card_data in commanders_board.get("cards", {}).items():
        name = card_data.get("card", {}).get("name", card_key)
        qty = card_data.get("quantity", 1)
        entries.append((qty, name, True))

    mainboard = data.get("boards", {}).get("mainboard", {})
    for card_key, card_data in mainboard.get("cards", {}).items():
        name = card_data.get("card", {}).get("name", card_key)
        qty = card_data.get("quantity", 1)
        entries.append((qty, name, False))

    if not entries:
        raise MoxfieldAPIError("No cards found in deck")

    suggested_name = _slugify(data.get("name") or "")
    return suggested_name, resolve_cards(entries)
