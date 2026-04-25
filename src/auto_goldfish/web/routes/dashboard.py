"""Dashboard route -- list saved decks."""

from __future__ import annotations

import json
import os

from flask import Blueprint, render_template

from auto_goldfish.decklist.loader import get_deckpath

bp = Blueprint("dashboard", __name__)

_DECK_DESCRIPTIONS: dict[str, str] = {
    "mana-starved-demo": "18 lands with CMC 5-7 spells. Severely under-landed — the optimizer should strongly recommend adding lands.",
    "overlanded-cantrips-demo": "45 lands with all CMC 1 spells. Way too many lands — the optimizer should recommend cutting lands.",
    "equilibrium-demo": "37 lands with uniform CMC 2 spells. Already near-optimal — changes should have negligible effect.",
}

# Pedagogical order: under-landed → balanced → over-landed.
_DEMO_DECK_ORDER: list[str] = [
    "mana-starved-demo",
    "equilibrium-demo",
    "overlanded-cantrips-demo",
]


def _deck_sort_key(name: str) -> tuple[int, str]:
    """Sort demo decks in pedagogical order, then everything else alphabetically."""
    if name in _DEMO_DECK_ORDER:
        return (0, f"{_DEMO_DECK_ORDER.index(name):03d}")
    return (1, name)


def _list_saved_decks() -> list[dict]:
    """Return metadata for each saved deck."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )
    decks_dir = os.path.join(project_root, "decks")
    if not os.path.isdir(decks_dir):
        return []

    decks = []
    for name in sorted(os.listdir(decks_dir), key=_deck_sort_key):
        deck_json = os.path.join(decks_dir, name, f"{name}.json")
        if os.path.isfile(deck_json):
            try:
                with open(deck_json) as f:
                    cards = json.load(f)
                card_count = len(cards)
                commanders = [c["name"] for c in cards if c.get("commander")]
                land_count = sum(
                    1 for c in cards if "Land" in c.get("types", [])
                )
            except (json.JSONDecodeError, KeyError):
                card_count = 0
                commanders = []
                land_count = 0
            decks.append({
                "name": name,
                "card_count": card_count,
                "commanders": commanders,
                "land_count": land_count,
                "description": _DECK_DESCRIPTIONS.get(name),
            })
    return decks


@bp.route("/")
def index():
    decks = _list_saved_decks()
    return render_template("dashboard.html", decks=decks)
