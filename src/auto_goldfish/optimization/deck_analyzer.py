"""Deck composition extraction for the hypergeometric mana model.

Extracts land count, CMC distribution, ramp/draw counts from card dicts
and the effect registry -- no Goldfisher dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auto_goldfish.effects.registry import CardEffects, EffectRegistry


@dataclass
class DeckComposition:
    """Summary of a deck's composition relevant to mana modelling."""

    deck_size: int = 99
    land_count: int = 0
    mdfc_count: int = 0
    cmc_distribution: Dict[int, int] = field(default_factory=dict)
    avg_cmc: float = 0.0
    ramp_cards: int = 0
    draw_cards: int = 0
    commander_cmc: int = 0
    commander_name: str = ""


def analyze_deck_composition(
    deck_list: List[Dict[str, Any]],
    registry: Optional[EffectRegistry] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DeckComposition:
    """Extract deck composition from a list of card dicts.

    Parameters
    ----------
    deck_list : list of card dicts (as stored in decks/<name>/<name>.json)
    registry : EffectRegistry for identifying ramp/draw cards
    overrides : user overrides dict {card_name: override_data}
    """
    if overrides is None:
        overrides = {}

    land_count = 0
    mdfc_count = 0
    cmc_counts: Counter[int] = Counter()
    ramp_cards = 0
    draw_cards = 0
    commander_cmc = 0
    commander_name = ""
    total_cmc = 0
    nonland_count = 0

    for card in deck_list:
        types = card.get("types", [])
        is_land = "Land" in types
        is_spell = any(t in types for t in ["Creature", "Artifact", "Enchantment",
                                             "Instant", "Sorcery", "Planeswalker", "Battle"])
        name = card.get("name", "")
        cmc = card.get("cmc", 0)
        qty = card.get("quantity", 1)

        if card.get("commander", False):
            commander_cmc = cmc
            commander_name = name

        if is_land:
            land_count += qty
            if is_spell:
                # MDFC: count as land but also note it
                mdfc_count += qty
            continue

        # Non-land card
        nonland_count += qty
        cmc_counts[cmc] += qty
        total_cmc += cmc * qty

        # Check if ramp or draw via registry or overrides
        is_ramp, is_draw = _classify_card(name, registry, overrides)
        if is_ramp:
            ramp_cards += qty
        if is_draw:
            draw_cards += qty

    deck_size = land_count + nonland_count
    avg_cmc = total_cmc / nonland_count if nonland_count > 0 else 0.0

    return DeckComposition(
        deck_size=deck_size,
        land_count=land_count,
        mdfc_count=mdfc_count,
        cmc_distribution=dict(cmc_counts),
        avg_cmc=round(avg_cmc, 2),
        ramp_cards=ramp_cards,
        draw_cards=draw_cards,
        commander_cmc=commander_cmc,
        commander_name=commander_name,
    )


def _classify_card(
    name: str,
    registry: Optional[EffectRegistry],
    overrides: Dict[str, Any],
) -> tuple[bool, bool]:
    """Determine if a card is ramp and/or draw.

    Checks user overrides first, then the effect registry.
    """
    is_ramp = False
    is_draw = False

    # Check user overrides
    if name in overrides:
        override = overrides[name]
        categories = override.get("categories", [])
        for cat in categories:
            if cat.get("category") == "ramp":
                is_ramp = True
            elif cat.get("category") == "draw":
                is_draw = True
        return is_ramp, is_draw

    # Check registry
    if registry is not None:
        effects: CardEffects | None = registry.get(name)
        if effects is not None:
            is_ramp = effects.ramp
            is_draw = effects.draw

    return is_ramp, is_draw
