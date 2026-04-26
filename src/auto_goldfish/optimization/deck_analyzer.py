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
    # Ramp counts bucketed by CMC (e.g. {0: 1, 1: 2, 2: 5}).
    ramp_by_cmc: Dict[int, int] = field(default_factory=dict)
    # Draw counts split by mechanic: cantrip (1-card draw on Instant/Sorcery),
    # instant_draw (one-shot multi-card draw or non-Instant/Sorcery one-shots),
    # repeatable_draw (PerTurnDraw or PerCastDraw triggers).
    draw_breakdown: Dict[str, int] = field(default_factory=dict)
    # Primary commander (first encountered) -- kept for backwards compat.
    commander_cmc: int = 0
    commander_name: str = ""
    # All commanders found, in deck order. For partner pairs both are listed.
    commander_cmcs: List[int] = field(default_factory=list)
    commander_names: List[str] = field(default_factory=list)


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
    ramp_by_cmc: Counter[int] = Counter()
    draw_breakdown: Counter[str] = Counter()
    commander_cmcs: List[int] = []
    commander_names: List[str] = []
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
            commander_cmcs.append(cmc)
            commander_names.append(name)

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
            ramp_by_cmc[cmc] += qty
        if is_draw:
            draw_cards += qty
            subcategory = _classify_draw_subcategory(types, name, registry)
            draw_breakdown[subcategory] += qty

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
        ramp_by_cmc=dict(ramp_by_cmc),
        draw_breakdown=dict(draw_breakdown),
        commander_cmc=commander_cmcs[0] if commander_cmcs else 0,
        commander_name=commander_names[0] if commander_names else "",
        commander_cmcs=commander_cmcs,
        commander_names=commander_names,
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


def _classify_draw_subcategory(
    types: List[str],
    name: str,
    registry: Optional[EffectRegistry],
) -> str:
    """Bucket a draw card into ``cantrip``, ``instant_draw``, or ``repeatable_draw``.

    Repeatable: card has any per-turn or cast-trigger draw effects in the
    registry (e.g. Phyrexian Arena, Rhystic Study). Cantrip: an Instant or
    Sorcery whose registered ``DrawCards`` effect draws exactly one card --
    the canonical "incidental draw" shape. Everything else (multi-card
    one-shots, ETB draw on permanents, draws not represented in the
    registry) falls into ``instant_draw``.
    """
    effects: CardEffects | None = registry.get(name) if registry else None
    if effects is not None and (effects.per_turn or effects.cast_trigger):
        return "repeatable_draw"

    is_spell_speed = any(t in types for t in ("Instant", "Sorcery"))
    if is_spell_speed and effects is not None:
        for e in effects.on_play:
            if type(e).__name__ == "DrawCards" and getattr(e, "amount", 0) == 1:
                return "cantrip"

    return "instant_draw"
