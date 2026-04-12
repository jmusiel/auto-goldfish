"""Mana model routes -- instant hypergeometric analysis API."""

from __future__ import annotations

import os

from flask import Blueprint, abort, jsonify, request

from auto_goldfish.decklist.loader import get_deckpath, load_decklist, load_overrides
from auto_goldfish.effects.card_database import DEFAULT_REGISTRY
from auto_goldfish.optimization.deck_analyzer import analyze_deck_composition
from auto_goldfish.optimization.mana_model import (
    expected_mana_table,
    land_count_comparison,
    mulligan_probability,
    optimal_land_count,
)

bp = Blueprint("mana_model", __name__, url_prefix="/mana-model")


@bp.route("/api/<deck_name>/analysis", methods=["GET", "POST"])
def analysis(deck_name: str):
    """Instant JSON analysis with recommendation, mana table, comparison, mulligan stats.

    GET: loads deck from disk (saved decks).
    POST: accepts {cards, overrides} in the request body (localStorage decks).
    """
    if request.method == "POST":
        try:
            body = request.get_json(force=True)
        except Exception:
            abort(400)
        deck_list = body.get("cards", [])
        overrides = body.get("overrides", {})
    else:
        path = get_deckpath(deck_name)
        if not os.path.isfile(path):
            abort(404)
        deck_list = load_decklist(deck_name)
        overrides = load_overrides(deck_name)

    comp = analyze_deck_composition(deck_list, DEFAULT_REGISTRY, overrides)

    # Get recommendation
    rec = optimal_land_count(
        deck_size=comp.deck_size,
        cmc_distribution=comp.cmc_distribution,
        ramp_cards=comp.ramp_cards,
        draw_cards=comp.draw_cards,
        commander_cmc=comp.commander_cmc,
    )

    # Mana table for current land count
    current_table = expected_mana_table(comp.deck_size, comp.land_count, max_turn=10)

    # Comparison: current vs recommended (and +/- 2)
    compare_counts = sorted(set([
        comp.land_count,
        rec["recommended_lands"],
        max(20, comp.land_count - 2),
        min(comp.deck_size - 10, comp.land_count + 2),
    ]))
    comparison = land_count_comparison(comp.deck_size, compare_counts, max_turn=10)

    # Mulligan stats
    p_mull = mulligan_probability(comp.deck_size, comp.land_count)
    p_mull_rec = mulligan_probability(comp.deck_size, rec["recommended_lands"])

    return jsonify({
        "deck_name": deck_name,
        "composition": {
            "deck_size": comp.deck_size,
            "land_count": comp.land_count,
            "mdfc_count": comp.mdfc_count,
            "avg_cmc": comp.avg_cmc,
            "ramp_cards": comp.ramp_cards,
            "draw_cards": comp.draw_cards,
            "commander_cmc": comp.commander_cmc,
            "commander_name": comp.commander_name,
            "cmc_distribution": comp.cmc_distribution,
        },
        "recommendation": rec,
        "current_mana_table": current_table,
        "comparison": comparison,
        "mulligan": {
            "current_rate": round(p_mull, 4),
            "recommended_rate": round(p_mull_rec, 4),
        },
    })


@bp.route("/api/calculate", methods=["POST"])
def calculate():
    """Ad-hoc calculation for interactive what-if sliders."""
    try:
        body = request.get_json(force=True)
    except Exception:
        abort(400)

    deck_size = body.get("deck_size", 99)
    land_count = body.get("land_count", 36)
    max_turn = min(body.get("max_turn", 10), 14)
    ramp_cards = body.get("ramp_cards", 0)
    draw_cards = body.get("draw_cards", 0)

    table = expected_mana_table(deck_size, land_count, max_turn)
    p_mull = mulligan_probability(deck_size, land_count)

    return jsonify({
        "deck_size": deck_size,
        "land_count": land_count,
        "mana_table": table,
        "mulligan_rate": round(p_mull, 4),
    })
