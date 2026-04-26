"""Deck import and viewer routes."""

from __future__ import annotations

import json
import os
import re

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, url_for

# Engine bookkeeping suffix used to keep simulator card identities unique.
_COPY_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")


def _base_card_name(name: str) -> str:
    return _COPY_SUFFIX_RE.sub("", name) if name else name


def _pool_cards(cards):
    """Collapse engine-uniquified copies into one entry per base name."""
    pools: dict[str, dict] = {}
    order: list[str] = []
    for card in cards:
        base = _base_card_name(card.get("name", ""))
        if base not in pools:
            pools[base] = {
                "base_name": base,
                "cmc": card.get("cmc"),
                "cost": card.get("cost"),
                "count": 0,
                "example": card,
            }
            order.append(base)
        pools[base]["count"] += int(card.get("quantity", 1) or 1)
    return [pools[k] for k in order]

from auto_goldfish.decklist.archidekt import fetch_and_save, fetch_decklist as fetch_archidekt
from auto_goldfish.decklist.card_resolver import resolve_cards
from auto_goldfish.decklist.loader import get_deckpath, load_decklist
from auto_goldfish.decklist.moxfield import (
    MoxfieldConfigError,
    fetch_decklist as fetch_moxfield,
    is_configured as moxfield_is_configured,
)
from auto_goldfish.decklist.text_import import parse_decklist

bp = Blueprint("decks", __name__, url_prefix="/decks")


@bp.route("/import")
def import_form():
    return render_template("import.html", moxfield_available=moxfield_is_configured())


@bp.route("/import", methods=["POST"])
def import_deck():
    default_url = "https://archidekt.com/decks/81320/the_rr_connection"

    deck_url = request.form.get("deck_url", "").strip() or default_url
    deck_name = request.form.get("deck_name", "").strip()

    # If no name provided, extract it from the URL's last path segment
    if not deck_name:
        deck_name = deck_url.rstrip("/").rsplit("/", 1)[-1]

    try:
        fetch_and_save(deck_url, deck_name)
    except Exception as e:
        flash(f"Import failed: {e}", "error")
        return render_template("import.html", moxfield_available=moxfield_is_configured()), 400

    flash(f"Deck '{deck_name}' imported successfully.", "success")
    return redirect(url_for("decks.view_deck", name=deck_name))


@bp.route("/import/api", methods=["POST"])
def import_deck_api():
    """Import a deck from Archidekt, Moxfield, or pasted text. Returns JSON."""
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    source = (body.get("source") or "archidekt").strip().lower()
    deck_name = (body.get("deck_name") or "").strip()

    try:
        if source == "text":
            decklist_text = (body.get("decklist_text") or "").strip()
            if not decklist_text:
                return jsonify({"ok": False, "error": "No decklist text provided"}), 400
            if not deck_name:
                return jsonify({"ok": False, "error": "Deck name is required for text import"}), 400
            entries = parse_decklist(decklist_text)
            if not entries:
                return jsonify({"ok": False, "error": "No cards found in decklist text"}), 400
            cards = resolve_cards(entries)

        elif source == "moxfield":
            deck_url = (body.get("deck_url") or "").strip()
            if not deck_url:
                return jsonify({"ok": False, "error": "Moxfield URL is required"}), 400
            if not deck_name:
                deck_name = deck_url.rstrip("/").rsplit("/", 1)[-1]
            cards = fetch_moxfield(deck_url)

        else:  # archidekt (default)
            default_url = "https://archidekt.com/decks/81320/the_rr_connection"
            deck_url = (body.get("deck_url") or "").strip() or default_url
            if not deck_name:
                deck_name = deck_url.rstrip("/").rsplit("/", 1)[-1]
            cards = fetch_archidekt(deck_url)

    except MoxfieldConfigError as e:
        return jsonify({"ok": False, "error": str(e)}), 501
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "deck_name": deck_name, "cards": cards})


@bp.route("/<name>", methods=["GET", "POST"])
def view_deck(name: str):
    if request.method == "POST":
        try:
            body = request.get_json(force=True)
        except Exception:
            abort(400)
        cards = body.get("cards", [])
        is_local = True
    else:
        path = get_deckpath(name)
        if not os.path.isfile(path):
            abort(404)
        cards = load_decklist(name)
        is_local = False

    # Group cards by user_category (or "Other"), then collapse engine-uniquified
    # copies inside each group so the deck list shows "9× Avatar of Woe" once.
    groups: dict[str, list] = {}
    commanders = []
    for card in cards:
        if card.get("commander"):
            commanders.append(card)
            continue
        category = card.get("user_category") or card.get("default_category") or "Other"
        groups.setdefault(category, []).append(card)

    pooled_groups = []
    for category, raw_cards in groups.items():
        pools = _pool_cards(raw_cards)
        pools.sort(key=lambda p: p["base_name"].lower())
        pooled_groups.append((category, pools))

    # Sort groups: Land last, Commander first handled separately
    pooled_groups.sort(key=lambda x: (x[0] == "Land", x[0]))

    total_cards = len(cards)
    land_count = sum(1 for c in cards if "Land" in c.get("types", []))

    return render_template(
        "deck_view.html",
        name=name,
        commanders=commanders,
        groups=pooled_groups,
        total_cards=total_cards,
        land_count=land_count,
        is_local=is_local,
    )
