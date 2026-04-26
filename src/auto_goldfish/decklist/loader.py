"""Load decklists from JSON files."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def get_deckpath(deck_name: str) -> str:
    """Return the path to a deck's JSON file.

    Decks are stored at ``<project_root>/decks/<name>/<name>.json``.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    deck_dir = os.path.join(project_root, "decks", deck_name)
    return os.path.join(deck_dir, f"{deck_name}.json")


def load_decklist(deck_name: str) -> List[Dict[str, Any]]:
    """Load a decklist from a saved JSON file."""
    path = get_deckpath(deck_name)
    with open(path) as f:
        return json.load(f)


def save_decklist(deck_name: str, decklist: List[Dict[str, Any]]) -> str:
    """Save a decklist to JSON. Returns the file path."""
    path = get_deckpath(deck_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(decklist, f, indent=4)
    return path


def get_overrides_path(deck_name: str) -> str:
    """Return the path to a deck's overrides file: decks/<name>/<name>.overrides.json."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    deck_dir = os.path.join(project_root, "decks", deck_name)
    return os.path.join(deck_dir, f"{deck_name}.overrides.json")


def load_overrides(deck_name: str) -> Dict[str, Any]:
    """Load saved overrides. Returns {} if file doesn't exist."""
    path = get_overrides_path(deck_name)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_overrides(deck_name: str, overrides: Dict[str, Any]) -> str:
    """Save overrides dict to JSON. Returns file path."""
    path = get_overrides_path(deck_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(overrides, f, indent=4)
    return path


def get_basic_island() -> Dict[str, Any]:
    """Return a card dict for a basic Island."""
    return {
        "name": "Island",
        "quantity": 1,
        "oracle_cmc": 0,
        "cmc": 0,
        "cost": "",
        "text": "({T}: Add {U}.)",
        "sub_types": ["Island"],
        "super_types": ["Basic"],
        "types": ["Land"],
        "identity": ["Blue"],
        "default_category": None,
        "user_category": "Land",
        "commander": False,
    }


def get_hare_apparent() -> Dict[str, Any]:
    """Return a card dict for Hare Apparent."""
    return {
        "name": "Hare Apparent",
        "quantity": 1,
        "oracle_cmc": 2,
        "cmc": 2,
        "cost": "{1}{W}",
        "text": "When this creature enters, create a number of 1/1 white Rabbit creature tokens equal to the number of other creatures you control named Hare Apparent.\nA deck can have any number of cards named Hare Apparent.",
        "sub_types": ["Rabbit", "Noble"],
        "super_types": [],
        "types": ["Creature"],
        "identity": ["White"],
        "default_category": "Tokens",
        "user_category": "hare apparent",
        "commander": False,
    }
