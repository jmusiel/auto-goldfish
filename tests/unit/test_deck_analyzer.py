"""Tests for the deck composition analyzer."""

import pytest

from auto_goldfish.effects.registry import CardEffects, EffectRegistry
from auto_goldfish.optimization.deck_analyzer import (
    DeckComposition,
    analyze_deck_composition,
)


def _make_card_dict(name, cmc=2, types=None, commander=False, quantity=1):
    return {
        "name": name,
        "cmc": cmc,
        "oracle_cmc": cmc,
        "types": types or ["Creature"],
        "commander": commander,
        "quantity": quantity,
        "cost": f"{{{cmc}}}",
        "text": "",
    }


def _make_land(name="Island"):
    return _make_card_dict(name, cmc=0, types=["Land"])


def _make_mdfc(name="Emeria's Call"):
    return _make_card_dict(name, cmc=7, types=["Land", "Sorcery"])


class TestAnalyzeDeckComposition:
    def test_basic_deck(self):
        deck = [_make_land()] * 36 + [_make_card_dict("Bear", cmc=2)] * 63
        comp = analyze_deck_composition(deck)
        assert comp.deck_size == 99
        assert comp.land_count == 36
        assert comp.avg_cmc == 2.0

    def test_counts_lands(self):
        deck = [_make_land()] * 40 + [_make_card_dict("Elf")] * 59
        comp = analyze_deck_composition(deck)
        assert comp.land_count == 40

    def test_cmc_distribution(self):
        deck = [_make_land()] * 36
        deck += [_make_card_dict("A", cmc=1)] * 10
        deck += [_make_card_dict("B", cmc=2)] * 20
        deck += [_make_card_dict("C", cmc=3)] * 15
        deck += [_make_card_dict("D", cmc=5)] * 18
        comp = analyze_deck_composition(deck)
        assert comp.cmc_distribution[1] == 10
        assert comp.cmc_distribution[2] == 20
        assert comp.cmc_distribution[3] == 15
        assert comp.cmc_distribution[5] == 18

    def test_commander_extraction(self):
        deck = [_make_land()] * 36
        deck += [_make_card_dict("Atraxa", cmc=4, commander=True)]
        deck += [_make_card_dict("Bear")] * 62
        comp = analyze_deck_composition(deck)
        assert comp.commander_cmc == 4
        assert comp.commander_name == "Atraxa"
        assert comp.commander_cmcs == [4]
        assert comp.commander_names == ["Atraxa"]

    def test_partner_commanders_both_tracked(self):
        deck = [_make_land()] * 36
        deck += [_make_card_dict("Tymna", cmc=2, commander=True)]
        deck += [_make_card_dict("Tana", cmc=4, commander=True)]
        deck += [_make_card_dict("Bear")] * 61
        comp = analyze_deck_composition(deck)
        # Primary (legacy fields) point at the first commander encountered.
        assert comp.commander_cmc == 2
        assert comp.commander_name == "Tymna"
        # Both partners listed in the new fields.
        assert comp.commander_cmcs == [2, 4]
        assert comp.commander_names == ["Tymna", "Tana"]

    def test_no_commanders_empty_lists(self):
        deck = [_make_land()] * 36 + [_make_card_dict("Bear")] * 63
        comp = analyze_deck_composition(deck)
        assert comp.commander_cmcs == []
        assert comp.commander_names == []
        assert comp.commander_cmc == 0
        assert comp.commander_name == ""

    def test_mdfc_counted_as_land(self):
        deck = [_make_land()] * 35 + [_make_mdfc()] + [_make_card_dict("Bear")] * 63
        comp = analyze_deck_composition(deck)
        assert comp.land_count == 36
        assert comp.mdfc_count == 1

    def test_ramp_from_registry(self):
        registry = EffectRegistry()
        registry.register("Sol Ring", CardEffects(ramp=True))
        deck = [_make_land()] * 36
        deck += [_make_card_dict("Sol Ring", cmc=1)] * 1
        deck += [_make_card_dict("Bear")] * 62
        comp = analyze_deck_composition(deck, registry=registry)
        assert comp.ramp_cards == 1

    def test_draw_from_registry(self):
        registry = EffectRegistry()
        registry.register("Harmonize", CardEffects(draw=True))
        deck = [_make_land()] * 36
        deck += [_make_card_dict("Harmonize", cmc=4)] * 1
        deck += [_make_card_dict("Bear")] * 62
        comp = analyze_deck_composition(deck, registry=registry)
        assert comp.draw_cards == 1

    def test_ramp_from_overrides(self):
        overrides = {
            "Mana Rock": {"categories": [{"category": "ramp"}]},
        }
        deck = [_make_land()] * 36
        deck += [_make_card_dict("Mana Rock", cmc=2)] * 1
        deck += [_make_card_dict("Bear")] * 62
        comp = analyze_deck_composition(deck, overrides=overrides)
        assert comp.ramp_cards == 1

    def test_override_takes_precedence_over_registry(self):
        """If a card has a user override, use that instead of registry."""
        registry = EffectRegistry()
        registry.register("Rock", CardEffects(ramp=True, draw=False))
        overrides = {
            "Rock": {"categories": [{"category": "draw"}]},
        }
        deck = [_make_land()] * 36
        deck += [_make_card_dict("Rock")] * 1
        deck += [_make_card_dict("Bear")] * 62
        comp = analyze_deck_composition(deck, registry=registry, overrides=overrides)
        # Override says draw, not ramp
        assert comp.draw_cards == 1
        assert comp.ramp_cards == 0

    def test_empty_deck(self):
        comp = analyze_deck_composition([])
        assert comp.deck_size == 0
        assert comp.land_count == 0
        assert comp.avg_cmc == 0.0
