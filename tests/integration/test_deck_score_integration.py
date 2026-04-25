"""Integration tests for deck scoring through the full simulation pipeline."""

import pytest

from auto_goldfish.engine.goldfisher import Goldfisher
from auto_goldfish.metrics.deck_score import compute_deck_score


def _simple_deck(num_lands: int = 37, num_spells: int = 62) -> list[dict]:
    deck = []
    deck.append({
        "name": "Test Commander",
        "cmc": 4,
        "cost": "{2}{U}{B}",
        "text": "",
        "types": ["Creature"],
        "commander": True,
    })
    for i in range(num_lands):
        deck.append({
            "name": f"Island {i}",
            "cmc": 0,
            "cost": "",
            "text": "",
            "types": ["Land"],
            "commander": False,
        })
    for i in range(num_spells):
        cmc = (i % 6) + 1
        deck.append({
            "name": f"Creature {i}",
            "cmc": cmc,
            "cost": f"{{{cmc}}}",
            "text": "",
            "types": ["Creature"],
            "commander": False,
        })
    return deck


SIMS = 200
SEED = 42


@pytest.fixture(scope="module")
def sequential_result():
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=10, sims=SIMS, seed=SEED)
    return gf.simulate()


@pytest.fixture(scope="module")
def parallel_result():
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=10, sims=SIMS, seed=SEED, workers=2)
    return gf.simulate()


class TestPerTurnFields:
    def test_mean_mana_per_turn_length(self, sequential_result):
        assert len(sequential_result.mean_mana_per_turn) == 10

    def test_mean_spells_per_turn_length(self, sequential_result):
        assert len(sequential_result.mean_spells_per_turn) == 10

    def test_mana_per_turn_monotonically_increases(self, sequential_result):
        mpt = sequential_result.mean_mana_per_turn
        # On average, later turns should spend more mana (not strict, but generally true)
        assert mpt[-1] > mpt[0]

    def test_std_mana_positive(self, sequential_result):
        assert sequential_result.std_mana > 0

    def test_mull_rate_between_0_and_1(self, sequential_result):
        assert 0.0 <= sequential_result.mull_rate <= 1.0

    def test_mull_mana_values_exist(self, sequential_result):
        assert sequential_result.mean_mana_with_mull > 0
        assert sequential_result.mean_mana_no_mull > 0


class TestParallelPathHasPerTurnData:
    def test_mean_mana_per_turn_length(self, parallel_result):
        assert len(parallel_result.mean_mana_per_turn) == 10

    def test_std_mana_positive(self, parallel_result):
        assert parallel_result.std_mana > 0

    def test_mull_rate_between_0_and_1(self, parallel_result):
        assert 0.0 <= parallel_result.mull_rate <= 1.0


class TestDeckScoreFromSimulation:
    def test_sequential_score_all_in_range(self, sequential_result):
        score = compute_deck_score(sequential_result, turns=10)
        for name, value in score.as_dict().items():
            assert 1 <= value <= 10, f"{name}={value} out of range"

    def test_parallel_score_all_in_range(self, parallel_result):
        score = compute_deck_score(parallel_result, turns=10)
        for name, value in score.as_dict().items():
            assert 1 <= value <= 10, f"{name}={value} out of range"

    def test_score_format_block_renders(self, sequential_result):
        score = compute_deck_score(sequential_result, turns=10)
        block = score.format_block()
        assert "ACCELERATION" in block
        assert "REACH" in block


class TestStructuralStatsPlumbed:
    def test_structural_fields_populated(self, sequential_result):
        # The simple_deck has 37 lands and 62 spells; mana_source_count
        # should at least equal the land count (no ramp registered).
        assert sequential_result.mana_source_count >= 37
        assert sequential_result.early_count > 0
        assert sequential_result.avg_cmc > 0
