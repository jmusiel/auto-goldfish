"""Unit tests for the deck scoring module."""

import pytest

from auto_goldfish.engine.goldfisher import SimulationResult
from auto_goldfish.metrics.deck_score import (
    DeckScore,
    _clamp,
    _compute_consistency,
    _compute_efficiency,
    _compute_momentum,
    _compute_power,
    _compute_resilience,
    _compute_speed,
    _scale,
    compute_deck_score,
)


# ---------------------------------------------------------------------------
# Helper to build SimulationResults with custom fields
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> SimulationResult:
    defaults = {
        "mean_mana": 20.0,
        "mean_mana_value": 15.0,
        "mean_mana_draw": 3.0,
        "mean_mana_ramp": 2.0,
        "mean_mana_total": 20.0,
        "consistency": 0.7,
        "mean_bad_turns": 1.5,
        "mean_mid_turns": 3.0,
        "mean_lands": 5.0,
        "mean_mulls": 0.3,
        "mean_spells_cast": 7.0,
        "percentile_25": 14.0,
        "percentile_50": 20.0,
        "percentile_75": 26.0,
        "ceiling_mana": 28.0,
        "mean_mana_per_turn": [0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "mean_spells_per_turn": [0.3, 0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2],
        "std_mana": 5.0,
        "mull_rate": 0.25,
        "mean_mana_with_mull": 18.0,
        "mean_mana_no_mull": 21.0,
    }
    defaults.update(overrides)
    return SimulationResult(**defaults)


# ---------------------------------------------------------------------------
# _clamp and _scale
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert _clamp(10.0) == 10

    def test_below_minimum(self):
        assert _clamp(-5.0) == 1

    def test_above_maximum(self):
        assert _clamp(25.0) == 20

    def test_rounds(self):
        assert _clamp(10.4) == 10
        assert _clamp(10.6) == 11


class TestScale:
    def test_midpoint(self):
        assert _scale(50.0, 0.0, 100.0) == 10

    def test_minimum(self):
        assert _scale(0.0, 0.0, 100.0) == 1

    def test_maximum(self):
        assert _scale(100.0, 0.0, 100.0) == 20

    def test_below_range(self):
        assert _scale(-10.0, 0.0, 100.0) == 1

    def test_above_range(self):
        assert _scale(110.0, 0.0, 100.0) == 20

    def test_equal_bounds_returns_10(self):
        assert _scale(5.0, 5.0, 5.0) == 10


# ---------------------------------------------------------------------------
# Individual stat computation
# ---------------------------------------------------------------------------

class TestSpeed:
    def test_fast_deck_scores_high(self):
        # 3+4+5+6 = 18 mana in first 4 turns (above 14 cap)
        result = _make_result(mean_mana_per_turn=[3.0, 4.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        assert _compute_speed(result, 10) >= 17

    def test_slow_deck_scores_low(self):
        # 0+0.5+0.5+0.5 = 1.5 mana in first 4 turns
        result = _make_result(mean_mana_per_turn=[0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert _compute_speed(result, 10) <= 4

    def test_empty_per_turn_returns_1(self):
        result = _make_result(mean_mana_per_turn=[])
        assert _compute_speed(result, 10) == 1

    def test_fewer_than_4_turns(self):
        result = _make_result(mean_mana_per_turn=[2.0, 3.0])
        score = _compute_speed(result, 2)
        assert 1 <= score <= 20


class TestPower:
    def test_high_power(self):
        result = _make_result(mean_mana=40.0, ceiling_mana=50.0)
        assert _compute_power(result, 10) >= 17

    def test_low_power(self):
        result = _make_result(mean_mana=5.0, ceiling_mana=8.0)
        assert _compute_power(result, 10) <= 4

    def test_scales_with_turns(self):
        result = _make_result(mean_mana=20.0, ceiling_mana=28.0)
        score_10 = _compute_power(result, 10)
        score_5 = _compute_power(result, 5)
        # With fewer turns, same raw values should score higher
        assert score_5 >= score_10


class TestConsistency:
    def test_perfect_consistency(self):
        result = _make_result(consistency=1.0, mean_bad_turns=0.0, std_mana=0.0, mean_mana=20.0)
        assert _compute_consistency(result, 10) == 20

    def test_terrible_consistency(self):
        result = _make_result(consistency=0.0, mean_bad_turns=6.0, std_mana=15.0, mean_mana=10.0)
        assert _compute_consistency(result, 10) <= 3

    def test_mid_consistency(self):
        result = _make_result(consistency=0.7, mean_bad_turns=1.5, std_mana=5.0, mean_mana=20.0)
        score = _compute_consistency(result, 10)
        assert 8 <= score <= 16


class TestResilience:
    def test_no_mull_penalty(self):
        # No difference between mull and no-mull games, low mull rate
        result = _make_result(
            mean_mana=20.0, mean_mana_with_mull=20.0, mean_mana_no_mull=20.0,
            mull_rate=0.1,
        )
        assert _compute_resilience(result) >= 16

    def test_severe_mull_penalty(self):
        result = _make_result(
            mean_mana=20.0, mean_mana_with_mull=10.0, mean_mana_no_mull=22.0,
            mull_rate=0.5,
        )
        assert _compute_resilience(result) <= 6

    def test_zero_mana_returns_10(self):
        result = _make_result(mean_mana=0.0)
        assert _compute_resilience(result) == 10


class TestEfficiency:
    def test_perfect_utilization(self):
        result = _make_result(mean_mana=55.0, mean_lands=10.0, mean_mid_turns=0.0)
        assert _compute_efficiency(result, 10) == 20

    def test_zero_lands_returns_1(self):
        result = _make_result(mean_mana=0.0, mean_lands=0.0, mean_mid_turns=5.0)
        assert _compute_efficiency(result, 10) == 1


class TestMomentum:
    def test_strong_acceleration(self):
        # Late game mana much higher than early
        result = _make_result(
            mean_mana_per_turn=[0.5, 1.0, 1.5, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        assert _compute_momentum(result, 10) >= 14

    def test_flat_curve(self):
        # Same mana every turn
        result = _make_result(
            mean_mana_per_turn=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        )
        score = _compute_momentum(result, 10)
        assert score <= 12

    def test_short_game_returns_10(self):
        result = _make_result(mean_mana_per_turn=[1.0, 2.0, 3.0])
        assert _compute_momentum(result, 3) == 10


# ---------------------------------------------------------------------------
# compute_deck_score integration
# ---------------------------------------------------------------------------

class TestComputeDeckScore:
    def test_returns_deck_score(self):
        result = _make_result()
        score = compute_deck_score(result, turns=10)
        assert isinstance(score, DeckScore)

    def test_all_stats_in_range(self):
        result = _make_result()
        score = compute_deck_score(result, turns=10)
        for name, value in score.as_dict().items():
            assert 1 <= value <= 20, f"{name}={value} out of range"

    def test_as_dict_keys(self):
        score = compute_deck_score(_make_result(), turns=10)
        keys = set(score.as_dict().keys())
        assert keys == {"speed", "power", "consistency", "resilience", "efficiency", "momentum"}

    def test_format_block_contains_all_stats(self):
        score = compute_deck_score(_make_result(), turns=10)
        block = score.format_block()
        for stat in ["SPEED", "POWER", "CONSISTENCY", "RESILIENCE", "EFFICIENCY", "MOMENTUM"]:
            assert stat in block


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_default_simulation_result(self):
        """Scoring a default (empty) SimulationResult should not crash."""
        result = SimulationResult()
        score = compute_deck_score(result, turns=10)
        for value in score.as_dict().values():
            assert 1 <= value <= 20

    def test_single_turn(self):
        result = _make_result(mean_mana_per_turn=[3.0], mean_spells_per_turn=[1.0])
        score = compute_deck_score(result, turns=1)
        for value in score.as_dict().values():
            assert 1 <= value <= 20
