"""Tests for the hypergeometric mana model -- pure math functions."""

import pytest

from auto_goldfish.optimization.mana_model import (
    adjusted_expected_mana,
    expected_mana_on_turn,
    expected_mana_table,
    hypergeometric_cdf,
    hypergeometric_pmf,
    land_count_comparison,
    mulligan_probability,
    optimal_land_count,
    prob_at_least,
)


# ---------------------------------------------------------------------------
# PMF tests
# ---------------------------------------------------------------------------

class TestHypergeometricPMF:
    def test_known_value(self):
        """P(exactly 3 lands in 7 cards from 99 cards, 36 lands)."""
        p = hypergeometric_pmf(3, 99, 36, 7)
        assert 0.0 < p < 1.0
        # Known approximate value ~0.278
        assert abs(p - 0.278) < 0.02

    def test_impossible_draw(self):
        """Can't draw 5 lands from 7 cards if only 4 lands in deck."""
        assert hypergeometric_pmf(5, 99, 4, 7) == 0.0

    def test_zero_successes(self):
        """P(0 lands in 7) should be positive for 36/99."""
        p = hypergeometric_pmf(0, 99, 36, 7)
        assert p > 0.0
        assert p < 0.05  # Unlikely with 36 lands (~3.7%)

    def test_all_pmf_sum_to_one(self):
        """Sum of all PMF values for a given draw should be ~1."""
        total = sum(hypergeometric_pmf(k, 99, 36, 7) for k in range(8))
        assert abs(total - 1.0) < 1e-10

    def test_edge_n_equals_zero(self):
        """Drawing 0 cards always gives 0 successes."""
        assert hypergeometric_pmf(0, 99, 36, 0) == 1.0
        assert hypergeometric_pmf(1, 99, 36, 0) == 0.0


# ---------------------------------------------------------------------------
# CDF and P(at least) tests
# ---------------------------------------------------------------------------

class TestCDF:
    def test_cdf_at_max(self):
        """CDF at k=7 (max possible lands in 7 cards) should be 1."""
        assert abs(hypergeometric_cdf(7, 99, 36, 7) - 1.0) < 1e-10

    def test_cdf_monotonic(self):
        """CDF should be non-decreasing."""
        prev = 0.0
        for k in range(8):
            cur = hypergeometric_cdf(k, 99, 36, 7)
            assert cur >= prev - 1e-12
            prev = cur

    def test_prob_at_least_complement(self):
        """P(X>=k) = 1 - P(X<=k-1)."""
        for k in range(1, 8):
            p = prob_at_least(k, 99, 36, 7)
            cdf = hypergeometric_cdf(k - 1, 99, 36, 7)
            assert abs(p - (1.0 - cdf)) < 1e-10

    def test_prob_at_least_zero(self):
        """P(X>=0) should be 1."""
        assert prob_at_least(0, 99, 36, 7) == 1.0


# ---------------------------------------------------------------------------
# Expected mana tests
# ---------------------------------------------------------------------------

class TestExpectedMana:
    def test_turn_1(self):
        """On turn 1, expected mana <= 1 (can play at most 1 land)."""
        e = expected_mana_on_turn(1, 99, 36)
        assert 0.0 < e <= 1.0
        # With 36/99 lands, P(>=1 land in 7 cards) is very high
        assert e > 0.9

    def test_increases_with_turns(self):
        """Expected mana should generally increase with turns."""
        prev = 0.0
        for t in range(1, 11):
            e = expected_mana_on_turn(t, 99, 36)
            assert e >= prev - 0.01  # Allow tiny floating point
            prev = e

    def test_more_lands_means_more_mana(self):
        """More lands should give more expected mana on any turn."""
        for t in [3, 5, 7]:
            e_low = expected_mana_on_turn(t, 99, 30)
            e_high = expected_mana_on_turn(t, 99, 40)
            assert e_high > e_low


class TestExpectedManaTable:
    def test_returns_correct_length(self):
        table = expected_mana_table(99, 36, max_turn=8)
        assert len(table) == 8

    def test_table_fields(self):
        table = expected_mana_table(99, 36, max_turn=1)
        row = table[0]
        assert "turn" in row
        assert "expected_mana" in row
        assert "prob_on_curve" in row
        assert "prob_screw" in row
        assert row["turn"] == 1

    def test_on_curve_decreases_over_turns(self):
        """P(on curve) generally decreases as turn increases (harder to hit all drops)."""
        table = expected_mana_table(99, 36, max_turn=10)
        # Turn 1 on-curve should be very high
        assert table[0]["prob_on_curve"] > 0.9


# ---------------------------------------------------------------------------
# Mulligan tests
# ---------------------------------------------------------------------------

class TestMulligan:
    def test_mulligan_rate_reasonable(self):
        """Mulligan rate for 36/99 with keep 2-5 should be low."""
        p = mulligan_probability(99, 36, keep_range=(2, 5))
        assert 0.0 < p < 0.3

    def test_extreme_land_count_high_mull(self):
        """Very few lands should give high mulligan rate."""
        p = mulligan_probability(99, 10, keep_range=(2, 5))
        assert p > 0.5


# ---------------------------------------------------------------------------
# Ramp/draw adjustment tests
# ---------------------------------------------------------------------------

class TestAdjustedMana:
    def test_ramp_increases_mana(self):
        """Ramp cards should increase expected mana on later turns."""
        e_base = adjusted_expected_mana(5, 99, 36, ramp_cards=0)
        e_ramp = adjusted_expected_mana(5, 99, 36, ramp_cards=8)
        assert e_ramp > e_base

    def test_draw_increases_mana(self):
        """Draw cards should increase expected mana on later turns."""
        e_base = adjusted_expected_mana(5, 99, 36, draw_cards=0)
        e_draw = adjusted_expected_mana(5, 99, 36, draw_cards=5)
        assert e_draw >= e_base

    def test_no_ramp_on_early_turns(self):
        """Ramp with avg_cmc=2 should not add mana on turn 1."""
        e_base = adjusted_expected_mana(1, 99, 36, ramp_cards=0)
        e_ramp = adjusted_expected_mana(1, 99, 36, ramp_cards=8, avg_ramp_cmc=2.0)
        assert abs(e_ramp - e_base) < 0.01


# ---------------------------------------------------------------------------
# Optimal land count tests
# ---------------------------------------------------------------------------

class TestOptimalLandCount:
    def test_returns_recommendation(self):
        result = optimal_land_count(deck_size=99)
        assert "recommended_lands" in result
        assert 25 <= result["recommended_lands"] <= 45

    def test_with_cmc_distribution(self):
        # Heavy curve deck
        cmc_dist = {1: 5, 2: 15, 3: 15, 4: 10, 5: 5, 6: 3, 7: 2}
        result = optimal_land_count(deck_size=99, cmc_distribution=cmc_dist)
        assert result["recommended_lands"] >= 30

    def test_low_curve_fewer_lands(self):
        """Low curve deck should recommend fewer lands than high curve."""
        low = optimal_land_count(
            deck_size=99,
            cmc_distribution={1: 20, 2: 25, 3: 10},
        )
        high = optimal_land_count(
            deck_size=99,
            cmc_distribution={3: 10, 4: 15, 5: 10, 6: 10, 7: 5},
        )
        assert low["recommended_lands"] <= high["recommended_lands"]

    def test_scores_list(self):
        result = optimal_land_count(deck_size=99, search_range=(33, 37))
        assert len(result["scores"]) == 5  # 33,34,35,36,37


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

class TestLandCountComparison:
    def test_returns_one_per_count(self):
        result = land_count_comparison(99, [34, 36, 38], max_turn=5)
        assert len(result) == 3

    def test_each_has_table(self):
        result = land_count_comparison(99, [36], max_turn=5)
        assert len(result[0]["mana_table"]) == 5
        assert "mulligan_rate" in result[0]
        assert "land_ratio" in result[0]
