"""Integration test: validate mana model predictions against simulation.

This is a stub that can be expanded once simulation results are available.
The model's expected mana should match simulation within ~2-5%.
"""

import pytest

from auto_goldfish.optimization.mana_model import (
    expected_mana_on_turn,
    expected_mana_table,
    hypergeometric_pmf,
    prob_at_least,
)


class TestMathematicalConsistency:
    """Validate internal consistency of the model (no simulation needed)."""

    def test_pmf_sums_to_one_various_params(self):
        """PMF should sum to 1 for various deck/land/draw combos."""
        params = [
            (99, 36, 7),
            (99, 40, 7),
            (60, 24, 7),
            (99, 36, 15),
            (99, 30, 10),
        ]
        for N, K, n in params:
            total = sum(hypergeometric_pmf(k, N, K, n) for k in range(n + 1))
            assert abs(total - 1.0) < 1e-10, f"PMF failed for N={N}, K={K}, n={n}: sum={total}"

    def test_expected_mana_bounded_by_turn(self):
        """Expected mana on turn T should never exceed T."""
        for t in range(1, 11):
            for k in [30, 36, 42]:
                e = expected_mana_on_turn(t, 99, k)
                assert e <= t + 0.001, f"E[mana] on turn {t} with {k} lands = {e} > {t}"

    def test_expected_mana_bounded_by_lands(self):
        """Expected mana should never exceed the number of lands in deck."""
        for k in [10, 20, 30]:
            for t in [5, 10]:
                e = expected_mana_on_turn(t, 99, k)
                assert e <= k + 0.001

    def test_on_curve_turn1_near_certain(self):
        """With 36/99 lands, P(>=1 land in 7 cards) should be very high."""
        p = prob_at_least(1, 99, 36, 7)
        assert p > 0.95  # ~96.3% with 36/99 lands

    def test_table_probabilities_valid(self):
        """All probabilities in the mana table should be in [0, 1]."""
        table = expected_mana_table(99, 36, max_turn=10)
        for row in table:
            assert 0.0 <= row["prob_on_curve"] <= 1.0
            assert 0.0 <= row["prob_screw"] <= 1.0
            assert row["expected_mana"] >= 0.0
