"""Comprehensive tests for all simulation metrics.

Tests cover every field of SimulationResult for both sequential
and parallel execution paths, including distribution_stats.
"""

import pytest

from auto_goldfish.engine.goldfisher import Goldfisher, SimulationResult


def _simple_deck(num_lands: int = 37, num_spells: int = 62) -> list[dict]:
    """Build a simple deck with lands and vanilla creatures."""
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


# Use enough sims for stable distribution stats (need > calibration sample)
SIMS = 500
SEED = 42


@pytest.fixture(scope="module")
def sequential_result() -> SimulationResult:
    """Run a single sequential simulation used by multiple tests."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=10, sims=SIMS, record_results="quartile", seed=SEED)
    return gf.simulate()


@pytest.fixture(scope="module")
def parallel_result() -> SimulationResult:
    """Run a single parallel simulation used by multiple tests."""
    deck = _simple_deck()
    gf = Goldfisher(
        deck, turns=10, sims=SIMS, record_results="quartile", seed=SEED, workers=2,
    )
    return gf.simulate()


# ---------------------------------------------------------------------------
# Core metric sanity checks (sequential)
# ---------------------------------------------------------------------------

class TestCoreMetrics:
    """Verify all core SimulationResult fields are populated and sensible."""

    def test_land_count(self, sequential_result):
        assert sequential_result.land_count == 37

    def test_mean_mana_positive(self, sequential_result):
        assert sequential_result.mean_mana > 0

    def test_mean_mana_value_nonnegative(self, sequential_result):
        assert sequential_result.mean_mana_value >= 0

    def test_mean_mana_draw_nonnegative(self, sequential_result):
        assert sequential_result.mean_mana_draw >= 0

    def test_mean_mana_ramp_nonnegative(self, sequential_result):
        assert sequential_result.mean_mana_ramp >= 0

    def test_mean_mana_total_nonnegative(self, sequential_result):
        assert sequential_result.mean_mana_total >= 0

    def test_mean_hand_sum_positive(self, sequential_result):
        assert sequential_result.mean_hand_sum > 0

    def test_mean_mana_equals_value_plus_draw(self, sequential_result):
        """mean_mana should equal mean_mana_value + mean_mana_draw (backward compat)."""
        r = sequential_result
        assert r.mean_mana == pytest.approx(r.mean_mana_value + r.mean_mana_draw, abs=0.01)

    def test_mean_mana_total_equals_all_three(self, sequential_result):
        """mean_mana_total should equal value + draw + ramp."""
        r = sequential_result
        assert r.mean_mana_total == pytest.approx(
            r.mean_mana_value + r.mean_mana_draw + r.mean_mana_ramp, abs=0.01
        )

    def test_vanilla_deck_all_value_mana(self, sequential_result):
        """A deck with only vanilla creatures should have all mana as value."""
        r = sequential_result
        # No ramp or draw cards in the test deck
        assert r.mean_mana_draw == 0
        assert r.mean_mana_ramp == 0
        assert r.mean_mana == pytest.approx(r.mean_mana_value, abs=0.01)

    def test_mean_ecms_nonnegative(self, sequential_result):
        assert sequential_result.mean_ecms >= 0

    def test_mean_ecms_at_least_mean_mana_value(self, sequential_result):
        """ECMS compounds mana forward, so it should always be >= mean_mana_value."""
        r = sequential_result
        assert r.mean_ecms >= r.mean_mana_value

    def test_mean_lands_positive(self, sequential_result):
        assert sequential_result.mean_lands > 0

    def test_mean_lands_bounded(self, sequential_result):
        """Over 10 turns, can't play more lands than turns (1 per turn default)."""
        assert sequential_result.mean_lands <= 10

    def test_mean_mulls_nonnegative(self, sequential_result):
        assert sequential_result.mean_mulls >= 0

    def test_mean_draws_positive(self, sequential_result):
        """Should draw at least 7 cards (opening hand)."""
        assert sequential_result.mean_draws >= 7

    def test_mean_bad_turns_nonnegative(self, sequential_result):
        assert sequential_result.mean_bad_turns >= 0

    def test_mean_bad_turns_bounded(self, sequential_result):
        assert sequential_result.mean_bad_turns <= 10

    def test_mean_mid_turns_nonnegative(self, sequential_result):
        assert sequential_result.mean_mid_turns >= 0

    def test_mean_mid_turns_bounded(self, sequential_result):
        assert sequential_result.mean_mid_turns <= 10

    def test_percentiles_ordered(self, sequential_result):
        """25th <= 50th <= 75th percentile."""
        r = sequential_result
        assert r.percentile_25 <= r.percentile_50 <= r.percentile_75

    def test_percentiles_positive(self, sequential_result):
        assert sequential_result.percentile_25 >= 0
        assert sequential_result.percentile_50 > 0
        assert sequential_result.percentile_75 > 0


class TestConsistency:
    """Tests for the left-tail ratio consistency metric."""

    def test_consistency_range(self, sequential_result):
        """Left-tail ratio should be between 0 and 1.0."""
        assert 0 < sequential_result.consistency <= 1.0

    def test_con_threshold(self, sequential_result):
        assert sequential_result.con_threshold == 0.25

    def test_threshold_percent_equals_con_threshold(self, sequential_result):
        assert sequential_result.threshold_percent == sequential_result.con_threshold

    def test_threshold_mana_positive(self, sequential_result):
        """threshold_mana is the left-tail mean (mean of bottom quartile)."""
        assert sequential_result.threshold_mana >= 0

    def test_threshold_mana_at_most_mean(self, sequential_result):
        """Left-tail mean should be <= overall mean mana."""
        assert sequential_result.threshold_mana <= sequential_result.mean_mana

    def test_ceiling_mana_at_least_mean(self, sequential_result):
        """Top-quartile mean should be >= overall mean mana."""
        assert sequential_result.ceiling_mana >= sequential_result.mean_mana

    def test_quartile_mana_has_all_keys(self, sequential_result):
        """quartile_mana should have bottom_25, mean, and top_25 sections."""
        qm = sequential_result.quartile_mana
        assert set(qm.keys()) == {"bottom_25", "mean", "top_25"}
        for section in qm.values():
            assert set(section.keys()) == {"value", "draw", "ramp", "vd", "all"}

    def test_quartile_mana_ordering(self, sequential_result):
        """Bottom-25% vd should be <= mean vd <= top-25% vd."""
        qm = sequential_result.quartile_mana
        assert qm["bottom_25"]["vd"] <= qm["mean"]["vd"] <= qm["top_25"]["vd"]


class TestConfidenceIntervals:
    """Tests for 95% confidence intervals."""

    def test_ci_mean_mana_brackets_mean(self, sequential_result):
        r = sequential_result
        assert r.ci_mean_mana[0] <= r.mean_mana <= r.ci_mean_mana[1]

    def test_ci_mean_mana_width_positive(self, sequential_result):
        r = sequential_result
        assert r.ci_mean_mana[1] > r.ci_mean_mana[0]

    def test_ci_mean_bad_turns_brackets_mean(self, sequential_result):
        r = sequential_result
        assert r.ci_mean_bad_turns[0] <= r.mean_bad_turns <= r.ci_mean_bad_turns[1]

    def test_ci_consistency_brackets_mean(self, sequential_result):
        r = sequential_result
        assert r.ci_consistency[0] <= r.consistency <= r.ci_consistency[1]

    def test_ci_consistency_width_positive(self, sequential_result):
        r = sequential_result
        assert r.ci_consistency[1] > r.ci_consistency[0]

    def test_ci_mana_value_positive(self, sequential_result):
        """CI half-width for value mana should be positive."""
        assert sequential_result.ci_mana_value > 0

    def test_ci_mana_positive(self, sequential_result):
        """CI half-width for V+D mana should be positive."""
        assert sequential_result.ci_mana > 0

    def test_ci_mana_total_positive(self, sequential_result):
        """CI half-width for total mana should be positive."""
        assert sequential_result.ci_mana_total > 0

    def test_ci_mana_draw_nonnegative(self, sequential_result):
        """CI half-width for draw mana should be non-negative (zero in vanilla deck)."""
        assert sequential_result.ci_mana_draw >= 0

    def test_ci_mana_ramp_nonnegative(self, sequential_result):
        """CI half-width for ramp mana should be non-negative (zero in vanilla deck)."""
        assert sequential_result.ci_mana_ramp >= 0

    def test_ci_half_widths_smaller_than_means(self, sequential_result):
        """CI half-widths should be much smaller than the means (with 500 sims)."""
        r = sequential_result
        assert r.ci_mana_value < r.mean_mana_value
        assert r.ci_mana < r.mean_mana
        assert r.ci_mana_total < r.mean_mana_total

    def test_ci_narrows_with_more_sims(self):
        """More sims should yield a tighter CI."""
        deck = _simple_deck()
        gf_small = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=SEED)
        r_small = gf_small.simulate()

        gf_big = Goldfisher(deck, turns=5, sims=2000, record_results="quartile", seed=SEED)
        r_big = gf_big.simulate()

        ci_width_small = r_small.ci_mean_mana[1] - r_small.ci_mean_mana[0]
        ci_width_big = r_big.ci_mean_mana[1] - r_big.ci_mean_mana[0]
        assert ci_width_big < ci_width_small

    def test_ci_half_width_narrows_with_more_sims(self):
        """CI half-widths should narrow with more sims."""
        deck = _simple_deck()
        gf_small = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=SEED)
        r_small = gf_small.simulate()

        gf_big = Goldfisher(deck, turns=5, sims=2000, record_results="quartile", seed=SEED)
        r_big = gf_big.simulate()

        assert r_big.ci_mana_value < r_small.ci_mana_value
        assert r_big.ci_mana < r_small.ci_mana
        assert r_big.ci_mana_total < r_small.ci_mana_total


# ---------------------------------------------------------------------------
# Distribution stats
# ---------------------------------------------------------------------------

class TestDistributionStats:
    """Tests for distribution_stats bucketing."""

    def test_all_keys_present(self, sequential_result):
        ds = sequential_result.distribution_stats
        expected = [
            "top_centile", "top_decile", "top_quartile", "top_half",
            "low_half", "low_quartile", "low_decile", "low_centile",
        ]
        for key in expected:
            assert key in ds, f"Missing key: {key}"

    def test_all_values_nonnegative(self, sequential_result):
        for key, val in sequential_result.distribution_stats.items():
            assert val >= 0, f"{key} is negative: {val}"

    def test_all_values_at_most_one(self, sequential_result):
        for key, val in sequential_result.distribution_stats.items():
            assert val <= 1.0, f"{key} exceeds 1.0: {val}"

    def test_top_half_nonzero(self, sequential_result):
        """Top 50% bucket should capture a significant fraction of games."""
        assert sequential_result.distribution_stats["top_half"] > 0.3

    def test_low_half_nonzero(self, sequential_result):
        assert sequential_result.distribution_stats["low_half"] > 0.3

    def test_top_quartile_nonzero(self, sequential_result):
        """Top 25% bucket should capture roughly 25% of games."""
        assert sequential_result.distribution_stats["top_quartile"] > 0.1

    def test_low_quartile_nonzero(self, sequential_result):
        assert sequential_result.distribution_stats["low_quartile"] > 0.1

    def test_top_decile_nonzero(self, sequential_result):
        assert sequential_result.distribution_stats["top_decile"] > 0.01

    def test_low_decile_nonzero(self, sequential_result):
        assert sequential_result.distribution_stats["low_decile"] > 0.01

    def test_bucket_ordering(self, sequential_result):
        """Larger buckets should capture at least as many games as smaller ones."""
        ds = sequential_result.distribution_stats
        assert ds["top_half"] >= ds["top_quartile"]
        assert ds["top_quartile"] >= ds["top_decile"]
        assert ds["top_decile"] >= ds["top_centile"]
        assert ds["low_half"] >= ds["low_quartile"]
        assert ds["low_quartile"] >= ds["low_decile"]
        assert ds["low_decile"] >= ds["low_centile"]

    def test_not_all_zero(self, sequential_result):
        """Distribution stats should not all be zero."""
        ds = sequential_result.distribution_stats
        assert sum(ds.values()) > 0

    def test_halves_sum_near_one(self, sequential_result):
        """top_half + low_half should be close to 1.0."""
        ds = sequential_result.distribution_stats
        total = ds["top_half"] + ds["low_half"]
        assert 0.9 <= total <= 1.1, f"Halves sum to {total}"


# ---------------------------------------------------------------------------
# Parallel path parity
# ---------------------------------------------------------------------------

class TestParallelParity:
    """Verify parallel path produces consistent results with sequential."""

    def test_mean_mana_matches(self, sequential_result, parallel_result):
        """CRN: same seed should give identical mean_mana."""
        assert sequential_result.mean_mana == parallel_result.mean_mana

    def test_mean_lands_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_lands == parallel_result.mean_lands

    def test_mean_mulls_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_mulls == parallel_result.mean_mulls

    def test_mean_draws_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_draws == parallel_result.mean_draws

    def test_mean_mana_value_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_mana_value == parallel_result.mean_mana_value

    def test_mean_mana_draw_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_mana_draw == parallel_result.mean_mana_draw

    def test_mean_mana_ramp_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_mana_ramp == parallel_result.mean_mana_ramp

    def test_mean_mana_total_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_mana_total == parallel_result.mean_mana_total

    def test_mean_ecms_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_ecms == parallel_result.mean_ecms

    def test_mean_hand_sum_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_hand_sum == parallel_result.mean_hand_sum

    def test_mean_bad_turns_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_bad_turns == parallel_result.mean_bad_turns

    def test_mean_mid_turns_matches(self, sequential_result, parallel_result):
        assert sequential_result.mean_mid_turns == parallel_result.mean_mid_turns

    def test_percentiles_match(self, sequential_result, parallel_result):
        assert sequential_result.percentile_25 == parallel_result.percentile_25
        assert sequential_result.percentile_50 == parallel_result.percentile_50
        assert sequential_result.percentile_75 == parallel_result.percentile_75

    def test_consistency_matches(self, sequential_result, parallel_result):
        assert sequential_result.consistency == parallel_result.consistency

    def test_threshold_matches(self, sequential_result, parallel_result):
        assert sequential_result.threshold_percent == parallel_result.threshold_percent
        assert sequential_result.threshold_mana == parallel_result.threshold_mana

    def test_ci_mana_value_matches(self, sequential_result, parallel_result):
        assert sequential_result.ci_mana_value == pytest.approx(parallel_result.ci_mana_value, rel=0.01)

    def test_ci_mana_draw_matches(self, sequential_result, parallel_result):
        assert sequential_result.ci_mana_draw == parallel_result.ci_mana_draw

    def test_ci_mana_ramp_matches(self, sequential_result, parallel_result):
        assert sequential_result.ci_mana_ramp == parallel_result.ci_mana_ramp

    def test_ci_mana_matches(self, sequential_result, parallel_result):
        assert sequential_result.ci_mana == pytest.approx(parallel_result.ci_mana, rel=0.01)

    def test_ci_mana_total_matches(self, sequential_result, parallel_result):
        assert sequential_result.ci_mana_total == pytest.approx(parallel_result.ci_mana_total, rel=0.01)


class TestParallelDistributionStats:
    """Verify parallel distribution_stats are populated and reasonable."""

    def test_all_keys_present(self, parallel_result):
        ds = parallel_result.distribution_stats
        expected = [
            "top_centile", "top_decile", "top_quartile", "top_half",
            "low_half", "low_quartile", "low_decile", "low_centile",
        ]
        for key in expected:
            assert key in ds, f"Missing key: {key}"

    def test_not_all_zero(self, parallel_result):
        ds = parallel_result.distribution_stats
        assert sum(ds.values()) > 0, "All distribution stats are zero in parallel path"

    def test_top_half_nonzero(self, parallel_result):
        assert parallel_result.distribution_stats["top_half"] > 0.3

    def test_low_half_nonzero(self, parallel_result):
        assert parallel_result.distribution_stats["low_half"] > 0.3

    def test_bucket_ordering(self, parallel_result):
        ds = parallel_result.distribution_stats
        assert ds["top_half"] >= ds["top_quartile"]
        assert ds["top_quartile"] >= ds["top_decile"]
        assert ds["top_decile"] >= ds["top_centile"]
        assert ds["low_half"] >= ds["low_quartile"]
        assert ds["low_quartile"] >= ds["low_decile"]
        assert ds["low_decile"] >= ds["low_centile"]

    def test_distribution_stats_close_to_sequential(self, sequential_result, parallel_result):
        """Both paths use the same raw data, so distribution stats should be identical."""
        ds_seq = sequential_result.distribution_stats
        ds_par = parallel_result.distribution_stats
        for key in ds_seq:
            assert ds_seq[key] == pytest.approx(ds_par[key], abs=0.001), (
                f"{key}: seq={ds_seq[key]}, par={ds_par[key]}"
            )


# ---------------------------------------------------------------------------
# as_row formatting
# ---------------------------------------------------------------------------

class TestAsRow:
    """Tests for SimulationResult.as_row() formatting."""

    def test_row_length(self, sequential_result):
        row = sequential_result.as_row()
        assert len(row) == 14

    def test_land_count_first(self, sequential_result):
        row = sequential_result.as_row()
        assert row[0] == sequential_result.land_count

    def test_mean_mana_formatted(self, sequential_result):
        row = sequential_result.as_row()
        assert "+/-" in row[1]
        assert str(f"{sequential_result.mean_mana:.2f}") in row[1]

    def test_consistency_formatted(self, sequential_result):
        row = sequential_result.as_row()
        assert "+/-" in row[2]

    def test_numeric_fields(self, sequential_result):
        """Fields 3-12 should be numeric."""
        row = sequential_result.as_row()
        for i in range(3, 13):
            assert isinstance(row[i], (int, float)), f"row[{i}] = {row[i]} is not numeric"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for simulation metrics."""

    def test_few_sims_no_crash(self):
        """Should not crash with very few sims (below calibration threshold)."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile", seed=1)
        result = gf.simulate()
        assert isinstance(result, SimulationResult)
        # Distribution stats should be all zeros (not enough data to calibrate)
        ds = result.distribution_stats
        assert all(v == 0 for v in ds.values())

    def test_record_results_none(self):
        """record_results=None should still produce valid metrics."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=200, record_results=None, seed=1)
        result = gf.simulate()
        assert result.mean_mana > 0
        ds = result.distribution_stats
        # record_half is always True, so half buckets are computed;
        # centile/decile/quartile should be zero when record_results=None
        assert ds["top_centile"] == 0
        assert ds["top_decile"] == 0
        assert ds["top_quartile"] == 0
        assert ds["low_centile"] == 0
        assert ds["low_decile"] == 0
        assert ds["low_quartile"] == 0

    def test_one_turn(self):
        """Simulation with 1 turn should still produce valid results."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=1, sims=200, record_results="quartile", seed=1)
        result = gf.simulate()
        assert result.mean_mana >= 0
        assert result.mean_lands >= 0
        assert result.mean_lands <= 1

    def test_more_lands_more_mean_lands(self):
        """A deck with more lands should result in higher mean lands played."""
        deck_low = _simple_deck(num_lands=30, num_spells=69)
        gf_low = Goldfisher(deck_low, turns=10, sims=500, record_results="quartile", seed=SEED)
        r_low = gf_low.simulate()

        deck_high = _simple_deck(num_lands=45, num_spells=54)
        gf_high = Goldfisher(deck_high, turns=10, sims=500, record_results="quartile", seed=SEED)
        r_high = gf_high.simulate()

        assert r_high.mean_lands > r_low.mean_lands

    def test_more_lands_fewer_bad_turns(self):
        """More lands should reduce bad turns (fewer missed land drops)."""
        deck_low = _simple_deck(num_lands=25, num_spells=74)
        gf_low = Goldfisher(deck_low, turns=10, sims=500, record_results="quartile", seed=SEED)
        r_low = gf_low.simulate()

        deck_high = _simple_deck(num_lands=40, num_spells=59)
        gf_high = Goldfisher(deck_high, turns=10, sims=500, record_results="quartile", seed=SEED)
        r_high = gf_high.simulate()

        # With enough lands, you should have fewer dead turns
        # (but too many lands also creates dead turns; 40 vs 25 should be better)
        assert r_high.mean_bad_turns <= r_low.mean_bad_turns


# ---------------------------------------------------------------------------
# Game records (sequential only — parallel doesn't record game logs)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Card performance
# ---------------------------------------------------------------------------

class TestCardPerformance:
    """Tests for card_performance tracking."""

    def test_card_performance_populated(self, sequential_result):
        cp = sequential_result.card_performance
        assert "high_performing" in cp
        assert "low_performing" in cp
        assert "total_games" in cp

    def test_high_performing_has_required_fields(self, sequential_result):
        for entry in sequential_result.card_performance["high_performing"]:
            assert "name" in entry
            assert "label" in entry
            assert "copies" in entry
            assert "cost" in entry
            assert "cmc" in entry
            assert "effects" in entry
            assert "mean_with" in entry
            assert "mean_without" in entry
            assert "score" in entry

    def test_high_scores_above_low_scores(self, sequential_result):
        """Best top performer should out-score worst low performer."""
        high = sequential_result.card_performance["high_performing"]
        low = sequential_result.card_performance["low_performing"]
        if high and low:
            assert max(e["score"] for e in high) >= min(e["score"] for e in low)

    def test_high_sorted_descending(self, sequential_result):
        scores = [e["score"] for e in sequential_result.card_performance["high_performing"]]
        assert scores == sorted(scores, reverse=True)

    def test_low_sorted_ascending(self, sequential_result):
        scores = [e["score"] for e in sequential_result.card_performance["low_performing"]]
        assert scores == sorted(scores)

    def test_max_10_entries(self, sequential_result):
        assert len(sequential_result.card_performance["high_performing"]) <= 10
        assert len(sequential_result.card_performance["low_performing"]) <= 10

    def test_mean_values_reasonable(self, sequential_result):
        for entries in (
            sequential_result.card_performance["high_performing"],
            sequential_result.card_performance["low_performing"],
        ):
            for entry in entries:
                assert entry["mean_with"] >= 0
                assert entry["mean_without"] >= 0

    def test_no_land_cards(self, sequential_result):
        """Card performance should not include land cards."""
        for entries in (
            sequential_result.card_performance["high_performing"],
            sequential_result.card_performance["low_performing"],
        ):
            for entry in entries:
                assert "Island" not in entry["name"]

    def test_parallel_has_card_performance(self, parallel_result):
        cp = parallel_result.card_performance
        assert "high_performing" in cp
        assert "low_performing" in cp

    def test_few_sims_returns_empty_dict(self):
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile", seed=1)
        result = gf.simulate()
        assert result.card_performance == {}

    def test_pooled_entries_have_label_and_copies(self, sequential_result):
        """Pooled output should include human-readable label and copy count."""
        cp = sequential_result.card_performance
        for entries in (cp["high_performing"], cp["low_performing"]):
            for entry in entries:
                assert isinstance(entry["label"], str) and entry["label"]
                assert entry["copies"] >= 1
                # vanilla pools (62 vanilla creatures across 6 cmc) should pool at least some
        # At least one pool with >1 copies expected from this deck shape
        all_entries = cp["high_performing"] + cp["low_performing"]
        assert any(e["copies"] > 1 for e in all_entries)


class TestReplayData:
    """Tests for structured replay data capture."""

    def test_replay_data_has_buckets(self, sequential_result):
        rd = sequential_result.replay_data
        assert "top" in rd
        assert "mid" in rd
        assert "low" in rd

    def test_buckets_have_games(self, sequential_result):
        rd = sequential_result.replay_data
        assert len(rd["top"]) > 0
        assert len(rd["low"]) > 0

    def test_buckets_capped_at_10(self, sequential_result):
        rd = sequential_result.replay_data
        for bucket in ("top", "mid", "low"):
            assert len(rd[bucket]) <= 10

    def test_game_has_required_fields(self, sequential_result):
        rd = sequential_result.replay_data
        for bucket in ("top", "mid", "low"):
            for game in rd[bucket]:
                assert "total_mana" in game
                assert "mulligans" in game
                assert "starting_hand" in game
                assert "turns" in game

    def test_starting_hand_is_list_of_strings(self, sequential_result):
        game = sequential_result.replay_data["top"][0]
        assert isinstance(game["starting_hand"], list)
        assert all(isinstance(name, str) for name in game["starting_hand"])

    def test_turns_have_snapshot_fields(self, sequential_result):
        game = sequential_result.replay_data["top"][0]
        for turn in game["turns"]:
            assert "turn" in turn
            assert "hand_before_draw" in turn
            assert "played" in turn
            assert "mana_spent_this_turn" in turn
            assert "total_mana_production" in turn
            assert "hand_after" in turn
            assert "battlefield" in turn
            assert "lands" in turn
            assert "graveyard" in turn

    def test_turn_numbers_sequential(self, sequential_result):
        game = sequential_result.replay_data["top"][0]
        turn_numbers = [t["turn"] for t in game["turns"]]
        assert turn_numbers == list(range(1, len(turn_numbers) + 1))

    def test_played_cards_have_fields(self, sequential_result):
        game = sequential_result.replay_data["top"][0]
        for turn in game["turns"]:
            for card in turn["played"]:
                assert "name" in card
                assert "cost" in card
                assert "mana_spent" in card
                assert "is_land" in card

    def test_top_bucket_higher_mana_than_low(self, sequential_result):
        rd = sequential_result.replay_data
        if rd["top"] and rd["low"]:
            avg_top = sum(g["total_mana"] for g in rd["top"]) / len(rd["top"])
            avg_low = sum(g["total_mana"] for g in rd["low"]) / len(rd["low"])
            assert avg_top > avg_low

    def test_parallel_returns_populated_replay_data(self, parallel_result):
        rd = parallel_result.replay_data
        assert "top" in rd
        assert "mid" in rd
        assert "low" in rd

    def test_parallel_replay_buckets_have_games(self, parallel_result):
        rd = parallel_result.replay_data
        assert len(rd["top"]) > 0
        assert len(rd["low"]) > 0

    def test_parallel_replay_game_has_required_fields(self, parallel_result):
        rd = parallel_result.replay_data
        for bucket in ("top", "mid", "low"):
            for game in rd[bucket]:
                assert "total_mana" in game
                assert "mulligans" in game
                assert "starting_hand" in game
                assert "turns" in game

    def test_parallel_replay_turns_have_snapshot_fields(self, parallel_result):
        game = parallel_result.replay_data["top"][0]
        for turn in game["turns"]:
            assert "turn" in turn
            assert "hand_before_draw" in turn
            assert "played" in turn
            assert "mana_spent_this_turn" in turn
            assert "total_mana_production" in turn
            assert "hand_after" in turn
            assert "battlefield" in turn
            assert "lands" in turn
            assert "graveyard" in turn

    def test_few_sims_returns_empty_replay_data(self):
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile", seed=1)
        result = gf.simulate()
        rd = result.replay_data
        assert rd["top"] == []
        assert rd["mid"] == []
        assert rd["low"] == []


class TestGameRecords:
    """Tests for game_records populated in sequential path."""

    def test_game_records_populated(self, sequential_result):
        """Game records should have entries for the distribution buckets."""
        gr = sequential_result.game_records
        assert len(gr) > 0

    def test_game_records_have_mana_data(self, sequential_result):
        gr = sequential_result.game_records
        # At least one bucket should have recorded mana data
        has_mana = any(
            "mana" in bucket_data and len(bucket_data["mana"]) > 0
            for bucket_data in gr.values()
        )
        assert has_mana

    def test_game_records_logs_capped(self, sequential_result):
        """Game record logs should be capped at 10 per bucket."""
        gr = sequential_result.game_records
        for bucket, data in gr.items():
            if "logs" in data:
                assert len(data["logs"]) <= 10, f"{bucket} has {len(data['logs'])} logs"


# ---------------------------------------------------------------------------
# Mana mode
# ---------------------------------------------------------------------------

class TestManaMode:
    """Tests for configurable mana_mode parameter."""

    def test_mana_mode_default_is_value(self):
        """Default mana_mode should be 'value' and produce identical results to explicit 'value'."""
        deck = _simple_deck()
        gf_default = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=SEED)
        gf_value = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=SEED, mana_mode="value")
        r_default = gf_default.simulate()
        r_value = gf_value.simulate()
        assert r_default.percentile_50 == r_value.percentile_50
        assert r_default.consistency == r_value.consistency

    def test_mana_mode_invalid_raises(self):
        """Invalid mana_mode should raise ValueError."""
        deck = _simple_deck()
        with pytest.raises(ValueError, match="Invalid mana_mode"):
            Goldfisher(deck, turns=5, sims=100, mana_mode="invalid")

    def test_mana_mode_value_draw_accepted(self):
        """mana_mode='value_draw' should run without error."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=SEED, mana_mode="value_draw")
        result = gf.simulate()
        assert isinstance(result, SimulationResult)

    def test_mana_mode_total_accepted(self):
        """mana_mode='total' should run without error."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=SEED, mana_mode="total")
        result = gf.simulate()
        assert isinstance(result, SimulationResult)

    def test_mana_mode_vanilla_deck_all_modes_equal(self):
        """For a vanilla deck (no draw/ramp), all mana modes should produce identical percentiles."""
        deck = _simple_deck()
        results = {}
        for mode in ("value", "value_draw", "total"):
            gf = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=SEED, mana_mode=mode)
            results[mode] = gf.simulate()
        # All modes should produce the same percentiles since there are no draw/ramp cards
        assert results["value"].percentile_50 == results["value_draw"].percentile_50
        assert results["value"].percentile_50 == results["total"].percentile_50
        assert results["value"].consistency == results["value_draw"].consistency
        assert results["value"].consistency == results["total"].consistency

    def test_mana_mode_parallel_parity(self):
        """Sequential and parallel paths should agree for mana_mode='value_draw'."""
        deck = _simple_deck()
        gf_seq = Goldfisher(deck, turns=5, sims=SIMS, record_results="quartile", seed=SEED, mana_mode="value_draw")
        r_seq = gf_seq.simulate()
        gf_par = Goldfisher(deck, turns=5, sims=SIMS, record_results="quartile", seed=SEED, mana_mode="value_draw", workers=2)
        r_par = gf_par.simulate()
        assert r_seq.percentile_50 == r_par.percentile_50
        assert r_seq.consistency == r_par.consistency
        assert r_seq.mean_mana_value == r_par.mean_mana_value
