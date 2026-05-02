"""Unit tests for the deck scoring module (CASTER profile)."""

import pytest

from auto_goldfish.engine.goldfisher import SimulationResult
from auto_goldfish.metrics.deck_score import (
    DEFAULT_ANCHORS,
    DeckRawStats,
    DeckScore,
    StatAnchors,
    _clamp,
    _compute_acceleration,
    _compute_consistency,
    _compute_efficiency,
    _compute_reach,
    _compute_snowball,
    _compute_tuning,
    _scale,
    compute_deck_score,
    compute_raw_stats,
    score_from_raw,
)
from auto_goldfish.optimization.curve_value import (
    CurveValueResult,
    CurveVerdict,
    CurveVerdictRow,
    ImpliedDrawResult,
    ImpliedSpellValueResult,
)


def _make_curve_value(
    *,
    verdict_rows=None,
    actual_deficit: float = 0.0,
    n_max: float = 20.0,
    no_ramp: bool = False,
) -> CurveValueResult:
    """Build a synthetic CurveValueResult for deck-score testing.

    `verdict_rows` is a list of dicts with keys (cmc, n_cards, kind);
    other CurveVerdictRow fields default to neutral values. `actual_deficit`
    and `n_max` populate the implied_draw side."""
    rows = [
        CurveVerdictRow(
            cmc=r["cmc"], n_cards=r["n_cards"], mt_per_slot=1.0,
            a_required=1.0, b_implicit=1.0, kind=r["kind"],
        )
        for r in (verdict_rows or [])
    ]
    verdict = CurveVerdict(
        delta=1.0 if no_ramp else 0.69, median_irr=float("nan") if no_ramp else 0.45,
        baseline_cmc=2, no_ramp=no_ramp, T=8, net_flat=0.0, rows=rows,
    )
    implied_draw = ImpliedDrawResult(
        L=37, R=0, V=60, D=99, V_avg_cmc=3.0,
        land_mana=20.0, ramp_excess=0.0, total_mana=20.0,
        commander_mana=0.0, value_mana=20.0,
        N_natural=14, N_max=n_max, deficit_max=max(0.0, n_max - 14),
        actual_deficit=actual_deficit, actual_total_draws=n_max - actual_deficit,
    )
    isv = ImpliedSpellValueResult(
        median_irr=verdict.median_irr, delta=verdict.delta, baseline_cmc=2,
        power_multipliers={2: 1.0}, per_card_irrs=[], no_ramp=no_ramp,
    )
    return CurveValueResult(
        turns=8, deck_size_effective=99,
        implied_draw=implied_draw, implied_spell_value=isv,
        curve_verdict=verdict,
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
        # Structural snapshot used by Toughness.
        "mana_source_count": 38,
        "draw_count": 10,
        "early_count": 22,
        "avg_cmc": 3.0,
    }
    defaults.update(overrides)
    return SimulationResult(**defaults)


# ---------------------------------------------------------------------------
# _clamp and _scale
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert _clamp(5.0) == 5

    def test_below_minimum(self):
        assert _clamp(-5.0) == 1

    def test_above_maximum(self):
        assert _clamp(25.0) == 10

    def test_rounds(self):
        assert _clamp(5.4) == 5
        assert _clamp(5.6) == 6


class TestScale:
    def test_midpoint(self):
        assert _scale(50.0, 0.0, 100.0) == 6

    def test_minimum(self):
        assert _scale(0.0, 0.0, 100.0) == 1

    def test_maximum(self):
        assert _scale(100.0, 0.0, 100.0) == 10

    def test_below_range(self):
        assert _scale(-10.0, 0.0, 100.0) == 1

    def test_above_range(self):
        assert _scale(110.0, 0.0, 100.0) == 10

    def test_equal_bounds_returns_5(self):
        assert _scale(5.0, 5.0, 5.0) == 5


# ---------------------------------------------------------------------------
# Individual stat computation
# ---------------------------------------------------------------------------

class TestAcceleration:
    def test_fast_deck_scores_high(self):
        result = _make_result(mean_mana_per_turn=[3.0, 4.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        assert _compute_acceleration(result, 10) >= 9

    def test_slow_deck_scores_low(self):
        result = _make_result(mean_mana_per_turn=[0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert _compute_acceleration(result, 10) <= 2

    def test_empty_per_turn_returns_1(self):
        result = _make_result(mean_mana_per_turn=[])
        assert _compute_acceleration(result, 10) == 1

    def test_fewer_than_4_turns(self):
        result = _make_result(mean_mana_per_turn=[2.0, 3.0])
        score = _compute_acceleration(result, 2)
        assert 1 <= score <= 10


class TestReach:
    def test_high_reach(self):
        result = _make_result(mean_mana=40.0, ceiling_mana=50.0)
        assert _compute_reach(result, 10) >= 9

    def test_low_reach(self):
        result = _make_result(mean_mana=5.0, ceiling_mana=8.0)
        assert _compute_reach(result, 10) <= 2

    def test_scales_with_turns(self):
        result = _make_result(mean_mana=20.0, ceiling_mana=28.0)
        score_10 = _compute_reach(result, 10)
        score_5 = _compute_reach(result, 5)
        # With fewer turns, same raw values should score higher.
        assert score_5 >= score_10


class TestConsistency:
    def test_perfect_consistency(self):
        result = _make_result(consistency=1.0, mean_bad_turns=0.0, std_mana=0.0, mean_mana=20.0)
        assert _compute_consistency(result, 10) == 10

    def test_terrible_consistency(self):
        result = _make_result(consistency=0.0, mean_bad_turns=6.0, std_mana=15.0, mean_mana=10.0)
        assert _compute_consistency(result, 10) <= 2

    def test_mid_consistency(self):
        result = _make_result(consistency=0.7, mean_bad_turns=1.5, std_mana=5.0, mean_mana=20.0)
        score = _compute_consistency(result, 10)
        assert 4 <= score <= 8


class TestTuning:
    def test_all_coherent_scores_high(self):
        """A deck with every slot tagged coherent or over_allocated maxes out."""
        cv = _make_curve_value(verdict_rows=[
            {"cmc": 2, "n_cards": 12, "kind": "baseline"},
            {"cmc": 3, "n_cards": 10, "kind": "coherent"},
            {"cmc": 4, "n_cards": 8, "kind": "over_allocated"},
        ])
        assert _compute_tuning(cv) == 10

    def test_all_ramp_over_aggressive_scores_low(self):
        """Every slot tagged ramp_over_aggressive (apart from baseline) bottoms out."""
        cv = _make_curve_value(verdict_rows=[
            {"cmc": 2, "n_cards": 8, "kind": "baseline"},
            {"cmc": 4, "n_cards": 8, "kind": "ramp_over_aggressive"},
            {"cmc": 6, "n_cards": 8, "kind": "ramp_over_aggressive"},
        ])
        # 16 of 24 cards penalized → coherence fraction = 8/24 ≈ 0.33
        # Below the (0.50, 1.00) anchor → score 1.
        assert _compute_tuning(cv) == 1

    def test_no_curve_value_returns_anchor_floor(self):
        """Tuning falls back to a neutral 0.5 raw when curve_value is
        unavailable. Anchors (0.50, 1.00) map that to score 1."""
        assert _compute_tuning(None) == 1

    def test_no_ramp_deck_scores_well(self):
        """No-ramp decks have delta=1.0; A_required is duration-only and
        most slots come back coherent or over_allocated."""
        cv = _make_curve_value(no_ramp=True, verdict_rows=[
            {"cmc": 1, "n_cards": 5, "kind": "below_baseline"},
            {"cmc": 2, "n_cards": 12, "kind": "baseline"},
            {"cmc": 3, "n_cards": 10, "kind": "coherent"},
        ])
        assert _compute_tuning(cv) >= 8


class TestEfficiency:
    def test_perfect_alignment_scores_high(self):
        """Zero deficit (deck draws everything it needs) maxes out efficiency."""
        cv = _make_curve_value(actual_deficit=0.0, n_max=22.0)
        assert _compute_efficiency(cv) == 10

    def test_large_deficit_scores_low(self):
        """A deck that's 70% short of its draw requirement scores 1."""
        # raw = 1 - 14/20 = 0.30 → at the lower anchor.
        cv = _make_curve_value(actual_deficit=14.0, n_max=20.0)
        assert _compute_efficiency(cv) == 1

    def test_no_curve_value_returns_mid_score(self):
        """Efficiency falls back to a neutral 0.5 raw. Anchors (0.30, 1.00)
        map that to ~score 4, mid-range but not maxed (signals "we have no
        signal" rather than "the deck is great")."""
        assert _compute_efficiency(None) == 4


class TestSnowball:
    def test_strong_acceleration(self):
        result = _make_result(
            mean_mana_per_turn=[0.5, 1.0, 1.5, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        assert _compute_snowball(result, 10) >= 7

    def test_flat_curve(self):
        result = _make_result(
            mean_mana_per_turn=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        )
        score = _compute_snowball(result, 10)
        assert score <= 6

    def test_short_game_returns_5(self):
        result = _make_result(mean_mana_per_turn=[1.0, 2.0, 3.0])
        assert _compute_snowball(result, 3) == 5


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
            assert 1 <= value <= 10, f"{name}={value} out of range"

    def test_as_dict_keys(self):
        score = compute_deck_score(_make_result(), turns=10)
        keys = set(score.as_dict().keys())
        assert keys == {
            "consistency", "acceleration", "snowball",
            "tuning", "efficiency", "reach",
        }

    def test_format_block_contains_all_stats(self):
        score = compute_deck_score(_make_result(), turns=10)
        block = score.format_block()
        for stat in [
            "CONSISTENCY", "ACCELERATION", "SNOWBALL",
            "TUNING", "EFFICIENCY", "REACH",
        ]:
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
            assert 1 <= value <= 10

    def test_single_turn(self):
        result = _make_result(mean_mana_per_turn=[3.0], mean_spells_per_turn=[1.0])
        score = compute_deck_score(result, turns=1)
        for value in score.as_dict().values():
            assert 1 <= value <= 10


# ---------------------------------------------------------------------------
# StatAnchors and raw stats
# ---------------------------------------------------------------------------

class TestStatAnchors:
    def test_default_anchors_match_historical_values(self):
        assert DEFAULT_ANCHORS.consistency == (0.0, 1.0)
        assert DEFAULT_ANCHORS.acceleration == (1.0, 14.0)
        assert DEFAULT_ANCHORS.snowball_ratio == (0.5, 4.0)
        assert DEFAULT_ANCHORS.snowball_late_avg_norm == (1.0, 8.0)
        assert DEFAULT_ANCHORS.tuning == (0.50, 1.00)
        assert DEFAULT_ANCHORS.efficiency == (0.30, 1.00)
        assert DEFAULT_ANCHORS.reach_norm == (5.0, 45.0)

    def test_anchors_are_immutable(self):
        with pytest.raises(Exception):
            DEFAULT_ANCHORS.consistency = (0.0, 2.0)


class TestComputeRawStats:
    def test_returns_top_level_fields(self):
        raw = compute_raw_stats(_make_result(), turns=10)
        assert isinstance(raw, DeckRawStats)
        keys = set(raw.as_dict().keys())
        # Six CASTER stats plus the secondary snowball input that ships
        # alongside so server-side persist can re-score snowball with
        # active anchors.
        assert keys == {
            "consistency", "acceleration", "snowball",
            "tuning", "efficiency", "reach",
            "snowball_late_avg_norm",
        }

    def test_raw_values_are_floats(self):
        raw = compute_raw_stats(_make_result(), turns=10)
        for value in raw.as_dict().values():
            assert isinstance(value, float)

    def test_raw_tuning_neutral_when_no_curve_value(self):
        result = _make_result()
        raw = compute_raw_stats(result, turns=10, curve_value=None)
        # Falls back to neutral 0.5 when no verdict available.
        assert raw.tuning == pytest.approx(0.5)

    def test_raw_tuning_uses_curve_verdict_kinds(self):
        result = _make_result()
        cv = _make_curve_value(verdict_rows=[
            {"cmc": 2, "n_cards": 8, "kind": "baseline"},
            {"cmc": 4, "n_cards": 4, "kind": "ramp_over_aggressive"},
        ])
        raw = compute_raw_stats(result, turns=10, curve_value=cv)
        # Bad fraction = 4/12 → coherence = 8/12 = 0.667.
        assert raw.tuning == pytest.approx(8.0 / 12.0, abs=1e-6)

    def test_raw_efficiency_uses_implied_draw_deficit(self):
        result = _make_result()
        cv = _make_curve_value(actual_deficit=4.0, n_max=20.0)
        raw = compute_raw_stats(result, turns=10, curve_value=cv)
        # raw = 1 - 4/20 = 0.80
        assert raw.efficiency == pytest.approx(0.80, abs=1e-6)

    def test_raw_acceleration_sums_first_four_turns(self):
        result = _make_result(mean_mana_per_turn=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        raw = compute_raw_stats(result, turns=10)
        assert raw.acceleration == pytest.approx(10.0)


class TestScoreFromRaw:
    def test_default_anchors_match_compute_deck_score(self):
        result = _make_result()
        raw = compute_raw_stats(result, turns=10)
        from_raw = score_from_raw(raw, DEFAULT_ANCHORS)
        direct = compute_deck_score(result, turns=10)
        assert from_raw.as_dict() == direct.as_dict()

    def test_custom_anchors_change_output(self):
        """Widening the tuning window should make a mid-deck score lower
        because the same raw value now lands further from the upper anchor."""
        result = _make_result()
        cv = _make_curve_value(verdict_rows=[
            {"cmc": 2, "n_cards": 12, "kind": "baseline"},
            {"cmc": 4, "n_cards": 4, "kind": "ramp_over_aggressive"},
        ])
        raw = compute_raw_stats(result, turns=10, curve_value=cv)
        wide = StatAnchors(tuning=(0.0, 2.0))
        narrow = StatAnchors(tuning=(0.50, 1.00))  # default
        wide_score = score_from_raw(raw, wide)
        narrow_score = score_from_raw(raw, narrow)
        assert wide_score.tuning < narrow_score.tuning

    def test_custom_anchors_only_affect_specified_stat(self):
        """Overriding one anchor leaves the other stats unchanged."""
        result = _make_result()
        raw = compute_raw_stats(result, turns=10)
        custom = StatAnchors(consistency=(0.5, 0.6))  # tight
        custom_score = score_from_raw(raw, custom)
        default_score = score_from_raw(raw, DEFAULT_ANCHORS)
        # Only consistency should differ.
        for stat in ["acceleration", "snowball", "tuning", "efficiency", "reach"]:
            assert getattr(custom_score, stat) == getattr(default_score, stat)

    def test_short_game_snowball_returns_5(self):
        """No late-game data => Snowball falls back to neutral 5."""
        result = _make_result(mean_mana_per_turn=[3.0, 4.0])
        raw = compute_raw_stats(result, turns=2)
        score = score_from_raw(raw)
        assert score.snowball == 5

    def test_compute_deck_score_accepts_anchors_kwarg(self):
        """compute_deck_score forwards the anchors arg to score_from_raw."""
        result = _make_result()
        custom = StatAnchors(reach_norm=(1.0, 2.0))  # very tight, easy 10
        score = compute_deck_score(result, turns=10, anchors=custom)
        assert score.reach == 10
