"""Unit tests for the factored optimizer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from auto_goldfish.optimization.candidate_cards import ALL_CANDIDATES
from auto_goldfish.optimization.deck_config import DeckConfig
from auto_goldfish.optimization.factored_optimizer import (
    FactoredOptimizer,
    MarginalResult,
    _classify_config,
    _paired_p_value,
)


class TestPairedPValue:
    def test_identical_values_returns_one(self):
        diffs = np.zeros(100)
        assert _paired_p_value(diffs) == 1.0

    def test_large_positive_diff_returns_small_p(self):
        diffs = np.ones(200) * 10.0
        p = _paired_p_value(diffs)
        assert p < 0.001

    def test_empty_returns_one(self):
        assert _paired_p_value(np.array([])) == 1.0

    def test_moderate_diff_returns_small_but_nonzero_p(self):
        rng = np.random.RandomState(42)
        diffs = rng.normal(0.3, 1.0, 200)
        p = _paired_p_value(diffs)
        assert 0.0 < p < 0.01  # 200 samples, mean 0.3, SD 1.0 -> strong signal


class TestClassifyConfig:
    def test_land_only(self):
        assert _classify_config(DeckConfig(land_delta=1)) == "land"

    def test_draw_card(self):
        assert _classify_config(DeckConfig(added_cards=("draw_2cmc_2",))) == "draw"

    def test_ramp_card(self):
        assert _classify_config(DeckConfig(added_cards=("ramp_2cmc_1",))) == "ramp"


class TestMarginalEnumeration:
    def _make_optimizer(self, candidates, land_range=2, max_draw=1, max_ramp=1):
        gf = MagicMock()
        gf.seed = 42
        return FactoredOptimizer(
            goldfisher=gf, candidates=candidates,
            land_range=land_range, max_draw=max_draw, max_ramp=max_ramp,
        )

    def test_land_only(self):
        opt = self._make_optimizer({}, land_range=2, max_draw=0, max_ramp=0)
        configs = opt._enumerate_marginals()
        # -2, -1, +1, +2
        assert len(configs) == 4
        deltas = sorted(c.land_delta for c in configs)
        assert deltas == [-2, -1, 1, 2]

    def test_with_candidates(self):
        candidates = {
            "draw_2cmc_2": ALL_CANDIDATES["draw_2cmc_2"],
            "ramp_2cmc_1": ALL_CANDIDATES["ramp_2cmc_1"],
        }
        opt = self._make_optimizer(candidates, land_range=1)
        configs = opt._enumerate_marginals()
        # -1, +1 lands + 1 draw + 1 ramp = 4
        assert len(configs) == 4

    def test_excludes_zero_delta(self):
        opt = self._make_optimizer({}, land_range=1)
        configs = opt._enumerate_marginals()
        assert all(c.land_delta != 0 for c in configs)

    def test_respects_land_delta_min_max(self):
        gf = MagicMock()
        gf.seed = 42
        opt = FactoredOptimizer(
            goldfisher=gf, candidates={},
            land_delta_min=-1, land_delta_max=3,
            max_draw=0, max_ramp=0,
        )
        configs = opt._enumerate_marginals()
        deltas = sorted(c.land_delta for c in configs)
        assert deltas == [-1, 1, 2, 3]


class TestCombinations:
    def _make_marginal(self, config, dimension, effect):
        return MarginalResult(
            config=config, dimension=dimension,
            effect_size=effect, se=0.1, p_value=0.01,
            n_games=200, significant=effect > 0, negligible=False,
        )

    def test_pairwise_and_triple(self):
        opt = FactoredOptimizer(
            goldfisher=MagicMock(), candidates={},
            max_draw=1, max_ramp=1,
        )
        opt.marginal_results = [
            self._make_marginal(DeckConfig(land_delta=1), "land", 1.0),
            self._make_marginal(DeckConfig(added_cards=("draw_2cmc_2",)), "draw", 0.5),
            self._make_marginal(DeckConfig(added_cards=("ramp_2cmc_1",)), "ramp", 0.3),
        ]
        combos = opt._build_combinations()
        # 3 pairwise + 1 triple = 4
        assert len(combos) >= 4

    def test_skips_negative_dimensions(self):
        opt = FactoredOptimizer(
            goldfisher=MagicMock(), candidates={},
            max_draw=1, max_ramp=1,
        )
        opt.marginal_results = [
            self._make_marginal(DeckConfig(land_delta=1), "land", 1.0),
            self._make_marginal(DeckConfig(added_cards=("draw_2cmc_2",)), "draw", -0.5),
            self._make_marginal(DeckConfig(added_cards=("ramp_2cmc_1",)), "ramp", -0.3),
        ]
        combos = opt._build_combinations()
        # Only land is positive, no pairwise possible
        assert len(combos) == 0

    def test_two_copy_variant(self):
        opt = FactoredOptimizer(
            goldfisher=MagicMock(), candidates={},
            max_draw=2, max_ramp=1,
        )
        opt.marginal_results = [
            self._make_marginal(DeckConfig(land_delta=1), "land", 1.0),
            self._make_marginal(DeckConfig(added_cards=("draw_2cmc_2",)), "draw", 0.5),
        ]
        combos = opt._build_combinations()
        two_copy = [c for c in combos if len(c.added_cards) == 2]
        assert len(two_copy) >= 1
        assert two_copy[0].added_cards == ("draw_2cmc_2", "draw_2cmc_2")


class TestScoring:
    def test_mean_mana(self):
        opt = FactoredOptimizer(
            goldfisher=MagicMock(), candidates={}, optimize_for="mean_mana",
        )
        values = np.array([10.0, 20.0, 30.0])
        assert opt._compute_score(values) == 20.0

    def test_consistency(self):
        opt = FactoredOptimizer(
            goldfisher=MagicMock(), candidates={}, optimize_for="consistency",
        )
        # [1, 2, 3, 4] -> bottom 25% = [1], mean=1.0; overall mean=2.5
        # consistency = 1.0/2.5 = 0.4
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(opt._compute_score(values) - 0.4) < 0.01

    def test_empty_values(self):
        opt = FactoredOptimizer(
            goldfisher=MagicMock(), candidates={}, optimize_for="mean_mana",
        )
        assert opt._compute_score(np.array([])) == 0.0

    def test_extract_score_from_dict_all_metrics(self):
        result_dict = {
            "mean_mana": 10.0,
            "consistency": 0.8,
            "mean_mana_value": 7.0,
            "mean_mana_total": 12.0,
            "mean_spells_cast": 5.0,
            "threshold_mana": 6.0,
        }
        metric_map = {
            "mean_mana": 10.0,
            "consistency": 0.8,
            "mean_mana_value": 7.0,
            "mean_mana_total": 12.0,
            "mean_spells_cast": 5.0,
            "floor_performance": 6.0,
        }
        for metric, expected in metric_map.items():
            opt = FactoredOptimizer(
                goldfisher=MagicMock(), candidates={}, optimize_for=metric,
            )
            assert opt._extract_score_from_dict(result_dict) == expected


class TestRecommendations:
    def test_significant_positive_produces_recommendation(self):
        from auto_goldfish.optimization.feature_analysis import (
            synthesize_factored_recommendations,
        )

        results = [
            MarginalResult(
                config=DeckConfig(added_cards=("draw_2cmc_2",)),
                dimension="draw", effect_size=1.5, se=0.3,
                p_value=0.001, n_games=200, significant=True, negligible=False,
            ),
        ]
        recs = synthesize_factored_recommendations(results, "mean_mana")
        assert len(recs) == 1
        assert recs[0]["confidence"] == "high"
        assert "Add" in recs[0]["label"]

    def test_negligible_produces_no_recommendation(self):
        from auto_goldfish.optimization.feature_analysis import (
            synthesize_factored_recommendations,
        )

        results = [
            MarginalResult(
                config=DeckConfig(land_delta=1),
                dimension="land", effect_size=0.01, se=0.5,
                p_value=0.9, n_games=200, significant=False, negligible=True,
            ),
        ]
        recs = synthesize_factored_recommendations(results, "mean_mana")
        assert len(recs) == 0

    def test_negative_produces_dont_recommendation(self):
        from auto_goldfish.optimization.feature_analysis import (
            synthesize_factored_recommendations,
        )

        results = [
            MarginalResult(
                config=DeckConfig(land_delta=1),
                dimension="land", effect_size=-2.0, se=0.3,
                p_value=0.001, n_games=200, significant=True, negligible=False,
            ),
        ]
        recs = synthesize_factored_recommendations(results, "mean_mana")
        assert len(recs) == 1
        assert "Don't" in recs[0]["label"] or "Cut" not in recs[0]["label"]

    def test_output_format(self):
        from auto_goldfish.optimization.feature_analysis import (
            synthesize_factored_recommendations,
        )

        results = [
            MarginalResult(
                config=DeckConfig(added_cards=("ramp_2cmc_1",)),
                dimension="ramp", effect_size=0.8, se=0.2,
                p_value=0.01, n_games=400, significant=True, negligible=False,
            ),
        ]
        recs = synthesize_factored_recommendations(results, "floor_performance")
        rec = recs[0]
        assert "recommendation" in rec
        assert "impact" in rec
        assert "confidence" in rec
        assert "label" in rec
        assert "detail" in rec
