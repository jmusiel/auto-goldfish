"""Tests for the Pyodide simulation runner entry point."""

import json
from unittest.mock import MagicMock

import pytest

from auto_goldfish.decklist.loader import get_basic_island
from auto_goldfish.pyodide_runner import run_optimization, run_simulation


def _make_deck_json(n_lands=10, n_spells=5):
    """Build a minimal deck JSON string for testing."""
    deck = [get_basic_island() for _ in range(n_lands)]
    for i in range(n_spells):
        deck.append({
            "name": f"Bear {i}",
            "quantity": 1,
            "oracle_cmc": 2,
            "cmc": 2,
            "cost": "{1}{G}",
            "text": "",
            "types": ["Creature"],
            "sub_types": [],
            "super_types": [],
            "identity": ["Green"],
            "user_category": "Creature",
            "commander": False,
        })
    return json.dumps(deck)


class TestRunSimulation:
    def test_returns_valid_json(self):
        """run_simulation returns a JSON string of result dicts."""
        deck_json = _make_deck_json()
        config = json.dumps({
            "turns": 3,
            "sims": 10,
            "min_lands": 10,
            "max_lands": 10,
            "seed": 42,
        })
        result_str = run_simulation(deck_json, config)
        results = json.loads(result_str)

        assert isinstance(results, list)
        assert len(results) == 1  # single land count
        r = results[0]
        assert r["land_count"] == 10
        assert "mean_mana" in r
        assert "consistency" in r
        assert "replay_data" in r
        assert "ci_mean_mana" in r

    def test_land_sweep(self):
        """Multiple land counts produce multiple results."""
        deck_json = _make_deck_json()
        config = json.dumps({
            "turns": 3,
            "sims": 10,
            "min_lands": 9,
            "max_lands": 11,
            "seed": 42,
        })
        results = json.loads(run_simulation(deck_json, config))
        assert len(results) == 3
        assert [r["land_count"] for r in results] == [9, 10, 11]

    def test_progress_callback(self):
        """Progress callback is called with global progress across land counts."""
        deck_json = _make_deck_json()
        sims = 10
        config = json.dumps({
            "turns": 3,
            "sims": sims,
            "min_lands": 10,
            "max_lands": 11,
            "seed": 42,
        })
        callback = MagicMock()
        run_simulation(deck_json, config, progress_callback=callback)

        # 2 land counts * 10 sims = 20 total calls
        assert callback.call_count == 2 * sims
        # First call: (0, 20)
        assert callback.call_args_list[0].args == (0, 20)
        # Last call: (19, 20)
        assert callback.call_args_list[-1].args == (19, 20)

    def test_with_effect_overrides(self):
        """Effect overrides are applied to the simulation."""
        deck_json = _make_deck_json()
        config = json.dumps({
            "turns": 3,
            "sims": 10,
            "min_lands": 10,
            "max_lands": 10,
            "seed": 42,
            "effect_overrides": {
                "Bear 0": {
                    "effects": [{"type": "draw_cards", "slot": "on_play", "params": {"amount": 2}}],
                },
            },
        })
        results = json.loads(run_simulation(deck_json, config))
        assert len(results) == 1
        assert results[0]["mean_mana"] >= 0

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        deck_json = _make_deck_json()
        config = json.dumps({
            "turns": 3,
            "sims": 50,
            "min_lands": 10,
            "max_lands": 10,
            "seed": 12345,
        })
        r1 = json.loads(run_simulation(deck_json, config))
        r2 = json.loads(run_simulation(deck_json, config))
        assert r1[0]["mean_mana"] == r2[0]["mean_mana"]
        assert r1[0]["consistency"] == r2[0]["consistency"]

    def test_defaults_land_range_to_deck_count(self):
        """When min/max lands not specified, uses deck's land count."""
        deck_json = _make_deck_json(n_lands=8)
        config = json.dumps({
            "turns": 3,
            "sims": 10,
            "seed": 42,
        })
        results = json.loads(run_simulation(deck_json, config))
        assert len(results) == 1
        assert results[0]["land_count"] == 8

    def test_curve_aware_mulligan(self):
        """Curve-aware mulligan strategy is supported."""
        deck_json = _make_deck_json()
        config = json.dumps({
            "turns": 3,
            "sims": 10,
            "min_lands": 10,
            "max_lands": 10,
            "seed": 42,
            "mulligan": "curve_aware",
        })
        results = json.loads(run_simulation(deck_json, config))
        assert len(results) == 1

    def test_result_dict_structure(self):
        """Result dict has all expected keys from result_to_dict."""
        deck_json = _make_deck_json()
        config = json.dumps({
            "turns": 3,
            "sims": 50,
            "min_lands": 10,
            "max_lands": 10,
            "seed": 42,
            "record_results": "quartile",
        })
        results = json.loads(run_simulation(deck_json, config))
        r = results[0]
        expected_keys = {
            "land_count", "mean_mana", "consistency",
            "mean_bad_turns", "mean_mid_turns", "mean_lands",
            "mean_mulls", "mean_draws",
            "percentile_25", "percentile_50", "percentile_75",
            "threshold_percent", "threshold_mana",
            "distribution_stats", "card_performance", "replay_data",
            "ci_mean_mana", "ci_consistency", "ci_mean_bad_turns",
        }
        assert expected_keys.issubset(set(r.keys()))


class TestRunOptimizationDispatch:
    """Verify the algorithm string routes to the correct optimizer class.

    Regression: the UI defaults to ``algorithm="factored"`` but pyodide_runner
    used to fall through to DeckOptimizer (Hyperband), so client-side runs
    never exercised the FactoredOptimizer despite the form selection.
    """

    def _make_optimization_config(self, algorithm: str, **extra) -> str:
        cfg = {
            "turns": 3,
            "sims": 5,
            "seed": 42,
            "min_lands": 10,
            "max_lands": 11,
            "algorithm": algorithm,
            "enabled_candidates": [],
            "max_draw_additions": 0,
            "max_ramp_additions": 0,
        }
        cfg.update(extra)
        return json.dumps(cfg)

    @pytest.mark.parametrize(
        "algorithm,expected_module,expected_class",
        [
            ("factored", "auto_goldfish.optimization.factored_optimizer", "FactoredOptimizer"),
            ("racing", "auto_goldfish.optimization.fast_optimizer", "FastDeckOptimizer"),
            ("hyperband", "auto_goldfish.optimization.optimizer", "DeckOptimizer"),
        ],
    )
    def test_algorithm_dispatch(self, monkeypatch, algorithm, expected_module, expected_class):
        """Each algorithm string must instantiate the matching optimizer class."""
        import importlib

        captured = {"class_name": None, "kwargs": None}

        for mod_path, cls_name in [
            ("auto_goldfish.optimization.factored_optimizer", "FactoredOptimizer"),
            ("auto_goldfish.optimization.fast_optimizer", "FastDeckOptimizer"),
            ("auto_goldfish.optimization.optimizer", "DeckOptimizer"),
        ]:
            mod = importlib.import_module(mod_path)
            real_cls = getattr(mod, cls_name)
            captured_name = cls_name

            def make_recording_init(_real_cls, _name):
                original_init = _real_cls.__init__

                def recording_init(self, *args, **kwargs):
                    captured["class_name"] = _name
                    captured["kwargs"] = kwargs
                    original_init(self, *args, **kwargs)

                return recording_init

            monkeypatch.setattr(real_cls, "__init__", make_recording_init(real_cls, captured_name))

        deck_json = _make_deck_json()
        config_json = self._make_optimization_config(algorithm)
        run_optimization(deck_json, config_json)

        assert captured["class_name"] == expected_class

    def _patch_factored(self, monkeypatch):
        """Capture kwargs passed to FactoredOptimizer."""
        import auto_goldfish.optimization.factored_optimizer as ff

        captured = {"kwargs": None}
        original_init = ff.FactoredOptimizer.__init__

        def recording_init(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(ff.FactoredOptimizer, "__init__", recording_init)
        return captured

    def test_factored_algorithm_forwards_mixed_combos_default_true(self, monkeypatch):
        captured = self._patch_factored(monkeypatch)
        run_optimization(_make_deck_json(), self._make_optimization_config("factored"))
        assert captured["kwargs"]["mixed_combos"] is True

    def test_factored_algorithm_forwards_mixed_combos_when_false(self, monkeypatch):
        captured = self._patch_factored(monkeypatch)
        run_optimization(
            _make_deck_json(),
            self._make_optimization_config("factored", mixed_combos=False),
        )
        assert captured["kwargs"]["mixed_combos"] is False
