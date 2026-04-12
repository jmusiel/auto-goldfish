"""Integration tests for deck optimization."""

import json

from auto_goldfish.engine.goldfisher import Goldfisher
from auto_goldfish.optimization.candidate_cards import ALL_CANDIDATES
from auto_goldfish.optimization.deck_config import DeckConfig, apply_config
from auto_goldfish.optimization.optimizer import DeckOptimizer


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


def test_apply_config_no_changes():
    """Applying base config leaves deck unchanged."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile")
    original_count = len(gf.decklist)
    apply_config(gf, DeckConfig(), {})
    assert len(gf.decklist) == original_count


def test_apply_config_add_land():
    """Applying a +1 land config increases land count."""
    deck = _simple_deck(num_lands=37)
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile")
    apply_config(gf, DeckConfig(land_delta=1), {})
    assert gf.land_count == 38


def test_apply_config_add_draw_spell():
    """Adding a draw spell increases deck size (no swap mode)."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile")
    original_count = len(gf.decklist)
    candidates = {"draw_2cmc_2": ALL_CANDIDATES["draw_2cmc_2"]}
    config = DeckConfig(added_cards=("draw_2cmc_2",))
    apply_config(gf, config, candidates, swap_mode=False)
    assert len(gf.decklist) == original_count + 1
    # The synthetic card should be in the decklist
    names = [c.name for c in gf.decklist]
    assert "[Opt] 2 Mana Draw 2" in names


def test_apply_config_swap_mode():
    """Swap mode keeps deck size constant."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile")
    original_count = len(gf.decklist)
    candidates = {"draw_2cmc_2": ALL_CANDIDATES["draw_2cmc_2"]}
    config = DeckConfig(added_cards=("draw_2cmc_2",))
    apply_config(gf, config, candidates, swap_mode=True)
    assert len(gf.decklist) == original_count


def test_restore_original_decklist():
    """restore_original_decklist undoes set_lands changes."""
    deck = _simple_deck(num_lands=37)
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile")
    gf.set_lands(40)
    assert gf.land_count == 40
    gf.restore_original_decklist()
    assert gf.land_count == 37


def test_apply_config_restores_between_calls():
    """Successive apply_config calls don't accumulate cards."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile")
    candidates = {"draw_2cmc_2": ALL_CANDIDATES["draw_2cmc_2"]}

    config1 = DeckConfig(added_cards=("draw_2cmc_2",))
    apply_config(gf, config1, candidates, swap_mode=False)
    count_after_first = len(gf.decklist)

    config2 = DeckConfig(added_cards=("draw_2cmc_2",))
    apply_config(gf, config2, candidates, swap_mode=False)
    count_after_second = len(gf.decklist)

    assert count_after_first == count_after_second


def test_optimizer_runs():
    """DeckOptimizer completes and returns ranked results."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=1,
        max_ramp=1,
        land_range=1,
        optimize_for="mean_mana",
        hyperband_max_sims=50,
    )

    results = optimizer.run(final_sims=50, final_top_k=3)
    top_results = [r for r in results if r[1].get("opt_baseline_rank") is None]
    assert len(top_results) > 0
    assert len(top_results) <= 3

    # Each result is (DeckConfig, result_dict)
    for config, result_dict in results:
        assert isinstance(config, DeckConfig)
        assert "mean_mana" in result_dict
        assert "consistency" in result_dict


def test_optimizer_results_have_source_tags():
    """With include_hyperband, each result includes an opt_source tag."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=1,
        max_ramp=1,
        land_range=1,
        optimize_for="mean_mana",
        hyperband_max_sims=50,
    )

    results = optimizer.run(final_sims=50, final_top_k=5, include_hyperband=True)
    valid_sources = {"regression", "hyperband", "baseline"}
    for config, result_dict in results:
        assert "opt_source" in result_dict, f"Missing opt_source for {config}"
        assert result_dict["opt_source"] in valid_sources

    # Baseline should be among the evaluated configs (may or may not survive top-5 cut)
    sources = {r[1]["opt_source"] for r in results}
    # At minimum regression configs should be present
    assert "regression" in sources


def test_optimizer_results_no_source_tags_by_default():
    """Without include_hyperband, results have no opt_source tags."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=1,
        max_ramp=1,
        land_range=1,
        optimize_for="mean_mana",
        hyperband_max_sims=50,
    )

    results = optimizer.run(final_sims=50, final_top_k=5)
    for config, result_dict in results:
        assert "opt_source" not in result_dict


def test_optimizer_finds_multi_card_configs():
    """DeckOptimizer evaluates configs with multiple added cards."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=20, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=2,
        max_ramp=2,
        land_range=0,
        optimize_for="mean_mana",
        hyperband_max_sims=20,
    )

    results = optimizer.run(final_sims=20, final_top_k=10)
    max_cards = max(len(cfg.added_cards) for cfg, _ in results)
    assert max_cards >= 2, "Expected configs with 2+ added cards in results"


def test_optimizer_hyperband_multiple_brackets():
    """DeckOptimizer uses Hyperband with multiple brackets when hyperband_max_sims is high enough."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    # hyperband_max_sims=200 gives s_max=2 (3 brackets)
    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=1,
        max_ramp=1,
        land_range=1,
        optimize_for="mean_mana",
        hyperband_max_sims=200,
    )

    # Track progress calls
    enum_calls = []
    eval_calls = []

    results = optimizer.run(
        final_sims=50,
        final_top_k=3,
        enum_progress=lambda c, t: enum_calls.append((c, t)),
        eval_progress=lambda c, t: eval_calls.append((c, t)),
    )

    top_results = [r for r in results if r[1].get("opt_baseline_rank") is None]
    assert len(top_results) > 0
    assert len(top_results) <= 3

    # Progress should have been reported
    assert len(enum_calls) > 0
    assert len(eval_calls) > 0

    # Final enum progress should reach total
    assert enum_calls[-1][0] == enum_calls[-1][1]

    # Each result should have valid metrics
    for config, result_dict in results:
        assert isinstance(config, DeckConfig)
        assert "mean_mana" in result_dict


def test_pyodide_runner_optimization():
    """run_optimization entry point returns valid JSON."""
    from auto_goldfish.pyodide_runner import run_optimization

    deck = _simple_deck()
    config = {
        "turns": 5,
        "sims": 50,
        "seed": 42,
        "optimize_for": "mean_mana",
        "swap_mode": False,
        "enabled_candidates": ["draw_2cmc_2", "ramp_2cmc_1"],
        "max_draw_additions": 1,
        "max_ramp_additions": 1,
    }

    result_json = run_optimization(
        json.dumps(deck),
        json.dumps(config),
    )

    results = json.loads(result_json)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "opt_config" in results[0]
    assert "mean_mana" in results[0]
    # Racing optimizer does not set opt_source tags
    assert "opt_source" not in results[0]


def test_optimizer_mean_spells_cast_target():
    """DeckOptimizer can optimize for mean_spells_cast."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=1,
        max_ramp=1,
        land_range=1,
        optimize_for="mean_spells_cast",
        hyperband_max_sims=50,
    )

    results = optimizer.run(final_sims=50, final_top_k=3)
    # May have +1 baseline reference row appended
    top_results = [r for r in results if r[1].get("opt_baseline_rank") is None]
    assert len(top_results) > 0
    assert len(top_results) <= 3

    for config, result_dict in results:
        assert isinstance(config, DeckConfig)
        assert "mean_spells_cast" in result_dict
        assert result_dict["mean_spells_cast"] >= 0

    # Top results should be sorted by mean_spells_cast descending
    scores = [r[1]["mean_spells_cast"] for r in top_results]
    assert scores == sorted(scores, reverse=True)


def test_optimizer_floor_performance_target():
    """DeckOptimizer can optimize for floor_performance (bottom-25% mean mana)."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=1,
        max_ramp=1,
        land_range=1,
        optimize_for="floor_performance",
        hyperband_max_sims=50,
    )

    results = optimizer.run(final_sims=50, final_top_k=3)
    top_results = [r for r in results if r[1].get("opt_baseline_rank") is None]
    assert len(top_results) > 0
    assert len(top_results) <= 3

    for config, result_dict in results:
        assert isinstance(config, DeckConfig)
        assert "threshold_mana" in result_dict
        assert result_dict["threshold_mana"] >= 0

    # Top results should be sorted by threshold_mana descending
    scores = [r[1]["threshold_mana"] for r in top_results]
    assert scores == sorted(scores, reverse=True)


def test_optimizer_karsten_ecms_target():
    """DeckOptimizer can optimize for karsten_ecms."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items()
        if cid in ("draw_2cmc_2", "ramp_2cmc_1")
    }

    optimizer = DeckOptimizer(
        goldfisher=gf,
        candidates=enabled,
        swap_mode=False,
        max_draw=1,
        max_ramp=1,
        land_range=1,
        optimize_for="karsten_ecms",
        hyperband_max_sims=50,
    )

    results = optimizer.run(final_sims=50, final_top_k=3)
    top_results = [r for r in results if r[1].get("opt_baseline_rank") is None]
    assert len(top_results) > 0
    assert len(top_results) <= 3

    for config, result_dict in results:
        assert isinstance(config, DeckConfig)
        assert "mean_ecms" in result_dict
        assert result_dict["mean_ecms"] >= 0

    # Top results should be sorted by mean_ecms descending
    scores = [r[1]["mean_ecms"] for r in top_results]
    assert scores == sorted(scores, reverse=True)
