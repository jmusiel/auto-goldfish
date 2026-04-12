"""Integration tests for the factored optimizer.

Includes three purpose-built test decks that target specific adaptive
sampling outcomes: significant, negligible, and ambiguous effects.
"""

from auto_goldfish.engine.goldfisher import Goldfisher
from auto_goldfish.optimization.candidate_cards import ALL_CANDIDATES
from auto_goldfish.optimization.deck_config import DeckConfig
from auto_goldfish.optimization.factored_optimizer import FactoredOptimizer


# ---------------------------------------------------------------------------
# Test deck helpers
# ---------------------------------------------------------------------------


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


def _mana_starved_deck() -> list[dict]:
    """18 lands, 81 spells all CMC 5-7. Extremely under-landed.

    Targets: significant positive effect for +2 lands.
    With only 18 lands and average CMC ~6, almost every game is mana-screwed.
    Adding +2 lands has a huge, consistent per-game impact because it
    meaningfully increases the chance of casting even one expensive spell.
    """
    deck = []
    deck.append({
        "name": "Test Commander",
        "cmc": 7,
        "cost": "{5}{U}{B}",
        "text": "",
        "types": ["Creature"],
        "commander": True,
    })
    for i in range(18):
        deck.append({
            "name": f"Island {i}",
            "cmc": 0,
            "cost": "",
            "text": "",
            "types": ["Land"],
            "commander": False,
        })
    for i in range(81):
        cmc = (i % 3) + 5  # CMC 5, 6, or 7
        deck.append({
            "name": f"Creature {i}",
            "cmc": cmc,
            "cost": f"{{{cmc}}}",
            "text": "",
            "types": ["Creature"],
            "commander": False,
        })
    return deck


def _overlanded_cantrip_deck() -> list[dict]:
    """45 lands, 54 spells all CMC 1. Severely over-landed.

    Targets: significant negative effect for adding lands, significant
    positive effect for cutting lands.
    With 45 lands and all 1-CMC spells, the deck draws too many lands
    and not enough spells. Cutting a land is strictly better (trade an
    unneeded land for a castable spell). Adding draw/ramp is unhelpful
    since spells cost 1 and mana is abundant.
    """
    deck = []
    deck.append({
        "name": "Test Commander",
        "cmc": 1,
        "cost": "{U}",
        "text": "",
        "types": ["Creature"],
        "commander": True,
    })
    for i in range(45):
        deck.append({
            "name": f"Island {i}",
            "cmc": 0,
            "cost": "",
            "text": "",
            "types": ["Land"],
            "commander": False,
        })
    for i in range(54):
        deck.append({
            "name": f"Creature {i}",
            "cmc": 1,
            "cost": "{1}",
            "text": "",
            "types": ["Creature"],
            "commander": False,
        })
    return deck


def _equilibrium_deck() -> list[dict]:
    """37 lands, 62 spells all CMC 2. At equilibrium, changes are negligible.

    Targets: negligible effect for all changes.
    With 37 lands and uniform CMC 2, the deck is already near-optimal.
    You always have enough mana to cast your 2-drops, and +/-1 land
    doesn't meaningfully change outcomes. Adding draw/ramp (CMC 2+)
    doesn't help since you're already casting everything you draw.
    """
    deck = []
    deck.append({
        "name": "Test Commander",
        "cmc": 2,
        "cost": "{1}{U}",
        "text": "",
        "types": ["Creature"],
        "commander": True,
    })
    for i in range(37):
        deck.append({
            "name": f"Island {i}",
            "cmc": 0,
            "cost": "",
            "text": "",
            "types": ["Land"],
            "commander": False,
        })
    for i in range(62):
        deck.append({
            "name": f"Creature {i}",
            "cmc": 2,
            "cost": "{2}",
            "text": "",
            "types": ["Creature"],
            "commander": False,
        })
    return deck


def _balanced_deck() -> list[dict]:
    """30 lands, 69 spells CMC 3-5. Moderately undercooked for ambiguous effects.

    Targets: ambiguous effect that requires more samples.
    With 30 lands and avg CMC ~4, +1 land has a moderate benefit in
    games where you're mana-screwed but is wasted in games where draw
    order was favorable. The effect exists but paired-difference
    variance is high enough to need >200 games for confidence.
    """
    deck = []
    deck.append({
        "name": "Test Commander",
        "cmc": 4,
        "cost": "{2}{U}{B}",
        "text": "",
        "types": ["Creature"],
        "commander": True,
    })
    for i in range(30):
        deck.append({
            "name": f"Island {i}",
            "cmc": 0,
            "cost": "",
            "text": "",
            "types": ["Land"],
            "commander": False,
        })
    for i in range(69):
        cmc = (i % 3) + 3  # CMC 3, 4, or 5
        deck.append({
            "name": f"Creature {i}",
            "cmc": cmc,
            "cost": f"{{{cmc}}}",
            "text": "",
            "types": ["Creature"],
            "commander": False,
        })
    return deck


# ---------------------------------------------------------------------------
# Basic integration tests
# ---------------------------------------------------------------------------


class TestFactoredOptimizerIntegration:
    """Basic integration tests with the simple deck."""

    def test_runs_and_returns_results(self):
        """FactoredOptimizer completes and returns ranked results."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
        enabled = {
            cid: c for cid, c in ALL_CANDIDATES.items()
            if cid in ("draw_2cmc_2", "ramp_2cmc_1")
        }

        optimizer = FactoredOptimizer(
            goldfisher=gf,
            candidates=enabled,
            swap_mode=False,
            max_draw=1,
            max_ramp=1,
            land_range=1,
            optimize_for="mean_mana",
            base_games=50,
            max_games=100,
        )

        results = optimizer.run(final_sims=50, final_top_k=3)
        assert len(results) > 0
        assert len(results) <= 4  # top_k + possible baseline

        for config, result_dict in results:
            assert isinstance(config, DeckConfig)
            assert "mean_mana" in result_dict
            assert "consistency" in result_dict

    def test_baseline_always_in_results(self):
        """Baseline config appears in results."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=1,
            optimize_for="mean_mana", base_games=50, max_games=100,
        )

        results = optimizer.run(final_sims=50, final_top_k=3)
        configs = [cfg for cfg, _ in results]
        assert DeckConfig() in configs

    def test_feature_analysis_attached(self):
        """First result has feature_analysis key."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=1,
            optimize_for="mean_mana", base_games=50, max_games=100,
        )

        results = optimizer.run(final_sims=50, final_top_k=3)
        assert "feature_analysis" in results[0][1]
        fa = results[0][1]["feature_analysis"]
        assert "recommendations" in fa
        assert "marginal_impact" in fa

    def test_marginal_results_populated(self):
        """After run, marginal_results contains entries for each marginal config."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")
        enabled = {
            "draw_2cmc_2": ALL_CANDIDATES["draw_2cmc_2"],
            "ramp_2cmc_1": ALL_CANDIDATES["ramp_2cmc_1"],
        }

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates=enabled,
            max_draw=1, max_ramp=1, land_range=1,
            optimize_for="mean_mana", base_games=50, max_games=100,
        )

        optimizer.run(final_sims=50, final_top_k=3)

        # 2 land deltas + 1 draw + 1 ramp = 4 marginals
        assert len(optimizer.marginal_results) == 4
        dims = {mr.dimension for mr in optimizer.marginal_results}
        assert dims == {"land", "draw", "ramp"}

    def test_floor_performance_target(self):
        """FactoredOptimizer can optimize for floor_performance."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=1,
            optimize_for="floor_performance",
            base_games=50, max_games=100,
        )

        results = optimizer.run(final_sims=50, final_top_k=3)
        assert len(results) > 0
        for _, result_dict in results:
            assert "threshold_mana" in result_dict

    def test_progress_callbacks(self):
        """Progress callbacks are called during optimization."""
        deck = _simple_deck()
        gf = Goldfisher(deck, turns=5, sims=50, seed=42, record_results="quartile")

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=1,
            optimize_for="mean_mana", base_games=50, max_games=100,
        )

        enum_calls = []
        eval_calls = []

        optimizer.run(
            final_sims=50, final_top_k=3,
            enum_progress=lambda c, t: enum_calls.append((c, t)),
            eval_progress=lambda c, t: eval_calls.append((c, t)),
        )

        assert len(enum_calls) > 0
        assert len(eval_calls) > 0


# ---------------------------------------------------------------------------
# Adaptive sampling case tests
# ---------------------------------------------------------------------------


class TestAdaptiveSamplingCases:
    """Integration tests using purpose-built decks for each sampling outcome."""

    def test_significant_effect_stops_early(self):
        """Mana-starved deck: +2 lands is obviously good, detected in first batch."""
        deck = _mana_starved_deck()
        gf = Goldfisher(
            deck, turns=8, sims=50, seed=42, record_results="quartile",
        )

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=2,
            optimize_for="mean_mana",
            base_games=200, max_games=800,
        )

        optimizer.run(final_sims=50, final_top_k=3)

        land_marginals = [
            m for m in optimizer.marginal_results if m.dimension == "land"
        ]
        plus_2 = next(
            (m for m in land_marginals if m.config.land_delta == 2), None
        )
        assert plus_2 is not None, "Expected +2 land marginal"
        assert plus_2.significant is True, (
            f"+2 land should be significant (effect={plus_2.effect_size:.3f},"
            f" se={plus_2.se:.3f}, p={plus_2.p_value:.4f})"
        )
        assert plus_2.n_games == 200, (
            f"Expected early stop at 200 games, got {plus_2.n_games}"
        )
        assert plus_2.effect_size > 0, (
            f"Expected positive effect, got {plus_2.effect_size:.3f}"
        )

    def test_overlanded_cut_lands_is_positive(self):
        """Over-landed cantrip deck: cutting lands should be clearly beneficial."""
        deck = _overlanded_cantrip_deck()
        gf = Goldfisher(
            deck, turns=8, sims=50, seed=42, record_results="quartile",
        )

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=2,
            optimize_for="mean_mana",
            base_games=200, max_games=800,
        )

        optimizer.run(final_sims=50, final_top_k=3)

        land_marginals = [
            m for m in optimizer.marginal_results if m.dimension == "land"
        ]
        # Cutting lands should show a positive effect (fewer dead draws)
        minus_2 = next(
            (m for m in land_marginals if m.config.land_delta == -2), None
        )
        assert minus_2 is not None, "Expected -2 land marginal"
        assert minus_2.effect_size > 0, (
            f"Cutting 2 lands should improve mana spent, "
            f"got effect={minus_2.effect_size:.3f}"
        )

    def test_negligible_effect_detected(self):
        """Equilibrium deck: +/-1 land on a well-tuned deck is negligible."""
        deck = _equilibrium_deck()
        gf = Goldfisher(
            deck, turns=8, sims=50, seed=42, record_results="quartile",
        )

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=1,
            optimize_for="mean_mana",
            base_games=200, max_games=800,
        )

        optimizer.run(final_sims=50, final_top_k=3)

        for mr in optimizer.marginal_results:
            # Effects should be tiny — either negligible or not significant
            assert abs(mr.effect_size) < 1.0, (
                f"{mr.config.describe()} has unexpectedly large effect"
                f" {mr.effect_size:.3f}"
            )

    def test_ambiguous_effect_needs_more_samples(self):
        """Balanced deck: +/-1 land is ambiguous, should need more than base_games."""
        deck = _balanced_deck()
        gf = Goldfisher(
            deck, turns=8, sims=50, seed=42, record_results="quartile",
        )

        optimizer = FactoredOptimizer(
            goldfisher=gf, candidates={},
            max_draw=0, max_ramp=0, land_range=1,
            optimize_for="mean_mana",
            base_games=200, max_games=800,
        )

        optimizer.run(final_sims=50, final_top_k=3)

        land_marginals = [
            m for m in optimizer.marginal_results if m.dimension == "land"
        ]
        max_games_used = max(m.n_games for m in land_marginals)
        # At least one land marginal should have needed re-evaluation.
        # If both stop at 200, the deck may need tuning — see note below.
        assert max_games_used > 200, (
            f"Expected at least one marginal to need >200 games, "
            f"but max was {max_games_used}. "
            f"Effects: {[(m.config.land_delta, m.effect_size, m.se) for m in land_marginals]}"
        )
