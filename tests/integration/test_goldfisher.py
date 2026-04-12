"""Integration tests for the goldfisher engine."""

import random

import pytest

from auto_goldfish.engine.goldfisher import Goldfisher, SimulationResult
from auto_goldfish.engine.mana_efficiency import VALID_MANA_EFFICIENCY_MODES
from auto_goldfish.engine.spell_priority import VALID_SPELL_PRIORITIES


def _simple_deck(num_lands: int = 37, num_spells: int = 62) -> list[dict]:
    """Build a simple deck with lands and vanilla creatures."""
    deck = []
    # Commander
    deck.append({
        "name": "Test Commander",
        "cmc": 4,
        "cost": "{2}{U}{B}",
        "text": "",
        "types": ["Creature"],
        "commander": True,
    })
    # Lands
    for i in range(num_lands):
        deck.append({
            "name": f"Island {i}",
            "cmc": 0,
            "cost": "",
            "text": "",
            "types": ["Land"],
            "commander": False,
        })
    # Creatures at various costs
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


def test_simulate_completes():
    """Simulation runs without errors and returns a SimulationResult."""
    random.seed(42)
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile")
    result = gf.simulate()
    assert isinstance(result, SimulationResult)
    assert result.mean_mana > 0
    assert result.mean_lands > 0


def test_simulate_with_effects_cards():
    """Simulation works with cards that have registered effects."""
    random.seed(42)
    deck = _simple_deck(num_lands=35, num_spells=60)
    # Add Sol Ring and Phyrexian Arena
    deck.append({
        "name": "Sol Ring",
        "cmc": 1,
        "cost": "{1}",
        "text": "",
        "types": ["Artifact"],
        "commander": False,
    })
    deck.append({
        "name": "Phyrexian Arena",
        "cmc": 3,
        "cost": "{1}{B}{B}",
        "text": "",
        "types": ["Enchantment"],
        "commander": False,
    })
    gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile")
    result = gf.simulate()
    assert isinstance(result, SimulationResult)
    assert result.mean_mana > 0


def test_as_row():
    result = SimulationResult(land_count=37, mean_mana=15.5, consistency=0.9)
    row = result.as_row()
    assert row[0] == 37
    assert "15.50" in row[1]  # formatted with CI margin
    assert "0.9000" in row[2]  # consistency with CI margin
    assert len(row) == 14


def test_set_lands():
    """set_lands adjusts land count correctly."""
    deck = _simple_deck(num_lands=35, num_spells=64)
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile")
    original = gf.land_count
    gf.set_lands(original + 2)
    assert gf.land_count == original + 2


def test_simulate_single_game_ecms():
    """simulate_single_game with ecms=True returns compounded mana."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=10, record_results="quartile", seed=42)
    normal = gf.simulate_single_game(42)
    ecms = gf.simulate_single_game(42, ecms=True)
    # ECMS compounds mana forward, so it should be >= normal primary mana
    assert ecms >= normal


def test_seed_reproducibility():
    """Same seed produces identical results."""
    deck = _simple_deck()
    gf1 = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r1 = gf1.simulate()

    gf2 = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r2 = gf2.simulate()

    assert r1.mean_mana == r2.mean_mana
    assert r1.mean_lands == r2.mean_lands
    assert r1.mean_mulls == r2.mean_mulls
    assert r1.consistency == r2.consistency


def test_different_seeds_differ():
    """Different seeds produce different results."""
    deck = _simple_deck()
    gf1 = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r1 = gf1.simulate()

    gf2 = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=99)
    r2 = gf2.simulate()

    # Very unlikely to be exactly equal with different seeds
    assert r1.mean_mana != r2.mean_mana


def test_confidence_intervals():
    """Confidence intervals are computed and make sense."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    result = gf.simulate()

    # CI should bracket the mean
    assert result.ci_mean_mana[0] <= result.mean_mana <= result.ci_mean_mana[1]
    assert result.ci_mean_bad_turns[0] <= result.mean_bad_turns <= result.ci_mean_bad_turns[1]
    # Consistency CI should be within reasonable bounds
    assert result.ci_consistency[0] <= result.consistency <= result.ci_consistency[1]
    # CI width should be positive (non-degenerate)
    assert result.ci_mean_mana[1] > result.ci_mean_mana[0]


def test_parallel_simulation():
    """Parallel simulation produces results consistent with sequential."""
    deck = _simple_deck()

    # Sequential with seed
    gf_seq = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r_seq = gf_seq.simulate()

    # Parallel with same seed (2 workers)
    gf_par = Goldfisher(deck, turns=5, sims=200, record_results="quartile",
                        seed=42, workers=2)
    r_par = gf_par.simulate()

    # With CRN (same seed), parallel and sequential should get the same raw stats
    # because each game index gets the same seed regardless of batching
    assert r_seq.mean_mana == r_par.mean_mana
    assert r_seq.mean_lands == r_par.mean_lands
    assert r_seq.mean_mulls == r_par.mean_mulls


def test_crn_across_land_counts():
    """CRN: same seed across land counts uses same random draws per game index."""
    deck = _simple_deck(num_lands=35, num_spells=64)
    seed = 123

    gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile", seed=seed)
    gf.set_lands(36)
    r36 = gf.simulate()

    gf.set_lands(38)
    r38 = gf.simulate()

    # With CRN, results should differ (different land counts) but be comparable.
    # The key property: both ran with the same per-game seeds.
    # We verify this indirectly: re-running 36 with same seed gives identical result.
    gf.set_lands(36)
    r36_again = gf.simulate()
    assert r36.mean_mana == r36_again.mean_mana


# ---------------------------------------------------------------------------
# Spell priority mode tests
# ---------------------------------------------------------------------------

def _ramp_deck() -> list[dict]:
    """Deck with a mix of ramp, draw, and value cards for priority testing."""
    deck = []
    deck.append({
        "name": "Test Commander",
        "cmc": 4, "cost": "{2}{U}{B}", "text": "",
        "types": ["Creature"], "commander": True,
    })
    for i in range(37):
        deck.append({
            "name": f"Island {i}", "cmc": 0, "cost": "", "text": "",
            "types": ["Land"], "commander": False,
        })
    # 20 value creatures
    for i in range(20):
        cmc = (i % 4) + 2
        deck.append({
            "name": f"Creature {i}", "cmc": cmc, "cost": f"{{{cmc}}}",
            "text": "", "types": ["Creature"], "commander": False,
        })
    # 20 more creatures to fill the deck
    for i in range(20, 42):
        cmc = (i % 5) + 1
        deck.append({
            "name": f"Creature {i}", "cmc": cmc, "cost": f"{{{cmc}}}",
            "text": "", "types": ["Creature"], "commander": False,
        })
    return deck


@pytest.mark.parametrize("mode", VALID_SPELL_PRIORITIES)
def test_spell_priority_modes_complete(mode):
    """Each spell priority mode runs a simulation without error."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile",
                    seed=42, spell_priority=mode)
    result = gf.simulate()
    assert isinstance(result, SimulationResult)
    assert result.mean_mana > 0


def test_spell_priority_default_matches_original():
    """priority_then_cmc mode produces identical results to unset (backward compat)."""
    deck = _simple_deck()
    gf_default = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r_default = gf_default.simulate()

    gf_explicit = Goldfisher(deck, turns=5, sims=200, record_results="quartile",
                             seed=42, spell_priority="priority_then_cmc")
    r_explicit = gf_explicit.simulate()

    assert r_default.mean_mana == r_explicit.mean_mana
    assert r_default.mean_lands == r_explicit.mean_lands
    assert r_default.consistency == r_explicit.consistency


def test_spell_priority_modes_differ():
    """Different priority modes produce different results (at least some)."""
    deck = _ramp_deck()
    results = {}
    for mode in VALID_SPELL_PRIORITIES:
        gf = Goldfisher(deck, turns=8, sims=500, record_results="quartile",
                        seed=42, spell_priority=mode)
        r = gf.simulate()
        results[mode] = r.mean_mana

    # Not all modes should produce identical results
    unique_values = set(results.values())
    assert len(unique_values) > 1, (
        f"All spell priority modes produced identical mean_mana: {results}"
    )


def test_invalid_spell_priority_raises():
    """Invalid spell_priority value raises ValueError."""
    deck = _simple_deck()
    with pytest.raises(ValueError, match="Invalid spell_priority"):
        Goldfisher(deck, turns=5, sims=10, spell_priority="bogus")


# ---------------------------------------------------------------------------
# Mana efficiency mode tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", VALID_MANA_EFFICIENCY_MODES)
def test_mana_efficiency_modes_complete(mode):
    """Each mana efficiency mode runs a simulation without error."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile",
                    seed=42, mana_efficiency=mode)
    result = gf.simulate()
    assert isinstance(result, SimulationResult)
    assert result.mean_mana > 0


def test_mana_efficiency_default_matches_original():
    """greedy mode produces identical results to unset (backward compat)."""
    deck = _simple_deck()
    gf_default = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r_default = gf_default.simulate()

    gf_explicit = Goldfisher(deck, turns=5, sims=200, record_results="quartile",
                             seed=42, mana_efficiency="greedy")
    r_explicit = gf_explicit.simulate()

    assert r_default.mean_mana == r_explicit.mean_mana
    assert r_default.mean_lands == r_explicit.mean_lands


def test_mana_efficient_spends_at_least_as_much():
    """mana_efficient mode should spend >= mana per game vs greedy."""
    deck = _simple_deck()
    gf_greedy = Goldfisher(deck, turns=8, sims=500, record_results="quartile",
                           seed=42, mana_efficiency="greedy")
    r_greedy = gf_greedy.simulate()

    gf_efficient = Goldfisher(deck, turns=8, sims=500, record_results="quartile",
                              seed=42, mana_efficiency="mana_efficient")
    r_efficient = gf_efficient.simulate()

    # mana_efficient should spend at least as much total mana
    assert r_efficient.mean_mana >= r_greedy.mean_mana - 0.5  # small tolerance


def test_invalid_mana_efficiency_raises():
    """Invalid mana_efficiency value raises ValueError."""
    deck = _simple_deck()
    with pytest.raises(ValueError, match="Invalid mana_efficiency"):
        Goldfisher(deck, turns=5, sims=10, mana_efficiency="bogus")


# ---------------------------------------------------------------------------
# Ramp cutoff turn tests
# ---------------------------------------------------------------------------

def test_ramp_cutoff_completes():
    """Simulation with ramp cutoff runs without error."""
    deck = _ramp_deck()
    gf = Goldfisher(deck, turns=8, sims=50, record_results="quartile",
                    seed=42, ramp_cutoff_turn=4)
    result = gf.simulate()
    assert isinstance(result, SimulationResult)
    assert result.mean_mana > 0


def test_ramp_cutoff_zero_matches_default():
    """ramp_cutoff_turn=0 (disabled) produces identical results to default."""
    deck = _simple_deck()
    gf1 = Goldfisher(deck, turns=5, sims=200, record_results="quartile",
                     seed=42, ramp_cutoff_turn=0)
    r1 = gf1.simulate()

    gf2 = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r2 = gf2.simulate()

    assert r1.mean_mana == r2.mean_mana


def test_invalid_ramp_cutoff_raises():
    deck = _simple_deck()
    with pytest.raises(ValueError, match="ramp_cutoff_turn"):
        Goldfisher(deck, turns=5, sims=10, ramp_cutoff_turn=-1)


# ---------------------------------------------------------------------------
# Min cost floor tests
# ---------------------------------------------------------------------------

def test_min_cost_floor_zero_completes():
    """Simulation with min_cost_floor=0 runs without error."""
    deck = _simple_deck()
    gf = Goldfisher(deck, turns=5, sims=50, record_results="quartile",
                    seed=42, min_cost_floor=0)
    result = gf.simulate()
    assert isinstance(result, SimulationResult)
    assert result.mean_mana > 0


def test_min_cost_floor_one_matches_default():
    """min_cost_floor=1 produces identical results to default."""
    deck = _simple_deck()
    gf1 = Goldfisher(deck, turns=5, sims=200, record_results="quartile",
                     seed=42, min_cost_floor=1)
    r1 = gf1.simulate()

    gf2 = Goldfisher(deck, turns=5, sims=200, record_results="quartile", seed=42)
    r2 = gf2.simulate()

    assert r1.mean_mana == r2.mean_mana


def test_invalid_min_cost_floor_raises():
    deck = _simple_deck()
    with pytest.raises(ValueError, match="min_cost_floor"):
        Goldfisher(deck, turns=5, sims=10, min_cost_floor=2)
