"""Tests for src/auto_goldfish/optimization/curve_value.py."""

from __future__ import annotations

import math

import pytest

from auto_goldfish.optimization.curve_value import (
    BASELINE_CMC,
    CommanderSpec,
    RampCardSpec,
    aggregate_deck_irr,
    cards_for_land_drops,
    cards_seen_by_turn,
    cards_to_spend,
    classify_for_curve_value,
    compute_curve_value,
    compute_implied_draw,
    compute_implied_spell_value,
    implied_power_multipliers,
    land_mana_over_T,
    ramp_contribution,
    ramp_irr,
    solve_irr,
)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def test_cards_seen_by_turn_on_the_play():
    assert cards_seen_by_turn(1) == 7    # opening hand only
    assert cards_seen_by_turn(2) == 8    # 7 + 1 draw
    assert cards_seen_by_turn(8) == 14


# ---------------------------------------------------------------------------
# IRR
# ---------------------------------------------------------------------------

def test_ramp_irr_2cmc_1tap_at_T8_around_45pct():
    """A 2-cmc 1-tap rock cast turn 2 in an 8-turn game should yield ~45% per turn."""
    rate = ramp_irr(2, 1.0, 8)
    assert 0.40 < rate < 0.50


def test_ramp_irr_sol_ring_high_rate():
    """Sol Ring (1-cmc, 2 mana/turn) cast turn 1 should be very profitable."""
    rate = ramp_irr(1, 2.0, 8)
    # The bisection caps near IRR_HI; just verify it's clearly above 100%/turn.
    assert rate > 1.0


def test_ramp_irr_4cmc_1tap_at_T8_breakeven():
    """A 4-cmc 1-tap rock cast turn 4 in T=8 returns 4 mana for 4 cost: ~0%."""
    rate = ramp_irr(4, 1.0, 8)
    assert -0.05 < rate < 0.05


def test_ramp_irr_5cmc_1tap_at_T8_negative():
    """A 5-cmc 1-tap rock cast turn 5 in T=8 returns 3 mana for 5 cost: negative IRR."""
    rate = ramp_irr(5, 1.0, 8)
    assert rate < 0


def test_ramp_irr_uncastable_returns_nan():
    """Ramp cmc > T can't be cast; IRR is undefined."""
    rate = ramp_irr(10, 1.0, 8)
    assert math.isnan(rate)


def test_solve_irr_zero_npv():
    """Solver finds the rate where NPV = 0 for a known cash flow."""
    # -100 at t=0, +110 at t=1 -> IRR = 10%.
    rate = solve_irr({0: -100.0, 1: 110.0})
    assert abs(rate - 0.10) < 1e-3


# ---------------------------------------------------------------------------
# Aggregate / multipliers
# ---------------------------------------------------------------------------

def test_aggregate_irr_median_uniform_ramp():
    """Three identical 2-cmc rocks: median IRR = single-card IRR."""
    ramp = [RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(3)]
    agg = aggregate_deck_irr(ramp, T=8)
    expected = ramp_irr(2, 1.0, 8)
    assert abs(agg["median_irr"] - expected) < 1e-3
    assert agg["n_valid"] == 3


def test_aggregate_irr_no_ramp_returns_nan():
    agg = aggregate_deck_irr([], T=8)
    assert math.isnan(agg["median_irr"])
    assert agg["n_valid"] == 0


def test_implied_power_multipliers_baseline_is_1():
    delta = 0.7
    mults = implied_power_multipliers(delta, baseline_cmc=2, max_cmc=6)
    assert abs(mults[2] - 1.0) < 1e-9


def test_implied_power_multipliers_high_irr_inflates_high_cmc():
    """Impatient deck (low delta) requires high-CMC slots to be more powerful."""
    impatient = implied_power_multipliers(delta=0.5, baseline_cmc=2, max_cmc=6)
    patient = implied_power_multipliers(delta=0.9, baseline_cmc=2, max_cmc=6)
    assert impatient[6] > patient[6]


def test_implied_power_multipliers_at_delta_1():
    """No-ramp baseline (delta=1.0): multiplier purely reflects per-slot mana efficiency."""
    mults = implied_power_multipliers(delta=1.0, baseline_cmc=2, max_cmc=6)
    # multiplier(c) = (2 * 1) / (c * 1) = 2 / c
    assert abs(mults[1] - 2.0) < 1e-9
    assert abs(mults[2] - 1.0) < 1e-9
    assert abs(mults[4] - 0.5) < 1e-9
    assert abs(mults[6] - (2.0 / 6.0)) < 1e-9


# ---------------------------------------------------------------------------
# Implied Draw
# ---------------------------------------------------------------------------

def test_implied_draw_no_ramp_no_commander_has_land_bottleneck_deficit():
    """A 37-land deck with no ramp / no commanders still needs to draw more
    than its natural 14 cards to reliably hit 8 land drops by turn 8 -- 8
    lands / (37/99) = 21.4 cards needed. The land bottleneck creates a real
    deficit even when the value bottleneck is small.
    """
    res = compute_implied_draw(
        L=37, V=62, V_avg_cmc=3.0,
        ramp_specs=[], commanders=[],
        D=99, T=8,
    )
    assert res.ramp_excess == 0.0
    assert res.commander_mana == 0.0
    # Land bottleneck at turn 8 = 8 * 99 / 37 ≈ 21.4
    expected_land_n_max = 8 * 99 / 37
    assert abs(res.N_max - expected_land_n_max) < 1e-6
    # Deficit ≈ 21.4 - 14 = 7.4
    assert 6.5 < res.deficit_max < 8.5
    # Land bottleneck dominates value bottleneck across all turns for this deck
    for t in range(8):
        assert res.per_turn_lands_required[t] >= res.per_turn_value_required[t]


def test_implied_draw_lots_of_ramp_creates_deficit():
    """Adding ramp generates excess mana, pushing N_max well above natural draw."""
    ramp = [RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(20)]
    res = compute_implied_draw(
        L=37, V=42, V_avg_cmc=3.5,
        ramp_specs=ramp, commanders=[],
        D=99, T=8,
    )
    assert res.ramp_excess > 0
    assert res.deficit_max > 5.0


def test_implied_draw_commander_reduces_deficit():
    """Commanders are guaranteed castable -> they consume mana that no longer needs drawing."""
    cmds = [CommanderSpec(name="Cmd", cmc=4)]
    res_with = compute_implied_draw(
        L=37, V=62, V_avg_cmc=3.0,
        ramp_specs=[], commanders=cmds,
        D=99, T=8,
    )
    res_without = compute_implied_draw(
        L=37, V=62, V_avg_cmc=3.0,
        ramp_specs=[], commanders=[],
        D=99, T=8,
    )
    assert res_with.commander_mana == 4.0
    assert res_with.value_mana < res_without.value_mana
    assert res_with.N_max <= res_without.N_max


def test_cards_for_land_drops():
    """Land bottleneck: cards needed to draw t lands from L-of-D deck."""
    # 1 land needed, 37/98 land density -> need 98/37 ≈ 2.65 cards
    assert abs(cards_for_land_drops(1, L=37, D=98) - 98 / 37) < 1e-6
    # 8 land drops by turn 8 -> 8 * 98/37 ≈ 21.2
    assert abs(cards_for_land_drops(8, L=37, D=98) - 8 * 98 / 37) < 1e-6
    # Edge cases
    assert cards_for_land_drops(0, 37, 98) == 0.0
    assert cards_for_land_drops(8, L=0, D=98) == 0.0
    assert cards_for_land_drops(8, L=37, D=0) == 0.0


def test_implied_draw_land_bottleneck_dominates_early_turns():
    """Without ramp, the land bottleneck is the active constraint at turn 1
    (and most early turns). Value mana is small, but you still need cards
    drawn to find the lands you'll play."""
    res = compute_implied_draw(
        L=37, V=62, V_avg_cmc=3.0,
        ramp_specs=[], commanders=[],
        D=99, T=8,
    )
    # Turn 1: 1 land needed -> ~99/37 ≈ 2.68; value bottleneck ≈ 0.5 (~1 mana / 1.86).
    # Land must dominate.
    assert res.per_turn_lands_required[0] > res.per_turn_value_required[0]
    assert res.per_turn_required[0] == res.per_turn_lands_required[0]
    # And the curve never starts below the natural opener of 7 in this deck:
    # natural[0] = 7, lands_required[0] ≈ 2.68 < 7, so deficit at turn 1 = 0.
    assert res.per_turn_required[0] < res.per_turn_natural[0]


def test_implied_draw_value_bottleneck_can_dominate_late():
    """A deck with lots of ramp generates excess value mana that can push the
    value bottleneck above the land bottleneck in late turns."""
    ramp = [RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(20)]
    res = compute_implied_draw(
        L=37, V=42, V_avg_cmc=3.0,
        ramp_specs=ramp, commanders=[],
        D=99, T=8,
    )
    # By turn 8, 20 ramps + low V_avg should make value bottleneck dominant.
    assert res.per_turn_value_required[-1] >= res.per_turn_lands_required[-1]
    assert res.per_turn_required[-1] == res.per_turn_value_required[-1]


def test_implied_draw_per_turn_required_is_max_of_both():
    """``per_turn_required[t]`` must equal max of the two component arrays."""
    res = compute_implied_draw(
        L=37, V=42, V_avg_cmc=3.0,
        ramp_specs=[RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(10)],
        commanders=[],
        D=99, T=8,
    )
    for t in range(8):
        assert res.per_turn_required[t] == max(
            res.per_turn_lands_required[t],
            res.per_turn_value_required[t],
        )


def test_implied_draw_more_ramp_inflates_value_bottleneck():
    """Running more ramp shrinks V (slot-for-slot tradeoff against value
    pool), which raises n_for_value because the same mana must be sourced
    from a thinner value pool."""
    base = compute_implied_draw(
        L=37, V=52, V_avg_cmc=3.0,
        ramp_specs=[RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(10)],
        commanders=[],
        D=99, T=8,
    )
    more_ramp = compute_implied_draw(
        L=37, V=42, V_avg_cmc=3.0,
        ramp_specs=[RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(20)],
        commanders=[],
        D=99, T=8,
    )
    # More ramp = bigger value-bottleneck demand at turn 8.
    assert more_ramp.per_turn_value_required[-1] > base.per_turn_value_required[-1]


def test_implied_draw_n_max_matches_last_turn_required():
    """N_max is the last-turn cumulative requirement under the new model."""
    res = compute_implied_draw(
        L=37, V=62, V_avg_cmc=3.0,
        ramp_specs=[],
        commanders=[],
        D=99, T=8,
    )
    assert abs(res.N_max - res.per_turn_required[-1]) < 1e-9


def test_implied_draw_per_turn_arrays_have_T_entries():
    res = compute_implied_draw(
        L=37, V=62, V_avg_cmc=3.0,
        ramp_specs=[], commanders=[],
        D=99, T=8,
    )
    assert len(res.per_turn_required) == 8
    assert len(res.per_turn_natural) == 8
    assert res.per_turn_natural == [7, 8, 9, 10, 11, 12, 13, 14]


def test_implied_draw_actual_deficit_uses_mc_input():
    res = compute_implied_draw(
        L=37, V=42, V_avg_cmc=3.0,
        ramp_specs=[RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(10)],
        commanders=[],
        D=99, T=8,
        actual_total_draws=15.0,
    )
    assert res.actual_total_draws == 15.0
    assert res.actual_deficit == max(0.0, res.N_max - 15.0)


# ---------------------------------------------------------------------------
# Implied Spell Value
# ---------------------------------------------------------------------------

def test_implied_spell_value_no_ramp_falls_back_to_delta_1():
    res = compute_implied_spell_value(ramp_specs=[], T=8, max_cmc=6)
    assert res.no_ramp is True
    assert res.delta == 1.0
    # at delta=1, mults follow 2/c
    assert abs(res.power_multipliers[2] - 1.0) < 1e-9
    assert abs(res.power_multipliers[4] - 0.5) < 1e-9


def test_implied_spell_value_uniform_ramp_yields_consistent_delta():
    ramp = [RampCardSpec(name=f"r{i}", cmc=2, mana_per_turn=1.0) for i in range(5)]
    res = compute_implied_spell_value(ramp_specs=ramp, T=8, max_cmc=6)
    # Median IRR = 45%, delta ~ 0.69
    assert 0.65 < res.delta < 0.75


def test_implied_spell_value_per_card_irrs_present():
    ramp = [
        RampCardSpec(name="Rock", cmc=2, mana_per_turn=1.0),
        RampCardSpec(name="Sol Ring", cmc=1, mana_per_turn=2.0),
    ]
    res = compute_implied_spell_value(ramp_specs=ramp, T=8, max_cmc=6)
    assert len(res.per_card_irrs) == 2
    names = {p.name for p in res.per_card_irrs}
    assert names == {"Rock", "Sol Ring"}


# ---------------------------------------------------------------------------
# Classification + end-to-end
# ---------------------------------------------------------------------------

def _land(name: str = "Plains", qty: int = 1) -> dict:
    return {"name": name, "cmc": 0, "types": ["Land"], "quantity": qty}


def _spell(name: str, cmc: int, qty: int = 1, commander: bool = False, types=None) -> dict:
    return {
        "name": name, "cmc": cmc, "quantity": qty,
        "types": types or ["Creature"],
        "commander": commander,
    }


def test_classify_excludes_commanders_from_deck_size():
    deck = [
        _land("Plains", qty=37),
        _spell("Cmdr", cmc=4, qty=1, commander=True),
        _spell("Filler", cmc=3, qty=62),
    ]
    cls = classify_for_curve_value(deck, registry=None, overrides=None)
    assert cls["D"] == 99
    assert len(cls["commanders"]) == 1
    assert cls["commanders"][0].cmc == 4
    assert cls["L"] == 37
    assert cls["V"] == 62


def test_classify_promotes_only_mana_producing_ramp():
    """With registry=None, no card has mana_per_turn > 0 so no ramp is promoted."""
    deck = [
        _land(qty=37),
        _spell("Filler", cmc=2, qty=62),
    ]
    cls = classify_for_curve_value(deck, registry=None, overrides=None)
    assert cls["ramp_specs"] == []


def test_compute_curve_value_end_to_end_no_registry():
    """Smoke test: end-to-end runs on a synthetic deck without a registry."""
    deck = (
        [_land(qty=37)]
        + [_spell("Cmdr", cmc=4, qty=1, commander=True)]
        + [_spell(f"v{i}", cmc=3, qty=1) for i in range(62)]
    )
    res = compute_curve_value(deck, registry=None, overrides=None, turns=8)
    assert res.turns == 8
    assert res.deck_size_effective == 99
    assert res.implied_draw.commander_mana == 4.0
    assert res.implied_spell_value.no_ramp is True
    assert res.implied_spell_value.delta == 1.0


def test_compute_curve_value_uses_actual_draws_when_provided():
    deck = (
        [_land(qty=37)]
        + [_spell(f"v{i}", cmc=3, qty=1) for i in range(63)]
    )
    res = compute_curve_value(
        deck, registry=None, overrides=None, turns=8,
        actual_total_draws=20.0,
        actual_per_turn_cumulative_draws=[7, 8, 9.5, 11, 13, 15, 18, 20],
    )
    assert res.implied_draw.actual_total_draws == 20.0
    assert res.implied_draw.per_turn_actual is not None
    assert len(res.implied_draw.per_turn_actual) == 8


# ---------------------------------------------------------------------------
# Real-registry sanity (uses DEFAULT_REGISTRY for known cards)
# ---------------------------------------------------------------------------

def test_real_registry_promotes_sol_ring_as_ramp():
    from auto_goldfish.effects.card_database import DEFAULT_REGISTRY
    deck = [
        _land(qty=37),
        {"name": "Sol Ring", "cmc": 1, "quantity": 1, "types": ["Artifact"]},
    ]
    cls = classify_for_curve_value(deck, registry=DEFAULT_REGISTRY)
    assert any(r.name == "Sol Ring" and r.mana_per_turn == 2.0 for r in cls["ramp_specs"])


def test_real_registry_promotes_arcane_signet_as_ramp():
    from auto_goldfish.effects.card_database import DEFAULT_REGISTRY
    deck = [
        _land(qty=37),
        {"name": "Arcane Signet", "cmc": 2, "quantity": 1, "types": ["Artifact"]},
    ]
    cls = classify_for_curve_value(deck, registry=DEFAULT_REGISTRY)
    assert any(r.name == "Arcane Signet" and r.mana_per_turn == 1.0 for r in cls["ramp_specs"])


def test_goldfisher_exposes_full_decklist_dicts_for_optimizer_paths():
    """Regression: the optimizer paths read `_original_full_decklist_dicts`
    to pass to `result_to_dict(deck_list=...)`. Without it, optimization
    runs render no curve_value panel. See commit 5a7d45d.
    """
    from auto_goldfish.engine.goldfisher import Goldfisher

    deck = (
        [_land(qty=37)]
        + [_spell("Cmdr", cmc=4, qty=1, commander=True)]
        + [_spell(f"v{i}", cmc=3, qty=1) for i in range(62)]
    )
    sim = Goldfisher(decklist_dicts=deck, turns=4, sims=5, seed=1)
    assert hasattr(sim, "_original_full_decklist_dicts")
    full = sim._original_full_decklist_dicts
    assert isinstance(full, list)
    # Commander-included copy: the count matches the input deck.
    assert len(full) == len(deck)
    # Commanders survive in the list (the non-full sibling field strips them).
    assert any(c.get("commander") for c in full)


def test_optimizer_paths_pass_deck_list_to_result_to_dict():
    """Regression: optimizer.py, factored_optimizer.py, and fast_optimizer.py
    must each pass `deck_list=...` to `result_to_dict`, otherwise the
    curve_value panel is silently disabled in optimization-mode results.
    See commit 5a7d45d.

    Source-level check rather than a full optimizer run because the optimizer
    setup is heavy and the bug is exactly "the keyword is missing from the
    call site".
    """
    import inspect

    from auto_goldfish.optimization import (
        factored_optimizer,
        fast_optimizer,
        optimizer,
    )

    for module in [optimizer, factored_optimizer, fast_optimizer]:
        source = inspect.getsource(module)
        assert "result_to_dict(" in source, (
            f"{module.__name__}: doesn't call result_to_dict at all -- "
            f"this test should be updated."
        )
        assert "deck_list=" in source, (
            f"{module.__name__}: result_to_dict call appears to omit deck_list, "
            f"which silently sets curve_value=None in the result JSON and "
            f"hides the curve-value panel for optimization runs. Pass "
            f"`deck_list=self.goldfisher._original_full_decklist_dicts`."
        )


def test_reporter_falls_back_to_default_registry_when_none():
    """Regression: result_to_dict must not need an explicit registry to classify ramp.

    The web simulation_runner passes registry=None when no user overrides
    exist; Goldfisher itself defaults to DEFAULT_REGISTRY, so result_to_dict
    must too — otherwise the curve_value panel always shows no_ramp=True.
    """
    from auto_goldfish.engine.goldfisher import Goldfisher, SimulationResult
    from auto_goldfish.metrics.reporter import result_to_dict

    deck = [
        _land(qty=37),
        {"name": "Sol Ring", "cmc": 1, "quantity": 1, "types": ["Artifact"]},
        {"name": "Arcane Signet", "cmc": 2, "quantity": 1, "types": ["Artifact"]},
    ] + [_spell(f"v{i}", cmc=3, qty=1) for i in range(60)]

    sim = Goldfisher(decklist_dicts=deck, turns=8, sims=20, seed=1)
    res = sim.simulate()
    out = result_to_dict(res, turns=8, deck_list=deck, registry=None, overrides=None)
    cv = out["curve_value"]
    assert cv is not None
    isv = cv["implied_spell_value"]
    # Should detect the two ramp cards via DEFAULT_REGISTRY fallback.
    assert isv["no_ramp"] is False
    assert len(isv["per_card_irrs"]) == 2
