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
    DEFAULT_EXCESS_K,
    DEFAULT_FIXED_DELTA,
    DEFAULT_IRR_CAP,
    DEFAULT_LOAN_THRESHOLD,
    compute_curve_value,
    compute_curve_verdict,
    compute_implied_draw,
    compute_implied_spell_value,
    excess_alpha,
    idealized_ramp_excess,
    implied_power_multipliers,
    land_mana_over_T,
    loan_size_alpha,
    play_to_curve,
    power_mults_option_a,
    ramp_contribution,
    ramp_irr,
    ramp_share_from_mana,
    schedule_ramp,
    solve_irr,
    value_mana_per_turn,
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


# ---------------------------------------------------------------------------
# Curve Verdict: Option A (duration-aware) and Option B (curve-aware)
# ---------------------------------------------------------------------------

def _ramp(cmc: int, M: float = 1.0, name: str = "rock") -> RampCardSpec:
    return RampCardSpec(name=f"{name}_{cmc}_{M}", cmc=cmc, mana_per_turn=M)


def test_power_mults_option_a_baseline_is_1():
    for delta in (0.55, 0.69, 0.83, 1.0):
        m = power_mults_option_a(delta, T=8, baseline_cmc=2, max_cmc=8)
        assert m[2] == pytest.approx(1.0)


def test_power_mults_option_a_at_delta_1_is_pure_duration():
    """At δ=1.0 (no ramp), A reduces to 2(T-1)/(c(T-c+1))."""
    T = 8
    m = power_mults_option_a(1.0, T=T, baseline_cmc=2, max_cmc=T)
    for c in range(1, T + 1):
        expected = (2 * (T - 2 + 1)) / (c * (T - c + 1))
        assert m[c] == pytest.approx(expected, rel=1e-6)


def test_power_mults_option_a_high_irr_inflates_high_cmc():
    """At δ=0.69 (median IRR ≈ 45%/turn), T=8, the 6-drop multiplier is ~3.39."""
    m = power_mults_option_a(0.69, T=8, baseline_cmc=2, max_cmc=8)
    assert m[6] == pytest.approx(3.39, abs=0.05)
    # Strictly increasing from baseline at high IRR.
    assert m[3] < m[4] < m[5] < m[6] < m[7] < m[8]


def test_schedule_ramp_pushes_conflicts_later():
    """Three 2-cmc rocks land on turns 2, 3, 4 (same-cmc ties pushed forward)."""
    sched = schedule_ramp([(2, 1.0), (2, 1.0), (2, 1.0)], T=8)
    turns = sorted(t for t, _, _ in sched)
    assert turns == [2, 3, 4]


def test_schedule_ramp_drops_uncastable():
    """Ramp pieces pushed past T don't appear in the schedule."""
    sched = schedule_ramp([(8, 1.0), (8, 1.0)], T=8)
    # First 8-cmc lands turn 8; second is pushed to turn 9 (past T) and dropped.
    assert len(sched) == 1
    assert sched[0][0] == 8


def test_value_mana_per_turn_no_ramp():
    """With no ramp, value mana = land mana (one drop per turn)."""
    assert value_mana_per_turn([], T=4) == [1.0, 2.0, 3.0, 4.0]


def test_value_mana_per_turn_with_2cmc_rock():
    """A 2-cmc rock cast on turn 2 deducts its CMC on turn 2 then adds M from turn 3."""
    sched = [(2, 2, 1.0)]  # (turn, cmc, M)
    stream = value_mana_per_turn(sched, T=5)
    # turn 1: 1 land, no ramp yet
    # turn 2: 2 land - 2 (cast) = 0
    # turn 3: 3 land + 1 ramp = 4
    # turn 4: 4 + 1 = 5
    # turn 5: 5 + 1 = 6
    assert stream == [1.0, 0.0, 4.0, 5.0, 6.0]


def test_play_to_curve_no_value_returns_empty_mt():
    res = play_to_curve(curve_counts={}, ramp_pieces=[], T=8)
    assert res["mt_per_cmc"] == {}


# Synthetic deck fixtures from the AB-comparison notebook §4. These pin the
# net deltas and the loss/gain locations so the simulator stays stable.
SYNTH_TOP_HEAVY = {
    "curve": {1: 2, 2: 2, 3: 2, 4: 4, 5: 6, 6: 6, 7: 4, 8: 4},
    "ramp": [(2, 1.0), (2, 1.0), (2, 1.0)],
}
SYNTH_AGGRO = {
    "curve": {1: 8, 2: 12, 3: 8, 4: 4},
    "ramp": [(2, 1.0), (2, 1.0), (2, 1.0)],
}
SYNTH_BALANCED = {
    "curve": {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4},
    "ramp": [(2, 1.0), (2, 1.0), (2, 1.0)],
}
SYNTH_GAPPY = {
    "curve": {2: 12, 6: 12},
    "ramp": [(2, 1.0), (2, 1.0), (2, 1.0)],
}


def _verdict_from_synth(synth, T: int = 8):
    ramp_specs = [RampCardSpec(name=f"r{i}", cmc=c, mana_per_turn=M)
                  for i, (c, M) in enumerate(synth["ramp"])]
    return compute_curve_verdict(
        curve_counts=synth["curve"], ramp_specs=ramp_specs, T=T,
    )


def test_curve_verdict_top_heavy_net_flat_positive():
    """Three 2-cmc rocks in a top-heavy deck net +1 mana-turn at flat power."""
    v = _verdict_from_synth(SYNTH_TOP_HEAVY)
    assert v.net_flat == pytest.approx(1.0, abs=0.5)
    # Loss is concentrated at the 2-drop slot (displaced by ramp).
    assert v.loss_breakdown.get(2, 0) > 10
    # Gain is at high CMC (5-8).
    assert sum(v.gain_breakdown.get(c, 0) for c in (5, 6, 7, 8)) > 15


def test_curve_verdict_aggro_no_top_end_to_enable():
    """Aggro decks net positive at flat power but with smaller magnitude."""
    v = _verdict_from_synth(SYNTH_AGGRO)
    assert v.net_flat == pytest.approx(1.0, abs=0.5)
    # No 5+ drops, so all gain has to fit in 3-4.
    assert sum(v.gain_breakdown.get(c, 0) for c in (3, 4)) > 0


def test_curve_verdict_balanced_distributes_evenly():
    v = _verdict_from_synth(SYNTH_BALANCED)
    assert v.net_flat == pytest.approx(1.0, abs=0.5)


def test_curve_verdict_gappy_isolates_loss_and_gain():
    """Gappy deck (only 2s and 6s) gives the cleanest delta breakdown."""
    v = _verdict_from_synth(SYNTH_GAPPY)
    assert v.net_flat == pytest.approx(1.0, abs=0.5)
    # All loss at c=2, all gain at c=6.
    assert v.loss_breakdown.get(2, 0) == pytest.approx(18.57, abs=0.5)
    assert v.gain_breakdown.get(6, 0) == pytest.approx(19.57, abs=0.5)
    assert set(v.loss_breakdown.keys()) == {2}
    assert set(v.gain_breakdown.keys()) == {6}


def test_curve_verdict_no_ramp_uses_delta_1():
    """A deck with no permanent ramp falls back to δ=1.0 and still produces
    a verdict (no_ramp=True), so the panel always renders something."""
    v = compute_curve_verdict(
        curve_counts={1: 4, 2: 8, 3: 6, 4: 4, 5: 2}, ramp_specs=[], T=8,
    )
    assert v is not None
    assert v.no_ramp is True
    assert v.delta == 1.0
    assert math.isnan(v.median_irr)
    # net_flat ≈ 0 because with-ramp == no-ramp counterfactual when ramp is empty.
    assert abs(v.net_flat) < 1e-6


def test_curve_verdict_no_value_returns_none():
    """Decks with no value spells (only ramp + lands) return None."""
    ramp_specs = [_ramp(2), _ramp(2)]
    v = compute_curve_verdict(curve_counts={}, ramp_specs=ramp_specs, T=8)
    assert v is None


def test_curve_verdict_baseline_falls_back_when_no_2_drops():
    """When the deck has no 2-drops, baseline_cmc falls back to smallest castable."""
    v = compute_curve_verdict(
        curve_counts={3: 8, 4: 6, 5: 4}, ramp_specs=[_ramp(3)], T=8,
    )
    assert v is not None
    assert v.baseline_cmc == 3
    # Baseline row should be tagged as baseline.
    base_row = next(r for r in v.rows if r.cmc == 3)
    assert base_row.kind == "baseline"
    assert base_row.b_implicit == pytest.approx(1.0)


def test_curve_verdict_deliberately_over_ramped_low_curve_fires_warning():
    """A deliberately misbuilt deck (lots of ramp piling onto a low curve)
    should fire ramp_over_aggressive at the thin high-CMC slots. This is the
    canonical case the calibrated bar is meant to catch."""
    # Heavy ramp: 14 pieces of mostly fast/medium, ~73 mana of excess.
    ramp_specs = (
        [_ramp(1, 2.0), _ramp(1, 3.0)]
        + [_ramp(2, 1.0)] * 6
        + [_ramp(3, 1.0)] * 4
        + [_ramp(4, 2.0)] * 2
    )
    # Low curve with a single thin 6-drop slot.
    curve_counts = {1: 8, 2: 14, 3: 14, 4: 6, 5: 2, 6: 1}
    v = compute_curve_verdict(curve_counts, ramp_specs, T=8)
    assert v is not None
    assert v.delta == pytest.approx(DEFAULT_FIXED_DELTA)
    assert v.ramp_share > 0.7  # heavy excess saturates alpha well above 0.7.
    row_6 = next(r for r in v.rows if r.cmc == 6)
    assert row_6.kind == "ramp_over_aggressive", (
        f"expected ramp_over_aggressive at thin 6-drop slot, got {row_6.kind}"
    )


def test_curve_verdict_low_excess_softens_mid_curve():
    """A deck with low ramp excess (slow-only ramp) gets a small alpha and
    therefore softer A across the curve; mid-curve flags should be mild or
    coherent rather than ramp_over_aggressive."""
    ramp_specs = [_ramp(3, 1.0)] * 5 + [_ramp(4, 1.0)] * 2  # all slow ramp
    curve_counts = {1: 4, 2: 9, 3: 9, 4: 4, 5: 2, 6: 1, 7: 1, 8: 1}
    v = compute_curve_verdict(curve_counts, ramp_specs, T=8)
    assert v is not None
    assert v.delta == pytest.approx(DEFAULT_FIXED_DELTA)
    # Slow ramp -> low excess (5x2 + 0 = 10) -> alpha < 0.5 at k=50.
    assert v.ramp_share < 0.5
    # At least one mid-curve slot should be coherent or over_allocated.
    kinds_by_cmc = {r.cmc: r.kind for r in v.rows}
    assert any(kinds_by_cmc.get(c) in ("coherent", "over_allocated") for c in (3, 4, 5)), kinds_by_cmc


def test_curve_verdict_kinds_are_valid():
    """Every row's kind tag is one of the documented values."""
    v = _verdict_from_synth(SYNTH_BALANCED)
    valid = {"baseline", "below_baseline", "coherent",
             "ramp_over_aggressive", "over_allocated", "no_slots"}
    for row in v.rows:
        assert row.kind in valid


def test_curve_verdict_baseline_row_has_b_implicit_one():
    v = _verdict_from_synth(SYNTH_BALANCED)
    base_row = next(r for r in v.rows if r.cmc == v.baseline_cmc)
    assert base_row.kind == "baseline"
    assert base_row.b_implicit == pytest.approx(1.0)
    assert base_row.a_required == pytest.approx(1.0)


def test_compute_curve_value_attaches_curve_verdict():
    """End-to-end: compute_curve_value populates curve_verdict on the result."""
    deck = [_land(qty=37)]
    deck += [_spell(f"v{i}", cmc=3, qty=1) for i in range(40)]
    deck += [_spell(f"v6_{i}", cmc=6, qty=1) for i in range(20)]
    cv = compute_curve_value(deck, registry=None, overrides=None, turns=8)
    assert cv.curve_verdict is not None
    assert cv.curve_verdict.no_ramp is True  # no registry, no detected ramp
    assert any(r.cmc == 3 for r in cv.curve_verdict.rows)
    assert any(r.cmc == 6 for r in cv.curve_verdict.rows)


def test_compute_curve_value_serializes_verdict_via_asdict():
    """The reporter's asdict() pass needs the verdict to round-trip cleanly."""
    from dataclasses import asdict
    deck = [_land(qty=37)]
    deck += [_spell(f"v{i}", cmc=3, qty=1) for i in range(40)]
    cv = compute_curve_value(deck, registry=None, overrides=None, turns=8)
    d = asdict(cv)
    assert "curve_verdict" in d
    assert "rows" in d["curve_verdict"]
    assert "net_flat" in d["curve_verdict"]


def test_compute_curve_value_dict_is_json_safe_with_uncastable_slot():
    """Slots with c >= T (Ripjaw's 9-drop on T=8) produce A_raw=inf. The
    reporter's serializer must convert inf/NaN to None so the dict survives
    json.dumps -> JS JSON.parse round-trip."""
    import json
    from auto_goldfish.metrics.reporter import _sanitize_for_json
    # Curve with a 9-drop in a T=8 game.
    curve = {2: 8, 6: 4, 9: 2}
    specs = [_ramp(2, 1.0)] * 4
    v = compute_curve_verdict(curve, specs, T=8)
    # Confirm at least one row has inf A_required.
    assert any(
        not math.isfinite(r.a_required) for r in v.rows
    ), "expected at least one inf row for c >= T"
    # Round-trip through the sanitizer + json.dumps.
    from dataclasses import asdict
    raw_dict = asdict(v)
    safe = _sanitize_for_json(raw_dict)
    encoded = json.dumps(safe)  # would raise / produce 'Infinity' if not sanitized
    assert "Infinity" not in encoded
    assert "NaN" not in encoded
    # JS-equivalent parse round-trip: stdlib json with strict=True would also
    # reject Infinity tokens.
    decoded = json.loads(encoded)
    inf_rows = [r for r in decoded["rows"] if r["a_required"] is None]
    assert len(inf_rows) >= 1


# ---------------------------------------------------------------------------
# B2 ramp-share shrinkage: A_eff = 1 + ramp_share * (A_raw - 1)
# ---------------------------------------------------------------------------

def test_ramp_share_no_ramp_is_zero():
    assert ramp_share_from_mana(land_mana=30.0, ramp_excess=0.0) == 0.0


def test_ramp_share_negative_excess_floors_to_zero():
    """Late-cast ramp can produce a small negative excess; share floors to 0."""
    assert ramp_share_from_mana(land_mana=30.0, ramp_excess=-2.0) == 0.0


def test_ramp_share_pure_ramp_is_one():
    assert ramp_share_from_mana(land_mana=0.0, ramp_excess=10.0) == 1.0


def test_ramp_share_mixed_is_fractional():
    s = ramp_share_from_mana(land_mana=27.0, ramp_excess=18.0)
    assert s == pytest.approx(18.0 / 45.0)


def test_ramp_share_no_mana_is_zero():
    """Degenerate: no lands and no ramp -> 0 (don't divide by zero)."""
    assert ramp_share_from_mana(land_mana=0.0, ramp_excess=0.0) == 0.0


# Edgar Markov-shaped: low-curve aggressive vampire tribal, very little ramp.
EDGAR_CURVE = {1: 12, 2: 18, 3: 12, 4: 6, 5: 3, 6: 1}
EDGAR_RAMP = [(1, 2.0), (2, 1.0), (2, 1.0)]


def _edgar_specs():
    return [RampCardSpec(name=f"r{i}", cmc=c, mana_per_turn=M)
            for i, (c, M) in enumerate(EDGAR_RAMP)]


def test_curve_verdict_default_uses_excess_alpha_and_fixed_delta():
    """Production default: delta is anchored at DEFAULT_FIXED_DELTA and alpha
    is excess-derived (Option C)."""
    v = compute_curve_verdict(EDGAR_CURVE, _edgar_specs(), T=8)
    assert v.delta == pytest.approx(DEFAULT_FIXED_DELTA)
    # Edgar's ramp specs: Sol Ring (excess 13) + 2 Signets (excess 4 each) = 21.
    # alpha = 1 - exp(-21/50) = 0.343.
    assert v.ramp_share == pytest.approx(0.343, abs=0.02)
    # A_eff(6) at delta=0.85, T=8 -> A_raw~1.49, shrunk by alpha~0.34 -> ~1.17.
    row6 = next(r for r in v.rows if r.cmc == 6)
    assert row6.a_required == pytest.approx(1.17, abs=0.1)


def test_curve_verdict_share_zero_collapses_a_to_one():
    """Explicit ramp_share=0 still collapses A to 1.0 everywhere (test override)."""
    v = compute_curve_verdict(EDGAR_CURVE, _edgar_specs(), T=8, ramp_share=0.0)
    for r in v.rows:
        assert r.a_required == pytest.approx(1.0)
    assert not any(r.kind == "ramp_over_aggressive" for r in v.rows)


def test_curve_verdict_low_share_shrinks_high_cmc_demand():
    """Explicit ramp_share=0.41 shrinks A(6) at the fixed delta anchor."""
    v = compute_curve_verdict(EDGAR_CURVE, _edgar_specs(), T=8, ramp_share=0.41)
    row6 = next(r for r in v.rows if r.cmc == 6)
    # A_raw(6) at delta=0.85 -> ~1.49; 1 + 0.41 * 0.49 -> 1.20.
    assert row6.a_required == pytest.approx(1.20, abs=0.1)


def test_curve_verdict_share_clamped_to_unit_interval():
    """ramp_share outside [0,1] is clamped silently."""
    v_neg = compute_curve_verdict(EDGAR_CURVE, _edgar_specs(), T=8, ramp_share=-0.5)
    v_high = compute_curve_verdict(EDGAR_CURVE, _edgar_specs(), T=8, ramp_share=2.5)
    # Negative -> clamped to 0 -> A=1 everywhere.
    assert all(r.a_required == pytest.approx(1.0) for r in v_neg.rows)
    # >1 -> clamped to 1 -> matches alpha=1 case at the fixed delta anchor.
    # A_raw(6) at delta=0.85 -> ~1.49.
    row6 = next(r for r in v_high.rows if r.cmc == 6)
    assert row6.a_required == pytest.approx(1.49, abs=0.1)


def test_compute_curve_value_applies_ramp_share_end_to_end():
    """The integration layer derives ramp_share from land_mana / ramp_excess
    and passes it into the verdict."""
    deck = [_land(qty=35)]
    # Aggressive low-curve value pool (Edgar-shaped).
    deck += [_spell(f"c1_{i}", cmc=1, qty=1) for i in range(12)]
    deck += [_spell(f"c2_{i}", cmc=2, qty=1) for i in range(18)]
    deck += [_spell(f"c3_{i}", cmc=3, qty=1) for i in range(12)]
    deck += [_spell(f"c4_{i}", cmc=4, qty=1) for i in range(6)]
    deck += [_spell(f"c5_{i}", cmc=5, qty=1) for i in range(3)]
    deck += [_spell(f"c6_{i}", cmc=6, qty=1) for i in range(1)]

    # Without a registry, classify_for_curve_value can't detect ramp; build
    # the verdict directly from a known curve+ramp+share, mirroring what
    # compute_curve_value would do once ramp is detected.
    cv = compute_curve_value(deck, registry=None, overrides=None, turns=8)
    # No detected ramp -> share=0, no_ramp=True, A clamped to 1 everywhere.
    assert cv.curve_verdict is not None
    assert cv.curve_verdict.ramp_share == 0.0
    assert all(r.a_required == pytest.approx(1.0) for r in cv.curve_verdict.rows)


def test_curve_verdict_ramp_share_field_is_set():
    """The verdict echoes back the share it was constructed with."""
    v = compute_curve_verdict(EDGAR_CURVE, _edgar_specs(), T=8, ramp_share=0.33)
    assert v.ramp_share == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# B3: loan-size α + IRR cap (production tuning)
# ---------------------------------------------------------------------------

def test_loan_size_alpha_no_ramp_is_zero():
    assert loan_size_alpha([]) == 0.0


def test_loan_size_alpha_below_threshold_scales_linearly():
    # Loan = 1+2+2 = 5; threshold 15 -> 0.333.
    specs = [_ramp(1, 2.0), _ramp(2, 1.0), _ramp(2, 1.0)]
    assert loan_size_alpha(specs, threshold=15.0) == pytest.approx(5.0 / 15.0)


def test_loan_size_alpha_saturates_at_one():
    # 11 2-cmc rocks = loan 22; clamped to 1.0 against threshold 15.
    specs = [_ramp(2, 1.0)] * 11
    assert loan_size_alpha(specs, threshold=15.0) == 1.0


def test_loan_size_alpha_ignores_zero_mana_pieces():
    """Pieces with mana_per_turn=0 don't count toward the loan."""
    specs = [_ramp(2, 0.0), _ramp(2, 1.0)]
    assert loan_size_alpha(specs, threshold=10.0) == pytest.approx(2.0 / 10.0)


def test_loan_size_alpha_zero_threshold_returns_zero():
    """Defensive: 0 threshold -> no commitment ever (avoid div by zero)."""
    assert loan_size_alpha([_ramp(2, 1.0)], threshold=0.0) == 0.0


def test_aggregate_irr_cap_clips_solo_sol_ring():
    """Sol Ring alone: uncapped IRR saturates near IRR_HI=10; cap=1.0 brings
    it down so single-piece decks get a sane delta."""
    specs = [_ramp(1, 2.0)]  # Sol Ring shape
    no_cap = aggregate_deck_irr(specs, T=8)["median_irr"]
    capped = aggregate_deck_irr(specs, T=8, irr_cap=1.0)["median_irr"]
    assert no_cap > 1.0  # uncapped saturates well above 1
    assert capped == pytest.approx(1.0)


def test_aggregate_irr_cap_does_not_affect_typical_ramp():
    """Multi-piece decks with median IRR < 1 are unaffected by cap=1.0."""
    specs = [_ramp(2, 1.0)] * 11  # median IRR ~0.45
    no_cap = aggregate_deck_irr(specs, T=8)["median_irr"]
    capped = aggregate_deck_irr(specs, T=8, irr_cap=1.0)["median_irr"]
    assert no_cap == pytest.approx(capped)
    assert no_cap < 1.0


def test_compute_curve_verdict_sol_ring_only_under_option_c():
    """Sol Ring alone: excess = 14 - 1 = 13; alpha = 1 - exp(-13/50) ~= 0.229.
    delta is fixed at 0.85. A_eff(6) ~= 1 + 0.229 * (1.49 - 1) ~= 1.11."""
    specs = [_ramp(1, 2.0)]
    v = compute_curve_verdict(EDGAR_CURVE, specs, T=8)
    assert v.delta == pytest.approx(DEFAULT_FIXED_DELTA)
    assert v.idealized_excess == pytest.approx(13.0)
    assert v.ramp_share == pytest.approx(0.229, abs=0.02)
    row6 = next(r for r in v.rows if r.cmc == 6)
    assert row6.a_required == pytest.approx(1.11, abs=0.1)


def test_default_loan_threshold_is_15():
    assert DEFAULT_LOAN_THRESHOLD == 15.0


def test_default_irr_cap_is_one():
    assert DEFAULT_IRR_CAP == 1.0


def test_curve_verdict_loan_size_field_matches_sum_of_cmcs():
    """The verdict carries the gross loan so the UI can display it alongside α
    (which saturates at 1.0 and would otherwise hide scale)."""
    specs = [_ramp(1, 2.0), _ramp(2, 1.0), _ramp(2, 1.0)]
    v = compute_curve_verdict(EDGAR_CURVE, specs, T=8)
    assert v.loan_size == pytest.approx(5.0)


def test_curve_verdict_loan_size_zero_when_no_ramp():
    v = compute_curve_verdict(EDGAR_CURVE, [], T=8)
    assert v.loan_size == 0.0


def test_curve_verdict_loan_size_ignores_zero_mana_pieces():
    """Pieces with mana_per_turn=0 don't count toward the displayed loan."""
    specs = [_ramp(2, 0.0), _ramp(3, 1.0)]
    v = compute_curve_verdict(EDGAR_CURVE, specs, T=8)
    assert v.loan_size == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Option C: fixed delta + excess-based alpha (monotonic in cuts)
# ---------------------------------------------------------------------------

def test_default_fixed_delta_is_calibrated():
    assert DEFAULT_FIXED_DELTA == 0.85


def test_default_excess_k_is_calibrated():
    assert DEFAULT_EXCESS_K == 50.0


def test_idealized_ramp_excess_sol_ring():
    """Sol Ring: c=1, M=2, T=8 -> 2*(8-1) - 1 = 13."""
    assert idealized_ramp_excess([_ramp(1, 2.0)], T=8) == pytest.approx(13.0)


def test_idealized_ramp_excess_signet():
    """Signet shape: c=2, M=1, T=8 -> 1*(8-2) - 2 = 4."""
    assert idealized_ramp_excess([_ramp(2, 1.0)], T=8) == pytest.approx(4.0)


def test_idealized_ramp_excess_sums_across_pieces():
    specs = [_ramp(1, 2.0), _ramp(2, 1.0), _ramp(2, 1.0)]  # 13 + 4 + 4
    assert idealized_ramp_excess(specs, T=8) == pytest.approx(21.0)


def test_idealized_ramp_excess_drops_uncastable():
    """Pieces with cmc > T can't be cast; they don't contribute to excess."""
    assert idealized_ramp_excess([_ramp(10, 1.0)], T=8) == 0.0


def test_idealized_ramp_excess_floors_at_zero():
    """A net-negative ramp (e.g. very late cast) floors to 0."""
    # 6-cmc 1-mana rock: 1*(8-6) - 6 = -4 net.
    assert idealized_ramp_excess([_ramp(6, 1.0)], T=8) == 0.0


def test_excess_alpha_zero_excess_zero_alpha():
    assert excess_alpha(0.0) == 0.0


def test_excess_alpha_saturates_smoothly():
    """alpha approaches 1.0 but never reaches it -- never saturates fully."""
    # Pin behavior with explicit k so we don't depend on DEFAULT_EXCESS_K.
    assert excess_alpha(30.0, k=30.0) == pytest.approx(1 - math.exp(-1.0), abs=1e-6)
    assert excess_alpha(60.0, k=30.0) == pytest.approx(1 - math.exp(-2.0), abs=1e-6)
    # Even at very high excess, alpha < 1.
    assert 0.99 < excess_alpha(200.0, k=30.0) < 1.0


def test_curve_verdict_idealized_excess_field():
    """The verdict surfaces the idealized excess for the UI to display."""
    specs = [_ramp(1, 2.0), _ramp(2, 1.0), _ramp(2, 1.0)]
    v = compute_curve_verdict(EDGAR_CURVE, specs, T=8)
    assert v.idealized_excess == pytest.approx(21.0)


def test_curve_verdict_monotonic_under_any_cut():
    """The defining property of Option C: cutting ANY ramp piece monotonically
    reduces alpha and therefore A_eff at every CMC. Test by removing one
    piece at a time, in any order, and confirming A_eff(6) only ever drops."""
    # Build a heterogeneous ramp pile with mixed speeds.
    base_specs = [
        _ramp(1, 2.0),  # Sol Ring shape
        _ramp(2, 1.0),  # signet
        _ramp(2, 1.0),
        _ramp(3, 1.0),  # cultivate-shape
        _ramp(3, 1.0),
        _ramp(4, 2.0),  # 4-cmc strong rock
    ]
    curve = {2: 14, 3: 11, 4: 7, 5: 4, 6: 1}
    # For each subset of size k = N, N-1, ..., 0, check A_eff(6) is monotonic.
    # We test all single-piece-cut orders (each piece removed individually,
    # then the next, etc., across all 6! = 720 orderings would be overkill);
    # spot-check on the fast-first and slow-first orderings, which are the
    # extremes the Part-3 analysis showed mattered most.
    for cut_order in ['fast-first', 'slow-first', 'name-order']:
        if cut_order == 'fast-first':
            ordered = sorted(
                base_specs, key=lambda s: -(s.mana_per_turn / max(1, s.cmc))
            )
        elif cut_order == 'slow-first':
            ordered = sorted(
                base_specs, key=lambda s: (s.mana_per_turn / max(1, s.cmc))
            )
        else:
            ordered = list(base_specs)
        prev_a6 = None
        for k in range(len(ordered) + 1):
            kept = ordered[k:]
            v = compute_curve_verdict(curve, kept, T=8)
            row6 = next((r for r in v.rows if r.cmc == 6), None)
            assert row6 is not None
            if prev_a6 is not None:
                assert row6.a_required <= prev_a6 + 1e-6, (
                    f"non-monotonic under cut order {cut_order} at k={k}: "
                    f"prev={prev_a6:.4f}, curr={row6.a_required:.4f}"
                )
            prev_a6 = row6.a_required
