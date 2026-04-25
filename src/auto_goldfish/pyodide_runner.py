"""Entry point for running simulations in Pyodide (client-side WebAssembly).

This module is loaded by the Web Worker and called from JavaScript.
It orchestrates a full simulation run using only sequential execution
(workers=1), which is the only option available in the Pyodide environment.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from auto_goldfish.effects.card_database import DEFAULT_REGISTRY
from auto_goldfish.effects.json_loader import build_overridden_registry
from auto_goldfish.engine.goldfisher import Goldfisher
from auto_goldfish.engine.mulligan import CurveAwareMulligan
from auto_goldfish.metrics.reporter import result_to_dict


def run_simulation(
    deck_json: str,
    config_json: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Entry point called from JavaScript via Pyodide.

    Args:
        deck_json: JSON string of deck card list (list of card dicts).
        config_json: JSON string with simulation configuration:
            - turns (int): Number of turns per game (default 10)
            - sims (int): Number of simulations per land count (default 1000)
            - min_lands (int): Minimum land count to sweep
            - max_lands (int): Maximum land count to sweep
            - seed (int|null): Random seed (default null)
            - record_results (str): Detail level - "quartile"/"decile"/"centile"
            - effect_overrides (dict): Card name -> override JSON format
            - mulligan (str): Mulligan strategy - "default"/"curve_aware"
        progress_callback: Optional callable(current, total) for progress
            updates. Called during each simulation run. The total reflects
            sims * number_of_land_counts.

    Returns:
        JSON string of list[result_to_dict(result)] for each land count.
    """
    deck_list: List[Dict[str, Any]] = json.loads(deck_json)
    config: Dict[str, Any] = json.loads(config_json)

    turns = config.get("turns", 8)
    sims = config.get("sims", 1000)
    min_lands = config.get("min_lands")
    max_lands = config.get("max_lands")
    seed = config.get("seed")
    record_results = config.get("record_results", "quartile")
    effect_overrides = config.get("effect_overrides", {})
    mulligan_type = config.get("mulligan", "default")

    # Build registry with overrides
    registry = None
    if effect_overrides:
        registry = build_overridden_registry(DEFAULT_REGISTRY, effect_overrides)

    # Build mulligan strategy
    mulligan_strategy = None
    if mulligan_type == "curve_aware":
        mulligan_strategy = CurveAwareMulligan()

    mana_mode = config.get("mana_mode", "value")
    spell_priority = config.get("spell_priority", "priority_then_cmc")
    mana_efficiency = config.get("mana_efficiency", "greedy")
    ramp_cutoff_turn = config.get("ramp_cutoff_turn", 0)
    min_cost_floor = config.get("min_cost_floor", 1)

    goldfisher = Goldfisher(
        deck_list,
        turns=turns,
        sims=sims,
        verbose=False,
        record_results=record_results,
        seed=seed,
        workers=1,  # Always sequential in Pyodide
        mulligan_strategy=mulligan_strategy,
        registry=registry,
        mana_mode=mana_mode,
        spell_priority=spell_priority,
        mana_efficiency=mana_efficiency,
        ramp_cutoff_turn=ramp_cutoff_turn,
        min_cost_floor=min_cost_floor,
    )

    # Determine land range
    if min_lands is None:
        min_lands = goldfisher.land_count
    if max_lands is None:
        max_lands = goldfisher.land_count

    num_land_counts = max_lands - min_lands + 1
    results: List[Dict[str, Any]] = []

    for land_idx, land_count in enumerate(range(min_lands, max_lands + 1)):
        goldfisher.set_lands(land_count, cuts=[])

        # Wrap progress callback to report global progress across all land counts
        land_callback = None
        if progress_callback is not None:
            offset = land_idx * sims

            def land_callback(current: int, total: int, _offset: int = offset) -> None:
                progress_callback(
                    _offset + current,
                    num_land_counts * sims,
                )

        result = goldfisher.simulate(progress_callback=land_callback)
        results.append(result_to_dict(result, turns=turns))

    return json.dumps(results)


def run_optimization(
    deck_json: str,
    config_json: str,
    enum_callback: Optional[Callable[[int, int], None]] = None,
    eval_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Run card optimization from JavaScript via Pyodide.

    Args:
        deck_json: JSON string of deck card list.
        config_json: JSON string with optimization configuration:
            - turns, sims, seed, record_results, effect_overrides, mulligan
              (same as run_simulation; sims controls final evaluation count)
            - optimize_for (str): "mean_mana", "consistency", or "mean_spells_cast"
            - swap_mode (bool): Replace cards or add extra
            - hyperband_max_sims (int): Max sims per candidate during Hyperband selection (default sims//2)
            - enabled_candidates (list[str]): Candidate IDs that are enabled
            - custom_draw (dict|null): {cmc, amount} for custom draw candidate
            - custom_ramp (dict|null): {cmc, amount} for custom ramp candidate
            - max_draw_additions (int): Max draw cards to add (0-2)
            - max_ramp_additions (int): Max ramp cards to add (0-2)
        enum_callback: Optional callable(current, total) for enumeration progress.
        eval_callback: Optional callable(current, total) for evaluation progress.

    Returns:
        JSON string of list[{config_description, ...result_fields}].
    """
    from auto_goldfish.optimization.candidate_cards import (
        ALL_CANDIDATES,
        make_custom_candidate,
    )
    from auto_goldfish.optimization.fast_optimizer import FastDeckOptimizer
    from auto_goldfish.optimization.optimizer import DeckOptimizer

    deck_list: List[Dict[str, Any]] = json.loads(deck_json)
    config: Dict[str, Any] = json.loads(config_json)

    turns = config.get("turns", 8)
    sims = config.get("sims", 500)
    seed = config.get("seed")
    record_results = config.get("record_results", "quartile")
    effect_overrides = config.get("effect_overrides", {})
    mulligan_type = config.get("mulligan", "default")

    # Optimization config
    optimize_for = config.get("optimize_for", "floor_performance")
    swap_mode = config.get("swap_mode", False)
    enabled_ids = set(config.get("enabled_candidates", []))
    custom_draw = config.get("custom_draw")
    custom_ramp = config.get("custom_ramp")
    max_draw = config.get("max_draw_additions", 2)
    max_ramp = config.get("max_ramp_additions", 2)

    # Build registry
    registry = None
    if effect_overrides:
        registry = build_overridden_registry(DEFAULT_REGISTRY, effect_overrides)

    mulligan_strategy = None
    if mulligan_type == "curve_aware":
        mulligan_strategy = CurveAwareMulligan()

    mana_mode = config.get("mana_mode", "value")
    spell_priority = config.get("spell_priority", "priority_then_cmc")
    mana_efficiency = config.get("mana_efficiency", "greedy")
    ramp_cutoff_turn = config.get("ramp_cutoff_turn", 0)
    min_cost_floor = config.get("min_cost_floor", 1)
    hyperband_max_sims = config.get("hyperband_max_sims", max(sims // 2, 100))
    eta = config.get("eta", 3)
    hyperband_min_sims = config.get("hyperband_min_sims", 20)
    hyperband_top_k = config.get("hyperband_top_k")
    include_hyperband = config.get("include_hyperband", False)

    goldfisher = Goldfisher(
        deck_list,
        turns=turns,
        sims=sims,
        verbose=False,
        record_results=record_results,
        seed=seed,
        workers=1,
        mulligan_strategy=mulligan_strategy,
        registry=registry,
        mana_mode=mana_mode,
        spell_priority=spell_priority,
        mana_efficiency=mana_efficiency,
        ramp_cutoff_turn=ramp_cutoff_turn,
        min_cost_floor=min_cost_floor,
    )

    # Build enabled candidates dict
    candidates = {
        cid: c for cid, c in ALL_CANDIDATES.items() if cid in enabled_ids
    }

    # Add custom candidates if provided
    if custom_draw and custom_draw.get("cmc") is not None:
        cc = make_custom_candidate("draw", custom_draw["cmc"], custom_draw["amount"])
        candidates[cc.id] = cc
    if custom_ramp and custom_ramp.get("cmc") is not None:
        cc = make_custom_candidate("ramp", custom_ramp["cmc"], custom_ramp["amount"])
        candidates[cc.id] = cc

    # Compute land deltas from absolute min/max lands
    min_lands = config.get("min_lands")
    max_lands = config.get("max_lands")
    land_delta_min = None
    land_delta_max = None
    if min_lands is not None:
        land_delta_min = min_lands - goldfisher.land_count
    if max_lands is not None:
        land_delta_max = max_lands - goldfisher.land_count

    algorithm = config.get("algorithm", "racing")

    if algorithm == "racing":
        optimizer = FastDeckOptimizer(
            goldfisher=goldfisher,
            candidates=candidates,
            swap_mode=swap_mode,
            max_draw=max_draw,
            max_ramp=max_ramp,
            land_delta_min=land_delta_min,
            land_delta_max=land_delta_max,
            optimize_for=optimize_for,
            hyperband_max_sims=hyperband_max_sims,
        )
    else:
        optimizer = DeckOptimizer(
            goldfisher=goldfisher,
            candidates=candidates,
            swap_mode=swap_mode,
            max_draw=max_draw,
            max_ramp=max_ramp,
            land_delta_min=land_delta_min,
            land_delta_max=land_delta_max,
            optimize_for=optimize_for,
            hyperband_max_sims=hyperband_max_sims,
            eta=eta,
            hyperband_min_sims=hyperband_min_sims,
            hyperband_top_k=hyperband_top_k,
        )

    ranked = optimizer.run(
        final_sims=sims,
        final_top_k=5,
        include_hyperband=include_hyperband,
        enum_progress=enum_callback,
        eval_progress=eval_callback,
    )

    # Annotate results with config descriptions
    output: List[Dict[str, Any]] = []
    for deck_config, result_dict in ranked:
        result_dict["opt_config"] = deck_config.describe()
        result_dict["opt_land_delta"] = deck_config.land_delta
        result_dict["opt_added_cards"] = list(deck_config.added_cards)
        output.append(result_dict)

    return json.dumps(output)
