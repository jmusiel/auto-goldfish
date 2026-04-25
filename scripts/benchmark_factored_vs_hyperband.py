"""Benchmark FactoredOptimizer vs DeckOptimizer (Hyperband) on the demo decks.

Measures wall time and surfaces top recommendations from each optimizer for
qualitative comparison.
"""
from __future__ import annotations

import time
from typing import Any

from auto_goldfish.decklist.loader import load_decklist
from auto_goldfish.engine.goldfisher import Goldfisher
from auto_goldfish.optimization.candidate_cards import ALL_CANDIDATES
from auto_goldfish.optimization.factored_optimizer import FactoredOptimizer
from auto_goldfish.optimization.optimizer import DeckOptimizer

DEMO_DECKS = [
    "mana-starved-demo",
    "overlanded-cantrips-demo",
    "equilibrium-demo",
]

TURNS = 8
SEED = 42
LAND_RANGE = 3
MAX_DRAW = 2
MAX_RAMP = 2
FINAL_SIMS = 500
HYPERBAND_MAX_SIMS = 200


def _run_optimizer(name: str, build_optimizer):
    t0 = time.perf_counter()
    optimizer = build_optimizer()
    ranked = optimizer.run(final_sims=FINAL_SIMS, final_top_k=5)
    elapsed = time.perf_counter() - t0
    return name, elapsed, ranked


def _summarize(name: str, elapsed: float, ranked):
    print(f"\n  [{name}]  {elapsed:6.2f}s   top {len(ranked)}:")
    for i, (cfg, result) in enumerate(ranked):
        score = result.get("mean_mana", float("nan"))
        cons = result.get("consistency", float("nan"))
        print(
            f"    {i+1}. {cfg.describe():40s}  "
            f"mean_mana={score:6.2f}  consistency={cons:5.2%}"
        )


def benchmark_deck(deck_name: str) -> None:
    print(f"\n{'=' * 70}\n  DECK: {deck_name}\n{'=' * 70}")
    cards = load_decklist(deck_name)

    def make_goldfisher() -> Goldfisher:
        return Goldfisher(
            cards,
            turns=TURNS,
            sims=200,
            seed=SEED,
            record_results="quartile",
        )

    def build_factored() -> FactoredOptimizer:
        return FactoredOptimizer(
            goldfisher=make_goldfisher(),
            candidates=ALL_CANDIDATES,
            max_draw=MAX_DRAW,
            max_ramp=MAX_RAMP,
            land_range=LAND_RANGE,
            base_games=200,
            max_games=800,
        )

    def build_hyperband() -> DeckOptimizer:
        return DeckOptimizer(
            goldfisher=make_goldfisher(),
            candidates=ALL_CANDIDATES,
            max_draw=MAX_DRAW,
            max_ramp=MAX_RAMP,
            land_range=LAND_RANGE,
            hyperband_max_sims=HYPERBAND_MAX_SIMS,
        )

    factored_result = _run_optimizer("FACTORED ", build_factored)
    hyperband_result = _run_optimizer("HYPERBAND", build_hyperband)

    _summarize(*factored_result)
    _summarize(*hyperband_result)

    f_time = factored_result[1]
    h_time = hyperband_result[1]
    speedup = h_time / f_time if f_time > 0 else float("inf")
    print(f"\n  Speedup: {speedup:.2f}x faster (factored)")


if __name__ == "__main__":
    print("Benchmark: Factored vs Hyperband")
    print(f"  turns={TURNS}, seed={SEED}, land_range={LAND_RANGE}")
    print(f"  max_draw={MAX_DRAW}, max_ramp={MAX_RAMP}")
    print(f"  final_sims={FINAL_SIMS}, hyperband_max_sims={HYPERBAND_MAX_SIMS}")

    for deck in DEMO_DECKS:
        benchmark_deck(deck)
