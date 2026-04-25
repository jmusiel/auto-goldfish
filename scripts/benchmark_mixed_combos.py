"""Benchmark FactoredOptimizer with mixed_combos=False vs True on demo decks.

Tests whether allowing within-dimension mixed combinations of distinct
ramp/draw spells (e.g. 1x Night's Whisper + 1x Concentrate) finds better
configs than the default behavior (which only tests 2 copies of the single
best card per dimension), and measures the simulation budget delta.
"""
from __future__ import annotations

import time
from typing import Any

from auto_goldfish.decklist.loader import load_decklist
from auto_goldfish.engine.goldfisher import Goldfisher
from auto_goldfish.optimization.candidate_cards import ALL_CANDIDATES
from auto_goldfish.optimization.deck_config import DeckConfig
from auto_goldfish.optimization.factored_optimizer import FactoredOptimizer

DEMO_DECKS = [
    "mana-starved-demo",
    "overlanded-cantrips-demo",
    "equilibrium-demo",
]

METRICS = ["floor_performance", "mean_mana"]

TURNS = 8
SEED = 42
LAND_RANGE = 3
MAX_DRAW = 2
MAX_RAMP = 2
FINAL_SIMS = 500


def _is_mixed_distinct(cfg: DeckConfig) -> bool:
    """True if config adds 2+ distinct cards within the same dimension."""
    if len(cfg.added_cards) < 2:
        return False
    by_dim: dict[str, set[str]] = {}
    for cid in cfg.added_cards:
        cand = ALL_CANDIDATES.get(cid)
        if cand is None:
            continue
        by_dim.setdefault(cand.card_type, set()).add(cid)
    return any(len(ids) >= 2 for ids in by_dim.values())


def _score(result_dict: dict[str, Any], metric: str) -> float:
    if metric == "floor_performance":
        return result_dict.get("threshold_mana", 0.0)
    if metric == "consistency":
        return result_dict.get("consistency", 0.0)
    return result_dict.get("mean_mana", 0.0)


def _run(
    name: str, mixed: bool, cards, metric: str,
) -> tuple[str, float, list, int]:
    gf = Goldfisher(
        cards, turns=TURNS, sims=200, seed=SEED, record_results="quartile",
    )
    opt = FactoredOptimizer(
        goldfisher=gf,
        candidates=ALL_CANDIDATES,
        max_draw=MAX_DRAW,
        max_ramp=MAX_RAMP,
        land_range=LAND_RANGE,
        base_games=200,
        max_games=800,
        optimize_for=metric,
        mixed_combos=mixed,
    )
    t0 = time.perf_counter()
    ranked = opt.run(final_sims=FINAL_SIMS, final_top_k=5)
    elapsed = time.perf_counter() - t0
    return name, elapsed, ranked, opt.total_sims


def _summarize(
    name: str, elapsed: float, ranked, total_sims: int, metric: str,
) -> None:
    print(f"\n  [{name}]  {elapsed:6.2f}s   sims={total_sims}   top {len(ranked)}:")
    for i, (cfg, result) in enumerate(ranked):
        score = _score(result, metric)
        mark = " *MIXED*" if _is_mixed_distinct(cfg) else ""
        rank_note = ""
        if "opt_baseline_rank" in result:
            rank_note = f"  [baseline rank={result['opt_baseline_rank']}]"
        print(
            f"    {i+1}. {cfg.describe():45s}  "
            f"score={score:6.3f}{mark}{rank_note}"
        )


def benchmark_deck(deck_name: str, metric: str) -> dict[str, Any]:
    print(f"\n{'=' * 78}\n  DECK: {deck_name}   METRIC: {metric}\n{'=' * 78}")
    cards = load_decklist(deck_name)

    base = _run("BASELINE   ", mixed=False, cards=cards, metric=metric)
    mixed = _run("MIXED-COMBO", mixed=True, cards=cards, metric=metric)

    _summarize(*base, metric=metric)
    _summarize(*mixed, metric=metric)

    base_top_cfg, base_top_res = base[2][0]
    mixed_top_cfg, mixed_top_res = mixed[2][0]
    base_top_score = _score(base_top_res, metric)
    mixed_top_score = _score(mixed_top_res, metric)

    sim_delta = mixed[3] - base[3]
    sim_pct = (sim_delta / base[3] * 100) if base[3] else 0.0

    print(f"\n  Budget delta: +{sim_delta} sims  ({sim_pct:+.1f}%)")
    print(
        f"  Top-1 score: baseline={base_top_score:.3f}  "
        f"mixed={mixed_top_score:.3f}  "
        f"delta={mixed_top_score - base_top_score:+.3f}"
    )
    if _is_mixed_distinct(mixed_top_cfg):
        print(f"  >>> Mixed mode top-1 is a MIXED-only config: {mixed_top_cfg.describe()}")
    else:
        print(f"  Mixed mode top-1 reuses a non-mixed config: {mixed_top_cfg.describe()}")
    return {
        "deck": deck_name,
        "metric": metric,
        "base_sims": base[3],
        "mixed_sims": mixed[3],
        "sim_delta": sim_delta,
        "sim_pct": sim_pct,
        "base_top_score": base_top_score,
        "mixed_top_score": mixed_top_score,
        "score_delta": mixed_top_score - base_top_score,
        "mixed_top_is_distinct": _is_mixed_distinct(mixed_top_cfg),
    }


if __name__ == "__main__":
    print("Benchmark: FactoredOptimizer mixed_combos=False vs True")
    print(f"  turns={TURNS}, seed={SEED}, land_range={LAND_RANGE}")
    print(f"  max_draw={MAX_DRAW}, max_ramp={MAX_RAMP}")
    print(f"  final_sims={FINAL_SIMS}, metrics={METRICS}")

    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        for deck in DEMO_DECKS:
            summary.append(benchmark_deck(deck, metric))

    print(f"\n{'=' * 78}\n  SUMMARY\n{'=' * 78}")
    print(
        f"{'deck':28s}{'metric':22s}"
        f"{'sims_base':>11s}{'sims_mix':>10s}{'delta':>9s}"
        f"{'score_d':>10s}{'mixed_won':>12s}"
    )
    for row in summary:
        print(
            f"{row['deck']:28s}{row['metric']:22s}"
            f"{row['base_sims']:>11d}{row['mixed_sims']:>10d}"
            f"{row['sim_pct']:>+8.1f}%"
            f"{row['score_delta']:>+10.3f}"
            f"{'YES' if row['mixed_top_is_distinct'] else 'no':>12s}"
        )
