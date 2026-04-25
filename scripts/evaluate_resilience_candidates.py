"""Phase 1 evaluation of candidate Toughness metrics.

(Originally written to evaluate Resilience replacements; the winning
candidate -- the structural composite -- has since been adopted as the
Toughness stat in deck_score.py.)

For each calibration deck this script:
  1. Runs the normal goldfish simulation.
  2. Runs a *disrupted* simulation where 5 random non-land, non-commander
     cards are replaced with cmc=99 dummies (Tier-1 disruption blanking).
  3. Reads structural counts from the decklist via the effect registry
     (mana sources, ramp cards, draw cards, low-cost cards).
  4. Computes several candidate Resilience metrics.

Output:
  - resilience_candidates_<timestamp>.csv with one row per deck.
  - Console report: per-candidate p5/p50/p95 spread + Pearson correlation
    with raw_consistency_composite (and with each other).

The goal is to pick a candidate with: WIDE spread (p10-p90 covers >=20% of
its full range) AND LOW correlation with consistency (|r| < ~0.6).
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse helpers from the calibration script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_stat_ranges import (  # noqa: E402
    CalibrationDeck,
    discover_decks,
    load_or_fetch,
)

from auto_goldfish.effects.card_database import DEFAULT_REGISTRY  # noqa: E402
from auto_goldfish.engine.goldfisher import Goldfisher, SimulationResult  # noqa: E402
from auto_goldfish.metrics.deck_score import compute_deck_score  # noqa: E402


# ---------------------------------------------------------------------------
# Tier-1 disruption blanking
# ---------------------------------------------------------------------------

BLANK_NAME = "_BLANK_DISRUPTED_"


def _make_blank_card() -> Dict[str, Any]:
    """A drawable card that can never be cast and never produces effects.

    cmc=99 is unreachable in a 10-turn goldfish so the card stays in hand.
    """
    return {
        "name": BLANK_NAME,
        "quantity": 1,
        "oracle_cmc": 99,
        "cmc": 99,
        "cost": "{99}",
        "text": "",
        "sub_types": [],
        "super_types": [],
        "types": ["Sorcery"],
        "identity": [],
        "default_category": None,
        "user_category": None,
        "commander": False,
    }


def blank_random_spells(
    cards: List[Dict[str, Any]], n: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """Return a deep copy of *cards* with *n* random non-land, non-commander
    cards replaced by blanks. Skips cards already classified as lands so
    the deck's mana base is preserved.
    """
    out = copy.deepcopy(cards)
    eligible = [
        i for i, c in enumerate(out)
        if not c.get("commander")
        and "Land" not in (c.get("types") or [])
    ]
    if len(eligible) < n:
        return out
    targets = rng.sample(eligible, n)
    for idx in targets:
        out[idx] = _make_blank_card()
    return out


# ---------------------------------------------------------------------------
# Structural metrics from the decklist
# ---------------------------------------------------------------------------

@dataclass
class StructuralStats:
    deck_size: int
    land_count: int
    ramp_count: int  # registered ramp cards (non-land mana sources)
    draw_count: int  # registered draw cards
    early_count: int  # cards with cmc <= 3 (not lands)
    mana_source_count: int  # lands + ramp
    avg_cmc: float


def compute_structural(cards: List[Dict[str, Any]]) -> StructuralStats:
    n = len(cards)
    lands = 0
    ramp = 0
    draw = 0
    early = 0
    total_cmc = 0.0
    non_land_cmc_count = 0
    for card in cards:
        types = card.get("types") or []
        is_land = "Land" in types
        if is_land:
            lands += 1
            continue
        cmc = card.get("cmc") or card.get("oracle_cmc") or 0
        if isinstance(cmc, (int, float)):
            if cmc <= 3:
                early += 1
            total_cmc += float(cmc)
            non_land_cmc_count += 1
        effects = DEFAULT_REGISTRY.get(card.get("name", ""))
        if effects is not None:
            if effects.ramp:
                ramp += 1
            if effects.draw:
                draw += 1
    avg_cmc = total_cmc / non_land_cmc_count if non_land_cmc_count > 0 else 0.0
    return StructuralStats(
        deck_size=n,
        land_count=lands,
        ramp_count=ramp,
        draw_count=draw,
        early_count=early,
        mana_source_count=lands + ramp,
        avg_cmc=avg_cmc,
    )


# ---------------------------------------------------------------------------
# Candidate metrics
# ---------------------------------------------------------------------------

def candidate_current(result: SimulationResult) -> float:
    """The existing formula's composite (raw, before [0.3, 1.0] -> [1, 10])."""
    if result.mean_mana == 0:
        return 0.5
    mull_ratio = min(result.mean_mana_with_mull / result.mean_mana, 1.0)
    mull_rate_score = max(0.0, 1.0 - result.mull_rate / 0.5)
    return 0.6 * mull_ratio + 0.4 * mull_rate_score


def candidate_p10_floor(result: SimulationResult) -> float:
    """Bottom-decile mana ratio: p10 / mean. Higher = better floor."""
    if result.mean_mana <= 0:
        return 0.0
    return result.percentile_10 / result.mean_mana


def candidate_p10_vs_theoretical(result: SimulationResult, turns: int) -> float:
    """p10 mana spent vs theoretical max for the deck's land count."""
    land_count = result.mean_lands
    theoretical_max = sum(min(i + 1, land_count) for i in range(turns))
    if theoretical_max <= 0:
        return 0.0
    return min(result.percentile_10 / theoretical_max, 1.0)


def candidate_floor_plus_dig(result: SimulationResult) -> float:
    """0.5 * (p25/mean) + 0.5 * (mean_mana_draw/mean)."""
    if result.mean_mana <= 0:
        return 0.0
    floor = result.percentile_25 / result.mean_mana
    dig = result.mean_mana_draw / result.mean_mana if result.mean_mana > 0 else 0.0
    return 0.5 * floor + 0.5 * min(dig, 1.0)


def candidate_structural(stats: StructuralStats) -> float:
    """Decklist-based redundancy composite, normalized to roughly [0, 1].

    Mana sources scaled around 40 (typical EDH baseline), draw around 12,
    early plays around 25. Combine with weights so each component
    contributes independently.
    """
    mana_norm = min(stats.mana_source_count / 45.0, 1.0)
    draw_norm = min(stats.draw_count / 15.0, 1.0)
    early_norm = min(stats.early_count / 30.0, 1.0)
    # Penalize high-curve decks slightly: avg_cmc 5+ is heavy.
    curve_norm = max(0.0, 1.0 - max(0.0, stats.avg_cmc - 3.0) / 3.0)
    return 0.4 * mana_norm + 0.3 * draw_norm + 0.2 * early_norm + 0.1 * curve_norm


def candidate_disruption_ratio(
    normal: SimulationResult, disrupted: SimulationResult
) -> float:
    """Disrupted/normal mean-mana ratio. Higher = handles disruption better."""
    if normal.mean_mana <= 0:
        return 0.0
    return min(disrupted.mean_mana / normal.mean_mana, 1.0)


def candidate_disruption_floor(
    normal: SimulationResult, disrupted: SimulationResult
) -> float:
    """Bottom-decile preservation under disruption."""
    if normal.percentile_10 <= 0:
        return 0.0
    return min(disrupted.percentile_10 / normal.percentile_10, 1.5)


# ---------------------------------------------------------------------------
# Per-deck row build
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "deck", "source", "tier",
    "mean_mana", "p10", "p25", "p90", "mean_mana_draw", "mean_lands",
    "deck_size", "land_count", "ramp_count", "draw_count", "early_count",
    "mana_sources", "avg_cmc",
    "mean_mana_disrupted", "p10_disrupted",
    # Candidates
    "cand_current", "cand_p10_floor", "cand_p10_vs_theoretical",
    "cand_floor_plus_dig", "cand_structural",
    "cand_disruption_ratio", "cand_disruption_floor",
    # For correlation analysis
    "raw_consistency_composite",
]


def _raw_consistency(result: SimulationResult, turns: int) -> float:
    tail_score = float(result.consistency)
    max_bad = max(turns * 0.6, 1)
    bad_score = max(0.0, 1.0 - result.mean_bad_turns / max_bad)
    if result.mean_mana > 0:
        cv = result.std_mana / result.mean_mana
        std_score = max(0.0, 1.0 - cv / 0.5)
    else:
        std_score = 0.0
    return 0.4 * tail_score + 0.3 * bad_score + 0.3 * std_score


def build_row(
    deck: CalibrationDeck,
    cards: List[Dict[str, Any]],
    normal: SimulationResult,
    disrupted: SimulationResult,
    turns: int,
) -> Dict[str, Any]:
    structural = compute_structural(cards)
    return {
        "deck": deck.name,
        "source": deck.source,
        "tier": deck.tier,
        "mean_mana": round(normal.mean_mana, 3),
        "p10": round(normal.percentile_10, 3),
        "p25": round(normal.percentile_25, 3),
        "p90": round(normal.percentile_90, 3),
        "mean_mana_draw": round(normal.mean_mana_draw, 3),
        "mean_lands": round(normal.mean_lands, 2),
        "deck_size": structural.deck_size,
        "land_count": structural.land_count,
        "ramp_count": structural.ramp_count,
        "draw_count": structural.draw_count,
        "early_count": structural.early_count,
        "mana_sources": structural.mana_source_count,
        "avg_cmc": round(structural.avg_cmc, 2),
        "mean_mana_disrupted": round(disrupted.mean_mana, 3),
        "p10_disrupted": round(disrupted.percentile_10, 3),
        "cand_current": round(candidate_current(normal), 4),
        "cand_p10_floor": round(candidate_p10_floor(normal), 4),
        "cand_p10_vs_theoretical": round(
            candidate_p10_vs_theoretical(normal, turns), 4
        ),
        "cand_floor_plus_dig": round(candidate_floor_plus_dig(normal), 4),
        "cand_structural": round(candidate_structural(structural), 4),
        "cand_disruption_ratio": round(
            candidate_disruption_ratio(normal, disrupted), 4
        ),
        "cand_disruption_floor": round(
            candidate_disruption_floor(normal, disrupted), 4
        ),
        "raw_consistency_composite": round(_raw_consistency(normal, turns), 4),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CANDIDATES = [
    "cand_current",
    "cand_p10_floor",
    "cand_p10_vs_theoretical",
    "cand_floor_plus_dig",
    "cand_structural",
    "cand_disruption_ratio",
    "cand_disruption_floor",
]


def _pct(values: List[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p))


def _pearson(a: List[float], b: List[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.std() == 0 or arr_b.std() == 0:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def print_report(rows: List[Dict[str, Any]]) -> None:
    real = [r for r in rows if r["tier"] == "real"]
    if not real:
        print("\nNo real-deck rows; skipping report.")
        return

    print("\n" + "=" * 78)
    print(f"  CANDIDATE SPREAD  (real-deck pool, n={len(real)})")
    print("=" * 78)
    head = (
        f"  {'candidate':<28}{'min':>9}{'p5':>9}{'p50':>9}{'p95':>9}"
        f"{'max':>9}{'spread':>9}"
    )
    print(head)
    print("  " + "-" * 73)
    for cand in CANDIDATES:
        vals = [r[cand] for r in real]
        lo, p5, med, p95, hi = (
            min(vals), _pct(vals, 5), _pct(vals, 50), _pct(vals, 95), max(vals)
        )
        spread = p95 - p5
        print(
            f"  {cand:<28}{lo:>9.3f}{p5:>9.3f}{med:>9.3f}{p95:>9.3f}"
            f"{hi:>9.3f}{spread:>9.3f}"
        )

    print("\n" + "=" * 78)
    print("  CORRELATION with raw_consistency_composite (we want |r| < 0.6)")
    print("=" * 78)
    cons_vals = [r["raw_consistency_composite"] for r in real]
    for cand in CANDIDATES:
        cand_vals = [r[cand] for r in real]
        r_val = _pearson(cand_vals, cons_vals)
        flag = "  <- low overlap" if abs(r_val) < 0.6 else ("  <- high overlap" if abs(r_val) > 0.8 else "")
        print(f"  {cand:<28} r = {r_val:+.3f}{flag}")

    print("\n" + "=" * 78)
    print("  CORRELATION matrix between candidates")
    print("=" * 78)
    short = [c.replace("cand_", "") for c in CANDIDATES]
    print("  " + " " * 22 + "".join(f"{s[:10]:>11}" for s in short))
    for cand_a, short_a in zip(CANDIDATES, short):
        a_vals = [r[cand_a] for r in real]
        line = f"  {short_a[:20]:<22}"
        for cand_b in CANDIDATES:
            b_vals = [r[cand_b] for r in real]
            line += f"{_pearson(a_vals, b_vals):>+11.2f}"
        print(line)

    # Synthetic anchors as a floor sanity check.
    synth = [r for r in rows if r["tier"] == "synthetic"]
    if synth:
        print("\n" + "=" * 78)
        print("  Synthetic anchors (should land low for a good candidate)")
        print("=" * 78)
        head2 = f"  {'deck':<28}" + "".join(f"{c.replace('cand_',''):>22}" for c in CANDIDATES)
        print(head2)
        for r in synth:
            line = f"  {r['deck'][:26]:<28}"
            for c in CANDIDATES:
                line += f"{r[c]:>22.3f}"
            print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_deck(
    deck: CalibrationDeck,
    cards: List[Dict[str, Any]],
    turns: int,
    sims: int,
    seed: int,
    workers: int,
    blank_n: int,
    rng: random.Random,
) -> Tuple[SimulationResult, SimulationResult]:
    gf_normal = Goldfisher(
        cards, turns=turns, sims=sims, seed=seed,
        workers=workers, record_results="quartile",
    )
    normal = gf_normal.simulate()

    blanked = blank_random_spells(cards, blank_n, rng)
    gf_dis = Goldfisher(
        blanked, turns=turns, sims=sims, seed=seed,
        workers=workers, record_results="quartile",
    )
    disrupted = gf_dis.simulate()
    return normal, disrupted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blank-n", type=int, default=5)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--users", type=str, default=None,
        help="Comma-separated Archidekt usernames; defaults to CALIBRATION_USERS.",
    )
    args = parser.parse_args()

    out_path = args.out or f"resilience_candidates_{datetime.now():%Y%m%d_%H%M%S}.csv"

    from calibrate_stat_ranges import CALIBRATION_USERS
    users = (
        [u.strip() for u in args.users.split(",") if u.strip()]
        if args.users else list(CALIBRATION_USERS)
    )

    decks = discover_decks(
        users=users,
        skip_fetch=args.skip_fetch,
        include_synthetic=not args.no_synthetic,
    )
    print(f"Evaluating {len(decks)} decks  (sims={args.sims}, blank_n={args.blank_n})")
    print(f"Output:  {out_path}\n")

    rng = random.Random(args.seed + 1)  # different stream from sim seed
    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for i, deck in enumerate(decks, start=1):
        print(f"[{i}/{len(decks)}] {deck.name}  ({deck.tier}, {deck.source})", flush=True)
        cards = load_or_fetch(deck, skip_fetch=args.skip_fetch, verbose=False)
        if cards is None:
            continue
        try:
            normal, disrupted = run_deck(
                deck, cards,
                turns=args.turns, sims=args.sims, seed=args.seed,
                workers=args.workers, blank_n=args.blank_n, rng=rng,
            )
        except Exception as exc:
            print(f"  [fail] {deck.name}: {exc}")
            continue
        row = build_row(deck, cards, normal, disrupted, turns=args.turns)
        rows.append(row)
        print(
            f"  current={row['cand_current']:.3f}  "
            f"p10_floor={row['cand_p10_floor']:.3f}  "
            f"struct={row['cand_structural']:.3f}  "
            f"disrupt={row['cand_disruption_ratio']:.3f}",
            flush=True,
        )

    if not rows:
        print("\nNo decks ran.")
        return 1

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    elapsed = time.perf_counter() - t_start
    print(f"\nWrote {len(rows)} rows to {out_path}  (total {elapsed:.1f}s)")
    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
