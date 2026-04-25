"""Calibrate the 1-10 ranges of the six DeckScore stats from real decks.

For each calibration deck, this script:
  1. Runs a Goldfisher simulation.
  2. Extracts the *raw* inputs that feed each of the six CASTER stat
     formulas (Consistency, Acceleration, Snowball, Toughness, Efficiency,
     Reach).
  3. Computes the *scaled* 1-10 score using the current bounds.
  4. Writes one CSV row with both raw and scaled values plus deck metadata.

After all decks run, it prints a per-stat summary (min / p10 / p50 / p90 /
max for both raw and scaled values) so we can see:
  - Which stats compress real decks into a narrow band (i.e. wasted
    resolution on the 1-10 scale).
  - Which raw bounds are too generous or too tight.
  - Whether anchor decks (synthetic low-end demos) actually score 1-3
    while typical decks land in 4-7.

Usage:
    .venv/bin/python scripts/calibrate_stat_ranges.py
    .venv/bin/python scripts/calibrate_stat_ranges.py --turns 10 --sims 1000
    .venv/bin/python scripts/calibrate_stat_ranges.py --skip-fetch  # cached only

Output:
    calibration_<timestamp>.csv  in the working directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from auto_goldfish.decklist.archidekt import list_user_decks
from auto_goldfish.decklist.loader import get_deckpath, load_decklist
from auto_goldfish.engine.goldfisher import Goldfisher, SimulationResult
from auto_goldfish.metrics.deck_score import compute_deck_score
from auto_goldfish.optimization.benchmark_decks import (
    BENCHMARK_DECKS,
    BenchmarkDeck,
    get_benchmark_deck_dicts,
)


# ---------------------------------------------------------------------------
# Calibration deck list
# ---------------------------------------------------------------------------
# We pull every complete (size==100) commander deck from a small set of
# Archidekt users and treat them as one undifferentiated pool. Tiers are
# *not* assigned a priori -- the goal is to look at the empirical
# distribution of raw metrics and decide where the [1, 10] anchors should
# sit. Synthetic low-end demo decks are included for sanity-check only.

CALIBRATION_USERS: List[str] = ["codudeol", "Tagazok"]


@dataclass(frozen=True)
class CalibrationDeck:
    name: str          # cache key under decks/<name>/<name>.json
    archetype: str     # known archetype for benchmark decks; "unknown" otherwise
    tier: str          # "synthetic" | "real"
    source: str        # "synthetic" or Archidekt username
    archidekt_url: Optional[str] = None


# Synthetic low-end anchors (already cached in decks/).
SYNTHETIC_DECKS: List[CalibrationDeck] = [
    CalibrationDeck("mana-starved-demo", "synthetic", "synthetic", "synthetic"),
    CalibrationDeck("overlanded-cantrips-demo", "synthetic", "synthetic", "synthetic"),
    CalibrationDeck("equilibrium-demo", "synthetic", "synthetic", "synthetic"),
]


# Map known Archidekt deck IDs to existing cached deck names so we reuse
# the already-fetched JSON files for the 14 benchmark decks instead of
# saving them again under a different name.
def _known_deck_id_to_name() -> Dict[int, Tuple[str, str]]:
    out: Dict[int, Tuple[str, str]] = {}
    for d in BENCHMARK_DECKS:
        try:
            deck_id = int(d.archidekt_url.rstrip("/").split("/")[-2])
        except (ValueError, IndexError):
            continue
        out[deck_id] = (d.name, d.archetype)
    return out


KNOWN_DECK_IDS: Dict[int, Tuple[str, str]] = _known_deck_id_to_name()


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    s = _SLUG_RE.sub("_", name.lower()).strip("_")
    return s or "deck"


def _user_deck_to_calibration(entry: Dict[str, Any], username: str) -> CalibrationDeck:
    deck_id = entry["id"]
    if deck_id in KNOWN_DECK_IDS:
        cache_name, archetype = KNOWN_DECK_IDS[deck_id]
    else:
        cache_name = f"{_slugify(entry['name'])}_{deck_id}"
        archetype = "unknown"
    slug = _slugify(entry["name"])
    url = f"https://archidekt.com/decks/{deck_id}/{slug}"
    return CalibrationDeck(
        name=cache_name,
        archetype=archetype,
        tier="real",
        source=username,
        archidekt_url=url,
    )


def discover_decks(
    users: List[str],
    skip_fetch: bool,
    include_synthetic: bool = True,
    verbose: bool = True,
) -> List[CalibrationDeck]:
    """Build the calibration deck list.

    With ``skip_fetch=True`` no network calls are made: we walk decks/ for
    cached entries and tag them ``source="cached"``.
    Otherwise we hit the Archidekt API once per user to list their public
    100-card commander decks.
    """
    decks: List[CalibrationDeck] = []
    if include_synthetic:
        decks.extend(SYNTHETIC_DECKS)

    if skip_fetch:
        # Walk decks/ and add anything cached that isn't already in decks.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_root = os.path.join(project_root, "decks")
        existing_names = {d.name for d in decks}
        if os.path.isdir(cache_root):
            for entry in sorted(os.listdir(cache_root)):
                if entry in existing_names or entry.startswith("__"):
                    continue
                json_path = os.path.join(cache_root, entry, f"{entry}.json")
                if not os.path.isfile(json_path):
                    continue
                # Tag with the matching benchmark archetype if known.
                bench_match = next(
                    (b for b in BENCHMARK_DECKS if b.name == entry), None
                )
                archetype = bench_match.archetype if bench_match else "unknown"
                decks.append(CalibrationDeck(
                    name=entry,
                    archetype=archetype,
                    tier="real",
                    source="cached",
                ))
        return decks

    for username in users:
        if verbose:
            print(f"Listing decks for user {username!r}...", flush=True)
        try:
            entries = list_user_decks(username)
        except Exception as exc:
            print(f"  [warn] could not list {username!r}: {exc}")
            continue
        if verbose:
            print(f"  found {len(entries)} complete commander decks")
        for entry in entries:
            decks.append(_user_deck_to_calibration(entry, username))

    # De-dupe by cache name in case the same deck appears in multiple users.
    seen: set = set()
    unique: List[CalibrationDeck] = []
    for d in decks:
        if d.name in seen:
            continue
        seen.add(d.name)
        unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Raw-metric extraction
# ---------------------------------------------------------------------------
# These mirror the formulas in deck_score.py. We re-derive the *raw* values
# (the input to the final _scale call) so we can see the un-clipped
# distribution and decide where the [1, 10] anchors should sit.

def _raw_acceleration(result: SimulationResult, turns: int) -> float:
    early_turns = min(4, turns, len(result.mean_mana_per_turn))
    if early_turns == 0:
        return 0.0
    return float(sum(result.mean_mana_per_turn[:early_turns]))


def _raw_reach(result: SimulationResult, turns: int) -> float:
    # Already turn-normalized for cross-deck comparison.
    turn_factor = turns / 10.0
    raw = 0.4 * result.mean_mana + 0.6 * result.ceiling_mana
    return float(raw / turn_factor) if turn_factor > 0 else float(raw)


def _raw_consistency(result: SimulationResult, turns: int) -> Tuple[float, float, float, float]:
    """Returns (composite, tail, bad_turns_score, std_score)."""
    tail_score = float(result.consistency)
    max_bad = max(turns * 0.6, 1)
    bad_score = max(0.0, 1.0 - result.mean_bad_turns / max_bad)
    if result.mean_mana > 0:
        cv = result.std_mana / result.mean_mana
        std_score = max(0.0, 1.0 - cv / 0.5)
    else:
        std_score = 0.0
    composite = 0.4 * tail_score + 0.3 * bad_score + 0.3 * std_score
    return composite, tail_score, bad_score, std_score


def _raw_toughness(result: SimulationResult) -> Tuple[float, float, float, float, float]:
    """Returns (composite, mana_norm, draw_norm, early_norm, curve_norm)."""
    mana_norm = min(result.mana_source_count / 45.0, 1.0)
    draw_norm = min(result.draw_count / 15.0, 1.0)
    early_norm = min(result.early_count / 30.0, 1.0)
    curve_norm = max(0.0, 1.0 - max(0.0, result.avg_cmc - 3.0) / 3.0)
    composite = 0.4 * mana_norm + 0.3 * draw_norm + 0.2 * early_norm + 0.1 * curve_norm
    return composite, mana_norm, draw_norm, early_norm, curve_norm


def _raw_efficiency(result: SimulationResult, turns: int) -> Tuple[float, float, float]:
    """Returns (composite, utilization, mid_score)."""
    land_count = result.mean_lands
    theoretical_max = sum(min(i + 1, land_count) for i in range(turns))
    if theoretical_max <= 0:
        return 0.0, 0.0, 0.0
    utilization = min(result.mean_mana / theoretical_max, 1.0)
    max_mid = max(turns * 0.7, 1)
    mid_score = max(0.0, 1.0 - result.mean_mid_turns / max_mid)
    composite = 0.6 * utilization + 0.4 * mid_score
    return composite, utilization, mid_score


def _raw_snowball(result: SimulationResult, turns: int) -> Tuple[float, float, float]:
    """Returns (acceleration_ratio, late_avg_normalized, early_avg)."""
    mpt = result.mean_mana_per_turn
    if len(mpt) < 4:
        return 1.0, 0.0, 0.0
    early_end = min(4, len(mpt))
    early_avg = sum(mpt[:early_end]) / early_end
    late_turns = mpt[early_end:]
    if not late_turns or early_avg <= 0:
        return 1.0, 0.0, float(early_avg)
    late_avg = sum(late_turns) / len(late_turns)
    acceleration = late_avg / early_avg
    turn_factor = turns / 10.0
    late_normalized = late_avg / turn_factor if turn_factor > 0 else late_avg
    return float(acceleration), float(late_normalized), float(early_avg)


# ---------------------------------------------------------------------------
# Per-deck calibration row
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "deck", "source", "archetype", "tier", "lands", "mean_mana", "ceiling_mana",
    "std_mana", "mull_rate", "mean_bad_turns", "mean_mid_turns",
    # Structural snapshot (feeds Toughness)
    "mana_source_count", "draw_count", "early_count", "avg_cmc",
    # Raw inputs (un-clipped) for each stat
    "raw_acceleration", "raw_reach",
    "raw_consistency_composite", "raw_consistency_tail",
    "raw_consistency_bad", "raw_consistency_std",
    "raw_toughness_composite", "raw_toughness_mana_norm",
    "raw_toughness_draw_norm", "raw_toughness_early_norm",
    "raw_toughness_curve_norm",
    "raw_efficiency_composite", "raw_efficiency_utilization",
    "raw_efficiency_mid",
    "raw_snowball_acceleration", "raw_snowball_late_avg_norm",
    # Scaled 1-10 scores (CASTER)
    "consistency", "acceleration", "snowball", "toughness", "efficiency", "reach",
]


def build_row(
    deck: CalibrationDeck, result: SimulationResult, turns: int
) -> Dict[str, Any]:
    raw_cons, c_tail, c_bad, c_std = _raw_consistency(result, turns)
    raw_tough, t_mana, t_draw, t_early, t_curve = _raw_toughness(result)
    raw_eff, e_util, e_mid = _raw_efficiency(result, turns)
    raw_snowball_accel, raw_snowball_late, _ = _raw_snowball(result, turns)
    score = compute_deck_score(result, turns=turns)

    return {
        "deck": deck.name,
        "source": deck.source,
        "archetype": deck.archetype,
        "tier": deck.tier,
        "lands": round(result.mean_lands, 2),
        "mean_mana": round(result.mean_mana, 2),
        "ceiling_mana": round(result.ceiling_mana, 2),
        "std_mana": round(result.std_mana, 2),
        "mull_rate": round(result.mull_rate, 4),
        "mean_bad_turns": round(result.mean_bad_turns, 3),
        "mean_mid_turns": round(result.mean_mid_turns, 3),
        "mana_source_count": result.mana_source_count,
        "draw_count": result.draw_count,
        "early_count": result.early_count,
        "avg_cmc": round(result.avg_cmc, 2),
        "raw_acceleration": round(_raw_acceleration(result, turns), 3),
        "raw_reach": round(_raw_reach(result, turns), 3),
        "raw_consistency_composite": round(raw_cons, 4),
        "raw_consistency_tail": round(c_tail, 4),
        "raw_consistency_bad": round(c_bad, 4),
        "raw_consistency_std": round(c_std, 4),
        "raw_toughness_composite": round(raw_tough, 4),
        "raw_toughness_mana_norm": round(t_mana, 4),
        "raw_toughness_draw_norm": round(t_draw, 4),
        "raw_toughness_early_norm": round(t_early, 4),
        "raw_toughness_curve_norm": round(t_curve, 4),
        "raw_efficiency_composite": round(raw_eff, 4),
        "raw_efficiency_utilization": round(e_util, 4),
        "raw_efficiency_mid": round(e_mid, 4),
        "raw_snowball_acceleration": round(raw_snowball_accel, 4),
        "raw_snowball_late_avg_norm": round(raw_snowball_late, 4),
        "consistency": score.consistency,
        "acceleration": score.acceleration,
        "snowball": score.snowball,
        "toughness": score.toughness,
        "efficiency": score.efficiency,
        "reach": score.reach,
    }


# ---------------------------------------------------------------------------
# Deck loading
# ---------------------------------------------------------------------------

def load_or_fetch(deck: CalibrationDeck, skip_fetch: bool, verbose: bool) -> Optional[List[Dict[str, Any]]]:
    """Return card dicts for *deck*, fetching from Archidekt if needed.

    Returns None if the deck is unavailable (cache miss + skip_fetch, or
    fetch failure).
    """
    cache_path = get_deckpath(deck.name)
    if os.path.isfile(cache_path):
        return load_decklist(deck.name)

    if skip_fetch or deck.archidekt_url is None:
        if verbose:
            print(f"  [skip] {deck.name}: not cached")
        return None

    bench = BenchmarkDeck(
        name=deck.name,
        archidekt_url=deck.archidekt_url,
        archetype=deck.archetype,
        description=f"calibration ({deck.tier})",
    )
    try:
        return get_benchmark_deck_dicts(bench, verbose=verbose)
    except Exception as exc:
        if verbose:
            print(f"  [fail] {deck.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

PERCENTILES = [0, 10, 50, 90, 100]

SCALED_STATS = ["consistency", "acceleration", "snowball", "toughness", "efficiency", "reach"]
RAW_KEYS = [
    ("consistency", "raw_consistency_composite"),
    ("acceleration", "raw_acceleration"),
    ("snowball", "raw_snowball_acceleration"),
    ("toughness", "raw_toughness_composite"),
    ("efficiency", "raw_efficiency_composite"),
    ("reach", "raw_reach"),
]


def percentiles(values: List[float]) -> Dict[int, float]:
    if not values:
        return {p: float("nan") for p in PERCENTILES}
    arr = np.asarray(values, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in PERCENTILES}


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("\nNo rows to summarize.")
        return

    print("\n" + "=" * 78)
    print("  RAW METRICS — distribution across calibration decks")
    print("=" * 78)
    header = f"  {'stat':<14}{'min':>10}{'p10':>10}{'p50':>10}{'p90':>10}{'max':>10}"
    print(header)
    print("  " + "-" * 64)
    for stat, key in RAW_KEYS:
        vals = [r[key] for r in rows]
        pct = percentiles(vals)
        print(
            f"  {stat:<14}"
            f"{pct[0]:>10.3f}{pct[10]:>10.3f}{pct[50]:>10.3f}"
            f"{pct[90]:>10.3f}{pct[100]:>10.3f}"
        )

    print("\n" + "=" * 78)
    print("  SCALED 1-10 SCORES — distribution across calibration decks")
    print("=" * 78)
    print(header)
    print("  " + "-" * 64)
    for stat in SCALED_STATS:
        vals = [r[stat] for r in rows]
        pct = percentiles(vals)
        print(
            f"  {stat:<14}"
            f"{pct[0]:>10.1f}{pct[10]:>10.1f}{pct[50]:>10.1f}"
            f"{pct[90]:>10.1f}{pct[100]:>10.1f}"
        )

    # Real-deck-only summary (excludes synthetic anchors). Tiers are *not*
    # predetermined; the empirical distribution below is what calibration
    # should target.
    real_rows = [r for r in rows if r["tier"] == "real"]
    if real_rows and len(real_rows) != len(rows):
        print("\n" + "=" * 78)
        print(f"  REAL-DECK POOL ONLY  (n={len(real_rows)})")
        print("=" * 78)
        print(header)
        print("  " + "-" * 64)
        for stat in SCALED_STATS:
            vals = [r[stat] for r in real_rows]
            pct = percentiles(vals)
            print(
                f"  {stat:<14}"
                f"{pct[0]:>10.1f}{pct[10]:>10.1f}{pct[50]:>10.1f}"
                f"{pct[90]:>10.1f}{pct[100]:>10.1f}"
            )

    # Per-source breakdown for traceability (which user contributed what).
    sources = sorted({r["source"] for r in rows})
    if len(sources) > 1:
        print("\n" + "=" * 78)
        print("  Median scaled score by source")
        print("=" * 78)
        src_header = f"  {'source':<18}" + "".join(f"{s:>14}" for s in SCALED_STATS)
        print(src_header)
        print("  " + "-" * (18 + 14 * len(SCALED_STATS)))
        for src in sources:
            src_rows = [r for r in rows if r["source"] == src]
            n = len(src_rows)
            line = f"  {src + f' (n={n})':<18}"
            for stat in SCALED_STATS:
                med = float(np.median([r[stat] for r in src_rows]))
                line += f"{med:>14.1f}"
            print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--sims", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Don't fetch from Archidekt; use only cached decks.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="CSV output path. Defaults to calibration_<timestamp>.csv.",
    )
    parser.add_argument(
        "--decks",
        type=str,
        default=None,
        help="Comma-separated subset of deck names to run (default: all).",
    )
    parser.add_argument(
        "--users",
        type=str,
        default=None,
        help=(
            "Comma-separated Archidekt usernames whose 100-card commander "
            f"decks should be pulled. Default: {','.join(CALIBRATION_USERS)}."
        ),
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Exclude the synthetic low-end demo decks from the pool.",
    )
    args = parser.parse_args()

    out_path = args.out or f"calibration_{datetime.now():%Y%m%d_%H%M%S}.csv"

    users = (
        [u.strip() for u in args.users.split(",") if u.strip()]
        if args.users else list(CALIBRATION_USERS)
    )

    decks = discover_decks(
        users=users,
        skip_fetch=args.skip_fetch,
        include_synthetic=not args.no_synthetic,
    )
    if args.decks:
        wanted = {d.strip() for d in args.decks.split(",") if d.strip()}
        decks = [d for d in decks if d.name in wanted]
        if not decks:
            print("No matching decks. Check --decks against listed names.")
            return 1

    print(f"Calibrating {len(decks)} decks  (turns={args.turns}, sims={args.sims})")
    print(f"Output:  {out_path}\n")

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for i, deck in enumerate(decks, start=1):
        print(f"[{i}/{len(decks)}] {deck.name}  ({deck.tier}, {deck.archetype})")
        cards = load_or_fetch(deck, skip_fetch=args.skip_fetch, verbose=True)
        if cards is None:
            continue

        gf = Goldfisher(
            cards,
            turns=args.turns,
            sims=args.sims,
            seed=args.seed,
            workers=args.workers,
            record_results="quartile",
        )
        t0 = time.perf_counter()
        result = gf.simulate()
        elapsed = time.perf_counter() - t0
        row = build_row(deck, result, args.turns)
        rows.append(row)
        print(
            f"  -> C={row['consistency']} A={row['acceleration']} "
            f"S={row['snowball']} T={row['toughness']} "
            f"E={row['efficiency']} R={row['reach']}  "
            f"({elapsed:.1f}s)"
        )

    if not rows:
        print("\nNo decks ran successfully.")
        return 1

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total = time.perf_counter() - t_start
    print(f"\nWrote {len(rows)} rows to {out_path}  (total {total:.1f}s)")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
