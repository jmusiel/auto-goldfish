"""Verify the on-the-fly calibration provider end-to-end.

Builds a fresh sqlite DB, runs simulations on every cached deck, and
records each deck's raw composites + default-anchored score. After all
sims complete, queries :func:`compute_anchors_from_db` to obtain the
empirical anchors, then re-scores every deck with those active anchors
and prints how scores shifted.

This script is the demo for "what does calibration *do* to scores?".
Output should make it obvious whether the empirical signal is moving
scores in a sane direction or pathologically compressing/inflating them.

Usage:
    .venv/bin/python scripts/verify_db_calibration.py
    .venv/bin/python scripts/verify_db_calibration.py --turns 10 --sims 500
    .venv/bin/python scripts/verify_db_calibration.py --decks deck-a,deck-b
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from auto_goldfish.db.persistence import (
    get_or_create_deck,
    save_simulation_run,
)
from auto_goldfish.db.session import get_session, init_db
from auto_goldfish.decklist.loader import get_deckpath, load_decklist
from auto_goldfish.engine.goldfisher import Goldfisher, SimulationResult
from auto_goldfish.metrics.calibration import (
    compute_anchors_from_db,
    reset_cache,
)
from auto_goldfish.metrics.deck_score import (
    DEFAULT_ANCHORS,
    DeckRawStats,
    compute_raw_stats,
    score_from_raw,
)
from auto_goldfish.metrics.reporter import result_to_dict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(PROJECT_ROOT, "decks")
STAT_KEYS = ["consistency", "acceleration", "snowball", "toughness", "efficiency", "reach"]


@dataclass
class DeckRunRecord:
    """One deck's contribution to the calibration evaluation."""

    name: str
    raw: DeckRawStats
    default_score: Dict[str, int]
    calibrated_score: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Discovery: walk decks/ for cached entries
# ---------------------------------------------------------------------------

def _discover_cached_decks() -> List[str]:
    if not os.path.isdir(CACHE_ROOT):
        return []
    out: List[str] = []
    for entry in sorted(os.listdir(CACHE_ROOT)):
        if entry.startswith("__"):
            continue
        json_path = os.path.join(CACHE_ROOT, entry, f"{entry}.json")
        if os.path.isfile(json_path):
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Per-deck simulation + persistence
# ---------------------------------------------------------------------------

def _simulate(deck_name: str, *, turns: int, sims: int, seed: int) -> SimulationResult:
    cards = load_decklist(deck_name)
    gf = Goldfisher(
        cards,
        turns=turns,
        sims=sims,
        seed=seed,
        workers=1,
        record_results="quartile",
    )
    return gf.simulate()


def _persist(deck_name: str, result: SimulationResult, *, turns: int, sims: int) -> Dict[str, Any]:
    """Persist via the real save_simulation_run path. Returns the result dict."""
    result_dict = result_to_dict(result, turns=turns)
    config = {
        "turns": turns,
        "sims": sims,
        "min_lands": int(result.land_count),
        "max_lands": int(result.land_count),
    }
    with get_session() as session:
        deck = get_or_create_deck(session, deck_name)
        save_simulation_run(
            session, f"verify-{deck_name}", deck, config, [result_dict]
        )
    return result_dict


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt_anchor(name: str, default: tuple, calibrated: tuple) -> str:
    return (
        f"  {name:<26}"
        f"  default=({default[0]:7.3f}, {default[1]:7.3f})"
        f"   calibrated=({calibrated[0]:7.3f}, {calibrated[1]:7.3f})"
    )


def _print_anchor_diff(default, calibrated) -> None:
    print("\n" + "=" * 78)
    print("  ANCHORS  (raw_min, raw_max) -- default vs DB-calibrated")
    print("=" * 78)
    print(_fmt_anchor("consistency", default.consistency, calibrated.consistency))
    print(_fmt_anchor("acceleration", default.acceleration, calibrated.acceleration))
    print(_fmt_anchor("snowball_ratio", default.snowball_ratio, calibrated.snowball_ratio))
    print(_fmt_anchor("snowball_late_avg_norm",
                      default.snowball_late_avg_norm,
                      calibrated.snowball_late_avg_norm))
    print(_fmt_anchor("toughness", default.toughness, calibrated.toughness))
    print(_fmt_anchor("efficiency", default.efficiency, calibrated.efficiency))
    print(_fmt_anchor("reach_norm", default.reach_norm, calibrated.reach_norm))


def _print_score_table(records: List[DeckRunRecord]) -> None:
    print("\n" + "=" * 110)
    print("  PER-DECK SCORE SHIFTS  (default | calibrated | delta)")
    print("=" * 110)
    head = f"  {'deck':<32}" + "  ".join(
        f"{k[:5]:>15}" for k in STAT_KEYS
    )
    print(head)
    print("  " + "-" * (32 + 17 * len(STAT_KEYS)))
    for rec in records:
        cells = []
        for k in STAT_KEYS:
            d = rec.default_score[k]
            c = rec.calibrated_score[k] if rec.calibrated_score else d
            delta = c - d
            sign = "+" if delta > 0 else ""
            cells.append(f"{d}|{c}|{sign}{delta}".rjust(15))
        print(f"  {rec.name[:32]:<32}" + "  ".join(cells))


def _print_aggregate_shifts(records: List[DeckRunRecord]) -> None:
    print("\n" + "=" * 78)
    print("  AGGREGATE SHIFT  (mean signed delta per stat across decks)")
    print("=" * 78)
    n = len(records)
    if n == 0:
        return
    for stat in STAT_KEYS:
        deltas = [
            (rec.calibrated_score[stat] - rec.default_score[stat])
            for rec in records
            if rec.calibrated_score is not None
        ]
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        max_pos = max(deltas)
        max_neg = min(deltas)
        print(
            f"  {stat:<14}  mean Δ {mean:+.2f}   "
            f"max +{max_pos}   max {max_neg}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--sims", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decks", type=str, default=None,
        help="Comma-separated subset of deck names (default: all cached decks).",
    )
    args = parser.parse_args()

    # Use a temp DB so we never touch the user's persistent store. Each
    # invocation calibrates from scratch against the deck pool below.
    db_path = os.path.join(tempfile.gettempdir(), f"verify_db_calibration_{os.getpid()}.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db(f"sqlite:///{db_path}")
    print(f"Using fresh sqlite DB: {db_path}")

    # Calibration uses the cached scoring path, so reset the in-process
    # cache between runs to avoid leaking state across invocations.
    reset_cache()

    deck_names = _discover_cached_decks()
    if args.decks:
        wanted = {d.strip() for d in args.decks.split(",") if d.strip()}
        deck_names = [d for d in deck_names if d in wanted]

    if not deck_names:
        print("No cached decks found under decks/. Aborting.")
        return 1

    print(f"\nRunning {len(deck_names)} cached decks  (turns={args.turns}, sims={args.sims})\n")
    records: List[DeckRunRecord] = []
    t_start = time.perf_counter()

    for i, name in enumerate(deck_names, start=1):
        try:
            t0 = time.perf_counter()
            result = _simulate(name, turns=args.turns, sims=args.sims, seed=args.seed)
            raw = compute_raw_stats(result, args.turns)
            default_score = score_from_raw(raw, DEFAULT_ANCHORS).as_dict()
            _persist(name, result, turns=args.turns, sims=args.sims)
            elapsed = time.perf_counter() - t0
            records.append(DeckRunRecord(
                name=name, raw=raw, default_score=default_score,
            ))
            print(f"  [{i:>3}/{len(deck_names)}] {name:<48}  ({elapsed:5.1f}s)")
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")

    if not records:
        print("\nNo decks ran successfully.")
        return 1

    # Re-score every deck against the now-calibrated anchors.
    with get_session() as session:
        anchors, meta = compute_anchors_from_db(session)
    for rec in records:
        rec.calibrated_score = score_from_raw(rec.raw, anchors).as_dict()

    total = time.perf_counter() - t_start
    print(f"\nDone in {total:.1f}s.  n_rows={meta.n_rows}, n_decks={meta.n_decks}\n")

    _print_anchor_diff(DEFAULT_ANCHORS, anchors)
    _print_score_table(records)
    _print_aggregate_shifts(records)

    return 0


if __name__ == "__main__":
    sys.exit(main())
