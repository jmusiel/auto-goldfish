"""D&D-style stat block for Commander decks.

Computes six stats (1--10 scale) from simulation results -- the **CASTER**
profile:

- **Consistency**: How rarely the deck has terrible games.
- **Acceleration**: How quickly the deck deploys mana in early turns.
- **Snowball**: How much advantage compounds over time -- the late-game
  mana curve climbs steeply relative to the early game and lands on a
  high plateau.
- **Toughness**: Structural redundancy of the decklist (mana sources,
  card draw, low-cost plays, and a controlled curve). A deck with broad
  redundancy can absorb a missing piece without falling apart.
- **Efficiency**: How well the deck uses available mana each turn.
- **Reach**: Peak mana output and ceiling performance.

The 1--10 mapping for each stat is governed by a :class:`StatAnchors`
container of (raw_min, raw_max) tuples. Defaults are anchored to a 76-deck
Archidekt sample; callers can pass tuned anchors to ``compute_deck_score``
or ``score_from_raw`` to override (e.g. for on-the-fly DB calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from auto_goldfish.engine.goldfisher import SimulationResult


@dataclass(frozen=True)
class StatAnchors:
    """``(raw_min, raw_max)`` anchors for the 1--10 scaling of each stat.

    Raw values feed ``_scale(raw, raw_min, raw_max)``: at/below ``raw_min``
    they map to 1; at/above ``raw_max`` they map to 10. Anchors marked
    ``_norm`` operate on turn-factor-normalized inputs (raw divided by
    ``turns / 10``) so a single set of anchors works across game lengths.

    Defaults derive from a 76-deck Archidekt calibration pool (see
    ``scripts/calibrate_stat_ranges.py``).
    """

    consistency: Tuple[float, float] = (0.0, 1.0)
    acceleration: Tuple[float, float] = (1.0, 14.0)
    snowball_ratio: Tuple[float, float] = (0.5, 4.0)
    snowball_late_avg_norm: Tuple[float, float] = (1.0, 8.0)
    toughness: Tuple[float, float] = (0.55, 1.00)
    efficiency: Tuple[float, float] = (0.0, 1.0)
    reach_norm: Tuple[float, float] = (5.0, 45.0)


DEFAULT_ANCHORS = StatAnchors()


@dataclass
class DeckRawStats:
    """Unscaled composite values that feed each CASTER stat's _scale call.

    Six floats are persistable to ``simulation_results`` for later
    distribution analysis and on-the-fly anchor calibration. Snowball
    has two underlying inputs (a ratio and a late-game average); only
    the ratio is exposed at the top level here.
    ``snowball_late_avg_norm`` is kept on the dataclass for completeness
    but is not currently persisted.
    """

    consistency: float
    acceleration: float
    snowball: float                # late/early acceleration ratio
    toughness: float
    efficiency: float
    reach: float                   # turn-factor-normalized
    snowball_late_avg_norm: float  # turn-factor-normalized late-game avg

    def as_dict(self) -> Dict[str, float]:
        """Return all raw inputs needed to re-score with new anchors.

        The six top-level CASTER raws plus the secondary snowball input.
        Only the six top-level fields are currently persisted as DB
        columns; the secondary rides along on the per-result dict so
        server-side persist-time re-scoring can recompute snowball
        accurately against active anchors.
        """
        return {
            "consistency": self.consistency,
            "acceleration": self.acceleration,
            "snowball": self.snowball,
            "toughness": self.toughness,
            "efficiency": self.efficiency,
            "reach": self.reach,
            "snowball_late_avg_norm": self.snowball_late_avg_norm,
        }


@dataclass
class DeckScore:
    """Six-stat profile for a deck (CASTER), each on a 1--10 scale."""

    consistency: int
    acceleration: int
    snowball: int
    toughness: int
    efficiency: int
    reach: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "consistency": self.consistency,
            "acceleration": self.acceleration,
            "snowball": self.snowball,
            "toughness": self.toughness,
            "efficiency": self.efficiency,
            "reach": self.reach,
        }

    def format_block(self) -> str:
        """Return an ASCII stat block suitable for terminal output."""
        bar_width = 10
        lines = []
        lines.append("+" + "-" * 26 + "+")
        lines.append("|     DECK STAT BLOCK      |")
        lines.append("+" + "-" * 26 + "+")
        for name, value in self.as_dict().items():
            filled = "█" * value + "░" * (bar_width - value)
            lines.append(f"| {name.upper():<13} {value:>2} {filled} |")
        lines.append("+" + "-" * 26 + "+")
        return "\n".join(lines)


def _clamp(value: float, lo: float = 1.0, hi: float = 10.0) -> int:
    """Clamp and round a float to an integer in [lo, hi]."""
    return int(max(lo, min(hi, round(value))))


def _scale(raw: float, raw_min: float, raw_max: float) -> int:
    """Linearly map *raw* from [raw_min, raw_max] to [1, 10]."""
    if raw_max <= raw_min:
        return 5
    normalized = (raw - raw_min) / (raw_max - raw_min)
    return _clamp(1 + 9 * normalized)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_deck_score(
    result: SimulationResult,
    turns: int = 10,
    anchors: StatAnchors = DEFAULT_ANCHORS,
) -> DeckScore:
    """Derive a :class:`DeckScore` from simulation results."""
    raw = compute_raw_stats(result, turns)
    return score_from_raw(raw, anchors)


def compute_raw_stats(result: SimulationResult, turns: int = 10) -> DeckRawStats:
    """Compute the six raw composite values that feed the 1--10 scaling.

    Independent of any anchor choice -- callers can later apply different
    :class:`StatAnchors` via :func:`score_from_raw` without re-simulating.
    """
    return DeckRawStats(
        consistency=_raw_consistency(result, turns),
        acceleration=_raw_acceleration(result, turns),
        snowball=_raw_snowball_ratio(result),
        snowball_late_avg_norm=_raw_snowball_late_avg_norm(result, turns),
        toughness=_raw_toughness(result),
        efficiency=_raw_efficiency(result, turns),
        reach=_raw_reach_norm(result, turns),
    )


def score_from_raw(raw: DeckRawStats, anchors: StatAnchors = DEFAULT_ANCHORS) -> DeckScore:
    """Apply 1--10 scaling using the given anchors."""
    if raw.snowball_late_avg_norm <= 0.0:
        # No late-game data (short game or all-mulligan); neutral score.
        snowball = 5
    else:
        accel_score = _scale(raw.snowball, *anchors.snowball_ratio)
        late_power = _scale(raw.snowball_late_avg_norm, *anchors.snowball_late_avg_norm)
        snowball = _clamp(0.5 * accel_score + 0.5 * late_power)
    return DeckScore(
        consistency=_scale(raw.consistency, *anchors.consistency),
        acceleration=_scale(raw.acceleration, *anchors.acceleration),
        snowball=snowball,
        toughness=_scale(raw.toughness, *anchors.toughness),
        efficiency=_scale(raw.efficiency, *anchors.efficiency),
        reach=_scale(raw.reach, *anchors.reach_norm),
    )


# ---------------------------------------------------------------------------
# Raw composite values (no scaling, no anchors)
# ---------------------------------------------------------------------------

def _raw_consistency(result: SimulationResult, turns: int) -> float:
    """Composite of left-tail ratio, bad-turn count, and mana CV."""
    tail_score = result.consistency
    max_bad = max(turns * 0.6, 1)
    bad_score = max(0.0, 1.0 - result.mean_bad_turns / max_bad)
    if result.mean_mana > 0:
        cv = result.std_mana / result.mean_mana
        std_score = max(0.0, 1.0 - cv / 0.5)
    else:
        std_score = 0.0
    return 0.4 * tail_score + 0.3 * bad_score + 0.3 * std_score


def _raw_acceleration(result: SimulationResult, turns: int) -> float:
    """Sum of mean mana spent across the first 4 turns."""
    early_turns = min(4, turns, len(result.mean_mana_per_turn))
    if early_turns == 0:
        return 0.0
    return float(sum(result.mean_mana_per_turn[:early_turns]))


def _raw_snowball_ratio(result: SimulationResult) -> float:
    """Late-game / early-game mean-mana ratio (the acceleration term)."""
    mpt = result.mean_mana_per_turn
    if len(mpt) < 4:
        return 1.0
    early_end = min(4, len(mpt))
    early_avg = sum(mpt[:early_end]) / early_end
    late_turns = mpt[early_end:]
    if not late_turns or early_avg <= 0:
        return 1.0
    late_avg = sum(late_turns) / len(late_turns)
    return float(late_avg / early_avg)


def _raw_snowball_late_avg_norm(result: SimulationResult, turns: int) -> float:
    """Mean mana per turn for late-game turns, normalized to a 10-turn game."""
    mpt = result.mean_mana_per_turn
    if len(mpt) < 4:
        return 0.0
    early_end = min(4, len(mpt))
    late_turns = mpt[early_end:]
    if not late_turns:
        return 0.0
    late_avg = sum(late_turns) / len(late_turns)
    turn_factor = turns / 10.0
    return float(late_avg / turn_factor) if turn_factor > 0 else float(late_avg)


def _raw_toughness(result: SimulationResult) -> float:
    """Structural-redundancy composite of the decklist."""
    mana_norm = min(result.mana_source_count / 45.0, 1.0)
    draw_norm = min(result.draw_count / 15.0, 1.0)
    early_norm = min(result.early_count / 30.0, 1.0)
    curve_norm = max(0.0, 1.0 - max(0.0, result.avg_cmc - 3.0) / 3.0)
    return 0.4 * mana_norm + 0.3 * draw_norm + 0.2 * early_norm + 0.1 * curve_norm


def _raw_efficiency(result: SimulationResult, turns: int) -> float:
    """Composite of mana utilization and avoided mid-game stalls."""
    land_count = result.mean_lands
    theoretical_max = sum(min(i + 1, land_count) for i in range(turns))
    if theoretical_max <= 0:
        return 0.0
    utilization = min(result.mean_mana / theoretical_max, 1.0)
    max_mid = max(turns * 0.7, 1)
    mid_score = max(0.0, 1.0 - result.mean_mid_turns / max_mid)
    return 0.6 * utilization + 0.4 * mid_score


def _raw_reach_norm(result: SimulationResult, turns: int) -> float:
    """Weighted blend of mean and ceiling mana, normalized to a 10-turn game."""
    raw = 0.4 * result.mean_mana + 0.6 * result.ceiling_mana
    turn_factor = turns / 10.0
    return float(raw / turn_factor) if turn_factor > 0 else float(raw)


# ---------------------------------------------------------------------------
# Per-stat scaled scores (kept for backward-compatible imports / tests)
# ---------------------------------------------------------------------------

def _compute_consistency(result: SimulationResult, turns: int) -> int:
    return _scale(_raw_consistency(result, turns), *DEFAULT_ANCHORS.consistency)


def _compute_acceleration(result: SimulationResult, turns: int) -> int:
    early_turns = min(4, turns, len(result.mean_mana_per_turn))
    if early_turns == 0:
        return 1
    return _scale(_raw_acceleration(result, turns), *DEFAULT_ANCHORS.acceleration)


def _compute_snowball(result: SimulationResult, turns: int) -> int:
    mpt = result.mean_mana_per_turn
    if len(mpt) < 4:
        return 5
    early_end = min(4, len(mpt))
    early_avg = sum(mpt[:early_end]) / early_end
    late_turns = mpt[early_end:]
    if not late_turns:
        return 5
    if early_avg <= 0:
        return 5

    accel_score = _scale(_raw_snowball_ratio(result), *DEFAULT_ANCHORS.snowball_ratio)
    late_power = _scale(
        _raw_snowball_late_avg_norm(result, turns), *DEFAULT_ANCHORS.snowball_late_avg_norm
    )
    return _clamp(0.5 * accel_score + 0.5 * late_power)


def _compute_toughness(result: SimulationResult) -> int:
    return _scale(_raw_toughness(result), *DEFAULT_ANCHORS.toughness)


def _compute_efficiency(result: SimulationResult, turns: int) -> int:
    land_count = result.mean_lands
    theoretical_max = sum(min(i + 1, land_count) for i in range(turns))
    if theoretical_max <= 0:
        return 1
    return _scale(_raw_efficiency(result, turns), *DEFAULT_ANCHORS.efficiency)


def _compute_reach(result: SimulationResult, turns: int) -> int:
    return _scale(_raw_reach_norm(result, turns), *DEFAULT_ANCHORS.reach_norm)
