"""D&D-style stat block for Commander decks.

Computes six stats (1--10 scale) from simulation results -- the **CASTER**
profile:

- **Consistency**: How rarely the deck has terrible games.
- **Acceleration**: How quickly the deck deploys mana in early turns.
- **Surge**: How well the deck sustains and accelerates output over time.
- **Toughness**: Structural redundancy of the decklist (mana sources,
  card draw, low-cost plays, and a controlled curve). A deck with broad
  redundancy can absorb a missing piece without falling apart.
- **Efficiency**: How well the deck uses available mana each turn.
- **Reach**: Peak mana output and ceiling performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from auto_goldfish.engine.goldfisher import SimulationResult


@dataclass
class DeckScore:
    """Six-stat profile for a deck (CASTER), each on a 1--10 scale."""

    consistency: int
    acceleration: int
    surge: int
    toughness: int
    efficiency: int
    reach: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "consistency": self.consistency,
            "acceleration": self.acceleration,
            "surge": self.surge,
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


def compute_deck_score(result: SimulationResult, turns: int = 10) -> DeckScore:
    """Derive a :class:`DeckScore` from simulation results."""
    return DeckScore(
        consistency=_compute_consistency(result, turns),
        acceleration=_compute_acceleration(result, turns),
        surge=_compute_surge(result, turns),
        toughness=_compute_toughness(result),
        efficiency=_compute_efficiency(result, turns),
        reach=_compute_reach(result, turns),
    )


# ---------------------------------------------------------------------------
# Individual stat computations
# ---------------------------------------------------------------------------

def _compute_consistency(result: SimulationResult, turns: int) -> int:
    """Consistency: how rarely the deck bricks.

    Combines the left-tail ratio (bottom 25% vs mean), bad turn count,
    and mana standard deviation into a single consistency score.
    """
    tail_score = result.consistency

    max_bad = max(turns * 0.6, 1)
    bad_score = max(0.0, 1.0 - result.mean_bad_turns / max_bad)

    if result.mean_mana > 0:
        cv = result.std_mana / result.mean_mana
        std_score = max(0.0, 1.0 - cv / 0.5)
    else:
        std_score = 0.0

    composite = 0.4 * tail_score + 0.3 * bad_score + 0.3 * std_score
    return _scale(composite, 0.0, 1.0)


def _compute_acceleration(result: SimulationResult, turns: int) -> int:
    """Acceleration: early-game mana deployment (turns 1-4).

    Measures the average mana spent in the first 4 turns. A deck that
    curves out Sol Ring -> Signet -> 3-drop -> 4-drop scores near 14.
    """
    early_turns = min(4, turns, len(result.mean_mana_per_turn))
    if early_turns == 0:
        return 1
    early_mana = sum(result.mean_mana_per_turn[:early_turns])
    return _scale(early_mana, 1.0, 14.0)


def _compute_surge(result: SimulationResult, turns: int) -> int:
    """Surge: how well the deck accelerates over time.

    Compares late-game mana per turn to early-game, blended with absolute
    late-game output. Strong ramp payoff yields a higher surge score.
    """
    mpt = result.mean_mana_per_turn
    if len(mpt) < 4:
        return 5

    early_end = min(4, len(mpt))
    early_avg = sum(mpt[:early_end]) / early_end
    late_turns = mpt[early_end:]
    if not late_turns:
        return 5
    late_avg = sum(late_turns) / len(late_turns)

    if early_avg <= 0:
        return 5

    acceleration = late_avg / early_avg

    turn_factor = turns / 10.0
    late_power = _scale(late_avg, 1.0 * turn_factor, 8.0 * turn_factor)
    accel_score = _scale(acceleration, 0.5, 4.0)

    composite = 0.5 * accel_score + 0.5 * late_power
    return _clamp(composite)


def _compute_toughness(result: SimulationResult) -> int:
    """Toughness: structural redundancy of the decklist.

    Composite of:
      - mana sources (lands + ramp), normalized to a 45-source baseline
      - card-draw cards, normalized to a 15-card baseline
      - low-cost (cmc <= 3) plays, normalized to a 30-card baseline
      - a curve term that penalises avg_cmc above 3

    A deck with broad redundancy absorbs a missing piece without
    falling apart -- the structural notion of "toughness" (area under
    the stress--strain curve, by analogy to materials science).
    """
    mana_norm = min(result.mana_source_count / 45.0, 1.0)
    draw_norm = min(result.draw_count / 15.0, 1.0)
    early_norm = min(result.early_count / 30.0, 1.0)
    curve_norm = max(0.0, 1.0 - max(0.0, result.avg_cmc - 3.0) / 3.0)

    composite = 0.4 * mana_norm + 0.3 * draw_norm + 0.2 * early_norm + 0.1 * curve_norm
    # Anchored to a 76-deck Archidekt sample (codudeol + Tagazok + cached
    # benchmarks): real-deck composite spans p10=0.70, p50=0.90, max=1.00.
    # [0.55, 1.00] -> [1, 10] gives full 1-10 spread without saturating
    # the median.
    return _scale(composite, 0.55, 1.00)


def _compute_efficiency(result: SimulationResult, turns: int) -> int:
    """Efficiency: how well the deck uses available mana each turn."""
    land_count = result.mean_lands
    theoretical_max = sum(min(i + 1, land_count) for i in range(turns))

    if theoretical_max <= 0:
        return 1

    utilization = min(result.mean_mana / theoretical_max, 1.0)

    max_mid = max(turns * 0.7, 1)
    mid_score = max(0.0, 1.0 - result.mean_mid_turns / max_mid)

    composite = 0.6 * utilization + 0.4 * mid_score
    return _scale(composite, 0.0, 1.0)


def _compute_reach(result: SimulationResult, turns: int) -> int:
    """Reach: peak output and ceiling performance.

    Combines mean mana spent with ceiling (top 25%) performance.
    A high-reach deck generates lots of mana in its best games.
    """
    turn_factor = turns / 10.0
    raw = 0.4 * result.mean_mana + 0.6 * result.ceiling_mana
    return _scale(raw, 5.0 * turn_factor, 45.0 * turn_factor)
