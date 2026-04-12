"""D&D-style stat block for Commander decks.

Computes six stats (1--20 scale) from simulation results:

- **Speed**: How quickly the deck deploys mana in early turns.
- **Power**: Peak mana output and ceiling performance.
- **Consistency**: How rarely the deck has terrible games.
- **Resilience**: How well the deck recovers from mulligans and bad starts.
- **Efficiency**: How well the deck uses available mana each turn.
- **Momentum**: How well the deck sustains and accelerates output over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from auto_goldfish.engine.goldfisher import SimulationResult


@dataclass
class DeckScore:
    """Six-stat profile for a deck, each on a 1--20 scale."""

    speed: int
    power: int
    consistency: int
    resilience: int
    efficiency: int
    momentum: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "speed": self.speed,
            "power": self.power,
            "consistency": self.consistency,
            "resilience": self.resilience,
            "efficiency": self.efficiency,
            "momentum": self.momentum,
        }

    def format_block(self) -> str:
        """Return an ASCII stat block suitable for terminal output."""
        bar_width = 20
        lines = []
        lines.append("+" + "-" * 34 + "+")
        lines.append("|       DECK STAT BLOCK            |")
        lines.append("+" + "-" * 34 + "+")
        for name, value in self.as_dict().items():
            filled = "█" * value + "░" * (bar_width - value)
            lines.append(f"| {name.upper():<13} {value:>2} {filled} |")
        lines.append("+" + "-" * 34 + "+")
        return "\n".join(lines)


def _clamp(value: float, lo: float = 1.0, hi: float = 20.0) -> int:
    """Clamp and round a float to an integer in [lo, hi]."""
    return int(max(lo, min(hi, round(value))))


def _scale(raw: float, raw_min: float, raw_max: float) -> int:
    """Linearly map *raw* from [raw_min, raw_max] to [1, 20]."""
    if raw_max <= raw_min:
        return 10
    normalized = (raw - raw_min) / (raw_max - raw_min)
    return _clamp(1 + 19 * normalized)


def compute_deck_score(result: SimulationResult, turns: int = 10) -> DeckScore:
    """Derive a :class:`DeckScore` from simulation results.

    Parameters
    ----------
    result : SimulationResult
        Output from ``Goldfisher.simulate()``.
    turns : int
        Number of turns used in the simulation (needed for bounds calibration).
    """
    speed = _compute_speed(result, turns)
    power = _compute_power(result, turns)
    consistency = _compute_consistency(result, turns)
    resilience = _compute_resilience(result)
    efficiency = _compute_efficiency(result, turns)
    momentum = _compute_momentum(result, turns)

    return DeckScore(
        speed=speed,
        power=power,
        consistency=consistency,
        resilience=resilience,
        efficiency=efficiency,
        momentum=momentum,
    )


# ---------------------------------------------------------------------------
# Individual stat computations
# ---------------------------------------------------------------------------

def _compute_speed(result: SimulationResult, turns: int) -> int:
    """Speed: early-game mana deployment (turns 1-4).

    Measures the average mana spent in the first 4 turns. A deck that
    curves out Sol Ring → Signet → 3-drop → 4-drop would score near 20.

    Bounds: 0 mana (do nothing) to ~14 mana (perfect curve with fast mana).
    """
    early_turns = min(4, turns, len(result.mean_mana_per_turn))
    if early_turns == 0:
        return 1
    early_mana = sum(result.mean_mana_per_turn[:early_turns])
    # Theoretical: turn 1=2, turn 2=3, turn 3=4, turn 4=5 with fast mana = 14
    # Realistic floor: ~1 mana in 4 turns (very slow deck)
    return _scale(early_mana, 1.0, 14.0)


def _compute_power(result: SimulationResult, turns: int) -> int:
    """Power: peak output and ceiling performance.

    Combines mean mana spent with ceiling (top 25%) performance.
    A high-power deck generates lots of mana in its best games.

    Bounds calibrated for 10-turn games. Scales linearly with turn count.
    """
    turn_factor = turns / 10.0
    # Weight: 40% overall mean, 60% ceiling
    raw = 0.4 * result.mean_mana + 0.6 * result.ceiling_mana
    # 10-turn bounds: ~5 (weak deck) to ~45 (powerhouse with lots of ramp)
    return _scale(raw, 5.0 * turn_factor, 45.0 * turn_factor)


def _compute_consistency(result: SimulationResult, turns: int) -> int:
    """Consistency: how rarely the deck bricks.

    Combines the left-tail ratio (bottom 25% vs mean), bad turn count,
    and mana standard deviation into a single consistency score.

    All three sub-scores are on a 0-1 scale and averaged.
    """
    # Left-tail ratio: already 0-1, where 1.0 = perfect consistency
    tail_score = result.consistency

    # Bad turns: 0 bad turns = 1.0, many bad turns = 0.0
    max_bad = max(turns * 0.6, 1)
    bad_score = max(0.0, 1.0 - result.mean_bad_turns / max_bad)

    # Low std dev relative to mean = consistent
    if result.mean_mana > 0:
        cv = result.std_mana / result.mean_mana  # coefficient of variation
        # CV of 0 = perfect, CV of 0.5+ = very inconsistent
        std_score = max(0.0, 1.0 - cv / 0.5)
    else:
        std_score = 0.0

    composite = 0.4 * tail_score + 0.3 * bad_score + 0.3 * std_score
    return _scale(composite, 0.0, 1.0)


def _compute_resilience(result: SimulationResult) -> int:
    """Resilience: how well the deck recovers from mulligans.

    Measures the performance gap between games with and without mulligans.
    A resilient deck performs nearly as well after mulliganing.

    Also factors in the mulligan rate itself -- decks that rarely need
    to mulligan are implicitly resilient.
    """
    if result.mean_mana == 0:
        return 10

    # Mull performance ratio: 1.0 = no penalty, lower = worse
    mull_ratio = result.mean_mana_with_mull / result.mean_mana if result.mean_mana > 0 else 1.0
    mull_ratio = min(mull_ratio, 1.0)  # cap at 1.0

    # Low mulligan rate = good (0% = 1.0, 50%+ = 0.0)
    mull_rate_score = max(0.0, 1.0 - result.mull_rate / 0.5)

    # Weight: 60% recovery quality, 40% mulligan avoidance
    composite = 0.6 * mull_ratio + 0.4 * mull_rate_score
    return _scale(composite, 0.3, 1.0)


def _compute_efficiency(result: SimulationResult, turns: int) -> int:
    """Efficiency: how well the deck uses available mana each turn.

    Approximates mana utilization by comparing mana spent to a
    theoretical maximum based on land count. Also penalizes mid turns
    (turns where the deck underperforms relative to the turn number).
    """
    # Theoretical max mana for a given land count over N turns:
    # sum(min(i+1, land_count) for i in range(turns))
    land_count = result.mean_lands
    theoretical_max = sum(min(i + 1, land_count) for i in range(turns))

    if theoretical_max <= 0:
        return 1

    # Utilization ratio: mana spent / theoretical max
    utilization = result.mean_mana / theoretical_max
    utilization = min(utilization, 1.0)

    # Mid-turn penalty: turns where deck underperformed
    max_mid = max(turns * 0.7, 1)
    mid_score = max(0.0, 1.0 - result.mean_mid_turns / max_mid)

    composite = 0.6 * utilization + 0.4 * mid_score
    return _scale(composite, 0.0, 1.0)


def _compute_momentum(result: SimulationResult, turns: int) -> int:
    """Momentum: how well the deck accelerates over time.

    Measures whether the deck's mana output grows faster than the
    natural land-per-turn baseline, indicating successful ramp and
    card advantage payoff.

    Uses the slope of the mana curve in turns 5-10 (or whatever is
    available) compared to the early turns.
    """
    mpt = result.mean_mana_per_turn
    if len(mpt) < 4:
        return 10

    # Split into early (turns 1-4) and late (turns 5+)
    early_end = min(4, len(mpt))
    early_avg = sum(mpt[:early_end]) / early_end
    late_turns = mpt[early_end:]
    if not late_turns:
        return 10
    late_avg = sum(late_turns) / len(late_turns)

    if early_avg <= 0:
        return 10

    # Acceleration ratio: how much more mana per turn in late game vs early
    # A ratio of 1.0 = flat (no acceleration). 3.0+ = strong ramp payoff.
    acceleration = late_avg / early_avg

    # Also factor in absolute late-game output
    turn_factor = turns / 10.0
    # Late-game mana per turn: ~1 (weak) to ~8 (strong)
    late_power = _scale(late_avg, 1.0 * turn_factor, 8.0 * turn_factor)
    accel_score = _scale(acceleration, 0.5, 4.0)

    # Blend acceleration shape with absolute late-game power
    composite = 0.5 * accel_score + 0.5 * late_power
    return _clamp(composite)
