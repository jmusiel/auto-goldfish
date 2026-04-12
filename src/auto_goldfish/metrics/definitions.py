"""Built-in metric functions."""

from __future__ import annotations

from typing import List

import numpy as np

from .collector import GameRecord


def mean_mana_spent(records: List[GameRecord]) -> float:
    return float(np.mean([r.total_mana_spent for r in records]))


def mean_mana_value(records: List[GameRecord]) -> float:
    return float(np.mean([r.mana_value for r in records]))


def mean_mana_draw(records: List[GameRecord]) -> float:
    return float(np.mean([r.mana_draw for r in records]))


def mean_mana_ramp(records: List[GameRecord]) -> float:
    return float(np.mean([r.mana_ramp for r in records]))


def mean_mana_total(records: List[GameRecord]) -> float:
    return float(np.mean([r.mana_value + r.mana_draw + r.mana_ramp for r in records]))


def mean_ecms(records: List[GameRecord]) -> float:
    return float(np.mean([r.ecms for r in records]))


def mean_hand_sum(records: List[GameRecord]) -> float:
    return float(np.mean([r.hand_sum for r in records]))


def mean_lands_played(records: List[GameRecord]) -> float:
    return float(np.mean([r.lands_played for r in records]))


def mean_mulligans(records: List[GameRecord]) -> float:
    return float(np.mean([r.mulligans for r in records]))


def mean_draws(records: List[GameRecord]) -> float:
    return float(np.mean([r.draws for r in records]))


def mean_bad_turns(records: List[GameRecord]) -> float:
    return float(np.mean([r.bad_turns for r in records]))


def mean_mid_turns(records: List[GameRecord]) -> float:
    return float(np.mean([r.mid_turns for r in records]))


def consistency(records: List[GameRecord], threshold: float = 0.25) -> float:
    """Left-tail ratio consistency metric.

    Computes the ratio of the mean mana spent in the worst-performing
    fraction of games (defined by *threshold*) to the overall mean.
    Higher values (closer to 1.0) indicate that bad games are not
    significantly worse than the average game.

    Parameters
    ----------
    records : list[GameRecord]
        Per-game simulation records.
    threshold : float
        Fraction of games (sorted ascending) to treat as the "bad tail".
        Default 0.25 (bottom quartile).

    Returns
    -------
    float
        ``mean(bottom threshold games) / mean(all games)``.
        Range is 0.0 – 1.0 where 1.0 means perfect consistency.
    """
    mana_list = [r.total_mana_spent for r in records]
    n = len(mana_list)
    sorted_mana = sorted(mana_list)
    cutoff = max(1, int(n * threshold))
    tail_mean = float(np.mean(sorted_mana[:cutoff]))
    overall_mean = float(np.mean(mana_list))
    if overall_mean == 0:
        return 1.0
    return tail_mean / overall_mean
