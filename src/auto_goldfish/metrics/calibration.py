"""On-the-fly calibration of CASTER stat anchors from persisted raw values.

Reads ``raw_*`` columns off ``simulation_results`` rows that correspond to
each run's optimal land count, computes ``(p10, p90)`` per stat, and blends
those empirical bounds with :data:`DEFAULT_ANCHORS` via Bayesian
shrinkage. The blend keeps cold-start sane: with zero rows the result is
exactly the defaults; the empirical signal smoothly takes over as more
decks accumulate.

``snowball_late_avg_norm`` is not persisted, so it always falls through to
the default anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from auto_goldfish.db.models import SimulationResultRow, SimulationRunRow
from auto_goldfish.metrics.deck_score import DEFAULT_ANCHORS, StatAnchors

# Stats whose anchors are calibrated from a single persisted raw column.
# Maps the StatAnchors field name to the SimulationResultRow column name.
_STAT_TO_RAW_COL = {
    "consistency": "raw_consistency",
    "acceleration": "raw_acceleration",
    "snowball_ratio": "raw_snowball",
    "toughness": "raw_toughness",
    "efficiency": "raw_efficiency",
    "reach_norm": "raw_reach",
}


@dataclass(frozen=True)
class CalibrationMetadata:
    """Diagnostic info about an anchor calibration."""

    n_rows: int             # optimal-land rows with non-NULL raw values
    n_decks: int            # distinct decks contributing
    pseudo_count: int       # Bayesian prior weight used
    low_pct: float          # percentile used for raw_min
    high_pct: float         # percentile used for raw_max


def _shrink(
    empirical: Tuple[float, float],
    default: Tuple[float, float],
    n_real: int,
    n_pseudo: int,
) -> Tuple[float, float]:
    """Bayesian-shrink empirical (low, high) toward default via pseudo-counts."""
    total = n_real + n_pseudo
    if total <= 0:
        return default
    w_real = n_real / total
    w_pseudo = n_pseudo / total
    low = w_real * empirical[0] + w_pseudo * default[0]
    high = w_real * empirical[1] + w_pseudo * default[1]
    return float(low), float(high)


def _percentiles(values: Iterable[float], low_pct: float, high_pct: float) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(np.percentile(arr, low_pct)), float(np.percentile(arr, high_pct))


def compute_anchors_from_db(
    session: Session,
    *,
    pseudo_count: int = 76,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
    defaults: StatAnchors = DEFAULT_ANCHORS,
) -> Tuple[StatAnchors, CalibrationMetadata]:
    """Read raw_* values from the DB and produce calibrated anchors.

    Only rows whose ``land_count`` matches their run's ``optimal_land_count``
    are used (one observation per run). Rows with NULL raw values (legacy
    pre-Phase-0 data) are skipped.

    Each stat's anchors are the per-stat ``(p_low, p_high)`` percentiles of
    the empirical distribution, then shrunk toward ``defaults`` by
    ``pseudo_count``. With zero usable rows the result is ``defaults``.
    """
    raw_cols = list(_STAT_TO_RAW_COL.values())

    stmt = (
        select(SimulationRunRow.deck_id, *[getattr(SimulationResultRow, c) for c in raw_cols])
        .join(SimulationResultRow, SimulationResultRow.run_id == SimulationRunRow.id)
        .where(SimulationResultRow.land_count == SimulationRunRow.optimal_land_count)
    )
    rows = session.execute(stmt).all()

    # Filter to rows where every raw_* column is non-NULL: a half-populated
    # row would skew different stats' percentiles by different N.
    usable = [r for r in rows if all(r[i + 1] is not None for i in range(len(raw_cols)))]

    n_rows = len(usable)
    n_decks = len({r[0] for r in usable})

    if n_rows == 0:
        return defaults, CalibrationMetadata(
            n_rows=0, n_decks=0, pseudo_count=pseudo_count,
            low_pct=low_pct, high_pct=high_pct,
        )

    columns = list(zip(*[r[1:] for r in usable]))

    fields: dict[str, Tuple[float, float]] = {}
    for stat_field, col_idx in zip(_STAT_TO_RAW_COL.keys(), range(len(raw_cols))):
        empirical = _percentiles(columns[col_idx], low_pct, high_pct)
        default_anchor = getattr(defaults, stat_field)
        fields[stat_field] = _shrink(empirical, default_anchor, n_rows, pseudo_count)

    anchors = StatAnchors(
        consistency=fields["consistency"],
        acceleration=fields["acceleration"],
        snowball_ratio=fields["snowball_ratio"],
        snowball_late_avg_norm=defaults.snowball_late_avg_norm,
        toughness=fields["toughness"],
        efficiency=fields["efficiency"],
        reach_norm=fields["reach_norm"],
    )

    metadata = CalibrationMetadata(
        n_rows=n_rows,
        n_decks=n_decks,
        pseudo_count=pseudo_count,
        low_pct=low_pct,
        high_pct=high_pct,
    )
    return anchors, metadata
