"""On-the-fly calibration of CASTER stat anchors from persisted raw values.

Reads ``raw_*`` columns off ``simulation_results`` rows that correspond to
each run's optimal land count, computes ``(p10, p90)`` per stat, and blends
those empirical bounds with :data:`DEFAULT_ANCHORS` via Bayesian
shrinkage. The blend keeps cold-start sane: with zero rows the result is
exactly the defaults; the empirical signal smoothly takes over as more
decks accumulate.

``snowball_late_avg_norm`` is not persisted, so it always falls through to
the default anchor.

For runtime use, :func:`get_active_anchors` wraps the heavy DB query in a
row-count-keyed cache and applies an env-var toggle. Calibration is
**enabled by default**; set ``AUTO_GOLDFISH_CALIBRATE=0`` to fall back to
defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from auto_goldfish.metrics.deck_score import DEFAULT_ANCHORS, StatAnchors

# sqlalchemy and auto_goldfish.db.models are intentionally imported lazily
# inside the functions that need them: this module is imported on the
# scoring path (via reporter.result_to_dict), which also runs in Pyodide
# where sqlalchemy is not installed. The runtime entry point
# get_active_anchors() short-circuits to defaults before any DB code runs.

logger = logging.getLogger(__name__)

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
    session: "Session",
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
    from sqlalchemy import select

    from auto_goldfish.db.models import SimulationResultRow, SimulationRunRow

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


# ---------------------------------------------------------------------------
# Runtime cached provider
# ---------------------------------------------------------------------------

_ENV_TOGGLE = "AUTO_GOLDFISH_CALIBRATE"

# Single-slot cache keyed by total simulation_results row count. New rows
# (the only normal mutation) bump the count and invalidate naturally.
_cache: dict = {"row_count": None, "result": None}


def _is_enabled() -> bool:
    """Calibration is on by default; ``AUTO_GOLDFISH_CALIBRATE=0`` disables."""
    raw = os.environ.get(_ENV_TOGGLE, "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def _row_count(session: "Session") -> int:
    from sqlalchemy import func, select

    from auto_goldfish.db.models import SimulationResultRow

    return int(session.execute(
        select(func.count()).select_from(SimulationResultRow)
    ).scalar_one())


def reset_cache() -> None:
    """Drop the cached anchors. Mainly for tests; runtime uses row-count
    invalidation automatically."""
    _cache["row_count"] = None
    _cache["result"] = None


def get_active_anchors(
    session: "Optional[Session]" = None,
) -> Tuple[StatAnchors, Optional[CalibrationMetadata]]:
    """Return the active anchors for live scoring.

    Returns calibrated anchors when:
      * the toggle env var is not set to a falsy value (default: enabled),
      * a usable DB session can be obtained, and
      * the underlying calibration query succeeds.

    Falls back to ``(DEFAULT_ANCHORS, None)`` otherwise -- never raises, so
    a calibration failure cannot break scoring.

    A session may be passed in (server-side flows that already hold one);
    otherwise a fresh session is opened via :func:`get_session`.
    """
    if not _is_enabled():
        return DEFAULT_ANCHORS, None

    if session is not None:
        return _from_session(session)

    try:
        # Local import: db.session imports calibration's siblings, so keep
        # this lazy to avoid pulling the DB layer at module import.
        from auto_goldfish.db.session import get_session
    except Exception:
        return DEFAULT_ANCHORS, None

    try:
        with get_session() as managed:
            return _from_session(managed)
    except Exception:
        # Includes "DB not initialized" (pyodide / CLI) and any query error.
        logger.debug("Calibration unavailable; using defaults", exc_info=True)
        return DEFAULT_ANCHORS, None


def _from_session(session: "Session") -> Tuple[StatAnchors, Optional[CalibrationMetadata]]:
    """Cached read against a live session."""
    try:
        count = _row_count(session)
    except Exception:
        logger.debug("Row-count probe failed; using defaults", exc_info=True)
        return DEFAULT_ANCHORS, None

    if _cache["row_count"] == count and _cache["result"] is not None:
        return _cache["result"]

    try:
        anchors, meta = compute_anchors_from_db(session)
    except Exception:
        logger.exception("Calibration query failed; using defaults")
        return DEFAULT_ANCHORS, None

    result = (anchors, meta)
    _cache["row_count"] = count
    _cache["result"] = result
    return result
