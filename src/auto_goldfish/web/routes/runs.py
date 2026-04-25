"""Runs route -- view and compare simulation runs by D&D-style deck scores."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template

logger = logging.getLogger(__name__)

bp = Blueprint("runs", __name__, url_prefix="/runs")

STAT_KEYS = ["consistency", "acceleration", "snowball", "toughness", "efficiency", "reach"]


def _load_runs() -> list[dict]:
    """Query all simulation runs with their optimal-land-count D&D scores."""
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import joinedload

        from auto_goldfish.db.models import SimulationRunRow
        from auto_goldfish.db.session import get_session
    except Exception:
        return []

    runs = []
    try:
        with get_session() as session:
            rows = (
                session.execute(
                    select(SimulationRunRow)
                    .options(
                        joinedload(SimulationRunRow.deck),
                        joinedload(SimulationRunRow.results),
                    )
                    .order_by(SimulationRunRow.created_at.desc())
                )
                .unique()
                .scalars()
                .all()
            )

            for run in rows:
                optimal_result = None
                if run.optimal_land_count is not None:
                    optimal_result = next(
                        (r for r in run.results if r.land_count == run.optimal_land_count),
                        None,
                    )

                # Build per-land-count series with scores for charts
                results_series = sorted(
                    [
                        {
                            "land_count": r.land_count,
                            "consistency": r.score_consistency,
                            "acceleration": r.score_acceleration,
                            "snowball": r.score_snowball,
                            "toughness": r.score_toughness,
                            "efficiency": r.score_efficiency,
                            "reach": r.score_reach,
                        }
                        for r in run.results
                    ],
                    key=lambda r: r["land_count"],
                )

                def _score(result, key):
                    val = getattr(result, f"score_{key}", None) if result else None
                    return val

                runs.append({
                    "id": run.id,
                    "job_id": run.job_id,
                    "deck_name": run.deck.name if run.deck else "Unknown",
                    "turns": run.turns,
                    "sims": run.sims,
                    "optimal_land_count": run.optimal_land_count,
                    "created_at": run.created_at.strftime("%Y-%m-%d %H:%M"),
                    **{k: _score(optimal_result, k) for k in STAT_KEYS},
                    "results": results_series,
                })
    except Exception:
        logger.exception("Failed to load simulation runs")

    return runs


_ANCHOR_FIELDS = (
    "consistency",
    "acceleration",
    "snowball_ratio",
    "snowball_late_avg_norm",
    "toughness",
    "efficiency",
    "reach_norm",
)


def _load_calibration_meta() -> dict | None:
    """Return the active calibration metadata + anchors, or None if defaults."""
    try:
        from auto_goldfish.metrics.calibration import get_active_anchors
        from auto_goldfish.metrics.deck_score import DEFAULT_ANCHORS
    except Exception:
        return None
    try:
        active, meta = get_active_anchors()
    except Exception:
        return None
    if meta is None:
        return None
    anchors = [
        {
            "name": field,
            "default": list(getattr(DEFAULT_ANCHORS, field)),
            "active": list(getattr(active, field)),
        }
        for field in _ANCHOR_FIELDS
    ]
    return {
        "n_rows": meta.n_rows,
        "n_decks": meta.n_decks,
        "pseudo_count": meta.pseudo_count,
        "low_pct": meta.low_pct,
        "high_pct": meta.high_pct,
        "anchors": anchors,
    }


@bp.route("/")
def index():
    runs = _load_runs()
    return render_template(
        "runs.html", runs=runs, calibration=_load_calibration_meta(),
    )


@bp.route("/api/data")
def api_data():
    """Return all runs as JSON."""
    runs = _load_runs()
    return jsonify({"runs": runs, "calibration": _load_calibration_meta()})
