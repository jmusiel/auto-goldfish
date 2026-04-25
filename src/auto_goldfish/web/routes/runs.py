"""Runs route -- view and compare simulation runs by D&D-style deck scores."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template

logger = logging.getLogger(__name__)

bp = Blueprint("runs", __name__, url_prefix="/runs")

STAT_KEYS = ["speed", "power", "consistency", "resilience", "efficiency", "momentum"]


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
                            "speed": r.score_speed,
                            "power": r.score_power,
                            "consistency": r.score_consistency,
                            "resilience": r.score_resilience,
                            "efficiency": r.score_efficiency,
                            "momentum": r.score_momentum,
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


@bp.route("/")
def index():
    runs = _load_runs()
    return render_template("runs.html", runs=runs)


@bp.route("/api/data")
def api_data():
    """Return all runs as JSON."""
    runs = _load_runs()
    return jsonify(runs)
