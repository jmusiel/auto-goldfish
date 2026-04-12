"""Runs route -- view all simulation runs stored in the database."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template

logger = logging.getLogger(__name__)

bp = Blueprint("runs", __name__, url_prefix="/runs")


def _load_runs() -> list[dict]:
    """Query all simulation runs with their optimal-land-count results."""
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import joinedload

        from auto_goldfish.db.models import SimulationResultRow, SimulationRunRow
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
                # Find the result at the optimal land count
                optimal_result = None
                if run.optimal_land_count is not None:
                    optimal_result = next(
                        (r for r in run.results if r.land_count == run.optimal_land_count),
                        None,
                    )

                # Build per-land-count series for charts
                results_series = sorted(
                    [
                        {
                            "land_count": r.land_count,
                            "mean_mana": round(r.mean_mana, 2),
                            "consistency": round(r.consistency, 2),
                            "mean_draws": round(r.mean_draws, 2),
                            "mean_bad_turns": round(r.mean_bad_turns, 2),
                            "mean_lands": round(r.mean_lands, 2),
                            "mean_mulls": round(r.mean_mulls, 2),
                            "mean_spells_cast": round(r.mean_spells_cast, 2),
                            "percentile_25": round(r.percentile_25, 2),
                            "percentile_50": round(r.percentile_50, 2),
                            "percentile_75": round(r.percentile_75, 2),
                        }
                        for r in run.results
                    ],
                    key=lambda r: r["land_count"],
                )

                runs.append({
                    "id": run.id,
                    "job_id": run.job_id,
                    "deck_name": run.deck.name if run.deck else "Unknown",
                    "turns": run.turns,
                    "sims": run.sims,
                    "min_lands": run.min_lands,
                    "max_lands": run.max_lands,
                    "optimal_land_count": run.optimal_land_count,
                    "mulligan_strategy": run.mulligan_strategy,
                    "created_at": run.created_at.strftime("%Y-%m-%d %H:%M"),
                    # Metrics at optimal land count
                    "mean_mana": round(optimal_result.mean_mana, 2) if optimal_result else None,
                    "consistency": round(optimal_result.consistency, 2) if optimal_result else None,
                    "mean_draws": round(optimal_result.mean_draws, 2) if optimal_result else None,
                    "mean_bad_turns": round(optimal_result.mean_bad_turns, 2) if optimal_result else None,
                    "mean_spells_cast": round(optimal_result.mean_spells_cast, 2) if optimal_result else None,
                    "mean_mulls": round(optimal_result.mean_mulls, 2) if optimal_result else None,
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
    """Return all runs as JSON (for potential future AJAX use)."""
    runs = _load_runs()
    return jsonify(runs)
