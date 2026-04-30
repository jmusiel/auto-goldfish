"""Runs route -- view and compare simulation runs by D&D-style deck scores."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from flask import Blueprint, jsonify, render_template, request

logger = logging.getLogger(__name__)

bp = Blueprint("runs", __name__, url_prefix="/runs")

STAT_KEYS = ["consistency", "acceleration", "snowball", "toughness", "efficiency", "reach"]
DEFAULT_LIMIT = 20


def _reconstruct_deck_list(
    session, deck_id: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the deck-list + overrides shape that ``analyze_deck_composition``
    expects, sourced entirely from the DB.

    Returns ``(deck_list, overrides)``. ``deck_list`` mirrors the on-disk
    ``decks/<name>/<name>.json`` format (one dict per card with ``name``,
    ``types``, ``cmc``, ``quantity``, ``commander``). ``overrides`` only
    contains rows flagged ``user_edited`` -- the analyzer treats those as
    authoritative for ramp/draw classification.
    """
    from sqlalchemy import select

    from auto_goldfish.db.models import CardRow, DeckCardRow, EffectLabelRow

    rows = session.execute(
        select(DeckCardRow, CardRow, EffectLabelRow)
        .join(CardRow, DeckCardRow.card_id == CardRow.id)
        .outerjoin(EffectLabelRow, DeckCardRow.label_id == EffectLabelRow.id)
        .where(DeckCardRow.deck_id == deck_id)
    ).all()

    deck_list: list[dict[str, Any]] = []
    overrides: dict[str, Any] = {}
    for deck_card, card, label in rows:
        types: list[str] = []
        if card.types_json:
            try:
                parsed = json.loads(card.types_json)
                if isinstance(parsed, list):
                    types = [str(t) for t in parsed]
            except json.JSONDecodeError:
                pass
        deck_list.append({
            "name": card.name,
            "types": types,
            "cmc": card.cmc if card.cmc is not None else 0,
            "quantity": deck_card.quantity,
            "commander": deck_card.is_commander,
        })
        if deck_card.user_edited and label is not None:
            try:
                overrides[card.name] = json.loads(label.effects_json)
            except json.JSONDecodeError:
                pass

    return deck_list, overrides


def _composition_to_breakdown(comp) -> dict:
    return {
        "commander_names": list(comp.commander_names),
        "land_count": comp.land_count,
        "mdfc_count": comp.mdfc_count,
        "avg_cmc": comp.avg_cmc,
        "cmc_distribution": {str(k): v for k, v in comp.cmc_distribution.items()},
        "ramp_cards": comp.ramp_cards,
        "ramp_by_cmc": {str(k): v for k, v in comp.ramp_by_cmc.items()},
        "draw_cards": comp.draw_cards,
        "draw_breakdown": dict(comp.draw_breakdown),
    }


def _load_deck_breakdown(
    session, deck_id: int, deck_name: str, cache: dict
) -> Optional[dict]:
    """Compute deck composition from the DB and return a JSON-friendly dict.

    ``cache`` is keyed by ``deck_id`` so the same deck loaded by multiple
    runs on a single page render only hits the analyzer once. Returns
    ``None`` when the deck has no DB-persisted cards (legacy run before the
    metadata schema landed) so the template can render a "composition
    unavailable" fallback.
    """
    if deck_id in cache:
        return cache[deck_id]

    try:
        from auto_goldfish.effects.card_database import DEFAULT_REGISTRY
        from auto_goldfish.optimization.deck_analyzer import analyze_deck_composition
    except Exception:
        cache[deck_id] = None
        return None

    breakdown: Optional[dict] = None
    try:
        deck_list, overrides = _reconstruct_deck_list(session, deck_id)
        if deck_list:
            comp = analyze_deck_composition(deck_list, DEFAULT_REGISTRY, overrides)
            breakdown = _composition_to_breakdown(comp)
    except Exception:
        logger.exception("Failed to load deck breakdown for %s", deck_name)
        breakdown = None

    cache[deck_id] = breakdown
    return breakdown


def _load_runs(view: str = "recent", stat: Optional[str] = None, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Query a slice of simulation runs along with deck composition breakdowns.

    ``view`` is one of:
      - ``recent``: most recently created runs (default).
      - ``top``: highest-scoring runs for ``stat`` at the run's optimal land count.
      - ``bottom``: lowest-scoring runs for ``stat``.

    Top/bottom views require ``stat`` to be one of :data:`STAT_KEYS`. They
    join through ``SimulationResultRow`` filtered to the run's optimal land
    count so each run contributes exactly one ranking score.
    """
    try:
        from sqlalchemy import asc, desc, select
        from sqlalchemy.orm import joinedload

        from auto_goldfish.db.models import SimulationResultRow, SimulationRunRow
        from auto_goldfish.db.session import get_session
    except Exception:
        return []

    if view in ("top", "bottom"):
        if stat not in STAT_KEYS:
            view = "recent"
            stat = None

    runs: list[dict] = []
    deck_cache: dict = {}
    try:
        with get_session() as session:
            base_select = (
                select(SimulationRunRow)
                .options(
                    joinedload(SimulationRunRow.deck),
                    joinedload(SimulationRunRow.results),
                )
            )

            if view in ("top", "bottom") and stat is not None:
                score_col = getattr(SimulationResultRow, f"score_{stat}")
                ordering = desc(score_col) if view == "top" else asc(score_col)
                # Join to the result row at the run's optimal land count so
                # each run contributes exactly one ranking score.
                stmt = (
                    base_select.join(
                        SimulationResultRow,
                        (SimulationResultRow.run_id == SimulationRunRow.id)
                        & (SimulationResultRow.land_count == SimulationRunRow.optimal_land_count),
                    )
                    .where(score_col.is_not(None))
                    .order_by(ordering, SimulationRunRow.created_at.desc())
                    .limit(limit)
                )
            else:
                stmt = base_select.order_by(SimulationRunRow.created_at.desc()).limit(limit)

            rows = session.execute(stmt).unique().scalars().all()

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

                deck_name = run.deck.name if run.deck else "Unknown"
                deck_id = run.deck.id if run.deck else None
                breakdown = (
                    _load_deck_breakdown(session, deck_id, deck_name, deck_cache)
                    if deck_id is not None
                    else None
                )
                runs.append({
                    "id": run.id,
                    "job_id": run.job_id,
                    "deck_name": deck_name,
                    "turns": run.turns,
                    "sims": run.sims,
                    "optimal_land_count": run.optimal_land_count,
                    "created_at": run.created_at.strftime("%Y-%m-%d %H:%M"),
                    **{k: _score(optimal_result, k) for k in STAT_KEYS},
                    "results": results_series,
                    "deck_breakdown": breakdown,
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


def load_caster_calibration() -> dict:
    """Return calibration data for the CASTER score explanation.

    Always returns a usable dict (falls back to defaults when calibration is
    unavailable). The shape matches :func:`_load_calibration_meta`, plus a
    boolean ``calibrated`` flag distinguishing live data from defaults.
    """
    try:
        from auto_goldfish.metrics.deck_score import DEFAULT_ANCHORS
    except Exception:
        return {"calibrated": False, "anchors": []}

    meta = _load_calibration_meta()
    if meta is not None:
        meta["calibrated"] = True
        return meta

    return {
        "calibrated": False,
        "n_rows": 0,
        "n_decks": 0,
        "pseudo_count": 0,
        "low_pct": 10.0,
        "high_pct": 90.0,
        "anchors": [
            {
                "name": field,
                "default": list(getattr(DEFAULT_ANCHORS, field)),
                "active": list(getattr(DEFAULT_ANCHORS, field)),
            }
            for field in _ANCHOR_FIELDS
        ],
    }


@bp.route("/")
def index():
    runs = _load_runs()
    return render_template(
        "runs.html", runs=runs, calibration=_load_calibration_meta(),
    )


@bp.route("/api/data")
def api_data():
    """Return a slice of runs as JSON.

    Query params:
      - ``view``: ``recent`` (default), ``top``, or ``bottom``.
      - ``stat``: required for ``top``/``bottom`` -- one of the CASTER stats.

    Invalid combinations (``top``/``bottom`` without a known stat) silently
    fall back to ``recent`` and the response reflects the applied view.
    """
    raw_view = request.args.get("view", "recent")
    raw_stat = request.args.get("stat")
    if raw_view in ("top", "bottom") and raw_stat in STAT_KEYS:
        view, stat = raw_view, raw_stat
    else:
        view, stat = "recent", None
    runs = _load_runs(view=view, stat=stat)
    return jsonify({
        "runs": runs,
        "calibration": _load_calibration_meta(),
        "view": view,
        "stat": stat,
    })
