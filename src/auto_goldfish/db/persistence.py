"""Persistence helpers -- get-or-create patterns and convenience wrappers."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    CardAnnotationRow,
    CardPerformanceRow,
    CardRow,
    DeckCardRow,
    DeckRow,
    EffectLabelRow,
    SimulationResultRow,
    SimulationRunRow,
)
from .session import get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# get-or-create primitives
# ---------------------------------------------------------------------------

def get_or_create_card(session: Session, name: str) -> CardRow:
    row = session.execute(select(CardRow).where(CardRow.name == name)).scalar_one_or_none()
    if row is None:
        row = CardRow(name=name)
        session.add(row)
        session.flush()
    return row


def get_or_create_effect_label(session: Session, effects: dict) -> EffectLabelRow:
    canonical = json.dumps(effects, sort_keys=True)
    row = session.execute(
        select(EffectLabelRow).where(EffectLabelRow.effects_json == canonical)
    ).scalar_one_or_none()
    if row is None:
        row = EffectLabelRow(effects_json=canonical)
        session.add(row)
        session.flush()
    return row


def get_or_create_deck(session: Session, name: str) -> DeckRow:
    row = session.execute(select(DeckRow).where(DeckRow.name == name)).scalar_one_or_none()
    if row is None:
        row = DeckRow(name=name)
        session.add(row)
        session.flush()
    return row


# ---------------------------------------------------------------------------
# Deck card persistence
# ---------------------------------------------------------------------------

def save_deck_cards(
    session: Session,
    deck: DeckRow,
    card_effects_list: List[Dict[str, Any]],
    overrides: Dict[str, Any],
) -> None:
    """Upsert deck cards with their effect labels and override status."""
    for entry in card_effects_list:
        card_name = entry.get("name", "")
        if not card_name:
            continue

        card = get_or_create_card(session, card_name)

        # Determine effect label
        label: Optional[EffectLabelRow] = None
        user_edited = False
        if card_name in overrides and overrides[card_name]:
            label = get_or_create_effect_label(session, overrides[card_name])
            user_edited = True
        elif entry.get("registry_override"):
            label = get_or_create_effect_label(session, entry["registry_override"])

        # Upsert deck_card
        existing = session.execute(
            select(DeckCardRow).where(
                DeckCardRow.deck_id == deck.id,
                DeckCardRow.card_id == card.id,
            )
        ).scalar_one_or_none()

        if existing is None:
            session.add(DeckCardRow(
                deck_id=deck.id,
                card_id=card.id,
                label_id=label.id if label else None,
                user_edited=user_edited,
            ))
        else:
            existing.label_id = label.id if label else None
            existing.user_edited = user_edited

    session.flush()


# ---------------------------------------------------------------------------
# Simulation run persistence
# ---------------------------------------------------------------------------

def save_simulation_run(
    session: Session,
    job_id: str,
    deck: DeckRow,
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> SimulationRunRow:
    """Save a complete simulation run with results and card performance."""

    # Determine optimal land count (highest consistency)
    optimal_land_count = None
    if results:
        best = max(results, key=lambda r: r.get("consistency", 0))
        optimal_land_count = best.get("land_count")

    run = SimulationRunRow(
        job_id=job_id,
        deck_id=deck.id,
        turns=config.get("turns", 8),
        sims=config.get("sims", 1000),
        min_lands=config.get("min_lands", 36),
        max_lands=config.get("max_lands", 39),
        seed=config.get("seed"),
        mulligan_strategy=config.get("mulligan", "default"),
        optimal_land_count=optimal_land_count,
    )
    session.add(run)
    session.flush()

    # The optimizer may return multiple top configs at the same land count
    # (e.g. different added-card combinations). The DB constrains one result
    # per (run, land_count), so keep only the highest-ranked entry per land.
    deduped: Dict[int, Dict[str, Any]] = {}
    for r in results:
        lc = r.get("land_count", 0)
        if lc not in deduped:
            deduped[lc] = r

    # The DB is the canonical source of calibrated scores: re-score every
    # row from its raw_* values using the currently-active anchors. This
    # ensures stored scores are consistent regardless of whether the
    # caller (Pyodide, server, CLI) had DB access at score time.
    from auto_goldfish.metrics.calibration import get_active_anchors
    from auto_goldfish.metrics.deck_score import (
        DeckRawStats,
        score_from_raw,
    )

    active_anchors, _ = get_active_anchors(session)

    # Save per-land-count results
    for r in deduped.values():
        ci_mana = r.get("ci_mean_mana", [0.0, 0.0])
        ci_con = r.get("ci_consistency", [0.0, 0.0])
        deck_score = r.get("deck_score", {})
        deck_raw = r.get("deck_raw", {})
        if all(deck_raw.get(k) is not None for k in (
            "consistency", "acceleration", "snowball",
            "toughness", "efficiency", "reach",
        )):
            raw_obj = DeckRawStats(
                consistency=deck_raw["consistency"],
                acceleration=deck_raw["acceleration"],
                snowball=deck_raw["snowball"],
                toughness=deck_raw["toughness"],
                efficiency=deck_raw["efficiency"],
                reach=deck_raw["reach"],
                # Secondary snowball input ships in the dict (not the DB)
                # so we can re-score snowball against active anchors.
                # Falls back to 0.0 (forces neutral 5) only on legacy dicts
                # that predate this field.
                snowball_late_avg_norm=deck_raw.get("snowball_late_avg_norm", 0.0),
            )
            deck_score = score_from_raw(raw_obj, active_anchors).as_dict()
        session.add(SimulationResultRow(
            run_id=run.id,
            land_count=r.get("land_count", 0),
            mean_mana=r.get("mean_mana", 0.0),
            mean_draws=r.get("mean_draws", 0.0),
            mean_bad_turns=r.get("mean_bad_turns", 0.0),
            mean_lands=r.get("mean_lands", 0.0),
            mean_mulls=r.get("mean_mulls", 0.0),
            mean_spells_cast=r.get("mean_spells_cast", 0.0),
            ci_mean_mana_low=ci_mana[0] if len(ci_mana) > 0 else 0.0,
            ci_mean_mana_high=ci_mana[1] if len(ci_mana) > 1 else 0.0,
            consistency=r.get("consistency", 0.0),
            ci_consistency_low=ci_con[0] if len(ci_con) > 0 else 0.0,
            ci_consistency_high=ci_con[1] if len(ci_con) > 1 else 0.0,
            percentile_25=r.get("percentile_25", 0.0),
            percentile_50=r.get("percentile_50", 0.0),
            percentile_75=r.get("percentile_75", 0.0),
            score_consistency=deck_score.get("consistency"),
            score_acceleration=deck_score.get("acceleration"),
            score_snowball=deck_score.get("snowball"),
            score_toughness=deck_score.get("toughness"),
            score_efficiency=deck_score.get("efficiency"),
            score_reach=deck_score.get("reach"),
            raw_consistency=deck_raw.get("consistency"),
            raw_acceleration=deck_raw.get("acceleration"),
            raw_snowball=deck_raw.get("snowball"),
            raw_toughness=deck_raw.get("toughness"),
            raw_efficiency=deck_raw.get("efficiency"),
            raw_reach=deck_raw.get("reach"),
        ))

    # Save card performance only at optimal land count (bottom 10 with effects)
    if optimal_land_count is not None:
        optimal_result = next(
            (r for r in results if r.get("land_count") == optimal_land_count),
            None,
        )
        if optimal_result:
            perf = optimal_result.get("card_performance", {})
            low_performing = perf.get("low_performing", [])

            # Filter to cards that have a non-empty effects description
            cards_with_effects = [
                c for c in low_performing if c.get("effects", "").strip()
            ]

            for rank, card_data in enumerate(cards_with_effects[:10], start=1):
                card = get_or_create_card(session, card_data["name"])
                session.add(CardPerformanceRow(
                    run_id=run.id,
                    card_id=card.id,
                    mean_with=card_data.get("mean_with", 0.0),
                    mean_without=card_data.get("mean_without", 0.0),
                    score=card_data.get("score", 0.0),
                    rank=rank,
                ))

    session.flush()
    return run


# ---------------------------------------------------------------------------
# Convenience wrappers (handle session internally)
# ---------------------------------------------------------------------------

def save_card_annotation(
    session: Session,
    card_name: str,
    effects_json: str,
    source: str = "human",
    session_id: str | None = None,
) -> CardAnnotationRow:
    """Save a card annotation, upserting within the same wizard session.

    If a row with the same (card_name, session_id) exists, update it.
    Otherwise insert a new row.  This ensures one row per card per
    wizard session, while still accumulating rows across sessions.
    """
    if session_id:
        existing = session.execute(
            select(CardAnnotationRow).where(
                CardAnnotationRow.card_name == card_name,
                CardAnnotationRow.session_id == session_id,
            )
        ).scalar_one_or_none()
        if existing:
            existing.effects_json = effects_json
            existing.source = source
            session.flush()
            return existing

    row = CardAnnotationRow(
        card_name=card_name,
        effects_json=effects_json,
        source=source,
        session_id=session_id,
    )
    session.add(row)
    session.flush()
    return row


def get_annotation_stats(
    session: Session,
    card_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Get annotation statistics for a list of cards.

    Returns dict of card_name -> {
        human_count: int,
        is_controversial: bool,
        has_human: bool,
        latest_effects_json: str | None,
        latest_source: str | None,
    }
    """
    if not card_names:
        return {}

    rows = session.execute(
        select(CardAnnotationRow)
        .where(CardAnnotationRow.card_name.in_(card_names))
        .order_by(CardAnnotationRow.created_at.desc())
    ).scalars().all()

    # Group by card name
    by_card: Dict[str, list] = {}
    for row in rows:
        by_card.setdefault(row.card_name, []).append(row)

    stats: Dict[str, Dict[str, Any]] = {}
    for name in card_names:
        annotations = by_card.get(name, [])
        human_annotations = [a for a in annotations if a.source == "human"]
        human_count = len(human_annotations)

        # Controversial = 2+ distinct effects_json among human annotations with no clear majority
        is_controversial = False
        if human_count >= 2:
            effects_counts: Dict[str, int] = {}
            for a in human_annotations:
                effects_counts[a.effects_json] = effects_counts.get(a.effects_json, 0) + 1
            # No clear majority = no single effects_json has > 50% of votes
            max_count = max(effects_counts.values())
            if max_count <= human_count / 2:
                is_controversial = True

        latest = annotations[0] if annotations else None
        stats[name] = {
            "human_count": human_count,
            "is_controversial": is_controversial,
            "has_human": human_count > 0,
            "latest_effects_json": latest.effects_json if latest else None,
            "latest_source": latest.source if latest else None,
        }

    return stats


def persist_completed_job(job: Any) -> None:
    """Persist a completed SimJob to the database.

    Called from simulation_runner after job.status = "completed".
    """
    with get_session() as session:
        deck = get_or_create_deck(session, job.deck_name)
        save_simulation_run(session, job.job_id, deck, job.config, job.results)
    logger.info("Persisted simulation run %s for deck %s", job.job_id, job.deck_name)


def persist_deck_cards(
    deck_name: str,
    card_effects_list: List[Dict[str, Any]],
    overrides: Dict[str, Any],
) -> None:
    """Persist deck card labels to the database.

    Called from the config route after building card_effects_list.
    """
    with get_session() as session:
        deck = get_or_create_deck(session, deck_name)
        save_deck_cards(session, deck, card_effects_list, overrides)
    logger.info("Persisted deck cards for %s", deck_name)
