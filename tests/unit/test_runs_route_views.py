"""Tests for /runs/api/data view-mode slicing (recent / top / bottom)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from auto_goldfish.db.models import (
    Base,
    DeckRow,
    SimulationResultRow,
    SimulationRunRow,
)
from auto_goldfish.metrics.calibration import reset_cache
from auto_goldfish.web import create_app


_REQUIRED_RESULT_FIELDS = dict(
    mean_mana=20.0, mean_draws=3.0, mean_bad_turns=1.0, mean_lands=5.0,
    mean_mulls=0.2, mean_spells_cast=7.0, ci_mean_mana_low=18.0,
    ci_mean_mana_high=22.0, consistency=0.8, ci_consistency_low=0.75,
    ci_consistency_high=0.85, percentile_25=15.0, percentile_50=20.0,
    percentile_75=25.0,
)


def _add_run(
    session: Session,
    deck_name: str,
    job_id: str,
    *,
    consistency_score: int,
    created_at: datetime | None = None,
) -> SimulationRunRow:
    deck = session.query(DeckRow).filter_by(name=deck_name).one_or_none()
    if deck is None:
        deck = DeckRow(name=deck_name)
        session.add(deck)
        session.flush()

    run = SimulationRunRow(
        job_id=job_id, deck_id=deck.id, turns=10, sims=1000,
        min_lands=37, max_lands=38, optimal_land_count=38,
    )
    if created_at is not None:
        run.created_at = created_at
    session.add(run)
    session.flush()

    # One row at the optimal land count carries the score used for top/bottom
    # ranking; a second row with a much higher score should be ignored
    # because the join filters to optimal_land_count.
    session.add(SimulationResultRow(
        run_id=run.id, land_count=38, **_REQUIRED_RESULT_FIELDS,
        score_consistency=consistency_score,
        score_acceleration=5, score_snowball=5,
        score_toughness=5, score_efficiency=5, score_reach=5,
    ))
    session.add(SimulationResultRow(
        run_id=run.id, land_count=37, **_REQUIRED_RESULT_FIELDS,
        score_consistency=99,  # would dominate ranking if join were wrong
        score_acceleration=1, score_snowball=1,
        score_toughness=1, score_efficiency=1, score_reach=1,
    ))
    session.flush()
    return run


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    reset_cache()
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "0")
    yield
    reset_cache()


@pytest.fixture
def client_with_db(monkeypatch):
    """Wire /runs/api/data up to an in-memory SQLite DB seeded with runs."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    base_time = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    with SessionLocal() as session:
        # Seed 25 runs with varying consistency scores and creation times so
        # we can distinguish "recent 20" from "top 20 by consistency".
        for i in range(25):
            _add_run(
                session,
                deck_name=f"deck-{i:02d}",
                job_id=f"job-{i:02d}",
                consistency_score=i,  # 0..24 -- higher i means newer + higher score
                created_at=base_time + timedelta(hours=i),
            )
        session.commit()

    @contextmanager
    def _fake_get_session():
        s = SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr(
        "auto_goldfish.db.session.get_session", _fake_get_session,
    )

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_recent_view_returns_at_most_20(client_with_db):
    resp = client_with_db.get("/runs/api/data?view=recent")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["runs"]) == 20
    assert data["view"] == "recent"
    assert data["stat"] is None


def test_recent_view_orders_newest_first(client_with_db):
    resp = client_with_db.get("/runs/api/data?view=recent")
    runs = resp.get_json()["runs"]
    # Newest 20 are deck-05 through deck-24, sorted desc.
    assert runs[0]["deck_name"] == "deck-24"
    assert runs[-1]["deck_name"] == "deck-05"


def test_top_view_orders_by_score_desc(client_with_db):
    resp = client_with_db.get("/runs/api/data?view=top&stat=consistency")
    assert resp.status_code == 200
    data = resp.get_json()
    runs = data["runs"]
    assert data["view"] == "top"
    assert data["stat"] == "consistency"
    assert len(runs) == 20
    # Highest consistency at optimal_land_count first; the runs were seeded
    # with consistency_score == i, so deck-24 wins.
    assert runs[0]["deck_name"] == "deck-24"
    assert runs[0]["consistency"] == 24
    # Ranking uses the optimal-land-count score (38), not the spurious 99
    # at land_count=37 -- if the join were wrong every run would tie at 99.
    consistency_scores = [r["consistency"] for r in runs]
    assert consistency_scores == sorted(consistency_scores, reverse=True)


def test_bottom_view_orders_by_score_asc(client_with_db):
    resp = client_with_db.get("/runs/api/data?view=bottom&stat=consistency")
    runs = resp.get_json()["runs"]
    assert len(runs) == 20
    assert runs[0]["deck_name"] == "deck-00"
    assert runs[0]["consistency"] == 0
    consistency_scores = [r["consistency"] for r in runs]
    assert consistency_scores == sorted(consistency_scores)


def test_invalid_stat_falls_back_to_recent(client_with_db):
    resp = client_with_db.get("/runs/api/data?view=top&stat=not_a_real_stat")
    assert resp.status_code == 200
    data = resp.get_json()
    # Falls back to the default recent slice rather than 500ing, and the
    # response reports the view that was actually applied.
    assert data["view"] == "recent"
    assert data["stat"] is None
    assert len(data["runs"]) == 20
    assert data["runs"][0]["deck_name"] == "deck-24"


def test_top_view_skips_runs_with_null_score(client_with_db, monkeypatch):
    """Runs whose optimal-land result has a NULL stat should drop out."""
    # Patch one of the seeded runs' optimal score to NULL.
    from auto_goldfish.db.session import get_session
    with get_session() as s:
        target = s.query(SimulationResultRow).filter_by(land_count=38).first()
        target.score_consistency = None
        s.flush()

    resp = client_with_db.get("/runs/api/data?view=top&stat=consistency")
    runs = resp.get_json()["runs"]
    # Still 20 returned because there are 25 valid runs total minus 1 NULL.
    assert len(runs) == 20
    # The NULL run shouldn't appear at all.
    assert all(r["consistency"] is not None for r in runs)
