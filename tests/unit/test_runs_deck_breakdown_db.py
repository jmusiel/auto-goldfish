"""Tests for the DB-backed deck breakdown reconstruction in /runs.

Confirms that ``_load_deck_breakdown`` rebuilds a usable composition from
``CardRow`` + ``DeckCardRow`` alone -- no disk reads -- so the runs page
works on Vercel where deck JSON files aren't deployed.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from auto_goldfish.db.models import (
    Base,
    DeckRow,
    SimulationResultRow,
    SimulationRunRow,
)
from auto_goldfish.db.persistence import save_deck_cards
from auto_goldfish.metrics.calibration import reset_cache
from auto_goldfish.web import create_app


_REQUIRED_RESULT_FIELDS = dict(
    mean_mana=20.0, mean_draws=3.0, mean_bad_turns=1.0, mean_lands=5.0,
    mean_mulls=0.2, mean_spells_cast=7.0, ci_mean_mana_low=18.0,
    ci_mean_mana_high=22.0, consistency=0.8, ci_consistency_low=0.75,
    ci_consistency_high=0.85, percentile_25=15.0, percentile_50=20.0,
    percentile_75=25.0,
)


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    reset_cache()
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "0")
    yield
    reset_cache()


def _seed_run_with_deck(
    session: Session,
    deck_name: str,
    deck_list: list[dict],
    *,
    optimal_land_count: int = 38,
) -> SimulationRunRow:
    deck = DeckRow(name=deck_name)
    session.add(deck)
    session.flush()

    save_deck_cards(session, deck, deck_list, {})

    run = SimulationRunRow(
        job_id=f"job-{deck_name}", deck_id=deck.id, turns=10, sims=1000,
        min_lands=37, max_lands=38, optimal_land_count=optimal_land_count,
        created_at=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    session.add(run)
    session.flush()

    session.add(SimulationResultRow(
        run_id=run.id, land_count=optimal_land_count,
        score_consistency=7, score_acceleration=5, score_snowball=5,
        score_toughness=5, score_efficiency=5, score_reach=5,
        **_REQUIRED_RESULT_FIELDS,
    ))
    session.flush()
    return run


@pytest.fixture
def client_with_full_deck(monkeypatch):
    """In-memory DB with one run whose deck has full metadata persisted."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    deck_list = [
        # Commander
        {"name": "Atraxa, Praetors' Voice", "cmc": 4,
         "types": ["Legendary", "Creature"], "quantity": 1, "commander": True},
        # Lands -- contribute to land_count, no CMC bucket
        {"name": "Forest", "cmc": 0, "types": ["Basic", "Land"],
         "quantity": 10, "commander": False},
        {"name": "Plains", "cmc": 0, "types": ["Basic", "Land"],
         "quantity": 8, "commander": False},
        # Nonland spells
        {"name": "Sol Ring", "cmc": 1, "types": ["Artifact"],
         "quantity": 1, "commander": False},
        {"name": "Cultivate", "cmc": 3, "types": ["Sorcery"],
         "quantity": 1, "commander": False},
    ]

    with SessionLocal() as session:
        _seed_run_with_deck(session, "atraxa-test", deck_list)
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


def test_breakdown_reconstructed_from_db(client_with_full_deck):
    resp = client_with_full_deck.get("/runs/api/data?view=recent")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["runs"]) == 1
    bd = data["runs"][0]["deck_breakdown"]

    assert bd is not None, "expected breakdown reconstructed from DB metadata"
    assert bd["commander_names"] == ["Atraxa, Praetors' Voice"]
    # 10 Forests + 8 Plains
    assert bd["land_count"] == 18
    # Nonland CMC distribution: 1 (Sol Ring), 3 (Cultivate), 4 (Atraxa)
    assert bd["cmc_distribution"] == {"1": 1, "3": 1, "4": 1}


def test_breakdown_none_when_deck_has_no_persisted_cards(monkeypatch):
    """Legacy decks with no deck_cards rows surface breakdown=None."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        deck = DeckRow(name="legacy-deck")
        session.add(deck)
        session.flush()
        run = SimulationRunRow(
            job_id="legacy-job", deck_id=deck.id, turns=10, sims=1000,
            min_lands=37, max_lands=38, optimal_land_count=38,
            created_at=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        session.add(run)
        session.flush()
        session.add(SimulationResultRow(
            run_id=run.id, land_count=38,
            score_consistency=5, score_acceleration=5, score_snowball=5,
            score_toughness=5, score_efficiency=5, score_reach=5,
            **_REQUIRED_RESULT_FIELDS,
        ))
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
    client = app.test_client()
    data = client.get("/runs/api/data?view=recent").get_json()
    assert data["runs"][0]["deck_breakdown"] is None
