"""Database engine and session management."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def init_db(database_url: str) -> None:
    """Create the engine, session factory, and all tables."""
    global _engine, _SessionFactory
    _engine = create_engine(database_url)
    _SessionFactory = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)
    _migrate(_engine)
    logger.info("Database initialized: %s", database_url.split("@")[-1] if "@" in database_url else "(local)")


def _migrate(engine) -> None:
    """Add columns that create_all won't add to existing tables."""
    insp = inspect(engine)
    if "card_annotations" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("card_annotations")}
        if "session_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE card_annotations ADD COLUMN session_id TEXT"))
            logger.info("Migrated card_annotations: added session_id column")

    if "simulation_results" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("simulation_results")}
        if "mean_spells_cast" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE simulation_results ADD COLUMN mean_spells_cast FLOAT NOT NULL DEFAULT 0.0"
                ))
            logger.info("Migrated simulation_results: added mean_spells_cast column")
        score_cols = [
            "score_speed", "score_power", "score_consistency",
            "score_resilience", "score_efficiency", "score_momentum",
        ]
        missing = [c for c in score_cols if c not in cols]
        if missing:
            with engine.begin() as conn:
                for col in missing:
                    conn.execute(text(f"ALTER TABLE simulation_results ADD COLUMN {col} INTEGER"))
            logger.info("Migrated simulation_results: added deck score columns")

    if "card_performance" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("card_performance")}
        renames = []
        if "top_rate" in cols and "mean_with" not in cols:
            renames.append(("top_rate", "mean_with"))
        if "low_rate" in cols and "mean_without" not in cols:
            renames.append(("low_rate", "mean_without"))
        if renames:
            with engine.begin() as conn:
                for old, new in renames:
                    conn.execute(text(
                        f"ALTER TABLE card_performance RENAME COLUMN {old} TO {new}"
                    ))
            logger.info("Migrated card_performance: renamed %s", renames)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a session that auto-commits on success and rolls back on error."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
