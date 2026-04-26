"""Database engine and session management."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def init_db(database_url: str) -> None:
    """Create the engine, session factory, and all tables.

    Uses ``NullPool`` and ``pool_pre_ping`` so that this works correctly under
    serverless runtimes (Vercel) where workers freeze between requests: pooled
    connections kept across a freeze go stale and the next request gets a dead
    socket. NullPool means every session opens a fresh connection (Neon's own
    pgbouncer handles pooling on the server side); pre_ping is belt-and-braces.
    """
    global _engine, _SessionFactory
    _engine = create_engine(
        database_url,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
    _SessionFactory = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)
    _migrate(_engine)
    logger.info("Database initialized: %s", database_url.split("@")[-1] if "@" in database_url else "(local)")


def is_db_configured() -> bool:
    """Return True if init_db has been called successfully."""
    return _SessionFactory is not None


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
        # Rename CASTER score columns from old names if present (idempotent).
        # The Snowball stat went through two renames (momentum -> surge ->
        # snowball); some production DBs ended up with *both* legacy
        # columns. We plan operations against a tracked column set so the
        # second rename to the same target turns into a DROP of the
        # redundant legacy column instead of a duplicate-column error.
        # Order matters: the most recent legacy name wins the rename
        # (preserving its data) and older legacy duplicates get dropped.
        column_renames = [
            ("score_speed", "score_acceleration"),
            ("score_power", "score_reach"),
            ("score_resilience", "score_toughness"),
            ("score_surge", "score_snowball"),
            ("score_momentum", "score_snowball"),
            ("raw_surge", "raw_snowball"),
        ]
        live_cols = set(cols)
        ops: list[tuple] = []
        for old, new in column_renames:
            if old not in live_cols:
                continue
            if new in live_cols:
                ops.append(("drop", old))
                live_cols.discard(old)
            else:
                ops.append(("rename", old, new))
                live_cols.discard(old)
                live_cols.add(new)
        if ops:
            with engine.begin() as conn:
                for op in ops:
                    if op[0] == "rename":
                        _, old, new = op
                        conn.execute(text(
                            f"ALTER TABLE simulation_results RENAME COLUMN {old} TO {new}"
                        ))
                    else:
                        _, old = op
                        conn.execute(text(
                            f"ALTER TABLE simulation_results DROP COLUMN {old}"
                        ))
            logger.info("Migrated simulation_results: applied %s", ops)
            cols = live_cols

        score_cols = [
            "score_consistency", "score_acceleration", "score_snowball",
            "score_toughness", "score_efficiency", "score_reach",
        ]
        missing = [c for c in score_cols if c not in cols]
        if missing:
            with engine.begin() as conn:
                for col in missing:
                    conn.execute(text(f"ALTER TABLE simulation_results ADD COLUMN {col} INTEGER"))
            logger.info("Migrated simulation_results: added deck score columns")
            cols = {c["name"] for c in inspect(engine).get_columns("simulation_results")}

        raw_cols = [
            "raw_consistency", "raw_acceleration", "raw_snowball",
            "raw_toughness", "raw_efficiency", "raw_reach",
        ]
        missing_raw = [c for c in raw_cols if c not in cols]
        if missing_raw:
            with engine.begin() as conn:
                for col in missing_raw:
                    conn.execute(text(f"ALTER TABLE simulation_results ADD COLUMN {col} FLOAT"))
            logger.info("Migrated simulation_results: added raw stat columns")

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
