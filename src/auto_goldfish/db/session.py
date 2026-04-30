"""Database engine and session management."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None

_SKIP_MIGRATE_ENV = "AUTO_GOLDFISH_SKIP_MIGRATE"

# Tables/columns the runtime relies on. ``verify_schema`` checks these in
# read-only mode when migrations are skipped at startup so a stale DB shows
# up loudly in the logs instead of silently 500ing on the first query.
_REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "simulation_results": (
        "score_consistency", "score_acceleration", "score_snowball",
        "score_toughness", "score_efficiency", "score_reach",
        "raw_consistency", "raw_acceleration", "raw_snowball",
        "raw_toughness", "raw_efficiency", "raw_reach",
        "mean_spells_cast",
    ),
    "card_annotations": ("session_id",),
    "card_performance": ("mean_with", "mean_without"),
    "calibration_cache": ("n_rows", "anchors_json", "metadata_json"),
    "cards": ("types_json", "cmc"),
    "deck_cards": ("quantity", "is_commander"),
}


def _skip_migrations_env() -> bool:
    """Return True if the env var disables auto-migrate at startup."""
    raw = os.environ.get(_SKIP_MIGRATE_ENV, "0").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def init_db(database_url: str, *, run_migrations: bool | None = None) -> None:
    """Create the engine, session factory, and (optionally) the schema.

    Uses ``NullPool`` and ``pool_pre_ping`` so that this works correctly under
    serverless runtimes (Vercel) where workers freeze between requests: pooled
    connections kept across a freeze go stale and the next request gets a dead
    socket. NullPool means every session opens a fresh connection (Neon's own
    pgbouncer handles pooling on the server side); pre_ping is belt-and-braces.

    ``run_migrations`` controls whether ``Base.metadata.create_all()`` and
    ``_migrate()`` run on startup. Default behaviour: migrate unless the
    ``AUTO_GOLDFISH_SKIP_MIGRATE`` env var is truthy. Production (Vercel)
    sets that var so schema changes happen once during the build step via
    ``scripts/migrate.py`` instead of on every cold-start worker, removing
    the ALTER-TABLE race that would otherwise fire under concurrent boots.
    Local dev/tests leave the var unset and keep the auto-create behaviour.
    """
    global _engine, _SessionFactory
    _engine = create_engine(
        database_url,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
    _SessionFactory = sessionmaker(bind=_engine)

    if run_migrations is None:
        run_migrations = not _skip_migrations_env()

    if run_migrations:
        Base.metadata.create_all(_engine)
        _migrate(_engine)
        logger.info(
            "Database initialized with migrations: %s",
            database_url.split("@")[-1] if "@" in database_url else "(local)",
        )
    else:
        verify_schema(_engine)
        logger.info(
            "Database initialized (migrations skipped): %s",
            database_url.split("@")[-1] if "@" in database_url else "(local)",
        )


def verify_schema(engine) -> list[str]:
    """Read-only check that required tables/columns exist.

    Returns a list of issue strings (empty when healthy). Logs at ERROR if
    anything is missing so an out-of-date production DB is impossible to
    miss in Vercel logs. Never raises -- the runtime keeps serving so
    requests that don't touch the missing columns still work.
    """
    issues: list[str] = []
    try:
        insp = inspect(engine)
        existing_tables = set(insp.get_table_names())
        for table, required_cols in _REQUIRED_SCHEMA.items():
            if table not in existing_tables:
                issues.append(f"missing table: {table}")
                continue
            present = {c["name"] for c in insp.get_columns(table)}
            missing = [c for c in required_cols if c not in present]
            if missing:
                issues.append(f"{table}: missing columns {missing}")
    except Exception:
        logger.exception("Schema verification probe failed")
        return ["schema verification probe failed"]

    if issues:
        logger.error(
            "Schema verification found issues; run scripts/migrate.py: %s",
            issues,
        )
    else:
        logger.info("Schema verification passed")
    return issues


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

    if "cards" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("cards")}
        adds: list[tuple[str, str]] = []
        if "types_json" not in cols:
            adds.append(("types_json", "TEXT"))
        if "cmc" not in cols:
            adds.append(("cmc", "INTEGER"))
        if adds:
            with engine.begin() as conn:
                for col, sql_type in adds:
                    conn.execute(text(f"ALTER TABLE cards ADD COLUMN {col} {sql_type}"))
            logger.info("Migrated cards: added %s", [a[0] for a in adds])

    if "deck_cards" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("deck_cards")}
        adds: list[tuple[str, str]] = []
        if "quantity" not in cols:
            adds.append(("quantity", "INTEGER NOT NULL DEFAULT 1"))
        if "is_commander" not in cols:
            adds.append(("is_commander", "BOOLEAN NOT NULL DEFAULT FALSE"))
        if adds:
            with engine.begin() as conn:
                for col, sql_type in adds:
                    conn.execute(text(
                        f"ALTER TABLE deck_cards ADD COLUMN {col} {sql_type}"
                    ))
            logger.info("Migrated deck_cards: added %s", [a[0] for a in adds])


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
