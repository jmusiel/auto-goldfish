"""Tests for init_db's run_migrations gate and verify_schema().

These cover the deploy-time / runtime split: production calls init_db with
run_migrations=False (gated by AUTO_GOLDFISH_SKIP_MIGRATE) so cold-start
workers don't race on ALTER TABLE; verify_schema flags any missing columns
the runtime relies on without taking the app down.
"""

from __future__ import annotations

import logging

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from auto_goldfish.db import session as session_mod
from auto_goldfish.db.models import Base


@pytest.fixture(autouse=True)
def _clear_module_state():
    """Make sure init_db() side-effects don't leak between tests."""
    session_mod._engine = None
    session_mod._SessionFactory = None
    yield
    session_mod._engine = None
    session_mod._SessionFactory = None


def _fresh_sqlite_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/test.db"


class TestRunMigrationsGate:
    def test_default_runs_migrations(self, tmp_path, monkeypatch):
        """No env var set -> migrations run -> tables exist."""
        monkeypatch.delenv("AUTO_GOLDFISH_SKIP_MIGRATE", raising=False)
        url = _fresh_sqlite_url(tmp_path)

        session_mod.init_db(url)

        insp = inspect(session_mod._engine)
        assert "simulation_runs" in insp.get_table_names()
        assert "calibration_cache" in insp.get_table_names()

    def test_env_skip_disables_migrations(self, tmp_path, monkeypatch):
        """AUTO_GOLDFISH_SKIP_MIGRATE=1 -> tables NOT created."""
        monkeypatch.setenv("AUTO_GOLDFISH_SKIP_MIGRATE", "1")
        url = _fresh_sqlite_url(tmp_path)

        session_mod.init_db(url)

        insp = inspect(session_mod._engine)
        # No tables: verify_schema runs but doesn't create anything.
        assert insp.get_table_names() == []

    def test_explicit_run_migrations_overrides_env(self, tmp_path, monkeypatch):
        """run_migrations=True wins over the skip env var (build-step path)."""
        monkeypatch.setenv("AUTO_GOLDFISH_SKIP_MIGRATE", "1")
        url = _fresh_sqlite_url(tmp_path)

        session_mod.init_db(url, run_migrations=True)

        insp = inspect(session_mod._engine)
        assert "simulation_runs" in insp.get_table_names()

    def test_explicit_skip_overrides_env(self, tmp_path, monkeypatch):
        """run_migrations=False wins even when env says auto-migrate."""
        monkeypatch.delenv("AUTO_GOLDFISH_SKIP_MIGRATE", raising=False)
        url = _fresh_sqlite_url(tmp_path)

        session_mod.init_db(url, run_migrations=False)

        insp = inspect(session_mod._engine)
        assert insp.get_table_names() == []


class TestVerifySchema:
    def test_passes_on_complete_schema(self, tmp_path, caplog):
        url = _fresh_sqlite_url(tmp_path)
        engine = create_engine(url)
        Base.metadata.create_all(engine)

        with caplog.at_level(logging.INFO, logger="auto_goldfish.db.session"):
            issues = session_mod.verify_schema(engine)

        assert issues == []
        assert "Schema verification passed" in caplog.text

    def test_flags_missing_table(self, tmp_path, caplog):
        """Empty DB -> every required table is missing."""
        url = _fresh_sqlite_url(tmp_path)
        engine = create_engine(url)
        # Don't create anything.

        with caplog.at_level(logging.ERROR, logger="auto_goldfish.db.session"):
            issues = session_mod.verify_schema(engine)

        assert any("missing table: simulation_results" in i for i in issues)
        assert any("missing table: calibration_cache" in i for i in issues)
        assert "run scripts/migrate.py" in caplog.text

    def test_flags_missing_columns(self, tmp_path, caplog):
        """A partial DB (missing recently-added columns) reports them."""
        url = _fresh_sqlite_url(tmp_path)
        engine = create_engine(url)
        Base.metadata.create_all(engine)

        # Drop a known column to simulate a stale prod DB.
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE simulation_results DROP COLUMN raw_consistency"))

        with caplog.at_level(logging.ERROR, logger="auto_goldfish.db.session"):
            issues = session_mod.verify_schema(engine)

        assert any(
            "simulation_results" in i and "raw_consistency" in i for i in issues
        )

    def test_does_not_raise_on_probe_failure(self, monkeypatch, caplog):
        """A flaky inspect() shouldn't crash the runtime; just log + return."""
        from auto_goldfish.db import session as sess

        class _BoomEngine:
            pass

        def _boom(_):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(sess, "inspect", _boom)
        with caplog.at_level(logging.ERROR, logger="auto_goldfish.db.session"):
            issues = sess.verify_schema(_BoomEngine())

        assert issues == ["schema verification probe failed"]


class TestInitDbSkipPath:
    def test_skip_init_still_yields_working_session_factory(self, tmp_path, monkeypatch):
        """Even with migrations skipped, get_session() should work."""
        monkeypatch.setenv("AUTO_GOLDFISH_SKIP_MIGRATE", "1")
        url = _fresh_sqlite_url(tmp_path)

        # Pre-create the schema as if a prior deploy migrated it.
        engine = create_engine(url)
        Base.metadata.create_all(engine)
        engine.dispose()

        session_mod.init_db(url)

        with session_mod.get_session() as sess:
            # No error: the session works against the pre-migrated DB.
            sess.execute(text("SELECT 1"))
