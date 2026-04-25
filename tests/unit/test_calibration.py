"""Tests for the on-the-fly anchor calibration provider."""

from __future__ import annotations

from typing import Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from auto_goldfish.db.models import (
    Base,
    DeckRow,
    SimulationResultRow,
    SimulationRunRow,
)
from auto_goldfish.metrics.calibration import (
    CalibrationMetadata,
    compute_anchors_from_db,
    get_active_anchors,
    reset_cache,
)
from auto_goldfish.metrics.deck_score import DEFAULT_ANCHORS, StatAnchors


@pytest.fixture(autouse=True)
def _clear_calibration_cache():
    """Each test gets a fresh module cache so order-of-tests doesn't leak."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Helpers: insert a deck + run + one result per land count.
# ---------------------------------------------------------------------------

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
    land_counts=(37, 38),
    optimal_land_count: Optional[int] = 38,
    raw_by_land: Optional[dict[int, dict[str, Optional[float]]]] = None,
) -> SimulationRunRow:
    """Insert a deck (if needed), a run, and N result rows.

    ``raw_by_land`` maps land_count -> dict of raw_* overrides; missing
    entries default to a fixed mid-range value. Pass ``{stat: None}`` to
    explicitly persist a NULL.
    """
    deck = session.query(DeckRow).filter_by(name=deck_name).one_or_none()
    if deck is None:
        deck = DeckRow(name=deck_name)
        session.add(deck)
        session.flush()

    run = SimulationRunRow(
        job_id=job_id, deck_id=deck.id, turns=10, sims=1000,
        min_lands=min(land_counts), max_lands=max(land_counts),
        optimal_land_count=optimal_land_count,
    )
    session.add(run)
    session.flush()

    raw_by_land = raw_by_land or {}
    for lc in land_counts:
        raws = {
            "raw_consistency": 0.5,
            "raw_acceleration": 7.0,
            "raw_snowball": 2.0,
            "raw_toughness": 0.7,
            "raw_efficiency": 0.5,
            "raw_reach": 20.0,
        }
        raws.update(raw_by_land.get(lc, {}))
        session.add(SimulationResultRow(
            run_id=run.id, land_count=lc, **_REQUIRED_RESULT_FIELDS, **raws,
        ))
    session.flush()
    return run


# ---------------------------------------------------------------------------
# Cold start: empty DB returns defaults.
# ---------------------------------------------------------------------------

class TestEmptyDb:
    def test_returns_defaults(self, db_session: Session):
        anchors, meta = compute_anchors_from_db(db_session)
        assert anchors == DEFAULT_ANCHORS

    def test_metadata_zero(self, db_session: Session):
        _, meta = compute_anchors_from_db(db_session)
        assert meta.n_rows == 0
        assert meta.n_decks == 0


# ---------------------------------------------------------------------------
# Bayesian shrinkage: small N stays near defaults.
# ---------------------------------------------------------------------------

class TestSmallSampleShrinkage:
    def test_single_deck_barely_moves_anchors(self, db_session: Session):
        # Insert one deck whose raw_consistency is wildly off-default.
        _add_run(
            db_session, "deck-1", "job-1",
            land_counts=(38,), optimal_land_count=38,
            raw_by_land={38: {"raw_consistency": 99.0}},
        )
        db_session.commit()

        anchors, meta = compute_anchors_from_db(db_session, pseudo_count=76)
        # Default consistency = (0.0, 1.0). With one row, shrinkage weight
        # on empirical = 1/77 ≈ 1.3%. Both bounds shift by ~99 * 0.013.
        expected_high = (1 / 77) * 99.0 + (76 / 77) * 1.0
        assert anchors.consistency[1] == pytest.approx(expected_high, rel=1e-6)
        assert meta.n_rows == 1
        assert meta.n_decks == 1


# ---------------------------------------------------------------------------
# Large N: empirical signal dominates.
# ---------------------------------------------------------------------------

class TestLargeSampleDominates:
    def test_many_decks_drive_anchors_to_empirical(self, db_session: Session):
        # 200 decks, raw_acceleration uniformly = 50.0. With pseudo_count=76
        # the shrunk anchor is heavily empirical.
        for i in range(200):
            _add_run(
                db_session, f"deck-{i}", f"job-{i}",
                land_counts=(38,), optimal_land_count=38,
                raw_by_land={38: {"raw_acceleration": 50.0}},
            )
        db_session.commit()

        anchors, meta = compute_anchors_from_db(db_session, pseudo_count=76)
        # All values are 50.0, so p10 and p90 are both 50.0.
        # Shrunk: w_real = 200/276; default is (1.0, 14.0).
        w_real = 200 / 276
        w_pseudo = 76 / 276
        expected_low = w_real * 50.0 + w_pseudo * 1.0
        expected_high = w_real * 50.0 + w_pseudo * 14.0
        assert anchors.acceleration[0] == pytest.approx(expected_low)
        assert anchors.acceleration[1] == pytest.approx(expected_high)
        assert meta.n_rows == 200
        assert meta.n_decks == 200


# ---------------------------------------------------------------------------
# Filtering: only optimal-land rows contribute.
# ---------------------------------------------------------------------------

class TestOptimalLandFilter:
    def test_non_optimal_rows_ignored(self, db_session: Session):
        # Run with optimal=38; off-optimal land 37 has wildly different raw.
        # Only the land=38 row should feed calibration.
        _add_run(
            db_session, "deck-1", "job-1",
            land_counts=(37, 38), optimal_land_count=38,
            raw_by_land={
                37: {"raw_reach": 999.0},  # should be ignored
                38: {"raw_reach": 25.0},
            },
        )
        db_session.commit()

        anchors, meta = compute_anchors_from_db(db_session, pseudo_count=0)
        # With pseudo_count=0, empirical dominates. p10 == p90 == 25.0
        assert anchors.reach_norm == pytest.approx((25.0, 25.0))
        assert meta.n_rows == 1

    def test_run_without_optimal_contributes_nothing(self, db_session: Session):
        # optimal_land_count is NULL -> no row matches the filter.
        _add_run(
            db_session, "deck-1", "job-1",
            land_counts=(37, 38), optimal_land_count=None,
        )
        db_session.commit()

        _, meta = compute_anchors_from_db(db_session)
        assert meta.n_rows == 0


# ---------------------------------------------------------------------------
# NULL handling: legacy rows without raw_* values are skipped.
# ---------------------------------------------------------------------------

class TestNullHandling:
    def test_null_raw_values_skipped(self, db_session: Session):
        _add_run(
            db_session, "deck-1", "job-1",
            land_counts=(38,), optimal_land_count=38,
            raw_by_land={38: {"raw_consistency": None}},  # legacy NULL
        )
        db_session.commit()

        _, meta = compute_anchors_from_db(db_session)
        assert meta.n_rows == 0


# ---------------------------------------------------------------------------
# n_decks counting (multiple runs per deck).
# ---------------------------------------------------------------------------

class TestDeckCounting:
    def test_multiple_runs_same_deck_count_as_one_deck(self, db_session: Session):
        _add_run(db_session, "deck-A", "job-1", land_counts=(38,), optimal_land_count=38)
        _add_run(db_session, "deck-A", "job-2", land_counts=(38,), optimal_land_count=38)
        _add_run(db_session, "deck-B", "job-3", land_counts=(38,), optimal_land_count=38)
        db_session.commit()

        _, meta = compute_anchors_from_db(db_session)
        assert meta.n_rows == 3
        assert meta.n_decks == 2


# ---------------------------------------------------------------------------
# Snowball secondary anchor: not persisted, falls through to default.
# ---------------------------------------------------------------------------

class TestSnowballSecondary:
    def test_snowball_late_avg_norm_uses_default(self, db_session: Session):
        for i in range(200):
            _add_run(
                db_session, f"deck-{i}", f"job-{i}",
                land_counts=(38,), optimal_land_count=38,
                raw_by_land={38: {"raw_snowball": 99.0}},
            )
        db_session.commit()

        anchors, _ = compute_anchors_from_db(db_session)
        # Even with strong empirical signal on the ratio, the secondary
        # snowball anchor must equal the default (it isn't persisted).
        assert anchors.snowball_late_avg_norm == DEFAULT_ANCHORS.snowball_late_avg_norm


# ---------------------------------------------------------------------------
# Custom defaults / pseudo_count are respected.
# ---------------------------------------------------------------------------

class TestKnobs:
    def test_custom_defaults_override(self, db_session: Session):
        custom = StatAnchors(consistency=(7.0, 8.0))
        anchors, _ = compute_anchors_from_db(db_session, defaults=custom)
        assert anchors.consistency == (7.0, 8.0)

    def test_pseudo_count_zero_full_empirical(self, db_session: Session):
        _add_run(
            db_session, "deck-1", "job-1",
            land_counts=(38,), optimal_land_count=38,
            raw_by_land={38: {"raw_efficiency": 0.42}},
        )
        db_session.commit()
        anchors, _ = compute_anchors_from_db(db_session, pseudo_count=0)
        # With pseudo=0 and a single row, both bounds collapse onto 0.42.
        assert anchors.efficiency == pytest.approx((0.42, 0.42))

    def test_metadata_records_knobs(self, db_session: Session):
        _, meta = compute_anchors_from_db(
            db_session, pseudo_count=42, low_pct=5.0, high_pct=95.0,
        )
        assert isinstance(meta, CalibrationMetadata)
        assert meta.pseudo_count == 42
        assert meta.low_pct == 5.0
        assert meta.high_pct == 95.0


# ---------------------------------------------------------------------------
# Runtime cached provider: get_active_anchors().
# ---------------------------------------------------------------------------

class TestGetActiveAnchorsToggle:
    def test_default_is_enabled(self, db_session: Session, monkeypatch):
        # No env var set -> calibration is on.
        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)
        # Push the row count high so empirical is meaningful.
        for i in range(150):
            _add_run(
                db_session, f"deck-{i}", f"job-{i}",
                land_counts=(38,), optimal_land_count=38,
                raw_by_land={38: {"raw_acceleration": 50.0}},
            )
        db_session.commit()

        anchors, meta = get_active_anchors(db_session)
        assert meta is not None  # calibrated path
        # 150 real decks with anchor 50.0 should bias acceleration upward
        # away from the default's max of 14.0.
        assert anchors.acceleration[1] > DEFAULT_ANCHORS.acceleration[1]

    @pytest.mark.parametrize("disable_value", ["0", "false", "no", "off"])
    def test_env_disable_returns_defaults(
        self, db_session: Session, monkeypatch, disable_value
    ):
        for i in range(150):
            _add_run(
                db_session, f"deck-{i}", f"job-{i}",
                land_counts=(38,), optimal_land_count=38,
                raw_by_land={38: {"raw_acceleration": 50.0}},
            )
        db_session.commit()
        monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", disable_value)

        anchors, meta = get_active_anchors(db_session)
        assert anchors == DEFAULT_ANCHORS
        assert meta is None

    def test_empty_db_returns_defaults_with_metadata(self, db_session: Session, monkeypatch):
        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)
        anchors, meta = get_active_anchors(db_session)
        assert anchors == DEFAULT_ANCHORS
        # Empty DB still goes through compute_anchors_from_db and returns
        # n_rows=0 metadata so the UI can show "calibrated (0 decks)".
        assert meta is not None
        assert meta.n_rows == 0


class TestGetActiveAnchorsCache:
    def test_repeated_calls_hit_cache(self, db_session: Session, monkeypatch):
        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)
        _add_run(db_session, "deck-1", "job-1", land_counts=(38,), optimal_land_count=38)
        db_session.commit()

        first = get_active_anchors(db_session)
        # Same identity tuple should be returned on the second hit because
        # nothing has changed in the DB.
        second = get_active_anchors(db_session)
        assert first is second

    def test_new_row_invalidates_cache(self, db_session: Session, monkeypatch):
        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)
        _add_run(db_session, "deck-1", "job-1", land_counts=(38,), optimal_land_count=38)
        db_session.commit()
        first = get_active_anchors(db_session)

        # Add another row -> count changes -> cache should miss.
        _add_run(db_session, "deck-2", "job-2", land_counts=(38,), optimal_land_count=38)
        db_session.commit()
        second = get_active_anchors(db_session)

        assert second is not first  # fresh tuple from recomputation
        assert second[1].n_rows == 2  # metadata reflects new row count


class TestGetActiveAnchorsFallback:
    def test_no_session_and_no_initialized_db_returns_defaults(self, monkeypatch):
        # No DB initialized via init_db -> get_session() raises -> fallback.
        # We don't pass a session and we ensure init_db hasn't been called
        # by patching get_session to raise.
        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)
        import auto_goldfish.db.session as session_mod

        def _boom():
            raise RuntimeError("not initialized")

        monkeypatch.setattr(session_mod, "get_session", _boom)

        anchors, meta = get_active_anchors()
        assert anchors == DEFAULT_ANCHORS
        assert meta is None

    def test_query_failure_returns_defaults(self, db_session: Session, monkeypatch):
        """A broken DB session shouldn't break scoring -- swallow + default."""
        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)

        # Force the row-count probe to raise.
        from auto_goldfish.metrics import calibration as cal

        def _boom(session):
            raise RuntimeError("simulated query failure")

        monkeypatch.setattr(cal, "_row_count", _boom)

        anchors, meta = get_active_anchors(db_session)
        assert anchors == DEFAULT_ANCHORS
        assert meta is None


# ---------------------------------------------------------------------------
# result_to_dict integration: scores come from the active anchors and the
# 'calibration' key carries metadata for the UI.
# ---------------------------------------------------------------------------

class TestResultToDictCalibrationKey:
    def test_no_db_initialized_calibration_is_none(self, monkeypatch):
        """No DB -> defaults path -> 'calibration' is None on the dict."""
        from auto_goldfish.engine.goldfisher import SimulationResult
        from auto_goldfish.metrics.reporter import result_to_dict
        import auto_goldfish.db.session as session_mod

        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)

        def _boom():
            raise RuntimeError("not initialized")

        monkeypatch.setattr(session_mod, "get_session", _boom)

        out = result_to_dict(SimulationResult(), turns=10)
        assert out["calibration"] is None

    def test_env_disabled_calibration_is_none(self, monkeypatch):
        from auto_goldfish.engine.goldfisher import SimulationResult
        from auto_goldfish.metrics.reporter import result_to_dict

        monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "0")
        out = result_to_dict(SimulationResult(), turns=10)
        assert out["calibration"] is None


# ---------------------------------------------------------------------------
# save_simulation_run re-scores from raw_* with the active anchors.
# ---------------------------------------------------------------------------

class TestPersistenceReScoring:
    def test_persisted_score_uses_active_anchors(
        self, db_session: Session, monkeypatch
    ):
        """A custom calibrated anchor on the DB shifts the stored score."""
        from sqlalchemy import select
        from auto_goldfish.db.persistence import (
            get_or_create_deck, save_simulation_run,
        )
        from auto_goldfish.metrics import calibration as cal

        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)

        # Force get_active_anchors to return a custom anchor that *only*
        # reaches 10 when raw_acceleration is huge -> caller's mid value
        # should land at 1.
        custom = StatAnchors(acceleration=(100.0, 200.0))

        def _stub(session=None):
            return custom, None

        monkeypatch.setattr(cal, "get_active_anchors", _stub)
        # save_simulation_run imports the symbol locally; patch the source.

        deck = get_or_create_deck(db_session, "calibrated-deck")
        results = [{
            "land_count": 38,
            "mean_mana": 7.5, "mean_draws": 3.0, "mean_bad_turns": 1.0,
            "mean_lands": 3.5, "mean_mulls": 0.2,
            "ci_mean_mana": [7.0, 8.0], "ci_consistency": [0.80, 0.90],
            "consistency": 0.85, "percentile_25": 6.0,
            "percentile_50": 7.5, "percentile_75": 9.0,
            "deck_score": {
                "consistency": 7, "acceleration": 7, "snowball": 5,
                "toughness": 6, "efficiency": 6, "reach": 6,
            },
            "deck_raw": {
                "consistency": 0.62, "acceleration": 7.5, "snowball": 1.8,
                "toughness": 0.81, "efficiency": 0.55, "reach": 22.0,
                "snowball_late_avg_norm": 4.0,
            },
            "card_performance": {"low_performing": [], "high_performing": []},
        }]
        config = {"turns": 10, "sims": 1000, "min_lands": 38, "max_lands": 38}

        save_simulation_run(db_session, "rescore-job", deck, config, results)
        db_session.commit()

        from auto_goldfish.db.models import SimulationResultRow
        row = db_session.execute(select(SimulationResultRow)).scalar_one()
        # raw_acceleration=7.5 with anchor (100, 200) maps below 100 -> score 1.
        assert row.score_acceleration == 1
        # The input dict said acceleration=7; it was overridden by the
        # active-anchor re-score, proving the calibration is applied.
        assert row.score_acceleration != 7

    def test_legacy_input_without_deck_raw_still_persists(
        self, db_session: Session, monkeypatch
    ):
        """Old result dicts without deck_raw skip re-scoring (no crash)."""
        from sqlalchemy import select
        from auto_goldfish.db.persistence import (
            get_or_create_deck, save_simulation_run,
        )

        monkeypatch.delenv("AUTO_GOLDFISH_CALIBRATE", raising=False)

        deck = get_or_create_deck(db_session, "legacy-deck")
        results = [{
            "land_count": 38,
            "mean_mana": 7.5, "mean_draws": 3.0, "mean_bad_turns": 1.0,
            "mean_lands": 3.5, "mean_mulls": 0.2,
            "ci_mean_mana": [7.0, 8.0], "ci_consistency": [0.80, 0.90],
            "consistency": 0.85, "percentile_25": 6.0,
            "percentile_50": 7.5, "percentile_75": 9.0,
            "deck_score": {
                "consistency": 7, "acceleration": 5, "snowball": 5,
                "toughness": 6, "efficiency": 6, "reach": 6,
            },
            # No deck_raw key -- e.g., a result dict from before Phase 0.
            "card_performance": {"low_performing": [], "high_performing": []},
        }]
        config = {"turns": 10, "sims": 1000, "min_lands": 38, "max_lands": 38}

        save_simulation_run(db_session, "legacy-job", deck, config, results)
        db_session.commit()

        from auto_goldfish.db.models import SimulationResultRow
        row = db_session.execute(select(SimulationResultRow)).scalar_one()
        # Without raw_*, persistence keeps the input scores verbatim.
        assert row.score_acceleration == 5
