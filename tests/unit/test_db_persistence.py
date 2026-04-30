"""Tests for persistence layer -- deduplication, upsert, filtering."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from auto_goldfish.db.models import (
    Base,
    CardPerformanceRow,
    CardRow,
    DeckCardRow,
    DeckRow,
    EffectLabelRow,
    SimulationResultRow,
    SimulationRunRow,
)
from auto_goldfish.db.persistence import (
    get_or_create_card,
    get_or_create_deck,
    get_or_create_effect_label,
    save_deck_cards,
    save_simulation_run,
)


@pytest.fixture
def db_session():
    """In-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# get-or-create tests
# ---------------------------------------------------------------------------

class TestGetOrCreateCard:
    def test_creates_new(self, db_session: Session):
        card = get_or_create_card(db_session, "Sol Ring")
        assert card.id is not None
        assert card.name == "Sol Ring"

    def test_returns_existing(self, db_session: Session):
        c1 = get_or_create_card(db_session, "Sol Ring")
        c2 = get_or_create_card(db_session, "Sol Ring")
        assert c1.id == c2.id

    def test_different_names(self, db_session: Session):
        c1 = get_or_create_card(db_session, "Sol Ring")
        c2 = get_or_create_card(db_session, "Mana Crypt")
        assert c1.id != c2.id


class TestGetOrCreateEffectLabel:
    def test_creates_new(self, db_session: Session):
        label = get_or_create_effect_label(db_session, {"effects": [{"type": "draw"}]})
        assert label.id is not None

    def test_deduplication(self, db_session: Session):
        l1 = get_or_create_effect_label(db_session, {"effects": [{"type": "draw"}]})
        l2 = get_or_create_effect_label(db_session, {"effects": [{"type": "draw"}]})
        assert l1.id == l2.id

    def test_key_order_irrelevant(self, db_session: Session):
        l1 = get_or_create_effect_label(db_session, {"b": 2, "a": 1})
        l2 = get_or_create_effect_label(db_session, {"a": 1, "b": 2})
        assert l1.id == l2.id


class TestGetOrCreateDeck:
    def test_creates_new(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        assert deck.id is not None
        assert deck.name == "test-deck"

    def test_returns_existing(self, db_session: Session):
        d1 = get_or_create_deck(db_session, "test-deck")
        d2 = get_or_create_deck(db_session, "test-deck")
        assert d1.id == d2.id


# ---------------------------------------------------------------------------
# save_deck_cards tests
# ---------------------------------------------------------------------------

class TestSaveDeckCards:
    def test_inserts_cards(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        deck_list = [
            {"name": "Sol Ring", "cmc": 1, "types": ["Artifact"], "quantity": 1},
            {"name": "Grizzly Bears", "cmc": 2, "types": ["Creature"], "quantity": 1},
        ]
        effects_list = [
            {"name": "Sol Ring", "registry_override": {"effects": [{"type": "ramp"}]}},
            {"name": "Grizzly Bears", "registry_override": None},
        ]
        save_deck_cards(db_session, deck, deck_list, {}, effects_list)
        db_session.commit()

        deck_cards = db_session.execute(select(DeckCardRow)).scalars().all()
        assert len(deck_cards) == 2

    def test_upsert_updates_label(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        deck_list = [{"name": "Sol Ring", "cmc": 1, "types": ["Artifact"], "quantity": 1}]
        effects_list = [{"name": "Sol Ring", "registry_override": {"effects": [{"type": "ramp"}]}}]

        save_deck_cards(db_session, deck, deck_list, {}, effects_list)
        db_session.commit()

        # Re-save with user override
        overrides = {"Sol Ring": {"effects": [{"type": "draw"}]}}
        save_deck_cards(db_session, deck, deck_list, overrides, effects_list)
        db_session.commit()

        deck_cards = db_session.execute(select(DeckCardRow)).scalars().all()
        assert len(deck_cards) == 1
        assert deck_cards[0].user_edited is True

    def test_skips_empty_names(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        deck_list = [{"name": "", "cmc": 0, "types": [], "quantity": 1}]
        save_deck_cards(db_session, deck, deck_list, {})
        db_session.commit()

        deck_cards = db_session.execute(select(DeckCardRow)).scalars().all()
        assert len(deck_cards) == 0

    def test_user_override_sets_flag(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        deck_list = [{"name": "Sol Ring", "cmc": 1, "types": ["Artifact"], "quantity": 1}]
        overrides = {"Sol Ring": {"effects": [{"type": "draw"}]}}

        save_deck_cards(db_session, deck, deck_list, overrides)
        db_session.commit()

        dc = db_session.execute(select(DeckCardRow)).scalar_one()
        assert dc.user_edited is True
        assert dc.label_id is not None

    def test_persists_card_metadata(self, db_session: Session):
        """Cards row gets types_json + cmc; deck_card row gets quantity + is_commander."""
        import json as _json

        deck = get_or_create_deck(db_session, "test-deck")
        deck_list = [
            {"name": "Atraxa, Praetors' Voice", "cmc": 4,
             "types": ["Legendary", "Creature", "Phyrexian", "Angel", "Horror"],
             "quantity": 1, "commander": True},
            {"name": "Forest", "cmc": 0, "types": ["Basic", "Land", "Forest"],
             "quantity": 5, "commander": False},
        ]
        save_deck_cards(db_session, deck, deck_list, {})
        db_session.commit()

        atraxa = db_session.execute(
            select(CardRow).where(CardRow.name == "Atraxa, Praetors' Voice")
        ).scalar_one()
        assert atraxa.cmc == 4
        assert "Creature" in _json.loads(atraxa.types_json)

        forest = db_session.execute(
            select(CardRow).where(CardRow.name == "Forest")
        ).scalar_one()
        assert forest.cmc == 0
        assert "Land" in _json.loads(forest.types_json)

        deck_cards = {
            db_session.get(CardRow, dc.card_id).name: dc
            for dc in db_session.execute(select(DeckCardRow)).scalars().all()
        }
        assert deck_cards["Atraxa, Praetors' Voice"].quantity == 1
        assert deck_cards["Atraxa, Praetors' Voice"].is_commander is True
        assert deck_cards["Forest"].quantity == 5
        assert deck_cards["Forest"].is_commander is False

    def test_includes_lands(self, db_session: Session):
        """save_deck_cards persists lands too -- they're needed for composition."""
        deck = get_or_create_deck(db_session, "test-deck")
        deck_list = [
            {"name": "Sol Ring", "cmc": 1, "types": ["Artifact"], "quantity": 1},
            {"name": "Plains", "cmc": 0, "types": ["Basic", "Land"], "quantity": 10},
        ]
        save_deck_cards(db_session, deck, deck_list, {})
        db_session.commit()

        deck_cards = db_session.execute(select(DeckCardRow)).scalars().all()
        assert len(deck_cards) == 2
        names = {db_session.get(CardRow, dc.card_id).name for dc in deck_cards}
        assert names == {"Sol Ring", "Plains"}


# ---------------------------------------------------------------------------
# save_simulation_run tests
# ---------------------------------------------------------------------------

def _make_results(land_counts=(37, 38)):
    """Build mock result dicts for testing."""
    results = []
    for lc in land_counts:
        results.append({
            "land_count": lc,
            "mean_mana": 7.5 + lc * 0.1,
            "mean_draws": 3.0,
            "mean_bad_turns": 1.0,
            "mean_lands": 3.5,
            "mean_mulls": 0.2,
            "ci_mean_mana": [7.0, 8.0],
            "ci_consistency": [0.80, 0.90],
            "consistency": 0.85 + (lc - 37) * 0.02,
            "percentile_25": 6.0,
            "percentile_50": 7.5,
            "percentile_75": 9.0,
            "card_performance": {
                "low_performing": [
                    {"name": "Bad Card", "effects": "draws 1", "mean_with": 8.0, "mean_without": 8.3, "score": -0.3},
                    {"name": "Effectless Card", "effects": "", "mean_with": 8.1, "mean_without": 8.2, "score": -0.1},
                ],
                "high_performing": [],
            },
        })
    return results


class TestSaveSimulationRun:
    def test_creates_run_and_results(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        results = _make_results()
        config = {"turns": 10, "sims": 1000, "min_lands": 37, "max_lands": 38}

        run = save_simulation_run(db_session, "job123", deck, config, results)
        db_session.commit()

        assert run.id is not None
        assert run.job_id == "job123"
        assert run.optimal_land_count == 38  # Higher consistency

        result_rows = db_session.execute(select(SimulationResultRow)).scalars().all()
        assert len(result_rows) == 2

    def test_optimal_land_count(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        results = _make_results(land_counts=(36, 37, 38))
        config = {"turns": 10, "sims": 1000, "min_lands": 36, "max_lands": 38}

        run = save_simulation_run(db_session, "job456", deck, config, results)
        db_session.commit()

        assert run.optimal_land_count == 38

    def test_card_performance_filters_effectless(self, db_session: Session):
        """Only cards with non-empty effects description are stored."""
        deck = get_or_create_deck(db_session, "test-deck")
        results = _make_results(land_counts=(38,))
        config = {"turns": 10, "sims": 1000, "min_lands": 38, "max_lands": 38}

        save_simulation_run(db_session, "job789", deck, config, results)
        db_session.commit()

        perf_rows = db_session.execute(select(CardPerformanceRow)).scalars().all()
        # Only "Bad Card" has effects, "Effectless Card" should be filtered out
        assert len(perf_rows) == 1
        card = db_session.get(CardRow, perf_rows[0].card_id)
        assert card.name == "Bad Card"

    def test_card_performance_rank(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        results = _make_results(land_counts=(38,))
        config = {"turns": 10, "sims": 1000, "min_lands": 38, "max_lands": 38}

        save_simulation_run(db_session, "jobrank", deck, config, results)
        db_session.commit()

        perf_rows = db_session.execute(select(CardPerformanceRow)).scalars().all()
        assert perf_rows[0].rank == 1

    def test_empty_results(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        config = {"turns": 10, "sims": 1000, "min_lands": 36, "max_lands": 39}

        run = save_simulation_run(db_session, "empty_job", deck, config, [])
        db_session.commit()

        assert run.optimal_land_count is None
        result_rows = db_session.execute(select(SimulationResultRow)).scalars().all()
        assert len(result_rows) == 0

    def test_raw_stats_persisted(self, db_session: Session):
        """save_simulation_run stores deck_raw values into raw_* columns."""
        deck = get_or_create_deck(db_session, "test-deck")
        results = _make_results(land_counts=(38,))
        # Inject deck_raw on the input dict (as result_to_dict now does).
        results[0]["deck_raw"] = {
            "consistency": 0.62,
            "acceleration": 7.5,
            "snowball": 1.8,
            "toughness": 0.81,
            "efficiency": 0.55,
            "reach": 22.0,
        }
        config = {"turns": 10, "sims": 1000, "min_lands": 38, "max_lands": 38}

        save_simulation_run(db_session, "rawjob", deck, config, results)
        db_session.commit()

        row = db_session.execute(select(SimulationResultRow)).scalar_one()
        assert row.raw_consistency == pytest.approx(0.62)
        assert row.raw_acceleration == pytest.approx(7.5)
        assert row.raw_snowball == pytest.approx(1.8)
        assert row.raw_toughness == pytest.approx(0.81)
        assert row.raw_efficiency == pytest.approx(0.55)
        assert row.raw_reach == pytest.approx(22.0)

    def test_raw_stats_nullable_when_absent(self, db_session: Session):
        """Legacy result dicts with no deck_raw key save NULLs (not a crash)."""
        deck = get_or_create_deck(db_session, "test-deck")
        results = _make_results(land_counts=(38,))
        # No deck_raw key on this dict.
        config = {"turns": 10, "sims": 1000, "min_lands": 38, "max_lands": 38}

        save_simulation_run(db_session, "legacyjob", deck, config, results)
        db_session.commit()

        row = db_session.execute(select(SimulationResultRow)).scalar_one()
        assert row.raw_consistency is None
        assert row.raw_reach is None

    def test_run_config_stored(self, db_session: Session):
        deck = get_or_create_deck(db_session, "test-deck")
        config = {
            "turns": 12,
            "sims": 5000,
            "min_lands": 35,
            "max_lands": 40,
            "seed": 42,
            "mulligan": "curve_aware",
        }

        run = save_simulation_run(db_session, "cfg_job", deck, config, [])
        db_session.commit()

        assert run.turns == 12
        assert run.sims == 5000
        assert run.min_lands == 35
        assert run.max_lands == 40
        assert run.seed == 42
        assert run.mulligan_strategy == "curve_aware"
