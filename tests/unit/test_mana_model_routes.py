"""Tests for the mana model web API routes."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from auto_goldfish.web import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _make_deck_json():
    """Minimal deck list for testing."""
    lands = [
        {"name": f"Island_{i}", "cmc": 0, "types": ["Land"], "quantity": 1,
         "oracle_cmc": 0, "cost": "", "text": "", "commander": False}
        for i in range(36)
    ]
    spells = [
        {"name": f"Bear_{i}", "cmc": 2, "types": ["Creature"], "quantity": 1,
         "oracle_cmc": 2, "cost": "{1}{G}", "text": "", "commander": False}
        for i in range(63)
    ]
    return lands + spells


class TestManaModelPage:
    def test_page_get_renders(self, client):
        """GET /mana-model/<deck> should render the mana model template."""
        deck_name = "__test_mana_page__"
        deck_data = _make_deck_json()

        with patch("auto_goldfish.web.routes.mana_model.load_decklist", return_value=deck_data):
            with patch("auto_goldfish.web.routes.mana_model.get_deckpath") as mock_path:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                    f.write(b"[]")
                    mock_path.return_value = f.name
                try:
                    resp = client.get(f"/mana-model/{deck_name}")
                finally:
                    os.unlink(f.name)

        assert resp.status_code == 200
        assert b"Mana Model" in resp.data

    def test_page_post_renders_local(self, client):
        """POST /mana-model/<deck> should render for localStorage decks."""
        deck_data = _make_deck_json()
        resp = client.post(
            "/mana-model/local_deck",
            data=json.dumps({"cards": deck_data}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert b"Mana Model" in resp.data
        assert b"Go to Simulator" in resp.data

    def test_page_404_for_missing_deck(self, client):
        with patch("auto_goldfish.web.routes.mana_model.get_deckpath", return_value="/nonexistent/path.json"):
            resp = client.get("/mana-model/no_such_deck")
        assert resp.status_code == 404


class TestAnalysisEndpoint:
    def test_analysis_returns_json(self, client):
        """GET /mana-model/api/<deck>/analysis should return JSON with expected keys."""
        deck_name = "__test_mana_model__"
        deck_data = _make_deck_json()

        with patch("auto_goldfish.web.routes.mana_model.load_decklist", return_value=deck_data):
            with patch("auto_goldfish.web.routes.mana_model.load_overrides", return_value={}):
                with patch("auto_goldfish.web.routes.mana_model.get_deckpath") as mock_path:
                    # Create a temp file so os.path.isfile returns True
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                        f.write(b"[]")
                        mock_path.return_value = f.name

                    try:
                        resp = client.get(f"/mana-model/api/{deck_name}/analysis")
                    finally:
                        os.unlink(f.name)

        assert resp.status_code == 200
        data = resp.get_json()
        assert "recommendation" in data
        assert "current_mana_table" in data
        assert "comparison" in data
        assert "mulligan" in data
        assert "composition" in data

    def test_analysis_404_for_missing_deck(self, client):
        with patch("auto_goldfish.web.routes.mana_model.get_deckpath", return_value="/nonexistent/path.json"):
            resp = client.get("/mana-model/api/no_such_deck/analysis")
        assert resp.status_code == 404

    def test_analysis_post_with_card_data(self, client):
        """POST /mana-model/api/<deck>/analysis should accept cards in body (localStorage decks)."""
        deck_data = _make_deck_json()
        resp = client.post(
            "/mana-model/api/local_deck/analysis",
            data=json.dumps({"cards": deck_data, "overrides": {}}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recommendation" in data
        assert data["composition"]["land_count"] == 36
        assert data["composition"]["deck_size"] == 99


class TestCalculateEndpoint:
    def test_calculate_returns_json(self, client):
        resp = client.post(
            "/mana-model/api/calculate",
            data=json.dumps({"deck_size": 99, "land_count": 36, "max_turn": 5}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "mana_table" in data
        assert "mulligan_rate" in data
        assert len(data["mana_table"]) == 5

    def test_calculate_default_values(self, client):
        resp = client.post(
            "/mana-model/api/calculate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["deck_size"] == 99
        assert data["land_count"] == 36

    def test_calculate_with_ramp_draw_returns_more_mana(self, client):
        """Ramp/draw should increase expected_mana on later turns vs baseline."""
        baseline = client.post(
            "/mana-model/api/calculate",
            data=json.dumps({"deck_size": 99, "land_count": 36, "max_turn": 6}),
            content_type="application/json",
        ).get_json()
        boosted = client.post(
            "/mana-model/api/calculate",
            data=json.dumps({"deck_size": 99, "land_count": 36, "max_turn": 6,
                             "ramp_cards": 10, "draw_cards": 10}),
            content_type="application/json",
        ).get_json()
        assert boosted["mana_table"][4]["expected_mana"] > baseline["mana_table"][4]["expected_mana"]

    def test_calculate_handles_empty_body(self, client):
        """No body should not 500 -- should fall back to defaults."""
        resp = client.post("/mana-model/api/calculate")
        assert resp.status_code == 200


class TestPartnerCommanderFilter:
    def _partner_deck(self):
        lands = [
            {"name": f"Forest_{i}", "cmc": 0, "types": ["Land"], "quantity": 1,
             "oracle_cmc": 0, "cost": "", "text": "", "commander": False}
            for i in range(36)
        ]
        partners = [
            {"name": "Tymna the Weaver", "cmc": 2, "types": ["Creature"], "quantity": 1,
             "oracle_cmc": 2, "cost": "{1}{W}", "text": "", "commander": True},
            {"name": "Tana the Bloodsower", "cmc": 4, "types": ["Creature"], "quantity": 1,
             "oracle_cmc": 4, "cost": "{2}{R}{G}", "text": "", "commander": True},
        ]
        spells = [
            {"name": f"Bear_{i}", "cmc": 2, "types": ["Creature"], "quantity": 1,
             "oracle_cmc": 2, "cost": "{1}{G}", "text": "", "commander": False}
            for i in range(61)
        ]
        return lands + partners + spells

    def test_partner_deck_returns_both_commanders(self, client):
        resp = client.post(
            "/mana-model/api/local_partners/analysis",
            data=json.dumps({"cards": self._partner_deck()}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["composition"]["commanders"]) == 2
        assert data["active_commander_cmcs"] == [2, 4]
        assert "partner_castable_prob" in data["recommendation"]

    def test_filter_to_single_partner(self, client):
        resp = client.post(
            "/mana-model/api/local_partners/analysis",
            data=json.dumps({
                "cards": self._partner_deck(),
                "commander_filter": "0",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active_commander_cmcs"] == [2]
        assert "partner_castable_prob" not in data["recommendation"]

    def test_filter_invalid_index_falls_back_to_all(self, client):
        resp = client.post(
            "/mana-model/api/local_partners/analysis",
            data=json.dumps({
                "cards": self._partner_deck(),
                "commander_filter": "99",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active_commander_cmcs"] == [2, 4]

    def test_single_commander_unaffected(self, client):
        """Single-commander deck must not get partner_castable_prob."""
        deck = _make_deck_json()
        # Mark one as commander.
        deck[36]["commander"] = True
        deck[36]["cmc"] = 4
        resp = client.post(
            "/mana-model/api/local_solo/analysis",
            data=json.dumps({"cards": deck}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["composition"]["commanders"]) == 1
        assert "partner_castable_prob" not in data["recommendation"]
