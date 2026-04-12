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
