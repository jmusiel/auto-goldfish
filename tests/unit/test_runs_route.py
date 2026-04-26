"""Tests for the /runs route, focused on calibration UI surfacing."""

from __future__ import annotations

import json

import pytest

from auto_goldfish.metrics.calibration import reset_cache
from auto_goldfish.web import create_app


@pytest.fixture(autouse=True)
def _clear_calibration_cache():
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_runs_page_returns_200(client):
    response = client.get("/runs/")
    assert response.status_code == 200


def test_runs_page_calibration_default_when_disabled(client, monkeypatch):
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "0")
    reset_cache()
    response = client.get("/runs/")
    assert b"Default anchors" in response.data
    assert b"calibration-badge-on" not in response.data


def test_runs_api_calibration_null_when_disabled(client, monkeypatch):
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "0")
    reset_cache()
    response = client.get("/runs/api/data")
    payload = response.get_json()
    assert payload["calibration"] is None


def test_runs_page_calibration_badge_on_when_enabled(client, monkeypatch):
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "1")

    fake_anchors = type("A", (), {
        "consistency": (0.1, 0.9),
        "acceleration": (3.0, 11.0),
        "snowball_ratio": (1.0, 3.5),
        "snowball_late_avg_norm": (1.0, 8.0),
        "toughness": (0.6, 1.0),
        "efficiency": (0.2, 0.8),
        "reach_norm": (15.0, 45.0),
    })()
    fake_meta = type("M", (), {
        "n_rows": 12, "n_decks": 9, "pseudo_count": 76,
        "low_pct": 10.0, "high_pct": 90.0,
    })()

    monkeypatch.setattr(
        "auto_goldfish.web.routes.runs.__name__", "auto_goldfish.web.routes.runs",
    )

    def fake_get_active_anchors():
        return fake_anchors, fake_meta

    monkeypatch.setattr(
        "auto_goldfish.metrics.calibration.get_active_anchors",
        fake_get_active_anchors,
    )

    response = client.get("/runs/")
    assert b"calibration-badge-on" in response.data
    assert b"N=12 from 9 decks" in response.data
    assert b"active anchors" in response.data
    # Anchor row formatted to 3 decimals
    assert b"(0.100, 0.900)" in response.data


def test_runs_api_exposes_anchor_pairs(client, monkeypatch):
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "1")

    fake_anchors = type("A", (), {
        "consistency": (0.25, 0.85),
        "acceleration": (3.5, 11.3),
        "snowball_ratio": (1.0, 3.5),
        "snowball_late_avg_norm": (1.0, 8.0),
        "toughness": (0.6, 1.0),
        "efficiency": (0.2, 0.8),
        "reach_norm": (15.0, 45.0),
    })()
    fake_meta = type("M", (), {
        "n_rows": 5, "n_decks": 4, "pseudo_count": 76,
        "low_pct": 10.0, "high_pct": 90.0,
    })()

    monkeypatch.setattr(
        "auto_goldfish.metrics.calibration.get_active_anchors",
        lambda: (fake_anchors, fake_meta),
    )

    response = client.get("/runs/api/data")
    cal = response.get_json()["calibration"]
    assert cal is not None
    assert cal["n_rows"] == 5
    assert cal["n_decks"] == 4
    by_name = {a["name"]: a for a in cal["anchors"]}
    assert by_name["consistency"]["active"] == [0.25, 0.85]
    assert by_name["consistency"]["default"] == [0.0, 1.0]


def test_load_caster_calibration_returns_default_when_disabled(monkeypatch):
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "0")
    reset_cache()

    from auto_goldfish.web.routes.runs import load_caster_calibration

    cal = load_caster_calibration()
    assert cal["calibrated"] is False
    assert cal["n_rows"] == 0
    by_name = {a["name"]: a for a in cal["anchors"]}
    # Active equals default when disabled (no live calibration)
    assert by_name["consistency"]["active"] == by_name["consistency"]["default"]
    # All six CASTER fields + the secondary snowball field are present
    assert set(by_name) == {
        "consistency", "acceleration", "snowball_ratio",
        "snowball_late_avg_norm", "toughness", "efficiency", "reach_norm",
    }


def test_load_caster_calibration_uses_live_meta_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "1")

    fake_anchors = type("A", (), {
        "consistency": (0.2, 0.95),
        "acceleration": (2.0, 10.0),
        "snowball_ratio": (1.0, 3.5),
        "snowball_late_avg_norm": (1.0, 8.0),
        "toughness": (0.6, 1.0),
        "efficiency": (0.2, 0.8),
        "reach_norm": (15.0, 45.0),
    })()
    fake_meta = type("M", (), {
        "n_rows": 7, "n_decks": 6, "pseudo_count": 76,
        "low_pct": 10.0, "high_pct": 90.0,
    })()
    monkeypatch.setattr(
        "auto_goldfish.metrics.calibration.get_active_anchors",
        lambda: (fake_anchors, fake_meta),
    )

    from auto_goldfish.web.routes.runs import load_caster_calibration

    cal = load_caster_calibration()
    assert cal["calibrated"] is True
    assert cal["n_rows"] == 7
    by_name = {a["name"]: a for a in cal["anchors"]}
    assert by_name["consistency"]["active"] == [0.2, 0.95]


def test_simulate_page_exposes_caster_calibration_global(client, tmp_path, monkeypatch):
    """The simulate config page must inject CASTER_CALIBRATION as a JS global
    so client_results.js can render the explanation panel without an extra
    fetch round-trip."""
    deck_path = tmp_path / "testdeck.txt"
    deck_path.write_text("1 Island\n", encoding="utf-8")
    monkeypatch.setattr(
        "auto_goldfish.web.routes.simulation.get_deckpath",
        lambda name: str(deck_path),
    )
    monkeypatch.setattr(
        "auto_goldfish.web.routes.simulation.load_decklist",
        lambda name: [{"name": "Island", "quantity": 1, "types": ["Land"], "cmc": 0}],
    )
    monkeypatch.setattr(
        "auto_goldfish.web.routes.simulation.load_overrides",
        lambda name: {},
    )
    monkeypatch.setenv("AUTO_GOLDFISH_CALIBRATE", "0")
    reset_cache()

    response = client.get("/sim/testdeck")
    assert response.status_code == 200
    body = response.data.decode("utf-8")
    assert "window.CASTER_CALIBRATION" in body
    # Default-anchor payload is serialized as JSON (calibrated=false)
    assert '"calibrated": false' in body or '"calibrated":false' in body
