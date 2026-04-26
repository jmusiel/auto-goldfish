"""Tests for the Moxfield adapter."""

from unittest.mock import patch, MagicMock

import pytest

from auto_goldfish.decklist.moxfield import (
    MoxfieldAPIError,
    _extract_deck_id,
    _get_user_agent,
    fetch_decklist,
)
from auto_goldfish.decklist import rate_limiter


@pytest.fixture(autouse=True)
def reset_limiter():
    rate_limiter.reset()
    yield
    rate_limiter.reset()


class TestExtractDeckId:
    def test_standard_url(self):
        assert _extract_deck_id("https://www.moxfield.com/decks/AbCdEf123") == "AbCdEf123"

    def test_no_www(self):
        assert _extract_deck_id("https://moxfield.com/decks/XyZ789") == "XyZ789"

    def test_trailing_slash(self):
        assert _extract_deck_id("https://www.moxfield.com/decks/AbCdEf123/") == "AbCdEf123"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid Moxfield URL"):
            _extract_deck_id("https://archidekt.com/decks/123/test")

    def test_deck_id_with_hyphens_and_underscores(self):
        assert _extract_deck_id("https://moxfield.com/decks/a-b_c") == "a-b_c"


class TestGetUserAgent:
    def test_default_identifies_project(self, monkeypatch):
        monkeypatch.delenv("MOXFIELD_USER_AGENT", raising=False)
        ua = _get_user_agent()
        assert ua.startswith("auto-goldfish/")
        assert "github.com/jmusiel/auto-goldfish" in ua

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MOXFIELD_USER_AGENT", "MoxKey; partner_user partner_pass")
        assert _get_user_agent() == "MoxKey; partner_user partner_pass"

    def test_blank_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MOXFIELD_USER_AGENT", "  ")
        assert _get_user_agent().startswith("auto-goldfish/")


class TestFetchDecklist:
    SAMPLE_RESPONSE = {
        "boards": {
            "commanders": {
                "cards": {
                    "vren": {
                        "card": {"name": "Vren, the Relentless"},
                        "quantity": 1,
                    }
                }
            },
            "mainboard": {
                "cards": {
                    "sol-ring": {
                        "card": {"name": "Sol Ring"},
                        "quantity": 1,
                    },
                    "island": {
                        "card": {"name": "Island"},
                        "quantity": 36,
                    },
                }
            },
        }
    }

    def _mock_moxfield_response(self, data, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = data
        resp.raise_for_status = MagicMock()
        return resp

    @patch("auto_goldfish.decklist.moxfield.resolve_cards")
    @patch("auto_goldfish.decklist.moxfield.requests.get")
    def test_fetch_parses_commanders_and_mainboard(self, mock_get, mock_resolve, monkeypatch):
        monkeypatch.delenv("MOXFIELD_USER_AGENT", raising=False)
        mock_get.return_value = self._mock_moxfield_response(self.SAMPLE_RESPONSE)
        mock_resolve.return_value = [{"name": "test"}]

        fetch_decklist("https://www.moxfield.com/decks/abc123")

        entries = mock_resolve.call_args[0][0]
        names = {name for _, name, _ in entries}
        assert "Vren, the Relentless" in names
        assert "Sol Ring" in names
        assert "Island" in names

        cmdr_entries = [(q, n, c) for q, n, c in entries if c]
        assert len(cmdr_entries) == 1
        assert cmdr_entries[0][1] == "Vren, the Relentless"

    @patch("auto_goldfish.decklist.moxfield.resolve_cards")
    @patch("auto_goldfish.decklist.moxfield.requests.get")
    def test_fetch_sends_default_user_agent(self, mock_get, mock_resolve, monkeypatch):
        monkeypatch.delenv("MOXFIELD_USER_AGENT", raising=False)
        mock_get.return_value = self._mock_moxfield_response(self.SAMPLE_RESPONSE)
        mock_resolve.return_value = []

        fetch_decklist("https://www.moxfield.com/decks/abc123")

        _, kwargs = mock_get.call_args
        ua = kwargs["headers"]["User-Agent"]
        assert ua.startswith("auto-goldfish/")
        assert "github.com/jmusiel/auto-goldfish" in ua

    @patch("auto_goldfish.decklist.moxfield.resolve_cards")
    @patch("auto_goldfish.decklist.moxfield.requests.get")
    def test_fetch_honors_user_agent_override(self, mock_get, mock_resolve, monkeypatch):
        monkeypatch.setenv("MOXFIELD_USER_AGENT", "MoxKey; secret_user secret_pass")
        mock_get.return_value = self._mock_moxfield_response(self.SAMPLE_RESPONSE)
        mock_resolve.return_value = []

        fetch_decklist("https://www.moxfield.com/decks/abc123")

        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["User-Agent"] == "MoxKey; secret_user secret_pass"

    @patch("auto_goldfish.decklist.moxfield.requests.get")
    def test_fetch_404_raises(self, mock_get, monkeypatch):
        monkeypatch.delenv("MOXFIELD_USER_AGENT", raising=False)
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp

        with pytest.raises(MoxfieldAPIError, match="not found"):
            fetch_decklist("https://www.moxfield.com/decks/nonexistent")

    @patch("auto_goldfish.decklist.moxfield.requests.get")
    def test_fetch_empty_deck_raises(self, mock_get, monkeypatch):
        monkeypatch.delenv("MOXFIELD_USER_AGENT", raising=False)
        mock_get.return_value = self._mock_moxfield_response({"boards": {}})

        with pytest.raises(MoxfieldAPIError, match="No cards found"):
            fetch_decklist("https://www.moxfield.com/decks/empty")

    @patch("auto_goldfish.decklist.moxfield.resolve_cards")
    @patch("auto_goldfish.decklist.moxfield.requests.get")
    def test_fetch_respects_rate_limiter(self, mock_get, mock_resolve, monkeypatch):
        monkeypatch.delenv("MOXFIELD_USER_AGENT", raising=False)
        mock_get.return_value = self._mock_moxfield_response(self.SAMPLE_RESPONSE)
        mock_resolve.return_value = []

        with patch("auto_goldfish.decklist.moxfield.rate_limiter.wait") as mock_wait:
            fetch_decklist("https://www.moxfield.com/decks/abc123")
            mock_wait.assert_called_with("moxfield")
