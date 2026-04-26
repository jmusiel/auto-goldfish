"""Regression test for the commander-vs-decklist index collision bug.

Card indices are stored directly in zone lists (state.battlefield,
state.command_zone, etc.). Before the fix, commanders started at index 0
*overlapping* with the decklist's index 0, so once a commander was cast onto
the battlefield the replay snapshot's "battlefield" list would render the
*non-commander* decklist[0] card name instead of the commander's name.

The fix renumbers commanders to start at Goldfisher._COMMANDER_INDEX_OFFSET
(a fixed high constant — the optimizer mutates the decklist mid-run, so a
length-based boundary is unstable), making indices disjoint, and routes all
zone-derived lookups through Goldfisher._card_by_id.
"""

from __future__ import annotations

import random

from auto_goldfish.engine.goldfisher import Goldfisher


def _build_deck_with_low_cost_commander():
    """Deck where the commander costs 1 (so it's cast on turn 1) and the
    first non-commander decklist card has a clearly distinct name."""
    deck = [
        {
            "name": "DISTINCT_DECK_INDEX_0_CARD",
            "cmc": 6, "cost": "{6}", "text": "",
            "types": ["Creature"], "commander": False,
        },
        {
            "name": "Test Commander",
            "cmc": 1, "cost": "{B}", "text": "",
            "types": ["Creature"], "commander": True,
        },
    ]
    for i in range(40):
        deck.append({
            "name": f"Swamp {i}", "cmc": 0, "cost": "", "text": "",
            "types": ["Land"], "commander": False,
        })
    for i in range(57):
        deck.append({
            "name": f"Filler {i}", "cmc": 6, "cost": "{6}",
            "text": "", "types": ["Creature"], "commander": False,
        })
    return deck


class TestCommanderIndicesAreDisjointFromDecklist:
    def test_commander_indices_use_fixed_offset(self):
        gf = Goldfisher(_build_deck_with_low_cost_commander(),
                        turns=4, sims=1, seed=1)
        assert len(gf.commanders) == 1
        assert gf.commanders[0].index == Goldfisher._COMMANDER_INDEX_OFFSET
        # And the offset must comfortably exceed any realistic decklist size,
        # so optimizer mutations can't accidentally collide with it.
        assert Goldfisher._COMMANDER_INDEX_OFFSET > len(gf.decklist) * 100

    def test_card_by_id_resolves_both_ranges(self):
        gf = Goldfisher(_build_deck_with_low_cost_commander(),
                        turns=4, sims=1, seed=1)
        assert gf._card_by_id(0) is gf.decklist[0]
        assert gf._card_by_id(0).name == "DISTINCT_DECK_INDEX_0_CARD"
        assert gf._card_by_id(gf.commanders[0].index) is gf.commanders[0]
        assert gf._card_by_id(gf.commanders[0].index).name == "Test Commander"


class TestCommanderShowsOnBattlefieldInReplay:
    def _run_one_turn_and_snapshot(self, gf):
        """Drive a single turn with the same logic the engine uses for
        replay capture and return the resulting snapshot dict."""
        random.seed(1)
        state = gf._reset()
        gf._mulligan(state)
        played = gf._take_turn(state)
        return played, {
            "hand_after": [gf._card_by_id(idx).name for idx in state.hand],
            "battlefield": [gf._card_by_id(idx).name for idx in state.battlefield],
            "lands": [gf._card_by_id(idx).name for idx in state.lands],
            "graveyard": [gf._card_by_id(idx).name for idx in state.yard],
        }

    def test_commander_cast_appears_on_battlefield_with_correct_name(self):
        gf = Goldfisher(_build_deck_with_low_cost_commander(),
                        turns=4, sims=1, seed=1)
        played, snap = self._run_one_turn_and_snapshot(gf)

        # The 1-mana commander should be cast on turn 1 (we have a Swamp in
        # the opening hand thanks to mulligans + a land-rich deck).
        played_names = [c.name for c in played]
        assert "Test Commander" in played_names

        # And it should appear on the rendered battlefield under its real
        # name — not aliased to DISTINCT_DECK_INDEX_0_CARD via the index
        # collision bug.
        assert "Test Commander" in snap["battlefield"], (
            f"commander missing from battlefield snapshot: {snap['battlefield']}"
        )
        assert "DISTINCT_DECK_INDEX_0_CARD" not in snap["battlefield"], (
            "decklist[0] leaked into battlefield via index-collision bug; "
            f"snapshot was {snap['battlefield']}"
        )

    def test_full_simulate_battlefield_snapshots_contain_commander(self):
        """End-to-end: run the public simulate() pipeline with replay
        capture and verify no captured snapshot mentions the impostor card
        on the battlefield once the commander has been cast."""
        gf = Goldfisher(_build_deck_with_low_cost_commander(),
                        turns=8, sims=200, seed=42, workers=1)
        result = gf.simulate()

        captured_any = False
        for bucket in ("top", "mid", "low"):
            for game in result.replay_data.get(bucket, []):
                for turn_snap in game.get("turns", []):
                    captured_any = True
                    assert "DISTINCT_DECK_INDEX_0_CARD" not in turn_snap["battlefield"], (
                        f"impostor card on battlefield in {bucket} replay turn "
                        f"{turn_snap['turn']}: {turn_snap['battlefield']}"
                    )
        assert captured_any, "expected at least one captured replay snapshot"
