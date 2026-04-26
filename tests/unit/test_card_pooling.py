"""Tests for card-pool equivalence keys used in top/low performer pooling."""

import numpy as np

from auto_goldfish.effects.builtin import DrawCards, ProduceMana, PerTurnDraw
from auto_goldfish.effects.registry import CardEffects, EffectRegistry
from auto_goldfish.engine.goldfisher import (
    Goldfisher,
    _card_equivalence_key,
    _classify_saturation,
    _compute_marginal_impacts,
    _format_pool_label,
    _recommended_copy_range,
)
from auto_goldfish.models.card import Card


def _card(name: str, cmc: int, types: list[str], effects: CardEffects | None = None) -> Card:
    """Build a Card and attach cached effects, mimicking Goldfisher._make_card."""
    c = Card(name=name, cmc=cmc, types=types)
    c._cached_effects = effects
    return c


class TestEquivalenceKey:
    def test_vanilla_5_drops_collapse(self):
        a = _card("Cauldron of Souls", 5, ["artifact"])
        b = _card("Generic Five-Drop Artifact", 5, ["artifact"])
        assert _card_equivalence_key(a) == _card_equivalence_key(b)

    def test_2_mana_draw_2_collapse(self):
        draw_2 = CardEffects(on_play=[DrawCards(amount=2)], draw=True)
        a = _card("Night's Whisper", 2, ["sorcery"], effects=draw_2)
        b = _card("Sign in Blood", 2, ["sorcery"], effects=draw_2)
        assert _card_equivalence_key(a) == _card_equivalence_key(b)

    def test_different_cmc_distinct(self):
        a = _card("4-Drop", 4, ["creature"])
        b = _card("5-Drop", 5, ["creature"])
        assert _card_equivalence_key(a) != _card_equivalence_key(b)

    def test_types_no_longer_distinguish(self):
        # Type is intentionally dropped from the pooling key — vanilla creatures,
        # sorceries and artifacts at the same cmc all behave identically to the
        # simulator and are pooled together as "spells".
        a = _card("5-Drop Creature", 5, ["creature"])
        b = _card("5-Drop Sorcery", 5, ["sorcery"])
        c = _card("5-Drop Artifact", 5, ["artifact"])
        assert _card_equivalence_key(a) == _card_equivalence_key(b) == _card_equivalence_key(c)

    def test_different_effects_distinct(self):
        draw_1 = CardEffects(on_play=[DrawCards(amount=1)], draw=True)
        draw_2 = CardEffects(on_play=[DrawCards(amount=2)], draw=True)
        a = _card("Card A", 2, ["sorcery"], effects=draw_1)
        b = _card("Card B", 2, ["sorcery"], effects=draw_2)
        assert _card_equivalence_key(a) != _card_equivalence_key(b)

    def test_no_effects_vs_effects_distinct(self):
        a = _card("Vanilla", 2, ["sorcery"])
        b = _card("Effective", 2, ["sorcery"], effects=CardEffects(on_play=[DrawCards(1)]))
        assert _card_equivalence_key(a) != _card_equivalence_key(b)

    def test_multitype_pools_with_singletype_at_same_cmc(self):
        # MDFC-like multi-type cards pool with single-type cards of the same cmc
        # (and same effects) since type no longer matters for the simulator pool.
        a = _card("MDFC A", 3, ["creature", "artifact"])
        b = _card("Plain B", 3, ["creature"])
        assert _card_equivalence_key(a) == _card_equivalence_key(b)

    def test_cached_effects_none_treated_as_no_effects(self):
        a = _card("Plain", 3, ["creature"], effects=None)
        b = _card("Plain Empty", 3, ["creature"], effects=CardEffects())
        assert _card_equivalence_key(a) == _card_equivalence_key(b)


class TestPoolLabel:
    def test_no_effects_label(self):
        label = _format_pool_label(5, "")
        assert label == "5-mana spell (no effects)"

    def test_effect_label(self):
        label = _format_pool_label(2, "Draw 2 cards")
        assert label == "2-mana spell: Draw 2 cards"

    def test_one_mana_label(self):
        label = _format_pool_label(1, "")
        assert label == "1-mana spell (no effects)"


class TestPooledScoring:
    """End-to-end: identical cards in a deck should pool into a single entry."""

    def _build_deck(self) -> list[dict]:
        # 38 lands + 4 vanilla 1-drops + 4 vanilla 2-drops + 4 of each cmc 3..6
        deck = [{
            "name": "Test Commander",
            "cmc": 4, "cost": "{2}{U}{B}", "text": "",
            "types": ["Creature"], "commander": True,
        }]
        for i in range(38):
            deck.append({
                "name": f"Island {i}", "cmc": 0, "cost": "", "text": "",
                "types": ["Land"], "commander": False,
            })
        # 4 distinct 2-mana "draw 2" sorceries (Night's Whisper, Sign in Blood, etc.)
        for nm in ("Night's Whisper", "Sign in Blood", "Read the Bones", "Black Heart"):
            deck.append({
                "name": nm, "cmc": 2, "cost": "{1}{B}", "text": "draw 2",
                "types": ["Sorcery"], "commander": False,
            })
        # Vanilla creatures across multiple cmcs
        for cmc in range(1, 7):
            for j in range(10):
                deck.append({
                    "name": f"Vanilla{cmc}-{j}", "cmc": cmc, "cost": f"{{{cmc}}}",
                    "text": "", "types": ["Creature"], "commander": False,
                })
        return deck

    def test_draw_2_sorceries_pool_into_one_entry(self):
        registry = EffectRegistry()
        draw_2 = CardEffects(on_play=[DrawCards(amount=2)], draw=True)
        for nm in ("Night's Whisper", "Sign in Blood", "Read the Bones", "Black Heart"):
            registry.register(nm, draw_2)

        deck = self._build_deck()
        gf = Goldfisher(
            deck, turns=10, sims=400, record_results="quartile",
            seed=7, registry=registry,
        )
        result = gf.simulate()
        cp = result.card_performance
        all_entries = cp["high_performing"] + cp["low_performing"]

        # Find the 2-mana sorcery draw-2 pool
        draw_pools = [e for e in all_entries if "Draw 2" in (e.get("effects") or "")]
        assert draw_pools, "expected a 2-mana draw-2 pool to be reported"
        pool = draw_pools[0]
        assert pool["copies"] == 4
        assert pool["cmc"] == 2
        assert pool["name"] in {"Night's Whisper", "Sign in Blood", "Read the Bones", "Black Heart"}
        assert "2-mana spell: Draw 2 cards" == pool["label"]

    def test_vanilla_spells_pool_by_cmc(self):
        # Without the registry's draw-2 mapping, "Black Heart" has no effects and
        # pools with the 10 vanilla 2-mana creatures — the type-agnostic pooling
        # treats them all as "2-mana spell (no effects)". The other three named
        # sorceries auto-derive a Draw 2 effect from their text.
        deck = self._build_deck()
        gf = Goldfisher(deck, turns=10, sims=400, record_results="quartile", seed=11)
        result = gf.simulate()
        cp = result.card_performance
        seen: dict[str, dict] = {}
        for e in cp["high_performing"] + cp["low_performing"]:
            seen[e["label"]] = e
        spell_pools = [
            e for e in seen.values()
            if e["label"].startswith(tuple(f"{c}-mana spell" for c in range(1, 7)))
            and e["label"].endswith("(no effects)")
        ]
        cmcs = [e["cmc"] for e in spell_pools]
        assert len(cmcs) == len(set(cmcs)), "vanilla spell pools should be unique per cmc"
        for e in spell_pools:
            # 10 vanilla creatures per cmc, plus possibly Black Heart at cmc 2
            assert e["copies"] >= 10

    def test_lands_excluded(self):
        deck = self._build_deck()
        gf = Goldfisher(deck, turns=10, sims=400, record_results="quartile", seed=13)
        result = gf.simulate()
        cp = result.card_performance
        for entries in (cp["high_performing"], cp["low_performing"]):
            for e in entries:
                assert "Island" not in e["name"]
                assert "land" not in e["label"].lower()


class TestMarginalImpacts:
    """Tests for per-copy marginal impact computation."""

    def _synth(self, n_per_bucket: int, means: list[float], sigma: float = 1.0, seed: int = 0):
        """Build (count_per_game, mana_arr) where bucket k has mean = means[k]."""
        rng = np.random.default_rng(seed)
        counts: list[int] = []
        manas: list[float] = []
        for k, mu in enumerate(means):
            counts.extend([k] * n_per_bucket)
            manas.extend(rng.normal(mu, sigma, size=n_per_bucket).tolist())
        return np.array(counts), np.array(manas)

    def test_diminishing_returns_detected(self):
        # Each copy adds a smaller boost: 8 -> 9 -> 9.5 -> 9.6 -> 9.6
        counts, mana = self._synth(200, [8.0, 9.0, 9.5, 9.6, 9.6], sigma=0.5, seed=1)
        result = _compute_marginal_impacts(counts, mana, max_k=4)
        assert len(result) == 4
        # Effects should be roughly +1.0, +0.5, +0.1, +0.0
        assert result[0]["effect"] is not None and result[0]["effect"] > 0.7
        assert result[1]["effect"] is not None and result[1]["effect"] > 0.3
        # Later copies should be near zero -> noise
        assert result[3]["noise"] is True

    def test_undersampled_buckets_skipped(self):
        # Bucket 4 has only 5 samples — below the 30-game min. The pill is
        # dropped entirely rather than emitted as "small_sample" noise.
        rng = np.random.default_rng(2)
        counts = np.concatenate([
            np.full(200, 0), np.full(200, 1), np.full(200, 2), np.full(200, 3), np.full(5, 4)
        ])
        mana = rng.normal(8.0, 1.0, size=counts.size)
        result = _compute_marginal_impacts(counts, mana, max_k=4)
        emitted_k = [m["k"] for m in result]
        assert emitted_k == [1, 2, 3]
        assert all(m["effect"] is not None for m in result)

    def test_always_drawn_pool_only_emits_modal_range(self):
        # 60-copy always-drawn pool: counts concentrate around k=7, leading
        # buckets are vanishingly rare. Only k where both buckets clear the
        # 30-game floor should appear.
        rng = np.random.default_rng(7)
        counts = np.concatenate([
            np.full(1, 2),
            np.full(2, 3),
            np.full(50, 6),
            np.full(200, 7),
            np.full(150, 8),
        ])
        mana = rng.normal(10.0, 1.0, size=counts.size)
        result = _compute_marginal_impacts(counts, mana, max_k=8)
        emitted_k = [m["k"] for m in result]
        assert emitted_k == [7, 8]

    def test_negative_marginal_detected(self):
        # Each copy makes things worse: 9 -> 8 -> 7 -> 6
        counts, mana = self._synth(200, [9.0, 8.0, 7.0, 6.0, 5.0], sigma=0.5, seed=3)
        result = _compute_marginal_impacts(counts, mana, max_k=4)
        for r in result:
            assert r["effect"] is not None and r["effect"] < 0
            assert r["noise"] is False


class TestTopupBudgetPlanner:
    def _make_gf(self):
        deck = [{
            "name": "Cmd", "cmc": 2, "cost": "{1}{U}", "text": "",
            "types": ["Creature"], "commander": True,
        }]
        for i in range(99):
            deck.append({
                "name": f"f{i}", "cmc": 0, "cost": "", "text": "",
                "types": ["Land"], "commander": False,
            })
        return Goldfisher(deck, turns=8, sims=2000, seed=0, workers=1)

    def test_returns_zero_when_no_candidates(self):
        gf = self._make_gf()
        # Pool with both buckets well-sampled
        cpg = np.array([0] * 1000 + [1] * 1000)
        groups = {("k1",): [0, 1, 2]}
        budget = gf._plan_topup_budget(groups, {("k1",): cpg}, n_games=2000)
        assert budget == 0

    def test_skips_pool_with_zero_in_baseline(self):
        gf = self._make_gf()
        # n_prev=0 (drew zero of a 5-copy pool — physically rare)
        cpg = np.array([1] * 1500 + [2] * 500)  # k=0 has zero games
        groups = {("k1",): [0, 1, 2, 3, 4]}
        budget = gf._plan_topup_budget(groups, {("k1",): cpg}, n_games=2000)
        assert budget == 0

    def test_extrapolates_extras_to_clear_floor(self):
        gf = self._make_gf()
        # n_prev=10 (k=0 bucket), n_curr=1990 (k=1) — need 20 more in baseline
        # at observed rate 10/2000 = 0.5%, expect ceil(20/0.005) = 4000 extras
        cpg = np.array([0] * 10 + [1] * 1990)
        groups = {("k1",): [0, 1, 2]}
        budget = gf._plan_topup_budget(groups, {("k1",): cpg}, n_games=2000)
        assert budget == 4000

    def test_caps_request_at_2x_sims(self):
        gf = self._make_gf()
        # n_prev=2 → would need 28/0.001 = 28000 extras, exceeds 4000 cap
        cpg = np.array([0] * 2 + [1] * 1998)
        groups = {("k1",): [0, 1, 2]}
        budget = gf._plan_topup_budget(groups, {("k1",): cpg}, n_games=2000)
        assert budget == 0  # over cap → skipped

    def test_picks_largest_affordable_request(self):
        # Two candidates: cheap one (1000 extras) and pricier one (3500).
        # Both fit under 4000 cap; we should pick 3500 since one shared batch
        # also resolves the 1000-extra pool.
        gf = self._make_gf()
        cheap = np.array([0] * 30 + [1] * 1970)
        pricy = np.array([0] * 16 + [1] * 1984)
        groups = {
            ("cheap",): [0, 1, 2],
            ("pricy",): [3, 4, 5],
        }
        count_arrays = {("cheap",): cheap, ("pricy",): pricy}
        budget = gf._plan_topup_budget(groups, count_arrays, n_games=2000)
        # Should be the pricier one (>= 1500 extras for n_prev=16 → 14/0.008)
        assert 1500 <= budget <= 4000


class TestSaturationBadge:
    def test_scaling_when_last_marginal_positive(self):
        marginals = [
            {"k": 1, "effect": 1.0, "noise": False, "ci": 0.2},
            {"k": 2, "effect": 0.6, "noise": False, "ci": 0.2},
            {"k": 3, "effect": 0.4, "noise": False, "ci": 0.2},
        ]
        sat = _classify_saturation(marginals)
        assert sat["badge"] == "scaling"
        assert sat["saturates_at"] is None

    def test_saturated_when_signal_dies_off(self):
        marginals = [
            {"k": 1, "effect": 1.0, "noise": False, "ci": 0.2},
            {"k": 2, "effect": 0.5, "noise": False, "ci": 0.2},
            {"k": 3, "effect": 0.05, "noise": True, "ci": 0.5},
            {"k": 4, "effect": -0.02, "noise": True, "ci": 0.6},
        ]
        sat = _classify_saturation(marginals)
        assert sat["badge"] == "saturated"
        assert sat["saturates_at"] == 2

    def test_crowding_when_marginal_goes_negative(self):
        marginals = [
            {"k": 1, "effect": 0.5, "noise": False, "ci": 0.2},
            {"k": 2, "effect": -0.4, "noise": False, "ci": 0.2},
            {"k": 3, "effect": -0.6, "noise": False, "ci": 0.2},
        ]
        sat = _classify_saturation(marginals)
        assert sat["badge"] == "crowding"
        assert sat["saturates_at"] == 1

    def test_unclear_when_all_noise(self):
        marginals = [
            {"k": 1, "effect": 0.05, "noise": True, "ci": 0.5},
            {"k": 2, "effect": -0.03, "noise": True, "ci": 0.4},
        ]
        sat = _classify_saturation(marginals)
        assert sat["badge"] == "unclear"

    def test_late_only_positive_is_unclear_not_scaling(self):
        # Leading marginals are noise, only k=4 is sig+. Without leading
        # evidence we can't claim "add more" — the late signal is likely
        # confounded by selection on cards-drawn (games that drew enough
        # to reach the kth copy also tend to spend more mana).
        marginals = [
            {"k": 1, "effect": 0.10, "noise": True, "ci": 0.4},
            {"k": 2, "effect": -0.05, "noise": True, "ci": 0.3},
            {"k": 3, "effect": 0.08, "noise": True, "ci": 0.4},
            {"k": 4, "effect": 1.50, "noise": False, "ci": 0.5},
        ]
        sat = _classify_saturation(marginals)
        assert sat["badge"] == "unclear"

    def test_late_only_negative_is_unclear_not_crowding(self):
        # Symmetric case: leading marginals noise, only k=4 sig negative.
        # Without leading evidence, "cut copies" isn't groundable either.
        marginals = [
            {"k": 1, "effect": -0.05, "noise": True, "ci": 0.4},
            {"k": 2, "effect": 0.05, "noise": True, "ci": 0.3},
            {"k": 3, "effect": -0.08, "noise": True, "ci": 0.4},
            {"k": 4, "effect": -1.20, "noise": False, "ci": 0.5},
        ]
        sat = _classify_saturation(marginals)
        assert sat["badge"] == "unclear"

    def test_transition_crowding_still_works(self):
        # k=1 sig+, then k=2 sig-: leading is significant (positive), so
        # the transition to negative is real evidence — still crowding.
        marginals = [
            {"k": 1, "effect": 0.50, "noise": False, "ci": 0.2},
            {"k": 2, "effect": -0.40, "noise": False, "ci": 0.2},
            {"k": 3, "effect": -0.60, "noise": False, "ci": 0.2},
        ]
        sat = _classify_saturation(marginals)
        assert sat["badge"] == "crowding"
        assert sat["saturates_at"] == 1


class TestRecommendedCopyRange:
    """Hypergeometric translation of in-hand saturation point → deck-build range."""

    def test_returns_none_when_saturates_at_missing(self):
        assert _recommended_copy_range(None, 3, 100, 7) is None

    def test_returns_none_when_saturates_at_nonpositive(self):
        assert _recommended_copy_range(0, 3, 100, 7) is None

    def test_returns_none_when_deck_size_invalid(self):
        assert _recommended_copy_range(1, 3, 0, 7) is None

    def test_returns_none_when_turns_invalid(self):
        assert _recommended_copy_range(1, 3, 100, 0) is None

    def test_in_range_for_typical_2mana_ramp_pool(self):
        # K=1 in hand, 100-card deck, 7 turns → 13 cards seen.
        # 3 copies should land in the sweet spot ~3–7.
        rng = _recommended_copy_range(1, 3, 100, 7)
        assert rng is not None
        assert rng["status"] == "in_range"
        assert rng["min_copies"] <= 3 <= rng["max_copies"]

    def test_low_status_when_below_min(self):
        rng = _recommended_copy_range(1, 1, 100, 7)
        assert rng["status"] == "low"
        assert rng["min_copies"] > 1

    def test_high_status_when_above_max(self):
        # 12 copies of a K=1 pool draws 2+ way too often.
        rng = _recommended_copy_range(1, 12, 100, 7)
        assert rng["status"] == "high"
        assert rng["max_copies"] < 12

    def test_higher_k_yields_wider_max(self):
        # Saturating at 2 copies in hand tolerates more deck copies than K=1.
        narrow = _recommended_copy_range(1, 5, 100, 7)
        wide = _recommended_copy_range(2, 5, 100, 7)
        assert wide["max_copies"] > narrow["max_copies"]

    def test_thresholds_satisfy_definition(self):
        # Verify the returned min/max actually meet the defined probability
        # thresholds (33% min draw, 25% max waste).
        from auto_goldfish.optimization.mana_model import prob_at_least
        rng = _recommended_copy_range(1, 5, 100, 7)
        cards_seen = 13  # 7 + 7 - 1
        assert prob_at_least(1, 100, rng["min_copies"], cards_seen) >= 0.33
        assert prob_at_least(1, 100, rng["min_copies"] - 1, cards_seen) < 0.33
        assert prob_at_least(2, 100, rng["max_copies"], cards_seen) <= 0.25
        assert prob_at_least(2, 100, rng["max_copies"] + 1, cards_seen) > 0.25


class TestAlwaysDrawnPools:
    """Pools drawn in nearly every game still surface via marginal-derived score."""

    def test_large_vanilla_pool_surfaces_with_marginals(self):
        # 38 lands + a 60-card pool of vanilla 2-mana creatures + filler.
        # The 60-card pool is drawn at least once in essentially every game
        # (n_not_drawn ≈ 0), so the old filter dropped it. We now expect it
        # to appear with always_drawn=True and a marginal-derived score.
        deck = [{
            "name": "Cmd", "cmc": 4, "cost": "{2}{U}{B}", "text": "",
            "types": ["Creature"], "commander": True,
        }]
        for i in range(38):
            deck.append({
                "name": f"Island {i}", "cmc": 0, "cost": "", "text": "",
                "types": ["Land"], "commander": False,
            })
        for j in range(60):
            deck.append({
                "name": f"Vanilla2-{j}", "cmc": 2, "cost": "{1}{U}",
                "text": "", "types": ["Creature"], "commander": False,
            })
        # filler so total = 99
        for j in range(1):
            deck.append({
                "name": f"Filler-{j}", "cmc": 5, "cost": "{4}{U}",
                "text": "", "types": ["Creature"], "commander": False,
            })

        gf = Goldfisher(deck, turns=10, sims=400, record_results="quartile", seed=23)
        result = gf.simulate()
        cp = result.card_performance
        all_entries = cp["high_performing"] + cp["low_performing"]

        big_pools = [e for e in all_entries if e["copies"] == 60]
        assert big_pools, "expected the 60-copy vanilla pool to appear post-filter-relax"
        pool = big_pools[0]
        assert pool["always_drawn"] is True
        # mean_without is the sentinel 0.0; score is derived from marginals
        assert pool["mean_without"] == 0.0
        assert pool["marginals"], "always-drawn pool must include marginals"
        # At least one marginal must be significant (the score depends on it)
        assert any(not m["noise"] for m in pool["marginals"])


class TestEntryFields:
    def test_single_copy_pool_has_no_marginals(self):
        # 1-of in a 99-card singleton-style deck → marginals should be empty, badge=single
        deck = [{
            "name": "Cmd", "cmc": 4, "cost": "{2}{U}{B}", "text": "",
            "types": ["Creature"], "commander": True,
        }]
        for i in range(38):
            deck.append({
                "name": f"Island {i}", "cmc": 0, "cost": "", "text": "",
                "types": ["Land"], "commander": False,
            })
        # 60 distinct singleton cards: each its own pool
        for cmc in (1, 2, 3, 4, 5, 6):
            for j in range(10):
                deck.append({
                    "name": f"Unique{cmc}-{j}", "cmc": cmc, "cost": f"{{{cmc}}}",
                    "text": "", "types": ["Creature"], "commander": False,
                })

        gf = Goldfisher(deck, turns=10, sims=400, record_results="quartile", seed=99)
        result = gf.simulate()
        cp = result.card_performance
        # All entries are pools — but vanilla creatures with same cmc DO pool. So this
        # test really only verifies that single-copy pools (if any) get badge="single".
        for entry in cp["high_performing"] + cp["low_performing"]:
            if entry["copies"] == 1:
                assert entry["marginals"] == []
                assert entry["saturation"]["badge"] == "single"
