"""Core goldfishing simulation engine.

Refactored to use the effects system, GameState, and Card dataclass.
No CLI code here -- returns structured ``SimulationResult``.
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
try:
    from concurrent.futures import ProcessPoolExecutor
except ImportError:
    ProcessPoolExecutor = None  # type: ignore[misc,assignment]

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        """No-op fallback when tqdm is unavailable (e.g. Pyodide)."""
        return iterable

from auto_goldfish.effects.card_database import DEFAULT_REGISTRY
from auto_goldfish.effects.registry import CardEffects, EffectRegistry
from auto_goldfish.effects.types import CastTriggerEffect, ManaFunctionEffect, OnPlayEffect, PerTurnEffect
from auto_goldfish.engine.mana import land_mana, mana_rocks
from auto_goldfish.engine.mana_efficiency import VALID_MANA_EFFICIENCY_MODES, select_cards_to_play
from auto_goldfish.engine.mulligan import DefaultMulligan, MulliganStrategy
from auto_goldfish.engine.spell_priority import VALID_SPELL_PRIORITIES, get_spell_sort_key
from auto_goldfish.models.card import Card
from auto_goldfish.models.game_state import GameState


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Structured return from ``Goldfisher.simulate()``.

    Replaces the old 15-element tuple.
    """

    land_count: int = 0
    mean_mana: float = 0.0
    mean_mana_value: float = 0.0
    mean_mana_draw: float = 0.0
    mean_mana_ramp: float = 0.0
    mean_mana_total: float = 0.0
    mean_hand_sum: float = 0.0
    consistency: float = 0.0
    mean_bad_turns: float = 0.0
    mean_mid_turns: float = 0.0
    mean_lands: float = 0.0
    mean_mulls: float = 0.0
    mean_draws: float = 0.0
    mean_spells_cast: float = 0.0
    mean_ecms: float = 0.0
    percentile_10: float = 0.0
    percentile_25: float = 0.0
    percentile_50: float = 0.0
    percentile_75: float = 0.0
    percentile_90: float = 0.0
    threshold_percent: float = 0.0
    threshold_mana: float = 0.0
    ceiling_mana: float = 0.0
    quartile_mana: Dict[str, Dict[str, float]] = field(default_factory=dict)
    con_threshold: float = 0.25
    distribution_stats: Dict[str, float] = field(default_factory=dict)
    card_performance: Dict[str, Any] = field(default_factory=dict)
    game_records: Dict[str, Dict[str, list]] = field(default_factory=dict)
    replay_data: Dict[str, Any] = field(default_factory=dict)

    # Per-turn averages (index 0 = turn 1)
    mean_mana_per_turn: List[float] = field(default_factory=list)
    mean_spells_per_turn: List[float] = field(default_factory=list)

    # Extra aggregates for deck scoring
    std_mana: float = 0.0
    mull_rate: float = 0.0
    mean_mana_with_mull: float = 0.0
    mean_mana_no_mull: float = 0.0

    # Structural snapshot of the decklist (independent of simulation outcomes).
    # Used by the Toughness stat. Computed once per Goldfisher build.
    mana_source_count: int = 0
    draw_count: int = 0
    early_count: int = 0
    avg_cmc: float = 0.0

    # 95% CI half-widths (z * std / sqrt(n))
    ci_mana_value: float = 0.0
    ci_mana_draw: float = 0.0
    ci_mana_ramp: float = 0.0
    ci_mana: float = 0.0
    ci_mana_total: float = 0.0

    # 95% confidence intervals (low, high) — legacy tuple form
    ci_mean_mana: Tuple[float, float] = (0.0, 0.0)
    ci_consistency: Tuple[float, float] = (0.0, 0.0)
    ci_mean_bad_turns: Tuple[float, float] = (0.0, 0.0)

    def as_row(self) -> list:
        """Return a flat list for tabulate (without distribution_stats)."""
        mana_margin = (self.ci_mean_mana[1] - self.ci_mean_mana[0]) / 2
        con_margin = (self.ci_consistency[1] - self.ci_consistency[0]) / 2
        return [
            self.land_count,
            f"{self.mean_mana:.2f} +/-{mana_margin:.2f}",
            f"{self.consistency:.4f} +/-{con_margin:.4f}",
            self.mean_bad_turns,
            self.mean_mid_turns,
            self.mean_lands,
            self.mean_mulls,
            self.mean_draws,
            self.mean_spells_cast,
            self.percentile_25,
            self.percentile_50,
            self.percentile_75,
            self.threshold_percent,
            self.threshold_mana,
        ]


# ---------------------------------------------------------------------------
# Module-level helpers (used by effects via import)
# ---------------------------------------------------------------------------


def _draw(state: GameState) -> None:
    """Draw a card from the deck into the hand."""
    if not state.deck:
        if state.should_log:
            state.log.append("Draw failed, deck is empty")
        state.draws += 1
        return
    drawn_i = state.deck.pop()
    drawn = state.decklist[drawn_i]
    drawn.zone = state.hand
    state.hand.append(drawn_i)
    state.draws += 1
    if state.should_log:
        state.log.append(f"Draw {drawn.printable}")


def _random_discard(state: GameState) -> None:
    """Discard a random card from hand."""
    discarded_i = random.choice(state.hand)
    discarded = state.decklist[discarded_i]
    discarded.change_zone(state.yard)
    if state.should_log:
        state.log.append(f"Discarded {discarded.printable}")


def _find_effectless_lands(state: GameState, count: int) -> list[int]:
    """Find up to *count* land cards in the deck that have no registered effects."""
    found: list[int] = []
    for idx in state.deck:
        card = state.decklist[idx]
        if card.land and not _has_effects(card):
            found.append(idx)
            if len(found) >= count:
                break
    return found


def _has_effects(card: Card) -> bool:
    """Return True if the card has any cached effects."""
    eff = getattr(card, '_cached_effects', None)
    if eff is None:
        return False
    return bool(eff.on_play or eff.per_turn or eff.cast_trigger or eff.mana_function)


def _card_equivalence_key(card: Card) -> tuple[int, str]:
    """Return a canonical key identifying simulator-equivalent cards.

    Two cards with the same key behave identically in the simulator: same cmc
    and same registered effects. Card type (creature vs sorcery vs artifact)
    is intentionally ignored so the UI can refer to everything as a "spell".
    """
    eff = getattr(card, "_cached_effects", None)
    effect_desc = eff.describe_effects() if eff else ""
    return (card.cmc, effect_desc)


def _format_pool_label(cmc: int, effect_desc: str) -> str:
    """Render a human-readable label for a pool, e.g. '2-mana spell: Draw 2 cards'."""
    if effect_desc:
        return f"{cmc}-mana spell: {effect_desc}"
    return f"{cmc}-mana spell (no effects)"


# Per-copy marginal analysis tunables.
_MARGINAL_MIN_BUCKET = 30   # both buckets must have >=30 games or we skip the pill
_MARGINAL_MAX_K = 4         # stop evaluating beyond the 4th copy
_MARGINAL_CI_Z = 1.645      # 90% normal CI
_SATURATION_THRESHOLD = 0.1  # marginal magnitude below this rounds to "saturated"

# Top-up tunables: when a pool's k=1 or k=2 marginal has too few games in
# either bucket, we run additional natural goldfishing games (rejection
# sampling) to fill them. _TOPUP_CAP_FACTOR caps total extras as a multiple
# of the original sim count.
_TOPUP_TARGET_KS = (1, 2)
_TOPUP_CAP_FACTOR = 2.0


def _compute_marginal_impacts(
    count_per_game: np.ndarray,
    mana_arr: np.ndarray,
    max_k: int,
) -> list[dict]:
    """Return marginal impact of the k-th copy for k=1..max_k.

    Uses exact-count buckets: marginal_k = E[mana | drew=k] - E[mana | drew=k-1].
    Each entry has effect, ci, n_curr, n_prev, and a 'noise' flag when the 90%
    CI straddles zero.

    Skips emission entirely whenever either bucket has fewer than
    _MARGINAL_MIN_BUCKET games. The badge already conveys the high-level
    verdict, and individual under-sampled pills add clutter without insight.
    """
    out: list[dict] = []
    for k in range(1, max_k + 1):
        mask_curr = count_per_game == k
        mask_prev = count_per_game == k - 1
        n_curr = int(mask_curr.sum())
        n_prev = int(mask_prev.sum())

        if min(n_curr, n_prev) < _MARGINAL_MIN_BUCKET:
            continue

        mean_curr = float(mana_arr[mask_curr].mean())
        mean_prev = float(mana_arr[mask_prev].mean())
        var_curr = float(mana_arr[mask_curr].var(ddof=1))
        var_prev = float(mana_arr[mask_prev].var(ddof=1))
        effect = mean_curr - mean_prev
        se = float(np.sqrt(var_curr / n_curr + var_prev / n_prev))
        ci = _MARGINAL_CI_Z * se
        noise = abs(effect) < ci  # CI straddles zero

        out.append({
            "k": k,
            "effect": round(effect, 3),
            "ci": round(ci, 3),
            "n_curr": n_curr,
            "n_prev": n_prev,
            "noise": noise,
            "reason": "ci_straddles_zero" if noise else None,
        })
    return out


def _classify_saturation(marginals: list[dict]) -> dict:
    """Return saturation badge + suggested copy count from a marginal series.

    Categories:
      * scaling   -- last evaluated marginal still significantly positive
      * saturated -- positive at k=1, then noise/near-zero
      * crowding  -- a marginal is significantly negative
      * unclear   -- not enough signal to classify

    Directional badges (`scaling`, `crowding`) require leading evidence: the
    first emitted marginal must itself be significantly directional. A lone
    late-k significant marginal isn't enough to recommend "add more" or "cut
    copies" — we don't know the value of earlier copies.
    """
    significant = [m for m in marginals if not m["noise"] and m["effect"] is not None]
    if not significant:
        return {"badge": "unclear", "saturates_at": None}

    first_emitted = marginals[0]
    # Directional badges require leading evidence — if the first emitted
    # marginal is itself noise, we have no idea what the early copies do, so
    # neither "add more" nor "cut copies" is groundable. Especially relevant
    # because high-k positives are inflated by selection bias (games that
    # drew enough cards to reach the kth copy also tend to spend more mana
    # for unrelated reasons).
    leading_sig = significant[0] is first_emitted

    if any(m["effect"] < -_SATURATION_THRESHOLD for m in significant):
        if not leading_sig:
            return {"badge": "unclear", "saturates_at": None}
        crowd_k = next(
            m["k"] for m in significant if m["effect"] < -_SATURATION_THRESHOLD
        )
        return {"badge": "crowding", "saturates_at": crowd_k - 1}

    last_sig = significant[-1]
    last_eval = marginals[-1]
    if last_sig is last_eval and last_sig["effect"] > _SATURATION_THRESHOLD:
        if not (leading_sig and first_emitted["effect"] > _SATURATION_THRESHOLD):
            return {"badge": "unclear", "saturates_at": None}
        return {"badge": "scaling", "saturates_at": None}

    return {"badge": "saturated", "saturates_at": last_sig["k"]}


def _card_to_dict(card: Card) -> dict:
    """Serialize a Card back to a dict for worker reconstruction."""
    return {
        "name": card.name, "quantity": card.quantity,
        "oracle_cmc": card.oracle_cmc, "cmc": card.cmc,
        "cost": card.cost, "text": card.text,
        "sub_types": card.sub_types, "super_types": card.super_types,
        "types": card.types, "identity": card.identity,
        "default_category": card.default_category,
        "user_category": card.user_category,
        "tag": card.tag, "commander": card.commander,
    }


_REPLAY_CAP_PER_WORKER = 15


def _worker_run_batch(
    deck_dicts: list[dict],
    turns: int,
    n_games: int,
    base_seed: int | None,
    game_offset: int,
    capture_replays: bool = False,
    extra_config: dict | None = None,
) -> dict:
    """Top-level function for ProcessPoolExecutor workers.

    Creates a fresh Goldfisher and runs ``n_games`` simulations.
    Returns raw per-game stats as lists.

    When *capture_replays* is ``True`` the worker records unclassified
    turn-by-turn snapshots for a sample of games (up to
    ``_REPLAY_CAP_PER_WORKER``).  Classification into quartile buckets
    happens later in ``_run_parallel`` once the full mana distribution
    is available.
    """
    extra_config = extra_config or {}
    gf = Goldfisher(
        deck_dicts, turns=turns, sims=n_games,
        record_results=None, seed=base_seed,
        **extra_config,
    )
    # Adjust seed offset so each batch uses the correct per-game seeds
    mana_spent = []
    mana_value = []
    mana_draw = []
    mana_ramp = []
    ecms = []
    hand_sum = []
    mulls = []
    lands_played = []
    cards_drawn = []
    spells_cast = []
    bad_turns = []
    mid_turns = []
    card_cast_turns: list[list] = [[] for _ in gf.decklist]
    card_mana_spent_list: list[list] = [[] for _ in gf.decklist]
    played_cards_per_game: list[set] = []
    drawn_cards_per_game: list[set] = []

    # Per-turn accumulators (index = turn number)
    mana_per_turn_sum = [0] * turns
    spells_per_turn_sum = [0] * turns

    # Unclassified replay snapshots: list of (total_mana, replay_dict)
    raw_replays: list[tuple[int, dict]] = []
    # Start capturing after the first 10% of games to get some variance
    replay_start = max(int(n_games * 0.1), 1)

    for j in range(n_games):
        global_j = game_offset + j
        if base_seed is not None:
            random.seed(base_seed + global_j)
        state = gf._reset()
        mulligans = gf._mulligan(state)

        total_mana_spent = 0
        game_mana_value = 0
        game_mana_draw = 0
        game_mana_ramp = 0
        game_ecms = 0
        game_hand_sum = 0
        game_lands = 0
        game_bad = 0
        game_mid = 0
        game_spells_cast = 0

        # Decide whether to capture this game's replay
        _capture_this = (
            capture_replays
            and j >= replay_start
            and len(raw_replays) < _REPLAY_CAP_PER_WORKER
        )
        turn_snapshots: list[dict] = []
        starting_hand_names: list[str] = []
        if _capture_this:
            starting_hand_names = [gf._card_by_id(idx).name for idx in state.hand]

        for i in range(turns):
            turn_mana_value = 0
            turn_mana_draw = 0
            turn_mana_ramp = 0
            spells_played = 0

            if _capture_this:
                hand_before = [gf._card_by_id(idx).name for idx in state.hand]

            played = gf._take_turn(state)
            for card in played:
                if card.land:
                    game_lands += 1
                if card.spell:
                    spells_played += 1
                    cost = card.mana_spent_when_played
                    if card.draw:
                        turn_mana_draw += cost
                    elif card.ramp:
                        turn_mana_ramp += cost
                    else:
                        turn_mana_value += cost
            turn_mana = turn_mana_value + turn_mana_draw
            game_mana_value += turn_mana_value
            game_mana_draw += turn_mana_draw
            game_mana_ramp += turn_mana_ramp
            # ECMS: compound mana forward for remaining turns
            if gf.mana_mode == "value":
                turn_ecms_mana = turn_mana_value
            elif gf.mana_mode == "value_draw":
                turn_ecms_mana = turn_mana_value + turn_mana_draw
            else:
                turn_ecms_mana = turn_mana_value + turn_mana_draw + turn_mana_ramp
            game_ecms += turn_ecms_mana * (turns - i)
            game_hand_sum += min(len(state.hand), 7)
            game_spells_cast += spells_played
            mana_per_turn_sum[i] += turn_mana
            spells_per_turn_sum[i] += spells_played
            if spells_played == 0 and state.deck:
                game_bad += 1
            if spells_played < 2 and state.deck and turn_mana < i + 1:
                game_mid += 1
            total_mana_spent += turn_mana

            if _capture_this:
                turn_snapshots.append({
                    "turn": i + 1,
                    "hand_before_draw": hand_before,
                    "played": [
                        {
                            "name": c.name,
                            "cost": c.cost,
                            "mana_spent": c.mana_spent_when_played,
                            "is_land": c.land,
                        }
                        for c in played
                    ],
                    "mana_spent_this_turn": turn_mana,
                    "total_mana_production": gf._get_mana(state),
                    "hand_after": [gf._card_by_id(idx).name for idx in state.hand],
                    "battlefield": [gf._card_by_id(idx).name for idx in state.battlefield],
                    "lands": [gf._card_by_id(idx).name for idx in state.lands],
                    "graveyard": [gf._card_by_id(idx).name for idx in state.yard],
                })

        mana_spent.append(total_mana_spent)
        mana_value.append(game_mana_value)
        mana_draw.append(game_mana_draw)
        mana_ramp.append(game_mana_ramp)
        ecms.append(game_ecms)
        hand_sum.append(game_hand_sum)
        lands_played.append(game_lands)
        mulls.append(mulligans)
        cards_drawn.append(state.draws)
        spells_cast.append(game_spells_cast)
        bad_turns.append(game_bad)
        mid_turns.append(game_mid)

        game_played = set()
        for k, turn in enumerate(state.card_cast_turn):
            if turn is not None and not gf.decklist[k].land:
                card_cast_turns[k].append(turn)
                card_mana_spent_list[k].append(gf.decklist[k].mana_spent_when_played)
                if gf.decklist[k].spell:
                    game_played.add(k)
        played_cards_per_game.append(game_played)

        # Cards drawn = all cards not still in deck or command zone
        deck_set = set(state.deck) | set(state.command_zone)
        game_drawn = {k for k in range(len(gf.decklist))
                      if k not in deck_set and gf.decklist[k].spell and not gf.decklist[k].land}
        drawn_cards_per_game.append(game_drawn)

        if _capture_this:
            raw_replays.append((game_mana_value, game_mana_draw, game_mana_ramp, total_mana_spent, {
                "total_mana": total_mana_spent,
                "mulligans": mulligans,
                "starting_hand": starting_hand_names,
                "turns": turn_snapshots,
            }))

    result: dict = {
        "mana_spent": mana_spent,
        "mana_value": mana_value,
        "mana_draw": mana_draw,
        "mana_ramp": mana_ramp,
        "ecms": ecms,
        "hand_sum": hand_sum,
        "mulls": mulls,
        "lands_played": lands_played,
        "cards_drawn": cards_drawn,
        "spells_cast": spells_cast,
        "bad_turns": bad_turns,
        "mid_turns": mid_turns,
        "card_cast_turns": card_cast_turns,
        "card_mana_spent_list": card_mana_spent_list,
        "played_cards_per_game": played_cards_per_game,
        "drawn_cards_per_game": drawn_cards_per_game,
        "mana_per_turn_sum": mana_per_turn_sum,
        "spells_per_turn_sum": spells_per_turn_sum,
        "n_games": n_games,
    }
    if capture_replays:
        result["raw_replays"] = raw_replays
    return result


# ---------------------------------------------------------------------------
# Goldfisher engine
# ---------------------------------------------------------------------------

class Goldfisher:
    """Core simulation engine.

    Parameters
    ----------
    decklist_dicts : list[dict]
        Raw card dicts (from JSON or Archidekt).
    turns : int
        Number of turns to simulate per game.
    sims : int
        Number of games to simulate.
    verbose : bool
        Print game logs.
    registry : EffectRegistry, optional
        Card effects registry. Uses DEFAULT_REGISTRY if not provided.
    mulligan_strategy : MulliganStrategy, optional
        Mulligan strategy. Uses DefaultMulligan if not provided.
    """

    # Commander indices live above this offset so they never collide with
    # decklist indices (the optimizer mutates the decklist post-construction,
    # so a length-based boundary is unstable).
    _COMMANDER_INDEX_OFFSET = 1_000_000

    def __init__(
        self,
        decklist_dicts: list[dict],
        turns: int,
        sims: int,
        verbose: bool = False,
        registry: EffectRegistry | None = None,
        mulligan_strategy: MulliganStrategy | None = None,
        record_results: str = "quartile",
        deck_name: str | None = None,
        seed: int | None = None,
        workers: int = 1,
        mana_mode: str = "value",
        spell_priority: str = "priority_then_cmc",
        mana_efficiency: str = "greedy",
        ramp_cutoff_turn: int = 0,
        min_cost_floor: int = 1,
        **kwargs,
    ):
        if mana_mode not in ("value", "value_draw", "total"):
            raise ValueError(f"Invalid mana_mode: {mana_mode!r}. Must be 'value', 'value_draw', or 'total'.")
        self.mana_mode = mana_mode
        if spell_priority not in VALID_SPELL_PRIORITIES:
            raise ValueError(
                f"Invalid spell_priority: {spell_priority!r}. "
                f"Must be one of {VALID_SPELL_PRIORITIES}"
            )
        self.spell_priority = spell_priority
        self._spell_sort_key = get_spell_sort_key(spell_priority)
        if mana_efficiency not in VALID_MANA_EFFICIENCY_MODES:
            raise ValueError(
                f"Invalid mana_efficiency: {mana_efficiency!r}. "
                f"Must be one of {VALID_MANA_EFFICIENCY_MODES}"
            )
        self.mana_efficiency = mana_efficiency
        if ramp_cutoff_turn < 0:
            raise ValueError(f"ramp_cutoff_turn must be >= 0, got {ramp_cutoff_turn}")
        self.ramp_cutoff_turn = ramp_cutoff_turn
        if min_cost_floor not in (0, 1):
            raise ValueError(f"min_cost_floor must be 0 or 1, got {min_cost_floor}")
        self.min_cost_floor = min_cost_floor
        self.registry = registry or DEFAULT_REGISTRY
        self.mulligan_strategy = mulligan_strategy or DefaultMulligan()
        self.turns = turns
        self.sims = sims
        self.verbose = verbose
        self.seed = seed
        self.workers = workers
        self._should_log = verbose or record_results is not None

        # Separate commanders from the decklist. Card indices double as the
        # value stored in zone lists (state.battlefield, state.command_zone,
        # etc.), so commander indices MUST be disjoint from decklist indices —
        # otherwise a commander cast onto the battlefield aliases a random
        # non-commander card during snapshot lookups.
        #
        # Decklists are mutated post-construction by the optimizer (cards
        # cut, indices reassigned, synthetic cards appended), so we can't
        # tie commander indices to the current decklist length. Use a fixed
        # high offset that real decklists will never reach.
        non_commander_dicts: list[dict] = []
        commander_dicts: list[dict] = []
        for card_dict in decklist_dicts:
            if card_dict.get("commander", False):
                commander_dicts.append(card_dict)
            else:
                non_commander_dicts.append(card_dict)

        self.decklist: list[Card] = [
            self._make_card(d, i) for i, d in enumerate(non_commander_dicts)
        ]
        self.commanders: list[Card] = [
            self._make_card(d, self._COMMANDER_INDEX_OFFSET + i)
            for i, d in enumerate(commander_dicts)
        ]
        self.deckdict: dict[str, Card] = {c.name: c for c in self.decklist}

        self.deck_name = deck_name or "_-_".join(
            c.name for c in self.commanders
        ).replace(" ", "_").replace(",", "")

        self.land_count = sum(1 for c in self.decklist if c.land)
        self.original_card_count = len(self.decklist)

        # Structural snapshot of the decklist (used by the Toughness score).
        # Counts mana sources (lands + ramp), card-draw cards, low-CMC plays,
        # and average CMC of non-land cards.
        non_land_cmc = [c.cmc for c in self.decklist if not c.land]
        self.mana_source_count = sum(
            1 for c in self.decklist if c.land or c.ramp
        )
        self.draw_count = sum(1 for c in self.decklist if c.draw)
        self.early_count = sum(
            1 for c in self.decklist if not c.land and c.cmc <= 3
        )
        self.avg_cmc = (
            sum(non_land_cmc) / len(non_land_cmc) if non_land_cmc else 0.0
        )

        # Save original state for optimizer restore
        self._original_decklist_dicts = list(non_commander_dicts)
        self._original_registry = self.registry
        self._original_land_count = self.land_count

        # Recording config
        self.record_quartile = False
        self.record_decile = False
        self.record_centile = False
        self.record_half = True
        if record_results == "quartile":
            self.record_quartile = self.record_decile = self.record_centile = True
        elif record_results == "decile":
            self.record_decile = self.record_centile = True
        elif record_results == "centile":
            self.record_centile = True

    def _card_by_id(self, idx: int) -> Card:
        """Resolve a zone card-id to its Card.

        Decklist indices start at 0; commander indices start at
        _COMMANDER_INDEX_OFFSET. Any zone (battlefield, command zone, hand,
        etc.) can in principle store either kind of id, so all zone-derived
        lookups should go through this helper.
        """
        if idx >= self._COMMANDER_INDEX_OFFSET:
            return self.commanders[idx - self._COMMANDER_INDEX_OFFSET]
        return self.decklist[idx]

    def _make_card(self, card_dict: dict, index: int) -> Card:
        """Create a Card from a raw dict, applying registry effects."""
        # Clean up dict for Card dataclass
        kw = {
            "name": card_dict.get("name", ""),
            "quantity": card_dict.get("quantity", 1),
            "oracle_cmc": card_dict.get("oracle_cmc", 0),
            "cmc": card_dict.get("cmc", 0),
            "cost": card_dict.get("cost", ""),
            "text": card_dict.get("text", ""),
            "sub_types": card_dict.get("sub_types", []),
            "super_types": card_dict.get("super_types", []),
            "types": list(card_dict.get("types", [])),
            "identity": card_dict.get("identity", []),
            "default_category": card_dict.get("default_category"),
            "user_category": card_dict.get("user_category"),
            "tag": card_dict.get("tag"),
            "commander": card_dict.get("commander", False),
            "index": index,
        }

        # Apply registry overrides
        effects = self.registry.get(kw["name"])
        if effects:
            if effects.extra_types:
                for t in effects.extra_types:
                    if t not in kw["types"]:
                        kw["types"].append(t)
            if effects.override_cmc is not None:
                kw["cmc"] = effects.override_cmc

        card = Card(**kw)

        # Cache registry effects on the card to avoid repeated lookups
        card._cached_effects = effects

        # Apply card-level flags from registry
        if effects:
            card.ramp = effects.ramp
            card.draw = effects.draw
            card.priority = effects.priority
            card.tapped = effects.tapped

        # Default: if it's a land, mark as not tapped unless registry says otherwise
        if card.land and not card.spell and not effects:
            card.tapped = False

        return card

    def _reset(self) -> GameState:
        """Create a fresh GameState for a new game."""
        state = GameState()
        state.should_log = self._should_log
        state.card_cast_turn = [None] * len(self.decklist)
        state.decklist = self.decklist
        state.deckdict = self.deckdict
        state.min_cost_floor = self.min_cost_floor

        # Place commanders in command zone
        for card in self.commanders:
            card.zone = state.command_zone
            state.command_zone.append(card.index)

        # Place all cards in deck
        for card in self.decklist:
            card.zone = state.deck
            state.deck.append(card.index)

        random.shuffle(state.deck)

        # Default mana functions
        state.mana_functions = [land_mana, mana_rocks]

        return state

    def _mulligan(self, state: GameState) -> int:
        """Execute mulligan logic. Returns the number of mulligans taken."""
        mulligans = -1
        while True:
            # Reset state for this mulligan attempt
            state.log = []
            state.turn = 0
            state.draws = 0
            state.command_zone = []
            state.deck = []
            state.yard = []
            state.hand = []
            state.battlefield = []
            state.exile = []
            state.lands = []
            state.mana_production = 0
            state.treasure = 0
            state.per_turn_effects = []
            state.cast_triggers = []
            state.mana_functions = [land_mana, mana_rocks]
            state.lands_per_turn = 1
            state.nonpermanent_cost_reduction = 0
            state.permanent_cost_reduction = 0
            state.spell_cost_reduction = 0
            state.creature_cost_reduction = 0
            state.enchantment_cost_reduction = 0
            state.creatures_played = 0
            state.enchantments_played = 0
            state.artifacts_played = 0
            state.card_cast_turn = [None] * len(self.decklist)

            for card in self.commanders:
                card.zone = state.command_zone
                state.command_zone.append(card.index)
            for card in self.decklist:
                card.zone = state.deck
                state.deck.append(card.index)
            random.shuffle(state.deck)

            if state.should_log:
                if mulligans == -1:
                    state.log.append("### Opening hand:")
                else:
                    state.log.append(f"### Mulligan #{mulligans + 1}")

            cards = 7
            if mulligans > 0:
                cards -= mulligans
            mulligans += 1

            lands_in_hand = 0
            for _ in range(cards):
                _draw(state)
            for i in state.hand:
                if self.decklist[i].land:
                    lands_in_hand += 1

            if self.mulligan_strategy.should_keep(state, len(state.hand), lands_in_hand):
                break

        state.starting_hand = [self.decklist[i] for i in state.hand]
        state.starting_hand_land_count = lands_in_hand
        if state.should_log:
            state.log.append(f"### Kept {lands_in_hand}/{len(state.hand)} lands/cards")
        return mulligans

    def _get_mana(self, state: GameState) -> int:
        """Calculate total available mana."""
        mana = 0
        for func in state.mana_functions:
            if callable(func):
                mana += func(state)
            elif isinstance(func, ManaFunctionEffect):
                mana += func.mana_function(state)
        return mana

    def _get_playables(self, state: GameState, available_mana: int) -> list[Card]:
        """Find all castable spells in hand and command zone."""
        playables = []
        for i in state.hand:
            card = self.decklist[i]
            if card.get_current_cost(state) <= available_mana and card.spell:
                playables.append(card)

        for i in state.command_zone:
            card = self._card_by_id(i)
            if card.get_current_cost(state) <= available_mana and card.spell:
                playables.append(card)

        if self._spell_sort_key is not None:
            playables = sorted(playables, key=self._spell_sort_key)
        else:
            playables = sorted(playables)

        # Ramp cutoff: after the cutoff turn, move ramp cards to lowest priority
        if self.ramp_cutoff_turn and state.turn >= self.ramp_cutoff_turn:
            non_ramp = [c for c in playables if not c.ramp]
            ramp = [c for c in playables if c.ramp]
            playables = ramp + non_ramp  # ramp at front = played last (reversed iteration)

        if state.should_log:
            playables_str = []
            for card in playables:
                if card.commander:
                    playables_str.append(f"{card.cmc}(c)")
                else:
                    playables_str.append(f"{card.cmc}")
            state.log.append(f"--Playable Spells: {playables_str}")

        return playables

    def _play_land(self, state: GameState) -> list[Card]:
        """Play lands from hand."""
        played = []
        land_cards = [self.decklist[i] for i in state.hand if self.decklist[i].land]
        if self._spell_sort_key is not None:
            playable_lands = sorted(land_cards, key=self._spell_sort_key)
        else:
            playable_lands = sorted(land_cards)
        for land in reversed(playable_lands):
            if state.played_land_this_turn < state.lands_per_turn:
                land.change_zone(state.lands)
                if state.should_log:
                    state.log.append(f"Played as land {land.printable}")
                land.mana_spent_when_played = 0
                played.append(land)
                state.played_land_this_turn += 1
                if not land.tapped:
                    state.untapped_land_this_turn += 1

                # Apply on_play effects for lands
                effects = land._cached_effects
                if effects:
                    for eff in effects.on_play:
                        if isinstance(eff, OnPlayEffect):
                            eff.on_play(land, state)
                    for eff in effects.mana_function:
                        state.mana_functions.append(eff)
            else:
                break
        return played

    def _play_card(self, card: Card, state: GameState) -> None:
        """Play a single spell card, applying all effects."""
        if card.spell and card.nonpermanent:
            card.change_zone(state.yard)
        elif card.spell and card.permanent:
            card.change_zone(state.battlefield)

        # Apply cast triggers from cards already in play
        for trigger_data in state.cast_triggers:
            trigger_card, trigger_eff = trigger_data
            trigger_eff.cast_trigger(trigger_card, card, state)

        # Register this card's effects
        effects = card._cached_effects
        if effects:
            for eff in effects.cast_trigger:
                state.cast_triggers.append((card, eff))
            for eff in effects.per_turn:
                state.per_turn_effects.append((card, eff))
            for eff in effects.mana_function:
                state.mana_functions.append(eff)

        # Track when this card was actually cast
        if not card.commander and state.card_cast_turn[card.index] is None:
            state.card_cast_turn[card.index] = state.turn + 1

        # Execute on_play effects
        if state.should_log:
            state.log.append(f"Played {card.printable}")
        card.mana_spent_when_played = card.get_current_cost(state)
        if card.creature:
            state.creatures_played += 1
        if card.enchantment:
            state.enchantments_played += 1
        if card.artifact:
            state.artifacts_played += 1

        if effects:
            for eff in effects.on_play:
                if isinstance(eff, OnPlayEffect):
                    eff.on_play(card, state)

    def _play_spells(self, state: GameState) -> list[Card]:
        """Play spells from hand."""
        mana_available = self._get_mana(state) + state.treasure
        played_effects: list[Card] = []

        played_effects.extend(self._play_land(state))
        mana_available += state.untapped_land_this_turn
        state.untapped_land_this_turn = 0

        playables = self._get_playables(state, mana_available)
        while playables:
            selected = select_cards_to_play(
                self.mana_efficiency, playables, mana_available, state,
            )
            if not selected:
                break
            for card in selected:
                cost = card.get_current_cost(state)
                mana_available -= cost
                self._play_card(card, state)
                played_effects.append(card)

            played_effects.extend(self._play_land(state))
            mana_available += state.untapped_land_this_turn
            state.untapped_land_this_turn = 0
            playables = self._get_playables(state, mana_available)

        if mana_available < state.treasure:
            if state.should_log:
                state.log.append(f"Spent treasures: [{state.treasure}] -> [{mana_available}]")
            state.treasure = mana_available

        return played_effects

    def _take_turn(self, state: GameState) -> list[Card]:
        """Execute one turn."""
        if state.should_log:
            state.log.append(
                f"### Turn {state.turn + 1} "
                f"(Lands: {len(state.lands)}, Mana: {self._get_mana(state)}[{state.treasure}], "
                f"Hand: {len(state.hand)})"
            )
        state.played_land_this_turn = 0
        state.untapped_land_this_turn = 0
        state.tapped_creatures_this_turn = 0
        _draw(state)

        # Per-turn effects
        for card, eff in state.per_turn_effects:
            eff.per_turn(card, state)

        return self._play_spells(state)

    def set_lands(self, land_count: int, cuts: list[str] | None = None) -> None:
        """Adjust the deck's land count."""
        from auto_goldfish.decklist.loader import get_basic_island

        cuts = cuts or []
        cutted = []
        spells_list = []
        lands_list = []

        for card in self.decklist:
            if card.land:
                lands_list.append(card)
            else:
                spells_list.append(card)

        land_diff = land_count - len(lands_list)
        while land_diff > 0:
            lands_list.append(self._make_card(get_basic_island(), len(spells_list) + len(lands_list)))
            land_diff -= 1
            if cuts and len(spells_list) + len(lands_list) > self.original_card_count:
                for card in spells_list:
                    if card.name in cuts:
                        spells_list.remove(card)
                        cutted.append(card.name)
                        break

        while land_diff < 0:
            for card in lands_list:
                if not card.spell:
                    lands_list.remove(card)
                    cutted.append(card.name)
                    break
            land_diff += 1

        updated = spells_list + lands_list
        self.decklist = [
            self._make_card(
                {
                    "name": c.name, "quantity": c.quantity, "oracle_cmc": c.oracle_cmc,
                    "cmc": c.cmc, "cost": c.cost, "text": c.text,
                    "sub_types": c.sub_types, "super_types": c.super_types,
                    "types": c.types, "identity": c.identity,
                    "default_category": c.default_category, "user_category": c.user_category,
                    "tag": c.tag, "commander": c.commander,
                },
                i,
            )
            for i, c in enumerate(updated)
        ]
        self.deckdict = {c.name: c for c in self.decklist}

        if cutted:
            print(f"Cutted: {cutted}")
        print(
            f"\nSet land count to {land_count} prev {self.land_count} "
            f"({len(lands_list)} lands, {len(spells_list)} spells, total {len(self.decklist)})"
        )
        self.land_count = sum(1 for c in self.decklist if c.land)

    def restore_original_decklist(self) -> None:
        """Reset decklist to its original state (before any set_lands or optimizer modifications)."""
        self.registry = self._original_registry
        self.decklist = [
            self._make_card(d, i) for i, d in enumerate(self._original_decklist_dicts)
        ]
        self.deckdict = {c.name: c for c in self.decklist}
        self.land_count = self._original_land_count

    def _compute_distribution_stats(self, mana_spent_list: list) -> Dict[str, float]:
        """Compute distribution bucket fractions from raw mana data.

        Uses the first 10% (min 100) games to calibrate percentile thresholds,
        then counts what fraction of remaining games fall into each bucket.
        """
        sample_games = int(max(len(mana_spent_list) / 10, 100))
        if sample_games >= len(mana_spent_list):
            # Not enough data for calibration
            return {k: 0.0 for k in [
                "top_centile", "top_decile", "top_quartile", "top_half",
                "low_half", "low_quartile", "low_decile", "low_centile",
            ]}

        calibration = mana_spent_list[:sample_games]
        evaluation = mana_spent_list[sample_games:]

        top_centile_threshold = float(np.percentile(calibration, 99))
        low_centile_threshold = float(np.percentile(calibration, 1))
        top_decile_threshold = float(np.percentile(calibration, 90))
        low_decile_threshold = float(np.percentile(calibration, 10))
        top_quartile_threshold = float(np.percentile(calibration, 75))
        low_quartile_threshold = float(np.percentile(calibration, 25))
        median_threshold = float(np.percentile(calibration, 50))

        counts = {k: 0 for k in [
            "top_centile", "top_decile", "top_quartile", "top_half",
            "low_half", "low_quartile", "low_decile", "low_centile",
        ]}

        for mana in evaluation:
            if self.record_centile and mana >= top_centile_threshold:
                counts["top_centile"] += 1
            if self.record_decile and mana >= top_decile_threshold:
                counts["top_decile"] += 1
            if self.record_quartile and mana >= top_quartile_threshold:
                counts["top_quartile"] += 1
            if self.record_half and mana >= median_threshold:
                counts["top_half"] += 1

            if self.record_centile and mana <= low_centile_threshold:
                counts["low_centile"] += 1
            if self.record_decile and mana <= low_decile_threshold:
                counts["low_decile"] += 1
            if self.record_quartile and mana <= low_quartile_threshold:
                counts["low_quartile"] += 1
            if self.record_half and mana < median_threshold:
                counts["low_half"] += 1

        total = len(evaluation)
        return {k: v / total for k, v in counts.items()}

    def _compute_card_performance(
        self,
        mana_spent_list: list,
        played_cards_per_game: list[set],
        drawn_cards_per_game: list[set] | None = None,
        card_mana_spent_list: list[list] | None = None,
    ) -> Dict[str, Any]:
        """Measure each pool's causal impact on game quality.

        Cards that are simulator-equivalent (same cmc, same effect signature,
        same simulator-relevant types) are pooled together — e.g. all 5-mana
        vanilla spells, or all 2-mana "draw 2" sorceries. For each pool, we
        compare the average mana spent in games where any pool member was
        drawn vs games where none were drawn. Pooling tightens estimates for
        otherwise-rare cards and exposes archetype-level signals.
        """
        if not drawn_cards_per_game or len(mana_spent_list) < 100:
            return {}

        n_games = len(mana_spent_list)
        mana_arr = np.array(mana_spent_list, dtype=float)

        # Group spells by simulator-equivalence signature. Pure lands are
        # excluded; MDFCs (cards with both land and spell faces) are kept
        # because their spell side affects mana spent.
        groups: Dict[tuple, list[int]] = {}
        for k, card in enumerate(self.decklist):
            if not card.spell:
                continue
            sig = _card_equivalence_key(card)
            groups.setdefault(sig, []).append(k)

        # Compute initial per-pool draw counts.
        count_arrays: dict[tuple, np.ndarray] = {}
        for sig, indices in groups.items():
            index_set = set(indices)
            count_arrays[sig] = np.fromiter(
                (len(index_set & drawn_cards_per_game[i]) for i in range(n_games)),
                dtype=int,
                count=n_games,
            )

        # Top-up: run extra natural goldfishing games when k=1/k=2 buckets
        # are underfilled. Each extra game contributes draw counts to *every*
        # pool, so one shared batch can fill many pools at once. Skipped when
        # there are no candidate buckets within the cap.
        n_supplemental = self._plan_topup_budget(groups, count_arrays, n_games)
        if n_supplemental > 0:
            extra_data = _worker_run_batch(
                self._get_deck_dicts(),
                self.turns,
                n_supplemental,
                base_seed=self.seed,
                game_offset=n_games,
                capture_replays=False,
                extra_config=self._get_worker_config(),
            )
            extra_drawn = extra_data["drawn_cards_per_game"]
            extra_mana = extra_data["mana_spent"]
            mana_arr = np.concatenate(
                [mana_arr, np.array(extra_mana, dtype=float)]
            )
            for sig, indices in groups.items():
                index_set = set(indices)
                extra_counts = np.fromiter(
                    (len(index_set & extra_drawn[i]) for i in range(n_supplemental)),
                    dtype=int,
                    count=n_supplemental,
                )
                count_arrays[sig] = np.concatenate(
                    [count_arrays[sig], extra_counts]
                )

        n_games_total = n_games + n_supplemental
        scores = []
        for sig, indices in groups.items():
            cmc, effect_desc = sig
            count_per_game = count_arrays[sig]
            drawn_mask = count_per_game >= 1
            n_drawn = int(drawn_mask.sum())
            n_not_drawn = n_games_total - n_drawn

            if n_drawn < 20:
                continue  # No "drew at least one" group to measure
            always_drawn = n_not_drawn < 20
            copies = len(indices)
            if always_drawn and copies < 2:
                continue  # Singleton always drawn → no comparison group available

            mean_with = float(np.mean(mana_arr[drawn_mask]))

            marginals: list[dict] = []
            saturation = {"badge": "single", "saturates_at": None}
            if copies >= 2:
                # Always-drawn pools have empty bucket k=0, so the early k bins
                # are small-sample noise. Extend max_k to where the data lives:
                # one above the most-populated count.
                max_k = min(copies, _MARGINAL_MAX_K)
                if always_drawn:
                    modal_k = int(np.bincount(count_per_game).argmax())
                    max_k = min(copies, max(max_k, modal_k + 1))
                marginals = _compute_marginal_impacts(
                    count_per_game,
                    mana_arr,
                    max_k=max_k,
                )
                saturation = _classify_saturation(marginals)

            if always_drawn:
                # Sum significant per-copy marginals stand in for "with vs without"
                sig_effects = [
                    m["effect"] for m in marginals
                    if not m["noise"] and m["effect"] is not None
                ]
                if not sig_effects:
                    continue  # No usable signal for an always-drawn pool
                impact = sum(sig_effects)
                mean_without = 0.0  # sentinel — not displayed when always_drawn
            else:
                mean_without = float(np.mean(mana_arr[~drawn_mask]))
                impact = mean_with - mean_without

            # Pooled average mana actually spent across all member casts
            pooled_spent: list[float] = []
            if card_mana_spent_list:
                for k in indices:
                    if card_mana_spent_list[k]:
                        pooled_spent.extend(card_mana_spent_list[k])
            avg_cost = round(float(np.mean(pooled_spent)), 1) if pooled_spent else cmc

            # Deterministic example (alphabetical first)
            sorted_members = sorted(indices, key=lambda k: self.decklist[k].name)
            example_card = self.decklist[sorted_members[0]]

            scores.append({
                "name": example_card.name,
                "label": _format_pool_label(cmc, effect_desc),
                "copies": copies,
                "cost": avg_cost,
                "cmc": cmc,
                "effects": effect_desc,
                "mean_with": round(mean_with, 2),
                "mean_without": round(mean_without, 2),
                "score": round(impact, 4),
                "drawn_pct": round(n_drawn / n_games_total, 4),
                "always_drawn": always_drawn,
                "marginals": marginals,
                "saturation": saturation,
            })

        # Sort and pick top/bottom 10
        high_performing = sorted(scores, key=lambda x: x["score"], reverse=True)[:10]
        low_performing = sorted(scores, key=lambda x: x["score"])[:10]

        return {
            "high_performing": high_performing,
            "low_performing": low_performing,
            "total_games": n_games,
            "supplemental_games": n_supplemental,
        }

    def _plan_topup_budget(
        self,
        groups: Dict[tuple, list[int]],
        count_arrays: dict,
        n_games: int,
    ) -> int:
        """Decide how many extra games to run for marginal top-up.

        Scans candidate buckets across all pools (k in _TOPUP_TARGET_KS where
        the smaller adjacent bucket is in [1, _MARGINAL_MIN_BUCKET)). For each
        candidate, extrapolates how many more natural goldfishing games would
        push the smaller bucket past the floor. Returns the largest such
        request that fits within _TOPUP_CAP_FACTOR x n_games (since one shared
        batch fills every pool simultaneously, running the largest affordable
        request also resolves all cheaper candidates as a side effect).
        Returns 0 when nothing is fillable.
        """
        cap = int(_TOPUP_CAP_FACTOR * n_games)
        best = 0
        for sig, indices in groups.items():
            copies = len(indices)
            if copies < 2:
                continue
            cpg = count_arrays[sig]
            for k in _TOPUP_TARGET_KS:
                if k > copies:
                    continue
                n_curr = int((cpg == k).sum())
                n_prev = int((cpg == k - 1).sum())
                n_smaller = min(n_curr, n_prev)
                if n_smaller >= _MARGINAL_MIN_BUCKET:
                    continue  # already resolved
                if n_smaller < 1:
                    continue  # bucket physically rare; rejection won't help
                p_smaller = n_smaller / n_games
                needed = (_MARGINAL_MIN_BUCKET - n_smaller) / p_smaller
                extras = math.ceil(needed)
                if extras <= cap and extras > best:
                    best = extras
        return best

    def _get_deck_dicts(self) -> list[dict]:
        """Serialize current decklist + commanders to dicts for workers."""
        dicts = [_card_to_dict(c) for c in self.commanders]
        dicts.extend(_card_to_dict(c) for c in self.decklist)
        return dicts

    def _get_worker_config(self) -> dict:
        """Collect algorithm settings that workers need to replicate."""
        return {
            "spell_priority": self.spell_priority,
            "mana_efficiency": self.mana_efficiency,
            "ramp_cutoff_turn": self.ramp_cutoff_turn,
            "min_cost_floor": self.min_cost_floor,
            "mana_mode": self.mana_mode,
        }

    def _run_parallel(self) -> dict:
        """Run simulations across multiple worker processes."""
        deck_dicts = self._get_deck_dicts()
        extra_config = self._get_worker_config()
        num_workers = min(self.workers, self.sims)
        batch_size = self.sims // num_workers
        remainder = self.sims % num_workers

        futures = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            offset = 0
            for w in range(num_workers):
                n = batch_size + (1 if w < remainder else 0)
                futures.append(
                    executor.submit(
                        _worker_run_batch,
                        deck_dicts, self.turns, n, self.seed, offset,
                        capture_replays=True,
                        extra_config=extra_config,
                    )
                )
                offset += n

        # Merge results from all workers
        merged = {
            "mana_spent": [], "mana_value": [], "mana_draw": [], "mana_ramp": [],
            "ecms": [],
            "hand_sum": [], "mulls": [], "lands_played": [],
            "cards_drawn": [], "spells_cast": [], "bad_turns": [], "mid_turns": [],
            "played_cards_per_game": [],
            "drawn_cards_per_game": [],
        }
        card_cast_turns: list[list] = [[] for _ in self.decklist]
        card_mana_spent: list[list] = [[] for _ in self.decklist]
        all_raw_replays: list[tuple[int, dict]] = []
        mana_per_turn_sum = [0] * self.turns
        spells_per_turn_sum = [0] * self.turns

        for future in futures:
            batch = future.result()
            for key in ["mana_spent", "mana_value", "mana_draw", "mana_ramp",
                        "ecms", "hand_sum", "mulls", "lands_played",
                        "cards_drawn", "spells_cast", "bad_turns", "mid_turns"]:
                merged[key].extend(batch[key])
            for k, turns_list in enumerate(batch["card_cast_turns"]):
                card_cast_turns[k].extend(turns_list)
            for k, spent_list in enumerate(batch["card_mana_spent_list"]):
                card_mana_spent[k].extend(spent_list)
            merged["played_cards_per_game"].extend(batch["played_cards_per_game"])
            merged["drawn_cards_per_game"].extend(batch["drawn_cards_per_game"])
            all_raw_replays.extend(batch.get("raw_replays", []))
            for t in range(self.turns):
                mana_per_turn_sum[t] += batch["mana_per_turn_sum"][t]
                spells_per_turn_sum[t] += batch["spells_per_turn_sum"][t]

        merged["card_cast_turns"] = card_cast_turns
        merged["card_mana_spent_list"] = card_mana_spent
        merged["mana_per_turn_sum"] = mana_per_turn_sum
        merged["spells_per_turn_sum"] = spells_per_turn_sum

        # Classify pooled replays using the primary mana distribution
        replay_buckets: dict[str, list] = {"top": [], "mid": [], "low": []}
        if all_raw_replays and merged["mana_value"]:
            primary_merged = self._get_primary_mana(
                merged["mana_value"], merged["mana_draw"], merged["mana_ramp"], merged["mana_spent"],
            )
            top_threshold = float(np.percentile(primary_merged, 75))
            low_threshold = float(np.percentile(primary_merged, 25))
            for mana_val, mana_drw, mana_rmp, mana_spt, replay in all_raw_replays:
                if self.mana_mode == "value":
                    primary_val = mana_val
                elif self.mana_mode == "value_draw":
                    primary_val = mana_spt
                else:
                    primary_val = mana_val + mana_drw + mana_rmp
                if primary_val >= top_threshold:
                    if len(replay_buckets["top"]) < 10:
                        replay_buckets["top"].append(replay)
                elif primary_val <= low_threshold:
                    if len(replay_buckets["low"]) < 10:
                        replay_buckets["low"].append(replay)
                else:
                    if len(replay_buckets["mid"]) < 10:
                        replay_buckets["mid"].append(replay)
        merged["replay_data"] = replay_buckets
        return merged

    def _get_primary_mana(self, mana_value, mana_draw, mana_ramp, mana_spent=None):
        """Return the mana list selected by ``self.mana_mode``."""
        if self.mana_mode == "value":
            return mana_value
        elif self.mana_mode == "value_draw":
            return mana_spent if mana_spent is not None else [v + d for v, d in zip(mana_value, mana_draw)]
        else:  # "total"
            return [v + d + r for v, d, r in zip(mana_value, mana_draw, mana_ramp)]

    def _simulate_from_raw(self, raw: dict) -> SimulationResult:
        """Compute summary stats from raw per-game data (used by parallel path)."""

        mana_spent_list = raw["mana_spent"]
        mana_value_list = raw["mana_value"]
        mana_draw_list = raw["mana_draw"]
        mana_ramp_list = raw["mana_ramp"]
        hand_sum_list = raw["hand_sum"]
        lands_played_list = raw["lands_played"]
        mulls_list = raw["mulls"]
        cards_drawn_list = raw["cards_drawn"]
        spells_cast_list = raw["spells_cast"]
        bad_turns_list = raw["bad_turns"]
        mid_turns_list = raw["mid_turns"]
        ecms_list = raw["ecms"]

        primary_list = self._get_primary_mana(mana_value_list, mana_draw_list, mana_ramp_list, mana_spent_list)

        mean_mana = float(np.mean(mana_spent_list))
        mean_mana_value = float(np.mean(mana_value_list))
        mean_mana_draw = float(np.mean(mana_draw_list))
        mean_mana_ramp = float(np.mean(mana_ramp_list))
        mean_ecms = float(np.mean(ecms_list))
        mean_mana_total = float(np.mean([v + d + r for v, d, r in zip(mana_value_list, mana_draw_list, mana_ramp_list)]))
        mean_hand_sum = float(np.mean(hand_sum_list))
        mean_lands = float(np.mean(lands_played_list))
        mean_mulls = float(np.mean(mulls_list))
        mean_draws = float(np.mean(cards_drawn_list))
        mean_spells_cast = float(np.mean(spells_cast_list))
        mean_bad_turns = float(np.mean(bad_turns_list))
        mean_mid_turns = float(np.mean(mid_turns_list))
        percentile_10 = float(np.percentile(primary_list, 10))
        percentile_25 = float(np.percentile(primary_list, 25))
        percentile_50 = float(np.percentile(primary_list, 50))
        percentile_75 = float(np.percentile(primary_list, 75))
        percentile_90 = float(np.percentile(primary_list, 90))

        # Left-tail ratio consistency: mean(bottom 25%) / mean(all)
        con_threshold = 0.25
        sorted_primary = sorted(primary_list)
        n = len(primary_list)
        cutoff = max(1, int(n * con_threshold))
        tail_mean = float(np.mean(sorted_primary[:cutoff]))
        overall_primary_mean = float(np.mean(primary_list))
        if overall_primary_mean == 0:
            consistency = 1.0
        else:
            consistency = tail_mean / overall_primary_mean
        threshold_percent = con_threshold
        threshold_mana = tail_mean

        # Top-25% mean (ceiling) and per-quartile mana breakdowns
        top_cutoff = max(1, int(n * (1 - con_threshold)))
        ceiling_mana = float(np.mean(sorted_primary[top_cutoff:]))

        primary_arr = np.array(primary_list)
        val_arr = np.array(mana_value_list)
        draw_arr = np.array(mana_draw_list)
        ramp_arr = np.array(mana_ramp_list)
        vd_arr = val_arr + draw_arr
        all_arr = val_arr + draw_arr + ramp_arr

        sort_idx = np.argsort(primary_arr)
        bot_idx = sort_idx[:cutoff]
        top_idx = sort_idx[top_cutoff:]

        quartile_mana = {
            "bottom_25": {
                "value": float(np.mean(val_arr[bot_idx])),
                "draw": float(np.mean(draw_arr[bot_idx])),
                "ramp": float(np.mean(ramp_arr[bot_idx])),
                "vd": float(np.mean(vd_arr[bot_idx])),
                "all": float(np.mean(all_arr[bot_idx])),
            },
            "mean": {
                "value": float(np.mean(val_arr)),
                "draw": float(np.mean(draw_arr)),
                "ramp": float(np.mean(ramp_arr)),
                "vd": float(np.mean(vd_arr)),
                "all": float(np.mean(all_arr)),
            },
            "top_25": {
                "value": float(np.mean(val_arr[top_idx])),
                "draw": float(np.mean(draw_arr[top_idx])),
                "ramp": float(np.mean(ramp_arr[top_idx])),
                "vd": float(np.mean(vd_arr[top_idx])),
                "all": float(np.mean(all_arr[top_idx])),
            },
        }

        z = 1.96

        sqrt_n = np.sqrt(n)
        mana_total_list = [v + d + r for v, d, r in zip(mana_value_list, mana_draw_list, mana_ramp_list)]
        ci_mana_value = z * float(np.std(mana_value_list, ddof=1)) / sqrt_n
        ci_mana_draw = z * float(np.std(mana_draw_list, ddof=1)) / sqrt_n
        ci_mana_ramp = z * float(np.std(mana_ramp_list, ddof=1)) / sqrt_n
        ci_mana = z * float(np.std(mana_spent_list, ddof=1)) / sqrt_n
        ci_mana_total = z * float(np.std(mana_total_list, ddof=1)) / sqrt_n

        ci_mean_mana = (mean_mana - ci_mana, mean_mana + ci_mana)

        bad_se = float(np.std(bad_turns_list, ddof=1) / sqrt_n)
        ci_mean_bad_turns = (mean_bad_turns - z * bad_se, mean_bad_turns + z * bad_se)

        # Bootstrap CI for consistency (left-tail ratio)
        n_boot = min(1000, n)
        boot_consistencies = []
        mana_arr = np.array(primary_list)
        boot_rng = np.random.RandomState(self.seed if self.seed is not None else 42)
        for _ in range(n_boot):
            boot_sample = boot_rng.choice(mana_arr, size=n, replace=True)
            boot_sorted = np.sort(boot_sample)
            boot_tail = float(np.mean(boot_sorted[:cutoff]))
            boot_overall = float(np.mean(boot_sample))
            if boot_overall == 0:
                boot_consistencies.append(1.0)
            else:
                boot_consistencies.append(boot_tail / boot_overall)
        ci_consistency = (
            float(np.percentile(boot_consistencies, 2.5)),
            float(np.percentile(boot_consistencies, 97.5)),
        )

        # Compute distribution stats (same calibration approach as sequential path)
        distribution_stats = self._compute_distribution_stats(primary_list)

        # Compute card performance
        played_cards_per_game = raw.get("played_cards_per_game", [])
        drawn_cards_per_game = raw.get("drawn_cards_per_game", [])
        card_mana_spent_raw = raw.get("card_mana_spent_list")
        card_performance = self._compute_card_performance(
            primary_list, played_cards_per_game, drawn_cards_per_game,
            card_mana_spent_list=card_mana_spent_raw,
        )

        replay_data = raw.get("replay_data", {})

        # Per-turn averages
        mana_per_turn_sum = raw.get("mana_per_turn_sum", [0] * self.turns)
        spells_per_turn_sum = raw.get("spells_per_turn_sum", [0] * self.turns)
        mean_mana_per_turn = [s / n for s in mana_per_turn_sum] if n > 0 else []
        mean_spells_per_turn = [s / n for s in spells_per_turn_sum] if n > 0 else []

        # Scoring aggregates
        std_mana = float(np.std(mana_spent_list, ddof=1)) if n > 1 else 0.0
        mulls_arr = np.array(mulls_list)
        mana_arr_full = np.array(mana_spent_list)
        mull_mask = mulls_arr > 0
        mull_rate = float(np.mean(mull_mask))
        mean_mana_with_mull = float(np.mean(mana_arr_full[mull_mask])) if mull_mask.any() else mean_mana
        mean_mana_no_mull = float(np.mean(mana_arr_full[~mull_mask])) if (~mull_mask).any() else mean_mana

        return SimulationResult(
            land_count=self.land_count,
            mean_mana=mean_mana,
            mean_mana_value=mean_mana_value,
            mean_mana_draw=mean_mana_draw,
            mean_mana_ramp=mean_mana_ramp,
            mean_mana_total=mean_mana_total,
            mean_hand_sum=mean_hand_sum,
            consistency=consistency,
            mean_bad_turns=mean_bad_turns,
            mean_mid_turns=mean_mid_turns,
            mean_lands=mean_lands,
            mean_mulls=mean_mulls,
            mean_draws=mean_draws,
            mean_spells_cast=mean_spells_cast,
            mean_ecms=mean_ecms,
            percentile_10=percentile_10,
            percentile_25=percentile_25,
            percentile_50=percentile_50,
            percentile_75=percentile_75,
            percentile_90=percentile_90,
            threshold_percent=threshold_percent,
            threshold_mana=threshold_mana,
            ceiling_mana=ceiling_mana,
            quartile_mana=quartile_mana,
            con_threshold=con_threshold,
            mean_mana_per_turn=mean_mana_per_turn,
            mean_spells_per_turn=mean_spells_per_turn,
            std_mana=std_mana,
            mull_rate=mull_rate,
            mean_mana_with_mull=mean_mana_with_mull,
            mean_mana_no_mull=mean_mana_no_mull,
            mana_source_count=self.mana_source_count,
            draw_count=self.draw_count,
            early_count=self.early_count,
            avg_cmc=self.avg_cmc,
            ci_mana_value=ci_mana_value,
            ci_mana_draw=ci_mana_draw,
            ci_mana_ramp=ci_mana_ramp,
            ci_mana=ci_mana,
            ci_mana_total=ci_mana_total,
            ci_mean_mana=ci_mean_mana,
            ci_consistency=ci_consistency,
            ci_mean_bad_turns=ci_mean_bad_turns,
            distribution_stats=distribution_stats,
            card_performance=card_performance,
            replay_data=replay_data,
        )

    def simulate(self, progress_callback=None) -> SimulationResult:
        """Run all simulations and return a ``SimulationResult``.

        Args:
            progress_callback: Optional callable(current, total) for progress updates.
        """
        if self.workers > 1 and ProcessPoolExecutor is not None:
            return self._simulate_from_raw(self._run_parallel())

        sample_games = max(self.sims / 10, 100)
        top_centile_threshold = None
        game_records: dict[str, dict[str, list]] = {
            k: defaultdict(list)
            for k in [
                "top_centile", "low_centile",
                "top_decile", "low_decile",
                "top_quartile", "low_quartile",
                "top_half", "low_half",
            ]
        }

        mana_spent_list = []
        mana_value_list = []
        mana_draw_list = []
        mana_ramp_list = []
        ecms_list = []
        primary_list = []
        hand_sum_list = []
        mulls_list = []
        lands_played_list = []
        cards_drawn_list = []
        spells_cast_list = []
        bad_turns_list = []
        mid_turns_list = []
        mana_per_turn_sum = [0] * self.turns
        spells_per_turn_sum = [0] * self.turns
        card_cast_turn_list: list[list] = [[] for _ in self.decklist]
        card_mana_spent_list: list[list] = [[] for _ in self.decklist]
        played_cards_per_game: list[set] = []
        drawn_cards_per_game: list[set] = []
        replay_buckets: dict[str, list] = {"top": [], "mid": [], "low": []}

        game_iter = range(self.sims)
        if progress_callback is None:
            game_iter = tqdm(game_iter, leave=False)

        for j in game_iter:
            if progress_callback is not None:
                progress_callback(j, self.sims)
            if self.seed is not None:
                random.seed(self.seed + j)
            state = self._reset()
            mulligans = self._mulligan(state)

            total_mana_spent = 0
            game_mana_value = 0
            game_mana_draw = 0
            game_mana_ramp = 0
            game_ecms = 0
            game_hand_sum = 0
            lands_played = 0
            bad_turns = 0
            mid_turns = 0
            total_spells_cast = 0
            all_cards_played: list[Card] = []

            # Replay capture: only when thresholds are available and buckets not full
            _capture_replay = (
                top_centile_threshold is not None
                and not all(len(b) >= 10 for b in replay_buckets.values())
            )
            turn_snapshots: list[dict] = []
            starting_hand_names: list[str] = []
            if _capture_replay:
                starting_hand_names = [self._card_by_id(idx).name for idx in state.hand]

            for i in range(self.turns):
                turn_mana_value = 0
                turn_mana_draw = 0
                turn_mana_ramp = 0
                spells_played = 0

                if _capture_replay:
                    hand_before = [self._card_by_id(idx).name for idx in state.hand]

                played = self._take_turn(state)

                for card in played:
                    all_cards_played.append(card)
                    if card.land:
                        lands_played += 1
                    if card.spell:
                        spells_played += 1
                        cost = card.mana_spent_when_played
                        if card.draw:
                            turn_mana_draw += cost
                        elif card.ramp:
                            turn_mana_ramp += cost
                        else:
                            turn_mana_value += cost
                mana_spent = turn_mana_value + turn_mana_draw
                game_mana_value += turn_mana_value
                game_mana_draw += turn_mana_draw
                game_mana_ramp += turn_mana_ramp
                # ECMS: compound mana forward for remaining turns
                if self.mana_mode == "value":
                    turn_ecms_mana = turn_mana_value
                elif self.mana_mode == "value_draw":
                    turn_ecms_mana = turn_mana_value + turn_mana_draw
                else:
                    turn_ecms_mana = turn_mana_value + turn_mana_draw + turn_mana_ramp
                game_ecms += turn_ecms_mana * (self.turns - i)
                game_hand_sum += min(len(state.hand), 7)
                total_spells_cast += spells_played
                mana_per_turn_sum[i] += mana_spent
                spells_per_turn_sum[i] += spells_played
                if spells_played == 0 and state.deck:
                    bad_turns += 1
                if spells_played < 2 and state.deck and mana_spent < i + 1:
                    mid_turns += 1
                total_mana_spent += mana_spent

                if _capture_replay:
                    turn_snapshots.append({
                        "turn": i + 1,
                        "hand_before_draw": hand_before,
                        "played": [
                            {
                                "name": c.name,
                                "cost": c.cost,
                                "mana_spent": c.mana_spent_when_played,
                                "is_land": c.land,
                            }
                            for c in played
                        ],
                        "mana_spent_this_turn": mana_spent,
                        "total_mana_production": self._get_mana(state),
                        "hand_after": [self._card_by_id(idx).name for idx in state.hand],
                        "battlefield": [self._card_by_id(idx).name for idx in state.battlefield],
                        "lands": [self._card_by_id(idx).name for idx in state.lands],
                        "graveyard": [self._card_by_id(idx).name for idx in state.yard],
                    })

            mana_spent_list.append(total_mana_spent)
            mana_value_list.append(game_mana_value)
            mana_draw_list.append(game_mana_draw)
            mana_ramp_list.append(game_mana_ramp)
            ecms_list.append(game_ecms)

            if self.mana_mode == "value":
                game_primary = game_mana_value
            elif self.mana_mode == "value_draw":
                game_primary = total_mana_spent
            else:
                game_primary = game_mana_value + game_mana_draw + game_mana_ramp
            primary_list.append(game_primary)
            hand_sum_list.append(game_hand_sum)
            lands_played_list.append(lands_played)
            mulls_list.append(mulligans)
            cards_drawn_list.append(state.draws)
            spells_cast_list.append(total_spells_cast)
            bad_turns_list.append(bad_turns)
            mid_turns_list.append(mid_turns)

            game_played = set()
            for k, turn in enumerate(state.card_cast_turn):
                if turn is not None and not self.decklist[k].land:
                    card_cast_turn_list[k].append(turn)
                    card_mana_spent_list[k].append(self.decklist[k].mana_spent_when_played)
                    if self.decklist[k].spell:
                        game_played.add(k)
            played_cards_per_game.append(game_played)

            # Cards drawn = all spell cards not still in deck or command zone
            deck_set = set(state.deck) | set(state.command_zone)
            game_drawn = {k for k in range(len(self.decklist))
                          if k not in deck_set and self.decklist[k].spell and not self.decklist[k].land}
            drawn_cards_per_game.append(game_drawn)

            # Record games in buckets (based on primary mana mode)
            if j > sample_games:
                if top_centile_threshold is None:
                    top_centile_threshold = np.percentile(primary_list, 99)
                    low_centile_threshold = np.percentile(primary_list, 1)
                    top_decile_threshold = np.percentile(primary_list, 90)
                    low_decile_threshold = np.percentile(primary_list, 10)
                    top_quartile_threshold = np.percentile(primary_list, 75)
                    low_quartile_threshold = np.percentile(primary_list, 25)
                    median_threshold = np.percentile(primary_list, 50)
                else:
                    record_games = []
                    if self.record_centile and game_primary >= top_centile_threshold:
                        record_games.extend(["top_centile", "top_decile", "top_quartile", "top_half"])
                    elif self.record_decile and game_primary >= top_decile_threshold:
                        record_games.extend(["top_decile", "top_quartile", "top_half"])
                    elif self.record_quartile and game_primary >= top_quartile_threshold:
                        record_games.extend(["top_quartile", "top_half"])
                    elif self.record_half and game_primary >= median_threshold:
                        record_games.append("top_half")

                    if self.record_centile and game_primary <= low_centile_threshold:
                        record_games.extend(["low_centile", "low_decile", "low_quartile", "low_half"])
                    elif self.record_decile and game_primary <= low_decile_threshold:
                        record_games.extend(["low_decile", "low_quartile", "low_half"])
                    elif self.record_quartile and game_primary <= low_quartile_threshold:
                        record_games.extend(["low_quartile", "low_half"])
                    elif self.record_half and game_primary < median_threshold:
                        record_games.append("low_half")

                    for rg in record_games:
                        if len(game_records[rg]["logs"]) < 10:
                            game_records[rg]["logs"].append(state.log)
                        game_records[rg]["mana"].append(total_mana_spent)
                        game_records[rg]["lands"].append(lands_played)
                        game_records[rg]["mulls"].append(mulligans)
                        game_records[rg]["draws"].append(state.draws)
                        game_records[rg]["bad_turns"].append(bad_turns)
                        game_records[rg]["mid_turns"].append(mid_turns)
                        game_records[rg]["surplus mana production"].append(
                            self._get_mana(state) - lands_played
                        )
                        game_records[rg]["nonpermanent cost reduction"].append(
                            state.nonpermanent_cost_reduction
                        )
                        game_records[rg]["permanent cost reduction"].append(
                            state.permanent_cost_reduction
                        )
                        game_records[rg]["spell cost reduction"].append(
                            state.spell_cost_reduction
                        )
                        game_records[rg]["creature cost reduction"].append(
                            state.creature_cost_reduction
                        )
                        game_records[rg]["starting hand land count"].append(
                            state.starting_hand_land_count
                        )
                        game_records[rg]["per turn effects"].append(
                            [c.unique_name for c, _ in state.per_turn_effects]
                        )
                        game_records[rg]["cast triggers"].append(
                            [c.unique_name for c, _ in state.cast_triggers]
                        )
                        game_records[rg]["starting hand"].append(
                            [c.unique_name for c in state.starting_hand]
                        )
                        game_records[rg]["played cards"].append(
                            [c.unique_name for c in all_cards_played]
                        )

            # Classify game into replay buckets (based on primary mana mode)
            if _capture_replay:
                game_replay = {
                    "total_mana": total_mana_spent,
                    "mulligans": mulligans,
                    "starting_hand": starting_hand_names,
                    "turns": turn_snapshots,
                }
                if game_primary >= top_quartile_threshold:
                    if len(replay_buckets["top"]) < 10:
                        replay_buckets["top"].append(game_replay)
                elif game_primary <= low_quartile_threshold:
                    if len(replay_buckets["low"]) < 10:
                        replay_buckets["low"].append(game_replay)
                else:
                    if len(replay_buckets["mid"]) < 10:
                        replay_buckets["mid"].append(game_replay)

            if self.verbose:
                for line in state.log:
                    print(line)
                print(f"\n### Game {j + 1} finished")

        # Compute summary stats
        mean_mana = float(np.mean(mana_spent_list))
        mean_mana_value = float(np.mean(mana_value_list))
        mean_mana_draw = float(np.mean(mana_draw_list))
        mean_mana_ramp = float(np.mean(mana_ramp_list))
        mean_mana_total = float(np.mean([v + d + r for v, d, r in zip(mana_value_list, mana_draw_list, mana_ramp_list)]))
        mean_ecms = float(np.mean(ecms_list))
        mean_hand_sum = float(np.mean(hand_sum_list))
        mean_lands = float(np.mean(lands_played_list))
        mean_mulls = float(np.mean(mulls_list))
        mean_draws = float(np.mean(cards_drawn_list))
        mean_spells_cast = float(np.mean(spells_cast_list))
        mean_bad_turns = float(np.mean(bad_turns_list))
        mean_mid_turns = float(np.mean(mid_turns_list))
        percentile_10 = float(np.percentile(primary_list, 10))
        percentile_25 = float(np.percentile(primary_list, 25))
        percentile_50 = float(np.percentile(primary_list, 50))
        percentile_75 = float(np.percentile(primary_list, 75))
        percentile_90 = float(np.percentile(primary_list, 90))

        # Left-tail ratio consistency: mean(bottom 25%) / mean(all)
        con_threshold = 0.25
        sorted_primary = sorted(primary_list)
        n = len(mana_spent_list)
        cutoff = max(1, int(n * con_threshold))
        tail_mean = float(np.mean(sorted_primary[:cutoff]))
        overall_primary_mean = float(np.mean(primary_list))
        if overall_primary_mean == 0:
            consistency = 1.0
        else:
            consistency = tail_mean / overall_primary_mean
        threshold_percent = con_threshold
        threshold_mana = tail_mean

        # Top-25% mean (ceiling) and per-quartile mana breakdowns
        top_cutoff = max(1, int(n * (1 - con_threshold)))
        ceiling_mana = float(np.mean(sorted_primary[top_cutoff:]))

        primary_arr = np.array(primary_list)
        val_arr = np.array(mana_value_list)
        draw_arr = np.array(mana_draw_list)
        ramp_arr = np.array(mana_ramp_list)
        vd_arr = val_arr + draw_arr
        all_arr = val_arr + draw_arr + ramp_arr

        sort_idx = np.argsort(primary_arr)
        bot_idx = sort_idx[:cutoff]
        top_idx = sort_idx[top_cutoff:]

        quartile_mana = {
            "bottom_25": {
                "value": float(np.mean(val_arr[bot_idx])),
                "draw": float(np.mean(draw_arr[bot_idx])),
                "ramp": float(np.mean(ramp_arr[bot_idx])),
                "vd": float(np.mean(vd_arr[bot_idx])),
                "all": float(np.mean(all_arr[bot_idx])),
            },
            "mean": {
                "value": float(np.mean(val_arr)),
                "draw": float(np.mean(draw_arr)),
                "ramp": float(np.mean(ramp_arr)),
                "vd": float(np.mean(vd_arr)),
                "all": float(np.mean(all_arr)),
            },
            "top_25": {
                "value": float(np.mean(val_arr[top_idx])),
                "draw": float(np.mean(draw_arr[top_idx])),
                "ramp": float(np.mean(ramp_arr[top_idx])),
                "vd": float(np.mean(vd_arr[top_idx])),
                "all": float(np.mean(all_arr[top_idx])),
            },
        }

        # Compute 95% confidence intervals
        z = 1.96
        sqrt_n = np.sqrt(n)

        mana_total_list = [v + d + r for v, d, r in zip(mana_value_list, mana_draw_list, mana_ramp_list)]
        ci_mana_value = z * float(np.std(mana_value_list, ddof=1)) / sqrt_n
        ci_mana_draw = z * float(np.std(mana_draw_list, ddof=1)) / sqrt_n
        ci_mana_ramp = z * float(np.std(mana_ramp_list, ddof=1)) / sqrt_n
        ci_mana = z * float(np.std(mana_spent_list, ddof=1)) / sqrt_n
        ci_mana_total = z * float(np.std(mana_total_list, ddof=1)) / sqrt_n

        ci_mean_mana = (mean_mana - ci_mana, mean_mana + ci_mana)

        bad_se = float(np.std(bad_turns_list, ddof=1) / sqrt_n)
        ci_mean_bad_turns = (mean_bad_turns - z * bad_se, mean_bad_turns + z * bad_se)

        # Bootstrap CI for consistency (left-tail ratio)
        n_boot = min(1000, n)
        boot_consistencies = []
        mana_arr = np.array(primary_list)
        boot_rng = np.random.RandomState(self.seed if self.seed is not None else 42)
        for _ in range(n_boot):
            boot_sample = boot_rng.choice(mana_arr, size=n, replace=True)
            boot_sorted = np.sort(boot_sample)
            boot_tail = float(np.mean(boot_sorted[:cutoff]))
            boot_overall = float(np.mean(boot_sample))
            if boot_overall == 0:
                boot_consistencies.append(1.0)
            else:
                boot_consistencies.append(boot_tail / boot_overall)
        ci_consistency = (
            float(np.percentile(boot_consistencies, 2.5)),
            float(np.percentile(boot_consistencies, 97.5)),
        )

        distribution_stats = self._compute_distribution_stats(primary_list)
        card_performance = self._compute_card_performance(
            primary_list, played_cards_per_game, drawn_cards_per_game,
            card_mana_spent_list=card_mana_spent_list,
        )

        # Per-turn averages
        mean_mana_per_turn = [s / n for s in mana_per_turn_sum] if n > 0 else []
        mean_spells_per_turn = [s / n for s in spells_per_turn_sum] if n > 0 else []

        # Scoring aggregates
        std_mana = float(np.std(mana_spent_list, ddof=1)) if n > 1 else 0.0
        mulls_arr = np.array(mulls_list)
        mana_arr_full = np.array(mana_spent_list)
        mull_mask = mulls_arr > 0
        mull_rate = float(np.mean(mull_mask))
        mean_mana_with_mull = float(np.mean(mana_arr_full[mull_mask])) if mull_mask.any() else mean_mana
        mean_mana_no_mull = float(np.mean(mana_arr_full[~mull_mask])) if (~mull_mask).any() else mean_mana

        return SimulationResult(
            land_count=self.land_count,
            mean_mana=mean_mana,
            mean_mana_value=mean_mana_value,
            mean_mana_draw=mean_mana_draw,
            mean_mana_ramp=mean_mana_ramp,
            mean_mana_total=mean_mana_total,
            mean_hand_sum=mean_hand_sum,
            consistency=consistency,
            mean_bad_turns=mean_bad_turns,
            mean_mid_turns=mean_mid_turns,
            mean_lands=mean_lands,
            mean_mulls=mean_mulls,
            mean_draws=mean_draws,
            mean_spells_cast=mean_spells_cast,
            mean_ecms=mean_ecms,
            percentile_10=percentile_10,
            percentile_25=percentile_25,
            percentile_50=percentile_50,
            percentile_75=percentile_75,
            percentile_90=percentile_90,
            threshold_percent=threshold_percent,
            threshold_mana=threshold_mana,
            ceiling_mana=ceiling_mana,
            quartile_mana=quartile_mana,
            con_threshold=con_threshold,
            mean_mana_per_turn=mean_mana_per_turn,
            mean_spells_per_turn=mean_spells_per_turn,
            std_mana=std_mana,
            mull_rate=mull_rate,
            mean_mana_with_mull=mean_mana_with_mull,
            mean_mana_no_mull=mean_mana_no_mull,
            mana_source_count=self.mana_source_count,
            draw_count=self.draw_count,
            early_count=self.early_count,
            avg_cmc=self.avg_cmc,
            ci_mana_value=ci_mana_value,
            ci_mana_draw=ci_mana_draw,
            ci_mana_ramp=ci_mana_ramp,
            ci_mana=ci_mana,
            ci_mana_total=ci_mana_total,
            distribution_stats=distribution_stats,
            card_performance=card_performance,
            game_records=dict(game_records),
            replay_data=replay_buckets,
            ci_mean_mana=ci_mean_mana,
            ci_consistency=ci_consistency,
            ci_mean_bad_turns=ci_mean_bad_turns,
        )

    def simulate_single_game(self, seed: int, ecms: bool = False) -> float:
        """Run one game with a specific seed, return the primary mana value.

        Lightweight path for optimization — skips recording, logging,
        replays, and all bookkeeping beyond what's needed for the primary
        mana metric. Uses the same game logic as ``simulate()``.

        Args:
            seed: Random seed for this game.
            ecms: If True, return the Karsten-ECMS value instead of
                primary mana.

        Returns:
            The primary mana value (depends on ``self.mana_mode``),
            or the ECMS value if *ecms* is True.
        """
        random.seed(seed)
        state = self._reset()
        self._mulligan(state)

        game_mana_value = 0
        game_mana_draw = 0
        game_mana_ramp = 0
        game_ecms = 0

        for turn_idx in range(self.turns):
            played = self._take_turn(state)
            turn_mana_value = 0
            turn_mana_draw = 0
            turn_mana_ramp = 0
            for card in played:
                if card.spell:
                    cost = card.mana_spent_when_played
                    if card.draw:
                        turn_mana_draw += cost
                    elif card.ramp:
                        turn_mana_ramp += cost
                    else:
                        turn_mana_value += cost
            game_mana_value += turn_mana_value
            game_mana_draw += turn_mana_draw
            game_mana_ramp += turn_mana_ramp
            if ecms:
                if self.mana_mode == "value":
                    turn_ecms_mana = turn_mana_value
                elif self.mana_mode == "value_draw":
                    turn_ecms_mana = turn_mana_value + turn_mana_draw
                else:
                    turn_ecms_mana = turn_mana_value + turn_mana_draw + turn_mana_ramp
                game_ecms += turn_ecms_mana * (self.turns - turn_idx)

        if ecms:
            return float(game_ecms)

        if self.mana_mode == "value":
            return float(game_mana_value)
        elif self.mana_mode == "value_draw":
            return float(game_mana_value + game_mana_draw)
        else:
            return float(game_mana_value + game_mana_draw + game_mana_ramp)
