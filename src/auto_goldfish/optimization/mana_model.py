"""Hypergeometric mana model -- closed-form land count recommendations.

Pure math functions using ``math.comb`` (no scipy needed).
Answers "how many lands should I run?" in microseconds.
"""

from __future__ import annotations

from math import comb
from typing import Any, Dict, List, Optional, Tuple

# Mulligan keep range: hands with 2-5 lands are kept, others mulliganed.
DEFAULT_KEEP_RANGE: Tuple[int, int] = (2, 5)

# Land-count search bounds for the optimization sweep.
DEFAULT_SEARCH_RANGE: Tuple[int, int] = (25, 45)

# Composite score weights (must sum to 1.0).
SCORE_WEIGHT_CURVE = 0.35
SCORE_WEIGHT_MANA = 0.25
SCORE_WEIGHT_MULLIGAN = 0.20
SCORE_WEIGHT_FLOOD = 0.20

# Fraction of draw spells assumed to fire by mid-game (conservative).
DRAW_EARLY_FRACTION = 0.5

# Flood threshold: 7+ lands drawn by turn 5 (in 11 cards seen).
FLOOD_LAND_COUNT = 7
FLOOD_TURN = 5

# Composite score weights when 2+ partner commanders are active.
# Carves 0.05 from each base component to make room for the partner term.
SCORE_WEIGHT_CURVE_PARTNER = 0.30
SCORE_WEIGHT_MANA_PARTNER = 0.20
SCORE_WEIGHT_MULLIGAN_PARTNER = 0.15
SCORE_WEIGHT_FLOOD_PARTNER = 0.15
SCORE_WEIGHT_PARTNER = 0.20


# ---------------------------------------------------------------------------
# Core hypergeometric distribution
# ---------------------------------------------------------------------------

def hypergeometric_pmf(k: int, N: int, K: int, n: int) -> float:
    """P(X = k): probability of drawing exactly *k* successes.

    Parameters
    ----------
    k : int  -- number of observed successes
    N : int  -- population size (deck size)
    K : int  -- number of success states in population (e.g. lands)
    n : int  -- number of draws (cards seen)
    """
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    return comb(K, k) * comb(N - K, n - k) / comb(N, n)


def hypergeometric_cdf(k: int, N: int, K: int, n: int) -> float:
    """P(X <= k): cumulative probability of at most *k* successes."""
    return sum(hypergeometric_pmf(i, N, K, n) for i in range(k + 1))


def prob_at_least(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) = 1 - CDF(k - 1)."""
    if k <= 0:
        return 1.0
    return 1.0 - hypergeometric_cdf(k - 1, N, K, n)


# ---------------------------------------------------------------------------
# Expected mana helpers
# ---------------------------------------------------------------------------

def expected_mana_on_turn(turn: int, N: int, K: int) -> float:
    """Expected mana available on a given turn (land-per-turn cap).

    On turn T you've seen 7 + (T - 1) = T + 6 cards.
    You can play at most T lands, so expected mana = E[min(lands_drawn, T)].
    """
    cards_seen = 7 + turn - 1  # opening hand + draws
    if cards_seen > N:
        cards_seen = N
    total = 0.0
    for lands in range(cards_seen + 1):
        p = hypergeometric_pmf(lands, N, K, cards_seen)
        total += p * min(lands, turn)
    return total


def expected_mana_table(
    N: int, K: int, max_turn: int = 10
) -> List[Dict[str, Any]]:
    """Turn-by-turn table with expected mana, on-curve probability, and screw probability.

    Returns a list of dicts, one per turn (1..max_turn).
    """
    rows = []
    for t in range(1, max_turn + 1):
        cards_seen = min(7 + t - 1, N)
        e_mana = expected_mana_on_turn(t, N, K)
        p_on_curve = prob_at_least(t, N, K, cards_seen)
        # Screw = fewer than ceil(t/2) lands (missing half your drops)
        screw_threshold = max(1, (t + 1) // 2)
        p_screw = hypergeometric_cdf(screw_threshold - 1, N, K, cards_seen)
        rows.append({
            "turn": t,
            "cards_seen": cards_seen,
            "expected_mana": round(e_mana, 3),
            "prob_on_curve": round(p_on_curve, 4),
            "prob_screw": round(p_screw, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Mulligan model (London mulligan approximation)
# ---------------------------------------------------------------------------

def mulligan_probability(
    N: int, K: int, keep_range: Tuple[int, int] = DEFAULT_KEEP_RANGE
) -> float:
    """Probability of mulliganing a 7-card hand.

    Keeps hands with *keep_range[0]* to *keep_range[1]* lands (inclusive).
    """
    p_keep = sum(
        hypergeometric_pmf(lands, N, K, 7)
        for lands in range(keep_range[0], keep_range[1] + 1)
    )
    return 1.0 - p_keep


# ---------------------------------------------------------------------------
# Ramp and draw adjustments
# ---------------------------------------------------------------------------

def adjusted_expected_mana(
    turn: int,
    N: int,
    K: int,
    ramp_cards: int = 0,
    draw_cards: int = 0,
    avg_ramp_cmc: float = 2.0,
    avg_ramp_amount: float = 1.0,
    avg_draw_amount: float = 1.0,
) -> float:
    """Expected mana with ramp and draw effects factored in.

    Ramp model: Each ramp card adds *avg_ramp_amount* mana for turns after
    it can be cast (turn > avg_ramp_cmc). Probability of having it in hand
    is approximated as ramp_cards / (N - K) * cards_seen / N.

    Draw model: Extra draws increase effective cards seen for subsequent turns.
    """
    base_mana = expected_mana_on_turn(turn, N, K)

    # Ramp contribution
    ramp_bonus = 0.0
    if ramp_cards > 0 and turn > avg_ramp_cmc:
        cards_seen = 7 + turn - 1
        non_lands = N - K
        if non_lands > 0:
            # P(at least one ramp card in hand by the cast turn)
            ramp_cast_turn = int(avg_ramp_cmc)
            cards_at_cast = min(7 + ramp_cast_turn - 1, N)
            p_ramp_in_hand = 1.0 - hypergeometric_cdf(
                0, N, ramp_cards, cards_at_cast
            )
            # P(enough mana to cast it)
            p_castable = prob_at_least(ramp_cast_turn, N, K, cards_at_cast)
            ramp_bonus = p_ramp_in_hand * p_castable * avg_ramp_amount

    # Draw contribution: extra cards seen improve land probability
    draw_bonus = 0.0
    if draw_cards > 0 and turn > 1:
        extra_cards = draw_cards * avg_draw_amount * DRAW_EARLY_FRACTION
        augmented_seen = min(7 + turn - 1 + extra_cards, N)
        e_augmented = 0.0
        # Approximate with fractional cards_seen via interpolation
        lo = int(augmented_seen)
        hi = lo + 1
        frac = augmented_seen - lo
        e_lo = expected_mana_on_turn(turn, N, K) if lo == 7 + turn - 1 else _expected_mana_seen(turn, N, K, lo)
        e_hi = _expected_mana_seen(turn, N, K, min(hi, N))
        e_augmented = e_lo * (1 - frac) + e_hi * frac
        draw_bonus = max(0, e_augmented - base_mana)

    return base_mana + ramp_bonus + draw_bonus


def _expected_mana_seen(turn: int, N: int, K: int, cards_seen: int) -> float:
    """Expected mana given a specific number of cards seen (helper)."""
    cards_seen = min(cards_seen, N)
    total = 0.0
    for lands in range(cards_seen + 1):
        p = hypergeometric_pmf(lands, N, K, cards_seen)
        total += p * min(lands, turn)
    return total


# ---------------------------------------------------------------------------
# Partner commanders -- joint castable probability
# ---------------------------------------------------------------------------

def prob_both_partners_castable(N: int, K: int, cmc_a: int, cmc_b: int) -> float:
    """P(cast cheaper partner on its CMC turn AND second partner on its CMC turn).

    Sequential model: cast partner with cmc_a on turn cmc_a, then partner
    with cmc_b on turn cmc_b. The events are not independent (lands persist
    across turns), so this iterates over the joint distribution of land
    draws at turn A and the additional draws between turns A and B.
    """
    if cmc_a <= 0 or cmc_b <= 0:
        return 0.0
    if cmc_a > cmc_b:
        cmc_a, cmc_b = cmc_b, cmc_a

    cards_at_a = min(7 + cmc_a - 1, N)
    cards_at_b = min(7 + cmc_b - 1, N)
    extra = cards_at_b - cards_at_a

    total = 0.0
    for lands_a in range(cmc_a, min(cards_at_a, K) + 1):
        p_a = hypergeometric_pmf(lands_a, N, K, cards_at_a)
        if p_a == 0.0:
            continue
        needed = max(0, cmc_b - lands_a)
        remaining_pop = N - cards_at_a
        remaining_lands = K - lands_a
        if needed > extra or needed > remaining_lands:
            continue
        p_more = prob_at_least(needed, remaining_pop, remaining_lands, extra)
        total += p_a * p_more
    return total


# ---------------------------------------------------------------------------
# Optimal land count
# ---------------------------------------------------------------------------

def optimal_land_count(
    deck_size: int = 99,
    cmc_distribution: Optional[Dict[int, int]] = None,
    ramp_cards: int = 0,
    draw_cards: int = 0,
    commander_cmc: int = 0,
    commander_cmcs: Optional[List[int]] = None,
    search_range: Tuple[int, int] = DEFAULT_SEARCH_RANGE,
) -> Dict[str, Any]:
    """Find the optimal land count by sweeping K and scoring each.

    Score is a weighted composite of:
    - On-curve probability through key turns
    - Expected mana at critical turns (based on CMC distribution)
    - Mulligan rate penalty
    - Flood penalty
    - Partner co-availability (only when 2+ commanders provided)

    *commander_cmc* (legacy, single int) and *commander_cmcs* (list) are
    both accepted; the list takes precedence when given.
    """
    if cmc_distribution is None:
        cmc_distribution = {}

    if commander_cmcs is None:
        commander_cmcs = [commander_cmc] if commander_cmc > 0 else []

    avg_cmc = _weighted_avg_cmc(cmc_distribution) if cmc_distribution else 3.0
    key_turns = _key_turns_from_cmc(cmc_distribution, commander_cmcs)

    best_score = -1.0
    best_k = search_range[0]
    scores: List[Dict[str, Any]] = []

    for k in range(search_range[0], search_range[1] + 1):
        score = _score_land_count(
            deck_size, k, key_turns, avg_cmc, ramp_cards, draw_cards, commander_cmcs
        )
        scores.append({"land_count": k, "score": round(score, 4)})
        if score > best_score:
            best_score = score
            best_k = k

    result = {
        "recommended_lands": best_k,
        "deck_size": deck_size,
        "avg_cmc": round(avg_cmc, 2),
        "key_turns": key_turns,
        "commander_cmcs": list(commander_cmcs),
        "scores": scores,
    }
    if len(commander_cmcs) >= 2:
        sorted_cmcs = sorted(c for c in commander_cmcs if c > 0)
        result["partner_castable_prob"] = round(
            prob_both_partners_castable(
                deck_size, best_k, sorted_cmcs[0], sorted_cmcs[1]
            ),
            4,
        )
    return result


def _weighted_avg_cmc(cmc_distribution: Dict[int, int]) -> float:
    """Weighted average CMC from a distribution {cmc: count}."""
    total_cards = sum(cmc_distribution.values())
    if total_cards == 0:
        return 3.0
    return sum(cmc * count for cmc, count in cmc_distribution.items()) / total_cards


def _key_turns_from_cmc(
    cmc_distribution: Dict[int, int],
    commander_cmcs: Optional[List[int]] = None,
) -> List[int]:
    """Determine which turns matter most based on curve."""
    turns = set()
    for cmc, count in cmc_distribution.items():
        if count > 0 and 1 <= cmc <= 10:
            turns.add(cmc)
    if commander_cmcs:
        for cmc in commander_cmcs:
            if cmc > 0:
                turns.add(cmc)
    # Always include common critical turns 3-5
    turns.update([3, 4, 5])
    return sorted(turns)


def _score_land_count(
    N: int, K: int,
    key_turns: List[int],
    avg_cmc: float,
    ramp_cards: int,
    draw_cards: int,
    commander_cmcs: Optional[List[int]] = None,
) -> float:
    """Composite score for a given land count.

    Components: on-curve probability, expected mana efficiency, mulligan
    penalty, flood penalty. When 2+ partner commanders are given, a 5th
    partner-castable component is added with rebalanced weights.
    """
    # 1. On-curve score (0-1): weighted average of P(on curve) at key turns
    curve_score = 0.0
    weight_sum = 0.0
    for t in key_turns:
        cards_seen = min(7 + t - 1, N)
        p = prob_at_least(t, N, K, cards_seen)
        # Earlier turns weighted more heavily
        w = 1.0 / t
        curve_score += w * p
        weight_sum += w
    if weight_sum > 0:
        curve_score /= weight_sum

    # 2. Mana efficiency at avg_cmc turn
    target_turn = max(1, int(round(avg_cmc)))
    e_mana = adjusted_expected_mana(target_turn, N, K, ramp_cards, draw_cards)
    mana_score = min(1.0, e_mana / max(target_turn, 1))

    # 3. Mulligan penalty
    p_mull = mulligan_probability(N, K)
    mull_score = 1.0 - p_mull

    # 4. Flood penalty: P(too many lands) by the flood turn
    cards_at_flood = min(7 + FLOOD_TURN - 1, N)
    p_flood = prob_at_least(FLOOD_LAND_COUNT, N, K, cards_at_flood)
    flood_score = 1.0 - p_flood

    if commander_cmcs and len([c for c in commander_cmcs if c > 0]) >= 2:
        sorted_cmcs = sorted(c for c in commander_cmcs if c > 0)
        partner_score = prob_both_partners_castable(
            N, K, sorted_cmcs[0], sorted_cmcs[1]
        )
        return (
            SCORE_WEIGHT_CURVE_PARTNER * curve_score
            + SCORE_WEIGHT_MANA_PARTNER * mana_score
            + SCORE_WEIGHT_MULLIGAN_PARTNER * mull_score
            + SCORE_WEIGHT_FLOOD_PARTNER * flood_score
            + SCORE_WEIGHT_PARTNER * partner_score
        )

    return (
        SCORE_WEIGHT_CURVE * curve_score
        + SCORE_WEIGHT_MANA * mana_score
        + SCORE_WEIGHT_MULLIGAN * mull_score
        + SCORE_WEIGHT_FLOOD * flood_score
    )


# ---------------------------------------------------------------------------
# Land count comparison
# ---------------------------------------------------------------------------

def land_count_comparison(
    deck_size: int,
    land_counts: List[int],
    max_turn: int = 10,
) -> List[Dict[str, Any]]:
    """Side-by-side comparison of multiple land counts.

    Returns a list of dicts, one per land count, each containing
    the mana table and summary stats.
    """
    results = []
    for k in land_counts:
        table = expected_mana_table(deck_size, k, max_turn)
        p_mull = mulligan_probability(deck_size, k)
        results.append({
            "land_count": k,
            "land_ratio": round(k / deck_size, 3),
            "mulligan_rate": round(p_mull, 4),
            "mana_table": table,
        })
    return results
