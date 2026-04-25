"""Engineer decks that exercise each recommendation badge.

For draw / ramp / no-effect (vanilla) test pools, build deck shapes that
should drive the recommendation toward `scaling` (↑ Add more), `saturated`
(≈ Enough at k), or `crowding` (↓ Cut copies). Run the simulator on each,
report the badge for the test pool, and flag any case that doesn't match
its target.

Goal: verify the full range of badges is reachable across realistic deck
shapes spanning all three effect types, not that every cell of the
3x3 matrix is engineerable.

Each test pool is given a unique CMC, with filler placed only at *other*
CMCs, so the test pool never merges with same-signature filler.

Known finding: in 8-turn goldfishing, draw cantrips show much smaller
per-game effect sizes than ramp or vanilla curve fillers — there is no
late-game inevitability, so digging deeper rarely changes mana spent in
8 turns. As a result `draw scaling` and `draw saturated` are difficult
to engineer reliably; both pass through "unclear" or flip to "crowding"
under noise. Cantrips do reach `crowding` cleanly when over-loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_goldfish.effects.builtin import DrawCards, ProduceMana
from auto_goldfish.effects.registry import CardEffects, EffectRegistry
from auto_goldfish.engine.goldfisher import Goldfisher


def build_registry() -> EffectRegistry:
    reg = EffectRegistry()
    reg.register(
        "TestPool_draw",
        CardEffects(on_play=[DrawCards(amount=1)], draw=True),
    )
    reg.register(
        "TestPool_ramp",
        CardEffects(on_play=[ProduceMana(1)], ramp=True),
    )
    return reg


def land(idx: int) -> dict:
    return {
        "name": f"Plains_{idx}", "cmc": 0, "cost": "", "text": "",
        "types": ["Land"], "commander": False,
    }


def filler(idx: int, cmc: int) -> dict:
    return {
        "name": f"Filler_{cmc}_{idx}", "cmc": cmc, "cost": f"{{{cmc}}}",
        "text": "", "types": ["Creature"], "commander": False,
    }


def commander() -> dict:
    return {
        "name": "TestCmd", "cmc": 3, "cost": "{1}{W}{U}", "text": "",
        "types": ["Creature"], "commander": True,
    }


def test_pool_card(idx: int, kind: str, cmc: int) -> dict:
    name_map = {
        "vanilla": "TestPool_vanilla",
        "draw": "TestPool_draw",
        "ramp": "TestPool_ramp",
    }
    return {
        "name": name_map[kind], "cmc": cmc, "cost": f"{{{cmc}}}",
        "text": "", "types": ["Sorcery"], "commander": False,
    }


def build_deck(*, kind: str, copies: int, pool_cmc: int, lands: int,
               filler_specs: list[tuple[int, int]]) -> list[dict]:
    """Compose a 100-card deck (1 cmd + 99) with test pool + filler.

    Caller MUST ensure no filler entry shares pool_cmc, otherwise the
    pool will merge with same-cmc vanilla filler and we lose isolation.
    """
    for cmc, _ in filler_specs:
        if cmc == pool_cmc:
            raise ValueError(f"filler at pool_cmc={pool_cmc} would merge")
    deck = [commander()]
    for i in range(lands):
        deck.append(land(i))
    for j in range(copies):
        deck.append(test_pool_card(j, kind, pool_cmc))
    fidx = 0
    for cmc, count in filler_specs:
        for _ in range(count):
            deck.append(filler(fidx, cmc))
            fidx += 1
    # Pad with cmc-4 filler (avoid pool_cmc) to land on exactly 99 non-cmd
    pad_cmc = 4 if pool_cmc != 4 else 5
    while len(deck) - 1 < 99:
        deck.append(filler(fidx, pad_cmc))
        fidx += 1
    while len(deck) - 1 > 99:
        deck.pop()
    return deck


def run_case(deck: list[dict], pool_cmc: int, registry: EffectRegistry,
             sims: int = 2000, seed: int = 7) -> tuple[str, dict | None]:
    gf = Goldfisher(deck, turns=8, sims=sims, record_results="quartile",
                    seed=seed, workers=1, registry=registry, verbose=False)
    result = gf.simulate()
    cp = result.card_performance
    if not cp:
        return ("(no cp)", None)
    all_entries = cp["high_performing"] + cp["low_performing"]
    pool = next((e for e in all_entries if e.get("cmc") == pool_cmc), None)
    if pool is None:
        return ("(pool missing)", None)
    return (pool["saturation"]["badge"], pool)


def fmt_marginals(pool: dict) -> str:
    parts = []
    for m in pool["marginals"]:
        if m["effect"] is None:
            parts.append(f"k{m['k']}=?")
        elif m["noise"]:
            parts.append(f"k{m['k']}≈{m['effect']:+.2f}")
        else:
            parts.append(f"k{m['k']}={m['effect']:+.2f}")
    return " ".join(parts) if parts else "—"


def main():
    reg = build_registry()
    cases = [
        # ---------- SCALING (Add more) ----------
        # Vanilla cmc-1 with mana-flooded deck and no other cheap plays —
        # every additional cmc-1 fills turn 1, where mana would be wasted.
        {
            "name": "vanilla scaling",
            "expected": "scaling",
            "pool_cmc": 1,
            "deck": build_deck(
                kind="vanilla", copies=2, pool_cmc=1,
                lands=55, filler_specs=[(5, 30), (6, 12)],
            ),
        },
        # Cantrip cmc-1 in high-curve mana-rich deck — pay 1 to draw a
        # card you'll cast plus fill turn 1. Use more copies + sims for
        # tighter CIs (cantrips have high per-game variance).
        {
            "name": "draw scaling",
            "expected": "scaling",
            "pool_cmc": 1,
            "sims": 5000,
            "deck": build_deck(
                kind="draw", copies=4, pool_cmc=1,
                lands=50, filler_specs=[(5, 25), (6, 13), (7, 7)],
            ),
        },
        # Ramp cmc-2 in high-curve deck — extra ramp lets you cast big
        # sinks earlier.
        {
            "name": "ramp scaling",
            "expected": "scaling",
            "pool_cmc": 2,
            "deck": build_deck(
                kind="ramp", copies=4, pool_cmc=2,
                lands=30, filler_specs=[(6, 30), (7, 20), (8, 15)],
            ),
        },

        # ---------- CROWDING (Cut copies) ----------
        # Vanilla cmc-8 with too few lands — extra copies of an
        # uncastable bomb just clog the hand.
        {
            "name": "vanilla crowding",
            "expected": "crowding",
            "pool_cmc": 8,
            "deck": build_deck(
                kind="vanilla", copies=8, pool_cmc=8,
                lands=20, filler_specs=[(1, 30), (2, 30), (3, 11)],
            ),
        },
        # Ramp cmc-3 piles up but there are no real sinks for the mana.
        {
            "name": "ramp crowding",
            "expected": "crowding",
            "pool_cmc": 3,
            "deck": build_deck(
                kind="ramp", copies=8, pool_cmc=3,
                lands=18, filler_specs=[(1, 70), (2, 3)],
            ),
        },
        # Cantrip cmc-2 piling up in a low-curve deck — pure padding,
        # displaces real spells.
        {
            "name": "draw crowding",
            "expected": "crowding",
            "pool_cmc": 2,
            "deck": build_deck(
                kind="draw", copies=10, pool_cmc=2,
                lands=18, filler_specs=[(1, 70), (3, 1)],
            ),
        },

        # ---------- SATURATED (Enough at k) ----------
        # Vanilla cmc-2: 4 copies in deck with lots of cmc-2 alternatives
        # already. First copy adds nothing new, marginals all noise but
        # not negative — saturated.
        {
            "name": "vanilla saturated",
            "expected": "saturated",
            "pool_cmc": 2,
            "deck": build_deck(
                kind="vanilla", copies=4, pool_cmc=2,
                lands=37, filler_specs=[(3, 25), (4, 25), (5, 8)],
            ),
        },
        # Ramp cmc-2: 3 copies, balanced curve. First ramp accelerates,
        # subsequent ramps redundant.
        {
            "name": "ramp saturated",
            "expected": "saturated",
            "pool_cmc": 2,
            "deck": build_deck(
                kind="ramp", copies=3, pool_cmc=2,
                lands=34, filler_specs=[(3, 25), (4, 25), (5, 12)],
            ),
        },
        # Cantrip cmc-1: 4 copies in balanced curve. First fills turn 1
        # for free; later copies just trade real spells for redundant
        # cantrips. Bump sims to push past noise floor.
        {
            "name": "draw saturated",
            "expected": "saturated",
            "pool_cmc": 1,
            "sims": 5000,
            "deck": build_deck(
                kind="draw", copies=4, pool_cmc=1,
                lands=37, filler_specs=[(3, 30), (4, 22), (5, 6)],
            ),
        },
    ]

    print(f"{'Case':<22} {'Expected':<10} {'Got':<14} {'Match':<6}  Marginals")
    print("-" * 100)
    hits = 0
    for c in cases:
        badge, pool = run_case(c["deck"], c["pool_cmc"], reg,
                               sims=c.get("sims", 2000))
        match = "OK" if badge == c["expected"] else "MISS"
        if match == "OK":
            hits += 1
        marg_str = fmt_marginals(pool) if pool else "—"
        print(f"{c['name']:<22} {c['expected']:<10} {badge:<14} {match:<6}  {marg_str}")
    print(f"\n{hits}/{len(cases)} cases hit their target badge.")


if __name__ == "__main__":
    main()
