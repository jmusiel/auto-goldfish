"""Output formatting -- text reports, plots, JSON-serializable dicts."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np

from auto_goldfish.engine.goldfisher import SimulationResult
from auto_goldfish.models.card import Card


def save_report(
    result: SimulationResult,
    decklist: List[Card],
    commanders: List[str],
    card_cast_turn_list: List[List[int]] | None = None,
    output_dir: str = ".",
    deck_name: str = "deck",
) -> None:
    """Write a text report and mana curve plot to *output_dir*."""
    if plt is None:
        raise ImportError("matplotlib is required for save_report")
    os.makedirs(output_dir, exist_ok=True)

    cmc_list = [c.cmc for c in decklist]

    # Build cast turn stats
    if card_cast_turn_list is None:
        card_cast_turn_list = [[] for _ in decklist]

    cast_turn_mean = []
    cast_turn_std = []
    for ct in card_cast_turn_list:
        if not ct:
            cast_turn_mean.append(np.nan)
            cast_turn_std.append(np.nan)
        else:
            cast_turn_mean.append(np.mean(ct))
            cast_turn_std.append(np.std(ct))

    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(cmc_list, cast_turn_mean, yerr=cast_turn_std, fmt="o")
    ax.set_xlabel("Mana Value")
    ax.set_ylabel("Cast Turn")
    ax.set_title(f"Card Cast Turn {deck_name} with {result.land_count} lands")
    fig.savefig(os.path.join(output_dir, f"{deck_name}_auto_goldfish_{result.land_count}_lands.png"))
    plt.close(fig)

    # Text report
    report_path = os.path.join(output_dir, f"{deck_name}_record_{result.land_count}_lands.txt")
    with open(report_path, "w") as f:
        f.write(f"Decklist: {deck_name}\n")
        f.write(f"Commanders: {', '.join(commanders)}\n")
        f.write(f"Land Count: {result.land_count}\n\n")

        f.write("Decklist:\n")
        f.write("Name | (cmc) | (cast turn)\n")
        for card, cmc, ct in zip(decklist, cmc_list, cast_turn_mean):
            f.write(f"{card.name} ({cmc}) ({ct})\n")
        f.write("\n")

        f.write("Game Records:\n")
        for quantile, record in result.game_records.items():
            f.write(f"{'=' * 70}\n")
            f.write(f"{'=' * 20} {quantile} games {'=' * 20}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"num games in {quantile}: {len(record.get('mana', []))}\n\n")

            card_stats: dict[str, list] = {}
            for key, value in record.items():
                if key == "logs":
                    continue
                if key in ("per turn effects", "cast triggers", "starting hand", "played cards"):
                    superlist = []
                    for sublist in value:
                        superlist.extend(sublist)
                    card_stats[key] = Counter(superlist).most_common(10)
                else:
                    f.write(f"{key}: {np.mean(value)}\n")
            f.write("\n")

            for key, value in card_stats.items():
                f.write(f"most common {key}:\n")
                for card_name, count in value:
                    f.write(f"\t{count} {card_name}\n")
            f.write("\n\n")

        # Example game logs
        for quantile, record in result.game_records.items():
            f.write(f"{'=' * 70}\n")
            f.write(f"{'=' * 20} {quantile} example games {'=' * 20}\n")
            f.write(f"{'=' * 70}\n")
            for i, log in enumerate(record.get("logs", [])):
                f.write(f"--- {quantile} example game #{i} ---\n")
                f.writelines(line + "\n" for line in log)
                f.write("\n")


def result_to_dict(result: SimulationResult, turns: int = 10) -> Dict[str, Any]:
    """Convert SimulationResult to a JSON-serializable dict."""
    from auto_goldfish.metrics.deck_score import compute_raw_stats, score_from_raw
    raw = compute_raw_stats(result, turns)
    score = score_from_raw(raw)
    return {
        "deck_score": score.as_dict(),
        "deck_raw": raw.as_dict(),
        "land_count": result.land_count,
        "mean_mana": result.mean_mana,
        "mean_mana_value": result.mean_mana_value,
        "mean_mana_draw": result.mean_mana_draw,
        "mean_mana_ramp": result.mean_mana_ramp,
        "mean_mana_total": result.mean_mana_total,
        "mean_hand_sum": result.mean_hand_sum,
        "consistency": result.consistency,
        "mean_bad_turns": result.mean_bad_turns,
        "mean_mid_turns": result.mean_mid_turns,
        "mean_lands": result.mean_lands,
        "mean_mulls": result.mean_mulls,
        "mean_draws": result.mean_draws,
        "mean_spells_cast": result.mean_spells_cast,
        "mean_ecms": result.mean_ecms,
        "percentile_25": result.percentile_25,
        "percentile_50": result.percentile_50,
        "percentile_75": result.percentile_75,
        "threshold_percent": result.threshold_percent,
        "threshold_mana": result.threshold_mana,
        "ceiling_mana": result.ceiling_mana,
        "quartile_mana": result.quartile_mana,
        "distribution_stats": result.distribution_stats,
        "card_performance": result.card_performance,
        "replay_data": result.replay_data,
        "ci_mana_value": result.ci_mana_value,
        "ci_mana_draw": result.ci_mana_draw,
        "ci_mana_ramp": result.ci_mana_ramp,
        "ci_mana": result.ci_mana,
        "ci_mana_total": result.ci_mana_total,
        "ci_mean_mana": list(result.ci_mean_mana),
        "ci_consistency": list(result.ci_consistency),
        "ci_mean_bad_turns": list(result.ci_mean_bad_turns),
        "mean_mana_per_turn": result.mean_mana_per_turn,
        "mean_spells_per_turn": result.mean_spells_per_turn,
        "std_mana": result.std_mana,
        "mull_rate": result.mull_rate,
        "mean_mana_with_mull": result.mean_mana_with_mull,
        "mean_mana_no_mull": result.mean_mana_no_mull,
    }
