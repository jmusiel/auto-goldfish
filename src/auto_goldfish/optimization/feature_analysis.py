"""Feature analysis for optimization results.

Analyzes which individual card/land changes have the most impact on
simulation scores, using data collected during Hyperband optimization.

Produces:
  1. Marginal impact rankings per feature
  2. OLS/WLS regression with standardized beta coefficients
  3. Synthesized plain-English recommendations
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, List, Tuple

import numpy as np

from auto_goldfish.optimization.candidate_cards import ALL_CANDIDATES
from auto_goldfish.optimization.deck_config import DeckConfig

# Type-only import to avoid circular dependency at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from auto_goldfish.optimization.factored_optimizer import MarginalResult


# ── Feature extraction ─────────────────────────────────────────────────


def extract_features(config: DeckConfig) -> dict[str, int]:
    """Decompose a DeckConfig into a feature dict.

    Features are integer-valued:
      - land_delta: raw value (-2..+2)
      - count_<card_compact_label>: how many copies of that candidate (0-2)
    """
    features: dict[str, int] = {}
    features["land_delta"] = config.land_delta

    card_counts: Counter = Counter(config.added_cards)
    enabled = {
        cid: c for cid, c in ALL_CANDIDATES.items() if c.default_enabled
    }
    for cid, candidate in enabled.items():
        features[f"count_{candidate.compact_label}"] = card_counts.get(cid, 0)

    return features


def configs_to_feature_matrix(
    configs: list[DeckConfig],
) -> tuple[np.ndarray, list[str], list[dict[str, int]]]:
    """Convert configs to a feature matrix.

    Returns (X, feature_names, raw_feature_dicts).
    """
    feature_dicts = [extract_features(c) for c in configs]
    feature_names = sorted(feature_dicts[0].keys())
    X = np.array(
        [[fd[name] for name in feature_names] for fd in feature_dicts],
        dtype=float,
    )
    return X, feature_names, feature_dicts


# ── Aggregate Hyperband scores ─────────────────────────────────────────


def aggregate_hyperband_scores(
    round_scores: list[tuple[DeckConfig, float, int]],
) -> tuple[list[DeckConfig], np.ndarray, np.ndarray]:
    """Aggregate multi-round Hyperband scores per config.

    For each config, computes a sim-weighted average score and a
    confidence weight based on total sims.

    Returns (configs, scores, weights).
    """
    per_config: dict[DeckConfig, list[tuple[float, int]]] = defaultdict(list)
    for config, score, n_sims in round_scores:
        per_config[config].append((score, n_sims))

    configs = []
    scores = []
    weights = []

    for config, observations in per_config.items():
        total_sims = sum(n for _, n in observations)
        weighted_score = sum(s * n for s, n in observations) / total_sims

        configs.append(config)
        scores.append(weighted_score)
        weights.append(np.sqrt(total_sims))

    order = np.argsort(scores)[::-1]
    configs = [configs[i] for i in order]
    scores_arr = np.array([scores[i] for i in order])
    weights_arr = np.array([weights[i] for i in order])

    return configs, scores_arr, weights_arr


# ── Marginal feature impact ───────────────────────────────────────────


def _feature_label(fname: str, value: int) -> str:
    """Human-readable label for a feature=value pair."""
    if fname == "land_delta":
        if value > 0:
            return f"+{value} land"
        elif value < 0:
            return f"{value} land"
        else:
            return "0 land change"
    elif fname.startswith("count_"):
        card = fname.removeprefix("count_")
        if value == 0:
            return f"no {card}"
        return f"{value}x {card}"
    return f"{fname}={value}"


def compute_marginal_impact(
    feature_dicts: list[dict[str, int]],
    scores: np.ndarray,
    feature_names: list[str],
    weights: np.ndarray | None = None,
    score_std: float | None = None,
) -> list[dict[str, Any]]:
    """For each feature value, compute (weighted) mean score when present vs absent.

    Returns list of {feature, value, label, mean_with, mean_without, delta, count}
    sorted by delta descending.
    """
    if weights is None:
        weights = np.ones(len(scores))

    results = []
    global_mean = float(np.average(scores, weights=weights))

    for fname in feature_names:
        values = sorted(set(fd[fname] for fd in feature_dicts))
        if len(values) <= 1:
            continue

        for val in values:
            mask = np.array([fd[fname] == val for fd in feature_dicts])
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            mean_with = float(np.average(scores[mask], weights=weights[mask]))
            mean_without = float(np.average(scores[~mask], weights=weights[~mask]))
            raw_delta = mean_with - mean_without
            entry = {
                "feature": fname,
                "value": val,
                "label": _feature_label(fname, val),
                "mean_with": round(mean_with, 4),
                "mean_without": round(mean_without, 4),
                "delta": round(raw_delta, 4),
                "delta_vs_global": round(mean_with - global_mean, 4),
                "count": int(mask.sum()),
            }
            if score_std is not None and score_std > 0:
                entry["delta_std"] = round(raw_delta / score_std, 4)
            results.append(entry)

    results.sort(key=lambda x: x["delta"], reverse=True)
    return results


# ── OLS / WLS regression ─────────────────────────────────────────────


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Fit (weighted) least squares regression: y = X @ beta + intercept.

    Returns (coefficients, intercept, r_squared).
    """
    n = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X])

    if weights is not None:
        sqrt_w = np.sqrt(weights)
        Xw = X_with_intercept * sqrt_w[:, np.newaxis]
        yw = y * sqrt_w
    else:
        Xw = X_with_intercept
        yw = y

    result = np.linalg.lstsq(Xw, yw, rcond=None)
    beta = result[0]

    intercept = beta[0]
    coefficients = beta[1:]

    y_pred = X_with_intercept @ beta
    if weights is not None:
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        y_wmean = np.average(y, weights=weights)
        ss_tot = np.sum(weights * (y - y_wmean) ** 2)
    else:
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors of coefficients via (X'X)^{-1} * MSE
    p = X_with_intercept.shape[1]
    dof = max(n - p, 1)
    mse = ss_res / dof
    try:
        cov = np.linalg.inv(Xw.T @ Xw) * mse
        std_errors = np.sqrt(np.diag(cov)[1:])  # skip intercept
    except np.linalg.LinAlgError:
        std_errors = np.full(len(coefficients), np.inf)

    return coefficients, intercept, r_squared, std_errors


def regression_analysis(
    X: np.ndarray,
    scores: np.ndarray,
    feature_names: list[str],
    weights: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run OLS/WLS and return ranked coefficients + predicted scores.

    Returns (regression_dict, y_pred).
    """
    coeffs, intercept, r_sq, std_errors = fit_ols(X, scores, weights)
    y_pred = X @ coeffs + intercept

    if weights is not None:
        x_mean = np.average(X, axis=0, weights=weights)
        x_std = np.sqrt(np.average((X - x_mean) ** 2, axis=0, weights=weights))
        y_std = np.sqrt(np.average(
            (scores - np.average(scores, weights=weights)) ** 2,
            weights=weights,
        ))
    else:
        x_std = np.std(X, axis=0)
        y_std = np.std(scores)

    safe_y_std = y_std if y_std > 0 else 1.0
    std_coeffs = coeffs * x_std / safe_y_std
    t_stats = np.where(std_errors > 0, coeffs / std_errors, 0.0)

    ranked = sorted(
        zip(feature_names, coeffs, std_coeffs, t_stats),
        key=lambda x: x[1],
        reverse=True,
    )

    reg_dict = {
        "r_squared": round(r_sq, 4),
        "intercept": round(intercept, 4),
        "score_std": round(float(safe_y_std), 6),
        "weighted": weights is not None,
        "coefficients": [
            {
                "feature": name,
                "label": _feature_label(name, 1),
                "coefficient": round(float(c), 4),
                "std_beta": round(float(sc), 4),
                "t_stat": round(float(t), 4),
            }
            for name, c, sc, t in ranked
        ],
    }
    return reg_dict, y_pred


# ── Synthesize recommendations ────────────────────────────────────────


# Mapping from compact labels to user-friendly descriptions and example cards
_EXAMPLE_CARDS: dict[str, dict[str, str]] = {
    "Draw1(mv1)": {
        "friendly": "1-mana cantrips",
        "description": "1-mana-draw-1 spell",
        "example": "Opt",
    },
    "Draw1(mv2)": {
        "friendly": "2-mana card selection",
        "description": "2-mana-draw-1 spell",
        "example": "Impulse",
    },
    "Draw2(mv2)": {
        "friendly": "2-mana draw",
        "description": "2-mana-draw-2 spell",
        "example": "Night's Whisper",
    },
    "Draw3(mv4)": {
        "friendly": "4-mana draw",
        "description": "4-mana-draw-3 spell",
        "example": "Concentrate",
    },
    "Draw1/t(mv3)": {
        "friendly": "repeatable draw",
        "description": "3-mana-draw-1-per-turn enchantment",
        "example": "Phyrexian Arena",
    },
    "Ramp+1(mv2)": {
        "friendly": "2-mana ramp",
        "description": "2-mana ramp spell",
        "example": "Arcane Signet",
    },
    "Ramp+1(mv3)": {
        "friendly": "3-mana ramp",
        "description": "3-mana ramp spell",
        "example": "Chromatic Lantern",
    },
    "Ramp+2(mv4)": {
        "friendly": "4-mana ramp",
        "description": "4-mana-ramp-2 spell",
        "example": "Explosive Vegetation",
    },
}


def _scryfall_url(card_name: str) -> str:
    """Build a Scryfall search URL for a card name."""
    from urllib.parse import quote
    return f"https://scryfall.com/search?q=!%22{quote(card_name)}%22"


def _scryfall_image_url(card_name: str) -> str:
    """Build a Scryfall image URL for hover preview."""
    from urllib.parse import quote
    return (
        f"https://api.scryfall.com/cards/named"
        f"?exact={quote(card_name)}&format=image&version=normal"
    )


def synthesize_recommendations(
    marginal: list[dict[str, Any]],
    regression: dict[str, Any],
    optimize_for: str,
) -> list[dict[str, Any]]:
    """Synthesize analysis into concrete, user-friendly recommendations.

    Uses regression coefficients as the primary signal (consistent with
    predict_top_configs which selects the ranking table configs) and
    supplements with marginal impact data for detail text.

    Returns list of {recommendation, impact, confidence, detail, label,
    example_card (optional)}.
    """
    metric_labels = {
        "mean_mana": "mana spent",
        "mean_mana_value": "mana spent on value",
        "mean_mana_total": "total mana spent",
        "floor_performance": "floor performance",
        "consistency": "consistency",
        "mean_spells_cast": "spells cast",
        "karsten_ecms": "Karsten-ECMS",
    }
    metric_label = metric_labels.get(optimize_for, optimize_for)

    recommendations: list[dict[str, Any]] = []

    # Index marginal impact by feature name for detail text
    marginal_by_feature: dict[str, list[dict[str, Any]]] = {}
    for m in marginal:
        marginal_by_feature.setdefault(m["feature"], []).append(m)

    # Build recommendations from regression coefficients
    for c in regression["coefficients"]:
        coeff = c["coefficient"]
        std_beta = c["std_beta"]
        fname = c["feature"]

        if abs(std_beta) < 0.01:
            continue

        impact_direction = "increase" if coeff > 0 else "decrease"
        example_card: dict[str, str] | None = None

        # Build label and recommendation text
        # Positive impact → "Add more X:", negative → "Don't add X:"
        is_positive = coeff > 0
        if fname == "land_delta":
            # Always frame land recommendations as a positive action
            if is_positive:
                label = "Add more lands"
                rec_text = (
                    f"Adding at least one more land tends to"
                    f" increase {metric_label}"
                )
            else:
                label = "Cut lands"
                rec_text = (
                    f"Cutting at least one land tends to"
                    f" increase {metric_label}"
                )
            # Flip impact to positive so it always appears under "improvements"
            coeff = abs(coeff)
        elif fname.startswith("count_"):
            card = fname.removeprefix("count_")
            card_info = _EXAMPLE_CARDS.get(card)
            if card_info:
                friendly = card_info["friendly"]
                description = card_info["description"]
                example_name = card_info["example"]
                label = (
                    f"Add more {friendly}" if is_positive
                    else f"Don't add {friendly}"
                )
                example_card = {
                    "name": example_name,
                    "url": _scryfall_url(example_name),
                    "image_url": _scryfall_image_url(example_name),
                }
                rec_text = (
                    f"Adding at least one more {description}"
                    f" (such as {example_name})"
                    f" tends to {impact_direction} {metric_label}"
                )
            else:
                label = (
                    f"Add more {card}" if is_positive
                    else f"Don't add {card}"
                )
                rec_text = (
                    f"Adding {card} tends to"
                    f" {impact_direction} {metric_label}"
                )
        else:
            label = fname
            rec_text = f"{fname} tends to {impact_direction} {metric_label}"

        # Confidence based on R² and agreement with marginal impact
        marginal_entries = marginal_by_feature.get(fname, [])
        marginal_agrees = True
        if marginal_entries:
            # Check if marginal impact direction agrees with regression
            positive_entries = [m for m in marginal_entries if m["value"] > 0]
            if positive_entries:
                avg_delta = sum(m["delta"] for m in positive_entries) / len(positive_entries)
                marginal_agrees = (avg_delta > 0) == (coeff > 0)

        # Per-feature confidence based on t-statistic (unit-invariant)
        abs_t = abs(c["t_stat"])
        if not marginal_agrees:
            confidence = "low"
        elif abs_t > 3.0:
            confidence = "high"
        elif abs_t > 1.5:
            confidence = "medium"
        else:
            confidence = "low"

        # Build detail text from marginal impact if available
        detail_parts = [f"Regression coefficient: {coeff:+.4f} (t={c['t_stat']:+.2f})"]
        for m in marginal_entries:
            if m["value"] != 0:
                detail_parts.append(
                    f"{m['label']}: avg {m['mean_with']:.3f} vs "
                    f"without: {m['mean_without']:.3f} "
                    f"(delta: {m['delta']:+.4f})"
                )

        rec_entry: dict[str, Any] = {
            "recommendation": rec_text,
            "impact": round(coeff, 4),
            "confidence": confidence,
            "label": label,
            "detail": "; ".join(detail_parts),
        }
        if example_card is not None:
            rec_entry["example_card"] = example_card
        recommendations.append(rec_entry)

    # Sort by absolute impact descending (positive first, then negative)
    recommendations.sort(key=lambda x: -x["impact"])

    return recommendations


# ── Predict top configs ───────────────────────────────────────────────


def predict_top_configs(
    all_round_scores: List[Tuple[DeckConfig, float, int]],
    all_configs: List[DeckConfig],
    top_k: int = 5,
) -> tuple[List[DeckConfig], dict[str, Any]]:
    """Use regression on Hyperband data to predict the best configs.

    Fits a WLS regression model on the aggregated Hyperband round scores,
    then predicts scores for ALL enumerated configs and returns the top-k
    by predicted score.

    Args:
        all_round_scores: List of (config, score, n_sims) from Hyperband.
        all_configs: Full list of enumerated configs to predict over.
        top_k: Number of top configs to return.

    Returns:
        (top_configs, regression_info) where regression_info contains
        the fitted coefficients and intercept needed for prediction.
    """
    if len(all_round_scores) < 3:
        return all_configs[:top_k], {}

    configs, scores, weights = aggregate_hyperband_scores(all_round_scores)
    X_train, feature_names, _ = configs_to_feature_matrix(configs)

    # Fit WLS on Hyperband data
    coeffs, intercept, r_sq, _std_errors = fit_ols(X_train, scores, weights)

    # Predict scores for ALL configs
    all_feature_dicts = [extract_features(c) for c in all_configs]
    X_all = np.array(
        [[fd[name] for name in feature_names] for fd in all_feature_dicts],
        dtype=float,
    )
    predicted = X_all @ coeffs + intercept

    # Select top-k by predicted score
    top_indices = np.argsort(predicted)[::-1][:top_k]
    top_configs = [all_configs[i] for i in top_indices]

    reg_info = {
        "coefficients": coeffs,
        "intercept": intercept,
        "feature_names": feature_names,
        "r_squared": r_sq,
    }

    return top_configs, reg_info


# ── Main entry point ──────────────────────────────────────────────────


def analyze_optimization(
    all_round_scores: List[Tuple[DeckConfig, float, int]],
    optimize_for: str,
) -> dict[str, Any]:
    """Run feature analysis on Hyperband round scores.

    Args:
        all_round_scores: List of (config, score, n_sims) from optimizer.
        optimize_for: The optimization target metric name.

    Returns dict with keys:
        - recommendations: list of synthesized recommendations
        - marginal_impact: raw marginal impact data
        - regression: regression analysis results
        - n_configs: number of unique configs analyzed
    """
    if not all_round_scores:
        return {}

    configs, scores, weights = aggregate_hyperband_scores(all_round_scores)

    if len(configs) < 3:
        return {}

    X, feature_names, feature_dicts = configs_to_feature_matrix(configs)

    # Regression analysis
    reg_dict, y_pred = regression_analysis(X, scores, feature_names, weights)

    # Marginal impact
    marginal = compute_marginal_impact(
        feature_dicts, scores, feature_names,
        weights=weights, score_std=reg_dict.get("score_std"),
    )

    # Synthesize recommendations
    recommendations = synthesize_recommendations(marginal, reg_dict, optimize_for)

    return {
        "recommendations": recommendations,
        "marginal_impact": marginal,
        "regression": reg_dict,
        "n_configs": len(configs),
    }


# ── Factored optimizer recommendations ───────────────────────────────


def synthesize_factored_recommendations(
    marginal_results: list[MarginalResult],
    optimize_for: str,
) -> list[dict[str, Any]]:
    """Convert factored marginal results into user-friendly recommendations.

    Each marginal result with a meaningful effect becomes a recommendation.
    Uses measured paired effects directly (no regression needed).
    """
    metric_labels = {
        "mean_mana": "mana spent",
        "mean_mana_value": "mana spent on value",
        "mean_mana_total": "total mana spent",
        "floor_performance": "floor performance",
        "consistency": "consistency",
        "mean_spells_cast": "spells cast",
    }
    metric_label = metric_labels.get(optimize_for, optimize_for)

    recommendations: list[dict[str, Any]] = []

    for mr in marginal_results:
        if mr.negligible:
            continue

        config = mr.config
        effect = mr.effect_size
        is_positive = effect > 0
        impact_direction = "increase" if is_positive else "decrease"
        example_card: dict[str, str] | None = None

        # Build label and recommendation text
        if mr.dimension == "land":
            delta = config.land_delta
            if delta > 0:
                if is_positive:
                    label = "Add more lands"
                    rec_text = (
                        f"Adding {delta} land(s) improves {metric_label}"
                        f" by {abs(effect):.2f}"
                    )
                else:
                    label = "Don't add lands"
                    rec_text = (
                        f"Adding {delta} land(s) decreases {metric_label}"
                        f" by {abs(effect):.2f}"
                    )
            else:
                if is_positive:
                    label = "Cut lands"
                    rec_text = (
                        f"Cutting {abs(delta)} land(s) improves {metric_label}"
                        f" by {abs(effect):.2f}"
                    )
                else:
                    label = "Don't cut lands"
                    rec_text = (
                        f"Cutting {abs(delta)} land(s) decreases {metric_label}"
                        f" by {abs(effect):.2f}"
                    )
        elif config.added_cards:
            card_id = config.added_cards[0]
            candidate = ALL_CANDIDATES.get(card_id)
            compact = candidate.compact_label if candidate else card_id
            card_info = _EXAMPLE_CARDS.get(compact)

            if card_info:
                friendly = card_info["friendly"]
                description = card_info["description"]
                example_name = card_info["example"]
                label = (
                    f"Add {friendly}" if is_positive
                    else f"Don't add {friendly}"
                )
                example_card = {
                    "name": example_name,
                    "url": _scryfall_url(example_name),
                    "image_url": _scryfall_image_url(example_name),
                }
                rec_text = (
                    f"Adding a {description} (e.g. {example_name})"
                    f" {impact_direction}s {metric_label} by {abs(effect):.2f}"
                )
            else:
                label = (
                    f"Add {compact}" if is_positive
                    else f"Don't add {compact}"
                )
                rec_text = (
                    f"Adding {compact} {impact_direction}s"
                    f" {metric_label} by {abs(effect):.2f}"
                )
        else:
            continue

        confidence = "high" if mr.p_value < 0.01 else (
            "medium" if mr.p_value < 0.05 else "low"
        )

        detail = (
            f"Effect: {effect:+.3f} +/- {mr.se:.3f}"
            f" (p={mr.p_value:.4f}, n={mr.n_games} games)"
        )

        rec_entry: dict[str, Any] = {
            "recommendation": rec_text,
            "impact": round(abs(effect), 4),
            "confidence": confidence,
            "label": label,
            "detail": detail,
        }
        if example_card is not None:
            rec_entry["example_card"] = example_card
        recommendations.append(rec_entry)

    recommendations.sort(key=lambda x: -x["impact"])
    return recommendations
