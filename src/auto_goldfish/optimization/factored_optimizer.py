"""Factored optimizer for deck configuration.

Evaluates each modification dimension independently (land count, draw cards,
ramp cards) using CRN-paired adaptive sampling, then tests promising
combinations. Much simpler and faster than Hyperband or racing for the
small, structured search space of deck modifications.

Typical budget: ~20-30 simulations instead of hundreds.
"""

from __future__ import annotations

import random as _stdlib_random
from dataclasses import dataclass
from math import erfc, sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from auto_goldfish.optimization.candidate_cards import CandidateCard
from auto_goldfish.optimization.deck_config import DeckConfig, apply_config


@dataclass
class MarginalResult:
    """Result of evaluating a single marginal change vs baseline."""

    config: DeckConfig
    dimension: str  # "land", "draw", or "ramp"
    effect_size: float
    se: float
    p_value: float
    n_games: int
    significant: bool  # True if effect is statistically significant
    negligible: bool  # True if effect is clearly near zero


def _paired_p_value(diffs: np.ndarray) -> float:
    """Two-sided p-value for paired differences using normal approximation."""
    n = len(diffs)
    if n == 0:
        return 1.0
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    if se < 1e-15:
        return 0.0 if abs(mean) > 1e-15 else 1.0
    z = abs(mean) / se
    return float(erfc(z / sqrt(2)))


def _classify_config(config: DeckConfig) -> str:
    """Classify a single-change config into its dimension."""
    if config.added_cards:
        from auto_goldfish.optimization.candidate_cards import ALL_CANDIDATES

        card_id = config.added_cards[0]
        candidate = ALL_CANDIDATES.get(card_id)
        if candidate and candidate.card_type == "draw":
            return "draw"
        return "ramp"
    return "land"


class FactoredOptimizer:
    """Factored optimizer with adaptive sampling.

    Evaluates each deck modification dimension independently, then tests
    promising combinations, using CRN pairing for low-variance comparisons
    and adaptive sample sizing to avoid wasting compute.

    Args:
        goldfisher: Goldfisher instance (will be mutated during optimization).
        candidates: Dict of candidate_id -> CandidateCard to consider.
        swap_mode: If True, remove no-effect spells to maintain deck size.
        max_draw: Maximum number of draw candidates to add (0-2).
        max_ramp: Maximum number of ramp candidates to add (0-2).
        land_range: Land delta range (-land_range to +land_range).
        land_delta_min: Explicit lower bound for land delta.
        land_delta_max: Explicit upper bound for land delta.
        optimize_for: Target metric name.
        base_games: Initial games per adaptive evaluation.
        max_games: Maximum games before stopping adaptive sampling.
        significance_threshold: z-multiplier for significance (effect > threshold * SE).
        negligible_threshold: z-multiplier for negligible (effect < threshold * SE).
        hyperband_max_sims: Accepted for API compatibility, ignored.
    """

    def __init__(
        self,
        goldfisher,
        candidates: Dict[str, CandidateCard],
        swap_mode: bool = False,
        max_draw: int = 2,
        max_ramp: int = 2,
        land_range: int = 2,
        land_delta_min: Optional[int] = None,
        land_delta_max: Optional[int] = None,
        optimize_for: str = "floor_performance",
        base_games: int = 200,
        max_games: int = 800,
        significance_threshold: float = 2.0,
        negligible_threshold: float = 0.5,
        hyperband_max_sims: Optional[int] = None,
    ) -> None:
        self.goldfisher = goldfisher
        self.candidates = candidates
        self.swap_mode = swap_mode
        self.max_draw = max_draw
        self.max_ramp = max_ramp
        self.land_range = land_range
        self.land_delta_min = land_delta_min
        self.land_delta_max = land_delta_max
        self.optimize_for = optimize_for
        self.base_games = base_games
        self.max_games = max_games
        self.significance_threshold = significance_threshold
        self.negligible_threshold = negligible_threshold

        # Populated during run()
        self.marginal_results: List[MarginalResult] = []
        self.all_round_scores: List[Tuple[DeckConfig, float, int]] = []
        self.feature_analysis: Optional[dict] = None

    def run(
        self,
        final_sims: int = 1000,
        final_top_k: int = 5,
        include_hyperband: bool = False,
        enum_progress: Optional[Callable[[int, int], None]] = None,
        eval_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[DeckConfig, Any]]:
        """Run optimization and return ranked (config, result_dict) pairs.

        Phase 1: Evaluate each individual change vs baseline with adaptive sampling.
        Phase 2: Test combinations of the best changes from each dimension.
        Phase 3: Full evaluation of top configs.
        """
        original_sims = self.goldfisher.sims
        base_seed = (
            self.goldfisher.seed
            if self.goldfisher.seed is not None
            else _stdlib_random.randrange(2**31)
        )

        # Build marginal configs
        marginal_configs = self._enumerate_marginals()

        # Estimate total budget for progress
        total_budget_est = (len(marginal_configs) + 1) * self.max_games
        done_sims = [0]

        def report_enum(sims: int) -> None:
            done_sims[0] += sims
            if enum_progress is not None:
                enum_progress(done_sims[0], total_budget_est)

        # Phase 1: Baseline + marginal evaluation with adaptive sampling
        baseline_values: list[float] = []
        self.marginal_results = []

        for config in marginal_configs:
            result = self._adaptive_evaluate(
                config, base_seed, baseline_values, report_enum,
            )
            self.marginal_results.append(result)

        # Record scores for all marginals (for compatibility with feature_analysis)
        self.all_round_scores = []
        baseline_score = self._compute_score(np.array(baseline_values))
        self.all_round_scores.append(
            (DeckConfig(), baseline_score, len(baseline_values))
        )
        for mr in self.marginal_results:
            score = baseline_score + mr.effect_size
            self.all_round_scores.append((mr.config, score, mr.n_games))

        # Phase 2: Test combinations of best changes per dimension
        combo_configs = self._build_combinations()
        combo_results: list[MarginalResult] = []
        for config in combo_configs:
            result = self._adaptive_evaluate(
                config, base_seed, baseline_values, report_enum,
            )
            combo_results.append(result)
            score = baseline_score + result.effect_size
            self.all_round_scores.append((config, score, result.n_games))

        # Report final enum progress
        if enum_progress is not None:
            enum_progress(done_sims[0], done_sims[0])

        # Build feature analysis from marginal results
        self.feature_analysis = self._build_feature_analysis()

        # Phase 3: Full evaluation of top configs
        all_candidates = self.marginal_results + combo_results
        all_candidates.sort(key=lambda m: m.effect_size, reverse=True)

        # Select top configs for final evaluation
        eval_configs: list[DeckConfig] = []
        seen: set[DeckConfig] = set()
        for mr in all_candidates:
            if mr.config not in seen and len(eval_configs) < final_top_k:
                eval_configs.append(mr.config)
                seen.add(mr.config)

        baseline = DeckConfig()
        if baseline not in seen:
            eval_configs.append(baseline)
            seen.add(baseline)

        # Full simulation on each
        self.goldfisher.sims = final_sims
        from auto_goldfish.metrics.reporter import result_to_dict

        results: list[tuple[DeckConfig, Any]] = []
        for j, config in enumerate(eval_configs):
            apply_config(self.goldfisher, config, self.candidates, self.swap_mode)
            result = self.goldfisher.simulate()
            result_dict = result_to_dict(result)
            results.append((config, result_dict))
            if eval_progress is not None:
                eval_progress(j + 1, len(eval_configs))

        self.goldfisher.sims = original_sims

        # Sort by target metric
        results.sort(
            key=lambda r: self._extract_score_from_dict(r[1]), reverse=True
        )

        # Handle baseline rank
        baseline_rank = None
        baseline_entry = None
        for rank, (cfg, rd) in enumerate(results, 1):
            if cfg == baseline:
                baseline_rank = rank
                baseline_entry = (cfg, rd)
                break

        results = results[:final_top_k]

        baseline_in_top = any(cfg == baseline for cfg, _ in results)
        if not baseline_in_top and baseline_entry is not None:
            baseline_entry[1]["opt_baseline_rank"] = baseline_rank
            results.append(baseline_entry)

        # Attach feature analysis to top result
        if self.feature_analysis and results:
            results[0][1]["feature_analysis"] = self.feature_analysis

        return results

    # -- Marginal enumeration --

    def _enumerate_marginals(self) -> list[DeckConfig]:
        """Enumerate individual single-change configs to test."""
        configs: list[DeckConfig] = []

        lo = (
            self.land_delta_min
            if self.land_delta_min is not None
            else -self.land_range
        )
        hi = (
            self.land_delta_max
            if self.land_delta_max is not None
            else self.land_range
        )

        for delta in range(lo, hi + 1):
            if delta == 0:
                continue
            configs.append(DeckConfig(land_delta=delta))

        for cid, candidate in self.candidates.items():
            if candidate.card_type == "draw" and self.max_draw > 0:
                configs.append(DeckConfig(added_cards=(cid,)))
            elif candidate.card_type == "ramp" and self.max_ramp > 0:
                configs.append(DeckConfig(added_cards=(cid,)))

        return configs

    # -- Adaptive evaluation --

    def _adaptive_evaluate(
        self,
        config: DeckConfig,
        base_seed: int,
        baseline_values: list[float],
        report: Callable[[int], None],
    ) -> MarginalResult:
        """Evaluate a config vs baseline with adaptive sample sizing.

        Uses CRN pairing (same seeds). Extends baseline_values in-place
        as needed so the baseline grows to match the largest evaluation.
        """
        n_games = self.base_games
        dimension = _classify_config(config)

        while n_games <= self.max_games:
            # Ensure baseline has enough values
            self._extend_baseline(baseline_values, base_seed, n_games, report)

            # Evaluate config on same seeds
            config_values = self._evaluate_on_seeds(
                config, base_seed, n_games, report,
            )

            bl_arr = np.array(baseline_values[:n_games])
            cfg_arr = np.array(config_values)

            # Compute paired statistics
            if self.optimize_for == "consistency":
                effect, se = self._bootstrap_paired_consistency(cfg_arr, bl_arr)
            else:
                diffs = cfg_arr - bl_arr
                effect = float(diffs.mean())
                se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))

            p_value = _paired_p_value(cfg_arr - bl_arr)

            # Check stopping criteria
            if se < 1e-15:
                # Zero variance — effect is exact
                return MarginalResult(
                    config=config, dimension=dimension,
                    effect_size=effect, se=se, p_value=p_value,
                    n_games=n_games, significant=abs(effect) > 1e-10,
                    negligible=abs(effect) <= 1e-10,
                )

            if abs(effect) > self.significance_threshold * se:
                return MarginalResult(
                    config=config, dimension=dimension,
                    effect_size=effect, se=se, p_value=p_value,
                    n_games=n_games, significant=True, negligible=False,
                )

            if abs(effect) < self.negligible_threshold * se:
                return MarginalResult(
                    config=config, dimension=dimension,
                    effect_size=effect, se=se, p_value=p_value,
                    n_games=n_games, significant=False, negligible=True,
                )

            # Ambiguous — double and try again
            if n_games >= self.max_games:
                break
            n_games = min(n_games * 2, self.max_games)

        # Reached max games without clear conclusion
        return MarginalResult(
            config=config, dimension=dimension,
            effect_size=effect, se=se, p_value=p_value,
            n_games=n_games, significant=abs(effect) > self.significance_threshold * se,
            negligible=abs(effect) < self.negligible_threshold * se,
        )

    def _extend_baseline(
        self,
        baseline_values: list[float],
        base_seed: int,
        target_n: int,
        report: Callable[[int], None],
    ) -> None:
        """Extend baseline values to at least target_n games."""
        if len(baseline_values) >= target_n:
            return

        baseline_config = DeckConfig()
        apply_config(
            self.goldfisher, baseline_config, self.candidates, self.swap_mode,
        )

        start = len(baseline_values)
        for j in range(start, target_n):
            seed = base_seed + j
            val = self.goldfisher.simulate_single_game(seed)
            baseline_values.append(val)

        report(target_n - start)

    def _evaluate_on_seeds(
        self,
        config: DeckConfig,
        base_seed: int,
        n_games: int,
        report: Callable[[int], None],
    ) -> list[float]:
        """Evaluate a config on seeds [base_seed, base_seed+n_games)."""
        apply_config(self.goldfisher, config, self.candidates, self.swap_mode)

        values: list[float] = []
        for j in range(n_games):
            seed = base_seed + j
            val = self.goldfisher.simulate_single_game(seed)
            values.append(val)

        report(n_games)
        return values

    def _bootstrap_paired_consistency(
        self,
        config_values: np.ndarray,
        baseline_values: np.ndarray,
        n_bootstrap: int = 200,
    ) -> tuple[float, float]:
        """Compute paired consistency difference via bootstrap.

        Returns (effect_size, standard_error).
        """
        n = len(config_values)
        rng = np.random.RandomState(42)

        boot_idx = rng.randint(0, n, size=(n_bootstrap, n))

        cfg_boot = config_values[boot_idx]
        bl_boot = baseline_values[boot_idx]

        def _consistency(arr: np.ndarray) -> np.ndarray:
            """Consistency for each bootstrap row."""
            sorted_arr = np.sort(arr, axis=1)
            cutoff = max(1, int(n * 0.25))
            tail_means = sorted_arr[:, :cutoff].mean(axis=1)
            overall_means = arr.mean(axis=1)
            safe_means = np.maximum(overall_means, 1e-10)
            return tail_means / safe_means

        cfg_con = _consistency(cfg_boot)
        bl_con = _consistency(bl_boot)
        diffs = cfg_con - bl_con

        return float(diffs.mean()), float(diffs.std(ddof=1))

    # -- Combinations --

    def _build_combinations(self) -> list[DeckConfig]:
        """Build combination configs from the best marginal per dimension."""
        best_per_dim: dict[str, MarginalResult] = {}
        for mr in self.marginal_results:
            if mr.effect_size <= 0:
                continue
            prev = best_per_dim.get(mr.dimension)
            if prev is None or mr.effect_size > prev.effect_size:
                best_per_dim[mr.dimension] = mr

        if len(best_per_dim) < 2:
            # Not enough positive dimensions for meaningful combinations
            # Still test 2-copy if available
            combos: list[DeckConfig] = []
            for dim, mr in best_per_dim.items():
                if dim in ("draw", "ramp") and mr.config.added_cards:
                    max_copies = (
                        self.max_draw if dim == "draw" else self.max_ramp
                    )
                    if max_copies >= 2:
                        cid = mr.config.added_cards[0]
                        combos.append(
                            DeckConfig(
                                land_delta=mr.config.land_delta,
                                added_cards=(cid, cid),
                            )
                        )
            return combos

        dims = list(best_per_dim.keys())
        combos: list[DeckConfig] = []
        seen: set[DeckConfig] = set()

        def _merge(*mrs: MarginalResult) -> DeckConfig:
            land_delta = 0
            cards: list[str] = []
            for m in mrs:
                land_delta += m.config.land_delta
                cards.extend(m.config.added_cards)
            return DeckConfig(
                land_delta=land_delta,
                added_cards=tuple(sorted(cards)),
            )

        # Pairwise combinations
        for i in range(len(dims)):
            for j in range(i + 1, len(dims)):
                cfg = _merge(best_per_dim[dims[i]], best_per_dim[dims[j]])
                if cfg not in seen:
                    combos.append(cfg)
                    seen.add(cfg)

        # All three together
        if len(dims) >= 3:
            cfg = _merge(*[best_per_dim[d] for d in dims])
            if cfg not in seen:
                combos.append(cfg)
                seen.add(cfg)

        # 2-copy variants for draw/ramp
        for dim in ("draw", "ramp"):
            mr = best_per_dim.get(dim)
            if mr and mr.config.added_cards:
                max_copies = self.max_draw if dim == "draw" else self.max_ramp
                if max_copies >= 2:
                    cid = mr.config.added_cards[0]
                    cfg = DeckConfig(
                        land_delta=0,
                        added_cards=(cid, cid),
                    )
                    if cfg not in seen:
                        combos.append(cfg)
                        seen.add(cfg)

        return combos

    # -- Feature analysis --

    def _build_feature_analysis(self) -> dict[str, Any]:
        """Build feature analysis dict from marginal results."""
        from auto_goldfish.optimization.feature_analysis import (
            synthesize_factored_recommendations,
        )

        recommendations = synthesize_factored_recommendations(
            self.marginal_results, self.optimize_for,
        )

        marginal_impact = []
        for mr in self.marginal_results:
            marginal_impact.append({
                "config": mr.config.describe(),
                "dimension": mr.dimension,
                "effect_size": round(mr.effect_size, 4),
                "se": round(mr.se, 4),
                "p_value": round(mr.p_value, 4),
                "n_games": mr.n_games,
                "significant": mr.significant,
                "negligible": mr.negligible,
            })

        return {
            "recommendations": recommendations,
            "marginal_impact": marginal_impact,
            "n_configs": len(self.marginal_results),
        }

    # -- Scoring --

    def _compute_score(self, mana_values: np.ndarray) -> float:
        """Compute the target metric from raw per-game mana values."""
        if len(mana_values) == 0:
            return 0.0
        if self.optimize_for == "consistency":
            return self._compute_consistency(mana_values)
        return float(mana_values.mean())

    @staticmethod
    def _compute_consistency(
        mana_values: np.ndarray, threshold: float = 0.25,
    ) -> float:
        """Left-tail ratio from raw per-game mana array."""
        n = len(mana_values)
        if n == 0:
            return 1.0
        sorted_vals = np.sort(mana_values)
        cutoff = max(1, int(n * threshold))
        tail_mean = float(sorted_vals[:cutoff].mean())
        overall_mean = float(mana_values.mean())
        if overall_mean == 0:
            return 1.0
        return tail_mean / overall_mean

    def _extract_score_from_dict(self, result_dict: dict) -> float:
        """Extract score from a result_to_dict output."""
        if self.optimize_for == "floor_performance":
            return result_dict.get("threshold_mana", 0.0)
        if self.optimize_for == "consistency":
            return result_dict.get("consistency", 0.0)
        if self.optimize_for == "mean_mana_value":
            return result_dict.get("mean_mana_value", 0.0)
        if self.optimize_for == "mean_mana_total":
            return result_dict.get("mean_mana_total", 0.0)
        if self.optimize_for == "mean_spells_cast":
            return result_dict.get("mean_spells_cast", 0.0)
        return result_dict.get("mean_mana", 0.0)
