"""Hyperband optimizer for deck configuration.

Uses the Hyperband algorithm (Li et al., 2018) to efficiently identify
the best deck configurations from the full enumerated search space.
Multiple brackets of successive halving with different aggressiveness
levels hedge against the explore-exploit tradeoff, allocating more
simulation budget to promising configurations while cheaply eliminating
obvious underperformers.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from auto_goldfish.optimization.candidate_cards import CandidateCard
from auto_goldfish.optimization.deck_config import DeckConfig, apply_config, enumerate_configs


class DeckOptimizer:
    """Hyperband optimizer for deck configuration.

    Enumerates all valid configurations within the given bounds,
    then uses Hyperband (multi-bracket successive halving) to
    efficiently identify the top performers before a final
    full-accuracy evaluation pass.

    Hyperband runs multiple brackets of successive halving:
    - The most aggressive bracket starts all configs with few sims
      and eliminates aggressively
    - Less aggressive brackets start fewer configs with more sims
    - This hedges against the tradeoff between breadth and depth

    Args:
        goldfisher: Goldfisher instance (will be mutated during optimization).
        candidates: Dict of candidate_id -> CandidateCard to consider.
        swap_mode: If True, remove no-effect spells to maintain deck size.
        max_draw: Maximum number of draw candidates to add (0-2).
        max_ramp: Maximum number of ramp candidates to add (0-2).
        land_range: Land delta range (-land_range to +land_range). Used as
            fallback when land_delta_min/land_delta_max are not specified.
        land_delta_min: Minimum land delta (e.g. -3). Overrides -land_range.
        land_delta_max: Maximum land delta (e.g. +2). Overrides +land_range.
        optimize_for: Target metric - "mean_mana", "consistency", or "mean_spells_cast".
        hyperband_max_sims: Max simulations per config during Hyperband
            enumeration (the R parameter). Higher values give more accurate
            rankings but take longer.
        eta: Halving rate for successive halving. Lower values (e.g. 2)
            keep more candidates per round (less aggressive pruning).
            Higher values (e.g. 4-5) prune harder. Default: 3.
        hyperband_min_sims: Minimum simulations per evaluation in any
            Hyperband round. Higher values give more reliable early-round
            filtering at the cost of fewer brackets. Default: 20.
        hyperband_top_k: Number of survivors from Hyperband filtering before
            final evaluation. If None, uses final_top_k from run().
            Over-selecting (e.g. 10-15) and re-ranking with final_sims
            can improve quality at modest extra cost.
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
        hyperband_max_sims: int = 500,
        eta: int = 3,
        hyperband_min_sims: int = 20,
        hyperband_top_k: Optional[int] = None,
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
        self.hyperband_max_sims = hyperband_max_sims
        self.ETA = eta
        self.HYPERBAND_MIN_SIMS = hyperband_min_sims
        self.hyperband_top_k = hyperband_top_k

        # Populated during run(): every (config, score, n_sims) from all
        # Hyperband rounds across all brackets.
        self.all_round_scores: List[Tuple[DeckConfig, float, int]] = []

    def run(
        self,
        final_sims: int = 1000,
        final_top_k: int = 5,
        include_hyperband: bool = False,
        enum_progress: Optional[Callable[[int, int], None]] = None,
        eval_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[DeckConfig, Any]]:
        """Run optimization and return ranked (config, result_dict) pairs.

        Phase 1 (Hyperband): Explore the config space using multi-bracket
        successive halving, collecting scores across all rounds.

        Phase 2 (Regression selection): Fit a WLS regression model on
        Hyperband data and predict the best configs from the full space.

        Phase 3 (Final evaluation): Run full simulations on the combined
        pool of regression-predicted configs, top hyperband survivors, and
        the no-changes baseline. Returns the top ``final_top_k`` results.

        The first result dict includes a ``feature_analysis`` key with
        recommendations, marginal impact, and regression details.

        Args:
            final_sims: Simulations for final evaluation of top configs.
            final_top_k: Number of top configs to fully evaluate.
            enum_progress: Optional callable(current, total) for enumeration.
                Current/total are measured in total simulations run.
            eval_progress: Optional callable(current, total) for evaluation.

        Returns:
            List of (DeckConfig, result_dict) sorted best-first.
        """
        configs = enumerate_configs(
            self.candidates,
            max_draw=self.max_draw,
            max_ramp=self.max_ramp,
            land_range=self.land_range,
            land_delta_min=self.land_delta_min,
            land_delta_max=self.land_delta_max,
        )

        original_sims = self.goldfisher.sims

        # Phase 1: Hyperband exploration (collects all_round_scores)
        self.all_round_scores = []
        hb_top_k = self.hyperband_top_k if self.hyperband_top_k is not None else final_top_k
        hyperband_survivors = self._hyperband_select(configs, hb_top_k, enum_progress)

        # Phase 2: Use regression to pick the best configs from the
        # full config space, based on Hyperband round data
        from auto_goldfish.optimization.feature_analysis import (
            analyze_optimization,
            predict_top_configs,
        )

        regression_configs, _ = predict_top_configs(
            self.all_round_scores, configs, top_k=final_top_k,
        )

        # Run feature analysis for UI display
        self.feature_analysis = analyze_optimization(
            self.all_round_scores, self.optimize_for,
        )

        # Build combined evaluation pool:
        # - regression-predicted configs (tagged "regression")
        # - top 4 hyperband survivors (tagged "hyperband") — only if include_hyperband
        # - no-changes baseline (tagged "baseline")
        # Source tags are only set when include_hyperband is True.
        baseline = DeckConfig()
        eval_pool: list[tuple[DeckConfig, str | None]] = []
        seen: set[DeckConfig] = set()

        tag_regression = "regression" if include_hyperband else None
        tag_hyperband = "hyperband" if include_hyperband else None
        tag_baseline = "baseline" if include_hyperband else None

        for cfg in regression_configs:
            if cfg not in seen:
                eval_pool.append((cfg, tag_regression))
                seen.add(cfg)

        if include_hyperband:
            for cfg in hyperband_survivors[:4]:
                if cfg not in seen:
                    eval_pool.append((cfg, tag_hyperband))
                    seen.add(cfg)

        if baseline not in seen:
            eval_pool.append((baseline, tag_baseline))
            seen.add(baseline)

        # Phase 3: Full evaluation of combined pool
        self.goldfisher.sims = final_sims
        from auto_goldfish.metrics.reporter import result_to_dict

        results: list[tuple[DeckConfig, Any]] = []
        total_eval_sims = len(eval_pool) * final_sims
        for j, (config, source) in enumerate(eval_pool):
            apply_config(self.goldfisher, config, self.candidates, self.swap_mode)

            # Sub-progress: report per-sim within each config evaluation
            sub_cb = None
            if eval_progress is not None:
                offset = j * final_sims

                def sub_cb(current: int, total: int, _offset: int = offset) -> None:
                    eval_progress(_offset + current, total_eval_sims)

            result = self.goldfisher.simulate(progress_callback=sub_cb)
            result_dict = result_to_dict(result)
            if source is not None:
                result_dict["opt_source"] = source
            results.append((config, result_dict))

        self.goldfisher.sims = original_sims

        # Sort final results by target metric
        results.sort(key=lambda r: self._extract_score_from_dict(r[1]), reverse=True)

        # Find baseline rank (1-indexed) before cutting
        baseline_rank = None
        baseline_entry = None
        for rank, (cfg, rd) in enumerate(results, 1):
            if cfg == baseline:
                baseline_rank = rank
                baseline_entry = (cfg, rd)
                break

        # Keep top final_top_k
        results = results[:final_top_k]

        # If baseline didn't make the cut, append it as a reference row
        baseline_in_top = any(cfg == baseline for cfg, _ in results)
        if not baseline_in_top and baseline_entry is not None:
            baseline_entry[1]["opt_baseline_rank"] = baseline_rank
            results.append(baseline_entry)

        # Attach feature analysis to the top-ranked result (after sorting)
        if self.feature_analysis and results:
            results[0][1]["feature_analysis"] = self.feature_analysis

        return results

    # -- Hyperband internals --

    def _hyperband_select(
        self,
        configs: List[DeckConfig],
        top_k: int,
        progress: Optional[Callable[[int, int], None]],
    ) -> List[DeckConfig]:
        """Select top-K configs using Hyperband multi-bracket successive halving."""
        eta = self.ETA
        R = self.hyperband_max_sims
        min_sims = max(self.HYPERBAND_MIN_SIMS, R // 10)

        # s_max determines number of brackets; capped so initial sims >= min_sims
        s_max = max(0, int(math.floor(
            math.log(max(R / min_sims, 1)) / math.log(eta)
        )))

        n_max = len(configs)

        # Degenerate case: fewer configs than requested
        if n_max <= top_k:
            return list(configs)

        # Pre-compute bracket plans for budget estimation
        brackets = self._plan_brackets(n_max, R, eta, s_max, min_sims, top_k)
        total_budget = sum(
            n_i * r_i for bracket in brackets for n_i, r_i in bracket
        )

        done = [0]  # mutable for closure
        original_sims = self.goldfisher.sims
        rng = random.Random(
            self.goldfisher.seed if self.goldfisher.seed is not None else 42
        )

        def report(sims: int) -> None:
            done[0] += sims
            if progress is not None:
                progress(done[0], total_budget)

        # config -> (best_score, sims_used_for_that_score)
        all_survivors: dict[DeckConfig, tuple[float, int]] = {}

        for s_idx, bracket_plan in enumerate(brackets):
            s = s_max - s_idx

            # Most aggressive bracket uses all configs;
            # less aggressive brackets sample a subset
            if s == s_max:
                bracket_configs = list(configs)
            else:
                bracket_n = min(
                    max(math.ceil(n_max / eta ** (s_max - s)), top_k),
                    n_max,
                )
                bracket_configs = rng.sample(configs, bracket_n)

            survivors = self._successive_halving(bracket_configs, bracket_plan, report)

            # Record survivors with their final-round sim count
            r_final = bracket_plan[-1][1]
            for cfg, score in survivors:
                prev = all_survivors.get(cfg)
                # Prefer scores from higher sim counts (more accurate)
                if prev is None or r_final > prev[1] or (
                    r_final == prev[1] and score > prev[0]
                ):
                    all_survivors[cfg] = (score, r_final)

        self.goldfisher.sims = original_sims

        # Rank by (sims_used desc, score desc) to prefer high-confidence estimates
        ranked = sorted(
            all_survivors.items(),
            key=lambda x: (x[1][1], x[1][0]),
            reverse=True,
        )
        return [cfg for cfg, _ in ranked[:top_k]]

    def _plan_brackets(
        self,
        n_max: int,
        R: int,
        eta: int,
        s_max: int,
        min_sims: int,
        top_k: int,
    ) -> list[list[tuple[int, int]]]:
        """Pre-compute (num_configs, sims_per_config) for each round of each bracket.

        Must exactly match the logic in _successive_halving so the budget
        estimate is accurate for progress reporting.
        """
        brackets: list[list[tuple[int, int]]] = []

        for s in range(s_max, -1, -1):
            if s == s_max:
                n = n_max
            else:
                n = min(max(math.ceil(n_max / eta ** (s_max - s)), top_k), n_max)

            r_0 = max(math.ceil(R / eta ** s), min_sims)
            rounds: list[tuple[int, int]] = []
            n_i = n

            for i in range(s + 1):
                r_i = min(round(r_0 * eta ** i), R)
                rounds.append((n_i, r_i))
                # Configs kept for next round (matches _successive_halving)
                n_i = max(int(n_i / eta), 1)

            brackets.append(rounds)

        return brackets

    def _successive_halving(
        self,
        configs: list[DeckConfig],
        plan: list[tuple[int, int]],
        report: Callable[[int], None],
    ) -> list[tuple[DeckConfig, float]]:
        """Run one bracket of successive halving.

        Args:
            configs: Starting configurations for this bracket.
            plan: List of (expected_n, sims_per_config) per round.
            report: Callback to report sims completed for progress tracking.

        Returns:
            List of (config, score) for the final survivors.
        """
        eta = self.ETA
        current = list(configs)

        for round_idx, (_expected_n, r_i) in enumerate(plan):
            self.goldfisher.sims = r_i

            scored: list[tuple[DeckConfig, float]] = []
            for config in current:
                score = self._evaluate(config)
                scored.append((config, score))
                self.all_round_scores.append((config, score, r_i))
                report(r_i)

            scored.sort(key=lambda x: x[1], reverse=True)

            # Keep top 1/eta fraction for next round (or as final survivors)
            keep = max(int(len(scored) / eta), 1)
            if round_idx < len(plan) - 1:
                current = [cfg for cfg, _ in scored[:keep]]
            else:
                scored = scored[:keep]

        return scored

    # -- Scoring --

    def _evaluate(self, config: DeckConfig) -> float:
        """Run a quick simulation and return the target metric score."""
        apply_config(self.goldfisher, config, self.candidates, self.swap_mode)
        result = self.goldfisher.simulate()
        return self._extract_score(result)

    def _extract_score(self, result) -> float:
        """Extract the optimization target from a SimulationResult."""
        if self.optimize_for == "floor_performance":
            return result.threshold_mana
        if self.optimize_for == "consistency":
            return result.consistency
        if self.optimize_for == "mean_mana_value":
            return result.mean_mana_value
        if self.optimize_for == "mean_mana_total":
            return result.mean_mana_total
        if self.optimize_for == "mean_spells_cast":
            return result.mean_spells_cast
        if self.optimize_for == "karsten_ecms":
            return result.mean_ecms
        return result.mean_mana

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
        if self.optimize_for == "karsten_ecms":
            return result_dict.get("mean_ecms", 0.0)
        return result_dict.get("mean_mana", 0.0)
