"""CRN-paired racing optimizer for deck configuration.

Uses Common Random Numbers (same seeds across configs) and sequential
elimination with paired bootstrap confidence intervals to identify
the best deck configuration with minimal simulation budget.

Compared to Hyperband, this approach:
- Uses CRN to reduce variance of pairwise comparisons by 3-5x
- Eliminates inferior configs as soon as statistical confidence allows
- Avoids wasting budget on noisy early-round eliminations
"""

from __future__ import annotations

import random as _stdlib_random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from auto_goldfish.optimization.candidate_cards import CandidateCard
from auto_goldfish.optimization.deck_config import DeckConfig, apply_config, enumerate_configs


class FastDeckOptimizer:
    """CRN-paired racing optimizer for deck configuration.

    Enumerates all valid configurations, then uses sequential elimination
    with Common Random Numbers to efficiently identify the top performers.

    All configs are evaluated on the *same* random seeds each round,
    enabling paired comparisons that dramatically reduce the number of
    simulations needed for confident rankings.

    Fidelity is controlled via ``hyperband_max_sims`` (matching the UI
    config key used by DeckOptimizer) which maps to the internal racing
    budget.  The UI fidelity presets translate as:

    - **Fast** (hyperband_max_sims=100): max_sims_per_config=300
    - **Balanced** (hyperband_max_sims=500): max_sims_per_config=500
    - **High** (hyperband_max_sims=500, eta=2): same as balanced for racing

    Args:
        goldfisher: Goldfisher instance (will be mutated during optimization).
        candidates: Dict of candidate_id -> CandidateCard to consider.
        swap_mode: If True, remove no-effect spells to maintain deck size.
        max_draw: Maximum number of draw candidates to add (0-2).
        max_ramp: Maximum number of ramp candidates to add (0-2).
        land_range: Land delta range (-land_range to +land_range).
            Ignored when ``land_delta_min``/``land_delta_max`` are provided.
        land_delta_min: Explicit lower bound for land delta.  Overrides
            ``-land_range`` when set.
        land_delta_max: Explicit upper bound for land delta.  Overrides
            ``+land_range`` when set.
        optimize_for: Target metric - "mean_mana", "consistency",
            "mean_mana_value", "mean_mana_total", or "mean_spells_cast".
        hyperband_max_sims: Fidelity budget from the UI (same config key
            as DeckOptimizer).  Mapped to racing's max_sims_per_config.
            Values ≤200 select the fast tier (300 racing sims), otherwise
            the balanced tier (500 racing sims).  Default ``None`` uses
            the balanced tier.
        batch_size: Games per evaluation round.
        confidence: Confidence level for elimination (0-1).
        min_games: Minimum games before any elimination begins.
        max_sims_per_config: Direct override for max games per config.
            When ``hyperband_max_sims`` is also provided, this is ignored.
        n_bootstrap: Number of bootstrap resamples for elimination tests.
    """

    # Fidelity tier thresholds
    _FAST_MAX_SIMS = 300
    _BALANCED_MAX_SIMS = 500

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
        hyperband_max_sims: Optional[int] = None,
        batch_size: int = 50,
        confidence: float = 0.95,
        min_games: int = 150,
        max_sims_per_config: int = 500,
        n_bootstrap: int = 200,
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
        self.batch_size = batch_size
        self.confidence = confidence
        self.min_games = min_games
        self.n_bootstrap = n_bootstrap

        # Map UI fidelity (hyperband_max_sims) to racing budget
        if hyperband_max_sims is not None:
            if hyperband_max_sims <= 200:
                self.max_sims_per_config = self._FAST_MAX_SIMS
            else:
                self.max_sims_per_config = self._BALANCED_MAX_SIMS
        else:
            self.max_sims_per_config = max_sims_per_config

        # Populated during run(): race scores for all evaluated configs,
        # in the same (config, score, n_sims) format as Hyperband's
        # all_round_scores, enabling feature analysis and regression.
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

        Phase 1 (Flat racing): Race all configs simultaneously using CRN
        pairing and sequential elimination.

        Phase 2 (Regression analysis): Fit a regression model on racing
        scores for interpretability (recommendations, marginal impact).
        Unlike hyperband, regression configs are *not* added to the eval
        pool because racing survivors are empirically better.

        Phase 3 (Final evaluation): Re-evaluate racing survivors and
        baseline with full simulations.

        The first result dict includes a ``feature_analysis`` key with
        recommendations, marginal impact, and regression details.

        Args:
            final_sims: Simulations for final evaluation of top configs.
            final_top_k: Number of top configs to fully evaluate.
            include_hyperband: Accepted for API compatibility with
                DeckOptimizer but ignored (racing does not use hyperband).
            enum_progress: Optional callable(current, total) for racing phase.
            eval_progress: Optional callable(current, total) for final eval.

        Returns:
            List of (DeckConfig, result_dict) sorted best-first.
        """
        original_sims = self.goldfisher.sims

        # Phase 1: Flat racing elimination over all configs
        self.all_round_scores = []
        all_configs = enumerate_configs(
            self.candidates,
            max_draw=self.max_draw,
            max_ramp=self.max_ramp,
            land_range=self.land_range,
            land_delta_min=self.land_delta_min,
            land_delta_max=self.land_delta_max,
        )
        top_configs = self._race(all_configs, top_k=final_top_k, progress=enum_progress)

        # Phase 2: Regression analysis for interpretability
        # Racing survivors are used as eval candidates (not regression picks,
        # since racing finds better candidates than regression prediction).
        # The regression model is still fitted to produce recommendations.
        from auto_goldfish.optimization.feature_analysis import (
            analyze_optimization,
        )

        self.feature_analysis = analyze_optimization(
            self.all_round_scores, self.optimize_for,
        )

        # Build evaluation pool: racing survivors + baseline
        baseline = DeckConfig()
        eval_pool: list[DeckConfig] = []
        seen: set[DeckConfig] = set()

        for cfg in top_configs:
            if cfg not in seen:
                eval_pool.append(cfg)
                seen.add(cfg)

        if baseline not in seen:
            eval_pool.append(baseline)
            seen.add(baseline)

        # Phase 3: Full evaluation of combined pool
        self.goldfisher.sims = final_sims
        from auto_goldfish.metrics.reporter import result_to_dict

        results: list[tuple[DeckConfig, Any]] = []
        total_eval_sims = len(eval_pool) * final_sims
        for j, config in enumerate(eval_pool):
            apply_config(self.goldfisher, config, self.candidates, self.swap_mode)

            # Sub-progress: report per-sim within each config evaluation
            sub_cb = None
            if eval_progress is not None:
                offset = j * final_sims

                def sub_cb(current: int, total: int, _offset: int = offset) -> None:
                    eval_progress(_offset + current, total_eval_sims)

            result = self.goldfisher.simulate(progress_callback=sub_cb)
            result_dict = result_to_dict(result)
            results.append((config, result_dict))

        self.goldfisher.sims = original_sims

        # Sort final results by target metric
        results.sort(key=lambda r: self._extract_score_from_dict(r[1]), reverse=True)

        # Find baseline rank before cutting
        baseline_rank = None
        baseline_entry = None
        for rank, (cfg, rd) in enumerate(results, 1):
            if cfg == baseline:
                baseline_rank = rank
                baseline_entry = (cfg, rd)
                break

        # Keep top final_top_k
        results = results[:final_top_k]

        # If baseline didn't make the cut, append as reference
        baseline_in_top = any(cfg == baseline for cfg, _ in results)
        if not baseline_in_top and baseline_entry is not None:
            baseline_entry[1]["opt_baseline_rank"] = baseline_rank
            results.append(baseline_entry)

        # Attach feature analysis to the top-ranked result
        if self.feature_analysis and results:
            results[0][1]["feature_analysis"] = self.feature_analysis

        return results

    # -- Racing internals --

    def _race(
        self,
        configs: List[DeckConfig],
        top_k: int,
        progress: Optional[Callable[[int, int], None]],
    ) -> List[DeckConfig]:
        """Sequential elimination with CRN pairing.

        All active configs are evaluated on the same batch of seeds each
        round. After accumulating enough games (>= min_games), configs
        that are statistically worse than the current best are eliminated.
        """
        if len(configs) <= top_k:
            return list(configs)

        base_seed = self.goldfisher.seed if self.goldfisher.seed is not None else _stdlib_random.randrange(2**31)
        max_rounds = self.max_sims_per_config // self.batch_size

        # Estimate total budget for progress reporting
        total_budget_est = len(configs) * self.max_sims_per_config
        done_sims = 0

        # Per-config accumulated mana values as numpy arrays
        # Use a list-of-lists during accumulation, convert to arrays for elimination
        mana_lists: dict[DeckConfig, list[float]] = {cfg: [] for cfg in configs}
        active = set(configs)

        for round_idx in range(max_rounds):
            if len(active) <= top_k:
                break

            # Generate seeds for this batch (same for all configs = CRN)
            batch_start = base_seed + round_idx * self.batch_size
            seeds = [batch_start + j for j in range(self.batch_size)]

            # Evaluate all active configs on this batch
            for cfg in list(active):
                apply_config(self.goldfisher, cfg, self.candidates, self.swap_mode)
                use_ecms = self.optimize_for == "karsten_ecms"
                batch_values = [
                    self.goldfisher.simulate_single_game(s, ecms=use_ecms) for s in seeds
                ]
                mana_lists[cfg].extend(batch_values)
                done_sims += self.batch_size

                if progress is not None:
                    progress(done_sims, total_budget_est)

            # Only start eliminating after min_games
            n_games = (round_idx + 1) * self.batch_size
            if n_games < self.min_games:
                continue

            # Compute scores and eliminate
            active = self._eliminate_round(mana_lists, active, top_k)

        # Collect scores for all evaluated configs (active + eliminated)
        # in the same format as Hyperband's all_round_scores for regression
        for cfg, values in mana_lists.items():
            if values:
                score = self._compute_score(np.array(values))
                self.all_round_scores.append((cfg, score, len(values)))

        # Return top_k by final score
        final_scores = {
            cfg: self._compute_score(np.array(mana_lists[cfg])) for cfg in active
        }
        ranked = sorted(active, key=lambda c: final_scores[c], reverse=True)

        # Report final progress
        if progress is not None:
            progress(done_sims, done_sims)

        return ranked[:top_k]

    def _eliminate_round(
        self,
        mana_lists: dict[DeckConfig, list[float]],
        active: set[DeckConfig],
        top_k: int,
    ) -> set[DeckConfig]:
        """Vectorized paired bootstrap elimination against the current best.

        For mean-based metrics, uses a faster t-test instead of bootstrap.
        For consistency, uses vectorized bootstrap over all candidates at once.
        """
        active_list = list(active)

        # Build matrix: (n_configs, n_games)
        n_games = len(mana_lists[active_list[0]])
        mana_matrix = np.array([mana_lists[cfg] for cfg in active_list])

        # Compute current scores
        scores = np.array([
            self._compute_score(mana_matrix[i]) for i in range(len(active_list))
        ])
        best_idx = int(np.argmax(scores))
        best_cfg = active_list[best_idx]

        if self.optimize_for != "consistency":
            # Fast path: paired t-test for mean-based metrics
            dominated = self._ttest_elimination(mana_matrix, best_idx)
        else:
            # Vectorized bootstrap for consistency
            dominated = self._bootstrap_elimination_vectorized(
                mana_matrix, best_idx, n_games
            )

        to_remove: set[DeckConfig] = set()
        for i, cfg in enumerate(active_list):
            if i != best_idx and dominated[i]:
                to_remove.add(cfg)

        # Don't eliminate below top_k
        max_removals = len(active) - top_k
        if len(to_remove) > max_removals:
            removable = sorted(to_remove, key=lambda c: scores[active_list.index(c)])
            to_remove = set(removable[:max_removals])

        return active - to_remove

    def _ttest_elimination(
        self, mana_matrix: np.ndarray, best_idx: int
    ) -> np.ndarray:
        """Paired t-test elimination for mean-based metrics.

        Much faster than bootstrap. Returns boolean array where True = dominated.
        """
        n_configs, n_games = mana_matrix.shape
        best_row = mana_matrix[best_idx]

        # Paired differences: cfg - best (positive means cfg is better)
        diffs = mana_matrix - best_row[np.newaxis, :]  # (n_configs, n_games)
        means = diffs.mean(axis=1)
        stds = diffs.std(axis=1, ddof=1)
        stds = np.maximum(stds, 1e-10)  # avoid division by zero

        # One-sided t-test: upper CI for (cfg - best)
        se = stds / np.sqrt(n_games)
        # Use z-approximation (n_games >= 40)
        z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(
            self.confidence, 1.645
        )
        upper_ci = means + z * se

        dominated = upper_ci < 0
        dominated[best_idx] = False
        return dominated

    def _bootstrap_elimination_vectorized(
        self, mana_matrix: np.ndarray, best_idx: int, n_games: int
    ) -> np.ndarray:
        """Vectorized paired bootstrap for consistency metric.

        Bootstraps all configs simultaneously using numpy broadcasting.
        Returns boolean array where True = dominated.
        """
        n_configs = mana_matrix.shape[0]
        n_boot = self.n_bootstrap
        rng = np.random.RandomState(42)

        # Generate bootstrap indices: (n_boot, n_games)
        boot_idx = rng.randint(0, n_games, size=(n_boot, n_games))

        # Resample all configs: (n_configs, n_boot, n_games)
        boot_samples = mana_matrix[:, boot_idx]

        # Compute consistency for each bootstrap sample
        # Sort along game axis
        boot_sorted = np.sort(boot_samples, axis=2)
        cutoff = max(1, int(n_games * 0.25))
        tail_means = boot_sorted[:, :, :cutoff].mean(axis=2)  # (n_configs, n_boot)
        overall_means = boot_samples.mean(axis=2)  # (n_configs, n_boot)

        # Avoid division by zero
        safe_means = np.maximum(overall_means, 1e-10)
        boot_consistency = tail_means / safe_means  # (n_configs, n_boot)

        # Paired differences vs best
        best_consistency = boot_consistency[best_idx]  # (n_boot,)
        diffs = boot_consistency - best_consistency[np.newaxis, :]  # (n_configs, n_boot)

        # Upper CI bound
        pct = 100 * self.confidence
        upper_ci = np.percentile(diffs, pct, axis=1)  # (n_configs,)

        dominated = upper_ci < 0
        dominated[best_idx] = False
        return dominated

    # -- Scoring --

    def _compute_score(self, mana_values) -> float:
        """Compute the target metric from raw per-game mana values."""
        if len(mana_values) == 0:
            return 0.0

        arr = np.asarray(mana_values, dtype=float)

        if self.optimize_for == "consistency":
            return self._compute_consistency(arr)
        return float(arr.mean())

    @staticmethod
    def _compute_consistency(mana_values: np.ndarray, threshold: float = 0.25) -> float:
        """Left-tail ratio from raw per-game mana array.

        Matches the computation in metrics.definitions.consistency().
        """
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
        if self.optimize_for == "karsten_ecms":
            return result_dict.get("mean_ecms", 0.0)
        return result_dict.get("mean_mana", 0.0)
