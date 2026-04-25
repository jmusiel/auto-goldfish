"""Background simulation job manager.

Runs Goldfisher simulations in daemon threads. GameState is self-contained
(no module-level globals), so concurrent simulations are safe.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from auto_goldfish.decklist.loader import load_decklist
from auto_goldfish.effects.card_database import DEFAULT_REGISTRY
from auto_goldfish.effects.json_loader import build_overridden_registry
from auto_goldfish.engine.goldfisher import Goldfisher
from auto_goldfish.engine.mulligan import CurveAwareMulligan
from auto_goldfish.metrics.reporter import result_to_dict


@dataclass
class SimJob:
    job_id: str
    deck_name: str
    config: Dict[str, Any]
    status: str = "pending"  # pending | running | completed | failed
    progress: int = 0
    total: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class SimulationRunner:
    """Manages background simulation jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, SimJob] = {}
        self._lock = threading.Lock()

    def submit(self, deck_name: str, config: Dict[str, Any]) -> str:
        """Start a background simulation. Returns the job ID."""
        job_id = uuid.uuid4().hex[:12]
        min_lands = config.get("min_lands", 36)
        max_lands = config.get("max_lands", 39)
        total = max_lands - min_lands + 1

        job = SimJob(
            job_id=job_id,
            deck_name=deck_name,
            config=config,
            total=total,
        )

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_simulation,
            args=(job,),
            daemon=True,
        )
        thread.start()
        return job_id

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the current status of a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.job_id,
                "deck_name": job.deck_name,
                "status": job.status,
                "progress": job.progress,
                "total": job.total,
                "config": job.config,
                "results": job.results,
                "error": job.error,
            }

    def _run_simulation(self, job: SimJob) -> None:
        """Execute the simulation in a background thread."""
        try:
            with self._lock:
                job.status = "running"

            deck_list = load_decklist(job.deck_name)

            mulligan_strategy = None
            if job.config.get("mulligan") == "curve_aware":
                mulligan_strategy = CurveAwareMulligan()

            # Build custom registry if overrides provided
            effect_overrides = job.config.get("effect_overrides", {})
            registry = None
            if effect_overrides:
                registry = build_overridden_registry(DEFAULT_REGISTRY, effect_overrides)

            goldfisher = Goldfisher(
                deck_list,
                turns=job.config.get("turns", 8),
                sims=job.config.get("sims", 1000),
                verbose=False,
                record_results=job.config.get("record_results", "quartile"),
                deck_name=job.deck_name,
                seed=job.config.get("seed"),
                workers=job.config.get("workers", 1),
                mulligan_strategy=mulligan_strategy,
                registry=registry,
                mana_mode=job.config.get("mana_mode", "value"),
                spell_priority=job.config.get("spell_priority", "priority_then_cmc"),
                mana_efficiency=job.config.get("mana_efficiency", "greedy"),
                ramp_cutoff_turn=job.config.get("ramp_cutoff_turn", 0),
                min_cost_floor=job.config.get("min_cost_floor", 1),
            )

            if job.config.get("optimization_enabled"):
                self._run_optimization(job, goldfisher)
            else:
                min_lands = job.config.get("min_lands", goldfisher.land_count)
                max_lands = job.config.get("max_lands", goldfisher.land_count)

                for i in range(min_lands, max_lands + 1):
                    goldfisher.set_lands(i, cuts=job.config.get("cuts", []))
                    result = goldfisher.simulate()
                    result_dict = result_to_dict(result, turns=job.config.get("turns", 10))

                    with self._lock:
                        job.results.append(result_dict)
                        job.progress = len(job.results)

            with self._lock:
                job.status = "completed"

            try:
                from auto_goldfish.db.persistence import persist_completed_job

                persist_completed_job(job)
            except Exception:
                logger.exception("Failed to persist simulation run to DB")

        except Exception as e:
            with self._lock:
                job.status = "failed"
                job.error = str(e)

    def _run_optimization(self, job: SimJob, goldfisher: Goldfisher) -> None:
        """Run enumerate-and-evaluate optimization within a simulation job."""
        from auto_goldfish.optimization.candidate_cards import (
            ALL_CANDIDATES,
            make_custom_candidate,
        )
        from auto_goldfish.optimization.fast_optimizer import FastDeckOptimizer
        from auto_goldfish.optimization.optimizer import DeckOptimizer

        config = job.config
        enabled_ids = set(config.get("enabled_candidates", []))
        candidates = {
            cid: c for cid, c in ALL_CANDIDATES.items() if cid in enabled_ids
        }

        custom_draw = config.get("custom_draw")
        custom_ramp = config.get("custom_ramp")
        if custom_draw and custom_draw.get("cmc") is not None:
            cc = make_custom_candidate("draw", custom_draw["cmc"], custom_draw["amount"])
            candidates[cc.id] = cc
        if custom_ramp and custom_ramp.get("cmc") is not None:
            cc = make_custom_candidate("ramp", custom_ramp["cmc"], custom_ramp["amount"])
            candidates[cc.id] = cc

        hyperband_max_sims = config.get("hyperband_max_sims", max(goldfisher.sims // 2, 100))
        eta = config.get("eta", 3)
        hyperband_min_sims = config.get("hyperband_min_sims", 20)
        hyperband_top_k = config.get("hyperband_top_k")
        final_sims = goldfisher.sims

        def enum_cb(current: int, total: int) -> None:
            with self._lock:
                job.progress = current
                job.total = total

        def eval_cb(current: int, total: int) -> None:
            with self._lock:
                job.progress = current
                job.total = total

        # Compute land deltas from absolute min/max lands
        min_lands = config.get("min_lands")
        max_lands = config.get("max_lands")
        land_delta_min = None
        land_delta_max = None
        if min_lands is not None:
            land_delta_min = min_lands - goldfisher.land_count
        if max_lands is not None:
            land_delta_max = max_lands - goldfisher.land_count

        algorithm = config.get("algorithm", "racing")
        optimize_for = config.get("optimize_for", "floor_performance")
        swap_mode = config.get("swap_mode", False)
        max_draw = config.get("max_draw_additions", 2)
        max_ramp = config.get("max_ramp_additions", 2)

        if algorithm == "racing":
            optimizer = FastDeckOptimizer(
                goldfisher=goldfisher,
                candidates=candidates,
                swap_mode=swap_mode,
                max_draw=max_draw,
                max_ramp=max_ramp,
                land_delta_min=land_delta_min,
                land_delta_max=land_delta_max,
                optimize_for=optimize_for,
                hyperband_max_sims=hyperband_max_sims,
            )
        else:
            optimizer = DeckOptimizer(
                goldfisher=goldfisher,
                candidates=candidates,
                swap_mode=swap_mode,
                max_draw=max_draw,
                max_ramp=max_ramp,
                land_delta_min=land_delta_min,
                land_delta_max=land_delta_max,
                optimize_for=optimize_for,
                hyperband_max_sims=hyperband_max_sims,
                eta=eta,
                hyperband_min_sims=hyperband_min_sims,
                hyperband_top_k=hyperband_top_k,
            )

        ranked = optimizer.run(
            final_sims=final_sims,
            final_top_k=5,
            enum_progress=enum_cb,
            eval_progress=eval_cb,
        )

        for deck_config, result_dict in ranked:
            result_dict["opt_config"] = deck_config.describe()
            result_dict["opt_land_delta"] = deck_config.land_delta
            result_dict["opt_added_cards"] = list(deck_config.added_cards)
            with self._lock:
                job.results.append(result_dict)
