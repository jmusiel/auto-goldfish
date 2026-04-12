"""MetricsCollector -- gathers per-game data during simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class GameRecord:
    """Data recorded for a single simulated game."""

    total_mana_spent: int = 0
    mana_value: int = 0
    mana_draw: int = 0
    mana_ramp: int = 0
    ecms: int = 0
    hand_sum: int = 0
    lands_played: int = 0
    mulligans: int = 0
    draws: int = 0
    bad_turns: int = 0
    mid_turns: int = 0
    cards_played: List[str] = field(default_factory=list)
    starting_hand: List[str] = field(default_factory=list)
    starting_hand_land_count: int = 0
    log: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects GameRecords and computes registered metrics.

    Users can register custom metric functions::

        collector = MetricsCollector()
        collector.register_metric("avg_mana", lambda records: sum(r.total_mana_spent for r in records) / len(records))
        result = collector.compute(records)
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, Callable[[List[GameRecord]], Any]] = {}

    def register_metric(self, name: str, func: Callable[[List[GameRecord]], Any]) -> None:
        self._metrics[name] = func

    def compute(self, records: List[GameRecord]) -> Dict[str, Any]:
        """Run all registered metrics over the collected records."""
        if not records:
            return {}
        return {name: func(records) for name, func in self._metrics.items()}
