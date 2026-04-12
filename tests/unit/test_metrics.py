"""Tests for metrics system."""

from auto_goldfish.metrics.collector import GameRecord, MetricsCollector
from auto_goldfish.metrics.definitions import (
    consistency,
    mean_ecms,
    mean_hand_sum,
    mean_mana_draw,
    mean_mana_ramp,
    mean_mana_spent,
    mean_mana_total,
    mean_mana_value,
)


def _make_records(mana_values: list[int]) -> list[GameRecord]:
    return [GameRecord(total_mana_spent=m) for m in mana_values]


def _make_detailed_records() -> list[GameRecord]:
    return [
        GameRecord(total_mana_spent=20, mana_value=12, mana_draw=8, mana_ramp=5, hand_sum=40),
        GameRecord(total_mana_spent=30, mana_value=18, mana_draw=12, mana_ramp=10, hand_sum=50),
        GameRecord(total_mana_spent=10, mana_value=6, mana_draw=4, mana_ramp=3, hand_sum=30),
    ]


def test_mean_mana_spent():
    records = _make_records([10, 20, 30])
    assert mean_mana_spent(records) == 20.0


def test_mean_mana_value():
    records = _make_detailed_records()
    assert mean_mana_value(records) == 12.0


def test_mean_mana_draw():
    records = _make_detailed_records()
    assert mean_mana_draw(records) == 8.0


def test_mean_mana_ramp():
    records = _make_detailed_records()
    assert mean_mana_ramp(records) == 6.0


def test_mean_mana_total():
    records = _make_detailed_records()
    # total = (12+8+5 + 18+12+10 + 6+4+3) / 3 = (25 + 40 + 13) / 3 = 26.0
    assert mean_mana_total(records) == 26.0


def test_mean_hand_sum():
    records = _make_detailed_records()
    assert mean_hand_sum(records) == 40.0


def test_consistency_uniform():
    # All same values -> left-tail ratio should be exactly 1.0
    records = _make_records([10] * 100)
    c = consistency(records)
    assert c == 1.0


def test_consistency_nonuniform():
    # Bottom 25% averages 2, overall mean = (2*25 + 10*75)/100 = 8.0
    # Expected: 2 / 8.0 = 0.25
    records = _make_records([2] * 25 + [10] * 75)
    c = consistency(records)
    assert abs(c - 0.25) < 0.01


def test_consistency_range():
    # Left-tail ratio should always be in (0, 1] for positive values
    records = _make_records([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    c = consistency(records)
    assert 0 < c <= 1.0


def test_mean_ecms():
    records = [
        GameRecord(ecms=100),
        GameRecord(ecms=200),
        GameRecord(ecms=300),
    ]
    assert mean_ecms(records) == 200.0


def test_collector_custom_metric():
    collector = MetricsCollector()
    collector.register_metric("total", lambda rs: sum(r.total_mana_spent for r in rs))
    records = _make_records([5, 10, 15])
    result = collector.compute(records)
    assert result["total"] == 30
