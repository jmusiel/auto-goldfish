"""SQLAlchemy 2.0 models for persisting simulation data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CardRow(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)


class EffectLabelRow(Base):
    __tablename__ = "effect_labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    effects_json: Mapped[str] = mapped_column(Text, unique=True, nullable=False)


class DeckRow(Base):
    __tablename__ = "decks"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
    )

    cards: Mapped[list["DeckCardRow"]] = relationship(back_populates="deck")
    runs: Mapped[list["SimulationRunRow"]] = relationship(back_populates="deck")


class DeckCardRow(Base):
    __tablename__ = "deck_cards"

    id: Mapped[int] = mapped_column(primary_key=True)
    deck_id: Mapped[int] = mapped_column(ForeignKey("decks.id"), nullable=False)
    card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"), nullable=False)
    label_id: Mapped[Optional[int]] = mapped_column(ForeignKey("effect_labels.id"), nullable=True)
    user_edited: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (
        UniqueConstraint("deck_id", "card_id", name="uq_deck_card"),
    )

    deck: Mapped["DeckRow"] = relationship(back_populates="cards")
    card: Mapped["CardRow"] = relationship()
    label: Mapped[Optional["EffectLabelRow"]] = relationship()


class SimulationRunRow(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    deck_id: Mapped[int] = mapped_column(ForeignKey("decks.id"), nullable=False)
    turns: Mapped[int] = mapped_column(Integer, nullable=False)
    sims: Mapped[int] = mapped_column(Integer, nullable=False)
    min_lands: Mapped[int] = mapped_column(Integer, nullable=False)
    max_lands: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mulligan_strategy: Mapped[str] = mapped_column(Text, nullable=False, default="default")
    optimal_land_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
    )

    deck: Mapped["DeckRow"] = relationship(back_populates="runs")
    results: Mapped[list["SimulationResultRow"]] = relationship(back_populates="run")
    card_performances: Mapped[list["CardPerformanceRow"]] = relationship(back_populates="run")


class SimulationResultRow(Base):
    __tablename__ = "simulation_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("simulation_runs.id"), nullable=False)
    land_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_mana: Mapped[float] = mapped_column(Float, nullable=False)
    mean_draws: Mapped[float] = mapped_column(Float, nullable=False)
    mean_bad_turns: Mapped[float] = mapped_column(Float, nullable=False)
    mean_lands: Mapped[float] = mapped_column(Float, nullable=False)
    mean_mulls: Mapped[float] = mapped_column(Float, nullable=False)
    mean_spells_cast: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ci_mean_mana_low: Mapped[float] = mapped_column(Float, nullable=False)
    ci_mean_mana_high: Mapped[float] = mapped_column(Float, nullable=False)
    consistency: Mapped[float] = mapped_column(Float, nullable=False)
    ci_consistency_low: Mapped[float] = mapped_column(Float, nullable=False)
    ci_consistency_high: Mapped[float] = mapped_column(Float, nullable=False)
    percentile_25: Mapped[float] = mapped_column(Float, nullable=False)
    percentile_50: Mapped[float] = mapped_column(Float, nullable=False)
    percentile_75: Mapped[float] = mapped_column(Float, nullable=False)
    score_speed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    score_power: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    score_consistency: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    score_resilience: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    score_efficiency: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    score_momentum: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "land_count", name="uq_run_land"),
    )

    run: Mapped["SimulationRunRow"] = relationship(back_populates="results")


class CardAnnotationRow(Base):
    __tablename__ = "card_annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    card_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    effects_json: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="human")
    session_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
    )


class CardPerformanceRow(Base):
    __tablename__ = "card_performance"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("simulation_runs.id"), nullable=False)
    card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"), nullable=False)
    mean_with: Mapped[float] = mapped_column(Float, nullable=False)
    mean_without: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "card_id", name="uq_run_card"),
    )

    run: Mapped["SimulationRunRow"] = relationship(back_populates="card_performances")
    card: Mapped["CardRow"] = relationship()
