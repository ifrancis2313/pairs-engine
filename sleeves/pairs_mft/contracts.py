"""Immutable pair definitions and screening evidence shared by the engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScreeningEvidence:
    """Screening statistics retained with a pair definition for later inspection."""

    sector: str
    full_period_p_value: float
    regime_p_values: dict[str, float]  # regime label -> Engle-Granger p-value
    fdr_q_value: float  # Benjamini-Hochberg-corrected q-value across the tested set
    correlation: float


@dataclass(frozen=True)
class PairDefinition:
    """Pair identifiers, signal settings, and screening evidence consumed by the engine."""

    name: str  # f"{driver}/{follower}", e.g. "AAA/BBB"
    driver: str
    follower: str
    hedge_window: int  # rolling window (bars) for hedge ratio + z-score
    entry_z: float
    exit_z: float
    screening_evidence: ScreeningEvidence
    notes: str = ""
    intraday_bar_size: str = "5min"
    validated: bool = False  # set True after independent validation
    extra: dict = field(default_factory=dict)
