"""Track daily risk state and halt new entries when the configured loss limit is breached."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    max_notional_per_pair: float = 10_000.0
    max_concurrent_pairs: int = 5
    daily_loss_limit: float = 2_000.0


@dataclass
class RiskState:
    daily_realized_pnl: float = 0.0
    halted: bool = False
    halt_reason: str | None = None
    current_day: object | None = None  # a date, compared against each bar's timestamp.date()


def maybe_roll_day(risk: RiskState, timestamp) -> None:
    """Resets the daily P&L counter and any halt at the first bar of a new calendar
    day - a kill switch trip is a daily limit, not a permanent shutdown."""
    day = timestamp.date() if hasattr(timestamp, "date") else timestamp
    if risk.current_day is not None and day == risk.current_day:
        return
    risk.current_day = day
    risk.daily_realized_pnl = 0.0
    risk.halted = False
    risk.halt_reason = None


def check_kill_switch(risk: RiskState, daily_pnl: float, config: RiskConfig) -> None:
    """Halts new entries (mutates risk in place) if daily_pnl breaches the limit.
    Does nothing if already halted - the halt_reason from the first breach is kept,
    not overwritten by a later, deeper breach the same day."""
    if risk.halted:
        return
    if daily_pnl <= -config.daily_loss_limit:
        risk.halted = True
        risk.halt_reason = f"daily P&L {daily_pnl:.2f} breached -{config.daily_loss_limit:.2f} limit"
