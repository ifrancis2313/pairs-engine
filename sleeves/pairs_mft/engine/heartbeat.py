"""Halt new entries when a feed exceeds its staleness limit; existing positions are not force-closed."""

from __future__ import annotations

import pandas as pd

from sleeves.pairs_mft.engine.risk import RiskState


def check_staleness(
    risk: RiskState,
    last_bar_at: pd.Timestamp | None,
    now: pd.Timestamp,
    max_staleness_seconds: float,
) -> None:
    """Halts new entries if `now - last_bar_at` exceeds `max_staleness_seconds`.
    No-ops if already halted (keeps the first halt_reason, same rule
    check_kill_switch already follows) or if no bar has arrived yet at all (a cold
    start isn't a disconnect - nothing to be stale relative to)."""
    if risk.halted or last_bar_at is None:
        return
    elapsed = (now - last_bar_at).total_seconds()
    if elapsed > max_staleness_seconds:
        risk.halted = True
        risk.halt_reason = f"no new bar in {elapsed:.0f}s (limit {max_staleness_seconds:.0f}s) - feed may be stale/disconnected"
