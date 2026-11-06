"""Reject duplicate or out-of-order timestamps for each pair."""

from __future__ import annotations

from sleeves.pairs_mft.engine.events import BarEvent


class PairEngineError(Exception):
    pass


def check_bar_ordering(event: BarEvent, last_seen_timestamp: object | None) -> None:
    """Raises if `event` is not strictly after the last bar seen for this pair -
    catches an out-of-order feed, a duplicate bar, or a replay driver whose input
    DataFrame wasn't sorted, before it can silently corrupt the rolling hedge-ratio
    history with a look-ahead bar."""
    if last_seen_timestamp is not None and event.timestamp <= last_seen_timestamp:
        raise PairEngineError(
            f"{event.pair_name}: bar at {event.timestamp} is not after the last-seen "
            f"bar at {last_seen_timestamp} - out-of-order or duplicate bar reached run_bar"
        )
