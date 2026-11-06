"""Create bar-event iterators from historical price frames or a polling callback."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Optional

import pandas as pd

from sleeves.pairs_mft.engine.events import BarEvent


def historical_replay(bars_by_pair: dict[str, pd.DataFrame]) -> Iterator[BarEvent]:
    """Merges each pair's [driver, follower] price DataFrame (indexed by timestamp,
    columns "driver"/"follower") into one globally timestamp-sorted stream of
    BarEvents, interleaved across pairs. Sorting globally (not just within each pair)
    matters for two reasons: it's what a live feed would actually deliver (events
    arrive in wall-clock order across every pair, not one pair's whole history then
    the next), and it's what lets the risk state's daily P&L / kill switch roll over
    at the correct real calendar-day boundary across the whole portfolio.
    """
    frames = []
    for pair_name, bars in bars_by_pair.items():
        frame = bars[["driver", "follower"]].copy()
        frame["pair_name"] = pair_name
        frames.append(frame)

    if not frames:
        return

    combined = pd.concat(frames).sort_index()
    for timestamp, row in combined.iterrows():
        yield BarEvent(
            pair_name=row["pair_name"],
            timestamp=timestamp,
            driver_price=float(row["driver"]),
            follower_price=float(row["follower"]),
        )


def live_feed(poll_for_bar: Callable[[], Optional[BarEvent]], poll_interval_seconds: float = 5.0) -> Iterator[BarEvent]:
    """Poll for new bar events and yield them, sleeping between polls as configured."""
    while True:
        event = poll_for_bar()
        if event is not None:
            yield event
        else:
            time.sleep(poll_interval_seconds)
