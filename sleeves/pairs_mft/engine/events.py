"""Bar-event and order dataclasses shared by event sources and the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BarEvent:
    """Prices for one pair at one timestamp, consumed by the bar handler."""

    pair_name: str
    timestamp: object  # pd.Timestamp
    driver_price: float
    follower_price: float


@dataclass(frozen=True)
class Order:
    """One leg's buy or sell action with an explicit ticker and share quantity."""

    pair_name: str
    leg: Literal["driver", "follower"]
    ticker: str
    side: Literal["buy", "sell"]
    shares: float
