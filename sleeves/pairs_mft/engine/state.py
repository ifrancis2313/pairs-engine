"""Engine state: one PairState per traded pair, plus the portfolio-wide RiskState.

Price history is kept as a bounded deque (not a growing pandas Series - O(1) append,
auto-drops the oldest bar) per leg, sized to hold enough bars to recompute the SAME
rolling hedge-ratio/spread/z-score `spread.py` computes for a batch backtest: a
rolling hedge ratio needs `hedge_window` price bars to produce one value, and the
z-score needs `hedge_window` of those spread values for its own rolling mean/std - so
`2 * hedge_window` price bars is the minimum warmup, kept with a little slack.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sleeves.pairs_mft.engine.risk import RiskState

HISTORY_SLACK_BARS = 5


@dataclass
class PairState:
    hedge_window: int
    hold: int = 0
    pos_driver: int = 0
    pos_follower: int = 0
    driver_history: deque = field(default_factory=deque)
    follower_history: deque = field(default_factory=deque)
    last_timestamp: object | None = None
    entry_price_driver: float | None = None
    entry_price_follower: float | None = None
    driver_shares: float = 0.0
    follower_shares: float = 0.0

    def __post_init__(self) -> None:
        maxlen = 2 * self.hedge_window + HISTORY_SLACK_BARS
        self.driver_history = deque(self.driver_history, maxlen=maxlen)
        self.follower_history = deque(self.follower_history, maxlen=maxlen)

    @property
    def min_history_bars(self) -> int:
        return 2 * self.hedge_window


@dataclass
class EngineState:
    pairs: dict[str, PairState] = field(default_factory=dict)
    risk: RiskState = field(default_factory=RiskState)

    def pair_state(self, pair_name: str, hedge_window: int) -> PairState:
        if pair_name not in self.pairs:
            self.pairs[pair_name] = PairState(hedge_window=hedge_window)
        return self.pairs[pair_name]
