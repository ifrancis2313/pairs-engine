"""Dollar-neutral, hedge-ratio-weighted position sizing.

Not equal-dollar-per-leg - the two legs are weighted so their share counts match the
current rolling hedge ratio (driver_shares = |hedge_ratio| * follower_shares), the
same relationship spread.py's spread = follower - hedge_ratio * driver assumes. This
keeps the position's dollar exposure matched to what the entry signal actually hedged
against, scaled to fit within a fixed notional budget per pair.
"""

from __future__ import annotations


def hedge_weighted_shares(
    driver_price: float, follower_price: float, hedge_ratio: float, notional_budget: float
) -> tuple[float, float]:
    """(driver_shares, follower_shares), both non-negative magnitudes - the caller
    applies sign via pos_driver/pos_follower. Total position notional (driver_shares *
    driver_price + follower_shares * follower_price) equals notional_budget."""
    ratio = abs(hedge_ratio)
    follower_shares = notional_budget / (follower_price + ratio * driver_price)
    driver_shares = ratio * follower_shares
    return driver_shares, follower_shares
