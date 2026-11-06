"""Process one price bar, update pair and risk state, and emit leg orders."""

from __future__ import annotations

import pandas as pd

from sleeves.pairs_mft.contracts import PairDefinition
from sleeves.pairs_mft.engine.events import BarEvent, Order
from sleeves.pairs_mft.engine.gate import check_bar_ordering
from sleeves.pairs_mft.engine.risk import RiskConfig, check_kill_switch, maybe_roll_day
from sleeves.pairs_mft.engine.sizing import hedge_weighted_shares
from sleeves.pairs_mft.engine.state import EngineState, PairState
from sleeves.pairs_mft.signals import next_position
from sleeves.pairs_mft.spread import compute_spread, rolling_hedge_ratio, zscore


def _position_pnl(pair_state: PairState) -> float:
    if pair_state.hold == 0 or pair_state.entry_price_driver is None or not pair_state.driver_history:
        return 0.0
    current_driver_price = pair_state.driver_history[-1][1]
    current_follower_price = pair_state.follower_history[-1][1]
    driver_pnl = pair_state.pos_driver * pair_state.driver_shares * (current_driver_price - pair_state.entry_price_driver)
    follower_pnl = (
        pair_state.pos_follower * pair_state.follower_shares * (current_follower_price - pair_state.entry_price_follower)
    )
    return driver_pnl + follower_pnl


def _current_zscore(pair_state: PairState, hedge_window: int) -> tuple[float | None, float | None]:
    """Compute the latest spread z-score and hedge ratio from bounded price history.
    Return (None, None) until enough observations are available."""
    if len(pair_state.driver_history) < pair_state.min_history_bars:
        return None, None

    driver_series = pd.Series(dict(pair_state.driver_history))
    follower_series = pd.Series(dict(pair_state.follower_history))
    hedge_ratio_series = rolling_hedge_ratio(driver_series, follower_series, hedge_window)
    spread_series = compute_spread(driver_series, follower_series, hedge_ratio_series)
    z_series = zscore(spread_series, hedge_window)

    if z_series.empty or pd.isna(z_series.iloc[-1]) or pd.isna(hedge_ratio_series.iloc[-1]):
        return None, None
    return float(z_series.iloc[-1]), float(hedge_ratio_series.iloc[-1])


def run_bar(state: EngineState, event: BarEvent, pair_def: PairDefinition, risk_config: RiskConfig) -> tuple[EngineState, list[Order]]:
    pair_state = state.pair_state(event.pair_name, pair_def.hedge_window)
    check_bar_ordering(event, pair_state.last_timestamp)
    pair_state.last_timestamp = event.timestamp
    maybe_roll_day(state.risk, event.timestamp)

    pair_state.driver_history.append((event.timestamp, event.driver_price))
    pair_state.follower_history.append((event.timestamp, event.follower_price))

    orders: list[Order] = []
    z_value, hedge_ratio = _current_zscore(pair_state, pair_def.hedge_window)
    if z_value is None:
        return state, orders  # not enough warmup yet

    new_hold, new_pos_driver, new_pos_follower = next_position(
        z_value, pair_state.hold, pair_state.pos_driver, pair_state.pos_follower, pair_def.entry_z, pair_def.exit_z
    )

    opening_new = pair_state.hold == 0 and new_hold == 1
    if opening_new:
        if state.risk.halted:
            new_hold, new_pos_driver, new_pos_follower = 0, 0, 0  # blocked: kill switch is active
        else:
            open_count = sum(1 for p in state.pairs.values() if p.hold == 1)
            if open_count >= risk_config.max_concurrent_pairs:
                new_hold, new_pos_driver, new_pos_follower = 0, 0, 0  # blocked: at the concurrency cap

    position_changed = new_pos_driver != pair_state.pos_driver or new_pos_follower != pair_state.pos_follower
    if position_changed:
        if new_hold == 1:  # opening
            driver_shares, follower_shares = hedge_weighted_shares(
                event.driver_price, event.follower_price, hedge_ratio, risk_config.max_notional_per_pair
            )
            pair_state.driver_shares = driver_shares
            pair_state.follower_shares = follower_shares
            pair_state.entry_price_driver = event.driver_price
            pair_state.entry_price_follower = event.follower_price
            orders.append(Order(event.pair_name, "driver", pair_def.driver, "buy" if new_pos_driver == 1 else "sell", driver_shares))
            orders.append(Order(event.pair_name, "follower", pair_def.follower, "buy" if new_pos_follower == 1 else "sell", follower_shares))
        else:  # closing - direction is the OPPOSITE of the position being closed,
            # and shares is the ORIGINAL opening size, both taken from pair_state
            # before it's overwritten below (new_pos_driver/new_pos_follower are
            # already 0 here, and driver_shares/follower_shares fields get zeroed
            # right after this block).
            state.risk.daily_realized_pnl += _position_pnl(pair_state)
            orders.append(
                Order(event.pair_name, "driver", pair_def.driver, "sell" if pair_state.pos_driver == 1 else "buy", pair_state.driver_shares)
            )
            orders.append(
                Order(event.pair_name, "follower", pair_def.follower, "sell" if pair_state.pos_follower == 1 else "buy", pair_state.follower_shares)
            )
            pair_state.driver_shares = 0.0
            pair_state.follower_shares = 0.0

    pair_state.hold, pair_state.pos_driver, pair_state.pos_follower = new_hold, new_pos_driver, new_pos_follower

    portfolio_unrealized = sum(_position_pnl(p) for p in state.pairs.values())
    daily_pnl = state.risk.daily_realized_pnl + portfolio_unrealized
    check_kill_switch(state.risk, daily_pnl, risk_config)

    return state, orders
