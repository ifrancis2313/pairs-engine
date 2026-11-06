"""Generate two-leg positions from spread z-scores with entry and exit hysteresis."""

from __future__ import annotations

import pandas as pd


def next_position(z_value: float, hold: int, prev_pos_driver: int, prev_pos_follower: int, entry: float, exit: float) -> tuple[int, int, int]:
    """One bar's position decision given the current hold state and prior position.

    Entry: z > +entry -> long driver, short follower (follower is rich relative to
    the driver-implied level). z < -entry -> short driver, long follower (follower is
    cheap relative to the driver-implied level).
    Exit: |z| < exit -> close both positions.

    Returns (new_hold, new_pos_driver, new_pos_follower).
    """
    if z_value > entry and hold == 0:
        return 1, 1, -1  # follower rich vs driver -> short follower, long driver
    if z_value < -entry and hold == 0:
        return 1, -1, 1  # follower cheap vs driver -> long follower, short driver
    if hold == 1 and abs(z_value) < exit:
        return 0, 0, 0
    if hold == 1 and abs(z_value) >= exit:
        return hold, prev_pos_driver, prev_pos_follower
    return hold, 0, 0


def generate_signals(z: pd.Series, entry: float = 2.0, exit: float = 0.25) -> tuple[pd.Series, pd.Series]:
    """Apply next_position over a z-score series and return both legs' position series."""
    hold = 0
    pos_driver: list[int] = []
    pos_follower: list[int] = []
    prev_d, prev_f = 0, 0

    for value in z:
        hold, prev_d, prev_f = next_position(value, hold, prev_d, prev_f, entry, exit)
        pos_driver.append(prev_d)
        pos_follower.append(prev_f)

    return (
        pd.Series(pos_driver, index=z.index, name="pos_driver"),
        pd.Series(pos_follower, index=z.index, name="pos_follower"),
    )
