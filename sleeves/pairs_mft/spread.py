"""Compute rolling hedge ratios, hedged spreads, and spread z-scores."""

from __future__ import annotations

import pandas as pd


def rolling_hedge_ratio(driver: pd.Series, follower: pd.Series, window: int = 60) -> pd.Series:
    """Rolling OLS slope of follower on driver: NaN for the first window-1 rows."""
    cov = driver.rolling(window).cov(follower)
    var = driver.rolling(window).var()
    return (cov / var).rename("hedge_ratio")


def compute_spread(driver: pd.Series, follower: pd.Series, hedge_ratio: pd.Series) -> pd.Series:
    """follower - hedge_ratio * driver, NaN-dropped (drops the rolling warmup rows)."""
    return (follower - hedge_ratio * driver).dropna().rename("spread")


def zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of the spread - same rolling window as the hedge ratio, so the
    entry/exit signal and the hedge ratio adapt on the same timescale."""
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std(ddof=1)
    return ((spread - rolling_mean) / rolling_std).rename("zscore")
