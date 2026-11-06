"""Test ordering, warmup, orders, risk halts, and replay/polling agreement with synthetic prices."""

from __future__ import annotations

import itertools
import unittest

import numpy as np
import pandas as pd

from sleeves.pairs_mft.contracts import PairDefinition, ScreeningEvidence
from sleeves.pairs_mft.engine.core import run_bar
from sleeves.pairs_mft.engine.driver import run
from sleeves.pairs_mft.engine.events import BarEvent
from sleeves.pairs_mft.engine.gate import PairEngineError, check_bar_ordering
from sleeves.pairs_mft.engine.risk import RiskConfig, RiskState, check_kill_switch, maybe_roll_day
from sleeves.pairs_mft.engine.sources import historical_replay, live_feed
from sleeves.pairs_mft.engine.state import EngineState
from sleeves.pairs_mft.spread import compute_spread, rolling_hedge_ratio, zscore


def _z_after_appending(driver_history, follower_history, hedge_window, timestamp, driver_price, follower_price) -> float:
    """The z-score run_bar would compute for a bar with the given prices, appended to
    COPIES of the existing history - uses the real spread.py functions against the
    actual post-append rolling window (not an approximation from the pre-append
    window), so callers can numerically solve for a target z exactly as run_bar would
    see it, without mutating engine state."""
    driver_series = pd.Series(dict(list(driver_history) + [(timestamp, driver_price)]))
    follower_series = pd.Series(dict(list(follower_history) + [(timestamp, follower_price)]))
    hedge_ratio_series = rolling_hedge_ratio(driver_series, follower_series, hedge_window)
    spread_series = compute_spread(driver_series, follower_series, hedge_ratio_series)
    z_series = zscore(spread_series, hedge_window)
    return float(z_series.iloc[-1])


def _solve_follower_price_for_zero_z(driver_history, follower_history, hedge_window, timestamp, driver_price, initial_guess) -> float:
    """Secant-method solve for the follower_price that makes _z_after_appending land
    at (approximately) zero - spread is linear in follower_price for a fixed
    driver_price, so this converges in a couple of iterations."""
    x0, x1 = initial_guess, initial_guess * 1.001
    f0 = _z_after_appending(driver_history, follower_history, hedge_window, timestamp, driver_price, x0)
    for _ in range(10):
        f1 = _z_after_appending(driver_history, follower_history, hedge_window, timestamp, driver_price, x1)
        if abs(f1) < 1e-6 or f1 == f0:
            return x1
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0, x1 = x1, f1, x2
    return x1


def _pair_def(name="A/B", driver="A", follower="B", hedge_window=10, entry_z=2.0, exit_z=0.25) -> PairDefinition:
    evidence = ScreeningEvidence(sector="test", full_period_p_value=0.01, regime_p_values={}, fdr_q_value=0.01, correlation=0.9)
    return PairDefinition(
        name=name, driver=driver, follower=follower, hedge_window=hedge_window,
        entry_z=entry_z, exit_z=exit_z, screening_evidence=evidence,
    )


def _synthetic_events(pair_name: str, n: int, seed: int = 0, entry_shock_at: int | None = None) -> list[BarEvent]:
    """A driver random walk plus a follower that tracks it with a mean-reverting
    residual, with an optional artificial one-bar shock to force a clean entry."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    driver = 100 + np.cumsum(rng.normal(0, 0.02, n))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.9 * residual[i - 1] + rng.normal(0, 0.05)
    if entry_shock_at is not None:
        residual[entry_shock_at:] += 5.0  # force a large, persistent spread deviation
    follower = 2.0 * driver + residual
    return [
        BarEvent(pair_name=pair_name, timestamp=idx[i], driver_price=float(driver[i]), follower_price=float(follower[i]))
        for i in range(n)
    ]


class TestGate(unittest.TestCase):
    def test_first_bar_always_passes(self):
        check_bar_ordering(BarEvent("A/B", pd.Timestamp("2024-01-01"), 1.0, 2.0), None)

    def test_out_of_order_bar_raises(self):
        with self.assertRaises(PairEngineError):
            check_bar_ordering(BarEvent("A/B", pd.Timestamp("2024-01-01"), 1.0, 2.0), pd.Timestamp("2024-01-02"))

    def test_duplicate_timestamp_raises(self):
        ts = pd.Timestamp("2024-01-01")
        with self.assertRaises(PairEngineError):
            check_bar_ordering(BarEvent("A/B", ts, 1.0, 2.0), ts)


class TestRisk(unittest.TestCase):
    def test_kill_switch_trips_at_limit(self):
        risk = RiskState(current_day=pd.Timestamp("2024-01-01").date())
        config = RiskConfig(daily_loss_limit=100.0)
        check_kill_switch(risk, daily_pnl=-50.0, config=config)
        self.assertFalse(risk.halted)
        check_kill_switch(risk, daily_pnl=-150.0, config=config)
        self.assertTrue(risk.halted)
        self.assertIn("breached", risk.halt_reason)

    def test_kill_switch_does_not_overwrite_first_reason(self):
        risk = RiskState(current_day=pd.Timestamp("2024-01-01").date())
        config = RiskConfig(daily_loss_limit=100.0)
        check_kill_switch(risk, daily_pnl=-150.0, config=config)
        first_reason = risk.halt_reason
        check_kill_switch(risk, daily_pnl=-500.0, config=config)
        self.assertEqual(risk.halt_reason, first_reason)

    def test_day_rollover_resets_halt_and_pnl(self):
        risk = RiskState(current_day=pd.Timestamp("2024-01-01").date(), daily_realized_pnl=-999, halted=True, halt_reason="x")
        maybe_roll_day(risk, pd.Timestamp("2024-01-02 09:30"))
        self.assertFalse(risk.halted)
        self.assertEqual(risk.daily_realized_pnl, 0.0)

    def test_same_day_does_not_reset(self):
        risk = RiskState(current_day=pd.Timestamp("2024-01-01").date(), daily_realized_pnl=-50, halted=True, halt_reason="x")
        maybe_roll_day(risk, pd.Timestamp("2024-01-01 10:00"))
        self.assertTrue(risk.halted)
        self.assertEqual(risk.daily_realized_pnl, -50)


class TestEngineDriver(unittest.TestCase):
    def test_no_orders_during_warmup(self):
        pair_def = _pair_def(hedge_window=10)
        events = _synthetic_events("A/B", n=15)  # < 2*hedge_window=20
        collected = []
        run(events, {"A/B": pair_def}, RiskConfig(), on_orders=lambda e, o: collected.append((e, o)))
        self.assertEqual(collected, [])

    def test_entry_produces_two_orders_for_both_legs(self):
        pair_def = _pair_def(hedge_window=10, entry_z=1.5)
        events = _synthetic_events("A/B", n=60, entry_shock_at=40)
        collected = []
        run(events, {"A/B": pair_def}, RiskConfig(), on_orders=lambda e, o: collected.append((e, o)))
        self.assertTrue(collected, "expected at least one order-producing bar")
        first_event, first_orders = collected[0]
        self.assertEqual(len(first_orders), 2)
        legs = {o.leg for o in first_orders}
        self.assertEqual(legs, {"driver", "follower"})
        sides = {o.side for o in first_orders}
        self.assertEqual(sides, {"buy", "sell"})  # opposite sides for a hedged entry

    def test_max_concurrent_pairs_blocks_new_entries(self):
        pair_defs = {name: _pair_def(name=name, driver=f"D{i}", follower=f"F{i}", hedge_window=10, entry_z=1.5) for i, name in enumerate(["P1", "P2", "P3"])}
        events = []
        for i, name in enumerate(pair_defs):
            events += _synthetic_events(name, n=60, seed=i, entry_shock_at=40)
        events.sort(key=lambda e: e.timestamp)

        risk_config = RiskConfig(max_concurrent_pairs=1)
        opened_pairs = set()
        def on_orders(event, orders):
            if any(o.side in ("buy", "sell") for o in orders) and event.pair_name not in opened_pairs:
                opened_pairs.add(event.pair_name)

        state = run(events, pair_defs, risk_config, on_orders=on_orders)
        open_now = sum(1 for p in state.pairs.values() if p.hold == 1)
        self.assertLessEqual(open_now, 1, "max_concurrent_pairs=1 must never allow more than 1 simultaneously open pair")

    def test_kill_switch_halts_new_entries_but_lets_existing_exit(self):
        """Deterministic setup (not relying on stochastic data to organically trip
        the switch): pre-seed an already-halted EngineState with one pair already
        open (C/D), then feed one bar each for a NEW pair (A/B, should be blocked
        from opening) and the already-open pair (C/D, whose z-score is pushed inside
        exit_z, and should be allowed to close despite the halt)."""
        hedge_window = 10
        pair_defs = {
            "A/B": _pair_def(name="A/B", driver="A", follower="B", hedge_window=hedge_window, entry_z=1.0, exit_z=0.1),
            "C/D": _pair_def(name="C/D", driver="C", follower="D", hedge_window=hedge_window, entry_z=1.0, exit_z=0.1),
        }

        # Warm up both pairs' history via ordinary (non-triggering) bars first.
        state = EngineState()
        ab_events = _synthetic_events("A/B", n=2 * hedge_window + 3, seed=1)
        cd_events = _synthetic_events("C/D", n=2 * hedge_window + 3, seed=2)
        risk_config = RiskConfig(daily_loss_limit=0.01, max_notional_per_pair=10_000, max_concurrent_pairs=5)
        for event in sorted(ab_events + cd_events, key=lambda e: e.timestamp):
            state, _ = run_bar(state, event, pair_defs[event.pair_name], risk_config)

        # Force C/D into an open position directly (bypassing signal generation, so
        # the test doesn't depend on the synthetic walk happening to trigger entry).
        cd_state = state.pairs["C/D"]
        cd_state.hold, cd_state.pos_driver, cd_state.pos_follower = 1, 1, -1
        cd_state.entry_price_driver = cd_state.driver_history[-1][1]
        cd_state.entry_price_follower = cd_state.follower_history[-1][1]
        cd_state.driver_shares, cd_state.follower_shares = 50.0, 25.0

        # Force the kill switch on directly - isolates "does halted block/allow
        # correctly" from "does some P&L path actually trip it", which is covered by
        # TestRisk.test_kill_switch_trips_at_limit already.
        state.risk.halted = True
        state.risk.halt_reason = "forced for test"

        last_ab_ts = ab_events[-1].timestamp
        last_cd_ts = cd_events[-1].timestamp
        next_ts = max(last_ab_ts, last_cd_ts) + pd.Timedelta(minutes=5)

        # A/B: a fresh, currently-flat pair - a strong entry-triggering bar must NOT open it.
        ab_shock = BarEvent("A/B", next_ts, driver_price=100.0, follower_price=300.0)
        state, ab_orders = run_bar(state, ab_shock, pair_defs["A/B"], risk_config)
        self.assertEqual(ab_orders, [])
        self.assertEqual(state.pairs["A/B"].hold, 0, "kill switch must block a new entry")

        # C/D: already open - numerically solve for the follower price that puts the
        # NEXT bar's z-score at ~0 (using the real post-append rolling window, not an
        # approximation from the pre-append one), so the exit condition
        # (|z| < exit_z) is deterministically guaranteed to trigger rather than left
        # to chance.
        exit_driver_price = cd_state.driver_history[-1][1]
        exit_follower_price = _solve_follower_price_for_zero_z(
            cd_state.driver_history, cd_state.follower_history, hedge_window, next_ts, exit_driver_price,
            initial_guess=cd_state.follower_history[-1][1],
        )

        cd_exit_bar = BarEvent("C/D", next_ts, driver_price=exit_driver_price, follower_price=exit_follower_price)
        state, cd_orders = run_bar(state, cd_exit_bar, pair_defs["C/D"], risk_config)

        self.assertEqual(state.pairs["C/D"].hold, 0, "an already-open pair must still be allowed to exit while halted")
        # C/D was opened long-driver/short-follower; closing must sell the driver
        # leg and buy back the follower leg, at the original entry size.
        by_leg = {o.leg: o for o in cd_orders}
        self.assertEqual(by_leg["driver"].side, "sell")
        self.assertEqual(by_leg["follower"].side, "buy")
        self.assertAlmostEqual(by_leg["driver"].shares, 50.0)
        self.assertAlmostEqual(by_leg["follower"].shares, 25.0)

    def test_replay_and_live_feed_drivers_agree_on_the_same_bar_sequence(self):
        """Assert identical orders and final positions for replay and polling over the same synthetic bars."""
        pair_def = _pair_def(hedge_window=10, entry_z=1.5)
        events = _synthetic_events("A/B", n=60, entry_shock_at=40)

        replay_source = historical_replay({"A/B": pd.DataFrame(
            {"driver": [e.driver_price for e in events], "follower": [e.follower_price for e in events]},
            index=[e.timestamp for e in events],
        )})
        replay_orders = []
        replay_state = run(replay_source, {"A/B": pair_def}, RiskConfig(), on_orders=lambda e, o: replay_orders.append((e.timestamp, tuple(o))))

        it = iter(events)
        def poll():
            return next(it, None)
        live_source = itertools.islice(live_feed(poll, poll_interval_seconds=0), len(events))
        live_orders = []
        live_state = run(live_source, {"A/B": pair_def}, RiskConfig(), on_orders=lambda e, o: live_orders.append((e.timestamp, tuple(o))))

        self.assertEqual(len(replay_orders), len(live_orders))
        for (rt, ro), (lt, lo) in zip(replay_orders, live_orders):
            self.assertEqual(rt, lt)
            self.assertEqual(ro, lo)

        replay_pair = replay_state.pairs["A/B"]
        live_pair = live_state.pairs["A/B"]
        self.assertEqual(replay_pair.hold, live_pair.hold)
        self.assertEqual(replay_pair.pos_driver, live_pair.pos_driver)
        self.assertEqual(replay_pair.pos_follower, live_pair.pos_follower)


if __name__ == "__main__":
    unittest.main()
