"""Test staleness halts and preservation of an existing halt reason."""

from __future__ import annotations

import unittest

import pandas as pd

from sleeves.pairs_mft.engine.heartbeat import check_staleness
from sleeves.pairs_mft.engine.risk import RiskState


class TestCheckStaleness(unittest.TestCase):
    def test_no_bar_yet_does_not_halt(self):
        risk = RiskState()
        check_staleness(risk, last_bar_at=None, now=pd.Timestamp.now(), max_staleness_seconds=60)
        self.assertFalse(risk.halted)

    def test_within_window_does_not_halt(self):
        risk = RiskState()
        last_bar_at = pd.Timestamp("2026-01-01 10:00:00")
        now = last_bar_at + pd.Timedelta(seconds=30)
        check_staleness(risk, last_bar_at, now, max_staleness_seconds=60)
        self.assertFalse(risk.halted)

    def test_exceeding_window_halts_with_reason(self):
        risk = RiskState()
        last_bar_at = pd.Timestamp("2026-01-01 10:00:00")
        now = last_bar_at + pd.Timedelta(seconds=120)
        check_staleness(risk, last_bar_at, now, max_staleness_seconds=60)
        self.assertTrue(risk.halted)
        self.assertIn("stale", risk.halt_reason)

    def test_already_halted_keeps_original_reason(self):
        risk = RiskState(halted=True, halt_reason="original reason")
        last_bar_at = pd.Timestamp("2026-01-01 10:00:00")
        now = last_bar_at + pd.Timedelta(seconds=999)
        check_staleness(risk, last_bar_at, now, max_staleness_seconds=60)
        self.assertEqual(risk.halt_reason, "original reason")


if __name__ == "__main__":
    unittest.main()
