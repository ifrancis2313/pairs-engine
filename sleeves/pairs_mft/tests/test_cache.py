"""Test cache registration using generated bars and temporary files."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sleeves.pairs_mft.validation import cache


class TestCache(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp_dir.name) / "Data"
        self.licensing_path = Path(self._tmp_dir.name) / "data_licensing.json"
        self._patches = [
            patch.object(cache, "DATA_DIR", self.data_dir),
            patch.object(cache, "LICENSING_PATH", self.licensing_path),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._tmp_dir.cleanup)
        for p in self._patches:
            self.addCleanup(p.stop)

    def _bars(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100]},
            index=pd.to_datetime(["2026-01-01"]),
        )

    def test_save_bars_registers_unconfirmed_by_default(self):
        cache.save_bars("AAPL", 5, "minute", self._bars())
        registry = json.loads(self.licensing_path.read_text())
        entry = registry["files"]["pairs_mft_polygon/AAPL_5minute.parquet"]
        self.assertEqual(entry["status"], "unconfirmed")

    def test_save_ibkr_bars_registers_live_eligible_immediately(self):
        cache.save_ibkr_bars("AAPL", 5, "minute", self._bars())
        registry = json.loads(self.licensing_path.read_text())
        entry = registry["files"]["pairs_mft_ibkr/AAPL_5minute.parquet"]
        self.assertEqual(entry["status"], "live_eligible")
        self.assertEqual(entry["source"], "IBKR")

    def test_save_ibkr_bars_never_clobbers_existing_classification(self):
        path = cache.cache_path("AAPL", 5, "minute", subdir=cache.IBKR_CACHE_SUBDIR)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.licensing_path.write_text(
            json.dumps({"version": 1, "files": {"pairs_mft_ibkr/AAPL_5minute.parquet": {"status": "research_only"}}})
        )
        cache.save_ibkr_bars("AAPL", 5, "minute", self._bars())
        registry = json.loads(self.licensing_path.read_text())
        self.assertEqual(registry["files"]["pairs_mft_ibkr/AAPL_5minute.parquet"]["status"], "research_only")


if __name__ == "__main__":
    unittest.main()
