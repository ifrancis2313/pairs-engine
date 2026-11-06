"""Test licensing enforcement against temporary synthetic registry entries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sleeves.pairs_mft.execution.data_licensing_gate import DataLicensingError, assert_live_eligible


class TestDataLicensingGate(unittest.TestCase):
    def _registry(self, files: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"version": 1, "files": files}, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_live_eligible_passes(self):
        path = self._registry({"foo.parquet": {"status": "live_eligible"}})
        with patch("sleeves.pairs_mft.execution.data_licensing_gate.LICENSING_PATH", path):
            assert_live_eligible("foo.parquet")  # must not raise

    def test_research_only_raises(self):
        path = self._registry({"foo.parquet": {"status": "research_only"}})
        with patch("sleeves.pairs_mft.execution.data_licensing_gate.LICENSING_PATH", path):
            with self.assertRaises(DataLicensingError):
                assert_live_eligible("foo.parquet")

    def test_unconfirmed_raises_not_treated_as_probably_fine(self):
        path = self._registry({"foo.parquet": {"status": "unconfirmed"}})
        with patch("sleeves.pairs_mft.execution.data_licensing_gate.LICENSING_PATH", path):
            with self.assertRaises(DataLicensingError):
                assert_live_eligible("foo.parquet")

    def test_unregistered_file_raises(self):
        path = self._registry({})
        with patch("sleeves.pairs_mft.execution.data_licensing_gate.LICENSING_PATH", path):
            with self.assertRaises(DataLicensingError):
                assert_live_eligible("never_seen.parquet")

    def test_missing_registry_file_raises(self):
        with patch("sleeves.pairs_mft.execution.data_licensing_gate.LICENSING_PATH", Path("/nonexistent/registry.json")):
            with self.assertRaises(DataLicensingError):
                assert_live_eligible("foo.parquet")


if __name__ == "__main__":
    unittest.main()
