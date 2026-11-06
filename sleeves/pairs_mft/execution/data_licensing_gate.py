"""Require an explicit live_eligible classification in the configured licensing registry."""

from __future__ import annotations

import json
import os
from pathlib import Path

LICENSING_PATH = Path(os.environ.get("PAIRS_LICENSING_PATH", "data_licensing.json"))


class DataLicensingError(RuntimeError):
    pass


def assert_live_eligible(relative_path: str) -> None:
    """Raise DataLicensingError unless relative_path is explicitly classified live_eligible."""
    if not LICENSING_PATH.exists():
        raise DataLicensingError(f"{LICENSING_PATH} does not exist - no file can be live_eligible")

    registry = json.loads(LICENSING_PATH.read_text())
    entry = registry.get("files", {}).get(relative_path)
    if entry is None:
        raise DataLicensingError(f"{relative_path!r} is not registered in {LICENSING_PATH} at all - cannot be live_eligible")

    status = entry.get("status")
    if status != "live_eligible":
        raise DataLicensingError(
            f"{relative_path!r} is classified {status!r}, not 'live_eligible' - "
            "this data source may not feed a live/paper execution path. "
            f"Confirm the license and upgrade it in {LICENSING_PATH} yourself before running live."
        )
