"""Cache price bars and register their source and licensing classification."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("PAIRS_DATA_DIR", "data"))
CACHE_SUBDIR = "pairs_mft_polygon"
IBKR_CACHE_SUBDIR = "pairs_mft_ibkr"
LICENSING_PATH = Path(os.environ.get("PAIRS_LICENSING_PATH", "data_licensing.json"))

_DEFAULT_NOTE = (
    "Auto-registered by sleeves/pairs_mft/validation/cache.py on write. "
    "Treated as research_only until the user confirms the Polygon subscription's "
    "terms cover live execution use and upgrades this entry "
    "to live_eligible himself."
)

_IBKR_NOTE = (
    "Auto-registered live_eligible by sleeves/pairs_mft/validation/cache.py on "
    "write. IBKR's own delayed data for this account's own trading "
    "uses an inherited automatic classification, whereas the Polygon helper "
    "requires a separate human classification upgrade."
)


def cache_path(ticker: str, multiplier: int, timespan: str, subdir: str = CACHE_SUBDIR) -> Path:
    return DATA_DIR / subdir / f"{ticker}_{multiplier}{timespan}.parquet"


def save_bars(
    ticker: str,
    multiplier: int,
    timespan: str,
    bars: pd.DataFrame,
    subdir: str = CACHE_SUBDIR,
    source: str = "Polygon.io",
    status: str = "unconfirmed",
    note: str = _DEFAULT_NOTE,
) -> Path:
    """Write bars to the cache and register their classification without replacing an existing entry."""
    path = cache_path(ticker, multiplier, timespan, subdir=subdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    _register(path, source=source, status=status, note=note)
    return path


def save_ibkr_bars(ticker: str, multiplier: int, timespan: str, bars: pd.DataFrame) -> Path:
    """Cache IBKR bars and register them as live_eligible without replacing an existing classification."""
    return save_bars(
        ticker, multiplier, timespan, bars, subdir=IBKR_CACHE_SUBDIR, source="IBKR", status="live_eligible", note=_IBKR_NOTE
    )


def load_cached_bars(
    ticker: str, multiplier: int, timespan: str, subdir: str = CACHE_SUBDIR
) -> pd.DataFrame | None:
    path = cache_path(ticker, multiplier, timespan, subdir=subdir)
    return pd.read_parquet(path) if path.exists() else None


def _register(path: Path, source: str, status: str, note: str) -> None:
    key = str(path.relative_to(DATA_DIR))
    registry = {"version": 1, "files": {}}
    if LICENSING_PATH.exists():
        registry = json.loads(LICENSING_PATH.read_text())

    if key in registry["files"]:
        return  # never clobber an existing (possibly human-reviewed) classification

    registry["files"][key] = {"status": status, "source": source, "note": note}
    LICENSING_PATH.write_text(json.dumps(registry, indent=2) + "\n")
