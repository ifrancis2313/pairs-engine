# Pairs engine

A small event-driven pairs-research engine with rolling hedge ratios, spread signals, hedge-weighted sizing, timestamp ordering checks, and daily risk halts. It includes reusable cache and licensing-registry utilities and synthetic unit tests. No datasets or selected trading configurations are included.

## Quick start

Requires Python 3.13 or newer; tested with Python 3.13.1.

```sh
git clone <mirror-url> pairs-engine
cd pairs-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
python -c "from sleeves.pairs_mft.signals import next_position; print(next_position(3.0, 0, 0, 0, 2.0, 0.25))"
```

The synthetic example prints `(1, 1, -1)`: hold a long driver and short follower.

## How it is tested

24 tests use generated prices and temporary synthetic fixtures. They check timestamp rejection, warmup, paired order directions, concurrency limits, daily halt/reset behavior, closing existing positions during a halt, replay/polling agreement, stale-feed halts, cache registration, and fail-closed licensing checks. These tests establish software behavior only.

## Design notes

- A single bar handler consumes ordered events and emits two leg orders.
- Batch and event-driven paths share spread and signal functions.
- Risk halts block new entries while allowing ordinary exits.
- `PAIRS_DATA_DIR` selects the cache root (default `data`); `PAIRS_LICENSING_PATH` selects the registry (default `data_licensing.json`, relative to the working directory).

## Known limitations

`sleeves/pairs_mft/engine/core.py` computes signal and fill on the same bar (idealized zero-latency fill). Its accounting does not model realistic execution latency, transaction costs, or partial fills. Timestamp ordering alone does not establish realistic information availability. The halt does not force liquidation. Cache classification is metadata, not proof of a vendor license; the inherited IBKR cache helper assigns `live_eligible` automatically. Broker adapters, data ingestion, screening, and empirical strategy validation are outside this mirror's scope.

## License

MIT; see [LICENSE](LICENSE).
