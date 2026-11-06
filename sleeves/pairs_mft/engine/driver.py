"""Consume bar events through one handler and dispatch resulting orders to a callback."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from sleeves.pairs_mft.contracts import PairDefinition
from sleeves.pairs_mft.engine.core import run_bar
from sleeves.pairs_mft.engine.events import BarEvent, Order
from sleeves.pairs_mft.engine.risk import RiskConfig
from sleeves.pairs_mft.engine.state import EngineState


def run(
    bar_source: Iterable[BarEvent],
    pair_defs: dict[str, PairDefinition],
    risk_config: RiskConfig,
    state: EngineState | None = None,
    on_orders: "callable[[BarEvent, list[Order]], None] | None" = None,
) -> EngineState:
    """Consume bar_source, update engine state, and invoke on_orders for each nonempty order list."""
    state = state if state is not None else EngineState()

    for event in bar_source:
        pair_def = pair_defs.get(event.pair_name)
        if pair_def is None:
            continue  # event for a pair this run isn't configured to trade - ignore, don't error
        state, orders = run_bar(state, event, pair_def, risk_config)
        if orders and on_orders is not None:
            on_orders(event, orders)

    return state
