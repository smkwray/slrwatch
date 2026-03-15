from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HeadroomResult:
    actual_slr: float
    required_slr: float
    headroom_pp: float
    headroom_dollars: float


def compute_actual_slr(
    tier1_capital: float,
    total_leverage_exposure: float,
) -> float:
    if total_leverage_exposure <= 0:
        raise ValueError("total_leverage_exposure must be positive")
    return tier1_capital / total_leverage_exposure


def headroom_pp(actual_slr: float, required_slr: float) -> float:
    return actual_slr - required_slr


def headroom_dollars(
    tier1_capital: float,
    required_slr: float,
    total_leverage_exposure: float,
) -> float:
    if required_slr <= 0:
        raise ValueError("required_slr must be positive")
    return (tier1_capital / required_slr) - total_leverage_exposure


def compute_headroom(
    *,
    tier1_capital: float,
    total_leverage_exposure: float,
    required_slr: float,
    actual_slr: Optional[float] = None,
) -> HeadroomResult:
    realized_slr = (
        actual_slr
        if actual_slr is not None
        else compute_actual_slr(
            tier1_capital=tier1_capital,
            total_leverage_exposure=total_leverage_exposure,
        )
    )
    return HeadroomResult(
        actual_slr=realized_slr,
        required_slr=required_slr,
        headroom_pp=headroom_pp(realized_slr, required_slr),
        headroom_dollars=headroom_dollars(
            tier1_capital=tier1_capital,
            required_slr=required_slr,
            total_leverage_exposure=total_leverage_exposure,
        ),
    )


def treasury_inventory_fair_value(
    *,
    htm_fair_value: float = 0.0,
    afs_fair_value: float = 0.0,
    trading_ust: float = 0.0,
) -> float:
    return htm_fair_value + afs_fair_value + trading_ust


def treasury_share_of_assets(ust_inventory_fv: float, total_assets: float) -> float:
    if total_assets <= 0:
        raise ValueError("total_assets must be positive")
    return ust_inventory_fv / total_assets


def treasury_share_of_headroom(ust_inventory_fv: float, headroom_dollars_value: float) -> float:
    if headroom_dollars_value == 0:
        raise ValueError("headroom_dollars_value must be non-zero")
    return ust_inventory_fv / headroom_dollars_value
