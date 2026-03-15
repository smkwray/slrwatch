from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class EventStudySpec:
    outcome: str
    treatment: str
    entity_col: str = "entity_id"
    time_col: str = "quarter_end"
    shock_date: date = date(2020, 4, 1)
    leads: int = 4
    lags: int = 4


def add_event_time(frame: pd.DataFrame, *, time_col: str, shock_date: date) -> pd.DataFrame:
    out = frame.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    shock = pd.Timestamp(shock_date)
    out["event_quarter"] = (
        (out[time_col].dt.year - shock.year) * 4
        + ((out[time_col].dt.quarter - shock.quarter))
    )
    return out


def add_event_dummies(frame: pd.DataFrame, spec: EventStudySpec) -> pd.DataFrame:
    out = add_event_time(frame, time_col=spec.time_col, shock_date=spec.shock_date)
    for k in range(-spec.leads, spec.lags + 1):
        col = f"event_{k:+d}".replace("+", "p").replace("-", "m")
        out[col] = (out["event_quarter"] == k).astype(int)
    out["post"] = (out["event_quarter"] >= 0).astype(int)
    treatment = out[spec.treatment].fillna(0).astype(int)
    out["treated_post"] = treatment * out["post"]
    return out


def did_formula(spec: EventStudySpec) -> str:
    return (
        f"{spec.outcome} ~ treated_post + C({spec.entity_col}) + C({spec.time_col})"
    )


def event_study_terms(spec: EventStudySpec) -> list[str]:
    return [
        f"event_{k:+d}".replace("+", "p").replace("-", "m")
        for k in range(-spec.leads, spec.lags + 1)
        if k != -1
    ]
