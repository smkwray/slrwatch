from __future__ import annotations

import pandas as pd

from ..headroom import compute_headroom
from ..models import EntityRegulatoryProfile, EntityType
from ..rules import required_slr_breakdown


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def enrich_with_headroom(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        quarter_end = pd.to_datetime(row["quarter_end"]).date()
        profile = EntityRegulatoryProfile(
            entity_type=EntityType(row["entity_type"]),
            slr_applies=_to_bool(row.get("slr_applies", True)),
            is_gsib_parent=_to_bool(row.get("is_gsib_parent", False)),
            is_covered_bank_subsidiary=_to_bool(row.get("is_covered_bank_subsidiary", False)),
            parent_method1_surcharge=(
                None if pd.isna(row.get("parent_method1_surcharge")) else float(row.get("parent_method1_surcharge"))
            ),
            early_adopter=_to_bool(row.get("early_adopter", False)),
        )
        rule = required_slr_breakdown(as_of_date=quarter_end, profile=profile)
        result = compute_headroom(
            tier1_capital=float(row["tier1_capital"]),
            total_leverage_exposure=float(row["total_leverage_exposure"]),
            actual_slr=(
                None if pd.isna(row.get("actual_slr")) else float(row.get("actual_slr"))
            ),
            required_slr=rule.required_slr,
        )
        payload = dict(row)
        payload["rule_regime"] = rule.regime.value
        payload["required_slr"] = rule.required_slr
        payload["rule_notes"] = rule.notes
        payload["computed_actual_slr"] = result.actual_slr
        payload["headroom_pp"] = result.headroom_pp
        payload["headroom_dollars"] = result.headroom_dollars
        rows.append(payload)
    return pd.DataFrame(rows)
