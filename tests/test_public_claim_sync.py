from __future__ import annotations

from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def test_constraint_decomposition_public_claims_match_live_outputs() -> None:
    root = _repo_root()
    regime = pd.read_csv(root / "output/reports/constraint_decomposition/regime_summary.csv")
    family = pd.read_csv(root / "output/reports/constraint_decomposition/family_alignment_summary.csv")

    def regime_value(entity_source: str, policy_regime: str, column: str) -> str:
        row = regime[
            (regime["entity_source"] == entity_source)
            & (regime["policy_regime"] == policy_regime)
        ].iloc[0]
        return _pct(row[column])

    insured_duration = regime_value("insured_bank", "duration_loss_window", "duration_loss_dominant_share")
    parent_duration = regime_value("parent_or_ihc", "duration_loss_window", "duration_loss_dominant_share")
    insured_late_duration = regime_value("insured_bank", "late_qt_normalization", "duration_loss_dominant_share")
    parent_late_leverage = regime_value("parent_or_ihc", "late_qt_normalization", "leverage_dominant_share")

    family_row = family[family["policy_regime"] == "duration_loss_window"].iloc[0]
    family_match = _pct(family_row["matched_dominant_constraint_share"])
    family_both_duration = _pct(family_row["both_duration_loss_share"])

    readme = (root / "README.md").read_text(encoding="utf-8")
    site = (root / "site/index.html").read_text(encoding="utf-8")

    assert (
        f"In the 2022-2023 duration-loss window, duration loss is the dominant bucket for insured banks in "
        f"{insured_duration} of observations and for parents/IHCs in {parent_duration}."
    ) in readme
    assert (
        f"By late QT normalization, insured banks still lean duration loss at {insured_late_duration}, "
        f"while parents/IHCs now lean back toward leverage at {parent_late_leverage}."
    ) in readme

    for needle in [
        insured_duration,
        parent_duration,
        family_match,
        family_both_duration,
        f"insured banks ({insured_duration}) and parents/IHCs ({parent_duration})",
        f"Linked families match on the dominant constraint in {family_match}",
        f"duration-loss dominated in {family_both_duration} of those observations.",
    ]:
        assert needle in site
