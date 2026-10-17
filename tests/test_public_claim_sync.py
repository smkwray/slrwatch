from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


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

    family_row = family[family["policy_regime"] == "duration_loss_window"].iloc[0]
    family_match = _pct(family_row["matched_dominant_constraint_share"])
    family_both_duration = _pct(family_row["both_duration_loss_share"])

    site = (root / "site/index.html").read_text(encoding="utf-8")

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


@pytest.mark.integration
def test_event_study_public_claims_match_live_outputs() -> None:
    root = _repo_root()
    if not (root / "output/reports/event_2020/sample_ladder.csv").exists():
        pytest.fail("Integration requires local event-study outputs; rebuild the configured analysis first.")
    ladder = pd.read_csv(root / "output/reports/event_2020/sample_ladder.csv").set_index("sample_name")
    primary = pd.read_csv(root / "output/reports/event_2020/did_results.csv")
    clustered = pd.read_csv(root / "output/reports/event_2020/flagship_per_parent_clustered/did_results.csv")
    event_study = json.loads((root / "site/assets/data/event_study.json").read_text(encoding="utf-8"))

    def did_value(frame: pd.DataFrame, treatment: str, outcome: str) -> pd.Series:
        return frame[(frame["treatment"] == treatment) & (frame["outcome"] == outcome)].iloc[0]

    low_primary = did_value(primary, "low_headroom_treated", "ust_inventory_fv_scaled")
    covered_primary = did_value(primary, "covered_bank_treated", "ust_inventory_fv_scaled")
    low_clustered = did_value(clustered, "low_headroom_treated", "ust_inventory_fv_scaled")
    covered_clustered = did_value(clustered, "covered_bank_treated", "ust_inventory_fv_scaled")
    readme = (root / "README.md").read_text(encoding="utf-8")
    site = (root / "site/index.html").read_text(encoding="utf-8")

    assert f"{int(ladder.loc['universe_f_flagship_primary', 'parent_family_count'])} parent clusters" in readme
    normalized_readme = " ".join(readme.split()).replace("−", "-")
    assert f"{100 * low_clustered['coef']:.2f} percentage-point" in normalized_readme
    assert f"{100 * covered_clustered['coef']:.2f} percentage-point" in normalized_readme
    assert f"(p = {low_clustered['pvalue']:.3f})" in readme
    assert f"(p = {covered_clustered['pvalue']:.3f})" in readme

    assert f"{int(event_study['samples']['descriptive_universe']['entities']):,} insured-bank filers" in site
    assert f"{int(event_study['samples']['primary_core']['entities'])} fully balanced banks" in site
    assert f"{int(event_study['samples']['flagship_primary']['clusters'])} parent clusters" in site
    assert f"p&nbsp;=&nbsp;{low_primary['pvalue']:.3f}" in site
    assert f"p&nbsp;=&nbsp;{covered_primary['pvalue']:.3f}" in site
    assert f"p&nbsp;=&nbsp;{low_clustered['pvalue']:.3f}" in site
    assert f"p&nbsp;=&nbsp;{covered_clustered['pvalue']:.3f}" in site
    assert "2020 temporary SLR exclusion window" in site
    assert "2020&ndash;Q2 as the first treated Call Report quarter" in site
    assert "treatment roster" in site.lower()

    assert len(event_study["diagnostics"]["placebo_grid"]) == 6
    assert len(event_study["diagnostics"]["leave_one_parent_out"]) == 2
