from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def _pp(value: float) -> str:
    return f"{100 * float(value):+.2f}pp"


def test_constraint_decomposition_public_claims_match_live_outputs() -> None:
    root = _repo_root()
    if not (root / "output/reports/constraint_decomposition/regime_summary.csv").exists():
        pytest.skip("Local generated constraint-decomposition outputs are not committed in clean clones.")
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


def test_event_study_public_claims_match_live_outputs() -> None:
    root = _repo_root()
    if not (root / "output/reports/event_2020/sample_ladder.csv").exists():
        pytest.skip("Local generated event-study outputs are not committed in clean clones.")
    ladder = pd.read_csv(root / "output/reports/event_2020/sample_ladder.csv").set_index("sample_name")
    primary = pd.read_csv(root / "output/reports/event_2020/did_results.csv")
    clustered = pd.read_csv(root / "output/reports/event_2020/flagship_per_parent_clustered/did_results.csv")
    pretrend = pd.read_csv(root / "output/reports/event_2020/pretrend_checks.csv")
    clustered_pretrend = pd.read_csv(root / "output/reports/event_2020/flagship_per_parent_clustered/pretrend_checks.csv")
    placebo = pd.read_csv(root / "output/reports/event_2020/flagship_per_parent_clustered/placebo_fake_date.csv")
    event_study = json.loads((root / "site/assets/data/event_study.json").read_text(encoding="utf-8"))

    def did_value(frame: pd.DataFrame, treatment: str, outcome: str) -> pd.Series:
        return frame[(frame["treatment"] == treatment) & (frame["outcome"] == outcome)].iloc[0]

    def pretrend_value(frame: pd.DataFrame, treatment: str, outcome: str) -> pd.Series:
        return frame[(frame["treatment"] == treatment) & (frame["outcome"] == outcome)].iloc[0]

    low_primary = did_value(primary, "low_headroom_treated", "ust_inventory_fv_scaled")
    covered_primary = did_value(primary, "covered_bank_treated", "ust_inventory_fv_scaled")
    low_clustered = did_value(clustered, "low_headroom_treated", "ust_inventory_fv_scaled")
    covered_clustered = did_value(clustered, "covered_bank_treated", "ust_inventory_fv_scaled")
    low_pretrend = pretrend_value(pretrend, "low_headroom_treated", "ust_inventory_fv_scaled")
    covered_pretrend = pretrend_value(pretrend, "covered_bank_treated", "ust_inventory_fv_scaled")
    covered_placebo_ps = placebo[
        (placebo["treatment"] == "covered_bank_treated") & (placebo["outcome"] == "ust_inventory_fv_scaled")
    ]["pvalue"].tolist()

    readme = (root / "README.md").read_text(encoding="utf-8")
    site = (root / "site/index.html").read_text(encoding="utf-8")

    assert f"{int(ladder.loc['universe_a_all_insured_banks', 'entity_count']):,} insured-bank filers" in readme
    assert f"{int(ladder.loc['universe_d_primary_core', 'entity_count'])} balanced-coverage entities" in readme
    assert (
        f"{int(ladder.loc['universe_d_covered_bank_primary', 'entity_count'])} entities / "
        f"{int(ladder.loc['universe_d_covered_bank_primary', 'observation_count'])} observations"
    ) in readme
    assert f"{int(ladder.loc['universe_f_flagship_primary', 'parent_family_count'])} parent clusters" in readme
    assert f"Low-headroom Treasury result: {_pp(low_primary['coef'])} (p = {low_primary['pvalue']:.3f})" in readme
    assert f"Covered-bank / direct-eligibility Treasury result: {_pp(covered_primary['coef'])} (p = {covered_primary['pvalue']:.3f})" in readme
    assert f"Low headroom: {_pp(low_clustered['coef'])} (p = {low_clustered['pvalue']:.3f})" in readme
    assert f"Covered bank / direct eligibility: {_pp(covered_clustered['coef'])} (p = {covered_clustered['pvalue']:.3f})" in readme
    assert f"{low_pretrend['pretrend_joint_pvalue']:.3f}" in readme
    assert f"{covered_pretrend['pretrend_joint_pvalue']:.3f}" in readme
    for value in covered_placebo_ps:
        assert f"{value:.3f}" in readme

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
