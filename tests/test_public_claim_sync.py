from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_public_copy_mentions_2019q4_baseline_nuance() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    assert "2019Q4 baseline" in readme


@pytest.mark.integration
def test_event_study_public_claims_match_live_outputs() -> None:
    root = _repo_root()
    reports = root / "output/reports/event_2020"
    if not (reports / "sample_ladder.csv").exists():
        pytest.fail("Integration requires local event-study outputs; rebuild the configured analysis first.")
    ladder = pd.read_csv(reports / "sample_ladder.csv").set_index("sample_name")
    clustered_dir = reports / "flagship_per_parent_clustered"
    clustered = pd.read_csv(clustered_dir / "did_results.csv")

    def did_value(treatment: str) -> pd.Series:
        return clustered[
            (clustered["treatment"] == treatment)
            & (clustered["outcome"] == "ust_inventory_fv_scaled")
        ].iloc[0]

    low_clustered = did_value("low_headroom_treated")
    covered_clustered = did_value("covered_bank_treated")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert f"{int(ladder.loc['universe_f_flagship_primary', 'parent_family_count'])} parent clusters" in readme
    normalized_readme = " ".join(readme.split()).replace("−", "-")
    assert f"{100 * low_clustered['coef']:.2f} percentage-point" in normalized_readme
    assert f"{100 * covered_clustered['coef']:.2f} percentage-point" in normalized_readme
    assert f"(p = {low_clustered['pvalue']:.3f})" in readme
    assert f"(p = {covered_clustered['pvalue']:.3f})" in readme

    placebo = pd.read_csv(clustered_dir / "placebo_fake_date.csv")
    treasury_placebos = placebo[
        placebo["outcome"].eq("ust_inventory_fv_scaled")
        & placebo["treatment"].isin(["low_headroom_treated", "covered_bank_treated"])
    ]
    assert len(treasury_placebos) == 6
    leave_one_out = pd.read_csv(clustered_dir / "leave_one_parent_out.csv")
    groups = leave_one_out.groupby(["treatment", "outcome"])
    assert len(groups) == 2
    for _, group in groups:
        assert group["coef"].min() <= 0 <= group["coef"].max()
