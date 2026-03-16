from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bank_fixture(tmp_path: Path) -> Path:
    rows = []
    for entity_id, top_parent_rssd, base_headroom, base_loss, base_liquidity in [
        ("B1", "9001", 0.018, 0.002, 0.20),
        ("B2", "9002", 0.010, 0.010, 0.12),
    ]:
        for quarter_end, ust_share, deposit_growth in [
            ("2019-12-31", 0.05, 0.03),
            ("2020-12-31", 0.07, -0.02),
            ("2021-12-31", 0.06, 0.01),
        ]:
            rows.append(
                {
                    "entity_id": entity_id,
                    "top_parent_rssd": top_parent_rssd,
                    "quarter_end": quarter_end,
                    "headroom_pp": base_headroom,
                    "ust_share_assets": ust_share,
                    "total_unrealized_loss_tier1": base_loss,
                    "liquid_asset_share_assets": base_liquidity,
                    "deposit_growth_qoq": deposit_growth,
                }
            )
    path = tmp_path / "bank.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _parent_fixture(tmp_path: Path) -> Path:
    rows = []
    for entity_id, top_parent_rssd, headroom_series, loss_series, liquidity_series, deposit_growth_series in [
        (
            "P1",
            "9001",
            [0.015, 0.013, 0.011, 0.010],
            [0.001, 0.006, 0.020, 0.018],
            [0.18, 0.16, 0.13, 0.14],
            [0.02, 0.00, -0.01, 0.01],
        ),
        (
            "P2",
            "9002",
            [0.008, 0.007, 0.006, 0.009],
            [0.003, 0.008, 0.012, 0.010],
            [0.12, 0.10, 0.08, 0.09],
            [0.01, -0.02, -0.04, -0.01],
        ),
    ]:
        for idx, quarter_end in enumerate(["2020-12-31", "2021-12-31", "2022-06-30", "2024-03-31"]):
            rows.append(
                {
                    "entity_id": entity_id,
                    "top_parent_rssd": top_parent_rssd,
                    "quarter_end": quarter_end,
                    "headroom_pp": headroom_series[idx],
                    "ust_share_assets": 0.06 + (idx * 0.01),
                    "total_unrealized_loss_tier1": loss_series[idx],
                    "liquid_asset_share_assets": liquidity_series[idx],
                    "deposit_growth_qoq": deposit_growth_series[idx],
                }
            )
    path = tmp_path / "parent.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_constraint_decomposition_report(tmp_path: Path) -> None:
    from slr_watch.analytics.constraint_decomposition import run_constraint_decomposition_report

    output_dir = tmp_path / "output"
    result = run_constraint_decomposition_report(
        bank_panel_path=_bank_fixture(tmp_path),
        parent_panel_path=_parent_fixture(tmp_path),
        output_dir=output_dir,
    )

    assert result == output_dir
    assert (output_dir / "prepared_panel.csv").exists()
    assert (output_dir / "coverage_summary.csv").exists()
    assert (output_dir / "regime_summary.csv").exists()
    assert (output_dir / "absorption_summary.csv").exists()
    assert (output_dir / "regime_comparison.csv").exists()
    assert (output_dir / "family_alignment_summary.csv").exists()
    assert (output_dir / "interaction_regime_summary.csv").exists()
    assert (output_dir / "summary.md").exists()

    prepared = pd.read_csv(output_dir / "prepared_panel.csv")
    assert "dominant_constraint" in prepared.columns
    assert "funding_stress_score" in prepared.columns
    assert "liquidity_stress_score" in prepared.columns
    assert set(prepared["entity_source"]) == {"insured_bank", "parent_or_ihc"}

    coverage = pd.read_csv(output_dir / "coverage_summary.csv")
    assert set(coverage["entity_source"]) == {"insured_bank", "parent_or_ihc"}

    regime_summary = pd.read_csv(output_dir / "regime_summary.csv")
    assert "duration_loss_window" in set(regime_summary["policy_regime"])
    assert "leverage_dominant_share" in regime_summary.columns
    assert "duration_loss_dominant_share" in regime_summary.columns
    assert "funding_dominant_share" in regime_summary.columns
    assert "mean_funding_stress_score" in regime_summary.columns
    assert "mean_liquidity_stress_score" in regime_summary.columns

    absorption_summary = pd.read_csv(output_dir / "absorption_summary.csv")
    assert "mean_ust_share_assets_qoq" in absorption_summary.columns
    assert "mean_safe_asset_buffer_share_assets" in absorption_summary.columns

    regime_comparison = pd.read_csv(output_dir / "regime_comparison.csv")
    assert "duration_loss_dominant_share_gap" in regime_comparison.columns
    assert "insured_dominant_constraint" in regime_comparison.columns
    assert "parent_dominant_constraint" in regime_comparison.columns

    family_alignment = pd.read_csv(output_dir / "family_alignment_summary.csv")
    assert "matched_dominant_constraint_share" in family_alignment.columns
    assert "both_duration_loss_share" in family_alignment.columns

    interaction_summary = pd.read_csv(output_dir / "interaction_regime_summary.csv")
    assert "focus_regime" in interaction_summary.columns
    assert "outcome" in interaction_summary.columns
    assert "status" in interaction_summary.columns

    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Constraint Decomposition Beyond SLR" in summary_text
    assert "insured-bank panel currently ends at 2021-12-31" in summary_text
    assert "duration_loss_window" in summary_text
    assert "Cross-panel regime comparison" in summary_text
    assert "Treasury absorption by dominant constraint" in summary_text
    assert "Parent-bank family alignment" in summary_text
    assert "Interaction regressions" in summary_text
