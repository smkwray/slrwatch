from __future__ import annotations

from pathlib import Path

import pandas as pd


def _make_bank_fixture(tmp_path: Path) -> Path:
    rows = []
    entities = [
        {"entity_id": "E1", "entity_name": "Bank A", "top_parent_rssd": "P1", "is_covered_bank_subsidiary": True},
        {"entity_id": "E2", "entity_name": "Bank B", "top_parent_rssd": "P2", "is_covered_bank_subsidiary": False},
    ]
    quarters = ["2019-12-31", "2020-03-31", "2020-06-30", "2020-12-31"]
    for entity in entities:
        for i, quarter in enumerate(quarters):
            direction = 1 if entity["entity_id"] == "E1" else -1
            rows.append(
                {
                    **entity,
                    "quarter_end": quarter,
                    "total_assets": 1000.0,
                    "ust_inventory_fv": 100.0 + direction * i * 12.0,
                    "balances_due_from_fed": 200.0 + direction * i * 8.0,
                    "reverse_repos": 50.0 + direction * i * 7.0,
                    "trading_assets_total": 40.0 + direction * i * 5.0,
                    "deposits": 500.0 + i * 20.0,
                    "loans": 400.0 + i * 10.0,
                    "tier1_capital": 80.0,
                    "total_leverage_exposure": 1200.0,
                    "headroom_pp": 1.5 if entity["entity_id"] == "E1" else 3.0,
                    "ust_share_assets": 0.10 if entity["entity_id"] == "E1" else 0.05,
                    "actual_slr": 0.067,
                }
            )
    path = tmp_path / "insured_bank_panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_market_fixture(tmp_path: Path) -> Path:
    rows = [
        {
            "quarter_end": "2019-12-31",
            "pd_ust_dealer_position_net_mn": 100.0,
            "pd_ust_repo_mn_weekly_avg": 500.0,
            "pd_ust_reverse_repo_mn_weekly_avg": 300.0,
            "trace_total_par_value_bn": 20.0,
        },
        {
            "quarter_end": "2020-03-31",
            "pd_ust_dealer_position_net_mn": 110.0,
            "pd_ust_repo_mn_weekly_avg": 520.0,
            "pd_ust_reverse_repo_mn_weekly_avg": 290.0,
            "trace_total_par_value_bn": 21.0,
        },
        {
            "quarter_end": "2020-06-30",
            "pd_ust_dealer_position_net_mn": 150.0,
            "pd_ust_repo_mn_weekly_avg": 600.0,
            "pd_ust_reverse_repo_mn_weekly_avg": 320.0,
            "trace_total_par_value_bn": 25.0,
        },
        {
            "quarter_end": "2020-12-31",
            "pd_ust_dealer_position_net_mn": 170.0,
            "pd_ust_repo_mn_weekly_avg": 650.0,
            "pd_ust_reverse_repo_mn_weekly_avg": 340.0,
            "trace_total_par_value_bn": 27.0,
        },
    ]
    path = tmp_path / "market_overlay_panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_treasury_intermediation_report(tmp_path: Path) -> None:
    from slr_watch.analytics.treasury_intermediation import run_treasury_intermediation_report

    bank_panel_path = _make_bank_fixture(tmp_path)
    market_panel_path = _make_market_fixture(tmp_path)
    output_dir = tmp_path / "output"

    result = run_treasury_intermediation_report(
        panel_path=bank_panel_path,
        market_panel_path=market_panel_path,
        output_dir=output_dir,
    )

    assert result == output_dir
    assert (output_dir / "prepared_panel.csv").exists()
    assert (output_dir / "intermediation_summary.csv").exists()
    assert (output_dir / "market_linkage_summary.csv").exists()
    assert (output_dir / "summary.md").exists()

    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Treasury Intermediation" in summary_text
    assert "repo" in summary_text.lower() or "trading" in summary_text.lower()

    summary_frame = pd.read_csv(output_dir / "intermediation_summary.csv")
    assert not summary_frame.empty
    assert set(summary_frame["outcome"]).issuperset({"reverse_repos_scaled", "trading_assets_scaled"})

    linkage_frame = pd.read_csv(output_dir / "market_linkage_summary.csv")
    assert not linkage_frame.empty
    assert "gap_market_correlation" in linkage_frame.columns
