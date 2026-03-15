from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bank_fixture(tmp_path: Path) -> Path:
    rows = []
    for entity_id in ["B1", "B2"]:
        for quarter_end, ust, fed, trading, reverse_repo, headroom in [
            ("2019-12-31", 0.08, 0.12, 0.03, 0.02, 2.0),
            ("2020-06-30", 0.10, 0.15, 0.02, 0.03, 1.7),
            ("2021-12-31", 0.09, 0.11, 0.02, 0.02, 1.9),
        ]:
            rows.append(
                {
                    "entity_id": entity_id,
                    "quarter_end": quarter_end,
                    "ust_share_assets": ust,
                    "balances_due_from_fed_share_assets": fed,
                    "trading_assets_total_share_assets": trading,
                    "reverse_repos_share_assets": reverse_repo,
                    "headroom_pp": headroom,
                }
            )
    path = tmp_path / "bank.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _parent_fixture(tmp_path: Path) -> Path:
    rows = []
    for entity_id in ["P1", "P2"]:
        for quarter_end, ust, fed, trading, reverse_repo, headroom in [
            ("2020-12-31", 0.07, 0.10, 0.14, 0.02, 1.5),
            ("2021-12-31", 0.08, 0.09, 0.13, 0.02, 1.6),
            ("2022-06-30", 0.09, 0.08, 0.11, 0.03, 1.8),
            ("2023-12-31", 0.10, 0.07, 0.10, 0.03, 1.9),
        ]:
            rows.append(
                {
                    "entity_id": entity_id,
                    "quarter_end": quarter_end,
                    "ust_share_assets": ust,
                    "balances_due_from_fed_share_assets": fed,
                    "trading_assets_total_share_assets": trading,
                    "reverse_repos_share_assets": reverse_repo,
                    "headroom_pp": headroom,
                }
            )
    path = tmp_path / "parent.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _market_fixture(tmp_path: Path) -> Path:
    rows = [
        {"quarter_end": "2019-12-31", "pd_ust_dealer_position_net_mn": 100.0, "pd_ust_repo_mn_weekly_avg": 500.0, "pd_ust_reverse_repo_mn_weekly_avg": 300.0, "trace_total_par_value_bn": 20.0},
        {"quarter_end": "2020-06-30", "pd_ust_dealer_position_net_mn": 140.0, "pd_ust_repo_mn_weekly_avg": 620.0, "pd_ust_reverse_repo_mn_weekly_avg": 320.0, "trace_total_par_value_bn": 23.0},
        {"quarter_end": "2020-12-31", "pd_ust_dealer_position_net_mn": 150.0, "pd_ust_repo_mn_weekly_avg": 640.0, "pd_ust_reverse_repo_mn_weekly_avg": 330.0, "trace_total_par_value_bn": 24.0},
        {"quarter_end": "2021-12-31", "pd_ust_dealer_position_net_mn": 145.0, "pd_ust_repo_mn_weekly_avg": 610.0, "pd_ust_reverse_repo_mn_weekly_avg": 325.0, "trace_total_par_value_bn": 25.0},
        {"quarter_end": "2022-06-30", "pd_ust_dealer_position_net_mn": 170.0, "pd_ust_repo_mn_weekly_avg": 700.0, "pd_ust_reverse_repo_mn_weekly_avg": 350.0, "trace_total_par_value_bn": 27.0},
        {"quarter_end": "2023-12-31", "pd_ust_dealer_position_net_mn": 190.0, "pd_ust_repo_mn_weekly_avg": 730.0, "pd_ust_reverse_repo_mn_weekly_avg": 360.0, "trace_total_par_value_bn": 29.0},
    ]
    path = tmp_path / "market.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_policy_regime_panel_report(tmp_path: Path) -> None:
    from slr_watch.analytics.policy_regime_panel import run_policy_regime_panel_report

    output_dir = tmp_path / "output"
    result = run_policy_regime_panel_report(
        bank_panel_path=_bank_fixture(tmp_path),
        parent_panel_path=_parent_fixture(tmp_path),
        market_panel_path=_market_fixture(tmp_path),
        output_dir=output_dir,
    )

    assert result == output_dir
    assert (output_dir / "regime_quarter_panel.csv").exists()
    assert (output_dir / "regime_summary.csv").exists()
    assert (output_dir / "summary.md").exists()

    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Policy-Regime" in summary_text
    assert "temporary_exclusion" in summary_text
    assert "qt_era" in summary_text

    regime_summary = pd.read_csv(output_dir / "regime_summary.csv")
    assert not regime_summary.empty
    assert set(regime_summary["policy_regime"]).issuperset({"pre_exclusion", "temporary_exclusion", "qt_era"})
