from pathlib import Path

import pandas as pd

from slr_watch.analytics.market_context import run_market_context_report


def test_run_market_context_report_smoke(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "quarter_end": ["2020-03-31", "2023-03-31", "2024-03-31"],
            "pd_ust_dealer_position_net_mn": [100.0, 120.0, 130.0],
            "pd_ust_repo_mn_weekly_avg": [200.0, 220.0, 260.0],
            "trace_total_par_value_bn": [10.0, 12.0, 15.0],
            "trace_total_trade_count": [pd.NA, 1100.0, 1200.0],
        }
    )
    panel_path = tmp_path / "market_overlay_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    output_dir = run_market_context_report(panel_path=panel_path, output_dir=tmp_path / "market_context")

    assert (output_dir / "prepared_panel.csv").exists()
    assert (output_dir / "common_overlap_panel.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "market_context_index.png").exists()

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Treasury Market Context" in summary
    assert "Latest quarter in common overlap" in summary
    assert "trade-count sub-sample" in summary
