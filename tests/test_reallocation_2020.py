from pathlib import Path

import pandas as pd

from slr_watch.analytics.reallocation_2020 import run_reallocation_report


def test_run_reallocation_report_smoke(tmp_path: Path):
    quarters = pd.date_range("2019-03-31", "2021-12-31", freq="QE-DEC")
    rows = []
    for entity_num in range(6):
        entity_id = f"bank_{entity_num}"
        for idx, quarter_end in enumerate(quarters):
            rows.append(
                {
                    "entity_id": entity_id,
                    "quarter_end": quarter_end.strftime("%Y-%m-%d"),
                    "headroom_pp": 0.01 + (entity_num * 0.002),
                    "ust_share_assets": 0.02 + (entity_num * 0.003),
                    "is_covered_bank_subsidiary": entity_num % 2 == 0,
                    "total_assets": 1000 + (entity_num * 100),
                    "ust_inventory_fv": 50 + idx + entity_num,
                    "balances_due_from_fed": 80 + idx,
                    "reverse_repos": 20 + entity_num,
                    "trading_assets_total": 10 + idx,
                    "deposits": 500 + (idx * 5) + entity_num,
                    "loans": 400 + (idx * 4) + entity_num,
                }
            )

    panel_path = tmp_path / "insured_bank_panel.parquet"
    pd.DataFrame(rows).to_parquet(panel_path, index=False)

    output_dir = run_reallocation_report(panel_path=panel_path, output_dir=tmp_path / "reallocation_2020")

    assert (output_dir / "prepared_panel.csv").exists()
    assert (output_dir / "reallocation_summary.csv").exists()
    assert (output_dir / "summary.md").exists()

    summary_frame = pd.read_csv(output_dir / "reallocation_summary.csv")
    assert not summary_frame.empty
    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Balance-Sheet Reallocation Under SLR" in summary_text
    assert "Largest net reallocations" in summary_text
