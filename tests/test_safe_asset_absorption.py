from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _make_fixture(tmp_path: Path) -> Path:
    """Build a tiny insured-bank style panel with two entities and four quarters."""
    rows = []
    entities = [
        {"entity_id": "E1", "entity_name": "Bank A", "top_parent_rssd": "P1",
         "is_covered_bank_subsidiary": True},
        {"entity_id": "E2", "entity_name": "Bank B", "top_parent_rssd": "P2",
         "is_covered_bank_subsidiary": False},
    ]
    quarters = ["2019-12-31", "2020-03-31", "2020-06-30", "2020-12-31"]
    for entity in entities:
        for i, q in enumerate(quarters):
            rows.append(
                {
                    **entity,
                    "quarter_end": q,
                    "total_assets": 1000.0,
                    "ust_inventory_fv": 100.0 + i * 10 * (1 if entity["entity_id"] == "E1" else -1),
                    "balances_due_from_fed": 200.0 - i * 5 * (1 if entity["entity_id"] == "E1" else -1),
                    "reverse_repos": 50.0,
                    "trading_assets_total": 30.0,
                    "deposits": 500.0 + i * 20,
                    "loans": 400.0 + i * 10,
                    "tier1_capital": 80.0,
                    "total_leverage_exposure": 1200.0,
                    "headroom_pp": 1.5 if entity["entity_id"] == "E1" else 3.0,
                    "ust_share_assets": 0.10 if entity["entity_id"] == "E1" else 0.05,
                    "actual_slr": 0.067,
                }
            )
    panel = pd.DataFrame(rows)
    path = tmp_path / "insured_bank_panel.csv"
    panel.to_csv(path, index=False)
    return path


def test_run_absorption_report(tmp_path: Path) -> None:
    from slr_watch.analytics.safe_asset_absorption import run_absorption_report

    panel_path = _make_fixture(tmp_path)
    output_dir = tmp_path / "output"

    result = run_absorption_report(panel_path=panel_path, output_dir=output_dir)

    assert result == output_dir
    assert (output_dir / "prepared_panel.csv").exists()
    assert (output_dir / "absorption_summary.csv").exists()
    assert (output_dir / "summary.md").exists()

    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert len(summary_text) > 0
    assert "Treasury" in summary_text
    assert "reserve" in summary_text.lower() or "Reserve" in summary_text or "Fed" in summary_text

    absorption = pd.read_csv(output_dir / "absorption_summary.csv")
    assert not absorption.empty

    prepared = pd.read_csv(output_dir / "prepared_panel.csv")
    assert "ust_inventory_fv_scaled" in prepared.columns
    assert "balances_due_from_fed_scaled" in prepared.columns
    assert "treasury_share_of_safe" in prepared.columns
