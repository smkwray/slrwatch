from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from slr_watch.analytics.parent_transmission import run_parent_transmission_report


def _make_bank_panel() -> pd.DataFrame:
    """Tiny synthetic insured-bank panel with two banks under one parent."""
    rows = []
    for qe in ["2020-03-31", "2020-06-30", "2020-09-30"]:
        rows.append({
            "entity_id": "BANK_A",
            "entity_name": "Bank A",
            "rssd_id": "100",
            "top_parent_rssd": "900",
            "quarter_end": qe,
            "total_assets": 1_000_000,
            "ust_inventory_fv": 100_000,
            "ust_share_assets": 0.10,
            "balances_due_from_fed_share_assets": 0.05,
            "trading_assets_total_share_assets": 0.02,
            "headroom_pp": 2.5,
            "headroom_dollars": 25_000,
            "tier1_capital": 80_000,
            "total_leverage_exposure": 1_200_000,
            "is_covered_bank_subsidiary": True,
            "parent_method1_surcharge": 3.0,
        })
        rows.append({
            "entity_id": "BANK_B",
            "entity_name": "Bank B",
            "rssd_id": "101",
            "top_parent_rssd": "900",
            "quarter_end": qe,
            "total_assets": 500_000,
            "ust_inventory_fv": 40_000,
            "ust_share_assets": 0.08,
            "balances_due_from_fed_share_assets": 0.03,
            "trading_assets_total_share_assets": 0.01,
            "headroom_pp": 3.0,
            "headroom_dollars": 15_000,
            "tier1_capital": 40_000,
            "total_leverage_exposure": 600_000,
            "is_covered_bank_subsidiary": True,
            "parent_method1_surcharge": 3.0,
        })
    return pd.DataFrame(rows)


def _make_parent_panel() -> pd.DataFrame:
    """Tiny synthetic parent panel with one parent matching the banks above."""
    rows = []
    for qe in ["2020-03-31", "2020-06-30", "2020-09-30"]:
        rows.append({
            "entity_id": "PARENT_900",
            "entity_name": "Parent Corp",
            "rssd_id": "900",
            "top_parent_rssd": "900",
            "quarter_end": qe,
            "total_assets": 2_000_000,
            "ust_inventory_fv": 200_000,
            "ust_share_assets": 0.10,
            "trading_assets_total_share_assets": 0.04,
            "headroom_pp": 1.8,
            "headroom_dollars": 50_000,
            "tier1_capital": 150_000,
            "total_leverage_exposure": 2_500_000,
            "parent_method1_surcharge": 3.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_panels(tmp_path: Path) -> tuple[Path, Path]:
    bank_path = tmp_path / "bank_panel.parquet"
    parent_path = tmp_path / "parent_panel.parquet"
    _make_bank_panel().to_parquet(bank_path, index=False)
    _make_parent_panel().to_parquet(parent_path, index=False)
    return bank_path, parent_path


def test_report_produces_outputs(synthetic_panels: tuple[Path, Path], tmp_path: Path) -> None:
    bank_path, parent_path = synthetic_panels
    output_dir = tmp_path / "report"

    result = run_parent_transmission_report(bank_path, parent_path, output_dir)

    assert result == output_dir
    assert (output_dir / "linked_panel.csv").exists()
    assert (output_dir / "family_quarter_summary.csv").exists()
    assert (output_dir / "coverage_manifest.csv").exists()
    assert (output_dir / "coverage_manifest.md").exists()
    assert (output_dir / "summary.md").exists()


def test_linked_panel_non_empty(synthetic_panels: tuple[Path, Path], tmp_path: Path) -> None:
    bank_path, parent_path = synthetic_panels
    output_dir = tmp_path / "report"
    run_parent_transmission_report(bank_path, parent_path, output_dir)

    linked = pd.read_csv(output_dir / "linked_panel.csv")
    assert len(linked) > 0
    assert "entity_id_bank" in linked.columns
    assert "entity_id_parent" in linked.columns


def test_summary_contains_coverage_language(synthetic_panels: tuple[Path, Path], tmp_path: Path) -> None:
    bank_path, parent_path = synthetic_panels
    output_dir = tmp_path / "report"
    run_parent_transmission_report(bank_path, parent_path, output_dir)

    text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Coverage" in text or "coverage" in text
    assert "Linkage" in text or "linkage" in text or "Linked" in text or "linked" in text
    assert "quarter" in text.lower()


def test_family_summary_has_rows(synthetic_panels: tuple[Path, Path], tmp_path: Path) -> None:
    bank_path, parent_path = synthetic_panels
    output_dir = tmp_path / "report"
    run_parent_transmission_report(bank_path, parent_path, output_dir)

    summary = pd.read_csv(output_dir / "family_quarter_summary.csv")
    assert len(summary) > 0
    assert "n_banks" in summary.columns


def test_summary_mentions_directional(synthetic_panels: tuple[Path, Path], tmp_path: Path) -> None:
    bank_path, parent_path = synthetic_panels
    output_dir = tmp_path / "report"
    run_parent_transmission_report(bank_path, parent_path, output_dir)

    text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "direction" in text.lower() or "co-movement" in text.lower()


def test_coverage_manifest_has_linkage_flags(synthetic_panels: tuple[Path, Path], tmp_path: Path) -> None:
    bank_path, parent_path = synthetic_panels
    output_dir = tmp_path / "report"
    run_parent_transmission_report(bank_path, parent_path, output_dir)

    manifest = pd.read_csv(output_dir / "coverage_manifest.csv")
    assert "included_linked_sample" in manifest.columns
    assert manifest["included_linked_sample"].all()
