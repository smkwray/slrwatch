from pathlib import Path

import pandas as pd
import pytest

from slr_watch.panels import build_crosswalk, build_insured_bank_panel, build_parent_panel


def test_build_crosswalk_rejects_duplicate_entity_ids(tmp_path: Path):
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "entity_id,entity_name,entity_type,rssd_id,fdic_cert,top_parent_rssd,country,is_gsib_parent,is_covered_bank_subsidiary,fr_y15_reporter\n"
        "dup,One,insured_bank_sub,1001,,,United States,false,true,\n"
        "dup,Two,insured_bank_sub,1002,,,United States,false,false,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="entity_id"):
        build_crosswalk(universe, output_path=tmp_path / "crosswalk.parquet")


def test_build_insured_bank_panel(tmp_path: Path):
    staged = pd.DataFrame(
        [
            {
                "rssd_id": "1001",
                "quarter_end": "2020-03-31",
                "tier1_capital": 55.0,
                "total_leverage_exposure": 1000.0,
                "ust_htm_fair_value": 10.0,
                "ust_afs_fair_value": 20.0,
                "ust_trading_assets": 5.0,
                "balances_due_from_fed": 300.0,
                "reverse_repos": 40.0,
                "trading_assets_total": 60.0,
                "total_assets": 2000.0,
                "deposits": 1500.0,
                "loans": 800.0,
            }
        ]
    )
    staged_path = tmp_path / "call_reports_normalized.parquet"
    staged.to_parquet(staged_path, index=False)

    crosswalk = pd.DataFrame(
        [
            {
                "entity_id": "bank_1001",
                "entity_name": "Demo Bank",
                "entity_type": "insured_bank_sub",
                "rssd_id": "1001",
                "fdic_cert": "",
                "top_parent_rssd": "",
                "country": "United States",
                "is_gsib_parent": False,
                "is_covered_bank_subsidiary": True,
                "fr_y15_reporter": "",
                "parent_method1_surcharge": None,
            }
        ]
    )
    crosswalk_path = tmp_path / "crosswalk.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)

    output_path = build_insured_bank_panel(staged_path, crosswalk_path, output_path=tmp_path / "insured.parquet")
    frame = pd.read_parquet(output_path)

    assert frame.loc[0, "entity_id"] == "bank_1001"
    assert round(frame.loc[0, "headroom_pp"], 6) == -0.005
    assert frame.loc[0, "ust_inventory_fv"] == 35.0


def test_build_parent_panel_with_fry15_overlay(tmp_path: Path):
    staged = pd.DataFrame(
        [
            {
                "rssd_id": "2001",
                "quarter_end": "2026-06-30",
                "tier1_capital": 120.0,
                "total_leverage_exposure": 2400.0,
                "ust_htm_fair_value": 10.0,
                "ust_afs_fair_value": 15.0,
                "ust_trading_assets": 5.0,
                "total_assets": 3000.0,
                "balances_due_from_fed": 200.0,
                "reverse_repos": 25.0,
                "trading_assets_total": 50.0,
                "deposits": 1500.0,
                "loans": 900.0,
            }
        ]
    )
    staged_path = tmp_path / "fry9c_normalized.parquet"
    staged.to_parquet(staged_path, index=False)

    crosswalk = pd.DataFrame(
        [
            {
                "entity_id": "parent_2001",
                "entity_name": "Demo Parent",
                "entity_type": "bhc_parent",
                "rssd_id": "2001",
                "fdic_cert": "",
                "top_parent_rssd": "",
                "country": "United States",
                "is_gsib_parent": True,
                "is_covered_bank_subsidiary": False,
                "fr_y15_reporter": "DEMO_PARENT",
                "parent_method1_surcharge": None,
            }
        ]
    )
    crosswalk_path = tmp_path / "crosswalk.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)

    overlay = pd.DataFrame(
        [
            {
                "fr_y15_reporter": "DEMO_PARENT",
                "quarter_end": "2026-06-30",
                "parent_method1_surcharge": 0.045,
            }
        ]
    )
    overlay_path = tmp_path / "fry15_overlay.csv"
    overlay.to_csv(overlay_path, index=False)

    output_path = build_parent_panel(
        staged_path,
        crosswalk_path,
        fry15_path=overlay_path,
        output_path=tmp_path / "parent.parquet",
    )
    frame = pd.read_parquet(output_path)

    assert frame.loc[0, "entity_id"] == "parent_2001"
    assert frame.loc[0, "required_slr"] == 0.0525


def test_build_parent_panel_with_fry15_overlay_carries_forward_latest_quarter(tmp_path: Path):
    staged = pd.DataFrame(
        [
            {
                "rssd_id": "2001",
                "quarter_end": "2026-06-30",
                "tier1_capital": 120.0,
                "total_leverage_exposure": 2400.0,
                "total_assets": 3000.0,
            }
        ]
    )
    staged_path = tmp_path / "fry9c_normalized.parquet"
    staged.to_parquet(staged_path, index=False)

    crosswalk = pd.DataFrame(
        [
            {
                "entity_id": "parent_2001",
                "entity_name": "Demo Parent",
                "entity_type": "bhc_parent",
                "rssd_id": "2001",
                "fdic_cert": "",
                "top_parent_rssd": "",
                "country": "United States",
                "is_gsib_parent": True,
                "is_covered_bank_subsidiary": False,
                "fr_y15_reporter": "DEMO_PARENT",
                "parent_method1_surcharge": None,
            }
        ]
    )
    crosswalk_path = tmp_path / "crosswalk.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)

    overlay = pd.DataFrame(
        [
            {
                "fr_y15_reporter": "DEMO_PARENT",
                "quarter_end": "2025-12-31",
                "parent_method1_surcharge": 0.045,
            }
        ]
    )
    overlay_path = tmp_path / "fry15_overlay.csv"
    overlay.to_csv(overlay_path, index=False)

    output_path = build_parent_panel(
        staged_path,
        crosswalk_path,
        fry15_path=overlay_path,
        output_path=tmp_path / "parent.parquet",
    )
    frame = pd.read_parquet(output_path)

    assert frame.loc[0, "parent_method1_surcharge"] == 0.045
    assert frame.loc[0, "required_slr"] == 0.0525


def test_build_parent_panel_drops_nonpositive_leverage_exposure(tmp_path: Path):
    staged = pd.DataFrame(
        [
            {
                "rssd_id": "2001",
                "quarter_end": "2026-06-30",
                "tier1_capital": 120.0,
                "total_leverage_exposure": 0.0,
                "total_assets": 3000.0,
            }
        ]
    )
    staged_path = tmp_path / "fry9c_normalized.parquet"
    staged.to_parquet(staged_path, index=False)

    crosswalk = pd.DataFrame(
        [
            {
                "entity_id": "parent_2001",
                "entity_name": "Demo Parent",
                "entity_type": "bhc_parent",
                "rssd_id": "2001",
                "fdic_cert": "",
                "top_parent_rssd": "",
                "country": "United States",
                "is_gsib_parent": True,
                "is_covered_bank_subsidiary": False,
                "fr_y15_reporter": "DEMO_PARENT",
                "parent_method1_surcharge": 0.045,
            }
        ]
    )
    crosswalk_path = tmp_path / "crosswalk.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)

    output_path = build_parent_panel(
        staged_path,
        crosswalk_path,
        output_path=tmp_path / "parent.parquet",
    )
    frame = pd.read_parquet(output_path)

    assert frame.empty
