from pathlib import Path

import pandas as pd

from slr_watch.insured_banks import build_all_insured_bank_panel


def _write_stage_pair(root: Path, quarter_end: pd.Timestamp, normalized: pd.DataFrame, merged: pd.DataFrame) -> None:
    quarter_dir = root / f"{quarter_end.year}Q{((quarter_end.month - 1) // 3) + 1}"
    quarter_dir.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(quarter_dir / "call_reports_normalized.parquet", index=False)
    merged.to_parquet(quarter_dir / "call_reports_merged.parquet", index=False)


def test_build_all_insured_bank_panel_generates_universe_and_sample_flags(tmp_path: Path) -> None:
    staged_root = tmp_path / "call_reports"
    quarters = pd.date_range("2019-03-31", "2021-12-31", freq="QE-DEC")

    for quarter_end in quarters:
        normalized_rows = [
            {
                "rssd_id": "1001",
                "quarter_end": quarter_end.strftime("%Y-%m-%d"),
                "tier1_capital": 60.0,
                "total_leverage_exposure": 1000.0,
                "ust_htm_fair_value": 10.0,
                "ust_afs_fair_value": 20.0,
                "ust_trading_assets": 5.0,
                "balances_due_from_fed": 100.0,
                "reverse_repos": 15.0,
                "trading_assets_total": 12.0,
                "total_assets": 2000.0,
                "deposits": 1400.0,
                "loans": 900.0,
            },
            {
                "rssd_id": "1002",
                "quarter_end": quarter_end.strftime("%Y-%m-%d"),
                "tier1_capital": 40.0,
                "total_leverage_exposure": 900.0 if quarter_end != pd.Timestamp("2019-12-31") else 0.0,
                "ust_htm_fair_value": 8.0,
                "ust_afs_fair_value": 18.0,
                "ust_trading_assets": 4.0,
                "balances_due_from_fed": 80.0,
                "reverse_repos": 10.0,
                "trading_assets_total": 10.0,
                "total_assets": 1500.0,
                "deposits": 1000.0,
                "loans": 700.0,
            },
        ]
        if quarter_end in {
            pd.Timestamp("2019-03-31"),
            pd.Timestamp("2019-06-30"),
            pd.Timestamp("2019-09-30"),
            pd.Timestamp("2019-12-31"),
            pd.Timestamp("2020-03-31"),
            pd.Timestamp("2020-06-30"),
            pd.Timestamp("2020-09-30"),
            pd.Timestamp("2020-12-31"),
            pd.Timestamp("2021-03-31"),
            pd.Timestamp("2021-06-30"),
        }:
            normalized_rows.append(
                {
                    "rssd_id": "1003",
                    "quarter_end": quarter_end.strftime("%Y-%m-%d"),
                    "tier1_capital": 45.0,
                    "total_leverage_exposure": 850.0,
                    "ust_htm_fair_value": 9.0,
                    "ust_afs_fair_value": 12.0,
                    "ust_trading_assets": 3.0,
                    "balances_due_from_fed": 70.0,
                    "reverse_repos": 8.0,
                    "trading_assets_total": 9.0,
                    "total_assets": 1400.0,
                    "deposits": 950.0,
                    "loans": 650.0,
                }
            )

        merged_rows = [
            {"RSSD9001": "1001", "RSSD9017": "Alpha Bank", "RSSD9050": "10001", "RSSD9130": "New York", "RSSD9200": "NY", "RSSD9220": "10001"},
            {"RSSD9001": "1002", "RSSD9017": "Beta Bank", "RSSD9050": "10002", "RSSD9130": "Boston", "RSSD9200": "MA", "RSSD9220": "02110"},
            {"RSSD9001": "1003", "RSSD9017": "Gamma Bank", "RSSD9050": "10003", "RSSD9130": "Chicago", "RSSD9200": "IL", "RSSD9220": "60601"},
        ]

        _write_stage_pair(
            staged_root,
            quarter_end,
            pd.DataFrame(normalized_rows),
            pd.DataFrame(merged_rows),
        )

    crosswalk = pd.DataFrame(
        [
            {
                "entity_id": "alpha_bank",
                "entity_name": "Alpha Bank",
                "entity_type": "insured_bank_sub",
                "rssd_id": "1001",
                "fdic_cert": "10001",
                "top_parent_rssd": "9001",
                "country": "United States",
                "is_gsib_parent": False,
                "is_covered_bank_subsidiary": True,
                "fr_y15_reporter": "",
            }
        ]
    )
    crosswalk_path = tmp_path / "crosswalk.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)

    outputs = build_all_insured_bank_panel(
        staged_root,
        crosswalk_path=crosswalk_path,
        overrides_path=tmp_path / "overrides.csv",
        output_path=tmp_path / "insured_bank_descriptive_panel.parquet",
        universe_output_path=tmp_path / "insured_bank_universe.csv",
        coverage_output_path=tmp_path / "insured_bank_coverage_by_quarter.csv",
        manifest_output_path=tmp_path / "insured_bank_sample_manifest.csv",
    )

    panel = pd.read_parquet(outputs["panel"])
    universe = pd.read_csv(outputs["universe"])
    manifest = pd.read_csv(outputs["manifest"])

    assert panel["entity_id"].nunique() == 3
    assert len(universe) == 3
    assert universe.loc[universe["entity_id"] == "alpha_bank", "slr_scope_class"].iloc[0] == "covered_bank_subsidiary"
    assert int(manifest["in_universe_b"].sum()) == 3
    assert int(manifest["in_universe_c"].sum()) == 2
    assert int(manifest["in_universe_d"].sum()) == 1
    assert int(manifest["in_universe_e"].sum()) == 2
    beta_reason = manifest.loc[manifest["entity_name"] == "Beta Bank", "universe_c_exclusion_reason"].iloc[0]
    gamma_reason = manifest.loc[manifest["entity_name"] == "Gamma Bank", "universe_d_exclusion_reason"].iloc[0]
    assert beta_reason == "missing_usable_2019q4_treatment_baseline"
    assert gamma_reason == "incomplete_2019q1_2021q4_coverage"


def test_build_all_insured_bank_panel_uses_fdic_overlay_for_nonseed_parent_linkage(tmp_path: Path) -> None:
    staged_root = tmp_path / "call_reports"
    quarter_end = pd.Timestamp("2019-12-31")

    _write_stage_pair(
        staged_root,
        quarter_end,
        pd.DataFrame(
            [
                {
                    "rssd_id": "2001",
                    "quarter_end": quarter_end.strftime("%Y-%m-%d"),
                    "tier1_capital": 50.0,
                    "total_leverage_exposure": 950.0,
                    "ust_htm_fair_value": 10.0,
                    "ust_afs_fair_value": 15.0,
                    "ust_trading_assets": 2.0,
                    "balances_due_from_fed": 75.0,
                    "reverse_repos": 8.0,
                    "trading_assets_total": 7.0,
                    "total_assets": 1200.0,
                    "deposits": 800.0,
                    "loans": 600.0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "RSSD9001": "2001",
                    "RSSD9017": "Delta Bank Raw",
                    "RSSD9050": "20001",
                    "RSSD9130": "Miami",
                    "RSSD9200": "FL",
                    "RSSD9220": "33101",
                }
            ]
        ),
    )

    fdic_overlay = pd.DataFrame(
        [
            {
                "rssd_id": "2001",
                "fdic_cert": "20001",
                "fdic_top_parent_rssd": "99001",
                "fdic_entity_name": "Delta Bank",
                "fdic_top_parent_name": "Delta Bancorp",
                "fdic_active": 1,
                "fdic_bank_class": "NM",
                "fdic_regulator": "FDIC",
            }
        ]
    )
    fdic_path = tmp_path / "fdic_institutions.csv"
    fdic_overlay.to_csv(fdic_path, index=False)

    outputs = build_all_insured_bank_panel(
        staged_root,
        crosswalk_path=tmp_path / "missing_crosswalk.parquet",
        overrides_path=tmp_path / "missing_overrides.csv",
        fdic_metadata_path=fdic_path,
        output_path=tmp_path / "insured_bank_descriptive_panel.parquet",
        universe_output_path=tmp_path / "insured_bank_universe.csv",
        coverage_output_path=tmp_path / "insured_bank_coverage_by_quarter.csv",
        manifest_output_path=tmp_path / "insured_bank_sample_manifest.csv",
    )

    universe = pd.read_csv(outputs["universe"], dtype={"top_parent_rssd": "string", "fdic_cert": "string"})

    assert len(universe) == 1
    assert universe.loc[0, "entity_name"] == "Delta Bank"
    assert universe.loc[0, "top_parent_rssd"] == "99001"
    assert universe.loc[0, "top_parent_name"] == "Delta Bancorp"
    assert universe.loc[0, "fdic_cert"] == "20001"
    assert universe.loc[0, "slr_scope_class"] == "slr_reporting_insured_bank"
