from pathlib import Path

import pandas as pd

from slr_watch.analytics.event_2020 import run_event_2020, select_flagship_per_parent


def test_select_flagship_per_parent_prefers_largest_baseline_bank():
    frame = pd.DataFrame(
        [
            {
                "entity_id": "bank_a",
                "top_parent_rssd": "1000",
                "quarter_end": "2019-12-31",
                "total_assets": 100.0,
            },
            {
                "entity_id": "bank_b",
                "top_parent_rssd": "1000",
                "quarter_end": "2019-12-31",
                "total_assets": 200.0,
            },
            {
                "entity_id": "bank_c",
                "top_parent_rssd": "2000",
                "quarter_end": "2019-12-31",
                "total_assets": 150.0,
            },
            {
                "entity_id": "bank_b",
                "top_parent_rssd": "1000",
                "quarter_end": "2020-03-31",
                "total_assets": 210.0,
            },
            {
                "entity_id": "bank_c",
                "top_parent_rssd": "2000",
                "quarter_end": "2020-03-31",
                "total_assets": 155.0,
            },
        ]
    )

    selected = select_flagship_per_parent(frame)

    assert sorted(selected["entity_id"].unique()) == ["bank_b", "bank_c"]


def test_run_event_2020_smoke(tmp_path: Path):
    quarters = pd.date_range("2019-03-31", "2021-12-31", freq="QE-DEC")
    rows = []
    for entity_num in range(6):
        entity_id = f"bank_{entity_num}"
        for idx, quarter_end in enumerate(quarters):
            rows.append(
                {
                    "entity_id": entity_id,
                    "rssd_id": f"{1000 + entity_num}",
                    "entity_name": f"Bank {entity_num}",
                    "top_parent_rssd": f"parent_{entity_num // 2}",
                    "quarter_end": quarter_end.strftime("%Y-%m-%d"),
                    "tier1_capital": 55.0 + entity_num,
                    "total_leverage_exposure": 1000.0 + (20 * entity_num),
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

    panel = pd.DataFrame(rows)
    panel_path = tmp_path / "insured_bank_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    market_panel = pd.DataFrame(
        {
            "quarter_end": quarters.strftime("%Y-%m-%d"),
            "pd_ust_dealer_position_net_mn": [100 + idx for idx in range(len(quarters))],
            "trace_total_par_value_bn": [1000 + (10 * idx) for idx in range(len(quarters))],
        }
    )
    market_panel_path = tmp_path / "market_overlay_panel.parquet"
    market_panel.to_parquet(market_panel_path, index=False)
    market_dir = tmp_path / "market_context"
    market_dir.mkdir()
    (market_dir / "summary.md").write_text("# Treasury Market Context\n", encoding="utf-8")

    output_dir = run_event_2020(panel_path, output_dir=tmp_path / "event_2020", market_panel_path=market_panel_path)

    assert (output_dir / "prepared_panel.csv").exists()
    assert (output_dir / "did_results.csv").exists()
    assert (output_dir / "sample_manifest.csv").exists()
    assert (output_dir / "sample_manifest.md").exists()
    assert (output_dir / "sample_ladder.csv").exists()
    assert (output_dir / "sample_ladder.md").exists()
    assert (output_dir / "methodology_memo.md").exists()
    assert (output_dir / "gpt_pro_next_steps_prompt.md").exists()
    assert (output_dir / "pretrend_checks.csv").exists()
    assert (output_dir / "pretrend_checks.md").exists()
    assert (output_dir / "expanded_sensitivity" / "prepared_panel.csv").exists()
    assert (output_dir / "historical_unbalanced" / "prepared_panel.csv").exists()
    assert (output_dir / "flagship_per_parent" / "prepared_panel.csv").exists()
    assert (output_dir / "flagship_per_parent_clustered" / "prepared_panel.csv").exists()
    assert (output_dir / "flagship_per_parent_clustered" / "placebo_fake_date.csv").exists()
    assert (output_dir / "flagship_per_parent_clustered" / "placebo_fake_date.md").exists()
    assert (output_dir / "flagship_per_parent_expanded" / "prepared_panel.csv").exists()
    assert (output_dir / "flagship_per_parent_expanded_clustered" / "prepared_panel.csv").exists()
    assert (output_dir / "sample_comparison.md").exists()
    assert (output_dir / "market_control_sensitivity.md").exists()
    assert (output_dir / "market_interaction_sensitivity.md").exists()
    assert (output_dir / "market_interaction_sensitivity.csv").exists()
    assert (output_dir / "market_aux_no_time_fe.md").exists()
    assert (output_dir / "market_aux_no_time_fe.csv").exists()
    did_results = pd.read_csv(output_dir / "did_results.csv")
    clustered_results = pd.read_csv(output_dir / "flagship_per_parent_clustered" / "did_results.csv")
    interaction_results = pd.read_csv(output_dir / "market_interaction_sensitivity.csv")
    aux_results = pd.read_csv(output_dir / "market_aux_no_time_fe.csv")
    assert not did_results.empty
    assert not clustered_results.empty
    assert not interaction_results.empty
    assert not aux_results.empty
    manifest = pd.read_csv(output_dir / "sample_manifest.csv")
    synthetic_manifest = manifest[manifest["entity_id"].str.startswith("bank_")].copy()
    assert synthetic_manifest["included_universe_b"].sum() == 6
    assert synthetic_manifest["included_universe_c"].sum() == 6
    assert synthetic_manifest["included_universe_d"].sum() == 6
    assert synthetic_manifest["included_universe_f_primary"].sum() == 3
    excluded_flagship = synthetic_manifest.loc[
        ~synthetic_manifest["included_universe_f_primary"],
        "universe_f_primary_exclusion_reason",
    ].dropna()
    assert set(excluded_flagship) == {"not_largest_2019q4_subsidiary_in_parent_family"}
    ladder = pd.read_csv(output_dir / "sample_ladder.csv")
    assert set(ladder["sample_name"]) >= {
        "universe_a_all_insured_banks",
        "universe_b_slr_reporting",
        "universe_c_treatment_definable",
        "universe_d_primary_core",
        "universe_e_expanded_sensitivity",
        "universe_f_flagship_primary",
    }
    assert set(clustered_results["cov_type"]) == {"cluster"}
    assert set(clustered_results["cluster_col"]) == {"top_parent_rssd"}
    assert set(interaction_results["cluster_col"]) == {"top_parent_rssd"}
    assert set(aux_results["cluster_col"]) == {"top_parent_rssd"}
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Market Context" in summary
    assert "auxiliary no-time-FE" in summary
    control_note = (output_dir / "market_control_sensitivity.md").read_text(encoding="utf-8")
    assert "absorbed by those fixed effects" in control_note
    interaction_note = (output_dir / "market_interaction_sensitivity.md").read_text(encoding="utf-8")
    assert "treated_post × standardized_market_level" in interaction_note
    aux_note = (output_dir / "market_aux_no_time_fe.md").read_text(encoding="utf-8")
    assert "drops quarter fixed effects" in aux_note
    manifest_note = (output_dir / "sample_manifest.md").read_text(encoding="utf-8")
    assert "Event Study Sample Manifest" in manifest_note
    assert "Universe D" in manifest_note
    prompt_note = (output_dir / "gpt_pro_next_steps_prompt.md").read_text(encoding="utf-8")
    assert "Stability across D/E/F" in prompt_note
    assert "## Diagnostics" in prompt_note
    pretrend_note = (output_dir / "pretrend_checks.md").read_text(encoding="utf-8")
    placebo_note = (output_dir / "flagship_per_parent_clustered" / "placebo_fake_date.md").read_text(encoding="utf-8")
    assert "Pre-Trend Checks" in pretrend_note
    assert "Fake-Date Placebo" in placebo_note
