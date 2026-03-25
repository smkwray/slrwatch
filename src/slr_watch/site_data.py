from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import repo_root, reports_path


EVENT_LABELS = {
    ("low_headroom_treated", "ust_inventory_fv_scaled"): ("Low Headroom", "Treasury Inventory / Assets"),
    ("covered_bank_treated", "ust_inventory_fv_scaled"): ("Covered Bank", "Treasury Inventory / Assets"),
    ("high_ust_share_treated", "ust_inventory_fv_scaled"): ("High UST Share", "Treasury Inventory / Assets"),
    ("high_ust_share_treated", "trading_assets_scaled"): ("High UST Share", "Trading Assets / Assets"),
}

TREATMENT_LABELS = {
    "low_headroom_treated": "Low Headroom",
    "covered_bank_treated": "Covered Bank",
    "high_ust_share_treated": "High UST Share",
}

REALLOCATION_OUTCOME_LABELS = {
    "ust_inventory_fv_scaled": "Treasury Inventory",
    "balances_due_from_fed_scaled": "Fed Balances",
    "deposit_growth": "Deposit Growth",
    "loan_growth": "Loan Growth",
}

REGIME_LABELS = {
    "pre_exclusion": {"name": "Pre-Exclusion", "period": "through 2020-Q1"},
    "temporary_exclusion": {"name": "Temporary Exclusion", "period": "2020-Q2 to 2021-Q1"},
    "post_exclusion_normalization": {"name": "Post-Exclusion", "period": "2021-Q2 to 2022-Q1"},
    "qt_era": {"name": "QT Era", "period": "2022-Q2 onward"},
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _write_json(payload: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _quarter_label(value: str | pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    quarter = ((ts.month - 1) // 3) + 1
    return f"{ts.year}-Q{quarter}"


def _pct_change(first: float | int | None, last: float | int | None) -> float | None:
    if first in (None, 0) or pd.isna(first) or pd.isna(last):
        return None
    return ((float(last) / float(first)) - 1.0) * 100.0


def _load_event_study_payload(reports_root: Path) -> dict[str, object]:
    sample_ladder = _read_csv(reports_root / "event_2020" / "sample_ladder.csv").set_index("sample_name")
    primary_prepared = _read_csv(reports_root / "event_2020" / "prepared_panel.csv")
    expanded_prepared = _read_csv(reports_root / "event_2020" / "expanded_sensitivity" / "prepared_panel.csv")
    flagship_prepared = _read_csv(reports_root / "event_2020" / "flagship_per_parent" / "prepared_panel.csv")
    primary = _read_csv(reports_root / "event_2020" / "did_results.csv")
    expanded = _read_csv(reports_root / "event_2020" / "expanded_sensitivity" / "did_results.csv")
    clustered = _read_csv(reports_root / "event_2020" / "flagship_per_parent_clustered" / "did_results.csv")

    clustered_clusters = 0
    if not clustered.empty and "n_clusters" in clustered.columns and clustered["n_clusters"].notna().any():
        clustered_clusters = int(clustered["n_clusters"].dropna().iloc[0])
    elif "top_parent_rssd" in flagship_prepared.columns:
        clustered_clusters = int(flagship_prepared["top_parent_rssd"].astype("string").replace({"": pd.NA}).dropna().nunique())

    coefficients: list[dict[str, object]] = []
    for key, (treatment_label, outcome_label) in EVENT_LABELS.items():
        treatment, outcome = key
        primary_row = primary[(primary["treatment"] == treatment) & (primary["outcome"] == outcome)]
        expanded_row = expanded[(expanded["treatment"] == treatment) & (expanded["outcome"] == outcome)]
        clustered_row = clustered[(clustered["treatment"] == treatment) & (clustered["outcome"] == outcome)]
        if primary_row.empty or expanded_row.empty or clustered_row.empty:
            continue
        coefficients.append(
            {
                "treatment": treatment_label,
                "outcome": outcome_label,
                "primary": {
                    "coef": float(primary_row.iloc[0]["coef"]),
                    "p": float(primary_row.iloc[0]["pvalue"]),
                },
                "expanded": {
                    "coef": float(expanded_row.iloc[0]["coef"]),
                    "p": float(expanded_row.iloc[0]["pvalue"]),
                },
                "flagship_clustered": {
                    "coef": float(clustered_row.iloc[0]["coef"]),
                    "p": float(clustered_row.iloc[0]["pvalue"]),
                },
            }
        )

    return {
        "samples": {
            "descriptive_universe": {
                "observations": int(sample_ladder.loc["universe_a_all_insured_banks", "observation_count"]),
                "entities": int(sample_ladder.loc["universe_a_all_insured_banks", "entity_count"]),
            },
            "slr_reporting": {
                "observations": int(sample_ladder.loc["universe_b_slr_reporting", "observation_count"]),
                "entities": int(sample_ladder.loc["universe_b_slr_reporting", "entity_count"]),
            },
            "treatment_definable": {
                "observations": int(sample_ladder.loc["universe_c_treatment_definable", "observation_count"]),
                "entities": int(sample_ladder.loc["universe_c_treatment_definable", "entity_count"]),
            },
            "primary_core": {
                "observations": int(len(primary_prepared)),
                "entities": int(primary_prepared["entity_id"].nunique()),
            },
            "expanded_sensitivity": {
                "observations": int(len(expanded_prepared)),
                "entities": int(expanded_prepared["entity_id"].nunique()),
            },
            "flagship_primary": {
                "observations": int(len(flagship_prepared)),
                "entities": int(flagship_prepared["entity_id"].nunique()),
                "clusters": clustered_clusters,
            },
        },
        "coefficients": coefficients,
    }


def _load_reallocation_payload(reports_root: Path) -> dict[str, object]:
    summary = _read_csv(reports_root / "reallocation_2020" / "reallocation_summary.csv")
    focus = summary[
        summary["treatment"].isin(["low_headroom_treated", "covered_bank_treated"])
        & summary["outcome"].isin(list(REALLOCATION_OUTCOME_LABELS))
    ].copy()
    focus["treatment_label"] = focus["treatment"].map(TREATMENT_LABELS)
    focus["outcome_short_label"] = focus["outcome"].map(REALLOCATION_OUTCOME_LABELS)
    focus = focus.sort_values(["treatment_label", "outcome_short_label"])
    return {
        "treatments": ["Low Headroom", "Covered Bank"],
        "outcomes": [REALLOCATION_OUTCOME_LABELS[key] for key in ["ust_inventory_fv_scaled", "balances_due_from_fed_scaled", "deposit_growth", "loan_growth"]],
        "data": [
            {
                "treatment": str(row["treatment_label"]),
                "outcome": str(row["outcome_short_label"]),
                "treated": float(row["treated_change"]),
                "control": float(row["control_change"]),
                "net": float(row["did_like_change"]),
            }
            for _, row in focus.iterrows()
        ],
    }


def _load_safe_assets_payload(reports_root: Path) -> dict[str, object]:
    summary = _read_csv(reports_root / "safe_asset_absorption" / "absorption_summary.csv")
    rows: list[dict[str, object]] = []
    for treatment in ["low_headroom_treated", "covered_bank_treated"]:
        subset = summary[summary["treatment"] == treatment].set_index("outcome")
        if subset.empty:
            continue
        rows.append(
            {
                "treatment": TREATMENT_LABELS[treatment],
                "treasury_share_net": float(subset.loc["treasury_share_of_safe", "did_like_change"]) if "treasury_share_of_safe" in subset.index else None,
                "fed_balance_net": float(subset.loc["balances_due_from_fed_scaled", "did_like_change"]) if "balances_due_from_fed_scaled" in subset.index else None,
                "treasury_inventory_net": float(subset.loc["ust_inventory_fv_scaled", "did_like_change"]) if "ust_inventory_fv_scaled" in subset.index else None,
            }
        )
    return {"data": rows}


def _load_intermediation_payload(reports_root: Path) -> dict[str, object]:
    summary = _read_csv(reports_root / "treasury_intermediation" / "intermediation_summary.csv")
    linkage = _read_csv(reports_root / "treasury_intermediation" / "market_linkage_summary.csv")

    net_changes: list[dict[str, object]] = []
    for treatment in ["low_headroom_treated", "covered_bank_treated"]:
        subset = summary[summary["treatment"] == treatment].set_index("outcome")
        if subset.empty:
            continue
        net_changes.append(
            {
                "treatment": TREATMENT_LABELS[treatment],
                "trading_assets": float(subset.loc["trading_assets_scaled", "did_like_change"]) if "trading_assets_scaled" in subset.index else None,
                "treasury_inventory": float(subset.loc["ust_inventory_fv_scaled", "did_like_change"]) if "ust_inventory_fv_scaled" in subset.index else None,
            }
        )

    ranked = linkage.dropna(subset=["gap_market_correlation"]).copy()
    if not ranked.empty:
        ranked["abs_corr"] = ranked["gap_market_correlation"].abs()
        ranked = ranked.sort_values("abs_corr", ascending=False).head(6)

    return {
        "net_changes": net_changes,
        "market_correlations": [
            {
                "treatment": TREATMENT_LABELS.get(str(row["treatment"]), str(row["treatment"])),
                "outcome": str(row["outcome_label"]).replace(" / assets", ""),
                "market_var": str(row["market_label"]).replace("NY Fed ", "").replace("TRACE ", ""),
                "corr": float(row["gap_market_correlation"]),
            }
            for _, row in ranked.iterrows()
        ],
    }


def _load_market_context_payload(reports_root: Path) -> dict[str, object]:
    prepared_path = reports_root / "market_context" / "prepared_panel.csv"
    common_path = reports_root / "market_context" / "common_overlap_panel.csv"
    panel = _read_csv(prepared_path)
    common = _read_csv(common_path if common_path.exists() else prepared_path)
    panel["quarter_end"] = pd.to_datetime(panel["quarter_end"])
    common["quarter_end"] = pd.to_datetime(common["quarter_end"])
    panel = panel.sort_values("quarter_end").reset_index(drop=True)
    common = common.sort_values("quarter_end").reset_index(drop=True)
    start = common.iloc[0]
    end = common.iloc[-1]
    return {
        "coverage": {
            "quarters": int(len(panel)),
            "start": _quarter_label(panel["quarter_end"].min()),
            "end": _quarter_label(panel["quarter_end"].max()),
        },
        "long_run_changes": [
            {
                "series": "NY Fed Dealer Net UST Position",
                "start_val": float(start["pd_ust_dealer_position_net_mn"]),
                "end_val": float(end["pd_ust_dealer_position_net_mn"]),
                "pct_change": round(float(_pct_change(start["pd_ust_dealer_position_net_mn"], end["pd_ust_dealer_position_net_mn"])), 1),
                "unit": "$M",
            },
            {
                "series": "NY Fed UST Repo",
                "start_val": float(start["pd_ust_repo_mn_weekly_avg"]),
                "end_val": float(end["pd_ust_repo_mn_weekly_avg"]),
                "pct_change": round(float(_pct_change(start["pd_ust_repo_mn_weekly_avg"], end["pd_ust_repo_mn_weekly_avg"])), 1),
                "unit": "$M",
            },
            {
                "series": "TRACE Total Par Volume",
                "start_val": float(start["trace_total_par_value_bn"]),
                "end_val": float(end["trace_total_par_value_bn"]),
                "pct_change": round(float(_pct_change(start["trace_total_par_value_bn"], end["trace_total_par_value_bn"])), 1),
                "unit": "$B",
            },
        ],
    }


def _directional_agreement_counts(linked: pd.DataFrame, bank_col: str, parent_col: str) -> dict[str, int | float] | None:
    if bank_col not in linked.columns or parent_col not in linked.columns:
        return None
    frame = linked.sort_values(["top_parent_rssd", "entity_id_bank", "quarter_end"]).copy()
    frame["quarter_end"] = pd.to_datetime(frame["quarter_end"])
    frame["delta_bank"] = frame.groupby("entity_id_bank")[bank_col].diff()
    frame["delta_parent"] = frame.groupby(["top_parent_rssd"])[parent_col].diff()
    valid = frame.dropna(subset=["delta_bank", "delta_parent"])
    if valid.empty:
        return None
    same_direction = int(((valid["delta_bank"] * valid["delta_parent"]) > 0).sum())
    total = int(len(valid))
    pct = (same_direction / total) * 100.0
    return {"same_direction": same_direction, "total": total, "pct": round(pct, 1)}


def _load_parent_transmission_payload(reports_root: Path) -> dict[str, object]:
    linked = _read_csv(reports_root / "parent_transmission" / "linked_panel.csv")
    family = _read_csv(reports_root / "parent_transmission" / "family_quarter_summary.csv")
    linked["quarter_end"] = pd.to_datetime(linked["quarter_end"])
    family["quarter_end"] = pd.to_datetime(family["quarter_end"])

    surcharge = pd.to_numeric(family.get("surcharge"), errors="coerce")
    median = surcharge.median()
    high = family[surcharge > median]
    low = family[surcharge <= median]
    return {
        "coverage": {
            "linked_observations": int(len(linked)),
            "bank_subsidiaries": int(linked["entity_id_bank"].nunique()),
            "parent_families": int(linked["top_parent_rssd"].nunique()),
            "quarters": int(linked["quarter_end"].nunique()),
            "period": f"{_quarter_label(linked['quarter_end'].min())} to {_quarter_label(linked['quarter_end'].max())}",
        },
        "co_movement": {
            "ust_holdings": _directional_agreement_counts(linked, "ust_share_assets_bank", "ust_share_assets_parent"),
            "trading_assets": _directional_agreement_counts(linked, "trading_assets_total_share_assets_bank", "trading_assets_total_share_assets_parent"),
        },
        "surcharge_comparison": {
            "high_surcharge": {
                "bank_ust_assets": float(high["mean_ust_share_assets_bank"].mean()) if not high.empty else None,
                "bank_headroom_pp": float(high["mean_headroom_pp_bank"].mean()) if not high.empty else None,
                "parent_trading_assets": float(high["mean_trading_share_parent"].mean()) if not high.empty else None,
                "obs": int(len(high)),
            },
            "low_surcharge": {
                "bank_ust_assets": float(low["mean_ust_share_assets_bank"].mean()) if not low.empty else None,
                "bank_headroom_pp": float(low["mean_headroom_pp_bank"].mean()) if not low.empty else None,
                "parent_trading_assets": float(low["mean_trading_share_parent"].mean()) if not low.empty else None,
                "obs": int(len(low)),
            },
        },
    }


def _load_policy_regime_payload(reports_root: Path) -> dict[str, object]:
    summary = _read_csv(reports_root / "policy_regime_panel" / "regime_summary.csv")
    indexed = summary.set_index("policy_regime")
    metrics = [
        ("Insured-Bank Treasury Share", "bank_ust_share_assets_mean"),
        ("Insured-Bank Fed-Balance Share", "bank_fed_share_assets_mean"),
        ("Parent Trading-Assets Share", "parent_trading_share_assets_mean"),
    ]
    market = [
        ("NY Fed Dealer Net UST Position ($M)", "pd_ust_dealer_position_net_mn"),
        ("TRACE Total Par Volume ($B)", "trace_total_par_value_bn"),
    ]
    return {
        "regimes": [
            {"name": meta["name"], "id": regime_id, "period": meta["period"]}
            for regime_id, meta in REGIME_LABELS.items()
        ],
        "metrics": [
            {
                "label": label,
                "pre_exclusion": None if "pre_exclusion" not in indexed.index or pd.isna(indexed.loc["pre_exclusion", column]) else float(indexed.loc["pre_exclusion", column]),
                "temporary_exclusion": None if "temporary_exclusion" not in indexed.index or pd.isna(indexed.loc["temporary_exclusion", column]) else float(indexed.loc["temporary_exclusion", column]),
                "post_exclusion": None if "post_exclusion_normalization" not in indexed.index or pd.isna(indexed.loc["post_exclusion_normalization", column]) else float(indexed.loc["post_exclusion_normalization", column]),
                "qt_era": None if "qt_era" not in indexed.index or pd.isna(indexed.loc["qt_era", column]) else float(indexed.loc["qt_era", column]),
            }
            for label, column in metrics
        ],
        "market": [
            {
                "label": label,
                "pre_exclusion": None if "pre_exclusion" not in indexed.index or pd.isna(indexed.loc["pre_exclusion", column]) else float(indexed.loc["pre_exclusion", column]),
                "temporary_exclusion": None if "temporary_exclusion" not in indexed.index or pd.isna(indexed.loc["temporary_exclusion", column]) else float(indexed.loc["temporary_exclusion", column]),
                "post_exclusion": None if "post_exclusion_normalization" not in indexed.index or pd.isna(indexed.loc["post_exclusion_normalization", column]) else float(indexed.loc["post_exclusion_normalization", column]),
                "qt_era": None if "qt_era" not in indexed.index or pd.isna(indexed.loc["qt_era", column]) else float(indexed.loc["qt_era", column]),
            }
            for label, column in market
        ],
    }


def _load_constraint_decomposition_payload(reports_root: Path) -> dict[str, object]:
    regime = _read_csv(reports_root / "constraint_decomposition" / "regime_summary.csv")
    family = _read_csv(reports_root / "constraint_decomposition" / "family_alignment_summary.csv")
    regime_indexed = regime.set_index(["entity_source", "policy_regime"])

    def shares(entity_source: str, policy_regime: str) -> dict[str, float]:
        row = regime_indexed.loc[(entity_source, policy_regime)]
        return {
            "leverage": float(row["leverage_dominant_share"]),
            "duration_loss": float(row["duration_loss_dominant_share"]),
            "funding": float(row["funding_dominant_share"]),
        }

    duration_alignment = family[family["policy_regime"] == "duration_loss_window"]
    duration_row = duration_alignment.iloc[0] if not duration_alignment.empty else None
    return {
        "regimes": [
            {
                "regime": "Duration Loss Window (2022-23)",
                "insured": shares("insured_bank", "duration_loss_window"),
                "parent": shares("parent_or_ihc", "duration_loss_window"),
            },
            {
                "regime": "Late QT Normalization (2024-25)",
                "insured": shares("insured_bank", "late_qt_normalization"),
                "parent": shares("parent_or_ihc", "late_qt_normalization"),
            },
        ],
        "highlights": {
            "insured_duration_loss_window": float(regime_indexed.loc[("insured_bank", "duration_loss_window"), "duration_loss_dominant_share"]),
            "parent_duration_loss_window": float(regime_indexed.loc[("parent_or_ihc", "duration_loss_window"), "duration_loss_dominant_share"]),
            "insured_observations": int(regime[regime["entity_source"] == "insured_bank"]["observation_count"].sum()),
            "parent_observations": int(regime[regime["entity_source"] == "parent_or_ihc"]["observation_count"].sum()),
            "family_match_duration_loss_window": None if duration_row is None else float(duration_row["matched_dominant_constraint_share"]),
            "family_both_duration_loss_window": None if duration_row is None else float(duration_row["both_duration_loss_share"]),
        },
    }


def build_site_data(
    *,
    reports_root: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    reports_root = reports_root or reports_path()
    output_dir = output_dir or repo_root() / "site" / "assets" / "data"

    payloads = {
        "event_study.json": _load_event_study_payload(reports_root),
        "reallocation.json": _load_reallocation_payload(reports_root),
        "safe_assets.json": _load_safe_assets_payload(reports_root),
        "intermediation.json": _load_intermediation_payload(reports_root),
        "market_context.json": _load_market_context_payload(reports_root),
        "parent_transmission.json": _load_parent_transmission_payload(reports_root),
        "policy_regimes.json": _load_policy_regime_payload(reports_root),
        "constraint_decomposition.json": _load_constraint_decomposition_payload(reports_root),
    }

    written: list[Path] = []
    for filename, payload in payloads.items():
        written.append(_write_json(payload, output_dir / filename))
    return written
