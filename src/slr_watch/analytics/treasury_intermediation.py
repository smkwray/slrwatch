from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import derived_data_path, reports_path
from ..pipeline import read_table, write_frame
from .event_2020 import TREATMENTS, add_treatments, prepare_event_2020_panel


INTERMEDIATION_OUTCOMES = {
    "reverse_repos_scaled": "Reverse repo / assets",
    "trading_assets_scaled": "Trading assets / assets",
    "ust_inventory_fv_scaled": "Treasury inventory / assets",
}

MARKET_COLUMNS = {
    "pd_ust_dealer_position_net_mn": "NY Fed dealer net UST position",
    "pd_ust_repo_mn_weekly_avg": "NY Fed UST repo",
    "pd_ust_reverse_repo_mn_weekly_avg": "NY Fed UST reverse repo",
    "trace_total_par_value_bn": "TRACE total par volume",
}


def _event_window(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    out = out[
        (out["quarter_end"] >= pd.Timestamp("2019-03-31"))
        & (out["quarter_end"] <= pd.Timestamp("2021-12-31"))
    ].copy()
    out["period"] = pd.NA
    out.loc[out["quarter_end"] <= pd.Timestamp("2020-03-31"), "period"] = "pre"
    out.loc[out["quarter_end"] >= pd.Timestamp("2020-06-30"), "period"] = "post"
    return out.dropna(subset=["period"]).reset_index(drop=True)


def _load_market(panel_path: Path | None = None) -> pd.DataFrame | None:
    source = panel_path or derived_data_path("market_overlay_panel.parquet")
    if not source.exists():
        return None
    market = read_table(source).copy()
    market["quarter_end"] = pd.to_datetime(market["quarter_end"])
    selected = [column for column in MARKET_COLUMNS if column in market.columns]
    if not selected:
        return None
    return market[["quarter_end", *selected]].drop_duplicates("quarter_end")


def _summarize_intermediation(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment in [item.name for item in TREATMENTS]:
        if treatment not in frame.columns:
            continue
        for outcome, label in INTERMEDIATION_OUTCOMES.items():
            sample = frame[[treatment, "period", outcome]].dropna()
            if sample.empty:
                continue
            grouped = (
                sample.groupby([treatment, "period"], dropna=False)[outcome]
                .mean()
                .unstack("period")
                .reindex(index=[0, 1], columns=["pre", "post"])
            )
            treated_pre = grouped.loc[1, "pre"] if 1 in grouped.index else pd.NA
            treated_post = grouped.loc[1, "post"] if 1 in grouped.index else pd.NA
            control_pre = grouped.loc[0, "pre"] if 0 in grouped.index else pd.NA
            control_post = grouped.loc[0, "post"] if 0 in grouped.index else pd.NA
            treated_change = treated_post - treated_pre if pd.notna(treated_pre) and pd.notna(treated_post) else pd.NA
            control_change = control_post - control_pre if pd.notna(control_pre) and pd.notna(control_post) else pd.NA
            did_like = treated_change - control_change if pd.notna(treated_change) and pd.notna(control_change) else pd.NA
            rows.append(
                {
                    "treatment": treatment,
                    "outcome": outcome,
                    "outcome_label": label,
                    "treated_pre": treated_pre,
                    "treated_post": treated_post,
                    "treated_change": treated_change,
                    "control_pre": control_pre,
                    "control_post": control_post,
                    "control_change": control_change,
                    "did_like_change": did_like,
                    "treated_obs": int(sample[sample[treatment] == 1].shape[0]),
                    "control_obs": int(sample[sample[treatment] == 0].shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _build_market_linkage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available_market = [column for column in MARKET_COLUMNS if column in frame.columns]
    if not available_market:
        return pd.DataFrame()

    for treatment in [item.name for item in TREATMENTS]:
        if treatment not in frame.columns:
            continue
        for outcome, label in INTERMEDIATION_OUTCOMES.items():
            sample = frame[["quarter_end", treatment, outcome, *available_market]].dropna(subset=[outcome]).copy()
            if sample.empty:
                continue
            quarter_gap = (
                sample.groupby(["quarter_end", treatment], dropna=False)[outcome]
                .mean()
                .unstack(treatment)
                .reindex(columns=[0, 1])
            )
            if quarter_gap.empty or 0 not in quarter_gap.columns or 1 not in quarter_gap.columns:
                continue
            quarter_gap = quarter_gap.rename(columns={0: "control_mean", 1: "treated_mean"}).reset_index()
            quarter_gap["treated_minus_control_gap"] = quarter_gap["treated_mean"] - quarter_gap["control_mean"]
            market_by_quarter = sample.groupby("quarter_end", dropna=False)[available_market].first().reset_index()
            merged = quarter_gap.merge(market_by_quarter, how="left", on="quarter_end")
            for market_column in available_market:
                corr_sample = merged[["treated_minus_control_gap", market_column]].dropna()
                if len(corr_sample) < 2:
                    corr = pd.NA
                else:
                    corr = corr_sample["treated_minus_control_gap"].corr(corr_sample[market_column])
                rows.append(
                    {
                        "treatment": treatment,
                        "outcome": outcome,
                        "outcome_label": label,
                        "market_column": market_column,
                        "market_label": MARKET_COLUMNS[market_column],
                        "quarters_observed": int(len(corr_sample)),
                        "gap_market_correlation": corr,
                    }
                )
    return pd.DataFrame(rows)


def _write_summary(
    prepared: pd.DataFrame,
    summary_frame: pd.DataFrame,
    linkage_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Treasury Intermediation Sensitivity",
        "",
        f"- Observations: {len(prepared)}",
        f"- Distinct entities: {prepared['entity_id'].nunique()}",
        f"- Quarter range: {prepared['quarter_end'].min().date()} to {prepared['quarter_end'].max().date()}",
        "",
        "## Overview",
        "- This report tracks whether constrained banks shifted reverse repo, trading assets, and Treasury inventory differently around the 2020 temporary exclusion.",
        "- It also links treated-minus-control quarterly gaps to public dealer-position, repo, and TRACE market context series.",
        "",
        "## Headline results",
    ]

    if summary_frame.empty:
        lines.append("- No intermediation rows were estimable.")
    else:
        focus = summary_frame[
            (summary_frame["treatment"].isin(["low_headroom_treated", "covered_bank_treated"]))
            & (summary_frame["outcome"].isin(["reverse_repos_scaled", "trading_assets_scaled", "ust_inventory_fv_scaled"]))
        ].sort_values(["treatment", "outcome"])
        for _, row in focus.iterrows():
            lines.append(
                f"- `{row['treatment']}` / `{row['outcome_label']}`: treated change {row['treated_change']:.4f}, "
                f"control change {row['control_change']:.4f}, net {row['did_like_change']:.4f}"
            )

    lines.extend(["", "## Market linkage"])
    if linkage_frame.empty:
        lines.append("- No market linkage rows were estimable.")
    else:
        strongest = linkage_frame.dropna(subset=["gap_market_correlation"]).copy()
        if strongest.empty:
            lines.append("- Market columns were present, but there were not enough overlapping quarters to estimate simple correlations.")
        else:
            strongest = strongest.reindex(strongest["gap_market_correlation"].abs().sort_values(ascending=False).index).head(6)
            for _, row in strongest.iterrows():
                lines.append(
                    f"- `{row['treatment']}` / `{row['outcome_label']}` vs `{row['market_label']}`: "
                    f"quarterly gap correlation {row['gap_market_correlation']:.3f} across {int(row['quarters_observed'])} quarters."
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_treasury_intermediation_report(
    panel_path: Path | None = None,
    market_panel_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    source = panel_path or derived_data_path("insured_bank_panel.parquet")
    panel = read_table(source)
    prepared = prepare_event_2020_panel(panel)
    prepared = _event_window(prepared)
    prepared = add_treatments(prepared)

    market = _load_market(market_panel_path)
    if market is not None:
        prepared = prepared.merge(market, how="left", on="quarter_end")

    summary_frame = _summarize_intermediation(prepared)
    linkage_frame = _build_market_linkage(prepared)

    destination = output_dir or reports_path("treasury_intermediation")
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(prepared.assign(quarter_end=prepared["quarter_end"].dt.strftime("%Y-%m-%d")), destination / "prepared_panel.csv")
    write_frame(summary_frame, destination / "intermediation_summary.csv")
    write_frame(linkage_frame, destination / "market_linkage_summary.csv")
    _write_summary(prepared, summary_frame, linkage_frame, destination / "summary.md")
    return destination
