from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from ..config import derived_data_path, reports_path
from ..insured_banks import (
    BASELINE_QUARTER,
    EVENT_WINDOW_END,
    EVENT_WINDOW_START,
    MIN_POST_QUARTERS_EXPANDED,
    MIN_PRE_QUARTERS_EXPANDED,
    POST_TREATMENT_START,
    PRE_TREATMENT_END,
)
from ..pipeline import read_table, write_frame
from .event_study import EventStudySpec, add_event_dummies, event_study_terms


@dataclass(frozen=True)
class TreatmentDefinition:
    name: str
    source_column: str
    mode: str


TREATMENTS = [
    TreatmentDefinition(name="low_headroom_treated", source_column="headroom_pp", mode="bottom_tercile"),
    TreatmentDefinition(name="high_ust_share_treated", source_column="ust_share_assets", mode="top_tercile"),
    TreatmentDefinition(name="covered_bank_treated", source_column="is_covered_bank_subsidiary", mode="boolean"),
]

OUTCOMES = [
    "ust_inventory_fv_scaled",
    "balances_due_from_fed_scaled",
    "reverse_repos_scaled",
    "trading_assets_scaled",
    "deposit_growth",
    "loan_growth",
]

KEY_COMPARISON_ROWS = [
    ("low_headroom_treated", "ust_inventory_fv_scaled"),
    ("covered_bank_treated", "ust_inventory_fv_scaled"),
    ("high_ust_share_treated", "ust_inventory_fv_scaled"),
    ("high_ust_share_treated", "trading_assets_scaled"),
]

MARKET_CONTROL_COLUMNS = {
    "pd_ust_dealer_position_net_mn": "NY Fed dealer net UST position",
    "trace_total_par_value_bn": "TRACE total par volume",
}
DEFAULT_MARKET_INTERACTION_CLUSTER = "top_parent_rssd"
PLACEBO_FAKE_SHOCK_DATE = date(2019, 10, 1)
PLACEBO_SAMPLE_END = pd.Timestamp("2020-03-31")


def _write_market_context_note(destination: Path) -> None:
    market_dir = destination.parent / "market_context"
    market_summary = market_dir / "summary.md"
    summary_path = destination / "summary.md"
    if not summary_path.exists():
        return

    lines = summary_path.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(
        [
            "",
            "## Market Context",
            f"- Related market-context report: `{market_summary}`" if market_summary.exists() else "- Related market-context report: not available",
            f"- Related market-control note: `{destination / 'market_control_sensitivity.md'}`",
            f"- Related market-interaction note: `{destination / 'market_interaction_sensitivity.md'}`",
            f"- Related auxiliary no-time-FE note: `{destination / 'market_aux_no_time_fe.md'}`",
            "- NY Fed quarterly market overlays are useful narrative context for the broader Treasury environment.",
            "- Free public TRACE context now reaches back into the event-study window through FINRA's weekly archive, with newer quarters coming from the current monthly files.",
            "- Raw quarter-level market controls remain outside the baseline 2020 DiD because they are absorbed by quarter fixed effects.",
            "- The repo now reports a separate interaction sensitivity that asks whether treated-minus-control gaps widen or narrow when market-level Treasury conditions are higher.",
            "- The repo also writes a weaker auxiliary no-time-FE check that lets raw market levels enter directly alongside an entity-fixed-effects trend specification.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_market_controls(panel_path: Path | None = None) -> pd.DataFrame | None:
    source = panel_path or derived_data_path("market_overlay_panel.parquet")
    if not source.exists():
        return None
    market = read_table(source).copy()
    market["quarter_end"] = pd.to_datetime(market["quarter_end"])
    selected = [column for column in MARKET_CONTROL_COLUMNS if column in market.columns]
    if not selected:
        return None
    return market[["quarter_end", *selected]].drop_duplicates("quarter_end")


def _prepare_market_interactions(
    frame: pd.DataFrame,
    *,
    market_panel_path: Path | None = None,
) -> pd.DataFrame | None:
    market = _load_market_controls(market_panel_path)
    if market is None:
        return None
    merged = frame.merge(market, how="left", on="quarter_end")
    added_any = False
    for column in MARKET_CONTROL_COLUMNS:
        if column not in merged.columns:
            continue
        values = pd.to_numeric(merged[column], errors="coerce")
        mean = values.mean(skipna=True)
        std = values.std(skipna=True, ddof=0)
        z_col = f"{column}_z"
        if pd.isna(std) or std == 0:
            merged[z_col] = pd.NA
            continue
        merged[z_col] = (values - mean) / std
        added_any = True
    return merged if added_any else None


def _write_market_control_sensitivity_note(
    prepared: pd.DataFrame,
    output_path: Path,
    *,
    market_panel_path: Path | None = None,
) -> None:
    market = _load_market_controls(market_panel_path)
    lines = [
        "# Market Control Sensitivity",
        "",
        "## Decision",
        "- No separate market-control regression is reported for the baseline event-study design.",
    ]

    if market is None:
        lines.extend(
            [
                "",
                "## Reason",
                "- Market overlay data was not available, so the absorption check could not be run.",
            ]
        )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    merged = prepared.merge(market, how="left", on="quarter_end")
    lines.extend(
        [
            "",
            "## Reason",
            "- The baseline DiD includes quarter fixed effects, so any control that is constant across banks within a quarter is absorbed by those fixed effects.",
            "- The candidate controls below are quarter-level market series shared by all banks in the sample.",
            "",
            "## Candidate controls",
        ]
    )

    all_absorbed = True
    for column, label in MARKET_CONTROL_COLUMNS.items():
        if column not in merged.columns:
            lines.append(f"- `{column}` ({label}): not available in the merged panel")
            all_absorbed = False
            continue
        within_quarter_unique = merged.groupby("quarter_end")[column].nunique(dropna=True)
        min_unique = int(within_quarter_unique.min()) if not within_quarter_unique.empty else 0
        max_unique = int(within_quarter_unique.max()) if not within_quarter_unique.empty else 0
        observed_quarters = int(merged.groupby("quarter_end")[column].apply(lambda s: s.notna().any()).sum())
        lines.append(
            f"- `{column}` ({label}): observed in {observed_quarters} quarters; within-quarter unique values min={min_unique}, max={max_unique}"
        )
        if max_unique > 1:
            all_absorbed = False

    lines.extend(
        [
            "",
            "## Implication",
            "- Because the candidate market controls do not vary within quarter, adding them alongside `C(quarter_end)` would not identify a new coefficient or change the fitted treatment effect meaningfully.",
        ]
    )
    if all_absorbed:
        lines.append("- The repo therefore keeps these market series as context variables rather than pretending to estimate a separate fixed-effects control specification.")
    else:
        lines.append("- At least one control varies within quarter, so a true control sensitivity spec could be added in a future revision.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cluster_groups(frame: pd.DataFrame, cluster_col: str | None) -> pd.Series | None:
    if cluster_col is None or cluster_col not in frame.columns:
        return None
    groups = frame[cluster_col].astype("string").replace({"": pd.NA})
    if groups.nunique(dropna=True) < 2:
        return None
    return groups.fillna("entity:" + frame["entity_id"].astype(str))


def _fit_model(formula: str, sample: pd.DataFrame, *, cluster_col: str | None = None):
    groups = _cluster_groups(sample, cluster_col)
    if groups is None:
        return smf.ols(formula, data=sample).fit(cov_type="HC1"), "HC1", None
    return (
        smf.ols(formula, data=sample).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        ),
        "cluster",
        cluster_col,
    )


def _tercile_labels(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="first")
    quantiles = min(3, max(1, ranked.notna().sum()))
    labels = list(range(quantiles))
    return pd.qcut(ranked, q=quantiles, labels=labels, duplicates="drop")


def prepare_event_2020_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    out = out.sort_values(["entity_id", "quarter_end"]).reset_index(drop=True)

    total_assets = out["total_assets"].replace({0: pd.NA})
    out["ust_inventory_fv_scaled"] = out["ust_inventory_fv"] / total_assets
    out["balances_due_from_fed_scaled"] = out["balances_due_from_fed"] / total_assets
    out["reverse_repos_scaled"] = out["reverse_repos"] / total_assets
    out["trading_assets_scaled"] = out["trading_assets_total"] / total_assets
    out["deposit_growth"] = out.groupby("entity_id")["deposits"].pct_change()
    out["loan_growth"] = out.groupby("entity_id")["loans"].pct_change()
    return out


def _entity_universe_for_manifest(panel: pd.DataFrame) -> pd.DataFrame:
    manifest_columns = [
        "entity_id",
        "entity_name",
        "entity_type",
        "rssd_id",
        "fdic_cert",
        "top_parent_rssd",
        "country",
        "slr_scope_class",
        "is_covered_bank_subsidiary",
    ]
    out = panel.copy()
    if out.empty:
        return pd.DataFrame(columns=manifest_columns)
    out["quarter_end"] = pd.to_datetime(out["quarter_end"], errors="coerce")
    for column in manifest_columns:
        if column not in out.columns:
            out[column] = pd.NA
    out = out.sort_values(["entity_id", "quarter_end"]).drop_duplicates("entity_id", keep="last")
    return out[manifest_columns].reset_index(drop=True)


def _event_window(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["quarter_end"] = pd.to_datetime(out["quarter_end"], errors="coerce")
    return out[(out["quarter_end"] >= EVENT_WINDOW_START) & (out["quarter_end"] <= EVENT_WINDOW_END)].copy()


def _baseline_ready_entities(frame: pd.DataFrame) -> set[str]:
    baseline = frame[frame["quarter_end"] == BASELINE_QUARTER].copy()
    if baseline.empty:
        return set()
    mask = (
        pd.to_numeric(baseline.get("tier1_capital"), errors="coerce").gt(0)
        & pd.to_numeric(baseline.get("total_leverage_exposure"), errors="coerce").gt(0)
        & pd.to_numeric(baseline.get("headroom_pp"), errors="coerce").notna()
        & pd.to_numeric(baseline.get("ust_share_assets"), errors="coerce").notna()
    )
    return set(baseline.loc[mask, "entity_id"].astype(str))


def _coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["entity_id"])
    out = frame.copy()
    out["is_pre"] = out["quarter_end"] <= PRE_TREATMENT_END
    out["is_post"] = out["quarter_end"] >= POST_TREATMENT_START
    summary = (
        out.groupby("entity_id", dropna=False)
        .agg(
            sample_observations=("quarter_end", "size"),
            sample_min_quarter=("quarter_end", "min"),
            sample_max_quarter=("quarter_end", "max"),
            pre_treatment_quarters=("is_pre", "sum"),
            post_treatment_quarters=("is_post", "sum"),
        )
        .reset_index()
    )
    summary["has_full_event_window_coverage"] = summary["sample_observations"].eq(
        len(pd.date_range(EVENT_WINDOW_START, EVENT_WINDOW_END, freq="QE-DEC"))
    )
    summary["has_expanded_event_window_coverage"] = (
        summary["pre_treatment_quarters"].ge(MIN_PRE_QUARTERS_EXPANDED)
        & summary["post_treatment_quarters"].ge(MIN_POST_QUARTERS_EXPANDED)
    )
    return summary


def _filter_entities(frame: pd.DataFrame, entity_ids: set[str]) -> pd.DataFrame:
    if not entity_ids:
        return frame.iloc[0:0].copy()
    return frame[frame["entity_id"].astype(str).isin(entity_ids)].copy()


def _baseline_parent_selection(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame["quarter_end"] == BASELINE_QUARTER].copy()
    if baseline.empty:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "flagship_parent_key",
                "flagship_parent_rank_2019q4",
                "flagship_parent_representative_entity_id",
            ]
        )
    baseline["flagship_parent_key"] = (
        baseline.get("top_parent_rssd", pd.Series(pd.NA, index=baseline.index, dtype="string"))
        .astype("string")
        .replace({"": pd.NA})
        .fillna("entity:" + baseline["entity_id"].astype(str))
    )
    ranked = baseline.sort_values(
        ["flagship_parent_key", "total_assets", "entity_id"],
        ascending=[True, False, True],
    ).copy()
    ranked["flagship_parent_rank_2019q4"] = ranked.groupby("flagship_parent_key", dropna=False).cumcount() + 1
    ranked["flagship_parent_representative_entity_id"] = ranked.groupby("flagship_parent_key", dropna=False)["entity_id"].transform("first")
    return ranked[
        [
            "entity_id",
            "flagship_parent_key",
            "flagship_parent_rank_2019q4",
            "flagship_parent_representative_entity_id",
        ]
    ].copy()


def _build_sample_manifest(
    full_panel: pd.DataFrame,
    universe_b: pd.DataFrame,
    universe_c: pd.DataFrame,
    universe_d: pd.DataFrame,
    universe_e: pd.DataFrame,
    universe_f_primary: pd.DataFrame,
    universe_f_expanded: pd.DataFrame,
) -> pd.DataFrame:
    manifest = _entity_universe_for_manifest(full_panel)
    if manifest.empty:
        return manifest

    full_event = _event_window(full_panel)
    full_summary = _coverage_summary(full_event)
    baseline_ready = pd.DataFrame({"entity_id": sorted(_baseline_ready_entities(full_event))})
    if not baseline_ready.empty:
        baseline_ready["has_usable_2019q4_baseline"] = True

    primary_parent = _baseline_parent_selection(universe_d)
    expanded_parent = _baseline_parent_selection(universe_e).rename(
        columns={
            "flagship_parent_key": "flagship_parent_key_expanded",
            "flagship_parent_rank_2019q4": "flagship_parent_rank_2019q4_expanded",
            "flagship_parent_representative_entity_id": "flagship_parent_representative_entity_id_expanded",
        }
    )

    def obs_summary(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["entity_id", f"{prefix}_observations"])
        return (
            frame.groupby("entity_id", dropna=False)
            .agg(
                **{
                    f"{prefix}_observations": ("quarter_end", "size"),
                    f"{prefix}_min_quarter": ("quarter_end", "min"),
                    f"{prefix}_max_quarter": ("quarter_end", "max"),
                }
            )
            .reset_index()
        )

    manifest = manifest.merge(full_summary, how="left", on="entity_id")
    manifest = manifest.merge(baseline_ready, how="left", on="entity_id")
    manifest = manifest.merge(obs_summary(universe_b, "universe_b"), how="left", on="entity_id")
    manifest = manifest.merge(obs_summary(universe_c, "universe_c"), how="left", on="entity_id")
    manifest = manifest.merge(obs_summary(universe_d, "universe_d"), how="left", on="entity_id")
    manifest = manifest.merge(obs_summary(universe_e, "universe_e"), how="left", on="entity_id")
    manifest = manifest.merge(obs_summary(universe_f_primary, "universe_f_primary"), how="left", on="entity_id")
    manifest = manifest.merge(obs_summary(universe_f_expanded, "universe_f_expanded"), how="left", on="entity_id")
    manifest = manifest.merge(primary_parent, how="left", on="entity_id")
    manifest = manifest.merge(expanded_parent, how="left", on="entity_id")

    manifest["included_universe_a"] = True
    manifest["included_universe_b"] = manifest["universe_b_observations"].fillna(0).astype(int) > 0
    manifest["has_usable_2019q4_baseline"] = manifest["has_usable_2019q4_baseline"].fillna(False).astype(bool)
    manifest["included_universe_c"] = manifest["universe_c_observations"].fillna(0).astype(int) > 0
    manifest["included_universe_d"] = manifest["universe_d_observations"].fillna(0).astype(int) > 0
    manifest["included_universe_e"] = manifest["universe_e_observations"].fillna(0).astype(int) > 0
    manifest["included_universe_f_primary"] = manifest["universe_f_primary_observations"].fillna(0).astype(int) > 0
    manifest["included_universe_f_expanded"] = manifest["universe_f_expanded_observations"].fillna(0).astype(int) > 0

    manifest["universe_b_exclusion_reason"] = pd.NA
    manifest.loc[~manifest["included_universe_b"], "universe_b_exclusion_reason"] = "no_positive_tier1_and_tle_in_event_window"

    manifest["universe_c_exclusion_reason"] = pd.NA
    manifest.loc[
        manifest["included_universe_b"] & ~manifest["included_universe_c"],
        "universe_c_exclusion_reason",
    ] = "missing_usable_2019q4_treatment_baseline"

    manifest["universe_d_exclusion_reason"] = pd.NA
    manifest.loc[
        manifest["included_universe_c"] & ~manifest["included_universe_d"],
        "universe_d_exclusion_reason",
    ] = "incomplete_2019q1_2021q4_coverage"

    manifest["universe_e_exclusion_reason"] = pd.NA
    manifest.loc[
        manifest["included_universe_c"] & ~manifest["included_universe_e"],
        "universe_e_exclusion_reason",
    ] = "insufficient_pre_or_post_event_coverage"

    manifest["universe_f_primary_exclusion_reason"] = pd.NA
    primary_missing = manifest["included_universe_d"] & ~manifest["included_universe_f_primary"]
    manifest.loc[
        primary_missing & manifest["flagship_parent_rank_2019q4"].gt(1),
        "universe_f_primary_exclusion_reason",
    ] = "not_largest_2019q4_subsidiary_in_parent_family"
    manifest.loc[
        primary_missing & manifest["universe_f_primary_exclusion_reason"].isna(),
        "universe_f_primary_exclusion_reason",
    ] = "not_selected_in_primary_flagship_sample"

    manifest["universe_f_expanded_exclusion_reason"] = pd.NA
    expanded_missing = manifest["included_universe_e"] & ~manifest["included_universe_f_expanded"]
    manifest.loc[
        expanded_missing & manifest["flagship_parent_rank_2019q4_expanded"].gt(1),
        "universe_f_expanded_exclusion_reason",
    ] = "not_largest_2019q4_subsidiary_in_parent_family"
    manifest.loc[
        expanded_missing & manifest["universe_f_expanded_exclusion_reason"].isna(),
        "universe_f_expanded_exclusion_reason",
    ] = "not_selected_in_expanded_flagship_sample"

    for column in manifest.columns:
        if column.endswith("_observations") or column.endswith("_quarters") or column.endswith("_2019q4"):
            if column in manifest.columns:
                try:
                    manifest[column] = manifest[column].astype("Int64")
                except TypeError:
                    pass
    for column in manifest.columns:
        if column.endswith("_quarter") and column in manifest.columns:
            manifest[column] = pd.to_datetime(manifest[column], errors="coerce").dt.strftime("%Y-%m-%d")

    return manifest.sort_values(
        ["included_universe_d", "included_universe_e", "included_universe_f_primary", "entity_name", "entity_id"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def _format_reason_counts(series: pd.Series) -> str:
    reasons = series.dropna().astype(str)
    if reasons.empty:
        return "none"
    counts = reasons.value_counts().sort_index()
    return "; ".join(f"{reason}={count}" for reason, count in counts.items())


def _write_sample_manifest_summary(manifest: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Event Study Sample Manifest",
        "",
        "## Universe ladder",
        f"- Universe A (all insured-bank filers in the descriptive panel): {int(manifest['included_universe_a'].sum())}",
        f"- Universe B (SLR-reporting insured banks in the event window): {int(manifest['included_universe_b'].sum())}",
        f"- Universe C (treatment-definable 2019Q4 baseline): {int(manifest['included_universe_c'].sum())}",
        f"- Universe D (primary causal core, balanced 2019Q1-2021Q4): {int(manifest['included_universe_d'].sum())}",
        f"- Universe E (expanded causal sensitivity): {int(manifest['included_universe_e'].sum())}",
        f"- Universe F primary (largest 2019Q4 insured subsidiary per family inside Universe D): {int(manifest['included_universe_f_primary'].sum())}",
        f"- Universe F expanded (largest 2019Q4 insured subsidiary per family inside Universe E): {int(manifest['included_universe_f_expanded'].sum())}",
        "",
        "## Exclusion summary",
        f"- Universe B exclusions: {_format_reason_counts(manifest['universe_b_exclusion_reason'])}",
        f"- Universe C exclusions: {_format_reason_counts(manifest['universe_c_exclusion_reason'])}",
        f"- Universe D exclusions: {_format_reason_counts(manifest['universe_d_exclusion_reason'])}",
        f"- Universe E exclusions: {_format_reason_counts(manifest['universe_e_exclusion_reason'])}",
        f"- Universe F primary exclusions: {_format_reason_counts(manifest['universe_f_primary_exclusion_reason'])}",
        f"- Universe F expanded exclusions: {_format_reason_counts(manifest['universe_f_expanded_exclusion_reason'])}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_flagship_per_parent(frame: pd.DataFrame, baseline_quarter: str = "2019-12-31") -> pd.DataFrame:
    out = frame.copy()
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    parent_key = out.get("top_parent_rssd", pd.Series(pd.NA, index=out.index, dtype="string"))
    parent_key = parent_key.astype("string").replace({"": pd.NA})
    out["sample_parent_key"] = parent_key.fillna("entity:" + out["entity_id"].astype(str))

    baseline = out[out["quarter_end"] == pd.Timestamp(baseline_quarter)].copy()
    if baseline.empty:
        return out

    selected = (
        baseline.sort_values(
            ["sample_parent_key", "total_assets", "entity_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("sample_parent_key", keep="first")
    )
    selected_ids = set(selected["entity_id"])
    return out[out["entity_id"].isin(selected_ids)].drop(columns=["sample_parent_key"])


def add_treatments(frame: pd.DataFrame, baseline_quarter: str = "2019-12-31") -> pd.DataFrame:
    out = frame.copy()
    baseline_mask = out["quarter_end"] == pd.Timestamp(baseline_quarter)
    baseline = out.loc[baseline_mask, ["entity_id", "headroom_pp", "ust_share_assets", "is_covered_bank_subsidiary"]].copy()

    for treatment in TREATMENTS:
        if treatment.mode == "boolean":
            values = baseline[["entity_id", treatment.source_column]].copy()
            values[treatment.name] = values[treatment.source_column].astype(bool).astype(int)
            out = out.merge(values[["entity_id", treatment.name]], how="left", on="entity_id")
            continue

        terciles = _tercile_labels(baseline[treatment.source_column])
        values = baseline[["entity_id"]].copy()
        if treatment.mode == "bottom_tercile":
            values[treatment.name] = (terciles == 0).astype(int)
        else:
            values[treatment.name] = (terciles == 2).astype(int)
        out = out.merge(values, how="left", on="entity_id")
    return out


def _run_did(frame: pd.DataFrame, treatment: str, outcome: str, *, cluster_col: str | None = None) -> pd.DataFrame:
    spec = EventStudySpec(outcome=outcome, treatment=treatment)
    regression_frame = add_event_dummies(frame, spec)
    selected_columns = [outcome, treatment, "treated_post", "entity_id", "quarter_end"]
    if cluster_col and cluster_col in regression_frame.columns:
        selected_columns.append(cluster_col)
    sample = regression_frame[selected_columns].dropna()
    if sample.empty:
        return pd.DataFrame()
    model, cov_type, model_cluster_col = _fit_model(
        f"{outcome} ~ treated_post + C(entity_id) + C(quarter_end)",
        sample,
        cluster_col=cluster_col,
    )
    groups = _cluster_groups(sample, cluster_col)
    return pd.DataFrame(
        [
            {
                "treatment": treatment,
                "outcome": outcome,
                "coef": model.params.get("treated_post"),
                "std_err": model.bse.get("treated_post"),
                "pvalue": model.pvalues.get("treated_post"),
                "nobs": model.nobs,
                "cov_type": cov_type,
                "cluster_col": model_cluster_col,
                "n_clusters": groups.nunique() if groups is not None else pd.NA,
            }
        ]
    )


def _run_event_time(frame: pd.DataFrame, treatment: str, outcome: str, *, cluster_col: str | None = None) -> pd.DataFrame:
    spec = EventStudySpec(outcome=outcome, treatment=treatment)
    regression_frame = add_event_dummies(frame, spec)
    interaction_columns: list[str] = []
    for term in event_study_terms(spec):
        interaction = f"{treatment}_{term}"
        regression_frame[interaction] = regression_frame[term] * regression_frame[treatment]
        interaction_columns.append(interaction)

    selected_columns = [outcome, "entity_id", "quarter_end", treatment, *interaction_columns]
    if cluster_col and cluster_col in regression_frame.columns:
        selected_columns.append(cluster_col)
    sample = regression_frame[selected_columns].dropna()
    if sample.empty:
        return pd.DataFrame()

    formula = f"{outcome} ~ {' + '.join(interaction_columns)} + C(entity_id) + C(quarter_end)"
    model, cov_type, model_cluster_col = _fit_model(formula, sample, cluster_col=cluster_col)
    groups = _cluster_groups(sample, cluster_col)

    rows = []
    for term in interaction_columns:
        rows.append(
            {
                "treatment": treatment,
                "outcome": outcome,
                "term": term,
                "coef": model.params.get(term),
                "std_err": model.bse.get(term),
                "pvalue": model.pvalues.get(term),
                "cov_type": cov_type,
                "cluster_col": model_cluster_col,
                "n_clusters": groups.nunique() if groups is not None else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _run_market_interaction_did(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    market_column: str,
    *,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    spec = EventStudySpec(outcome=outcome, treatment=treatment)
    regression_frame = add_event_dummies(frame, spec)
    interaction_col = f"{treatment}_{market_column}_treated_post"
    regression_frame[interaction_col] = regression_frame["treated_post"] * regression_frame[market_column]
    selected_columns = [outcome, "treated_post", interaction_col, "entity_id", "quarter_end"]
    if cluster_col and cluster_col in regression_frame.columns:
        selected_columns.append(cluster_col)
    sample = regression_frame[selected_columns].dropna()
    if sample.empty:
        return pd.DataFrame()
    model, cov_type, model_cluster_col = _fit_model(
        f"{outcome} ~ treated_post + {interaction_col} + C(entity_id) + C(quarter_end)",
        sample,
        cluster_col=cluster_col,
    )
    groups = _cluster_groups(sample, cluster_col)
    return pd.DataFrame(
        [
            {
                "treatment": treatment,
                "outcome": outcome,
                "market_column": market_column,
                "coef_treated_post": model.params.get("treated_post"),
                "pvalue_treated_post": model.pvalues.get("treated_post"),
                "coef_interaction": model.params.get(interaction_col),
                "std_err_interaction": model.bse.get(interaction_col),
                "pvalue_interaction": model.pvalues.get(interaction_col),
                "cov_type": cov_type,
                "cluster_col": model_cluster_col,
                "nobs": model.nobs,
                "n_clusters": groups.nunique() if groups is not None else pd.NA,
            }
        ]
    )


def _run_market_aux_no_time_fe_did(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    market_column: str,
    *,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    spec = EventStudySpec(outcome=outcome, treatment=treatment)
    regression_frame = add_event_dummies(frame, spec)
    interaction_col = f"{treatment}_{market_column}_treated_post"
    regression_frame[interaction_col] = regression_frame["treated_post"] * regression_frame[market_column]
    selected_columns = [outcome, "treated_post", "post", "event_quarter", market_column, interaction_col, "entity_id", "quarter_end"]
    if cluster_col and cluster_col in regression_frame.columns:
        selected_columns.append(cluster_col)
    sample = regression_frame[selected_columns].dropna()
    if sample.empty:
        return pd.DataFrame()
    model, cov_type, model_cluster_col = _fit_model(
        f"{outcome} ~ treated_post + post + event_quarter + {market_column} + {interaction_col} + C(entity_id)",
        sample,
        cluster_col=cluster_col,
    )
    groups = _cluster_groups(sample, cluster_col)
    return pd.DataFrame(
        [
            {
                "treatment": treatment,
                "outcome": outcome,
                "market_column": market_column,
                "coef_treated_post": model.params.get("treated_post"),
                "pvalue_treated_post": model.pvalues.get("treated_post"),
                "coef_market": model.params.get(market_column),
                "pvalue_market": model.pvalues.get(market_column),
                "coef_interaction": model.params.get(interaction_col),
                "std_err_interaction": model.bse.get(interaction_col),
                "pvalue_interaction": model.pvalues.get(interaction_col),
                "cov_type": cov_type,
                "cluster_col": model_cluster_col,
                "nobs": model.nobs,
                "n_clusters": groups.nunique() if groups is not None else pd.NA,
            }
        ]
    )


def _run_pretrend_test(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    *,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    spec = EventStudySpec(outcome=outcome, treatment=treatment)
    regression_frame = add_event_dummies(frame, spec)
    pre_terms = [term for term in event_study_terms(spec) if term.startswith("event_m")]
    if not pre_terms:
        return pd.DataFrame()

    interaction_columns: list[str] = []
    for term in pre_terms:
        interaction = f"{treatment}_{term}"
        regression_frame[interaction] = regression_frame[term] * regression_frame[treatment]
        interaction_columns.append(interaction)

    selected_columns = [outcome, "entity_id", "quarter_end", treatment, *interaction_columns]
    if cluster_col and cluster_col in regression_frame.columns:
        selected_columns.append(cluster_col)
    sample = regression_frame[selected_columns].dropna()
    if sample.empty:
        return pd.DataFrame()

    formula = f"{outcome} ~ {' + '.join(interaction_columns)} + C(entity_id) + C(quarter_end)"
    model, cov_type, model_cluster_col = _fit_model(formula, sample, cluster_col=cluster_col)
    hypothesis = ", ".join(f"{term} = 0" for term in interaction_columns)
    try:
        joint_test = model.f_test(hypothesis)
        joint_pvalue = float(joint_test.pvalue)
        joint_f = float(joint_test.fvalue)
    except Exception:
        joint_pvalue = pd.NA
        joint_f = pd.NA
    coef_values = pd.Series({term: model.params.get(term) for term in interaction_columns}, dtype="float64")
    return pd.DataFrame(
        [
            {
                "treatment": treatment,
                "outcome": outcome,
                "pretrend_joint_f": joint_f,
                "pretrend_joint_pvalue": joint_pvalue,
                "max_abs_pre_coef": float(coef_values.abs().max()) if not coef_values.empty else pd.NA,
                "mean_abs_pre_coef": float(coef_values.abs().mean()) if not coef_values.empty else pd.NA,
                "nobs": model.nobs,
                "cov_type": cov_type,
                "cluster_col": model_cluster_col,
            }
        ]
    )


def _run_placebo_fake_date_did(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    *,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    placebo_frame = frame.copy()
    placebo_frame["quarter_end"] = pd.to_datetime(placebo_frame["quarter_end"], errors="coerce")
    placebo_frame = placebo_frame[placebo_frame["quarter_end"] <= PLACEBO_SAMPLE_END].copy()
    if placebo_frame.empty:
        return pd.DataFrame()
    spec = EventStudySpec(outcome=outcome, treatment=treatment, shock_date=PLACEBO_FAKE_SHOCK_DATE)
    regression_frame = add_event_dummies(placebo_frame, spec)
    selected_columns = [outcome, treatment, "treated_post", "entity_id", "quarter_end"]
    if cluster_col and cluster_col in regression_frame.columns:
        selected_columns.append(cluster_col)
    sample = regression_frame[selected_columns].dropna()
    if sample.empty:
        return pd.DataFrame()
    model, cov_type, model_cluster_col = _fit_model(
        f"{outcome} ~ treated_post + C(entity_id) + C(quarter_end)",
        sample,
        cluster_col=cluster_col,
    )
    return pd.DataFrame(
        [
            {
                "treatment": treatment,
                "outcome": outcome,
                "coef": model.params.get("treated_post"),
                "std_err": model.bse.get("treated_post"),
                "pvalue": model.pvalues.get("treated_post"),
                "nobs": model.nobs,
                "cov_type": cov_type,
                "cluster_col": model_cluster_col,
                "placebo_shock_date": PLACEBO_FAKE_SHOCK_DATE.isoformat(),
                "placebo_sample_end": PLACEBO_SAMPLE_END.strftime("%Y-%m-%d"),
            }
        ]
    )


def _plot_event_time(frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = frame.copy()
    plot_frame["event_quarter"] = (
        plot_frame["term"]
        .str.replace(".*_event_", "", regex=True)
        .str.replace("m", "-", regex=False)
        .str.replace("p", "", regex=False)
        .astype(int)
    )
    plot_frame = plot_frame.sort_values("event_quarter")
    ci = 1.96 * plot_frame["std_err"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_frame["event_quarter"], plot_frame["coef"], marker="o")
    ax.fill_between(plot_frame["event_quarter"], plot_frame["coef"] - ci, plot_frame["coef"] + ci, alpha=0.2)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(-1, color="gray", linestyle="--", linewidth=1)
    ax.set_title("2020 temporary exclusion event-study coefficients")
    ax.set_xlabel("Event quarter")
    ax.set_ylabel("Coefficient")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _run_event_2020_sample(
    prepared: pd.DataFrame,
    destination: Path,
    *,
    cluster_col: str | None = None,
    sample_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(prepared, destination / "prepared_panel.csv")
    if prepared.empty or any(treatment.name not in prepared.columns for treatment in TREATMENTS):
        write_frame(pd.DataFrame(), destination / "did_results.csv")
        write_frame(pd.DataFrame(), destination / "event_time_coefficients.csv")
        summary_lines = [
            "# 2020 Temporary Exclusion Event Study",
            "",
            f"- Sample: {sample_label or destination.name}",
            f"- Observations: {len(prepared)}",
            f"- Distinct entities: {prepared['entity_id'].nunique() if 'entity_id' in prepared.columns else 0}",
            f"- Covariance: {'parent-clustered' if cluster_col else 'HC1'}",
            f"- Cluster column: {cluster_col if cluster_col else 'n/a'}",
            "- DID specifications run: 0",
            "- Event-time coefficient sets: 0",
            "- Status: sample empty or treatment assignments unavailable.",
        ]
        (destination / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        return pd.DataFrame(), pd.DataFrame()
    did_results = []
    event_results = []
    for treatment in [item.name for item in TREATMENTS]:
        for outcome in OUTCOMES:
            did = _run_did(prepared, treatment, outcome, cluster_col=cluster_col)
            if not did.empty:
                did_results.append(did)
            event_time = _run_event_time(prepared, treatment, outcome, cluster_col=cluster_col)
            if not event_time.empty:
                event_results.append(event_time)

    did_frame = pd.concat(did_results, ignore_index=True) if did_results else pd.DataFrame()
    event_frame = pd.concat(event_results, ignore_index=True) if event_results else pd.DataFrame()
    write_frame(did_frame, destination / "did_results.csv")
    write_frame(event_frame, destination / "event_time_coefficients.csv")

    baseline_plot = event_frame[
        (event_frame["treatment"] == "low_headroom_treated")
        & (event_frame["outcome"] == "ust_inventory_fv_scaled")
    ]
    if not baseline_plot.empty:
        _plot_event_time(baseline_plot, destination / "low_headroom_ust_inventory_event_time.png")

    summary_lines = [
        "# 2020 Temporary Exclusion Event Study",
        "",
        f"- Sample: {sample_label or destination.name}",
        f"- Observations: {len(prepared)}",
        f"- Distinct entities: {prepared['entity_id'].nunique()}",
        f"- Covariance: {'parent-clustered' if cluster_col else 'HC1'}",
        f"- Cluster column: {cluster_col if cluster_col else 'n/a'}",
        f"- DID specifications run: {len(did_frame)}",
        f"- Event-time coefficient sets: {event_frame[['treatment', 'outcome']].drop_duplicates().shape[0] if not event_frame.empty else 0}",
    ]
    (destination / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return did_frame, event_frame


def _write_sample_comparison(
    primary_prepared: pd.DataFrame,
    primary_did: pd.DataFrame,
    expanded_prepared: pd.DataFrame,
    expanded_did: pd.DataFrame,
    flagship_primary_prepared: pd.DataFrame,
    flagship_primary_did: pd.DataFrame,
    flagship_primary_clustered_did: pd.DataFrame,
    historical_prepared: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    primary_lookup = primary_did.set_index(["treatment", "outcome"]) if not primary_did.empty else pd.DataFrame()
    expanded_lookup = expanded_did.set_index(["treatment", "outcome"]) if not expanded_did.empty else pd.DataFrame()
    flagship_lookup = (
        flagship_primary_did.set_index(["treatment", "outcome"]) if not flagship_primary_did.empty else pd.DataFrame()
    )
    clustered_lookup = (
        flagship_primary_clustered_did.set_index(["treatment", "outcome"]) if not flagship_primary_clustered_did.empty else pd.DataFrame()
    )

    lines = [
        "# Event Study Sample Comparison",
        "",
        "## Sample sizes",
        f"- Historical treatment-definable sample (Universe C) observations: {len(historical_prepared)}",
        f"- Historical treatment-definable sample (Universe C) entities: {historical_prepared['entity_id'].nunique()}",
        f"- Primary causal core (Universe D) observations: {len(primary_prepared)}",
        f"- Primary causal core (Universe D) entities: {primary_prepared['entity_id'].nunique()}",
        f"- Expanded causal sensitivity (Universe E) observations: {len(expanded_prepared)}",
        f"- Expanded causal sensitivity (Universe E) entities: {expanded_prepared['entity_id'].nunique()}",
        f"- Flagship primary (Universe F on D) observations: {len(flagship_primary_prepared)}",
        f"- Flagship primary (Universe F on D) entities: {flagship_primary_prepared['entity_id'].nunique()}",
        f"- Manifest rows: {len(sample_manifest)}",
        f"- Universe B entities with a usable 2019Q4 treatment baseline: {int(sample_manifest['included_universe_c'].sum())}",
        "",
        "## Key DID rows",
    ]
    for treatment, outcome in KEY_COMPARISON_ROWS:
        if (
            (treatment, outcome) not in primary_lookup.index
            or (treatment, outcome) not in expanded_lookup.index
            or (treatment, outcome) not in flagship_lookup.index
            or (treatment, outcome) not in clustered_lookup.index
        ):
            continue
        primary_row = primary_lookup.loc[(treatment, outcome)]
        expanded_row = expanded_lookup.loc[(treatment, outcome)]
        flagship_row = flagship_lookup.loc[(treatment, outcome)]
        clustered_row = clustered_lookup.loc[(treatment, outcome)]
        lines.append(
            f"- `{treatment}` / `{outcome}`: "
            f"primary coef={primary_row['coef']:.4f}, p={primary_row['pvalue']:.4f}; "
            f"expanded coef={expanded_row['coef']:.4f}, p={expanded_row['pvalue']:.4f}; "
            f"flagship coef={flagship_row['coef']:.4f}, p={flagship_row['pvalue']:.4f}; "
            f"flagship clustered coef={clustered_row['coef']:.4f}, p={clustered_row['pvalue']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Transparency artifacts",
            f"- Detailed sample manifest: `{output_path.parent / 'sample_manifest.csv'}`",
            f"- Human-readable manifest summary: `{output_path.parent / 'sample_manifest.md'}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_ladder_row(
    sample_name: str,
    sample_label: str,
    frame: pd.DataFrame,
    inclusion_rule: str,
    exclusion_summary: str,
) -> dict[str, object]:
    out: dict[str, object] = {
        "sample_name": sample_name,
        "sample_label": sample_label,
        "entity_count": int(frame["entity_id"].nunique()) if not frame.empty else 0,
        "observation_count": int(len(frame)),
        "quarter_span": "n/a",
        "parent_family_count": int(
            frame.get("top_parent_rssd", pd.Series(dtype="string")).astype("string").replace({"": pd.NA}).dropna().nunique()
        ) if not frame.empty and "top_parent_rssd" in frame.columns else 0,
        "balanced_panel": bool(frame.groupby("entity_id").size().nunique() == 1) if not frame.empty else False,
        "inclusion_rule": inclusion_rule,
        "exclusion_summary": exclusion_summary,
    }
    if not frame.empty:
        quarters = pd.to_datetime(frame["quarter_end"], errors="coerce")
        out["quarter_span"] = f"{quarters.min().strftime('%Y-%m-%d')} to {quarters.max().strftime('%Y-%m-%d')}"
    return out


def _write_sample_ladder(sample_ladder: pd.DataFrame, output_csv: Path, output_md: Path) -> None:
    write_frame(sample_ladder, output_csv)
    lines = [
        "# Event Study Sample Ladder",
        "",
        "## Samples",
    ]
    for _, row in sample_ladder.iterrows():
        lines.append(
            f"- {row['sample_label']}: entities={row['entity_count']}, observations={row['observation_count']}, "
            f"parents={row['parent_family_count']}, balanced={str(bool(row['balanced_panel'])).lower()}, "
            f"span={row['quarter_span']}; rule={row['inclusion_rule']}; exclusions={row['exclusion_summary']}"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coefficient_summary(lookup: pd.DataFrame, treatment: str, outcome: str) -> str:
    if lookup.empty or (treatment, outcome) not in lookup.index:
        return "not estimable"
    row = lookup.loc[(treatment, outcome)]
    return f"coef={row['coef']:.4f}, p={row['pvalue']:.4f}"


def _write_pretrend_checks(
    prepared: pd.DataFrame,
    output_dir: Path,
    *,
    sample_label: str,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    checks: list[pd.DataFrame] = []
    for treatment in [item.name for item in TREATMENTS]:
        for outcome in OUTCOMES:
            check = _run_pretrend_test(prepared, treatment, outcome, cluster_col=cluster_col)
            if not check.empty:
                check["sample_label"] = sample_label
                checks.append(check)
    frame = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    write_frame(frame, output_dir / "pretrend_checks.csv")

    lines = [
        "# Pre-Trend Checks",
        "",
        f"- Sample: {sample_label}",
        f"- Covariance: {'parent-clustered' if cluster_col else 'HC1'}",
        "",
        "## Key rows",
    ]
    if frame.empty:
        lines.append("- No pre-trend checks were estimable.")
    else:
        focus = frame.set_index(["treatment", "outcome"])
        for treatment, outcome in KEY_COMPARISON_ROWS:
            if (treatment, outcome) not in focus.index:
                continue
            row = focus.loc[(treatment, outcome)]
            lines.append(
                f"- `{treatment}` / `{outcome}`: joint pretrend p={row['pretrend_joint_pvalue']:.4f}, "
                f"max |pre coef|={row['max_abs_pre_coef']:.4f}"
            )
    (output_dir / "pretrend_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def _write_fake_date_placebo(
    prepared: pd.DataFrame,
    output_dir: Path,
    *,
    sample_label: str,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    checks: list[pd.DataFrame] = []
    for treatment in [item.name for item in TREATMENTS]:
        for outcome in OUTCOMES:
            check = _run_placebo_fake_date_did(prepared, treatment, outcome, cluster_col=cluster_col)
            if not check.empty:
                check["sample_label"] = sample_label
                checks.append(check)
    frame = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    write_frame(frame, output_dir / "placebo_fake_date.csv")

    lines = [
        "# Fake-Date Placebo",
        "",
        f"- Sample: {sample_label}",
        f"- Fake shock date: {PLACEBO_FAKE_SHOCK_DATE.isoformat()}",
        f"- Pre-policy sample end: {PLACEBO_SAMPLE_END.strftime('%Y-%m-%d')}",
        "",
        "## Key rows",
    ]
    if frame.empty:
        lines.append("- No fake-date placebo checks were estimable.")
    else:
        focus = frame.set_index(["treatment", "outcome"])
        for treatment, outcome in KEY_COMPARISON_ROWS:
            if (treatment, outcome) not in focus.index:
                continue
            row = focus.loc[(treatment, outcome)]
            lines.append(
                f"- `{treatment}` / `{outcome}`: placebo coef={row['coef']:.4f}, p={row['pvalue']:.4f}"
            )
    (output_dir / "placebo_fake_date.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def _write_methodology_memo(
    sample_ladder: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    primary_did: pd.DataFrame,
    expanded_did: pd.DataFrame,
    flagship_clustered_did: pd.DataFrame,
    primary_pretrend: pd.DataFrame,
    flagship_placebo: pd.DataFrame,
    output_path: Path,
) -> None:
    primary_lookup = primary_did.set_index(["treatment", "outcome"]) if not primary_did.empty else pd.DataFrame()
    expanded_lookup = expanded_did.set_index(["treatment", "outcome"]) if not expanded_did.empty else pd.DataFrame()
    flagship_lookup = (
        flagship_clustered_did.set_index(["treatment", "outcome"]) if not flagship_clustered_did.empty else pd.DataFrame()
    )
    excluded_b = sample_manifest.loc[sample_manifest["universe_b_exclusion_reason"].notna(), ["entity_name", "entity_id", "universe_b_exclusion_reason"]]
    excluded_c = sample_manifest.loc[sample_manifest["universe_c_exclusion_reason"].notna(), ["entity_name", "entity_id", "universe_c_exclusion_reason"]]
    excluded_d = sample_manifest.loc[sample_manifest["universe_d_exclusion_reason"].notna(), ["entity_name", "entity_id", "universe_d_exclusion_reason"]]
    lines = [
        "# SLR Exemption Sample Memo",
        "",
        "## Current posture",
        "- The descriptive universe is now all insured-bank Call Report filers observed in the staged FFIEC bulk data, not only the legacy curated subset.",
        "- The causal event-study sample is restricted in steps: SLR-reporting banks, then treatment-definable 2019Q4 baseline banks, then a balanced event-window core.",
        "",
        "## Sample ladder",
    ]
    for _, row in sample_ladder.iterrows():
        lines.append(f"- {row['sample_label']}: {row['entity_count']} entities / {row['observation_count']} observations")
    lines.extend(
        [
            "",
            "## Explicit exclusions",
            f"- Universe B exclusions: {_format_reason_counts(sample_manifest['universe_b_exclusion_reason'])}",
            f"- Universe C exclusions: {_format_reason_counts(sample_manifest['universe_c_exclusion_reason'])}",
            f"- Universe D exclusions: {_format_reason_counts(sample_manifest['universe_d_exclusion_reason'])}",
        ]
    )
    for label, frame, column in [
        ("Universe B", excluded_b, "universe_b_exclusion_reason"),
        ("Universe C", excluded_c, "universe_c_exclusion_reason"),
        ("Universe D", excluded_d, "universe_d_exclusion_reason"),
    ]:
        if frame.empty:
            continue
        lines.append(f"- {label} named exclusions: " + "; ".join(
            f"{row['entity_name']} (`{row['entity_id']}`) -> {row[column]}"
            for _, row in frame.sort_values('entity_name').head(25).iterrows()
        ))
    lines.extend(
        [
            "",
            "## Stability check",
            f"- Low-headroom Treasury result, Universe D: {_coefficient_summary(primary_lookup, 'low_headroom_treated', 'ust_inventory_fv_scaled')}",
            f"- Low-headroom Treasury result, Universe E: {_coefficient_summary(expanded_lookup, 'low_headroom_treated', 'ust_inventory_fv_scaled')}",
            f"- Low-headroom Treasury result, Universe F clustered: {_coefficient_summary(flagship_lookup, 'low_headroom_treated', 'ust_inventory_fv_scaled')}",
            f"- Covered-bank Treasury result, Universe D: {_coefficient_summary(primary_lookup, 'covered_bank_treated', 'ust_inventory_fv_scaled')}",
            f"- Covered-bank Treasury result, Universe E: {_coefficient_summary(expanded_lookup, 'covered_bank_treated', 'ust_inventory_fv_scaled')}",
            f"- Covered-bank Treasury result, Universe F clustered: {_coefficient_summary(flagship_lookup, 'covered_bank_treated', 'ust_inventory_fv_scaled')}",
        ]
    )
    if not primary_pretrend.empty:
        lookup = primary_pretrend.set_index(["treatment", "outcome"])
        lines.extend(
            [
                "",
                "## Diagnostics",
                f"- Universe D low-headroom Treasury pretrend joint p={lookup.loc[('low_headroom_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
                f"- Universe D covered-bank Treasury pretrend joint p={lookup.loc[('covered_bank_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
            ]
        )
    if not flagship_placebo.empty:
        lookup = flagship_placebo.set_index(["treatment", "outcome"])
        lines.extend(
            [
                f"- Universe F clustered low-headroom Treasury fake-date placebo={_coefficient_summary(lookup, 'low_headroom_treated', 'ust_inventory_fv_scaled')}",
                f"- Universe F clustered covered-bank Treasury fake-date placebo={_coefficient_summary(lookup, 'covered_bank_treated', 'ust_inventory_fv_scaled')}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gpt_pro_prompt(
    sample_ladder: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    primary_did: pd.DataFrame,
    expanded_did: pd.DataFrame,
    flagship_clustered_did: pd.DataFrame,
    primary_pretrend: pd.DataFrame,
    flagship_placebo: pd.DataFrame,
    output_path: Path,
) -> None:
    key_rows: list[str] = []
    for label, frame in [
        ("Universe D primary causal core", primary_did),
        ("Universe E expanded sensitivity", expanded_did),
        ("Universe F primary clustered", flagship_clustered_did),
    ]:
        lookup = frame.set_index(["treatment", "outcome"]) if not frame.empty else pd.DataFrame()
        key_rows.append(
            f"{label}: low_headroom_treated × ust_inventory_fv_scaled = {_coefficient_summary(lookup, 'low_headroom_treated', 'ust_inventory_fv_scaled')}; "
            f"covered_bank_treated × ust_inventory_fv_scaled = {_coefficient_summary(lookup, 'covered_bank_treated', 'ust_inventory_fv_scaled')}"
        )

    prompt = [
        "# GPT Pro Prompt",
        "",
        "Use the repo state below to advise on the next methodological steps for identifying the effects of the 2020 temporary SLR exemption on SLR-constrained banks.",
        "",
        "## Sample ladder",
    ]
    for _, row in sample_ladder.iterrows():
        prompt.append(
            f"- {row['sample_label']} ({row['sample_name']}): {row['entity_count']} entities, {row['observation_count']} observations, "
            f"parents={row['parent_family_count']}, balanced={str(bool(row['balanced_panel'])).lower()}, rule={row['inclusion_rule']}, exclusions={row['exclusion_summary']}"
        )
    prompt.extend(
        [
            "",
            "## Exclusion ledger",
            f"- Universe B: {_format_reason_counts(sample_manifest['universe_b_exclusion_reason'])}",
            f"- Universe C: {_format_reason_counts(sample_manifest['universe_c_exclusion_reason'])}",
            f"- Universe D: {_format_reason_counts(sample_manifest['universe_d_exclusion_reason'])}",
            f"- Universe F primary: {_format_reason_counts(sample_manifest['universe_f_primary_exclusion_reason'])}",
            "",
            "## Stability across D/E/F",
            *[f"- {row}" for row in key_rows],
        ]
    )
    prompt.extend(["", "## Diagnostics"])
    if not primary_pretrend.empty:
        lookup = primary_pretrend.set_index(["treatment", "outcome"])
        prompt.extend(
            [
                f"- Universe D low_headroom_treated Treasury pretrend joint p={lookup.loc[('low_headroom_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
                f"- Universe D covered_bank_treated Treasury pretrend joint p={lookup.loc[('covered_bank_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
            ]
        )
    if not flagship_placebo.empty:
        lookup = flagship_placebo.set_index(["treatment", "outcome"])
        prompt.extend(
            [
                f"- Universe F clustered low_headroom_treated Treasury fake-date placebo={_coefficient_summary(lookup, 'low_headroom_treated', 'ust_inventory_fv_scaled')}",
                f"- Universe F clustered covered_bank_treated Treasury fake-date placebo={_coefficient_summary(lookup, 'covered_bank_treated', 'ust_inventory_fv_scaled')}",
            ]
        )
    prompt.extend(
        [
            "",
            "## Open questions",
            "- Is the Universe D balanced-core definition methodologically defensible as the primary causal sample for the SLR exemption analysis?",
            "- Should additional banks from Universe E or from outside the current SLR-reporting filter be in the primary core, or only in sensitivity/descriptive layers?",
            "- What placebo, falsification, or specification checks are most important next?",
            "- Is the statement 'all banks were affected' conceptually wrong, partially true, or in need of a different design than the current SLR event study?",
        ]
    )
    output_path.write_text("\n".join(prompt) + "\n", encoding="utf-8")


def _write_market_interaction_sensitivity(
    prepared: pd.DataFrame,
    output_dir: Path,
    *,
    market_panel_path: Path | None = None,
    cluster_col: str | None = DEFAULT_MARKET_INTERACTION_CLUSTER,
) -> None:
    required_treatments = [item.name for item in TREATMENTS]
    merged = _prepare_market_interactions(prepared, market_panel_path=market_panel_path)
    note_path = output_dir / "market_interaction_sensitivity.md"
    results_path = output_dir / "market_interaction_sensitivity.csv"
    if merged is None or prepared.empty or any(column not in prepared.columns for column in required_treatments):
        note_path.write_text(
            "# Market Interaction Sensitivity\n\n- Market interaction sensitivity was not run because usable market series or treatment assignments were not available.\n",
            encoding="utf-8",
        )
        write_frame(pd.DataFrame(), results_path)
        return

    did_results = []
    available_market_columns = [f"{column}_z" for column in MARKET_CONTROL_COLUMNS if f"{column}_z" in merged.columns]
    for treatment in [item.name for item in TREATMENTS]:
        for outcome in OUTCOMES:
            for market_column in available_market_columns:
                did = _run_market_interaction_did(
                    merged,
                    treatment,
                    outcome,
                    market_column,
                    cluster_col=cluster_col,
                )
                if not did.empty:
                    did_results.append(did)

    results = pd.concat(did_results, ignore_index=True) if did_results else pd.DataFrame()
    write_frame(results, results_path)

    lines = [
        "# Market Interaction Sensitivity",
        "",
        "## Design",
        "- This sensitivity keeps the entity and quarter fixed effects from the baseline DiD.",
        "- It adds one interaction term at a time: `treated_post × standardized_market_level`.",
        "- The coefficient on the interaction term measures whether the treated-minus-control post gap gets larger or smaller when the market series is higher.",
        f"- Covariance: {'parent-clustered' if cluster_col else 'HC1'}",
        f"- Cluster column: {cluster_col if cluster_col else 'n/a'}",
        "",
        "## Key rows",
    ]

    if results.empty:
        lines.append("- No market-interaction results were estimable.")
    else:
        lookup = results.set_index(["treatment", "outcome", "market_column"])
        for treatment, outcome in KEY_COMPARISON_ROWS:
            for market_column in available_market_columns:
                key = (treatment, outcome, market_column)
                if key not in lookup.index:
                    continue
                row = lookup.loc[key]
                lines.append(
                    f"- `{treatment}` / `{outcome}` / `{market_column}`: "
                    f"treated_post coef={row['coef_treated_post']:.4f}, p={row['pvalue_treated_post']:.4f}; "
                    f"interaction coef={row['coef_interaction']:.4f}, p={row['pvalue_interaction']:.4f}"
                )

    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_market_aux_no_time_fe_sensitivity(
    prepared: pd.DataFrame,
    output_dir: Path,
    *,
    market_panel_path: Path | None = None,
    cluster_col: str | None = DEFAULT_MARKET_INTERACTION_CLUSTER,
) -> None:
    required_treatments = [item.name for item in TREATMENTS]
    merged = _prepare_market_interactions(prepared, market_panel_path=market_panel_path)
    note_path = output_dir / "market_aux_no_time_fe.md"
    results_path = output_dir / "market_aux_no_time_fe.csv"
    if merged is None or prepared.empty or any(column not in prepared.columns for column in required_treatments):
        note_path.write_text(
            "# Auxiliary No-Time-FE Market Sensitivity\n\n- Auxiliary no-time-FE sensitivity was not run because usable market series or treatment assignments were not available.\n",
            encoding="utf-8",
        )
        write_frame(pd.DataFrame(), results_path)
        return

    did_results = []
    available_market_columns = [f"{column}_z" for column in MARKET_CONTROL_COLUMNS if f"{column}_z" in merged.columns]
    for treatment in [item.name for item in TREATMENTS]:
        for outcome in OUTCOMES:
            for market_column in available_market_columns:
                did = _run_market_aux_no_time_fe_did(
                    merged,
                    treatment,
                    outcome,
                    market_column,
                    cluster_col=cluster_col,
                )
                if not did.empty:
                    did_results.append(did)

    results = pd.concat(did_results, ignore_index=True) if did_results else pd.DataFrame()
    write_frame(results, results_path)

    lines = [
        "# Auxiliary No-Time-FE Market Sensitivity",
        "",
        "## Design",
        "- This auxiliary check drops quarter fixed effects and is weaker than the baseline DiD.",
        "- It keeps entity fixed effects and adds `post`, a linear `event_quarter` trend, the standardized market level, and `treated_post × standardized_market_level`.",
        "- Use this only as a directional robustness check, not as a replacement for the baseline fixed-effects design.",
        f"- Covariance: {'parent-clustered' if cluster_col else 'HC1'}",
        f"- Cluster column: {cluster_col if cluster_col else 'n/a'}",
        "",
        "## Key rows",
    ]

    if results.empty:
        lines.append("- No auxiliary no-time-FE results were estimable.")
    else:
        lookup = results.set_index(["treatment", "outcome", "market_column"])
        for treatment, outcome in KEY_COMPARISON_ROWS:
            for market_column in available_market_columns:
                key = (treatment, outcome, market_column)
                if key not in lookup.index:
                    continue
                row = lookup.loc[key]
                lines.append(
                    f"- `{treatment}` / `{outcome}` / `{market_column}`: "
                    f"treated_post coef={row['coef_treated_post']:.4f}, p={row['pvalue_treated_post']:.4f}; "
                    f"market coef={row['coef_market']:.4f}, p={row['pvalue_market']:.4f}; "
                    f"interaction coef={row['coef_interaction']:.4f}, p={row['pvalue_interaction']:.4f}"
                )

    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_event_2020(
    panel_path: Path,
    output_dir: Path | None = None,
    *,
    market_panel_path: Path | None = None,
) -> Path:
    panel = read_table(panel_path)
    prepared = _event_window(prepare_event_2020_panel(panel))
    slr_reporting_entities = set(
        prepared.loc[
            pd.to_numeric(prepared.get("tier1_capital"), errors="coerce").gt(0)
            & pd.to_numeric(prepared.get("total_leverage_exposure"), errors="coerce").gt(0),
            "entity_id",
        ].astype(str)
    )
    universe_b = _filter_entities(prepared, slr_reporting_entities)
    baseline_ready_entities = _baseline_ready_entities(universe_b)
    universe_c = _filter_entities(universe_b, baseline_ready_entities)

    coverage = _coverage_summary(universe_c)
    primary_entities = set(coverage.loc[coverage["has_full_event_window_coverage"], "entity_id"].astype(str))
    expanded_entities = set(coverage.loc[coverage["has_expanded_event_window_coverage"], "entity_id"].astype(str))

    universe_d = add_treatments(_filter_entities(universe_c, primary_entities))
    universe_e = add_treatments(_filter_entities(universe_c, expanded_entities))
    universe_f_primary = select_flagship_per_parent(universe_d)
    universe_f_expanded = select_flagship_per_parent(universe_e)
    historical_prepared = universe_c.copy()

    sample_manifest = _build_sample_manifest(
        panel,
        universe_b,
        universe_c,
        universe_d,
        universe_e,
        universe_f_primary,
        universe_f_expanded,
    )

    destination = output_dir or reports_path("event_2020")
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(sample_manifest, destination / "sample_manifest.csv")
    _write_sample_manifest_summary(sample_manifest, destination / "sample_manifest.md")
    primary_did, _ = _run_event_2020_sample(universe_d, destination, sample_label="Universe D primary causal core")
    expanded_dir = destination / "expanded_sensitivity"
    expanded_did, _ = _run_event_2020_sample(
        universe_e,
        expanded_dir,
        sample_label="Universe E expanded causal sensitivity",
    )
    historical_dir = destination / "historical_unbalanced"
    _run_event_2020_sample(
        historical_prepared,
        historical_dir,
        sample_label="Universe C historical treatment-definable sample",
    )
    flagship_dir = destination / "flagship_per_parent"
    flagship_did, _ = _run_event_2020_sample(
        universe_f_primary,
        flagship_dir,
        sample_label="Universe F flagship primary",
    )
    flagship_clustered_dir = destination / "flagship_per_parent_clustered"
    flagship_clustered_did, _ = _run_event_2020_sample(
        universe_f_primary,
        flagship_clustered_dir,
        cluster_col="top_parent_rssd",
        sample_label="Universe F flagship primary clustered",
    )
    primary_pretrend = _write_pretrend_checks(
        universe_d,
        destination,
        sample_label="Universe D primary causal core",
    )
    flagship_placebo = _write_fake_date_placebo(
        universe_f_primary,
        flagship_clustered_dir,
        sample_label="Universe F flagship primary clustered",
        cluster_col="top_parent_rssd",
    )
    flagship_expanded_dir = destination / "flagship_per_parent_expanded"
    _run_event_2020_sample(
        universe_f_expanded,
        flagship_expanded_dir,
        sample_label="Universe F flagship expanded sensitivity",
    )
    flagship_expanded_clustered_dir = destination / "flagship_per_parent_expanded_clustered"
    _run_event_2020_sample(
        universe_f_expanded,
        flagship_expanded_clustered_dir,
        cluster_col="top_parent_rssd",
        sample_label="Universe F flagship expanded clustered",
    )
    _write_sample_comparison(
        universe_d,
        primary_did,
        universe_e,
        expanded_did,
        universe_f_primary,
        flagship_did,
        flagship_clustered_did,
        historical_prepared,
        sample_manifest,
        destination / "sample_comparison.md",
    )
    sample_ladder = pd.DataFrame(
        [
            _sample_ladder_row(
                "universe_a_all_insured_banks",
                "Universe A all insured-bank descriptive universe",
                _event_window(panel),
                "All insured-bank Call Report filers observed in the staged FFIEC bulk data.",
                "Descriptive-only universe.",
            ),
            _sample_ladder_row(
                "universe_b_slr_reporting",
                "Universe B SLR-reporting insured banks",
                universe_b,
                "Positive tier1_capital and positive total_leverage_exposure in at least one event-window quarter.",
                _format_reason_counts(sample_manifest["universe_b_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_c_treatment_definable",
                "Universe C treatment-definable SLR sample",
                universe_c,
                "Universe B banks with a usable 2019Q4 treatment baseline.",
                _format_reason_counts(sample_manifest["universe_c_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_d_primary_core",
                "Universe D primary causal core",
                universe_d,
                "Universe C banks with full 2019Q1-2021Q4 coverage.",
                _format_reason_counts(sample_manifest["universe_d_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_e_expanded_sensitivity",
                "Universe E expanded causal sensitivity",
                universe_e,
                f"Universe C banks with at least {MIN_PRE_QUARTERS_EXPANDED} pre-treatment and {MIN_POST_QUARTERS_EXPANDED} post-treatment quarters.",
                _format_reason_counts(sample_manifest["universe_e_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_f_flagship_primary",
                "Universe F flagship primary",
                universe_f_primary,
                "One largest 2019Q4 insured subsidiary per parent family within Universe D.",
                _format_reason_counts(sample_manifest["universe_f_primary_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_f_flagship_expanded",
                "Universe F flagship expanded sensitivity",
                universe_f_expanded,
                "One largest 2019Q4 insured subsidiary per parent family within Universe E.",
                _format_reason_counts(sample_manifest["universe_f_expanded_exclusion_reason"]),
            ),
        ]
    )
    _write_sample_ladder(sample_ladder, destination / "sample_ladder.csv", destination / "sample_ladder.md")
    _write_methodology_memo(
        sample_ladder,
        sample_manifest,
        primary_did,
        expanded_did,
        flagship_clustered_did,
        primary_pretrend,
        flagship_placebo,
        destination / "methodology_memo.md",
    )
    _write_gpt_pro_prompt(
        sample_ladder,
        sample_manifest,
        primary_did,
        expanded_did,
        flagship_clustered_did,
        primary_pretrend,
        flagship_placebo,
        destination / "gpt_pro_next_steps_prompt.md",
    )
    _write_market_control_sensitivity_note(
        universe_d,
        destination / "market_control_sensitivity.md",
        market_panel_path=market_panel_path,
    )
    _write_market_interaction_sensitivity(
        universe_f_primary,
        destination,
        market_panel_path=market_panel_path,
        cluster_col=DEFAULT_MARKET_INTERACTION_CLUSTER,
    )
    _write_market_aux_no_time_fe_sensitivity(
        universe_f_primary,
        destination,
        market_panel_path=market_panel_path,
        cluster_col=DEFAULT_MARKET_INTERACTION_CLUSTER,
    )
    _write_market_context_note(destination)
    return destination
