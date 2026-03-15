from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from ..config import derived_data_path, reports_path
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(prepared, destination / "prepared_panel.csv")
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
    broad_prepared: pd.DataFrame,
    broad_did: pd.DataFrame,
    flagship_prepared: pd.DataFrame,
    flagship_did: pd.DataFrame,
    flagship_clustered_did: pd.DataFrame,
    output_path: Path,
) -> None:
    broad_lookup = broad_did.set_index(["treatment", "outcome"]) if not broad_did.empty else pd.DataFrame()
    flagship_lookup = flagship_did.set_index(["treatment", "outcome"]) if not flagship_did.empty else pd.DataFrame()
    clustered_lookup = (
        flagship_clustered_did.set_index(["treatment", "outcome"]) if not flagship_clustered_did.empty else pd.DataFrame()
    )

    lines = [
        "# Event Study Sample Comparison",
        "",
        "## Sample sizes",
        f"- Broad sample observations: {len(broad_prepared)}",
        f"- Broad sample entities: {broad_prepared['entity_id'].nunique()}",
        f"- Flagship sample observations: {len(flagship_prepared)}",
        f"- Flagship sample entities: {flagship_prepared['entity_id'].nunique()}",
        "",
        "## Key DID rows",
    ]
    for treatment, outcome in KEY_COMPARISON_ROWS:
        if (
            (treatment, outcome) not in broad_lookup.index
            or (treatment, outcome) not in flagship_lookup.index
            or (treatment, outcome) not in clustered_lookup.index
        ):
            continue
        broad_row = broad_lookup.loc[(treatment, outcome)]
        flagship_row = flagship_lookup.loc[(treatment, outcome)]
        clustered_row = clustered_lookup.loc[(treatment, outcome)]
        lines.append(
            f"- `{treatment}` / `{outcome}`: "
            f"broad coef={broad_row['coef']:.4f}, p={broad_row['pvalue']:.4f}; "
            f"flagship coef={flagship_row['coef']:.4f}, p={flagship_row['pvalue']:.4f}; "
            f"flagship clustered coef={clustered_row['coef']:.4f}, p={clustered_row['pvalue']:.4f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_market_interaction_sensitivity(
    prepared: pd.DataFrame,
    output_dir: Path,
    *,
    market_panel_path: Path | None = None,
    cluster_col: str | None = DEFAULT_MARKET_INTERACTION_CLUSTER,
) -> None:
    merged = _prepare_market_interactions(prepared, market_panel_path=market_panel_path)
    note_path = output_dir / "market_interaction_sensitivity.md"
    results_path = output_dir / "market_interaction_sensitivity.csv"
    if merged is None:
        note_path.write_text(
            "# Market Interaction Sensitivity\n\n- Market interaction sensitivity was not run because usable market series were not available.\n",
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
    merged = _prepare_market_interactions(prepared, market_panel_path=market_panel_path)
    note_path = output_dir / "market_aux_no_time_fe.md"
    results_path = output_dir / "market_aux_no_time_fe.csv"
    if merged is None:
        note_path.write_text(
            "# Auxiliary No-Time-FE Market Sensitivity\n\n- Auxiliary no-time-FE sensitivity was not run because usable market series were not available.\n",
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
    prepared = prepare_event_2020_panel(panel)
    prepared = prepared[
        (prepared["quarter_end"] >= pd.Timestamp("2019-03-31"))
        & (prepared["quarter_end"] <= pd.Timestamp("2021-12-31"))
    ].copy()

    broad_prepared = add_treatments(prepared)
    flagship_prepared = add_treatments(select_flagship_per_parent(prepared))

    destination = output_dir or reports_path("event_2020")
    destination.mkdir(parents=True, exist_ok=True)
    broad_did, _ = _run_event_2020_sample(broad_prepared, destination)
    flagship_dir = destination / "flagship_per_parent"
    flagship_did, _ = _run_event_2020_sample(flagship_prepared, flagship_dir)
    flagship_clustered_dir = destination / "flagship_per_parent_clustered"
    flagship_clustered_did, _ = _run_event_2020_sample(
        flagship_prepared,
        flagship_clustered_dir,
        cluster_col="top_parent_rssd",
    )
    _write_sample_comparison(
        broad_prepared,
        broad_did,
        flagship_prepared,
        flagship_did,
        flagship_clustered_did,
        destination / "sample_comparison.md",
    )
    _write_market_control_sensitivity_note(
        broad_prepared,
        destination / "market_control_sensitivity.md",
        market_panel_path=market_panel_path,
    )
    _write_market_interaction_sensitivity(
        flagship_prepared,
        destination,
        market_panel_path=market_panel_path,
        cluster_col=DEFAULT_MARKET_INTERACTION_CLUSTER,
    )
    _write_market_aux_no_time_fe_sensitivity(
        flagship_prepared,
        destination,
        market_panel_path=market_panel_path,
        cluster_col=DEFAULT_MARKET_INTERACTION_CLUSTER,
    )
    _write_market_context_note(destination)
    return destination
