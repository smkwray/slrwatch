from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from ..config import derived_data_path, reference_data_path, reports_path
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
from ..panels import _to_bool_series
from .event_study import EventStudySpec, add_event_dummies, event_study_terms


@dataclass(frozen=True)
class TreatmentDefinition:
    name: str
    source_column: str
    mode: str
    baseline_columns: tuple[str, ...] = ()


TREATMENT_MAP_COLUMNS = (
    "rssd_id",
    "entity_id",
    "entity_name",
    "fdic_cert",
    "top_parent_rssd_2019q4",
    "top_parent_name_2019q4",
    "slr_reporting_2019q4",
    "eslr_covered_6pct",
    "di_relief_eligible_2020",
    "di_relief_elected_2020",
    "parent_hc_relief_scope_2020",
    "treatment_scope_2020",
    "classification_source",
    "provenance_notes",
)
DEFAULT_TREATMENT_MAP_PATH = reference_data_path("insured_bank_treatment_map_2020.csv")


TREATMENTS = [
    TreatmentDefinition(
        name="low_headroom_treated",
        source_column="headroom_pp",
        mode="bottom_tercile",
        baseline_columns=("headroom_pp",),
    ),
    TreatmentDefinition(
        name="high_ust_share_treated",
        source_column="ust_share_assets",
        mode="top_tercile",
        baseline_columns=("ust_share_assets",),
    ),
    TreatmentDefinition(
        name="covered_bank_treated",
        source_column="di_relief_eligible_2020",
        mode="boolean",
        baseline_columns=("di_relief_eligible_2020",),
    ),
]
TREATMENT_LOOKUP = {item.name: item for item in TREATMENTS}

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
PLACEBO_GRID = (
    ("2019-07-01", pd.Timestamp("2019-12-31")),
    ("2019-10-01", pd.Timestamp("2019-12-31")),
    ("2019-10-01", pd.Timestamp("2020-03-31")),
)


def _to_nullable_bool_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "n": False,
        "": pd.NA,
    }
    return text.map(mapping).astype("boolean")


def _load_treatment_map(treatment_map_path: Path | None = None) -> pd.DataFrame:
    source = treatment_map_path or DEFAULT_TREATMENT_MAP_PATH
    if not source.exists():
        return pd.DataFrame(columns=TREATMENT_MAP_COLUMNS)
    frame = read_table(source).copy()
    for column in TREATMENT_MAP_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["rssd_id"] = frame["rssd_id"].astype("string").str.strip()
    for column in ["entity_id", "entity_name", "fdic_cert", "top_parent_rssd_2019q4", "top_parent_name_2019q4", "treatment_scope_2020", "classification_source", "provenance_notes"]:
        frame[column] = frame[column].astype("string").str.strip()
    for column in ["slr_reporting_2019q4", "eslr_covered_6pct", "di_relief_eligible_2020", "parent_hc_relief_scope_2020"]:
        frame[column] = _to_bool_series(frame[column])
    if "di_relief_elected_2020" in frame.columns:
        frame["di_relief_elected_2020"] = _to_nullable_bool_series(frame["di_relief_elected_2020"])
    return frame[list(TREATMENT_MAP_COLUMNS)].drop_duplicates("rssd_id", keep="first")


def _enrich_with_treatment_map(frame: pd.DataFrame, *, treatment_map_path: Path | None = None) -> pd.DataFrame:
    out = frame.copy()
    treatment_map = _load_treatment_map(treatment_map_path)
    if not treatment_map.empty:
        if "rssd_id" in out.columns and "rssd_id" in treatment_map.columns:
            join_key = "rssd_id"
        elif "entity_id" in out.columns and "entity_id" in treatment_map.columns:
            join_key = "entity_id"
        else:
            join_key = None
        if join_key is not None:
            out = out.merge(treatment_map, how="left", on=join_key, suffixes=("", "__map"))
            for column in TREATMENT_MAP_COLUMNS:
                if column == join_key:
                    continue
                map_column = f"{column}__map"
                if map_column not in out.columns:
                    if column not in out.columns:
                        out[column] = pd.NA
                    continue
                if column in out.columns:
                    out[column] = out[map_column].combine_first(out[column])
                else:
                    out[column] = out[map_column]
                out = out.drop(columns=[map_column])
    for column in TREATMENT_MAP_COLUMNS:
        if column == "rssd_id":
            continue
        if column not in out.columns:
            out[column] = pd.NA
    if "di_relief_eligible_2020" in out.columns:
        eligible = out["di_relief_eligible_2020"]
        eligible = eligible.where(eligible.notna(), _to_bool_series(out.get("is_covered_bank_subsidiary", pd.Series(False, index=out.index))))
        out["di_relief_eligible_2020"] = eligible.astype(bool)
    if "classification_source" in out.columns:
        out["classification_source"] = out["classification_source"].where(
            out["classification_source"].notna() & (out["classification_source"].astype("string").str.strip() != ""),
            "legacy_is_covered_bank_subsidiary",
        )
    if "provenance_notes" in out.columns:
        out["provenance_notes"] = out["provenance_notes"].where(
            out["provenance_notes"].notna() & (out["provenance_notes"].astype("string").str.strip() != ""),
            "Fallback derived from the legacy covered-bank flag.",
        )
    if "treatment_scope_2020" in out.columns:
        out["treatment_scope_2020"] = out["treatment_scope_2020"].where(
            out["treatment_scope_2020"].notna() & (out["treatment_scope_2020"].astype("string").str.strip() != ""),
            "legacy_fallback",
        )
    if "top_parent_rssd_2019q4" in out.columns:
        out["top_parent_rssd_2019q4"] = out["top_parent_rssd_2019q4"].where(
            out["top_parent_rssd_2019q4"].notna() & (out["top_parent_rssd_2019q4"].astype("string").str.strip() != ""),
            out.get("top_parent_rssd", pd.Series(pd.NA, index=out.index, dtype="string")),
        )
    if "top_parent_name_2019q4" in out.columns:
        out["top_parent_name_2019q4"] = out["top_parent_name_2019q4"].where(
            out["top_parent_name_2019q4"].notna() & (out["top_parent_name_2019q4"].astype("string").str.strip() != ""),
            out.get("top_parent_name", pd.Series(pd.NA, index=out.index, dtype="string")),
        )
    return out


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
    baseline = _enrich_with_treatment_map(frame)
    baseline = baseline[baseline["quarter_end"] == BASELINE_QUARTER].copy()
    if baseline.empty:
        return set()
    mask = (
        pd.to_numeric(baseline.get("tier1_capital"), errors="coerce").gt(0)
        & pd.to_numeric(baseline.get("total_leverage_exposure"), errors="coerce").gt(0)
    )
    for treatment in TREATMENTS:
        for column in treatment.baseline_columns:
            series = baseline.get(column)
            if series is None:
                mask &= False
                continue
            if column in {"headroom_pp", "ust_share_assets"}:
                mask &= pd.to_numeric(series, errors="coerce").notna()
            else:
                mask &= series.notna()
    return set(baseline.loc[mask, "entity_id"].astype(str))


def _baseline_ready_entities_for_treatment(frame: pd.DataFrame, treatment_name: str) -> set[str]:
    baseline = _enrich_with_treatment_map(frame)
    baseline = baseline[baseline["quarter_end"] == BASELINE_QUARTER].copy()
    if baseline.empty:
        return set()
    treatment = TREATMENT_LOOKUP[treatment_name]
    mask = (
        pd.to_numeric(baseline.get("tier1_capital"), errors="coerce").gt(0)
        & pd.to_numeric(baseline.get("total_leverage_exposure"), errors="coerce").gt(0)
    )
    for column in treatment.baseline_columns:
        series = baseline.get(column)
        if series is None:
            mask &= False
            continue
        if column in {"headroom_pp", "ust_share_assets"}:
            mask &= pd.to_numeric(series, errors="coerce").notna()
        else:
            mask &= series.notna()
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
    treatment_columns = [item.name for item in TREATMENTS if item.name in baseline.columns]
    if treatment_columns:
        baseline = baseline[baseline[treatment_columns].notna().any(axis=1)].copy()
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
    *,
    treatment_map_path: Path | None = None,
) -> pd.DataFrame:
    manifest = _entity_universe_for_manifest(full_panel)
    if manifest.empty:
        return manifest
    baseline = full_panel.copy()
    baseline["quarter_end"] = pd.to_datetime(baseline["quarter_end"], errors="coerce")
    baseline = baseline[baseline["quarter_end"] == BASELINE_QUARTER].copy()
    if not baseline.empty:
        baseline["top_parent_rssd"] = baseline.get("top_parent_rssd", pd.Series(pd.NA, index=baseline.index, dtype="string")).astype("string").str.strip()
        if "top_parent_name" in baseline.columns:
            baseline["top_parent_name"] = baseline["top_parent_name"].astype("string").str.strip()
        elif "fdic_top_parent_name" in baseline.columns:
            baseline["top_parent_name"] = baseline["fdic_top_parent_name"].astype("string").str.strip()
        else:
            baseline["top_parent_name"] = pd.NA
        baseline["tier1_capital"] = pd.to_numeric(baseline.get("tier1_capital"), errors="coerce")
        baseline["total_leverage_exposure"] = pd.to_numeric(baseline.get("total_leverage_exposure"), errors="coerce")
        if "is_covered_bank_subsidiary" not in baseline.columns:
            baseline["is_covered_bank_subsidiary"] = False
        baseline_summary = (
            baseline.groupby("entity_id", dropna=False)
            .agg(
                top_parent_rssd_2019q4=("top_parent_rssd", lambda s: s.replace({"": pd.NA}).dropna().iloc[0] if s.replace({"": pd.NA}).dropna().size else pd.NA),
                top_parent_name_2019q4=("top_parent_name", lambda s: s.replace({"": pd.NA}).dropna().iloc[0] if s.replace({"": pd.NA}).dropna().size else pd.NA),
                slr_reporting_2019q4=("tier1_capital", lambda s: pd.to_numeric(s, errors="coerce").gt(0).any()),
                eslr_covered_6pct=("is_covered_bank_subsidiary", lambda s: _to_bool_series(s).any()),
                parent_hc_relief_scope_2020=("is_covered_bank_subsidiary", lambda s: _to_bool_series(s).any()),
            )
            .reset_index()
        )
        manifest = manifest.merge(baseline_summary, how="left", on="entity_id")
    manifest = _enrich_with_treatment_map(manifest, treatment_map_path=treatment_map_path)

    full_event = _event_window(full_panel)
    full_summary = _coverage_summary(full_event)
    baseline_ready = pd.DataFrame({"entity_id": sorted(_baseline_ready_entities(full_event))})
    if not baseline_ready.empty:
        baseline_ready["has_usable_2019q4_baseline"] = True
    baseline_ready_low_headroom = pd.DataFrame(
        {"entity_id": sorted(_baseline_ready_entities_for_treatment(full_event, "low_headroom_treated"))}
    )
    if not baseline_ready_low_headroom.empty:
        baseline_ready_low_headroom["has_usable_2019q4_low_headroom_baseline"] = True
    baseline_ready_high_ust = pd.DataFrame(
        {"entity_id": sorted(_baseline_ready_entities_for_treatment(full_event, "high_ust_share_treated"))}
    )
    if not baseline_ready_high_ust.empty:
        baseline_ready_high_ust["has_usable_2019q4_high_ust_share_baseline"] = True
    baseline_ready_covered = pd.DataFrame(
        {"entity_id": sorted(_baseline_ready_entities_for_treatment(full_event, "covered_bank_treated"))}
    )
    if not baseline_ready_covered.empty:
        baseline_ready_covered["has_usable_2019q4_covered_bank_baseline"] = True

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
    manifest = manifest.merge(baseline_ready_low_headroom, how="left", on="entity_id")
    manifest = manifest.merge(baseline_ready_high_ust, how="left", on="entity_id")
    manifest = manifest.merge(baseline_ready_covered, how="left", on="entity_id")
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
    manifest["has_usable_2019q4_low_headroom_baseline"] = (
        manifest["has_usable_2019q4_low_headroom_baseline"].fillna(False).astype(bool)
    )
    manifest["has_usable_2019q4_high_ust_share_baseline"] = (
        manifest["has_usable_2019q4_high_ust_share_baseline"].fillna(False).astype(bool)
    )
    manifest["has_usable_2019q4_covered_bank_baseline"] = (
        manifest["has_usable_2019q4_covered_bank_baseline"].fillna(False).astype(bool)
    )
    manifest["included_universe_c"] = manifest["universe_c_observations"].fillna(0).astype(int) > 0
    manifest["included_universe_c_low_headroom"] = (
        manifest["included_universe_b"] & manifest["has_usable_2019q4_low_headroom_baseline"]
    )
    manifest["included_universe_c_high_ust_share"] = (
        manifest["included_universe_b"] & manifest["has_usable_2019q4_high_ust_share_baseline"]
    )
    manifest["included_universe_c_covered_bank"] = (
        manifest["included_universe_b"] & manifest["has_usable_2019q4_covered_bank_baseline"]
    )
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
        manifest["included_universe_b"] & ~manifest["included_universe_d"],
        "universe_d_exclusion_reason",
    ] = "incomplete_2019q1_2021q4_coverage"

    manifest["universe_e_exclusion_reason"] = pd.NA
    manifest.loc[
        manifest["included_universe_b"] & ~manifest["included_universe_e"],
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
        if column.endswith("_observations") or column.endswith("_quarters"):
            try:
                manifest[column] = manifest[column].astype("Int64")
            except Exception:
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
        f"- Universe C joint baseline (banks that can support all current treatment splits): {int(manifest['included_universe_c'].sum())}",
        f"- Universe C low-headroom baseline: {int(manifest['included_universe_c_low_headroom'].sum())}",
        f"- Universe C high-UST-share baseline: {int(manifest['included_universe_c_high_ust_share'].sum())}",
        f"- Universe C covered-bank baseline: {int(manifest['included_universe_c_covered_bank'].sum())}",
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
    treatment_columns = [item.name for item in TREATMENTS if item.name in baseline.columns]
    if treatment_columns:
        baseline = baseline[baseline[treatment_columns].notna().any(axis=1)].copy()
    if baseline.empty:
        return out.iloc[0:0].drop(columns=["sample_parent_key"]).copy()

    selected = (
        baseline.sort_values(
            ["sample_parent_key", "total_assets", "entity_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("sample_parent_key", keep="first")
    )
    selected_ids = set(selected["entity_id"])
    return out[out["entity_id"].isin(selected_ids)].drop(columns=["sample_parent_key"])


def add_treatments(
    frame: pd.DataFrame,
    *,
    assignment_frame: pd.DataFrame | None = None,
    treatment_map_path: Path | None = None,
    baseline_quarter: str = "2019-12-31",
) -> pd.DataFrame:
    out = _enrich_with_treatment_map(frame, treatment_map_path=treatment_map_path)
    source = _enrich_with_treatment_map(assignment_frame.copy() if assignment_frame is not None else out.copy(), treatment_map_path=treatment_map_path)
    source["quarter_end"] = pd.to_datetime(source["quarter_end"], errors="coerce")
    baseline_mask = source["quarter_end"] == pd.Timestamp(baseline_quarter)
    baseline = source.loc[
        baseline_mask,
        ["entity_id", "headroom_pp", "ust_share_assets", "di_relief_eligible_2020"],
    ].copy()

    existing_columns = [item.name for item in TREATMENTS if item.name in out.columns]
    if existing_columns:
        out = out.drop(columns=existing_columns)

    for treatment in TREATMENTS:
        eligible_ids = _baseline_ready_entities_for_treatment(source, treatment.name)
        eligible = baseline[baseline["entity_id"].astype(str).isin(eligible_ids)].copy()
        if treatment.mode == "boolean":
            values = eligible[["entity_id", treatment.source_column]].copy()
            values[treatment.name] = values[treatment.source_column].astype("boolean").astype("Int64")
            out = out.merge(values[["entity_id", treatment.name]], how="left", on="entity_id")
            continue

        terciles = _tercile_labels(eligible[treatment.source_column])
        values = eligible[["entity_id"]].copy()
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
    shock_date: date,
    sample_end: pd.Timestamp,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    placebo_frame = frame.copy()
    placebo_frame["quarter_end"] = pd.to_datetime(placebo_frame["quarter_end"], errors="coerce")
    placebo_frame = placebo_frame[placebo_frame["quarter_end"] <= sample_end].copy()
    if placebo_frame.empty:
        return pd.DataFrame()
    spec = EventStudySpec(outcome=outcome, treatment=treatment, shock_date=shock_date)
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
                "placebo_shock_date": shock_date.isoformat(),
                "placebo_sample_end": sample_end.strftime("%Y-%m-%d"),
            }
        ]
    )

def _plot_event_time(frame: pd.DataFrame, output_path: Path, *, reference_period: int = -1) -> None:
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
    ax.axvline(reference_period, color="gray", linestyle="--", linewidth=1)
    ax.set_title("2020 temporary exclusion event-study coefficients")
    ax.set_xlabel("Event quarter")
    ax.set_ylabel("Coefficient")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _run_leave_one_parent_out_did(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    *,
    cluster_col: str = "top_parent_rssd",
) -> pd.DataFrame:
    if cluster_col not in frame.columns:
        return pd.DataFrame()
    clusters = (
        frame[cluster_col]
        .astype("string")
        .replace({"": pd.NA})
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if len(clusters) < 2:
        return pd.DataFrame()
    baseline = _run_did(frame, treatment, outcome, cluster_col=cluster_col)
    baseline_coef = baseline.iloc[0]["coef"] if not baseline.empty else pd.NA
    rows: list[dict[str, object]] = []
    for omitted in clusters:
        sample = frame[frame[cluster_col].astype("string") != omitted].copy()
        did = _run_did(sample, treatment, outcome, cluster_col=cluster_col)
        if did.empty:
            continue
        row = did.iloc[0].to_dict()
        row["omitted_cluster"] = omitted
        row["baseline_coef"] = baseline_coef
        row["coef_delta_vs_full"] = row["coef"] - baseline_coef if pd.notna(baseline_coef) else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def _run_event_2020_sample(
    prepared: pd.DataFrame,
    destination: Path,
    *,
    cluster_col: str | None = None,
    sample_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(prepared, destination / "prepared_panel.csv")
    available_treatments = [treatment.name for treatment in TREATMENTS if treatment.name in prepared.columns]
    if prepared.empty or not available_treatments:
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
    for treatment in available_treatments:
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
        _plot_event_time(baseline_plot, destination / "low_headroom_ust_inventory_event_time.png", reference_period=-2)

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
        "- Treatment assignment coverage:",
    ]
    for treatment in available_treatments:
        eligible = _treatment_defined_frame(prepared, treatment)
        summary_lines.append(
            f"  - `{treatment}`: {eligible['entity_id'].nunique()} entities / {len(eligible)} observations with baseline treatment assignments"
        )
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
        f"- Historical unbalanced SLR-reporting sample observations: {len(historical_prepared)}",
        f"- Historical unbalanced SLR-reporting sample entities: {historical_prepared['entity_id'].nunique()}",
        f"- Primary causal core (Universe D) observations: {len(primary_prepared)}",
        f"- Primary causal core (Universe D) entities: {primary_prepared['entity_id'].nunique()}",
        f"- Expanded causal sensitivity (Universe E) observations: {len(expanded_prepared)}",
        f"- Expanded causal sensitivity (Universe E) entities: {expanded_prepared['entity_id'].nunique()}",
        f"- Flagship primary (Universe F on D) observations: {len(flagship_primary_prepared)}",
        f"- Flagship primary (Universe F on D) entities: {flagship_primary_prepared['entity_id'].nunique()}",
        f"- Manifest rows: {len(sample_manifest)}",
        f"- Universe B entities with a jointly usable 2019Q4 treatment baseline: {int(sample_manifest['included_universe_c'].sum())}",
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


def _treatment_defined_frame(frame: pd.DataFrame, treatment_name: str) -> pd.DataFrame:
    if treatment_name not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame[treatment_name].notna()].copy()


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
    placebo_grid: tuple[tuple[str, pd.Timestamp], ...] = PLACEBO_GRID,
) -> pd.DataFrame:
    checks: list[pd.DataFrame] = []
    for shock_date_str, sample_end in placebo_grid:
        shock_date = date.fromisoformat(shock_date_str)
        for treatment in [item.name for item in TREATMENTS]:
            for outcome in OUTCOMES:
                check = _run_placebo_fake_date_did(
                    prepared,
                    treatment,
                    outcome,
                    shock_date=shock_date,
                    sample_end=sample_end,
                    cluster_col=cluster_col,
                )
                if not check.empty:
                    check["sample_label"] = sample_label
                    check["placebo_grid_label"] = f"{shock_date.isoformat()} to {sample_end.strftime('%Y-%m-%d')}"
                    checks.append(check)
    frame = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    write_frame(frame, output_dir / "placebo_fake_date.csv")

    lines = [
        "# Fake-Date Placebo",
        "",
        f"- Sample: {sample_label}",
        "- Placebo grid:",
    ]
    for shock_date_str, sample_end in placebo_grid:
        lines.append(f"- `{shock_date_str}` through `{sample_end.strftime('%Y-%m-%d')}`")
    lines.extend([
        "",
        "## Key rows",
    ])
    if frame.empty:
        lines.append("- No fake-date placebo checks were estimable.")
    else:
        focus = frame.set_index(["placebo_grid_label", "treatment", "outcome"])
        for treatment, outcome in KEY_COMPARISON_ROWS:
            for placebo_grid_label in frame["placebo_grid_label"].dropna().astype(str).drop_duplicates():
                key = (placebo_grid_label, treatment, outcome)
                if key not in focus.index:
                    continue
                row = focus.loc[key]
                lines.append(
                    f"- `{placebo_grid_label}` / `{treatment}` / `{outcome}`: placebo coef={row['coef']:.4f}, p={row['pvalue']:.4f}"
                )
    (output_dir / "placebo_fake_date.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def _write_leave_one_parent_out(
    prepared: pd.DataFrame,
    output_dir: Path,
    *,
    sample_label: str,
    cluster_col: str = "top_parent_rssd",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for treatment, outcome in [
        ("covered_bank_treated", "ust_inventory_fv_scaled"),
        ("low_headroom_treated", "ust_inventory_fv_scaled"),
    ]:
        check = _run_leave_one_parent_out_did(prepared, treatment, outcome, cluster_col=cluster_col)
        if not check.empty:
            check["sample_label"] = sample_label
            rows.append(check)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_frame(frame, output_dir / "leave_one_parent_out.csv")

    lines = [
        "# Leave-One-Parent-Out",
        "",
        f"- Sample: {sample_label}",
        f"- Cluster column: {cluster_col}",
        "",
        "## Key rows",
    ]
    if frame.empty:
        lines.append("- No leave-one-parent-out rows were estimable.")
    else:
        focus = frame.groupby(["treatment", "outcome"], dropna=False)
        for (treatment, outcome), subset in focus:
            lines.append(
                f"- `{treatment}` / `{outcome}`: omitted parents={subset['omitted_cluster'].nunique()}, "
                f"coef range={subset['coef'].min():.4f} to {subset['coef'].max():.4f}"
            )
    (output_dir / "leave_one_parent_out.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def _write_treatment_roster(sample_manifest: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    roster_columns = [
        "entity_id",
        "entity_name",
        "rssd_id",
        "fdic_cert",
        "top_parent_rssd",
        "top_parent_rssd_2019q4",
        "top_parent_name_2019q4",
        "slr_reporting_2019q4",
        "eslr_covered_6pct",
        "di_relief_eligible_2020",
        "di_relief_elected_2020",
        "parent_hc_relief_scope_2020",
        "treatment_scope_2020",
        "classification_source",
        "provenance_notes",
        "slr_scope_class",
        "is_covered_bank_subsidiary",
        "included_universe_b",
        "included_universe_c",
        "included_universe_d",
        "included_universe_f_primary",
        "included_universe_f_expanded",
    ]
    roster = sample_manifest.loc[sample_manifest["included_universe_b"].fillna(False).astype(bool)].copy()
    for column in roster_columns:
        if column not in roster.columns:
            roster[column] = pd.NA
    roster = roster[roster_columns].sort_values(["included_universe_d", "entity_name", "entity_id"], ascending=[False, True, True])
    write_frame(roster, output_path)
    return roster


def _write_methodology_memo(
    sample_ladder: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    primary_did: pd.DataFrame,
    expanded_did: pd.DataFrame,
    flagship_clustered_did: pd.DataFrame,
    primary_pretrend: pd.DataFrame,
    flagship_clustered_pretrend: pd.DataFrame,
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
        "- Treatment assignment is now frozen on the relevant 2019Q4 baseline subset for each treatment instead of being recomputed inside narrower D/E samples.",
        "- The joint Universe C intersection is still reported for transparency, but the covered-bank treatment no longer requires headroom and Treasury-share baselines that belong to other treatment families.",
        "- `covered_bank_treated` is sourced from the direct-eligibility field when available, with the legacy covered-bank flag preserved as compatibility fallback.",
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
    if not flagship_clustered_pretrend.empty:
        lookup = flagship_clustered_pretrend.set_index(["treatment", "outcome"])
        lines.extend(
            [
                f"- Universe F clustered low-headroom Treasury pretrend joint p={lookup.loc[('low_headroom_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
                f"- Universe F clustered covered-bank Treasury pretrend joint p={lookup.loc[('covered_bank_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
            ]
        )
    clustered_dir = output_path.parent / "flagship_per_parent_clustered"
    clustered_pretrend = clustered_dir / "pretrend_checks.csv"
    if clustered_pretrend.exists():
        lines.append(f"- Flagship clustered pretrend checks: `{clustered_pretrend}`")
    clustered_leave_one_out = clustered_dir / "leave_one_parent_out.csv"
    if clustered_leave_one_out.exists():
        lines.append(f"- Flagship leave-one-parent-out summary: `{clustered_leave_one_out}`")
    if not flagship_placebo.empty:
        lookup = flagship_placebo.set_index(["placebo_grid_label", "treatment", "outcome"])
        lines.append("## Placebo grid")
        for placebo_grid_label in flagship_placebo["placebo_grid_label"].dropna().astype(str).drop_duplicates():
            low_key = (placebo_grid_label, "low_headroom_treated", "ust_inventory_fv_scaled")
            covered_key = (placebo_grid_label, "covered_bank_treated", "ust_inventory_fv_scaled")
            if low_key in lookup.index:
                low_row = lookup.loc[low_key]
                lines.append(
                    f"- `{placebo_grid_label}` low-headroom Treasury placebo coef={low_row['coef']:.4f}, p={low_row['pvalue']:.4f}"
                )
            if covered_key in lookup.index:
                covered_row = lookup.loc[covered_key]
                lines.append(
                    f"- `{placebo_grid_label}` covered-bank Treasury placebo coef={covered_row['coef']:.4f}, p={covered_row['pvalue']:.4f}"
                )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gpt_pro_prompt(
    sample_ladder: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    primary_did: pd.DataFrame,
    expanded_did: pd.DataFrame,
    flagship_clustered_did: pd.DataFrame,
    primary_pretrend: pd.DataFrame,
    flagship_clustered_pretrend: pd.DataFrame,
    flagship_placebo: pd.DataFrame,
    flagship_leave_one_out: pd.DataFrame,
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
        "Use the repo state below to perform a final external audit before commit/push.",
        "",
        "This is a stop/go review, not a brainstorming prompt.",
        "Assume the team wants to stop iterating unless you identify a concrete blocker, overclaim, missing validation step, or a small high-leverage fix that should happen before commit.",
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
    if not flagship_clustered_pretrend.empty:
        lookup = flagship_clustered_pretrend.set_index(["treatment", "outcome"])
        prompt.extend(
            [
                f"- Universe F clustered low_headroom_treated Treasury pretrend joint p={lookup.loc[('low_headroom_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
                f"- Universe F clustered covered_bank_treated Treasury pretrend joint p={lookup.loc[('covered_bank_treated', 'ust_inventory_fv_scaled'), 'pretrend_joint_pvalue']:.4f}",
            ]
        )
    if not flagship_placebo.empty:
        lookup = flagship_placebo.set_index(["placebo_grid_label", "treatment", "outcome"])
        prompt.append("- Universe F clustered fake-date placebo grid:")
        for placebo_grid_label in flagship_placebo["placebo_grid_label"].dropna().astype(str).drop_duplicates():
            low_key = (placebo_grid_label, "low_headroom_treated", "ust_inventory_fv_scaled")
            covered_key = (placebo_grid_label, "covered_bank_treated", "ust_inventory_fv_scaled")
            if low_key in lookup.index:
                low_row = lookup.loc[low_key]
                prompt.append(
                    f"- `{placebo_grid_label}` low-headroom Treasury placebo coef={low_row['coef']:.4f}, p={low_row['pvalue']:.4f}"
                )
            if covered_key in lookup.index:
                covered_row = lookup.loc[covered_key]
                prompt.append(
                    f"- `{placebo_grid_label}` covered-bank Treasury placebo coef={covered_row['coef']:.4f}, p={covered_row['pvalue']:.4f}"
                )
    if not flagship_leave_one_out.empty:
        prompt.append("- Universe F clustered leave-one-parent-out Treasury ranges:")
        for treatment in ["covered_bank_treated", "low_headroom_treated"]:
            subset = flagship_leave_one_out[
                (flagship_leave_one_out["treatment"] == treatment)
                & (flagship_leave_one_out["outcome"] == "ust_inventory_fv_scaled")
            ]
            if subset.empty:
                continue
            prompt.append(
                f"- `{treatment}`: omitted parents={subset['omitted_cluster'].nunique()}, "
                f"coef range={subset['coef'].min():.4f} to {subset['coef'].max():.4f}, "
                f"p range={subset['pvalue'].min():.4f} to {subset['pvalue'].max():.4f}"
            )
    prompt.extend(
        [
            "",
            "## Audit artifacts",
            "- Treatment roster with B/C/D/F membership and classification fields: `output/reports/event_2020/treatment_roster.csv`",
            "- Versioned 2020 treatment map: `data/reference/insured_bank_treatment_map_2020.csv`",
            "- Methodology memo: `output/reports/event_2020/methodology_memo.md`",
            "",
            "## Exact questions",
            "1. Is the repo now at a reasonable stopping point for commit/push, given the weaker map-backed Treasury results and the current public framing?",
            "2. Are there any remaining methodological or public-copy issues that are severe enough to block commit/push?",
            "3. Does the current discreet roster disclosure and committed `treatment_roster.csv` solve the auditability problem of which banks are in the headline samples?",
            "4. Which remaining ideas should be treated as explicitly post-stop backlog rather than pre-push work?",
            "",
            "## Required output",
            "1. Verdict: `stop`, `stop with 1-2 fixes`, or `do not stop`",
            "2. Findings ordered by severity, with concrete reasons",
            "3. Any blocking or near-blocking fixes, limited to the minimum needed before commit/push",
            "4. A short list of items that should stay in the backlog and not delay this push",
            "5. Explicit note if the current public wording is appropriately cautious",
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
    treatment_map_path: Path | None = None,
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

    coverage = _coverage_summary(universe_b)
    primary_entities = set(coverage.loc[coverage["has_full_event_window_coverage"], "entity_id"].astype(str))
    expanded_entities = set(coverage.loc[coverage["has_expanded_event_window_coverage"], "entity_id"].astype(str))

    universe_d = add_treatments(
        _filter_entities(universe_b, primary_entities),
        assignment_frame=universe_b,
        treatment_map_path=treatment_map_path,
    )
    universe_e = add_treatments(
        _filter_entities(universe_b, expanded_entities),
        assignment_frame=universe_b,
        treatment_map_path=treatment_map_path,
    )
    universe_f_primary = select_flagship_per_parent(universe_d)
    universe_f_expanded = select_flagship_per_parent(universe_e)
    historical_prepared = add_treatments(universe_b, assignment_frame=universe_b, treatment_map_path=treatment_map_path)

    sample_manifest = _build_sample_manifest(
        panel,
        universe_b,
        universe_c,
        universe_d,
        universe_e,
        universe_f_primary,
        universe_f_expanded,
        treatment_map_path=treatment_map_path,
    )

    destination = output_dir or reports_path("event_2020")
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(sample_manifest, destination / "sample_manifest.csv")
    _write_sample_manifest_summary(sample_manifest, destination / "sample_manifest.md")
    _write_treatment_roster(sample_manifest, destination / "treatment_roster.csv")
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
        sample_label="Historical unbalanced SLR-reporting sample",
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
    flagship_clustered_pretrend = _write_pretrend_checks(
        universe_f_primary,
        flagship_clustered_dir,
        sample_label="Universe F flagship primary clustered",
        cluster_col="top_parent_rssd",
    )
    flagship_placebo = _write_fake_date_placebo(
        universe_f_primary,
        flagship_clustered_dir,
        sample_label="Universe F flagship primary clustered",
        cluster_col="top_parent_rssd",
    )
    flagship_leave_one_out = _write_leave_one_parent_out(
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
                "Universe C joint treatment-definable SLR sample",
                universe_c,
                "Universe B banks with a usable 2019Q4 treatment baseline.",
                _format_reason_counts(sample_manifest["universe_c_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_c_low_headroom_baseline",
                "Universe C low-headroom-definable sample",
                _filter_entities(universe_b, _baseline_ready_entities_for_treatment(universe_b, "low_headroom_treated")),
                "Universe B banks with a usable 2019Q4 headroom baseline.",
                "Excludes banks without a valid 2019Q4 headroom measure.",
            ),
            _sample_ladder_row(
                "universe_c_high_ust_share_baseline",
                "Universe C high-UST-share-definable sample",
                _filter_entities(universe_b, _baseline_ready_entities_for_treatment(universe_b, "high_ust_share_treated")),
                "Universe B banks with a usable 2019Q4 Treasury-share baseline.",
                "Excludes banks without a valid 2019Q4 Treasury-share measure.",
            ),
            _sample_ladder_row(
                "universe_c_covered_bank_baseline",
                "Universe C covered-bank-definable sample",
                _filter_entities(universe_b, _baseline_ready_entities_for_treatment(universe_b, "covered_bank_treated")),
                "Universe B banks with a usable 2019Q4 direct-treatment classification baseline.",
                "Excludes banks without a usable 2019Q4 direct-treatment classification.",
            ),
            _sample_ladder_row(
                "universe_d_primary_core",
                "Universe D primary causal core",
                universe_d,
                "Universe B banks with full 2019Q1-2021Q4 coverage; treatment assignment is frozen on the relevant 2019Q4 baseline subset for each treatment.",
                _format_reason_counts(sample_manifest["universe_d_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_e_expanded_sensitivity",
                "Universe E expanded causal sensitivity",
                universe_e,
                f"Universe B banks with at least {MIN_PRE_QUARTERS_EXPANDED} pre-treatment and {MIN_POST_QUARTERS_EXPANDED} post-treatment quarters; treatment assignment is frozen on the relevant 2019Q4 baseline subset for each treatment.",
                _format_reason_counts(sample_manifest["universe_e_exclusion_reason"]),
            ),
            _sample_ladder_row(
                "universe_d_low_headroom_primary",
                "Universe D low-headroom estimable core",
                _treatment_defined_frame(universe_d, "low_headroom_treated"),
                "Universe D rows with nonmissing low-headroom treatment assignment.",
                "Drops balanced-core banks without a usable 2019Q4 headroom baseline.",
            ),
            _sample_ladder_row(
                "universe_d_high_ust_share_primary",
                "Universe D high-UST-share estimable core",
                _treatment_defined_frame(universe_d, "high_ust_share_treated"),
                "Universe D rows with nonmissing high-UST-share treatment assignment.",
                "Drops balanced-core banks without a usable 2019Q4 Treasury-share baseline.",
            ),
            _sample_ladder_row(
                "universe_d_covered_bank_primary",
                "Universe D covered-bank estimable core",
                _treatment_defined_frame(universe_d, "covered_bank_treated"),
                "Universe D rows with nonmissing covered-bank treatment assignment.",
                "Drops balanced-core banks without a usable 2019Q4 direct-treatment classification baseline.",
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
        flagship_clustered_pretrend,
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
        flagship_clustered_pretrend,
        flagship_placebo,
        flagship_leave_one_out,
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
