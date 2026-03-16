from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from ..config import derived_data_path, reports_path
from ..pipeline import read_table, write_frame


REGIME_ORDER = [
    "pre_exclusion",
    "temporary_exclusion",
    "post_exclusion_normalization",
    "duration_loss_window",
    "late_qt_normalization",
]


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def _assign_regime(series: pd.Series) -> pd.Series:
    quarter = pd.to_datetime(series)
    regime = pd.Series(pd.NA, index=quarter.index, dtype="object")
    regime.loc[quarter <= pd.Timestamp("2020-03-31")] = "pre_exclusion"
    regime.loc[(quarter >= pd.Timestamp("2020-06-30")) & (quarter <= pd.Timestamp("2021-03-31"))] = "temporary_exclusion"
    regime.loc[(quarter >= pd.Timestamp("2021-06-30")) & (quarter <= pd.Timestamp("2022-03-31"))] = "post_exclusion_normalization"
    regime.loc[(quarter >= pd.Timestamp("2022-06-30")) & (quarter <= pd.Timestamp("2023-12-31"))] = "duration_loss_window"
    regime.loc[quarter >= pd.Timestamp("2024-03-31")] = "late_qt_normalization"
    return regime


def _pct_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="Float64")
    valid = numeric.dropna()
    if valid.empty:
        return out
    if valid.nunique(dropna=True) <= 1:
        out.loc[valid.index] = 0.5
        return out
    out.loc[valid.index] = valid.rank(method="average", pct=True).astype("Float64")
    return out


def _mean_available(columns: list[pd.Series], index: pd.Index) -> pd.Series:
    if not columns:
        return pd.Series(pd.NA, index=index, dtype="Float64")
    frame = pd.concat(columns, axis=1)
    values = frame.mean(axis=1, skipna=True)
    return values.where(frame.notna().any(axis=1), pd.NA).astype("Float64")


def _dominant_label(row: pd.Series, mapping: dict[str, str]) -> object:
    scores = {label: row[column] for label, column in mapping.items() if column in row.index}
    available = {label: value for label, value in scores.items() if pd.notna(value)}
    if not available:
        return pd.NA
    return max(available, key=available.get)


def _prepare_panel(frame: pd.DataFrame, entity_source: str) -> pd.DataFrame:
    out = frame.copy()
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    out["entity_source"] = entity_source
    out = out.sort_values(["entity_id", "quarter_end"]).reset_index(drop=True)
    if "top_parent_rssd" in out.columns:
        out["top_parent_rssd"] = out["top_parent_rssd"].astype("string").str.strip()

    if "deposit_growth_qoq" not in out.columns and "deposits" in out.columns:
        out["deposit_growth_qoq"] = out.groupby("entity_id")["deposits"].pct_change()
    if "loan_growth_qoq" not in out.columns and "loans" in out.columns:
        out["loan_growth_qoq"] = out.groupby("entity_id")["loans"].pct_change()
    if "ust_share_assets_qoq" not in out.columns and "ust_share_assets" in out.columns:
        out["ust_share_assets_qoq"] = out.groupby("entity_id")["ust_share_assets"].diff()

    out["policy_regime"] = _assign_regime(out["quarter_end"])

    leverage_score = _pct_rank(-_numeric_series(out, "headroom_pp"))

    duration_raw = _numeric_series(out, "total_unrealized_loss_tier1")
    if duration_raw.isna().all():
        duration_raw = _numeric_series(out, "total_unrealized_loss_share_assets")
    duration_score = _pct_rank(duration_raw)

    deposit_runoff_score = _pct_rank(-_numeric_series(out, "deposit_growth_qoq"))
    deposit_funding_gap_score = _pct_rank(_numeric_series(out, "deposit_funding_gap_share_assets"))
    loan_to_deposit_score = _pct_rank(_numeric_series(out, "loan_to_deposit_ratio"))
    non_deposit_funding_score = _pct_rank(_numeric_series(out, "non_deposit_funding_share_assets"))
    repo_funding_score = _pct_rank(_numeric_series(out, "repos_share_assets"))
    funding_stress_score = _mean_available(
        [
            deposit_runoff_score,
            deposit_funding_gap_score,
            loan_to_deposit_score,
            non_deposit_funding_score,
            repo_funding_score,
        ],
        out.index,
    )
    low_liquid_asset_score = _pct_rank(-_numeric_series(out, "liquid_asset_share_assets"))
    low_safe_asset_buffer_score = _pct_rank(-_numeric_series(out, "safe_asset_buffer_share_assets"))
    low_liquid_asset_to_deposits_score = _pct_rank(-_numeric_series(out, "liquid_asset_to_deposits"))
    low_safe_asset_buffer_to_deposits_score = _pct_rank(-_numeric_series(out, "safe_asset_buffer_to_deposits"))
    high_htm_share_ust_score = _pct_rank(_numeric_series(out, "htm_share_ust"))
    liquidity_stress_score = _mean_available(
        [
            low_liquid_asset_score,
            low_safe_asset_buffer_score,
            low_liquid_asset_to_deposits_score,
            low_safe_asset_buffer_to_deposits_score,
            high_htm_share_ust_score,
        ],
        out.index,
    )
    funding_score = _mean_available(
        [funding_stress_score, liquidity_stress_score],
        out.index,
    )

    out["leverage_pressure_score"] = leverage_score
    out["duration_pressure_score"] = duration_score
    out["funding_stress_score"] = funding_stress_score
    out["liquidity_stress_score"] = liquidity_stress_score
    out["funding_pressure_score"] = funding_score

    def dominant_constraint(row: pd.Series) -> object:
        scores = {
            "leverage": row["leverage_pressure_score"],
            "duration_loss": row["duration_pressure_score"],
            "funding": row["funding_pressure_score"],
        }
        available = {name: value for name, value in scores.items() if pd.notna(value)}
        if not available:
            return pd.NA
        return max(available, key=available.get)

    out["dominant_constraint"] = out.apply(dominant_constraint, axis=1)
    return out


def _coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("entity_source", dropna=False)
        .agg(
            entity_count=("entity_id", "nunique"),
            observation_count=("entity_id", "size"),
            first_quarter=("quarter_end", "min"),
            last_quarter=("quarter_end", "max"),
        )
        .reset_index()
    )
    summary["first_quarter"] = pd.to_datetime(summary["first_quarter"])
    summary["last_quarter"] = pd.to_datetime(summary["last_quarter"])
    return summary


def _regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (entity_source, policy_regime), sample in frame.dropna(subset=["policy_regime"]).groupby(
        ["entity_source", "policy_regime"],
        dropna=False,
    ):
        dominant_share = sample["dominant_constraint"].value_counts(normalize=True, dropna=True)
        rows.append(
            {
                "entity_source": entity_source,
                "policy_regime": policy_regime,
                "quarter_count": sample["quarter_end"].nunique(),
                "entity_count": sample["entity_id"].nunique(),
                "observation_count": len(sample),
                "mean_headroom_pp": _numeric_series(sample, "headroom_pp").mean(),
                "mean_ust_share_assets": _numeric_series(sample, "ust_share_assets").mean(),
                "mean_unrealized_loss_tier1": _numeric_series(sample, "total_unrealized_loss_tier1").mean(),
                "mean_deposit_growth_qoq": _numeric_series(sample, "deposit_growth_qoq").mean(),
                "mean_liquid_asset_share_assets": _numeric_series(sample, "liquid_asset_share_assets").mean(),
                "mean_leverage_pressure_score": _numeric_series(sample, "leverage_pressure_score").mean(),
                "mean_duration_pressure_score": _numeric_series(sample, "duration_pressure_score").mean(),
                "mean_funding_stress_score": _numeric_series(sample, "funding_stress_score").mean(),
                "mean_liquidity_stress_score": _numeric_series(sample, "liquidity_stress_score").mean(),
                "mean_funding_pressure_score": _numeric_series(sample, "funding_pressure_score").mean(),
                "leverage_dominant_share": dominant_share.get("leverage", 0.0),
                "duration_loss_dominant_share": dominant_share.get("duration_loss", 0.0),
                "funding_dominant_share": dominant_share.get("funding", 0.0),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["policy_regime"] = pd.Categorical(summary["policy_regime"], categories=REGIME_ORDER, ordered=True)
    return summary.sort_values(["entity_source", "policy_regime"]).reset_index(drop=True)


def _absorption_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.dropna(subset=["policy_regime", "dominant_constraint"]).groupby(
        ["entity_source", "policy_regime", "dominant_constraint"],
        dropna=False,
    )
    for (entity_source, policy_regime, dominant_constraint), sample in grouped:
        rows.append(
            {
                "entity_source": entity_source,
                "policy_regime": policy_regime,
                "dominant_constraint": dominant_constraint,
                "quarter_count": sample["quarter_end"].nunique(),
                "entity_count": sample["entity_id"].nunique(),
                "observation_count": len(sample),
                "mean_ust_share_assets": _numeric_series(sample, "ust_share_assets").mean(),
                "mean_ust_share_assets_qoq": _numeric_series(sample, "ust_share_assets_qoq").mean(),
                "mean_safe_asset_buffer_share_assets": _numeric_series(sample, "safe_asset_buffer_share_assets").mean(),
                "mean_balances_due_from_fed_share_assets": _numeric_series(sample, "balances_due_from_fed_share_assets").mean(),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["policy_regime"] = pd.Categorical(summary["policy_regime"], categories=REGIME_ORDER, ordered=True)
    return summary.sort_values(["entity_source", "policy_regime", "dominant_constraint"]).reset_index(drop=True)


def _regime_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    insured = summary[summary["entity_source"] == "insured_bank"].copy()
    parent = summary[summary["entity_source"] == "parent_or_ihc"].copy()
    if insured.empty or parent.empty:
        return pd.DataFrame()

    drop_columns = ["entity_source"]
    insured = insured.drop(columns=drop_columns).add_prefix("insured_").rename(
        columns={"insured_policy_regime": "policy_regime"}
    )
    parent = parent.drop(columns=drop_columns).add_prefix("parent_").rename(
        columns={"parent_policy_regime": "policy_regime"}
    )

    comparison = insured.merge(parent, how="inner", on="policy_regime")
    if comparison.empty:
        return comparison

    comparison["duration_loss_dominant_share_gap"] = (
        comparison["insured_duration_loss_dominant_share"] - comparison["parent_duration_loss_dominant_share"]
    )
    comparison["funding_dominant_share_gap"] = (
        comparison["insured_funding_dominant_share"] - comparison["parent_funding_dominant_share"]
    )
    comparison["leverage_dominant_share_gap"] = (
        comparison["insured_leverage_dominant_share"] - comparison["parent_leverage_dominant_share"]
    )
    comparison["ust_share_assets_gap"] = (
        comparison["insured_mean_ust_share_assets"] - comparison["parent_mean_ust_share_assets"]
    )
    comparison["insured_dominant_constraint"] = comparison.apply(
        lambda row: _dominant_label(
            row,
            {
                "leverage": "insured_leverage_dominant_share",
                "duration_loss": "insured_duration_loss_dominant_share",
                "funding": "insured_funding_dominant_share",
            },
        ),
        axis=1,
    )
    comparison["parent_dominant_constraint"] = comparison.apply(
        lambda row: _dominant_label(
            row,
            {
                "leverage": "parent_leverage_dominant_share",
                "duration_loss": "parent_duration_loss_dominant_share",
                "funding": "parent_funding_dominant_share",
            },
        ),
        axis=1,
    )
    comparison["policy_regime"] = pd.Categorical(comparison["policy_regime"], categories=REGIME_ORDER, ordered=True)
    return comparison.sort_values("policy_regime").reset_index(drop=True)


def _family_alignment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"top_parent_rssd", "quarter_end", "policy_regime", "dominant_constraint", "entity_source"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    bank = frame[frame["entity_source"] == "insured_bank"].copy()
    parent = frame[frame["entity_source"] == "parent_or_ihc"].copy()
    if bank.empty or parent.empty:
        return pd.DataFrame()

    bank = bank.dropna(subset=["top_parent_rssd", "quarter_end", "policy_regime"]).copy()
    parent = parent.dropna(subset=["top_parent_rssd", "quarter_end", "policy_regime"]).copy()
    if bank.empty or parent.empty:
        return pd.DataFrame()

    bank["ust_share_assets"] = _numeric_series(bank, "ust_share_assets")
    bank["ust_share_assets_qoq"] = _numeric_series(bank, "ust_share_assets_qoq")
    bank["safe_asset_buffer_share_assets"] = _numeric_series(bank, "safe_asset_buffer_share_assets")
    bank["is_leverage"] = (bank["dominant_constraint"] == "leverage").astype("Float64")
    bank["is_duration_loss"] = (bank["dominant_constraint"] == "duration_loss").astype("Float64")
    bank["is_funding"] = (bank["dominant_constraint"] == "funding").astype("Float64")

    bank_family = (
        bank.groupby(["top_parent_rssd", "quarter_end", "policy_regime"], dropna=False)
        .agg(
            bank_entity_count=("entity_id", "nunique"),
            bank_observation_count=("entity_id", "size"),
            bank_mean_ust_share_assets=("ust_share_assets", "mean"),
            bank_mean_ust_share_assets_qoq=("ust_share_assets_qoq", "mean"),
            bank_mean_safe_asset_buffer_share_assets=("safe_asset_buffer_share_assets", "mean"),
            bank_leverage_share=("is_leverage", "mean"),
            bank_duration_loss_share=("is_duration_loss", "mean"),
            bank_funding_share=("is_funding", "mean"),
        )
        .reset_index()
    )
    bank_family["bank_family_dominant_constraint"] = bank_family.apply(
        lambda row: _dominant_label(
            row,
            {
                "leverage": "bank_leverage_share",
                "duration_loss": "bank_duration_loss_share",
                "funding": "bank_funding_share",
            },
        ),
        axis=1,
    )

    parent["ust_share_assets"] = _numeric_series(parent, "ust_share_assets")
    parent["ust_share_assets_qoq"] = _numeric_series(parent, "ust_share_assets_qoq")
    parent["safe_asset_buffer_share_assets"] = _numeric_series(parent, "safe_asset_buffer_share_assets")
    parent_family = parent[
        [
            "top_parent_rssd",
            "quarter_end",
            "policy_regime",
            "entity_id",
            "dominant_constraint",
            "ust_share_assets",
            "ust_share_assets_qoq",
            "safe_asset_buffer_share_assets",
        ]
    ].copy()
    parent_family = parent_family.rename(
        columns={
            "entity_id": "parent_entity_id",
            "dominant_constraint": "parent_dominant_constraint",
            "ust_share_assets": "parent_ust_share_assets",
            "ust_share_assets_qoq": "parent_ust_share_assets_qoq",
            "safe_asset_buffer_share_assets": "parent_safe_asset_buffer_share_assets",
        }
    )

    linked = bank_family.merge(
        parent_family,
        how="inner",
        on=["top_parent_rssd", "quarter_end", "policy_regime"],
    )
    if linked.empty:
        return linked

    linked["matched_dominant_constraint"] = (
        linked["bank_family_dominant_constraint"] == linked["parent_dominant_constraint"]
    ).astype("Float64")
    linked["both_duration_loss"] = (
        (linked["bank_family_dominant_constraint"] == "duration_loss")
        & (linked["parent_dominant_constraint"] == "duration_loss")
    ).astype("Float64")
    linked["bank_minus_parent_ust_share_assets"] = (
        linked["bank_mean_ust_share_assets"] - linked["parent_ust_share_assets"]
    )
    linked["bank_minus_parent_ust_share_assets_qoq"] = (
        linked["bank_mean_ust_share_assets_qoq"] - linked["parent_ust_share_assets_qoq"]
    )

    summary = (
        linked.groupby("policy_regime", dropna=False)
        .agg(
            family_quarter_count=("top_parent_rssd", "size"),
            linked_parent_count=("top_parent_rssd", "nunique"),
            mean_bank_entity_count=("bank_entity_count", "mean"),
            matched_dominant_constraint_share=("matched_dominant_constraint", "mean"),
            both_duration_loss_share=("both_duration_loss", "mean"),
            bank_duration_loss_family_share=("bank_duration_loss_share", "mean"),
            parent_duration_loss_share=("parent_dominant_constraint", lambda s: (s == "duration_loss").mean()),
            mean_bank_ust_share_assets=("bank_mean_ust_share_assets", "mean"),
            mean_parent_ust_share_assets=("parent_ust_share_assets", "mean"),
            mean_bank_minus_parent_ust_share_assets=("bank_minus_parent_ust_share_assets", "mean"),
            mean_bank_minus_parent_ust_share_assets_qoq=("bank_minus_parent_ust_share_assets_qoq", "mean"),
        )
        .reset_index()
    )
    summary["policy_regime"] = pd.Categorical(summary["policy_regime"], categories=REGIME_ORDER, ordered=True)
    return summary.sort_values("policy_regime").reset_index(drop=True)


def _cluster_groups(frame: pd.DataFrame) -> pd.Series | None:
    if "top_parent_rssd" in frame.columns:
        groups = frame["top_parent_rssd"].astype("string").replace({"": pd.NA})
        if groups.nunique(dropna=True) >= 2:
            return groups.fillna("entity:" + frame["entity_id"].astype(str))
    groups = frame["entity_id"].astype("string").replace({"": pd.NA})
    if groups.nunique(dropna=True) >= 2:
        return groups.fillna("obs:" + frame.index.astype(str))
    return None


def _interaction_regression_summary(frame: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["ust_share_assets", "ust_share_assets_qoq"]
    focus_regimes = ["duration_loss_window", "late_qt_normalization"]
    score_terms = ["leverage_pressure_score", "duration_pressure_score", "funding_pressure_score"]
    min_observations = 12
    rows: list[dict[str, object]] = []

    prepared = frame.copy()
    prepared["quarter_end"] = pd.to_datetime(prepared["quarter_end"])

    for entity_source, sample in prepared.groupby("entity_source", dropna=False):
        for outcome in outcomes:
            outcome_sample = sample.dropna(
                subset=[
                    outcome,
                    "leverage_pressure_score",
                    "duration_pressure_score",
                    "funding_pressure_score",
                    "quarter_end",
                    "entity_id",
                ]
            ).copy()
            for focus_regime in focus_regimes:
                design = outcome_sample.copy()
                design["focus_regime"] = (design["policy_regime"] == focus_regime).astype(int)
                term_count = 1 + len(score_terms) + len(score_terms) + max(design["entity_id"].nunique() - 1, 0) + max(
                    design["quarter_end"].nunique() - 1, 0
                )
                if (
                    len(design) < min_observations
                    or design["focus_regime"].nunique() < 2
                    or design["entity_id"].nunique() < 2
                    or design["quarter_end"].nunique() < 2
                    or len(design) <= term_count
                ):
                    rows.append(
                        {
                            "entity_source": entity_source,
                            "outcome": outcome,
                            "focus_regime": focus_regime,
                            "term": pd.NA,
                            "coef": pd.NA,
                            "stderr": pd.NA,
                            "pvalue": pd.NA,
                            "nobs": len(design),
                            "rsquared": pd.NA,
                            "covariance": pd.NA,
                            "status": "insufficient_data",
                        }
                    )
                    continue

                formula = (
                    f"{outcome} ~ leverage_pressure_score + duration_pressure_score + funding_pressure_score + "
                    "focus_regime:leverage_pressure_score + "
                    "focus_regime:duration_pressure_score + "
                    "focus_regime:funding_pressure_score + "
                    "C(entity_id) + C(quarter_end)"
                )
                groups = _cluster_groups(design)
                try:
                    if groups is None:
                        model = smf.ols(formula, data=design).fit(cov_type="HC1")
                        covariance = "HC1"
                    else:
                        model = smf.ols(formula, data=design).fit(
                            cov_type="cluster",
                            cov_kwds={"groups": groups},
                        )
                        covariance = "cluster_top_parent_or_entity"
                except Exception as exc:  # pragma: no cover - defensive for live panels
                    rows.append(
                        {
                            "entity_source": entity_source,
                            "outcome": outcome,
                            "focus_regime": focus_regime,
                            "term": pd.NA,
                            "coef": pd.NA,
                            "stderr": pd.NA,
                            "pvalue": pd.NA,
                            "nobs": len(design),
                            "rsquared": pd.NA,
                            "covariance": pd.NA,
                            "status": f"error:{type(exc).__name__}",
                        }
                    )
                    continue

                for term in [
                    "focus_regime:leverage_pressure_score",
                    "focus_regime:duration_pressure_score",
                    "focus_regime:funding_pressure_score",
                ]:
                    rows.append(
                        {
                            "entity_source": entity_source,
                            "outcome": outcome,
                            "focus_regime": focus_regime,
                            "term": term,
                            "coef": model.params.get(term, pd.NA),
                            "stderr": model.bse.get(term, pd.NA),
                            "pvalue": model.pvalues.get(term, pd.NA),
                            "nobs": int(model.nobs),
                            "rsquared": model.rsquared,
                            "covariance": covariance,
                            "status": "ok",
                        }
                    )

    return pd.DataFrame(rows)


def _write_summary(
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    absorption: pd.DataFrame,
    regime_comparison: pd.DataFrame,
    family_alignment: pd.DataFrame,
    interaction_regressions: pd.DataFrame,
    output_path: Path,
) -> None:
    def fmt_metric(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.4f}"

    lines = [
        "# Constraint Decomposition Beyond SLR",
        "",
        "This report adds three public-data constraint channels to the current `slrwatch` stack:",
        "- leverage tightness (`headroom_pp`)",
        "- duration-loss pressure (Treasury unrealized-loss proxies from amortized vs fair-value gaps)",
        "- funding pressure (funding stress plus liquidity stress from deposits, repos, liquid buffers, and HTM mix)",
        "",
        "## Coverage",
    ]

    if coverage.empty:
        lines.append("- No bank or parent panel rows were available.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    coverage_by_source = {row["entity_source"]: row for _, row in coverage.iterrows()}
    for source in ["insured_bank", "parent_or_ihc"]:
        row = coverage_by_source.get(source)
        if row is None:
            continue
        lines.append(
            f"- `{source}`: {int(row['entity_count'])} entities, {int(row['observation_count'])} rows, "
            f"{row['first_quarter'].date()} to {row['last_quarter'].date()}"
        )

    insured = coverage_by_source.get("insured_bank")
    parent = coverage_by_source.get("parent_or_ihc")
    if insured is not None and parent is not None and insured["last_quarter"] < parent["last_quarter"]:
        lines.extend(
            [
                "",
                "## Current limitation",
                f"- The insured-bank panel currently ends at {insured['last_quarter'].date()}, so later duration-loss regimes are parent-heavy until more Call Report quarters are staged.",
            ]
        )

    lines.extend(["", "## Dominant constraint by regime"])
    if summary.empty:
        lines.append("- No regime summary rows were available.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for _, row in summary.iterrows():
        shares = {
            "leverage": row["leverage_dominant_share"],
            "duration_loss": row["duration_loss_dominant_share"],
            "funding": row["funding_dominant_share"],
        }
        dominant = max(shares, key=shares.get)
        lines.append(
            f"- `{row['entity_source']}` / `{row['policy_regime']}`: "
            f"`{dominant}` dominates {shares[dominant]:.1%} of scored rows; "
            f"mean headroom {fmt_metric(row['mean_headroom_pp'])}, "
            f"mean unrealized-loss / Tier 1 {fmt_metric(row['mean_unrealized_loss_tier1'])}, "
            f"mean deposit growth {fmt_metric(row['mean_deposit_growth_qoq'])}."
        )

    lines.extend(["", "## Cross-panel regime comparison"])
    if regime_comparison.empty:
        lines.append("- No insured-bank versus parent regime overlaps were available.")
    else:
        for _, row in regime_comparison.iterrows():
            lines.append(
                f"- `{row['policy_regime']}`: insured banks lean `{row['insured_dominant_constraint']}` "
                f"while parents/IHCs lean `{row['parent_dominant_constraint']}`; "
                f"duration-loss share gap {fmt_metric(row['duration_loss_dominant_share_gap'])}, "
                f"funding share gap {fmt_metric(row['funding_dominant_share_gap'])}, "
                f"Treasury-share gap {fmt_metric(row['ust_share_assets_gap'])}."
            )

    lines.extend(["", "## Treasury absorption by dominant constraint"])
    if absorption.empty:
        lines.append("- No absorption summary rows were available.")
    else:
        focus = absorption[
            absorption["policy_regime"].isin(["duration_loss_window", "late_qt_normalization"])
        ].copy()
        if focus.empty:
            focus = absorption.copy()
        focus = focus.sort_values(["entity_source", "policy_regime", "mean_ust_share_assets_qoq"], ascending=[True, True, False])
        for _, row in focus.iterrows():
            lines.append(
                f"- `{row['entity_source']}` / `{row['policy_regime']}` / `{row['dominant_constraint']}`: "
                f"mean Treasury share {fmt_metric(row['mean_ust_share_assets'])}, "
                f"mean QoQ Treasury-share change {fmt_metric(row['mean_ust_share_assets_qoq'])}, "
                f"mean safe-asset buffer {fmt_metric(row['mean_safe_asset_buffer_share_assets'])}."
            )

    lines.extend(["", "## Parent-bank family alignment"])
    if family_alignment.empty:
        lines.append("- No linked parent-bank family quarters were available for alignment analysis.")
    else:
        for _, row in family_alignment.iterrows():
            lines.append(
                f"- `{row['policy_regime']}`: matched dominant constraint in {row['matched_dominant_constraint_share']:.1%} "
                f"of linked family-quarters; both bank and parent are duration-loss dominant in {row['both_duration_loss_share']:.1%}; "
                f"mean bank-minus-parent Treasury share {fmt_metric(row['mean_bank_minus_parent_ust_share_assets'])}."
            )

    lines.extend(["", "## Interaction regressions"])
    if interaction_regressions.empty:
        lines.append("- No interaction-regression rows were available.")
    else:
        successful = interaction_regressions[interaction_regressions["status"] == "ok"].copy()
        if successful.empty:
            lines.append("- Interaction regressions were skipped because the available sample was too small or a model failed.")
        else:
            ranked = successful.copy()
            ranked["abs_coef"] = pd.to_numeric(ranked["coef"], errors="coerce").abs()
            ranked = ranked.sort_values(["entity_source", "outcome", "focus_regime", "abs_coef"], ascending=[True, True, True, False])
            top_rows = ranked.groupby(["entity_source", "outcome", "focus_regime"], dropna=False).head(1)
            term_labels = {
                "focus_regime:leverage_pressure_score": "leverage",
                "focus_regime:duration_pressure_score": "duration_loss",
                "focus_regime:funding_pressure_score": "funding",
            }
            for _, row in top_rows.iterrows():
                lines.append(
                    f"- `{row['entity_source']}` / `{row['outcome']}` / `{row['focus_regime']}`: "
                    f"largest interaction term is `{term_labels.get(row['term'], row['term'])}` with coef {fmt_metric(row['coef'])} "
                    f"and p-value {fmt_metric(row['pvalue'])}."
                )
        skipped = interaction_regressions[interaction_regressions["status"] != "ok"]
        if not skipped.empty:
            lines.append(
                f"- {len(skipped)} interaction-spec rows were skipped or failed; see `interaction_regime_summary.csv` for model status."
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_constraint_decomposition_report(
    bank_panel_path: Path | None = None,
    parent_panel_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    bank = read_table(bank_panel_path or derived_data_path("insured_bank_panel.parquet"))
    parent = read_table(parent_panel_path or derived_data_path("parent_panel.parquet"))

    prepared = pd.concat(
        [
            _prepare_panel(bank, "insured_bank"),
            _prepare_panel(parent, "parent_or_ihc"),
        ],
        ignore_index=True,
    )
    prepared = prepared.dropna(subset=["policy_regime"]).reset_index(drop=True)

    coverage = _coverage_summary(prepared)
    summary = _regime_summary(prepared)
    absorption = _absorption_summary(prepared)
    regime_comparison = _regime_comparison(summary)
    family_alignment = _family_alignment_summary(prepared)
    interaction_regressions = _interaction_regression_summary(prepared)

    destination = output_dir or reports_path("constraint_decomposition")
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(
        prepared.assign(quarter_end=prepared["quarter_end"].dt.strftime("%Y-%m-%d")),
        destination / "prepared_panel.csv",
    )
    if not coverage.empty:
        write_frame(
            coverage.assign(
                first_quarter=coverage["first_quarter"].dt.strftime("%Y-%m-%d"),
                last_quarter=coverage["last_quarter"].dt.strftime("%Y-%m-%d"),
            ),
            destination / "coverage_summary.csv",
        )
    else:
        write_frame(coverage, destination / "coverage_summary.csv")
    write_frame(summary, destination / "regime_summary.csv")
    write_frame(absorption, destination / "absorption_summary.csv")
    write_frame(regime_comparison, destination / "regime_comparison.csv")
    write_frame(family_alignment, destination / "family_alignment_summary.csv")
    write_frame(interaction_regressions, destination / "interaction_regime_summary.csv")
    _write_summary(
        coverage,
        summary,
        absorption,
        regime_comparison,
        family_alignment,
        interaction_regressions,
        destination / "summary.md",
    )
    return destination
