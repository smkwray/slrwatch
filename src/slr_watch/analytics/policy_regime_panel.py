from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import derived_data_path, reports_path
from ..pipeline import read_table, write_frame


REGIME_ORDER = [
    "pre_exclusion",
    "temporary_exclusion",
    "post_exclusion_normalization",
    "qt_era",
]


def _assign_regime(series: pd.Series) -> pd.Series:
    quarter = pd.to_datetime(series)
    regime = pd.Series(pd.NA, index=quarter.index, dtype="object")
    regime.loc[quarter <= pd.Timestamp("2020-03-31")] = "pre_exclusion"
    regime.loc[(quarter >= pd.Timestamp("2020-06-30")) & (quarter <= pd.Timestamp("2021-03-31"))] = "temporary_exclusion"
    regime.loc[(quarter >= pd.Timestamp("2021-06-30")) & (quarter <= pd.Timestamp("2022-03-31"))] = "post_exclusion_normalization"
    regime.loc[quarter >= pd.Timestamp("2022-06-30")] = "qt_era"
    return regime


def _quarter_aggregate(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    metrics = {
        "ust_share_assets": f"{prefix}_ust_share_assets_mean",
        "balances_due_from_fed_share_assets": f"{prefix}_fed_share_assets_mean",
        "trading_assets_total_share_assets": f"{prefix}_trading_share_assets_mean",
        "reverse_repos_share_assets": f"{prefix}_reverse_repo_share_assets_mean",
        "headroom_pp": f"{prefix}_headroom_pp_mean",
    }
    existing = {source: target for source, target in metrics.items() if source in frame.columns}
    if not existing:
        return pd.DataFrame(columns=["quarter_end", f"{prefix}_entity_count"])
    grouped = (
        frame.groupby("quarter_end", dropna=False)
        .agg(
            **{target: (source, "mean") for source, target in existing.items()},
            **{f"{prefix}_entity_count": ("entity_id", "nunique")},
        )
        .reset_index()
    )
    grouped["quarter_end"] = pd.to_datetime(grouped["quarter_end"])
    return grouped


def _market_quarter_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keep = ["quarter_end"]
    for column in [
        "pd_ust_dealer_position_net_mn",
        "pd_ust_repo_mn_weekly_avg",
        "pd_ust_reverse_repo_mn_weekly_avg",
        "trace_total_par_value_bn",
    ]:
        if column in frame.columns:
            keep.append(column)
    out = frame[keep].copy()
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    return out.drop_duplicates("quarter_end")


def _build_regime_quarter_panel(
    bank: pd.DataFrame,
    parent: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    bank_q = _quarter_aggregate(bank, "bank")
    parent_q = _quarter_aggregate(parent, "parent")
    market_q = _market_quarter_aggregate(market)
    merged = market_q.merge(bank_q, how="outer", on="quarter_end").merge(parent_q, how="outer", on="quarter_end")
    merged["quarter_end"] = pd.to_datetime(merged["quarter_end"])
    merged = merged.sort_values("quarter_end").reset_index(drop=True)
    merged["policy_regime"] = _assign_regime(merged["quarter_end"])
    return merged.dropna(subset=["policy_regime"]).reset_index(drop=True)


def _regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in frame.columns if column not in {"quarter_end", "policy_regime"}]
    summary = (
        frame.groupby("policy_regime", dropna=False)
        .agg(
            regime_quarters=("quarter_end", "nunique"),
            **{column: (column, "mean") for column in metric_columns},
        )
        .reset_index()
    )
    summary["policy_regime"] = pd.Categorical(summary["policy_regime"], categories=REGIME_ORDER, ordered=True)
    return summary.sort_values("policy_regime").reset_index(drop=True)


def _line_for_change(
    summary: pd.DataFrame,
    base_regime: str,
    compare_regime: str,
    column: str,
    label: str,
) -> str | None:
    if column not in summary.columns:
        return None
    indexed = summary.set_index("policy_regime")
    if base_regime not in indexed.index or compare_regime not in indexed.index:
        return None
    base = indexed.loc[base_regime, column]
    compare = indexed.loc[compare_regime, column]
    if pd.isna(base) or pd.isna(compare):
        return None
    return f"- `{label}` moved from {base:.4f} in `{base_regime}` to {compare:.4f} in `{compare_regime}` ({compare - base:+.4f})."


def _write_summary(regime_quarter: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Broader Policy-Regime Panel",
        "",
        f"- Quarter rows: {len(regime_quarter)}",
        f"- Quarter range: {regime_quarter['quarter_end'].min().date()} to {regime_quarter['quarter_end'].max().date()}",
        "",
        "## Regimes",
        "- `pre_exclusion`: through 2020-03-31",
        "- `temporary_exclusion`: 2020-06-30 through 2021-03-31",
        "- `post_exclusion_normalization`: 2021-06-30 through 2022-03-31",
        "- `qt_era`: 2022-06-30 onward",
        "",
        "## Readout",
    ]

    for item in [
        _line_for_change(summary, "pre_exclusion", "temporary_exclusion", "bank_ust_share_assets_mean", "insured-bank Treasury share"),
        _line_for_change(summary, "pre_exclusion", "temporary_exclusion", "bank_fed_share_assets_mean", "insured-bank Fed-balance share"),
        _line_for_change(summary, "temporary_exclusion", "qt_era", "parent_trading_share_assets_mean", "parent trading-assets share"),
        _line_for_change(summary, "temporary_exclusion", "qt_era", "pd_ust_dealer_position_net_mn", "NY Fed dealer net UST position"),
        _line_for_change(summary, "temporary_exclusion", "qt_era", "trace_total_par_value_bn", "TRACE total par volume"),
    ]:
        if item:
            lines.append(item)

    if len(lines) == 11:
        lines.append("- Not enough overlapping data was available to compare the configured regimes.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_policy_regime_panel_report(
    bank_panel_path: Path | None = None,
    parent_panel_path: Path | None = None,
    market_panel_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    bank = read_table(bank_panel_path or derived_data_path("insured_bank_panel.parquet"))
    parent = read_table(parent_panel_path or derived_data_path("parent_panel.parquet"))
    market = read_table(market_panel_path or derived_data_path("market_overlay_panel.parquet"))

    regime_quarter = _build_regime_quarter_panel(bank, parent, market)
    summary = _regime_summary(regime_quarter)

    destination = output_dir or reports_path("policy_regime_panel")
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(regime_quarter.assign(quarter_end=regime_quarter["quarter_end"].dt.strftime("%Y-%m-%d")), destination / "regime_quarter_panel.csv")
    write_frame(summary, destination / "regime_summary.csv")
    _write_summary(regime_quarter, summary, destination / "summary.md")
    return destination
