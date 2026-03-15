from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import derived_data_path, reports_path
from ..pipeline import read_table, write_frame

PLOT_COLUMNS = {
    "pd_ust_dealer_position_net_mn": "NY Fed dealer net UST position",
    "pd_ust_repo_mn_weekly_avg": "NY Fed UST repo",
    "trace_total_par_value_bn": "TRACE total par volume",
}


def _pct_change(first: float | int | None, last: float | int | None) -> float | None:
    if first in (None, 0) or pd.isna(first) or pd.isna(last):
        return None
    return ((float(last) / float(first)) - 1.0) * 100.0


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1f}%"


def _format_level(value: float | int | None, *, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _plot_market_context(common: pd.DataFrame, output_path: Path) -> None:
    plot_ready = common.copy()
    for column in PLOT_COLUMNS:
        base = plot_ready[column].iloc[0]
        plot_ready[column] = (plot_ready[column] / base) * 100.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for column, label in PLOT_COLUMNS.items():
        ax.plot(pd.to_datetime(plot_ready["quarter_end"]), plot_ready[column], marker="o", label=label)
    ax.axhline(100, color="gray", linewidth=1, linestyle="--")
    ax.set_title("Treasury Market Context Index (common overlap = 100 at start)")
    ax.set_xlabel("Quarter end")
    ax.set_ylabel("Index")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_market_context_report(panel_path: Path | None = None, output_dir: Path | None = None) -> Path:
    source = panel_path or derived_data_path("market_overlay_panel.parquet")
    panel = read_table(source).copy()
    panel["quarter_end"] = pd.to_datetime(panel["quarter_end"])
    panel = panel.sort_values("quarter_end").reset_index(drop=True)

    required_common = list(PLOT_COLUMNS)
    common = panel.dropna(subset=required_common).copy()

    destination = output_dir or reports_path("market_context")
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(panel.assign(quarter_end=panel["quarter_end"].dt.strftime("%Y-%m-%d")), destination / "prepared_panel.csv")
    write_frame(common.assign(quarter_end=common["quarter_end"].dt.strftime("%Y-%m-%d")), destination / "common_overlap_panel.csv")

    summary_lines = [
        "# Treasury Market Context",
        "",
        "## Coverage",
        f"- Full quarterly overlay rows: {len(panel)}",
        f"- Full overlay range: {panel['quarter_end'].min().date()} to {panel['quarter_end'].max().date()}",
        f"- Common NY Fed + TRACE overlap rows: {len(common)}",
    ]

    if common.empty:
        summary_lines.extend(
            [
                "- Common overlap range: n/a",
                "",
                "## Readout",
                "- TRACE columns are not yet populated alongside NY Fed data in the market overlay panel.",
            ]
        )
        (destination / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        return destination

    summary_lines.append(f"- Common overlap range: {common['quarter_end'].min().date()} to {common['quarter_end'].max().date()}")
    summary_lines.extend(["", "## Readout"])

    start = common.iloc[0]
    latest = common.iloc[-1]
    prev_year = common[common["quarter_end"] == (latest["quarter_end"] - pd.DateOffset(years=1))]
    previous = prev_year.iloc[0] if not prev_year.empty else None

    summary_lines.extend(
        [
            f"- NY Fed dealer net UST position moved from {start['pd_ust_dealer_position_net_mn']:.0f} to {latest['pd_ust_dealer_position_net_mn']:.0f} million, a {_format_pct(_pct_change(start['pd_ust_dealer_position_net_mn'], latest['pd_ust_dealer_position_net_mn']))} change over the common sample.",
            f"- NY Fed UST repo moved from {start['pd_ust_repo_mn_weekly_avg']:.0f} to {latest['pd_ust_repo_mn_weekly_avg']:.0f} million, a {_format_pct(_pct_change(start['pd_ust_repo_mn_weekly_avg'], latest['pd_ust_repo_mn_weekly_avg']))} change over the common sample.",
            f"- TRACE total par volume moved from {start['trace_total_par_value_bn']:.1f} to {latest['trace_total_par_value_bn']:.1f} billion, a {_format_pct(_pct_change(start['trace_total_par_value_bn'], latest['trace_total_par_value_bn']))} change over the common sample.",
        ]
    )

    trade_count_common = common.dropna(subset=["trace_total_trade_count"]).copy() if "trace_total_trade_count" in common.columns else pd.DataFrame()
    if trade_count_common.empty:
        summary_lines.append("- TRACE total trade count is not available for the full common sample because the older free weekly archive publishes par-value aggregates but not trade counts.")
    else:
        trade_count_start = trade_count_common.iloc[0]
        trade_count_latest = trade_count_common.iloc[-1]
        summary_lines.append(
            f"- TRACE total trade count moved from {_format_level(trade_count_start['trace_total_trade_count'])} to {_format_level(trade_count_latest['trace_total_trade_count'])}, a {_format_pct(_pct_change(trade_count_start['trace_total_trade_count'], trade_count_latest['trace_total_trade_count']))} change over the trade-count sub-sample ({trade_count_common['quarter_end'].min().date()} to {trade_count_common['quarter_end'].max().date()})."
        )

    if previous is not None:
        summary_lines.extend(
            [
                "",
                "## Latest quarter vs prior year",
                f"- Latest quarter in common overlap: {latest['quarter_end'].date()}",
                f"- NY Fed dealer net UST position: {_format_pct(_pct_change(previous['pd_ust_dealer_position_net_mn'], latest['pd_ust_dealer_position_net_mn']))}",
                f"- NY Fed UST repo: {_format_pct(_pct_change(previous['pd_ust_repo_mn_weekly_avg'], latest['pd_ust_repo_mn_weekly_avg']))}",
                f"- TRACE total par volume: {_format_pct(_pct_change(previous['trace_total_par_value_bn'], latest['trace_total_par_value_bn']))}",
            ]
        )
        if "trace_total_trade_count" in common.columns and not pd.isna(previous.get("trace_total_trade_count")) and not pd.isna(latest.get("trace_total_trade_count")):
            summary_lines.append(
                f"- TRACE total trade count: {_format_pct(_pct_change(previous['trace_total_trade_count'], latest['trace_total_trade_count']))}"
            )

    _plot_market_context(common.assign(quarter_end=common["quarter_end"].dt.strftime("%Y-%m-%d")), destination / "market_context_index.png")
    (destination / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return destination
