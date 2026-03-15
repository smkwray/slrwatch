from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import derived_data_path, reports_path
from ..pipeline import read_table, write_frame
from .event_2020 import TREATMENTS, add_treatments, prepare_event_2020_panel

REALLOCATION_OUTCOMES = {
    "ust_inventory_fv_scaled": "Treasury inventory / assets",
    "balances_due_from_fed_scaled": "Fed balances / assets",
    "reverse_repos_scaled": "Reverse repo / assets",
    "trading_assets_scaled": "Trading assets / assets",
    "deposit_growth": "Deposit growth",
    "loan_growth": "Loan growth",
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


def _summarize_reallocation(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment in [item.name for item in TREATMENTS]:
        if treatment not in frame.columns:
            continue
        for outcome in REALLOCATION_OUTCOMES:
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
                    "outcome_label": REALLOCATION_OUTCOMES[outcome],
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


def _write_summary(summary_frame: pd.DataFrame, prepared: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Balance-Sheet Reallocation Under SLR",
        "",
        f"- Observations: {len(prepared)}",
        f"- Distinct entities: {prepared['entity_id'].nunique()}",
        f"- Quarter range: {prepared['quarter_end'].min().date()} to {prepared['quarter_end'].max().date()}",
        "",
        "## Headline read",
    ]
    if summary_frame.empty:
        lines.append("- No reallocation summary rows were estimable.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    focus = summary_frame[
        (
            summary_frame["treatment"].isin(["low_headroom_treated", "covered_bank_treated"])
        )
        & (
            summary_frame["outcome"].isin(
            [
                "ust_inventory_fv_scaled",
                "balances_due_from_fed_scaled",
                "reverse_repos_scaled",
                "loan_growth",
                "deposit_growth",
            ]
            )
        )
    ].copy()
    focus = focus.sort_values(["treatment", "outcome"])
    for _, row in focus.iterrows():
        lines.append(
            f"- `{row['treatment']}` / `{row['outcome']}`: treated change {row['treated_change']:.4f}, "
            f"control change {row['control_change']:.4f}, net {row['did_like_change']:.4f}"
        )

    strongest = summary_frame.dropna(subset=["did_like_change"]).copy()
    if not strongest.empty:
        strongest = strongest.reindex(strongest["did_like_change"].abs().sort_values(ascending=False).index).head(6)
        lines.extend(["", "## Largest net reallocations"])
        for _, row in strongest.iterrows():
            lines.append(
                f"- `{row['treatment']}` shifted `{row['outcome_label']}` by {row['did_like_change']:.4f} more than its comparison group."
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reallocation_report(panel_path: Path | None = None, output_dir: Path | None = None) -> Path:
    source = panel_path or derived_data_path("insured_bank_panel.parquet")
    panel = read_table(source)
    prepared = prepare_event_2020_panel(panel)
    prepared = _event_window(prepared)
    prepared = add_treatments(prepared)

    summary_frame = _summarize_reallocation(prepared)

    destination = output_dir or reports_path("reallocation_2020")
    destination.mkdir(parents=True, exist_ok=True)
    write_frame(prepared.assign(quarter_end=prepared["quarter_end"].dt.strftime("%Y-%m-%d")), destination / "prepared_panel.csv")
    write_frame(summary_frame, destination / "reallocation_summary.csv")
    _write_summary(summary_frame, prepared, destination / "summary.md")
    return destination
