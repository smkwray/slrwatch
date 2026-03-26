from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..pipeline import read_table, write_frame


def _link_panels(bank: pd.DataFrame, parent: pd.DataFrame) -> pd.DataFrame:
    """Link insured-bank rows to parent rows on quarter and parent identifier."""
    b = bank.copy()
    p = parent.copy()
    b["quarter_end"] = pd.to_datetime(b["quarter_end"])
    p["quarter_end"] = pd.to_datetime(p["quarter_end"])

    b["top_parent_rssd"] = b["top_parent_rssd"].astype("string").str.strip()
    p["rssd_id"] = p["rssd_id"].astype("string").str.strip()

    bank_cols = [
        "entity_id", "entity_name", "quarter_end", "top_parent_rssd",
        "total_assets", "ust_inventory_fv", "ust_share_assets",
        "balances_due_from_fed_share_assets",
        "trading_assets_total_share_assets",
        "headroom_pp", "headroom_dollars",
        "tier1_capital", "total_leverage_exposure",
        "is_covered_bank_subsidiary", "parent_method1_surcharge",
    ]
    bank_cols = [c for c in bank_cols if c in b.columns]

    parent_cols = [
        "entity_id", "entity_name", "quarter_end", "rssd_id",
        "total_assets", "ust_inventory_fv", "ust_share_assets",
        "trading_assets_total_share_assets",
        "headroom_pp", "headroom_dollars",
        "tier1_capital", "total_leverage_exposure",
        "parent_method1_surcharge",
    ]
    parent_cols = [c for c in parent_cols if c in p.columns]

    merged = b[bank_cols].merge(
        p[parent_cols],
        how="inner",
        left_on=["top_parent_rssd", "quarter_end"],
        right_on=["rssd_id", "quarter_end"],
        suffixes=("_bank", "_parent"),
    )
    return merged


def _family_quarter_summary(linked: pd.DataFrame) -> pd.DataFrame:
    """Aggregate linked panel to family-quarter level."""
    group_cols = ["top_parent_rssd", "quarter_end"]
    if "entity_name_parent" in linked.columns:
        group_cols.append("entity_name_parent")

    agg: dict[str, tuple[str, str]] = {}
    agg["n_banks"] = ("entity_id_bank", "nunique")
    for suffix in ("_bank", "_parent"):
        col = f"ust_share_assets{suffix}"
        if col in linked.columns:
            agg[f"mean_ust_share_assets{suffix}"] = (col, "mean")
        col = f"trading_assets_total_share_assets{suffix}"
        if col in linked.columns:
            agg[f"mean_trading_share{suffix}"] = (col, "mean")
        col = f"headroom_pp{suffix}"
        if col in linked.columns:
            agg[f"mean_headroom_pp{suffix}"] = (col, "mean")

    if "parent_method1_surcharge_parent" in linked.columns:
        agg["surcharge"] = ("parent_method1_surcharge_parent", "first")
    elif "parent_method1_surcharge_bank" in linked.columns:
        agg["surcharge"] = ("parent_method1_surcharge_bank", "first")

    summary = linked.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    return summary


def _coverage_manifest(bank: pd.DataFrame, parent: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    bank_frame = bank.copy()
    parent_frame = parent.copy()
    linked_frame = linked.copy()
    bank_frame["quarter_end"] = pd.to_datetime(bank_frame["quarter_end"])
    parent_frame["quarter_end"] = pd.to_datetime(parent_frame["quarter_end"])
    if not linked_frame.empty:
        linked_frame["quarter_end"] = pd.to_datetime(linked_frame["quarter_end"])

    bank_summary = (
        bank_frame.groupby(["entity_id", "entity_name", "top_parent_rssd"], dropna=False)
        .agg(
            bank_panel_observations=("quarter_end", "size"),
            bank_panel_first_quarter=("quarter_end", "min"),
            bank_panel_last_quarter=("quarter_end", "max"),
        )
        .reset_index()
    )
    parent_summary = (
        parent_frame.groupby("rssd_id", dropna=False)
        .agg(
            parent_panel_observations=("quarter_end", "size"),
            parent_panel_first_quarter=("quarter_end", "min"),
            parent_panel_last_quarter=("quarter_end", "max"),
        )
        .reset_index()
        .rename(columns={"rssd_id": "top_parent_rssd"})
    )

    if linked_frame.empty:
        linked_summary = pd.DataFrame(
            columns=[
                "entity_id",
                "linked_observations",
                "linked_first_quarter",
                "linked_last_quarter",
                "linked_parent_entity_id",
            ]
        )
    else:
        linked_summary = (
            linked_frame.groupby("entity_id_bank", dropna=False)
            .agg(
                linked_observations=("quarter_end", "size"),
                linked_first_quarter=("quarter_end", "min"),
                linked_last_quarter=("quarter_end", "max"),
                linked_parent_entity_id=("entity_id_parent", "first"),
            )
            .reset_index()
            .rename(columns={"entity_id_bank": "entity_id"})
        )

    manifest = bank_summary.merge(parent_summary, how="left", on="top_parent_rssd")
    manifest = manifest.merge(linked_summary, how="left", on="entity_id")
    manifest["included_linked_sample"] = manifest["linked_observations"].fillna(0).astype(int) > 0
    manifest["exclusion_reason"] = pd.NA
    manifest.loc[
        ~manifest["included_linked_sample"] & manifest["parent_panel_observations"].isna(),
        "exclusion_reason",
    ] = "parent_panel_missing"
    manifest.loc[
        ~manifest["included_linked_sample"] & manifest["exclusion_reason"].isna(),
        "exclusion_reason",
    ] = "no_overlapping_parent_quarters"

    for column in ["bank_panel_observations", "parent_panel_observations", "linked_observations"]:
        manifest[column] = manifest[column].astype("Int64")
    for column in [
        "bank_panel_first_quarter",
        "bank_panel_last_quarter",
        "parent_panel_first_quarter",
        "parent_panel_last_quarter",
        "linked_first_quarter",
        "linked_last_quarter",
    ]:
        manifest[column] = pd.to_datetime(manifest[column], errors="coerce").dt.strftime("%Y-%m-%d")

    return manifest.sort_values(["included_linked_sample", "entity_name"], ascending=[False, True]).reset_index(drop=True)


def _write_coverage_manifest_summary(manifest: pd.DataFrame, output_path: Path) -> None:
    included = manifest[manifest["included_linked_sample"]].copy()
    excluded = manifest[~manifest["included_linked_sample"]].copy()
    lines = [
        "# Parent Transmission Coverage Manifest",
        "",
        "## Linked sample",
        f"- Included bank subsidiaries: {len(included)}",
        f"- Excluded bank subsidiaries: {len(excluded)}",
    ]
    if not included.empty:
        lines.append(
            "- Included entities: " + "; ".join(included["entity_name"].dropna().astype(str).sort_values().tolist())
        )
    if excluded.empty:
        lines.append("- Excluded entities: none")
    else:
        for _, row in excluded.sort_values("entity_name").iterrows():
            lines.append(
                f"- Excluded: {row['entity_name']} (`{row['entity_id']}`) -> {row['exclusion_reason']}"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _directional_agreement(linked: pd.DataFrame, col_bank: str, col_parent: str) -> str:
    """Check whether quarter-over-quarter changes move in the same direction."""
    if col_bank not in linked.columns or col_parent not in linked.columns:
        return "not available (column missing)"

    frame = linked.sort_values(["top_parent_rssd", "entity_id_bank", "quarter_end"]).copy()
    frame["delta_bank"] = frame.groupby("entity_id_bank")[col_bank].diff()
    frame["delta_parent"] = frame.groupby(["top_parent_rssd"])[col_parent].diff()
    valid = frame.dropna(subset=["delta_bank", "delta_parent"])
    if valid.empty:
        return "not enough data to assess"
    same_sign = ((valid["delta_bank"] * valid["delta_parent"]) > 0).sum()
    total = len(valid)
    pct = same_sign / total * 100
    return f"{same_sign}/{total} quarter-over-quarter changes move in the same direction ({pct:.0f}%)"


def _surcharge_comparison(family_summary: pd.DataFrame) -> list[str]:
    """Compare high-surcharge vs low-surcharge families."""
    if "surcharge" not in family_summary.columns:
        return ["- Surcharge comparison: surcharge data not available in panels."]

    surcharge = pd.to_numeric(family_summary["surcharge"], errors="coerce")
    if surcharge.notna().sum() < 2:
        return ["- Surcharge comparison: not enough surcharge observations."]

    median_surcharge = surcharge.median()
    high = family_summary[surcharge > median_surcharge]
    low = family_summary[surcharge <= median_surcharge]
    if high.empty or low.empty:
        return ["- Surcharge comparison: could not split families into high/low groups."]

    lines = [
        f"- Surcharge split: median = {median_surcharge:.2f}; "
        f"{len(high)} high-surcharge obs, {len(low)} low-surcharge obs.",
    ]
    for col, label in [
        ("mean_ust_share_assets_bank", "bank UST/assets"),
        ("mean_headroom_pp_bank", "bank headroom pp"),
        ("mean_trading_share_parent", "parent trading/assets"),
    ]:
        if col in family_summary.columns:
            h_mean = high[col].mean(skipna=True)
            l_mean = low[col].mean(skipna=True)
            lines.append(
                f"  - {label}: high-surcharge avg = {h_mean:.4f}, "
                f"low-surcharge avg = {l_mean:.4f}"
            )
    return lines


def _write_summary(
    linked: pd.DataFrame,
    family_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write summary.md with plain-English findings."""
    n_obs = len(linked)
    quarters = sorted(linked["quarter_end"].dropna().unique())
    q_min = pd.Timestamp(quarters[0]).strftime("%Y-%m-%d") if quarters else "n/a"
    q_max = pd.Timestamp(quarters[-1]).strftime("%Y-%m-%d") if quarters else "n/a"
    n_quarters = len(quarters)
    n_families = linked["top_parent_rssd"].nunique()
    n_banks = linked["entity_id_bank"].nunique()

    lines = [
        "# Parent-vs-Bank Transmission Report",
        "",
        "## Coverage and Linkage",
        f"- Linked bank-parent quarter observations: {n_obs}",
        f"- Unique bank subsidiaries: {n_banks}",
        f"- Unique parent families: {n_families}",
        f"- Quarters covered: {n_quarters} ({q_min} to {q_max})",
        f"- Detailed coverage manifest: `{output_dir / 'coverage_manifest.csv'}`",
        "",
        "## Directional Co-movement",
    ]

    ust_agreement = _directional_agreement(
        linked, "ust_share_assets_bank", "ust_share_assets_parent"
    )
    lines.append(f"- UST holdings / assets: {ust_agreement}")

    trading_agreement = _directional_agreement(
        linked, "trading_assets_total_share_assets_bank", "trading_assets_total_share_assets_parent"
    )
    lines.append(f"- Trading assets / assets: {trading_agreement}")

    lines.extend(["", "## Surcharge / Headroom Comparison"])
    lines.extend(_surcharge_comparison(family_summary))

    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_parent_transmission_report(
    bank_panel_path: Path,
    parent_panel_path: Path,
    output_dir: Path,
) -> Path:
    """Build a parent-vs-bank transmission starter report.

    Returns the output directory path.
    """
    bank = read_table(bank_panel_path)
    parent = read_table(parent_panel_path)

    linked = _link_panels(bank, parent)
    family_summary = _family_quarter_summary(linked)
    coverage_manifest = _coverage_manifest(bank, parent, linked)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(linked, output_dir / "linked_panel.csv")
    write_frame(family_summary, output_dir / "family_quarter_summary.csv")
    write_frame(coverage_manifest, output_dir / "coverage_manifest.csv")
    _write_coverage_manifest_summary(coverage_manifest, output_dir / "coverage_manifest.md")
    _write_summary(linked, family_summary, output_dir)

    return output_dir
