from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analytics.headroom_panel import enrich_with_headroom
from .config import derived_data_path
from .pipeline import discover_tables, read_table, read_tables, validate_crosswalk_frame, write_frame


def _to_bool_series(series: pd.Series) -> pd.Series:
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
        "": False,
    }
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.map(mapping).fillna(False).astype(bool)


def _read_crosswalk(path: Path) -> pd.DataFrame:
    return validate_crosswalk_frame(read_table(path))


def build_crosswalk(universe_path: Path, output_path: Path | None = None) -> Path:
    frame = pd.read_csv(universe_path)
    normalized = validate_crosswalk_frame(frame)
    if "parent_method1_surcharge" in normalized.columns:
        normalized["parent_method1_surcharge"] = pd.to_numeric(
            normalized["parent_method1_surcharge"],
            errors="coerce",
        )

    destination = output_path or derived_data_path("crosswalk_v1.parquet")
    return write_frame(normalized, destination)


def _load_stage_frames(path: Path, suffix: str = ".parquet") -> pd.DataFrame:
    return read_tables(discover_tables(path, suffix=suffix))


def _add_treasury_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def series_or_zero(column: str) -> pd.Series:
        if column in out.columns:
            return pd.to_numeric(out[column], errors="coerce").fillna(0)
        return pd.Series(0, index=out.index, dtype="Float64")

    out["ust_inventory_fv"] = (
        series_or_zero("ust_htm_fair_value")
        + series_or_zero("ust_afs_fair_value")
        + series_or_zero("ust_trading_assets")
    )
    total_assets_raw = series_or_zero("total_assets")
    total_assets = total_assets_raw.where(total_assets_raw != 0)
    out["ust_share_assets"] = out["ust_inventory_fv"] / total_assets
    out["balances_due_from_fed_share_assets"] = series_or_zero("balances_due_from_fed") / total_assets
    out["repos_share_assets"] = series_or_zero("repos") / total_assets
    out["reverse_repos_share_assets"] = series_or_zero("reverse_repos") / total_assets
    out["trading_assets_total_share_assets"] = series_or_zero("trading_assets_total") / total_assets
    out["ust_share_headroom"] = out["ust_inventory_fv"] / series_or_zero("headroom_dollars").replace({0: pd.NA})
    return out


def _add_constraint_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def series_or_zero(column: str) -> pd.Series:
        if column in out.columns:
            return pd.to_numeric(out[column], errors="coerce").fillna(0)
        return pd.Series(0, index=out.index, dtype="Float64")

    total_assets_raw = series_or_zero("total_assets")
    total_assets = total_assets_raw.where(total_assets_raw != 0)
    tier1_capital_raw = series_or_zero("tier1_capital")
    tier1_capital = tier1_capital_raw.where(tier1_capital_raw != 0)
    ust_inventory = pd.to_numeric(out.get("ust_inventory_fv"), errors="coerce")

    htm_amortized = series_or_zero("ust_htm_amortized")
    htm_fair_value = series_or_zero("ust_htm_fair_value")
    afs_amortized = series_or_zero("ust_afs_amortized")
    afs_fair_value = series_or_zero("ust_afs_fair_value")
    balances_due_from_fed = series_or_zero("balances_due_from_fed")
    reverse_repos = series_or_zero("reverse_repos")
    deposits = series_or_zero("deposits")
    loans = series_or_zero("loans")

    out["htm_unrealized_loss"] = (htm_amortized - htm_fair_value).clip(lower=0)
    out["afs_unrealized_loss"] = (afs_amortized - afs_fair_value).clip(lower=0)
    out["total_unrealized_loss"] = out["htm_unrealized_loss"] + out["afs_unrealized_loss"]
    out["total_unrealized_loss_share_assets"] = out["total_unrealized_loss"] / total_assets
    out["total_unrealized_loss_tier1"] = out["total_unrealized_loss"] / tier1_capital
    out["liquid_asset_share_assets"] = (balances_due_from_fed + reverse_repos) / total_assets
    out["deposit_share_assets"] = deposits / total_assets
    out["loan_share_assets"] = loans / total_assets
    out["loan_to_deposit_ratio"] = loans / deposits.replace({0: pd.NA})
    out["deposit_funding_gap_share_assets"] = (loans - deposits).clip(lower=0) / total_assets
    out["non_deposit_funding_share_assets"] = (total_assets_raw - deposits - tier1_capital_raw).clip(lower=0) / total_assets
    out["safe_asset_buffer_share_assets"] = (ust_inventory.fillna(0) + balances_due_from_fed + reverse_repos) / total_assets
    out["liquid_asset_to_deposits"] = (balances_due_from_fed + reverse_repos) / deposits.replace({0: pd.NA})
    out["safe_asset_buffer_to_deposits"] = (ust_inventory.fillna(0) + balances_due_from_fed + reverse_repos) / deposits.replace({0: pd.NA})
    out["htm_share_ust"] = htm_fair_value / ust_inventory.replace({0: pd.NA})
    out["afs_share_ust"] = afs_fair_value / ust_inventory.replace({0: pd.NA})
    return out


def _add_panel_dynamics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty or "quarter_end" not in out.columns or "entity_id" not in out.columns:
        return out
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    out = out.sort_values(["entity_id", "quarter_end"]).reset_index(drop=True)

    if "deposits" in out.columns:
        out["deposit_growth_qoq"] = out.groupby("entity_id")["deposits"].pct_change()
    if "loans" in out.columns:
        out["loan_growth_qoq"] = out.groupby("entity_id")["loans"].pct_change()
    if "ust_share_assets" in out.columns:
        out["ust_share_assets_qoq"] = out.groupby("entity_id")["ust_share_assets"].diff()
    return out


def _prepare_panel(
    staged: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    entity_type: str,
) -> pd.DataFrame:
    subset = crosswalk[crosswalk["entity_type"] == entity_type].copy()
    merged = staged.merge(
        subset,
        how="inner",
        on="rssd_id",
        suffixes=("", "_crosswalk"),
    )
    merged["entity_type"] = entity_type
    merged["slr_applies"] = True
    merged["is_gsib_parent"] = _to_bool_series(merged["is_gsib_parent"])
    merged["is_covered_bank_subsidiary"] = _to_bool_series(merged["is_covered_bank_subsidiary"])
    if "parent_method1_surcharge" in merged.columns:
        merged["parent_method1_surcharge"] = pd.to_numeric(merged["parent_method1_surcharge"], errors="coerce")
    else:
        merged["parent_method1_surcharge"] = pd.NA
    merged["tier1_capital"] = pd.to_numeric(merged["tier1_capital"], errors="coerce")
    merged["total_leverage_exposure"] = pd.to_numeric(merged["total_leverage_exposure"], errors="coerce")
    merged = merged.dropna(subset=["tier1_capital", "total_leverage_exposure"]).copy()
    merged = merged[
        (merged["tier1_capital"] > 0) & (merged["total_leverage_exposure"] > 0)
    ].copy()
    return merged


def build_insured_bank_panel(
    staged_path: Path,
    crosswalk_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    staged = _load_stage_frames(staged_path)
    crosswalk = _read_crosswalk(crosswalk_path)
    prepared = _prepare_panel(staged, crosswalk, entity_type="insured_bank_sub")
    enriched = enrich_with_headroom(prepared)
    enriched = _add_treasury_metrics(enriched)
    enriched = _add_constraint_metrics(enriched)
    enriched = _add_panel_dynamics(enriched)
    destination = output_path or derived_data_path("insured_bank_panel.parquet")
    return write_frame(enriched, destination)


def _merge_fry15_overlay(frame: pd.DataFrame, fry15_path: Path | None) -> pd.DataFrame:
    if fry15_path is None:
        return frame
    overlay = read_table(fry15_path)
    required = {"fr_y15_reporter", "parent_method1_surcharge"}
    missing = required - set(overlay.columns)
    if missing:
        raise ValueError(f"FR Y-15 overlay is missing required columns: {sorted(missing)}")

    columns = ["fr_y15_reporter", "parent_method1_surcharge"]
    if "quarter_end" not in overlay.columns:
        merged = frame.merge(
            overlay[columns],
            how="left",
            on="fr_y15_reporter",
            suffixes=("", "_fry15"),
        )
    else:
        left = frame.copy()
        right = overlay[columns + ["quarter_end"]].copy()
        left["fr_y15_reporter"] = left["fr_y15_reporter"].astype(str)
        right["fr_y15_reporter"] = right["fr_y15_reporter"].astype(str)
        left["quarter_end_dt"] = pd.to_datetime(left["quarter_end"])
        right["quarter_end_dt"] = pd.to_datetime(right["quarter_end"])
        pieces: list[pd.DataFrame] = []
        overlay_groups = {
            reporter: group.sort_values("quarter_end_dt")
            for reporter, group in right.groupby("fr_y15_reporter", sort=False, dropna=False)
        }
        for reporter, left_group in left.groupby("fr_y15_reporter", sort=False, dropna=False):
            ordered_left = left_group.sort_values("quarter_end_dt")
            right_group = overlay_groups.get(reporter)
            if right_group is None:
                ordered_left["parent_method1_surcharge_fry15"] = pd.NA
                pieces.append(ordered_left)
                continue
            merged_group = pd.merge_asof(
                ordered_left,
                right_group.drop(columns=["quarter_end", "fr_y15_reporter"]),
                on="quarter_end_dt",
                direction="backward",
                suffixes=("", "_fry15"),
            )
            pieces.append(merged_group)
        merged = pd.concat(pieces, ignore_index=True).drop(columns=["quarter_end_dt"])

    if "parent_method1_surcharge_fry15" in merged.columns:
        merged["parent_method1_surcharge"] = merged["parent_method1_surcharge_fry15"].combine_first(
            merged["parent_method1_surcharge"]
        )
        merged = merged.drop(columns=["parent_method1_surcharge_fry15"])
    return merged


def build_parent_panel(
    staged_path: Path,
    crosswalk_path: Path,
    *,
    fry15_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    staged = _load_stage_frames(staged_path)
    crosswalk = _read_crosswalk(crosswalk_path)
    parent_crosswalk = crosswalk[crosswalk["entity_type"].isin(["bhc_parent", "ihc_fbo_us"])].copy()
    prepared = staged.merge(parent_crosswalk, how="inner", on="rssd_id", suffixes=("", "_crosswalk"))
    prepared["slr_applies"] = True
    prepared["is_gsib_parent"] = _to_bool_series(prepared["is_gsib_parent"])
    prepared["is_covered_bank_subsidiary"] = False
    if "parent_method1_surcharge" in prepared.columns:
        prepared["parent_method1_surcharge"] = pd.to_numeric(prepared["parent_method1_surcharge"], errors="coerce")
    prepared["tier1_capital"] = pd.to_numeric(prepared["tier1_capital"], errors="coerce")
    prepared["total_leverage_exposure"] = pd.to_numeric(prepared["total_leverage_exposure"], errors="coerce")
    prepared = prepared.dropna(subset=["tier1_capital", "total_leverage_exposure"]).copy()
    prepared = prepared[
        (prepared["tier1_capital"] > 0) & (prepared["total_leverage_exposure"] > 0)
    ].copy()
    prepared = _merge_fry15_overlay(prepared, fry15_path)
    enriched = enrich_with_headroom(prepared)
    enriched = _add_treasury_metrics(enriched)
    enriched = _add_constraint_metrics(enriched)
    enriched = _add_panel_dynamics(enriched)
    destination = output_path or derived_data_path("parent_panel.parquet")
    return write_frame(enriched, destination)
