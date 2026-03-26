from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analytics.headroom_panel import enrich_with_headroom
from .config import derived_data_path, reference_data_path
from .ingest.fdic_institutions import build_fdic_institutions_reference
from .pipeline import read_table, write_frame
from .quarters import QuarterRef
from .panels import _add_constraint_metrics, _add_panel_dynamics, _add_treasury_metrics, _to_bool_series


EVENT_WINDOW_START = pd.Timestamp("2019-03-31")
EVENT_WINDOW_END = pd.Timestamp("2021-12-31")
BASELINE_QUARTER = pd.Timestamp("2019-12-31")
EVENT_WINDOW_QUARTERS = pd.date_range(EVENT_WINDOW_START, EVENT_WINDOW_END, freq="QE-DEC")
PRE_TREATMENT_END = pd.Timestamp("2020-03-31")
POST_TREATMENT_START = pd.Timestamp("2020-06-30")
MIN_PRE_QUARTERS_EXPANDED = 3
MIN_POST_QUARTERS_EXPANDED = 5

MERGED_IDENTITY_COLUMNS = {
    "RSSD9001": "rssd_id",
    "RSSD9017": "entity_name_raw",
    "RSSD9050": "fdic_cert_raw",
    "RSSD9130": "hq_city",
    "RSSD9200": "hq_state",
    "RSSD9220": "hq_zip",
}

OVERRIDE_COLUMNS = [
    "rssd_id",
    "entity_id",
    "entity_name",
    "fdic_cert",
    "top_parent_rssd",
    "country",
    "is_covered_bank_subsidiary",
    "notes",
]

TREATMENT_MAP_COLUMNS = [
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
]

TREATMENT_MAP_REQUIRED_COLUMNS = [
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
]


def _quarter_dirs(staged_path: Path) -> list[Path]:
    if staged_path.is_file():
        raise ValueError("Expected a staged Call Report root directory, not a single file")
    return sorted(
        path for path in staged_path.iterdir()
        if path.is_dir() and (path / "call_reports_normalized.parquet").exists() and (path / "call_reports_merged.parquet").exists()
    )


def _read_identity_frame(quarter_dir: Path) -> pd.DataFrame:
    merged = pd.read_parquet(quarter_dir / "call_reports_merged.parquet", columns=list(MERGED_IDENTITY_COLUMNS))
    identity = merged.rename(columns=MERGED_IDENTITY_COLUMNS).copy()
    identity["rssd_id"] = identity["rssd_id"].astype("string").str.strip()
    for column in ["entity_name_raw", "fdic_cert_raw", "hq_city", "hq_state", "hq_zip"]:
        identity[column] = identity[column].astype("string").str.strip()
    return identity.drop_duplicates("rssd_id")


def _read_normalized_frame(quarter_dir: Path) -> pd.DataFrame:
    frame = pd.read_parquet(quarter_dir / "call_reports_normalized.parquet").copy()
    frame["rssd_id"] = frame["rssd_id"].astype("string").str.strip()
    frame["quarter_end"] = pd.to_datetime(frame["quarter_end"])
    return frame


def _load_seed_overlay(
    crosswalk_path: Path | None,
    overrides_path: Path | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if crosswalk_path and crosswalk_path.exists():
        crosswalk = read_table(crosswalk_path).copy()
        insured = crosswalk[crosswalk["entity_type"].astype("string") == "insured_bank_sub"].copy()
        if not insured.empty:
            insured["rssd_id"] = insured["rssd_id"].astype("string").str.strip()
            insured["seed_entity_id"] = insured["entity_id"].astype("string").str.strip()
            insured["seed_entity_name"] = insured["entity_name"].astype("string").str.strip()
            insured["seed_fdic_cert"] = insured["fdic_cert"].astype("string").str.strip()
            insured["seed_top_parent_rssd"] = insured["top_parent_rssd"].astype("string").str.strip()
            insured["seed_country"] = insured["country"].astype("string").str.strip()
            insured["seed_is_covered_bank_subsidiary"] = _to_bool_series(insured["is_covered_bank_subsidiary"])
            frames.append(
                insured[
                    [
                        "rssd_id",
                        "seed_entity_id",
                        "seed_entity_name",
                        "seed_fdic_cert",
                        "seed_top_parent_rssd",
                        "seed_country",
                        "seed_is_covered_bank_subsidiary",
                    ]
                ]
            )

    if overrides_path and overrides_path.exists():
        override = pd.read_csv(overrides_path).copy()
        if not override.empty:
            for column in OVERRIDE_COLUMNS:
                if column not in override.columns:
                    override[column] = pd.NA
            override["rssd_id"] = override["rssd_id"].astype("string").str.strip()
            override["override_entity_id"] = override["entity_id"].astype("string").str.strip()
            override["override_entity_name"] = override["entity_name"].astype("string").str.strip()
            override["override_fdic_cert"] = override["fdic_cert"].astype("string").str.strip()
            override["override_top_parent_rssd"] = override["top_parent_rssd"].astype("string").str.strip()
            override["override_country"] = override["country"].astype("string").str.strip()
            override["override_is_covered_bank_subsidiary"] = _to_bool_series(override["is_covered_bank_subsidiary"])
            override["override_notes"] = override["notes"].astype("string").str.strip()
            frames.append(
                override[
                    [
                        "rssd_id",
                        "override_entity_id",
                        "override_entity_name",
                        "override_fdic_cert",
                        "override_top_parent_rssd",
                        "override_country",
                        "override_is_covered_bank_subsidiary",
                        "override_notes",
                    ]
                ]
            )

    if not frames:
        return pd.DataFrame(columns=["rssd_id"])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, how="outer", on="rssd_id")
    return out


def _load_gsib_parent_ids(crosswalk_path: Path | None) -> set[str]:
    if crosswalk_path is None or not crosswalk_path.exists():
        return set()
    crosswalk = read_table(crosswalk_path).copy()
    if crosswalk.empty:
        return set()
    parents = crosswalk[
        crosswalk["entity_type"].astype("string").isin(["bhc_parent"])
        & _to_bool_series(crosswalk["is_gsib_parent"])
    ].copy()
    return set(parents["rssd_id"].astype("string").str.strip().replace({"": pd.NA}).dropna().astype(str))


def _load_parent_name_lookup(crosswalk_path: Path | None) -> dict[str, str]:
    if crosswalk_path is None or not crosswalk_path.exists():
        return {}
    crosswalk = read_table(crosswalk_path).copy()
    if crosswalk.empty:
        return {}
    parents = crosswalk[crosswalk["entity_type"].astype("string").isin(["bhc_parent", "ihc_fbo_us"])].copy()
    parents["rssd_id"] = parents["rssd_id"].astype("string").str.strip()
    parents["entity_name"] = parents["entity_name"].astype("string").str.strip()
    parents = parents.dropna(subset=["rssd_id", "entity_name"])
    return {
        str(row["rssd_id"]): str(row["entity_name"])
        for _, row in parents.iterrows()
        if str(row["rssd_id"]).strip()
    }


def _load_fdic_overlay(fdic_metadata_path: Path | None) -> pd.DataFrame:
    if fdic_metadata_path is None:
        return pd.DataFrame(columns=["rssd_id"])
    if not fdic_metadata_path.exists():
        build_fdic_institutions_reference(output_path=fdic_metadata_path)
    overlay = pd.read_csv(fdic_metadata_path).copy()
    if overlay.empty:
        return overlay
    overlay["rssd_id"] = overlay["rssd_id"].astype("string").str.strip()
    for column in ["fdic_cert", "fdic_top_parent_rssd"]:
        if column in overlay.columns:
            overlay[column] = overlay[column].astype("string").str.strip()
    for column in ["fdic_entity_name", "fdic_top_parent_name", "fdic_city", "fdic_state", "fdic_regulator", "fdic_bank_class"]:
        if column in overlay.columns:
            overlay[column] = overlay[column].astype("string").str.strip()
    return overlay.drop_duplicates("rssd_id", keep="first")


def _to_nullable_bool_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    mapped = values.map(
        {
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
        }
    )
    mapped[values.isna() | values.eq("")] = pd.NA
    return mapped.astype("boolean")


def _load_treatment_map(treatment_map_path: Path | None) -> pd.DataFrame:
    path = treatment_map_path or reference_data_path("insured_bank_treatment_map_2020.csv")
    if path is None or not path.exists():
        return pd.DataFrame(columns=["rssd_id", *TREATMENT_MAP_COLUMNS])

    treatment_map = pd.read_csv(path).copy()
    if treatment_map.empty:
        return pd.DataFrame(columns=["rssd_id", *TREATMENT_MAP_COLUMNS])

    missing = [column for column in TREATMENT_MAP_REQUIRED_COLUMNS if column not in treatment_map.columns]
    if missing:
        raise ValueError(f"Treatment map {path} is missing required columns: {', '.join(missing)}")

    treatment_map["rssd_id"] = treatment_map["rssd_id"].astype("string").str.strip()
    treatment_map = treatment_map[treatment_map["rssd_id"].notna() & (treatment_map["rssd_id"] != "")].copy()
    treatment_map = treatment_map.drop_duplicates("rssd_id", keep="last")

    treatment_map = treatment_map.rename(
        columns={
            "entity_id": "map_entity_id",
            "entity_name": "map_entity_name",
            "fdic_cert": "map_fdic_cert",
        }
    )
    for column in [
        "map_entity_id",
        "map_entity_name",
        "map_fdic_cert",
        "top_parent_rssd_2019q4",
        "top_parent_name_2019q4",
        "treatment_scope_2020",
        "classification_source",
        "provenance_notes",
    ]:
        treatment_map[column] = treatment_map[column].astype("string").str.strip()
    for column in [
        "slr_reporting_2019q4",
        "eslr_covered_6pct",
        "di_relief_eligible_2020",
        "di_relief_elected_2020",
        "parent_hc_relief_scope_2020",
    ]:
        treatment_map[column] = _to_nullable_bool_series(treatment_map[column])

    treatment_map = treatment_map[
        [
            "rssd_id",
            "map_entity_id",
            "map_entity_name",
            "map_fdic_cert",
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
        ]
    ].copy()
    return treatment_map


def _merge_stage_frames(
    staged_path: Path,
    *,
    crosswalk_path: Path | None = None,
    overrides_path: Path | None = None,
    fdic_metadata_path: Path | None = None,
    treatment_map_path: Path | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seed_overlay = _load_seed_overlay(crosswalk_path, overrides_path)
    fdic_overlay = _load_fdic_overlay(fdic_metadata_path)
    treatment_map = _load_treatment_map(treatment_map_path)
    gsib_parent_ids = _load_gsib_parent_ids(crosswalk_path)
    parent_name_lookup = _load_parent_name_lookup(crosswalk_path)

    for quarter_dir in _quarter_dirs(staged_path):
        normalized = _read_normalized_frame(quarter_dir)
        identity = _read_identity_frame(quarter_dir)
        merged = normalized.merge(identity, how="left", on="rssd_id")
        frames.append(merged)

    if not frames:
        raise FileNotFoundError(f"No staged Call Report quarter pairs found under {staged_path}")

    out = pd.concat(frames, ignore_index=True)
    if not fdic_overlay.empty:
        out = out.merge(fdic_overlay, how="left", on="rssd_id")
    if not seed_overlay.empty:
        out = out.merge(seed_overlay, how="left", on="rssd_id")
    if not treatment_map.empty:
        out = out.merge(treatment_map, how="left", on="rssd_id")

    out["entity_type"] = "insured_bank_sub"
    out["classification_source"] = out.get(
        "classification_source",
        pd.Series(pd.NA, index=out.index, dtype="string"),
    ).astype("string").str.strip()
    out["provenance_notes"] = out.get(
        "provenance_notes",
        pd.Series(pd.NA, index=out.index, dtype="string"),
    ).astype("string").str.strip()
    out["entity_id"] = out.get("override_entity_id", pd.Series(pd.NA, index=out.index, dtype="string")).combine_first(
        out.get("map_entity_id", pd.Series(pd.NA, index=out.index, dtype="string"))
    ).combine_first(
        out.get("seed_entity_id", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    out["entity_id"] = out["entity_id"].astype("string").str.strip()
    fallback_entity_id = "insured_bank_rssd_" + out["rssd_id"].astype("string")
    out["entity_id"] = out["entity_id"].where(out["entity_id"].notna() & (out["entity_id"] != ""), fallback_entity_id)

    out["entity_name"] = out.get("override_entity_name", pd.Series(pd.NA, index=out.index, dtype="string")).combine_first(
        out.get("map_entity_name", pd.Series(pd.NA, index=out.index, dtype="string"))
    ).combine_first(
        out.get("seed_entity_name", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    out["entity_name"] = out["entity_name"].combine_first(out.get("fdic_entity_name", pd.Series(pd.NA, index=out.index, dtype="string")))
    out["entity_name"] = out["entity_name"].combine_first(out["entity_name_raw"].astype("string").str.strip())

    out["fdic_cert"] = out.get("override_fdic_cert", pd.Series(pd.NA, index=out.index, dtype="string")).combine_first(
        out.get("map_fdic_cert", pd.Series(pd.NA, index=out.index, dtype="string"))
    ).combine_first(
        out.get("seed_fdic_cert", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    out["fdic_cert"] = out["fdic_cert"].combine_first(out["fdic_cert_raw"].astype("string").str.strip())

    out["top_parent_rssd"] = out.get(
        "override_top_parent_rssd", pd.Series(pd.NA, index=out.index, dtype="string")
    ).combine_first(
        out.get("top_parent_rssd_2019q4", pd.Series(pd.NA, index=out.index, dtype="string"))
    ).combine_first(
        out.get("seed_top_parent_rssd", pd.Series(pd.NA, index=out.index, dtype="string"))
    ).combine_first(
        out.get("fdic_top_parent_rssd", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    out["country"] = out.get("override_country", pd.Series(pd.NA, index=out.index, dtype="string")).combine_first(
        out.get("seed_country", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    out["country"] = out["country"].where(out["country"].notna() & (out["country"] != ""), "United States")

    covered_override = out.get(
        "override_is_covered_bank_subsidiary",
        pd.Series(False, index=out.index, dtype=bool),
    )
    covered_seed = out.get(
        "seed_is_covered_bank_subsidiary",
        pd.Series(False, index=out.index, dtype=bool),
    )
    covered_map = out.get("di_relief_eligible_2020", pd.Series(pd.NA, index=out.index, dtype="boolean"))
    if str(covered_map.dtype) != "boolean":
        covered_map = _to_nullable_bool_series(covered_map.astype("string"))
    covered_gsib_parent = out["top_parent_rssd"].astype("string").isin(gsib_parent_ids)
    legacy_covered = covered_override.fillna(False) | covered_seed.fillna(False) | covered_gsib_parent.fillna(False)
    out["is_covered_bank_subsidiary"] = covered_map.fillna(legacy_covered)
    out["top_parent_rssd_2019q4"] = out.get(
        "top_parent_rssd_2019q4",
        pd.Series(pd.NA, index=out.index, dtype="string"),
    ).combine_first(out["top_parent_rssd"].astype("string"))
    out["top_parent_rssd_2019q4"] = out["top_parent_rssd_2019q4"].astype("string").str.strip()
    out["top_parent_name_2019q4"] = out.get(
        "top_parent_name_2019q4",
        pd.Series(pd.NA, index=out.index, dtype="string"),
    ).combine_first(out.get("top_parent_name", pd.Series(pd.NA, index=out.index, dtype="string")))
    out["top_parent_name_2019q4"] = out["top_parent_name_2019q4"].astype("string").str.strip()
    out["slr_reporting_2019q4"] = out.get(
        "slr_reporting_2019q4",
        pd.Series(pd.NA, index=out.index, dtype="boolean"),
    )
    if str(out["slr_reporting_2019q4"].dtype) != "boolean":
        out["slr_reporting_2019q4"] = _to_nullable_bool_series(out["slr_reporting_2019q4"].astype("string"))
    out["eslr_covered_6pct"] = out.get(
        "eslr_covered_6pct",
        pd.Series(pd.NA, index=out.index, dtype="boolean"),
    )
    if str(out["eslr_covered_6pct"].dtype) != "boolean":
        out["eslr_covered_6pct"] = _to_nullable_bool_series(out["eslr_covered_6pct"].astype("string"))
    out["di_relief_eligible_2020"] = out.get(
        "di_relief_eligible_2020",
        pd.Series(pd.NA, index=out.index, dtype="boolean"),
    )
    if str(out["di_relief_eligible_2020"].dtype) != "boolean":
        out["di_relief_eligible_2020"] = _to_nullable_bool_series(out["di_relief_eligible_2020"].astype("string"))
    out["di_relief_elected_2020"] = out.get(
        "di_relief_elected_2020",
        pd.Series(pd.NA, index=out.index, dtype="boolean"),
    )
    if str(out["di_relief_elected_2020"].dtype) != "boolean":
        out["di_relief_elected_2020"] = _to_nullable_bool_series(out["di_relief_elected_2020"].astype("string"))
    out["parent_hc_relief_scope_2020"] = out.get(
        "parent_hc_relief_scope_2020",
        pd.Series(pd.NA, index=out.index, dtype="boolean"),
    )
    if str(out["parent_hc_relief_scope_2020"].dtype) != "boolean":
        out["parent_hc_relief_scope_2020"] = _to_nullable_bool_series(out["parent_hc_relief_scope_2020"].astype("string"))
    out["treatment_scope_2020"] = out.get(
        "treatment_scope_2020",
        pd.Series(pd.NA, index=out.index, dtype="string"),
    ).astype("string").str.strip()
    out["classification_source"] = out["classification_source"].fillna(
        pd.Series(pd.NA, index=out.index, dtype="string")
    )
    out["provenance_notes"] = out["provenance_notes"].fillna(
        pd.Series(pd.NA, index=out.index, dtype="string")
    )
    seed_entity_id = out.get("seed_entity_id", pd.Series(pd.NA, index=out.index, dtype="string"))
    fdic_entity_name = out.get("fdic_entity_name", pd.Series(pd.NA, index=out.index, dtype="string"))
    out.loc[out["classification_source"].isna() & out["map_entity_id"].notna(), "classification_source"] = "treatment_map_2020"
    out.loc[out["classification_source"].isna() & seed_entity_id.notna(), "classification_source"] = "crosswalk_fallback"
    out.loc[out["classification_source"].isna() & seed_entity_id.isna() & fdic_entity_name.notna(), "classification_source"] = "fdic_fallback"
    out.loc[out["classification_source"].isna(), "classification_source"] = "derived_default"
    out.loc[out["provenance_notes"].isna() & out["map_entity_id"].notna(), "provenance_notes"] = "Treatment map row supplied the authoritative 2020 classification."
    out.loc[out["provenance_notes"].isna() & seed_entity_id.notna(), "provenance_notes"] = "Fallback classification derived from curated crosswalk."
    out.loc[out["provenance_notes"].isna() & seed_entity_id.isna() & fdic_entity_name.notna(), "provenance_notes"] = "Fallback classification derived from FDIC overlay."
    out.loc[out["provenance_notes"].isna(), "provenance_notes"] = "Derived default classification."
    out["parent_hc_relief_scope_2020"] = out["parent_hc_relief_scope_2020"].fillna(
        out["top_parent_rssd"].astype("string").isin(gsib_parent_ids)
    )
    out["treatment_scope_2020"] = out["treatment_scope_2020"].fillna(
        pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )
    )
    out.loc[out["treatment_scope_2020"].isna() & out["di_relief_eligible_2020"].fillna(False), "treatment_scope_2020"] = "direct_bank_relief_scope"
    out.loc[
        out["treatment_scope_2020"].isna() & ~out["di_relief_eligible_2020"].fillna(False) & out["parent_hc_relief_scope_2020"].fillna(False),
        "treatment_scope_2020",
    ] = "parent_hc_relief_scope"
    out.loc[out["treatment_scope_2020"].isna(), "treatment_scope_2020"] = "not_in_relief_scope"
    out["is_gsib_parent"] = False
    out["fr_y15_reporter"] = pd.NA
    out["parent_method1_surcharge"] = pd.NA
    out["is_legacy_seed_entity"] = out.get("seed_entity_id", pd.Series(pd.NA, index=out.index, dtype="string")).notna()
    out["override_notes"] = out.get("override_notes", pd.Series(pd.NA, index=out.index, dtype="string"))
    out["top_parent_name"] = out["top_parent_rssd"].astype("string").map(parent_name_lookup)
    out["top_parent_name"] = out["top_parent_name"].combine_first(
        out.get("fdic_top_parent_name", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    out["fdic_active"] = out.get("fdic_active", pd.Series(pd.NA, index=out.index, dtype="Int64"))
    out["fdic_bank_class"] = out.get("fdic_bank_class", pd.Series(pd.NA, index=out.index, dtype="string"))
    out["fdic_regulator"] = out.get("fdic_regulator", pd.Series(pd.NA, index=out.index, dtype="string"))
    return out


def _slr_reporting_mask(frame: pd.DataFrame) -> pd.Series:
    tier1 = pd.to_numeric(frame.get("tier1_capital"), errors="coerce")
    tle = pd.to_numeric(frame.get("total_leverage_exposure"), errors="coerce")
    return tier1.gt(0) & tle.gt(0)


def _add_scope_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    reporting = _slr_reporting_mask(out)
    out["slr_reporting_observation"] = reporting
    out["slr_applies"] = reporting
    out["slr_scope_class"] = "non_slr_reporting_insured_bank"
    out.loc[reporting, "slr_scope_class"] = "slr_reporting_insured_bank"
    out.loc[out["is_covered_bank_subsidiary"].fillna(False), "slr_scope_class"] = "covered_bank_subsidiary"
    return out


def _add_headroom_where_available(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    eligible = _slr_reporting_mask(out)
    if not eligible.any():
        for column in [
            "rule_regime",
            "required_slr",
            "rule_notes",
            "computed_actual_slr",
            "headroom_pp",
            "headroom_dollars",
        ]:
            out[column] = pd.NA
        return out

    enriched = enrich_with_headroom(out.loc[eligible].copy())
    add_columns = [
        "rule_regime",
        "required_slr",
        "rule_notes",
        "computed_actual_slr",
        "headroom_pp",
        "headroom_dollars",
    ]
    for column in add_columns:
        out[column] = pd.NA
        out.loc[eligible, column] = enriched[column].values
    return out


def _baseline_entity_flags(frame: pd.DataFrame) -> pd.DataFrame:
    event = frame[(frame["quarter_end"] >= EVENT_WINDOW_START) & (frame["quarter_end"] <= EVENT_WINDOW_END)].copy()
    if event.empty:
        return pd.DataFrame(columns=["entity_id"])

    baseline = event[event["quarter_end"] == BASELINE_QUARTER].copy()
    baseline["slr_reporting_2019q4"] = (
        pd.to_numeric(baseline.get("tier1_capital"), errors="coerce").gt(0)
        & pd.to_numeric(baseline.get("total_leverage_exposure"), errors="coerce").gt(0)
    )
    baseline["has_usable_2019q4_low_headroom_baseline"] = (
        baseline["slr_reporting_2019q4"]
        & pd.to_numeric(baseline.get("headroom_pp"), errors="coerce").notna()
    )
    baseline["has_usable_2019q4_high_ust_share_baseline"] = (
        baseline["slr_reporting_2019q4"]
        & pd.to_numeric(baseline.get("ust_share_assets"), errors="coerce").notna()
    )
    di_eligible = baseline.get("di_relief_eligible_2020", pd.Series(pd.NA, index=baseline.index, dtype="boolean"))
    if str(di_eligible.dtype) != "boolean":
        di_eligible = _to_nullable_bool_series(di_eligible.astype("string"))
    baseline["has_usable_2019q4_covered_bank_baseline"] = (
        baseline["slr_reporting_2019q4"]
        & di_eligible.fillna(False)
    )
    baseline["has_usable_2019q4_baseline"] = (
        baseline["has_usable_2019q4_low_headroom_baseline"]
        & baseline["has_usable_2019q4_high_ust_share_baseline"]
    )
    return baseline[
        [
            "entity_id",
            "slr_reporting_2019q4",
            "has_usable_2019q4_baseline",
            "has_usable_2019q4_low_headroom_baseline",
            "has_usable_2019q4_high_ust_share_baseline",
            "has_usable_2019q4_covered_bank_baseline",
        ]
    ].drop_duplicates("entity_id")


def _event_window_panel_flags(frame: pd.DataFrame) -> pd.DataFrame:
    event = frame[(frame["quarter_end"] >= EVENT_WINDOW_START) & (frame["quarter_end"] <= EVENT_WINDOW_END)].copy()
    if event.empty:
        return pd.DataFrame(columns=["entity_id"])

    event["is_pre"] = event["quarter_end"] <= PRE_TREATMENT_END
    event["is_post"] = event["quarter_end"] >= POST_TREATMENT_START
    summary = (
        event.groupby("entity_id", dropna=False)
        .agg(
            event_window_observations=("quarter_end", "size"),
            event_window_first_quarter=("quarter_end", "min"),
            event_window_last_quarter=("quarter_end", "max"),
            pre_treatment_quarters=("is_pre", "sum"),
            post_treatment_quarters=("is_post", "sum"),
            in_universe_b=("slr_reporting_observation", "any"),
        )
        .reset_index()
    )
    summary["has_full_event_window_coverage"] = summary["event_window_observations"].eq(len(EVENT_WINDOW_QUARTERS))
    summary["has_expanded_event_window_coverage"] = (
        summary["pre_treatment_quarters"].ge(MIN_PRE_QUARTERS_EXPANDED)
        & summary["post_treatment_quarters"].ge(MIN_POST_QUARTERS_EXPANDED)
    )
    return summary


def build_insured_bank_universe(frame: pd.DataFrame) -> pd.DataFrame:
    latest = frame.sort_values(["rssd_id", "quarter_end"]).drop_duplicates("rssd_id", keep="last").copy()
    summary = (
        frame.groupby(["entity_id", "rssd_id"], dropna=False)
        .agg(
            has_call_report_data=("quarter_end", "size"),
            has_tier1_capital=("tier1_capital", lambda s: pd.to_numeric(s, errors="coerce").notna().any()),
            has_total_leverage_exposure=("total_leverage_exposure", lambda s: pd.to_numeric(s, errors="coerce").gt(0).any()),
            first_quarter=("quarter_end", "min"),
            last_quarter=("quarter_end", "max"),
            slr_reporting_bank=("slr_reporting_observation", "any"),
        )
        .reset_index()
    )
    baseline_flags = _baseline_entity_flags(frame)
    event_flags = _event_window_panel_flags(frame)

    overlapping_baseline_columns = [column for column in baseline_flags.columns if column != "entity_id" and column in latest.columns]
    if overlapping_baseline_columns:
        baseline_flags = baseline_flags.drop(columns=overlapping_baseline_columns)

    universe = latest.merge(summary, how="left", on=["entity_id", "rssd_id"])
    universe = universe.merge(baseline_flags, how="left", on="entity_id")
    universe = universe.merge(event_flags, how="left", on="entity_id")

    universe["has_call_report_data"] = universe["has_call_report_data"].fillna(0).astype(int).gt(0)
    universe["has_tier1_capital"] = universe["has_tier1_capital"].fillna(False).astype(bool)
    universe["has_total_leverage_exposure"] = universe["has_total_leverage_exposure"].fillna(False).astype(bool)
    universe["has_usable_2019q4_baseline"] = universe["has_usable_2019q4_baseline"].fillna(False).astype(bool)
    universe["has_usable_2019q4_low_headroom_baseline"] = universe["has_usable_2019q4_low_headroom_baseline"].fillna(False).astype(bool)
    universe["has_usable_2019q4_high_ust_share_baseline"] = universe["has_usable_2019q4_high_ust_share_baseline"].fillna(False).astype(bool)
    universe["has_usable_2019q4_covered_bank_baseline"] = universe["has_usable_2019q4_covered_bank_baseline"].fillna(False).astype(bool)
    universe["slr_reporting_2019q4"] = universe["slr_reporting_2019q4"].fillna(False).astype(bool)
    universe["has_full_event_window_coverage"] = universe["has_full_event_window_coverage"].fillna(False).astype(bool)
    universe["has_expanded_event_window_coverage"] = universe["has_expanded_event_window_coverage"].fillna(False).astype(bool)
    universe["slr_reporting_bank"] = universe["slr_reporting_bank"].fillna(False).astype(bool)
    universe["is_us_insured_bank"] = True
    universe["charter_status"] = "active_call_report_filer"
    max_quarter = pd.to_datetime(frame["quarter_end"]).max()
    universe.loc[pd.to_datetime(universe["last_quarter"]) < max_quarter, "charter_status"] = "inactive_or_merged"
    if "fdic_active" in universe.columns:
        universe.loc[pd.to_numeric(universe["fdic_active"], errors="coerce").eq(0), "charter_status"] = "inactive_or_merged"
    universe["slr_scope_class"] = "non_slr_reporting_insured_bank"
    universe.loc[universe["slr_reporting_bank"], "slr_scope_class"] = "slr_reporting_insured_bank"
    universe.loc[universe["is_covered_bank_subsidiary"].fillna(False), "slr_scope_class"] = "covered_bank_subsidiary"

    keep_columns = [
        "entity_id",
        "entity_name",
        "rssd_id",
        "fdic_cert",
        "top_parent_rssd",
        "charter_status",
        "country",
        "is_us_insured_bank",
        "has_call_report_data",
        "has_tier1_capital",
        "has_total_leverage_exposure",
        "first_quarter",
        "last_quarter",
        "slr_scope_class",
        "is_covered_bank_subsidiary",
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
        "is_legacy_seed_entity",
        "override_notes",
        "hq_city",
        "hq_state",
        "hq_zip",
        "top_parent_name",
        "fdic_bank_class",
        "fdic_regulator",
        "has_usable_2019q4_baseline",
        "has_usable_2019q4_low_headroom_baseline",
        "has_usable_2019q4_high_ust_share_baseline",
        "has_usable_2019q4_covered_bank_baseline",
        "has_full_event_window_coverage",
        "has_expanded_event_window_coverage",
    ]
    universe = universe[keep_columns].copy()
    for column in ["first_quarter", "last_quarter"]:
        universe[column] = pd.to_datetime(universe[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return universe.sort_values(["is_legacy_seed_entity", "entity_name", "entity_id"], ascending=[False, True, True]).reset_index(drop=True)


def build_insured_bank_coverage_by_quarter(frame: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    coverage_flags = universe[
        [
            "entity_id",
            "has_usable_2019q4_baseline",
            "has_usable_2019q4_low_headroom_baseline",
            "has_usable_2019q4_high_ust_share_baseline",
            "has_usable_2019q4_covered_bank_baseline",
            "has_full_event_window_coverage",
        ]
    ].copy()
    merged = frame.merge(coverage_flags, how="left", on="entity_id")
    summary = (
        merged.groupby("quarter_end", dropna=False)
        .agg(
            total_insured_banks_observed=("entity_id", "nunique"),
            slr_reporting_banks_observed=("slr_reporting_observation", "sum"),
            banks_with_usable_2019q4_baseline_observed=("has_usable_2019q4_baseline", "sum"),
            banks_with_usable_2019q4_low_headroom_baseline_observed=("has_usable_2019q4_low_headroom_baseline", "sum"),
            banks_with_usable_2019q4_high_ust_share_baseline_observed=("has_usable_2019q4_high_ust_share_baseline", "sum"),
            banks_with_usable_2019q4_covered_bank_baseline_observed=("has_usable_2019q4_covered_bank_baseline", "sum"),
            continuous_event_window_banks_observed=("has_full_event_window_coverage", "sum"),
        )
        .reset_index()
        .sort_values("quarter_end")
    )
    summary["quarter_end"] = pd.to_datetime(summary["quarter_end"]).dt.strftime("%Y-%m-%d")
    return summary


def build_insured_bank_sample_manifest(frame: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    event_flags = _event_window_panel_flags(frame)
    merge_columns = [
        "entity_id",
        "event_window_observations",
        "event_window_first_quarter",
        "event_window_last_quarter",
        "pre_treatment_quarters",
        "post_treatment_quarters",
        "in_universe_b",
    ]
    event_flags = event_flags[[column for column in merge_columns if column in event_flags.columns]].copy()
    manifest = universe.merge(event_flags, how="left", on="entity_id")
    manifest["in_universe_a"] = True
    manifest["in_universe_b"] = manifest["in_universe_b"].fillna(False).astype(bool)
    manifest["in_universe_c"] = manifest["in_universe_b"] & manifest["has_usable_2019q4_baseline"].fillna(False).astype(bool)
    manifest["in_universe_c_low_headroom"] = manifest["in_universe_b"] & manifest["has_usable_2019q4_low_headroom_baseline"].fillna(False).astype(bool)
    manifest["in_universe_c_high_ust_share"] = manifest["in_universe_b"] & manifest["has_usable_2019q4_high_ust_share_baseline"].fillna(False).astype(bool)
    manifest["in_universe_c_covered_bank"] = manifest["in_universe_b"] & manifest["has_usable_2019q4_covered_bank_baseline"].fillna(False).astype(bool)
    manifest["in_universe_d"] = manifest["in_universe_c"] & manifest["has_full_event_window_coverage"].fillna(False).astype(bool)
    manifest["in_universe_e"] = manifest["in_universe_c"] & manifest["has_expanded_event_window_coverage"].fillna(False).astype(bool)

    manifest["universe_b_exclusion_reason"] = pd.NA
    manifest.loc[~manifest["in_universe_b"], "universe_b_exclusion_reason"] = "no_positive_tier1_and_tle_in_event_window"

    manifest["universe_c_exclusion_reason"] = pd.NA
    manifest.loc[manifest["in_universe_b"] & ~manifest["in_universe_c"], "universe_c_exclusion_reason"] = "missing_usable_2019q4_treatment_baseline"

    manifest["universe_c_low_headroom_exclusion_reason"] = pd.NA
    manifest.loc[
        manifest["in_universe_b"] & ~manifest["in_universe_c_low_headroom"],
        "universe_c_low_headroom_exclusion_reason",
    ] = "missing_usable_2019q4_low_headroom_baseline"

    manifest["universe_c_high_ust_share_exclusion_reason"] = pd.NA
    manifest.loc[
        manifest["in_universe_b"] & ~manifest["in_universe_c_high_ust_share"],
        "universe_c_high_ust_share_exclusion_reason",
    ] = "missing_usable_2019q4_high_ust_share_baseline"

    manifest["universe_c_covered_bank_exclusion_reason"] = pd.NA
    manifest.loc[
        manifest["in_universe_b"] & ~manifest["in_universe_c_covered_bank"],
        "universe_c_covered_bank_exclusion_reason",
    ] = "missing_usable_2019q4_covered_bank_baseline"

    manifest["universe_d_exclusion_reason"] = pd.NA
    manifest.loc[manifest["in_universe_c"] & ~manifest["in_universe_d"], "universe_d_exclusion_reason"] = "incomplete_2019q1_2021q4_coverage"

    manifest["universe_e_exclusion_reason"] = pd.NA
    manifest.loc[manifest["in_universe_c"] & ~manifest["in_universe_e"], "universe_e_exclusion_reason"] = "insufficient_pre_or_post_event_coverage"

    for column in ["event_window_observations", "pre_treatment_quarters", "post_treatment_quarters"]:
        if column in manifest.columns:
            manifest[column] = manifest[column].astype("Int64")
    for column in ["event_window_first_quarter", "event_window_last_quarter"]:
        if column in manifest.columns:
            manifest[column] = pd.to_datetime(manifest[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return manifest.sort_values(["in_universe_e", "in_universe_d", "in_universe_c", "entity_name"], ascending=[False, False, False, True]).reset_index(drop=True)


def build_all_insured_bank_panel(
    staged_path: Path,
    *,
    crosswalk_path: Path | None = None,
    overrides_path: Path | None = None,
    fdic_metadata_path: Path | None = None,
    treatment_map_path: Path | None = None,
    output_path: Path | None = None,
    universe_output_path: Path | None = None,
    coverage_output_path: Path | None = None,
    manifest_output_path: Path | None = None,
) -> dict[str, Path]:
    prepared = _merge_stage_frames(
        staged_path,
        crosswalk_path=crosswalk_path,
        overrides_path=overrides_path,
        fdic_metadata_path=fdic_metadata_path,
        treatment_map_path=treatment_map_path,
    )
    prepared = _add_scope_columns(prepared)
    prepared = _add_headroom_where_available(prepared)
    prepared = _add_treasury_metrics(prepared)
    prepared = _add_constraint_metrics(prepared)
    prepared = _add_panel_dynamics(prepared)

    panel_output = output_path or derived_data_path("insured_bank_descriptive_panel.parquet")
    universe_output = universe_output_path or derived_data_path("insured_bank_universe.csv")
    coverage_output = coverage_output_path or derived_data_path("insured_bank_coverage_by_quarter.csv")
    manifest_output = manifest_output_path or derived_data_path("insured_bank_sample_manifest.csv")

    write_frame(prepared, panel_output)
    universe = build_insured_bank_universe(prepared)
    write_frame(universe, universe_output)
    coverage = build_insured_bank_coverage_by_quarter(prepared, universe)
    write_frame(coverage, coverage_output)
    manifest = build_insured_bank_sample_manifest(prepared, universe)
    write_frame(manifest, manifest_output)
    return {
        "panel": panel_output,
        "universe": universe_output,
        "coverage": coverage_output,
        "manifest": manifest_output,
    }


def ensure_insured_bank_override_template(path: Path | None = None) -> Path:
    destination = path or reference_data_path("insured_bank_metadata_overrides.csv")
    if destination.exists():
        return destination
    pd.DataFrame(columns=OVERRIDE_COLUMNS).to_csv(destination, index=False)
    return destination


def infer_stage_quarter_label(quarter_dir: Path) -> str:
    return QuarterRef.parse(quarter_dir.name).label
