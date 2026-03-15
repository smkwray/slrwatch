from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .variables import load_variable_registry


RSSD_KEY_CANDIDATES = ("RSSD9001", "IDRSSD", "rssd_id")


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        text = str(column).strip()
        lookup[text.upper()] = text
    return lookup


def detect_key_column(frame: pd.DataFrame, candidates: Iterable[str] = RSSD_KEY_CANDIDATES) -> str:
    lookup = _column_lookup(frame.columns)
    for candidate in candidates:
        actual = lookup.get(candidate.upper())
        if actual:
            return actual
    raise ValueError(f"Could not find a key column from {tuple(candidates)}")


def _present_candidates(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    lookup = _column_lookup(frame.columns)
    present: list[str] = []
    for candidate in candidates:
        actual = lookup.get(candidate.upper())
        if actual and actual not in present:
            present.append(actual)
    return present


def coalesce_numeric_fields(frame: pd.DataFrame, candidates: Iterable[str]) -> tuple[pd.Series, pd.Series]:
    present = _present_candidates(frame, candidates)
    index = frame.index
    values = pd.Series(pd.NA, index=index, dtype="Float64")
    lineage = pd.Series(pd.NA, index=index, dtype="string")
    for column in present:
        numeric = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        mask = values.isna() & numeric.notna()
        values.loc[mask] = numeric.loc[mask]
        lineage.loc[mask] = column
    return values, lineage


def normalize_source_frame(
    frame: pd.DataFrame,
    *,
    source_name: str,
    quarter_end: str,
) -> pd.DataFrame:
    registry = load_variable_registry().variables
    key_column = detect_key_column(frame)
    normalized = pd.DataFrame(index=frame.index)
    normalized["rssd_id"] = frame[key_column].astype("string").str.strip()
    normalized["quarter_end"] = quarter_end

    for variable_name, spec in registry.items():
        fields = spec.get("fields", {}).get(source_name, [])
        if not fields:
            continue
        values, lineage = coalesce_numeric_fields(frame, fields)
        normalized[variable_name] = values
        normalized[f"{variable_name}_source"] = lineage

    if "actual_slr_ratio" in normalized.columns:
        normalized["actual_slr"] = normalized["actual_slr_ratio"]
    numeric_mask = normalized["rssd_id"].str.fullmatch(r"\d+").fillna(False)
    return normalized.loc[numeric_mask].reset_index(drop=True)


def write_frame(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        frame.to_parquet(output_path, index=False)
    return output_path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def read_tables(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [read_table(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No input tables were provided")
    return pd.concat(frames, ignore_index=True)


def discover_tables(path: Path, suffix: str = ".parquet") -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob(f"*{suffix}") if candidate.is_file())


def validate_crosswalk_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "entity_id",
        "entity_name",
        "entity_type",
        "rssd_id",
        "fdic_cert",
        "top_parent_rssd",
        "country",
        "is_gsib_parent",
        "is_covered_bank_subsidiary",
        "fr_y15_reporter",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Crosswalk is missing required columns: {missing}")

    normalized = frame.copy()
    for column in ["entity_id", "entity_name", "entity_type", "rssd_id", "fdic_cert", "top_parent_rssd", "country", "fr_y15_reporter"]:
        normalized[column] = normalized[column].astype("string").str.strip()

    for column in ["is_gsib_parent", "is_covered_bank_subsidiary"]:
        normalized[column] = normalized[column].fillna(False).astype(bool)

    duplicates = normalized["entity_id"].duplicated(keep=False)
    if duplicates.any():
        raise ValueError("Crosswalk entity_id values must be unique")
    return normalized
