from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd

from ..config import derived_data_path
from ..pipeline import write_frame

DATE_CANDIDATES = ("execution_date", "trade_date", "date", "tradeDate", "Execution Date", "Trade Date")
VOLUME_CANDIDATES = ("reported_volume", "par_value", "quantity", "volume", "Reported Volume", "Par Value", "Quantity")
PRICE_CANDIDATES = ("price", "execution_price", "Price", "Execution Price")
FINRA_CATEGORY_ROWS = {
    "Bills": "trace_bills",
    "FRNs": "trace_frns",
    "Nominal Coupons": "trace_nominal_coupons",
    "TIPS": "trace_tips",
    "Total": "trace_total",
}


def _discover_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = [
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.suffix.lower() in {".csv", ".xlsx", ".xls", ".zip"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No TRACE input files found under {path}")
    return candidates


def _read_trace_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _read_finra_monthly_aggregate_workbook(source: object, *, label: str) -> pd.DataFrame:
    frame = pd.read_excel(source, sheet_name="Summary", header=None)
    title = str(frame.iloc[0, 0])
    if not title.startswith("TRACE Volumes - "):
        raise ValueError(f"Workbook {label} is not a FINRA Treasury monthly aggregate workbook")

    report_date = pd.to_datetime(title.replace("TRACE Volumes - ", ""), errors="raise")
    table = frame.iloc[5:40, [0, 5, 6]].copy()
    table.columns = ["category", "trade_count", "par_value_bn"]
    table["category"] = table["category"].astype("string").str.strip()
    table["trade_count"] = pd.to_numeric(table["trade_count"], errors="coerce")
    table["par_value_bn"] = pd.to_numeric(table["par_value_bn"], errors="coerce")

    row: dict[str, object] = {"trade_date": report_date}
    for category, prefix in FINRA_CATEGORY_ROWS.items():
        subset = table[table["category"] == category]
        if subset.empty:
            continue
        row[f"{prefix}_trade_count"] = subset["trade_count"].iloc[0]
        row[f"{prefix}_par_value_bn"] = subset["par_value_bn"].iloc[0]
    row["trace_source_granularity"] = "monthly"
    return pd.DataFrame([row])


def _read_finra_weekly_aggregate_workbook(source: object, *, label: str) -> pd.DataFrame:
    frame = pd.read_excel(source, sheet_name="Summary", header=None)
    title = str(frame.iloc[0, 0])
    if not title.startswith("TRACE Volumes - Week of "):
        raise ValueError(f"Workbook {label} is not a FINRA Treasury weekly aggregate workbook")

    report_date = pd.to_datetime(title.replace("TRACE Volumes - Week of ", ""), errors="raise")
    table = frame.iloc[4:40, [0, 3]].copy()
    table.columns = ["category", "par_value_bn"]
    table["category"] = table["category"].astype("string").str.strip()
    table["par_value_bn"] = pd.to_numeric(table["par_value_bn"], errors="coerce")

    row: dict[str, object] = {"trade_date": report_date}
    for category, prefix in FINRA_CATEGORY_ROWS.items():
        subset = table[table["category"] == category]
        if subset.empty:
            continue
        row[f"{prefix}_par_value_bn"] = subset["par_value_bn"].iloc[0]
    row["trace_source_granularity"] = "weekly"
    return pd.DataFrame([row])


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        actual = lookup.get(candidate.strip().lower())
        if actual:
            return actual
    raise ValueError(f"Could not find a TRACE column from {candidates}")


def normalize_trace_treasury_frame(frame: pd.DataFrame) -> pd.DataFrame:
    date_col = _resolve_column(frame, DATE_CANDIDATES)
    volume_col = _resolve_column(frame, VOLUME_CANDIDATES)
    price_col = _resolve_column(frame, PRICE_CANDIDATES)

    normalized = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(frame[date_col]),
            "reported_volume": pd.to_numeric(frame[volume_col], errors="coerce"),
            "price": pd.to_numeric(frame[price_col], errors="coerce"),
        }
    )
    return normalized.dropna(subset=["trade_date"]).reset_index(drop=True)


def _load_trace_excel(source: object, *, label: str) -> pd.DataFrame:
    workbook = pd.ExcelFile(source)
    if "Summary" in workbook.sheet_names:
        summary = pd.read_excel(workbook, sheet_name="Summary", header=None, nrows=1)
        title = str(summary.iloc[0, 0])
        if title.startswith("TRACE Volumes - Week of "):
            return _read_finra_weekly_aggregate_workbook(workbook, label=label)
        if title.startswith("TRACE Volumes - "):
            return _read_finra_monthly_aggregate_workbook(workbook, label=label)
    return normalize_trace_treasury_frame(pd.read_excel(workbook))


def _load_trace_zip(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            member for member in archive.namelist() if Path(member).suffix.lower() in {".xlsx", ".xls", ".csv"}
        )
        if not members:
            raise ValueError(f"ZIP archive {path} does not contain any supported TRACE files")
        for member in members:
            with archive.open(member) as handle:
                payload = handle.read()
            suffix = Path(member).suffix.lower()
            if suffix == ".csv":
                frames.append(normalize_trace_treasury_frame(pd.read_csv(BytesIO(payload))))
            else:
                frames.append(_load_trace_excel(BytesIO(payload), label=f"{path.name}:{member}"))
    return pd.concat(frames, ignore_index=True)


def _load_trace_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        return _load_trace_zip(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return _load_trace_excel(path, label=str(path))
    return normalize_trace_treasury_frame(pd.read_csv(path))


def build_trace_treasury_overlay(input_path: Path, *, output_path: Path | None = None) -> Path:
    frames = [_load_trace_source(path) for path in _discover_trace_files(input_path)]
    combined = pd.concat(frames, ignore_index=True)
    combined["quarter_end"] = combined["trade_date"].dt.to_period("Q").dt.end_time.dt.normalize()

    if "trace_total_par_value_bn" in combined.columns:
        aggregations = {
            "trace_total_trade_count": "sum",
            "trace_total_par_value_bn": "sum",
            "trace_bills_trade_count": "sum",
            "trace_bills_par_value_bn": "sum",
            "trace_frns_trade_count": "sum",
            "trace_frns_par_value_bn": "sum",
            "trace_nominal_coupons_trade_count": "sum",
            "trace_nominal_coupons_par_value_bn": "sum",
            "trace_tips_trade_count": "sum",
            "trace_tips_par_value_bn": "sum",
        }
        present = [column for column in aggregations if column in combined.columns]
        overlay = combined.groupby("quarter_end", as_index=False)[present].sum(min_count=1)
        if "trace_source_granularity" in combined.columns:
            monthly_counts = (
                combined.loc[combined["trace_source_granularity"] == "monthly"]
                .groupby("quarter_end")["trade_date"]
                .count()
                .rename("trace_months_observed")
            )
            weekly_counts = (
                combined.loc[combined["trace_source_granularity"] == "weekly"]
                .groupby("quarter_end")["trade_date"]
                .count()
                .rename("trace_weeks_observed")
            )
            overlay = overlay.merge(monthly_counts, how="left", on="quarter_end")
            overlay = overlay.merge(weekly_counts, how="left", on="quarter_end")
        overlay = overlay.sort_values("quarter_end").reset_index(drop=True)
        overlay["quarter_end"] = pd.to_datetime(overlay["quarter_end"]).dt.strftime("%Y-%m-%d")
        destination = output_path or derived_data_path("trace_treasury_overlay.parquet")
        return write_frame(overlay, destination)

    daily = (
        combined.groupby("trade_date", as_index=False)
        .agg(
            trace_trade_count=("reported_volume", "size"),
            trace_total_par_volume=("reported_volume", "sum"),
        )
    )
    daily["quarter_end"] = daily["trade_date"].dt.to_period("Q").dt.end_time.dt.normalize()
    daily_quarterly = (
        daily.groupby("quarter_end", as_index=False)
        .agg(
            trace_avg_daily_trade_count=("trace_trade_count", "mean"),
            trace_avg_daily_par_volume=("trace_total_par_volume", "mean"),
        )
    )

    overlay = (
        combined.groupby("quarter_end", as_index=False)
        .agg(
            trace_trade_count=("reported_volume", "size"),
            trace_total_par_volume=("reported_volume", "sum"),
            trace_mean_price=("price", "mean"),
            trace_median_price=("price", "median"),
        )
        .merge(daily_quarterly, how="left", on="quarter_end")
        .sort_values("quarter_end")
        .reset_index(drop=True)
    )
    overlay["quarter_end"] = pd.to_datetime(overlay["quarter_end"]).dt.strftime("%Y-%m-%d")

    destination = output_path or derived_data_path("trace_treasury_overlay.parquet")
    return write_frame(overlay, destination)
