from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..config import raw_data_path, staging_data_path
from ..pipeline import normalize_source_frame, write_frame
from ..quarters import QuarterRef


CALL_REPORT_BULK_URL = "https://cdr.ffiec.gov/public/pws/downloadbulkdata.aspx"
CALL_REPORT_PRODUCT_SINGLE_PERIOD = "ReportingSeriesSinglePeriod"
CALL_REPORT_TAB_DELIMITED = "TSVRadioButton"
_FILENAME_DATE = re.compile(r"(\d{8})")
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class ParsedSchedule:
    path: Path
    frame: pd.DataFrame


def expected_bulk_folder_name(report_date_mmddyyyy: str) -> str:
    return f"ffiec_cdr_call_bulk_all_schedules_{report_date_mmddyyyy}"


def _extract_hidden_fields(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    fields: dict[str, str] = {}
    for element in soup.select("input[type='hidden'][name]"):
        fields[element["name"]] = element.get("value", "")
    return fields


def _date_option_value(html: str, report_date: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for option in soup.select("#DatesDropDownList option"):
        if option.get_text(strip=True) == report_date:
            return option.get("value", report_date)
    raise ValueError(f"Could not find Call Report option value for {report_date}")


def build_bulk_download_form_data(
    html: str,
    *,
    product: str,
    report_date_value: str,
    format_type: str = CALL_REPORT_TAB_DELIMITED,
) -> dict[str, str]:
    payload = _extract_hidden_fields(html)
    payload.update(
        {
            "__EVENTTARGET": "",
            "__EVENTARGUMENT": "",
            "__LASTFOCUS": "",
            "ctl00$MainContentHolder$ListBox1": product,
            "ctl00$MainContentHolder$DatesDropDownList": report_date_value,
            "ctl00$MainContentHolder$FormatType": format_type,
            format_type: format_type,
            "ctl00$MainContentHolder$TabStrip1$Download_0": "Download",
        }
    )
    return payload


def quarter_from_bulk_filename(filename: str) -> QuarterRef:
    match = _FILENAME_DATE.search(filename)
    if not match:
        raise ValueError(f"Could not infer report date from {filename}")
    report_date = pd.to_datetime(match.group(1), format="%m%d%Y").date()
    return QuarterRef.from_date(report_date)


def download_call_report_bulk_zip(
    quarter: QuarterRef,
    *,
    output_path: Path | None = None,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> Path:
    client = session or requests.Session()
    client.headers.update(DEFAULT_HEADERS)
    landing = client.get(CALL_REPORT_BULK_URL, timeout=timeout)
    landing.raise_for_status()

    selection_payload = _extract_hidden_fields(landing.text)
    selection_payload.update(
        {
            "__EVENTTARGET": "ctl00$MainContentHolder$ListBox1",
            "__EVENTARGUMENT": "",
            "__LASTFOCUS": "",
            "ctl00$MainContentHolder$ListBox1": CALL_REPORT_PRODUCT_SINGLE_PERIOD,
        }
    )
    selection = client.post(CALL_REPORT_BULK_URL, data=selection_payload, timeout=timeout)
    selection.raise_for_status()

    report_date_value = _date_option_value(selection.text, quarter.report_date_mmddyyyy)
    payload = build_bulk_download_form_data(
        selection.text,
        product=CALL_REPORT_PRODUCT_SINGLE_PERIOD,
        report_date_value=report_date_value,
    )
    response = client.post(CALL_REPORT_BULK_URL, data=payload, timeout=timeout)
    response.raise_for_status()

    destination = output_path or raw_data_path(
        "call_reports",
        quarter.label,
        f"FFIEC-CDR-Call-Bulk-All-Schedules-{quarter.report_date_mmddyyyy_compact}.zip",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def extract_call_report_zip(zip_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    return output_dir


def _exclude_schedule(path: Path) -> bool:
    name = path.name.lower()
    return (
        name == "readme.txt"
        or name.startswith("ffiec cdr call bulk por")
        or "schedule narr" in name
        or not path.is_file()
        or path.suffix.lower() != ".txt"
    )


def iter_schedule_files(root: Path):
    for path in sorted(root.rglob("*.txt")):
        if not _exclude_schedule(path):
            yield path


def _detect_header_row(path: Path, search_rows: int = 8) -> int:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines[:search_rows]):
        cells = [cell.strip().strip('"').upper() for cell in line.split("\t")]
        if "IDRSSD" in cells or "RSSD9001" in cells:
            return idx
    return 1


def read_schedule(path: Path) -> ParsedSchedule:
    header_row = _detect_header_row(path)
    frame = pd.read_csv(
        path,
        sep="\t",
        header=header_row,
        dtype=str,
        low_memory=False,
    )
    frame.columns = [str(col).strip().strip('"') for col in frame.columns]

    key_candidates = [column for column in frame.columns if column.upper() in {"IDRSSD", "RSSD9001"}]
    if not key_candidates:
        raise ValueError(f"Could not identify RSSD key column in {path}")
    key = key_candidates[0]

    frame = frame.dropna(how="all")
    frame = frame[frame[key].notna()].copy()
    frame[key] = frame[key].astype(str).str.strip()

    if key != "RSSD9001":
        frame = frame.rename(columns={key: "RSSD9001"})

    frame = frame.drop_duplicates(subset=["RSSD9001"], keep="first")
    return ParsedSchedule(path=path, frame=frame)


def merge_call_report_bulk_folder(root: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for schedule_path in iter_schedule_files(root):
        parsed = read_schedule(schedule_path)
        schedule_frame = parsed.frame
        if merged is None:
            merged = schedule_frame
            continue
        overlapping = [column for column in schedule_frame.columns if column in merged.columns and column != "RSSD9001"]
        if overlapping:
            schedule_frame = schedule_frame.drop(columns=overlapping)
        merged = merged.merge(schedule_frame, how="outer", on="RSSD9001")
    if merged is None:
        raise FileNotFoundError(f"No call report schedule files found under {root}")
    return merged


def write_merged_call_report_bulk_folder(root: Path, output_path: Path) -> Path:
    merged = merge_call_report_bulk_folder(root)
    return write_frame(merged, output_path)


def stage_call_report_folder(
    root: Path,
    *,
    quarter: QuarterRef,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    destination = output_dir or staging_data_path("call_reports", quarter.label)
    destination.mkdir(parents=True, exist_ok=True)

    merged = merge_call_report_bulk_folder(root)
    merged_path = destination / "call_reports_merged.parquet"
    write_frame(merged, merged_path)

    normalized = normalize_source_frame(
        merged,
        source_name="call_report",
        quarter_end=quarter.quarter_end_iso,
    )
    normalized_path = destination / "call_reports_normalized.parquet"
    write_frame(normalized, normalized_path)
    return merged_path, normalized_path


def stage_call_report_zip(
    zip_path: Path,
    *,
    quarter: QuarterRef | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    inferred_quarter = quarter or quarter_from_bulk_filename(zip_path.name)
    destination = output_dir or staging_data_path("call_reports", inferred_quarter.label)
    extract_dir = destination / "extracted"
    extract_call_report_zip(zip_path, extract_dir)
    return stage_call_report_folder(extract_dir, quarter=inferred_quarter, output_dir=destination)
