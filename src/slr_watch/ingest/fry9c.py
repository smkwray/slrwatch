from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
import re
from zipfile import ZipFile

import pandas as pd
import requests
from pandas.errors import ParserError

from ..config import raw_data_path, staging_data_path
from ..pipeline import normalize_source_frame, write_frame
from ..quarters import QuarterRef
from .browser import chromium_page


CHICAGO_FED_BASE = (
    "https://www.chicagofed.org/~/media/others/banking/"
    "financial-institution-reports/bhc-data"
)
NIC_FINANCIAL_DOWNLOAD_PAGE = "https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload"


def _is_historical_quarter(quarter: QuarterRef) -> bool:
    return (quarter.year, quarter.quarter) <= (2021, 1)


def historical_chicago_fed_url(quarter: QuarterRef) -> str:
    if quarter.year < 1986:
        raise ValueError("Chicago Fed historical FR Y-9C archive starts in 1986")
    if quarter.year == 1986 and quarter.quarter < 3:
        raise ValueError("Chicago Fed FR Y-9C historical archive starts at 1986Q3")
    if not _is_historical_quarter(quarter):
        raise ValueError("Chicago Fed direct historical FR Y-9C URLs are only supported through 2021Q1")
    return f"{CHICAGO_FED_BASE}/bhcf{quarter.yy}{quarter.quarter_end.strftime('%m')}.csv"


def download_historical_fry9c(
    quarter: QuarterRef,
    *,
    output_path: Path | None = None,
    timeout: int = 60,
) -> Path:
    url = historical_chicago_fed_url(quarter)
    destination = output_path or raw_data_path("fry9c", quarter.label, f"bhcf{quarter.yy}{quarter.quarter_end.strftime('%m')}.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def expected_nic_zip_filename(quarter: QuarterRef) -> str:
    return f"BHCF{quarter.quarter_end.strftime('%Y%m%d')}.ZIP"


def pick_nic_zip_filename(filenames: list[str], quarter: QuarterRef) -> str:
    expected = expected_nic_zip_filename(quarter)
    for filename in filenames:
        if filename.upper() == expected.upper():
            return filename
    raise FileNotFoundError(f"{expected} was not present on the NIC page")


def _run_playwright(mode: str, year: int, filename: str = "", output_path: Path | None = None) -> str:
    with chromium_page() as page:
        page.goto(NIC_FINANCIAL_DOWNLOAD_PAGE, wait_until="domcontentloaded", timeout=120000)
        page.locator("#DropDownlistYears").wait_for(state="visible", timeout=120000)
        page.select_option("#DropDownlistYears", str(year))
        page.wait_for_timeout(3000)

        zip_names = []
        for text in page.locator("button").all_inner_texts():
            label = " ".join(text.split())
            if re.fullmatch(r"BHCF\d{8}\.ZIP", label, flags=re.IGNORECASE):
                zip_names.append(label)

        if mode == "list":
            return json.dumps(zip_names)

        if output_path is None:
            raise ValueError("output_path is required when mode='download'")
        if not filename:
            raise ValueError("filename is required when mode='download'")
        if filename.upper() not in {name.upper() for name in zip_names}:
            raise FileNotFoundError(f"{filename} was not present on the NIC page")
        wanted = filename
        button = page.locator("button", has_text=wanted).last
        button.wait_for(state="visible", timeout=120000)
        try:
            with page.expect_download(timeout=15000) as download_info:
                button.click()
            download = download_info.value
        except Exception as exc:  # pragma: no cover - live-site behavior
            title = page.title()
            if "Just a moment" in title or "CAPTCHA" in title:
                raise RuntimeError(
                    "FFIEC blocked the automated NIC ZIP download in this environment. "
                    "Download the requested BHCF ZIP manually from the NIC page and rerun "
                    "the command with --zip-path."
                ) from exc
            raise
        download.save_as(str(output_path))
        return str(output_path)


def list_nic_zip_filenames(year: int) -> list[str]:
    output = _run_playwright("list", year)
    return json.loads(output or "[]")


def download_nic_fry9c(
    quarter: QuarterRef,
    *,
    output_path: Path | None = None,
) -> Path:
    filenames = list_nic_zip_filenames(quarter.year)
    filename = pick_nic_zip_filename(filenames, quarter)
    destination = output_path or raw_data_path("fry9c", quarter.label, filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_playwright("download", quarter.year, filename=filename, output_path=destination)
    return destination


def copy_manual_fry9c_zip(zip_path: Path, *, quarter: QuarterRef, output_path: Path | None = None) -> Path:
    destination = output_path or raw_data_path("fry9c", quarter.label, zip_path.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(zip_path, destination)
    return destination


def _resolve_case_insensitive_match(directory: Path, filename: str) -> Path | None:
    if not directory.exists() or not directory.is_dir():
        return None
    target = filename.upper()
    for child in directory.iterdir():
        if child.is_file() and child.name.upper() == target:
            return child
    return None


def find_manual_fry9c_zip(quarter: QuarterRef, search_paths: list[Path] | None = None) -> Path | None:
    filename = expected_nic_zip_filename(quarter)
    candidates = search_paths or [
        raw_data_path("fry9c", quarter.label),
        Path.cwd(),
        Path.home() / "Downloads",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.name.upper() == filename.upper():
            return candidate
        match = _resolve_case_insensitive_match(candidate, filename)
        if match is not None:
            return match
    return None


def download_fry9c(
    quarter: QuarterRef,
    *,
    zip_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    if zip_path is not None:
        return copy_manual_fry9c_zip(zip_path, quarter=quarter, output_path=output_path)
    if _is_historical_quarter(quarter):
        return download_historical_fry9c(quarter, output_path=output_path)
    try:
        return download_nic_fry9c(quarter, output_path=output_path)
    except RuntimeError as exc:
        manual_zip = find_manual_fry9c_zip(quarter)
        if manual_zip is not None:
            return copy_manual_fry9c_zip(manual_zip, quarter=quarter, output_path=output_path)
        expected = expected_nic_zip_filename(quarter)
        raise RuntimeError(
            f"{exc} Expected a manual fallback file named {expected} in "
            f"{raw_data_path('fry9c', quarter.label)}, the current working directory, or ~/Downloads."
        ) from exc


def _read_fry9c_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    separator = "^" if "^" in first_line else ","
    try:
        return pd.read_csv(path, sep=separator, dtype=str, low_memory=False)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, sep=separator, dtype=str, encoding="latin1", low_memory=False)
        except ParserError:
            if separator != "^":
                raise
            return pd.read_csv(
                path,
                sep=separator,
                dtype=str,
                encoding="latin1",
                engine="python",
                quoting=csv.QUOTE_NONE,
            )
    except ParserError:
        if separator != "^":
            raise
        # Some newer NIC text exports contain unescaped quotes inside caret-delimited
        # rows, which breaks the default C parser. Fall back to the Python parser and
        # disable quote handling for those files.
        return pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            engine="python",
            quoting=csv.QUOTE_NONE,
        )


def _extract_fry9c_zip(zip_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    candidates = sorted(
        candidate
        for candidate in output_dir.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".csv", ".txt"}
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV files were found inside {zip_path}")
    return candidates[0]


def stage_fry9c_file(
    input_path: Path,
    *,
    quarter: QuarterRef,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    destination = output_dir or staging_data_path("fry9c", quarter.label)
    destination.mkdir(parents=True, exist_ok=True)

    source_path = input_path
    if input_path.suffix.lower() == ".zip":
        source_path = _extract_fry9c_zip(input_path, destination / "extracted")

    frame = _read_fry9c_table(source_path)
    raw_path = destination / "fry9c_raw.parquet"
    write_frame(frame, raw_path)

    normalized = normalize_source_frame(
        frame,
        source_name="fry9c",
        quarter_end=quarter.quarter_end_iso,
    )
    normalized_path = destination / "fry9c_normalized.parquet"
    write_frame(normalized, normalized_path)
    return raw_path, normalized_path


def nic_current_download_notes() -> str:
    return (
        "Current and revised FR Y-9C files are published through the NIC Financial Data "
        "Download page. This repo uses Python Playwright-backed browser automation "
        "for NIC downloads and supports a --zip-path fallback for manually "
        "downloaded files. If automation is blocked, the downloader also checks "
        "for the expected BHCF ZIP in the raw quarter folder, the current working "
        "directory, and ~/Downloads before failing."
    )
