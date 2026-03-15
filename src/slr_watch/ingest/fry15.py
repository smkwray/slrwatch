from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..config import derived_data_path, raw_data_path
from ..pipeline import write_frame
from .browser import chromium_page

FRY15_SNAPSHOT_PAGE = "https://www.ffiec.gov/npw/FinancialReport/FRY15Reports"
OFR_BSRM_XLSX_URL = "https://www.financialresearch.gov/bank-systemic-risk-monitor/data/ofr_bsrm.xlsx"
FRY15_LABEL_PREFIXES = {
    "G-SIB Indicators": "G-SIB Indicators",
    "All Line Items": "All Line Items",
    "Summary Visualizations": "Summary Visualizations",
}
OFR_BANK_NAME_TO_FRY15_REPORTER = {
    "Bank of America": "BANK OF AMERICA",
    "Bank of New York Mellon": "BANK OF NEW YORK MELLON",
    "Citigroup": "CITIGROUP",
    "Goldman Sachs": "GOLDMAN SACHS",
    "JPMorgan Chase": "JPMORGAN CHASE",
    "Morgan Stanley": "MORGAN STANLEY",
    "State Street": "STATE STREET",
    "Wells Fargo": "WELLS FARGO & COMPANY",
}


@dataclass(frozen=True)
class FRY15Link:
    report_date: str
    label: str
    href: str


def parse_fry15_snapshot_links(html: str) -> list[FRY15Link]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    links: list[FRY15Link] = []

    current_report_date: str | None = None
    # Walk the DOM in document order and update current report date when present.
    for element in soup.find_all(["a", "div", "span", "p", "li", "td", "th", "strong"]):
        content = " ".join(element.get_text(" ", strip=True).split())
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", content):
            current_report_date = content
        if element.name == "a":
            raw_label = " ".join(element.get_text(" ", strip=True).split())
            href = element.get("href", "").strip()
            label = next(
                (prefix for prefix in FRY15_LABEL_PREFIXES if raw_label.startswith(prefix)),
                None,
            )
            if label is not None:
                links.append(
                    FRY15Link(
                        report_date=current_report_date or "unknown",
                        label=label,
                        href=urljoin(FRY15_SNAPSHOT_PAGE, href),
                    )
                )

    # deduplicate
    unique: dict[tuple[str, str, str], FRY15Link] = {}
    for link in links:
        unique[(link.report_date, link.label, link.href)] = link
    return list(unique.values())


def fetch_fry15_snapshot_links(timeout: int = 60) -> list[FRY15Link]:
    with chromium_page() as page:
        page.goto(FRY15_SNAPSHOT_PAGE, wait_until="domcontentloaded", timeout=timeout * 1000)
        page.wait_for_timeout(1000)
        return parse_fry15_snapshot_links(page.content())


def latest_all_line_items_link(links: list[FRY15Link]) -> FRY15Link | None:
    def sort_key(link: FRY15Link) -> tuple[str, str]:
        return (link.report_date, link.label)

    candidates = [link for link in links if link.label == "All Line Items"]
    if not candidates:
        return None
    return sorted(candidates, key=sort_key, reverse=True)[0]


def download_ofr_basel_scores_workbook(
    *,
    output_path: Path | None = None,
    timeout: int = 60,
) -> Path:
    destination = output_path or raw_data_path("fry15", "annual", "ofr_bsrm.xlsx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(OFR_BSRM_XLSX_URL, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def build_method1_surcharge_overlay(
    workbook_path: Path | None = None,
    *,
    output_path: Path | None = None,
) -> Path:
    source_path = workbook_path or download_ofr_basel_scores_workbook()
    frame = pd.read_excel(source_path, sheet_name="Basel Scores")
    frame = frame[frame["Parent Country"] == "United States"].copy()
    frame["fr_y15_reporter"] = frame["Bank Name"].map(OFR_BANK_NAME_TO_FRY15_REPORTER)
    frame = frame.dropna(subset=["fr_y15_reporter"]).copy()
    frame["quarter_end"] = pd.to_datetime(frame["Year"].astype(int).astype(str) + "-12-31").dt.strftime("%Y-%m-%d")
    frame["parent_method1_surcharge"] = pd.to_numeric(frame["Capital Surcharge"], errors="coerce")
    overlay = (
        frame[
            [
                "fr_y15_reporter",
                "quarter_end",
                "parent_method1_surcharge",
                "Year",
                "Systemic Importance Score",
                "Bank Name",
            ]
        ]
        .rename(
            columns={
                "Year": "source_year",
                "Systemic Importance Score": "basel_systemic_importance_score",
                "Bank Name": "ofr_bank_name",
            }
        )
        .sort_values(["fr_y15_reporter", "quarter_end"])
        .drop_duplicates(["fr_y15_reporter", "quarter_end"], keep="last")
        .reset_index(drop=True)
    )

    destination = output_path or derived_data_path("fry15_method1_overlay.parquet")
    return write_frame(overlay, destination)
