from pathlib import Path

import pandas as pd

from slr_watch.ingest.fry15 import build_method1_surcharge_overlay, parse_fry15_snapshot_links


def test_parse_fry15_snapshot_links_resolves_absolute_urls():
    html = """
    <div>
      <p>12/31/2024</p>
      <a href="/npw/StaticData/Y15SnapShot/20241231_all.csv">All Line Items</a>
      <a href="/npw/StaticData/Y15SnapShot/20241231_indicators.csv">G-SIB Indicators</a>
    </div>
    """

    links = parse_fry15_snapshot_links(html)

    assert len(links) == 2
    assert links[0].report_date == "12/31/2024"
    assert links[0].href.startswith("https://www.ffiec.gov/")


def test_build_method1_surcharge_overlay_from_workbook(tmp_path: Path):
    workbook_path = tmp_path / "ofr_bsrm.xlsx"
    frame = pd.DataFrame(
        [
            {
                "Institution Name": "JP Morgan (US)",
                "Bank Name": "JPMorgan Chase",
                "Parent Country": "United States",
                "Year": 2024,
                "Capital Surcharge": 0.025,
                "Systemic Importance Score": 442,
            },
            {
                "Institution Name": "Bank of America (US)",
                "Bank Name": "Bank of America",
                "Parent Country": "United States",
                "Year": 2024,
                "Capital Surcharge": 0.015,
                "Systemic Importance Score": 327,
            },
            {
                "Institution Name": "BNP Paribas (FR)",
                "Bank Name": "BNP Paribas",
                "Parent Country": "France",
                "Year": 2024,
                "Capital Surcharge": 0.02,
                "Systemic Importance Score": 344,
            },
        ]
    )
    with pd.ExcelWriter(workbook_path) as writer:
        frame.to_excel(writer, sheet_name="Basel Scores", index=False)

    output_path = build_method1_surcharge_overlay(workbook_path, output_path=tmp_path / "overlay.parquet")
    overlay = pd.read_parquet(output_path)

    assert overlay["fr_y15_reporter"].tolist() == ["BANK OF AMERICA", "JPMORGAN CHASE"]
    assert overlay["quarter_end"].tolist() == ["2024-12-31", "2024-12-31"]
    assert overlay["parent_method1_surcharge"].tolist() == [0.015, 0.025]
