from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from slr_watch.ingest.call_reports import (
    CALL_REPORT_PRODUCT_SINGLE_PERIOD,
    build_bulk_download_form_data,
    quarter_from_bulk_filename,
    stage_call_report_zip,
)
from slr_watch.quarters import QuarterRef


def test_build_bulk_download_form_data():
    html = """
    <html>
      <body>
        <form>
          <input type="hidden" name="__VIEWSTATE" value="abc123" />
          <input type="hidden" name="__EVENTVALIDATION" value="xyz789" />
        </form>
      </body>
    </html>
    """
    payload = build_bulk_download_form_data(
        html,
        product=CALL_REPORT_PRODUCT_SINGLE_PERIOD,
        report_date_value="150",
    )
    assert payload["__VIEWSTATE"] == "abc123"
    assert payload["ctl00$MainContentHolder$ListBox1"] == CALL_REPORT_PRODUCT_SINGLE_PERIOD
    assert payload["ctl00$MainContentHolder$DatesDropDownList"] == "150"
    assert payload["ctl00$MainContentHolder$TabStrip1$Download_0"] == "Download"


def test_quarter_from_bulk_filename():
    assert quarter_from_bulk_filename("FFIEC-CDR-Call-Bulk-All-Schedules-12312025.zip") == QuarterRef(2025, 4)


def test_stage_call_report_zip(tmp_path: Path):
    zip_path = tmp_path / "FFIEC-CDR-Call-Bulk-All-Schedules-03312020.zip"
    schedule = (
        "metadata row\n"
        "RSSD9001\tRCFA8274\tRCFAH015\tRCFD0213\tRCFD1287\tRCFD3531\tRCFD2170\tRCFD0090\tRCFD1350\tRCFD3545\tRCON2200\tRCON2122\n"
        "1111\t55\t1000\t10\t20\t5\t2000\t300\t40\t60\t1500\t800\n"
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("ffiec_cdr_call_bulk_all_schedules_03312020/RCON_schedule.txt", schedule)

    _, normalized_path = stage_call_report_zip(zip_path, output_dir=tmp_path / "staged")
    frame = pd.read_parquet(normalized_path)

    assert frame.loc[0, "rssd_id"] == "1111"
    assert frame.loc[0, "quarter_end"] == "2020-03-31"
    assert frame.loc[0, "tier1_capital"] == 55
    assert frame.loc[0, "tier1_capital_source"] == "RCFA8274"
    assert frame.loc[0, "total_leverage_exposure"] == 1000
    assert frame.loc[0, "deposits"] == 1500
    assert frame.loc[0, "loans"] == 800


def test_stage_call_report_zip_skips_narrative_schedule(tmp_path: Path):
    zip_path = tmp_path / "FFIEC-CDR-Call-Bulk-All-Schedules-12312022.zip"
    numeric_schedule = (
        "metadata row\n"
        "RSSD9001\tRCFA8274\tRCFAH015\tRCFD2170\n"
        "1111\t55\t1000\t2000\n"
    )
    narrative_schedule = (
        "\"IDRSSD\"\tRCON6979\tTEXT6980\t\n"
        "\tNO COMMENT-BK MANAGEMENT STATEMENT\tBANK MANAGEMENT STATEMENT\t\n"
        "1111\ttrue\tEdit Name:\tEdit Type:\tLong Message:\n"
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("ffiec_cdr_call_bulk_all_schedules_12312022/RCON_schedule.txt", numeric_schedule)
        archive.writestr("ffiec_cdr_call_bulk_all_schedules_12312022/FFIEC CDR Call Schedule NARR 12312022.txt", narrative_schedule)

    _, normalized_path = stage_call_report_zip(zip_path, output_dir=tmp_path / "staged")
    frame = pd.read_parquet(normalized_path)

    assert frame.loc[0, "rssd_id"] == "1111"
    assert frame.loc[0, "tier1_capital"] == 55
    assert frame.loc[0, "total_leverage_exposure"] == 1000
