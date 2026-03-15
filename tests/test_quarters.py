from datetime import date

from slr_watch.quarters import QuarterRef


def test_parse_quarter_label():
    quarter = QuarterRef.parse("2025Q4")
    assert quarter.year == 2025
    assert quarter.quarter == 4
    assert quarter.report_date_mmddyyyy == "12/31/2025"


def test_from_report_date():
    quarter = QuarterRef.from_report_date("06/30/2020")
    assert quarter == QuarterRef(year=2020, quarter=2)
    assert quarter.quarter_end == date(2020, 6, 30)
