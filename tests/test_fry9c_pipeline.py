from pathlib import Path

import pandas as pd
import pytest

from slr_watch.ingest import fry9c
from slr_watch.quarters import QuarterRef


def test_historical_url():
    quarter = QuarterRef.parse("2020Q1")
    assert fry9c.historical_chicago_fed_url(quarter).endswith("/bhcf2003.csv")


def test_pick_nic_zip_filename():
    quarter = QuarterRef.parse("2021Q3")
    name = fry9c.pick_nic_zip_filename(["BHCF20210630.ZIP", "BHCF20210930.ZIP"], quarter)
    assert name == "BHCF20210930.ZIP"


def test_download_nic_fry9c_uses_playwright_listing(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(mode: str, year: int, filename: str = "", output_path: Path | None = None) -> str:
        calls.append((mode, year, filename, output_path))
        if mode == "list":
            return '["BHCF20210630.ZIP", "BHCF20210930.ZIP"]'
        output_path.write_bytes(b"zip-bytes")
        return str(output_path)

    monkeypatch.setattr(fry9c, "_run_playwright", fake_run)
    quarter = QuarterRef.parse("2021Q3")
    output = fry9c.download_nic_fry9c(quarter, output_path=tmp_path / "BHCF20210930.ZIP")

    assert output.exists()
    assert calls[0][:3] == ("list", 2021, "")
    assert calls[1][0] == "download"
    assert calls[1][2] == "BHCF20210930.ZIP"


def test_find_manual_fry9c_zip_prefers_expected_filename(tmp_path: Path):
    quarter = QuarterRef.parse("2021Q3")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    expected = downloads / "BHCF20210930.ZIP"
    expected.write_bytes(b"zip-bytes")

    found = fry9c.find_manual_fry9c_zip(quarter, search_paths=[downloads])

    assert found == expected


def test_download_fry9c_uses_manual_zip_when_nic_blocked(monkeypatch, tmp_path: Path):
    quarter = QuarterRef.parse("2021Q3")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    manual_zip = downloads / "BHCF20210930.ZIP"
    manual_zip.write_bytes(b"zip-bytes")

    def blocked_download(*args, **kwargs):
        raise RuntimeError("FFIEC blocked the automated NIC ZIP download in this environment.")

    monkeypatch.setattr(fry9c, "download_nic_fry9c", blocked_download)
    monkeypatch.setattr(fry9c, "find_manual_fry9c_zip", lambda quarter_arg, search_paths=None: manual_zip)

    output = fry9c.download_fry9c(quarter, output_path=tmp_path / "copied.zip")

    assert output.exists()
    assert output.read_bytes() == b"zip-bytes"


def test_download_fry9c_blocked_message_mentions_manual_locations(monkeypatch, tmp_path: Path):
    quarter = QuarterRef.parse("2021Q3")

    def blocked_download(*args, **kwargs):
        raise RuntimeError("FFIEC blocked the automated NIC ZIP download in this environment.")

    monkeypatch.setattr(fry9c, "download_nic_fry9c", blocked_download)
    monkeypatch.setattr(fry9c, "find_manual_fry9c_zip", lambda quarter_arg, search_paths=None: None)

    with pytest.raises(RuntimeError, match="~/Downloads"):
        fry9c.download_fry9c(quarter, output_path=tmp_path / "copied.zip")


def test_stage_fry9c_file(tmp_path: Path):
    input_path = tmp_path / "bhcf2003.csv"
    input_path.write_text(
        "IDRSSD,BHCA8274,BHCALE88,BHCK0213,BHCK1287,BHCK3531,BHCK2170,BHCK0090,BHDM1350,BHCK3545,BHCB2200,BHCK2122\n"
        "2001,100,2000,11,22,7,3000,400,50,80,2500,1200\n",
        encoding="utf-8",
    )

    _, normalized_path = fry9c.stage_fry9c_file(input_path, quarter=QuarterRef.parse("2020Q1"))
    frame = pd.read_parquet(normalized_path)

    assert frame.loc[0, "rssd_id"] == "2001"
    assert frame.loc[0, "tier1_capital"] == 100
    assert frame.loc[0, "tier1_capital_source"] == "BHCA8274"
    assert frame.loc[0, "ust_afs_fair_value"] == 22


def test_read_fry9c_table_falls_back_for_unescaped_quotes(tmp_path: Path):
    input_path = tmp_path / "bhcf20231231.txt"
    cols = ["IDRSSD", "BHCA8274", "BHCALE88"]
    row = ['2001', '100', 'name with " quote']
    input_path.write_text(
        "^".join(cols) + "\n" + "^".join(row) + "\n",
        encoding="utf-8",
    )

    frame = fry9c._read_fry9c_table(input_path)

    assert frame.loc[0, "IDRSSD"] == "2001"
    assert frame.loc[0, "BHCA8274"] == "100"
    assert frame.loc[0, "BHCALE88"] == 'name with " quote'


def test_read_fry9c_table_falls_back_for_latin1(tmp_path: Path):
    input_path = tmp_path / "bhcf20241231.txt"
    payload = "IDRSSD^BHCA8274^BHCALE88\n2001^100^señor\n"
    input_path.write_bytes(payload.encode("latin1"))

    frame = fry9c._read_fry9c_table(input_path)

    assert frame.loc[0, "IDRSSD"] == "2001"
    assert frame.loc[0, "BHCALE88"] == "señor"
