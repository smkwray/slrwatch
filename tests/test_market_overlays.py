from pathlib import Path
import zipfile

import pandas as pd

from slr_watch.ingest.market_overlays import build_market_overlay_panel
from slr_watch.ingest.nyfed_primary_dealers import build_primary_dealer_overlay
from slr_watch.ingest.trace_treasury import build_trace_treasury_overlay, normalize_trace_treasury_frame


class StubResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class StubSession:
    def __init__(self, dictionary_payload: dict, read_payload: dict):
        self.dictionary_payload = dictionary_payload
        self.read_payload = read_payload

    def get(self, url: str, **_: object) -> StubResponse:
        if "list/timeseries" in url:
            return StubResponse(self.dictionary_payload)
        return StubResponse(self.read_payload)


def test_build_primary_dealer_overlay_quarterly_aggregates(tmp_path: Path):
    session = StubSession(
        dictionary_payload={
            "pd": {
                "timeseries": [
                    {"keyid": "PDPOSGST-TOT", "description": "Net Treasury dealer position"},
                    {"keyid": "PDGSWOEXTTOT", "description": "Treasury transactions"},
                    {"keyid": "PDSORA-UTSETTOT", "description": "Treasury repo"},
                    {"keyid": "PDSIRRA-UTSETTOT", "description": "Treasury reverse repo"},
                    {"keyid": "PDSIOSB-UTSETTOT", "description": "Treasury securities borrowed"},
                    {"keyid": "PDSOOS-UTSETTOT", "description": "Treasury securities lent"},
                ]
            }
        },
        read_payload={
            "data": [
                {
                    "_id": {"date": "2020-03-18"},
                    "values": [
                        {"keyId": "PDPOSGST-TOT", "value": "100"},
                        {"keyId": "PDGSWOEXTTOT", "value": "200"},
                        {"keyId": "PDSORA-UTSETTOT", "value": "300"},
                        {"keyId": "PDSIRRA-UTSETTOT", "value": "400"},
                        {"keyId": "PDSIOSB-UTSETTOT", "value": "500"},
                        {"keyId": "PDSOOS-UTSETTOT", "value": "600"},
                    ],
                },
                {
                    "_id": {"date": "2020-03-25"},
                    "values": [
                        {"keyId": "PDPOSGST-TOT", "value": "120"},
                        {"keyId": "PDGSWOEXTTOT", "value": "220"},
                        {"keyId": "PDSORA-UTSETTOT", "value": "320"},
                        {"keyId": "PDSIRRA-UTSETTOT", "value": "420"},
                        {"keyId": "PDSIOSB-UTSETTOT", "value": "520"},
                        {"keyId": "PDSOOS-UTSETTOT", "value": "620"},
                    ],
                },
            ]
        },
    )

    output_path = build_primary_dealer_overlay(
        start_date="2020-01-01",
        end_date="2020-03-31",
        output_path=tmp_path / "nyfed_overlay.parquet",
        session=session,
    )
    overlay = pd.read_parquet(output_path)

    assert overlay["quarter_end"].tolist() == ["2020-03-31"]
    assert overlay["pd_ust_dealer_position_net_mn"].tolist() == [120.0]
    assert overlay["pd_ust_transactions_mn_weekly_avg"].tolist() == [210.0]
    assert overlay["pd_ust_repo_mn_weekly_avg"].tolist() == [310.0]
    assert overlay["pd_ust_reverse_repo_mn_weekly_avg"].tolist() == [410.0]
    assert overlay["pd_ust_sec_borrowed_mn_weekly_avg"].tolist() == [510.0]
    assert overlay["pd_ust_sec_lent_mn_weekly_avg"].tolist() == [610.0]


def test_normalize_trace_treasury_frame_accepts_common_column_names():
    frame = pd.DataFrame(
        {
            "Trade Date": ["2020-03-30", "2020-03-31"],
            "Par Value": [1000, 1500],
            "Execution Price": [99.5, 100.2],
        }
    )

    normalized = normalize_trace_treasury_frame(frame)

    assert normalized["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-03-30", "2020-03-31"]
    assert normalized["reported_volume"].tolist() == [1000, 1500]
    assert normalized["price"].tolist() == [99.5, 100.2]


def test_build_trace_treasury_overlay_and_market_panel(tmp_path: Path):
    trace_input = pd.DataFrame(
        {
            "trade_date": ["2020-03-30", "2020-03-30", "2020-06-29"],
            "reported_volume": [1000, 2000, 3000],
            "price": [99.0, 100.0, 101.0],
        }
    )
    trace_path = tmp_path / "trace.csv"
    trace_input.to_csv(trace_path, index=False)

    trace_overlay_path = build_trace_treasury_overlay(trace_path, output_path=tmp_path / "trace_overlay.parquet")
    trace_overlay = pd.read_parquet(trace_overlay_path)
    assert trace_overlay["quarter_end"].tolist() == ["2020-03-31", "2020-06-30"]
    assert trace_overlay["trace_trade_count"].tolist() == [2, 1]
    assert trace_overlay["trace_total_par_volume"].tolist() == [3000, 3000]

    primary_dealer_overlay = pd.DataFrame(
        {
            "quarter_end": ["2020-03-31", "2020-06-30"],
            "pd_ust_dealer_position_net_mn": [120.0, 140.0],
        }
    )
    primary_dealer_path = tmp_path / "primary_dealer.parquet"
    primary_dealer_overlay.to_parquet(primary_dealer_path, index=False)

    market_panel_path = build_market_overlay_panel(
        primary_dealer_path=primary_dealer_path,
        trace_path=trace_overlay_path,
        output_path=tmp_path / "market_overlay_panel.parquet",
    )
    market_panel = pd.read_parquet(market_panel_path)
    assert market_panel["quarter_end"].tolist() == ["2020-03-31", "2020-06-30"]
    assert market_panel["pd_ust_dealer_position_net_mn"].tolist() == [120.0, 140.0]
    assert market_panel["trace_trade_count"].tolist() == [2, 1]


def test_build_trace_treasury_overlay_from_finra_monthly_workbooks(tmp_path: Path):
    monthly_dir = tmp_path / "monthly"
    monthly_dir.mkdir()

    for month_end, bills_trades, total_trades, total_par in [
        ("2026-01-31", 100, 1000, 10.5),
        ("2026-02-28", 120, 1200, 12.5),
    ]:
        workbook_path = monthly_dir / f"ts-monthly-aggregates-{month_end[:7]}.xlsx"
        rows = [
            [f"TRACE Volumes - {pd.Timestamp(month_end).strftime('%B %d, %Y')}"],
            ["Par Value amounts in $Billions"],
            [None],
            ["Category", "ATS & Interdealer", None, "Dealer to Customer", None, "Total", None],
            [None, "Trades", "Par Value", "Trades", "Par Value", "Trades", "Par Value"],
            ["Bills", 10, 1.0, 90, 2.0, bills_trades, 3.0],
            ["FRNs", 5, 0.1, 6, 0.2, 11, 0.3],
            ["Nominal Coupons", 20, 4.0, 30, 5.0, 50, 9.0],
            ["TIPS", 2, 0.5, 3, 0.7, 5, 1.2],
            ["Total", 37, 5.6, 129, 6.8, total_trades, total_par],
        ]
        pd.DataFrame(rows).to_excel(workbook_path, sheet_name="Summary", header=False, index=False)

    overlay_path = build_trace_treasury_overlay(monthly_dir, output_path=tmp_path / "trace_finra_overlay.parquet")
    overlay = pd.read_parquet(overlay_path)

    assert overlay["quarter_end"].tolist() == ["2026-03-31"]
    assert overlay["trace_months_observed"].tolist() == [2]
    assert overlay["trace_total_trade_count"].tolist() == [2200]
    assert overlay["trace_total_par_value_bn"].tolist() == [23.0]
    assert overlay["trace_bills_trade_count"].tolist() == [220]


def test_build_trace_treasury_overlay_from_finra_weekly_zip(tmp_path: Path):
    archive_path = tmp_path / "ts-weekly-aggregates-2020.zip"

    weekly_rows = [
        ("2020-03-06", 400.0, 8.0, 2500.0, 20.0, 15.0),
        ("2020-03-13", 420.0, 9.0, 2600.0, 21.0, 16.0),
    ]

    with zipfile.ZipFile(archive_path, "w") as archive:
        for week_end, bills_par, frns_par, nominal_par, tips_par, total_par in weekly_rows:
            workbook_path = tmp_path / f"ts-weekly-aggregates-{week_end}.xlsx"
            rows = [
                [f"TRACE Volumes - Week of {pd.Timestamp(week_end).strftime('%B %d, %Y')}"],
                ["Amounts in $Billions"],
                [None],
                ["Category", "ATS & Interdealer", "Dealer to Customer", "Total"],
                ["Bills", 100.0, 300.0, bills_par],
                ["FRNs", 2.0, 6.0, frns_par],
                ["Nominal Coupons", 1200.0, 1300.0, nominal_par],
                ["TIPS", 10.0, 11.0, tips_par],
                ["Total", 1312.0, 1617.0, total_par],
            ]
            pd.DataFrame(rows).to_excel(workbook_path, sheet_name="Summary", header=False, index=False)
            archive.write(workbook_path, arcname=workbook_path.name)

    overlay_path = build_trace_treasury_overlay(archive_path, output_path=tmp_path / "trace_weekly_overlay.parquet")
    overlay = pd.read_parquet(overlay_path)

    assert overlay["quarter_end"].tolist() == ["2020-03-31"]
    assert overlay["trace_weeks_observed"].tolist() == [2]
    assert overlay["trace_total_par_value_bn"].tolist() == [31.0]
    assert overlay["trace_bills_par_value_bn"].tolist() == [820.0]
    assert "trace_total_trade_count" not in overlay.columns
