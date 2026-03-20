from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

from ..config import derived_data_path
from ..pipeline import write_frame

NYFED_PRIMARY_DEALER_DICTIONARY_URL = "https://markets.newyorkfed.org/api/pd/list/timeseries.json"
NYFED_PRIMARY_DEALER_READ_URL = "https://markets.newyorkfed.org/read"

DEFAULT_PRIMARY_DEALER_SERIES = {
    "PDPOSGST-TOT": {
        "column": "pd_ust_dealer_position_net_mn",
        "aggregation": "last",
    },
    "PDGSWOEXTTOT": {
        "column": "pd_ust_transactions_mn_weekly_avg",
        "aggregation": "mean",
    },
    "PDSORA-UTSETTOT": {
        "column": "pd_ust_repo_mn_weekly_avg",
        "aggregation": "mean",
    },
    "PDSIRRA-UTSETTOT": {
        "column": "pd_ust_reverse_repo_mn_weekly_avg",
        "aggregation": "mean",
    },
    "PDSIOSB-UTSETTOT": {
        "column": "pd_ust_sec_borrowed_mn_weekly_avg",
        "aggregation": "mean",
    },
    "PDSOOS-UTSETTOT": {
        "column": "pd_ust_sec_lent_mn_weekly_avg",
        "aggregation": "mean",
    },
}


def _default_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "slr-watch/0.1"})
    return session


def fetch_primary_dealer_dictionary(session: requests.Session | None = None) -> pd.DataFrame:
    active = session or _default_session()
    response = active.get(NYFED_PRIMARY_DEALER_DICTIONARY_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()["pd"]["timeseries"]
    dictionary = pd.DataFrame(payload)
    return dictionary.rename(columns={"keyid": "series_code", "description": "series_description"})


def fetch_primary_dealer_timeseries(
    start_date: str,
    end_date: str,
    series_codes: list[str],
    session: requests.Session | None = None,
) -> pd.DataFrame:
    active = session or _default_session()
    response = active.get(
        NYFED_PRIMARY_DEALER_READ_URL,
        params={
            "productCode": "40",
            "startDt": start_date,
            "endDt": end_date,
            "keyIds": ",".join(series_codes),
            "format": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json().get("data", [])

    rows: list[dict[str, object]] = []
    for item in payload:
        asof_date = item.get("_id", {}).get("date")
        for value in item.get("values", []):
            rows.append(
                {
                    "asof_date": asof_date,
                    "series_code": value.get("keyId"),
                    "value_raw": value.get("value"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["asof_date", "series_code", "value"])
    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    frame["value"] = pd.to_numeric(frame["value_raw"].replace({"*": pd.NA}), errors="coerce")
    return frame.drop(columns=["value_raw"]).sort_values(["asof_date", "series_code"]).reset_index(drop=True)


def build_primary_dealer_overlay(
    *,
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    output_path: Path | None = None,
    session: requests.Session | None = None,
) -> Path:
    """Build quarterly market overlay from NY Fed primary dealer data.

    When *end_date* is not provided, defaults to today's date, so outputs
    may differ between runs.  Pass an explicit end_date for reproducibility.
    """
    final_end_date = end_date or date.today().isoformat()
    if end_date is None:
        logger.info("No end_date specified; using today: %s", final_end_date)
    dictionary = fetch_primary_dealer_dictionary(session=session)
    weekly = fetch_primary_dealer_timeseries(
        start_date,
        final_end_date,
        list(DEFAULT_PRIMARY_DEALER_SERIES),
        session=session,
    )
    merged = weekly.merge(dictionary, how="left", on="series_code")
    merged["quarter_end"] = merged["asof_date"].dt.to_period("Q").dt.end_time.dt.normalize()

    pieces: list[pd.DataFrame] = []
    for series_code, spec in DEFAULT_PRIMARY_DEALER_SERIES.items():
        subset = merged[merged["series_code"] == series_code].copy()
        if subset.empty:
            continue
        grouped = subset.groupby("quarter_end", as_index=False)["value"].agg(spec["aggregation"])
        grouped = grouped.rename(columns={"value": spec["column"]})
        grouped[f"{spec['column']}_series_code"] = series_code
        grouped[f"{spec['column']}_series_description"] = subset["series_description"].dropna().iloc[0] if subset["series_description"].notna().any() else pd.NA
        grouped[f"{spec['column']}_weeks_observed"] = subset.groupby("quarter_end")["value"].count().to_numpy()
        pieces.append(grouped)

    if not pieces:
        overlay = pd.DataFrame(columns=["quarter_end"])
    else:
        overlay = pieces[0]
        for piece in pieces[1:]:
            overlay = overlay.merge(piece, how="outer", on="quarter_end")

    overlay = overlay.sort_values("quarter_end").reset_index(drop=True)
    overlay["quarter_end"] = pd.to_datetime(overlay["quarter_end"]).dt.strftime("%Y-%m-%d")
    destination = output_path or derived_data_path("nyfed_primary_dealer_overlay.parquet")
    return write_frame(overlay, destination)
