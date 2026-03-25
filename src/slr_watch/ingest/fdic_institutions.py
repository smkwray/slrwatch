from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..config import derived_data_path
from ..pipeline import write_frame


FDIC_INSTITUTIONS_API = "https://banks.data.fdic.gov/api/institutions"
DEFAULT_FIELDS = [
    "ACTIVE",
    "BKCLASS",
    "CERT",
    "CHARTER",
    "CITY",
    "FED",
    "FED_RSSD",
    "HCTMULT",
    "NAME",
    "NAMEHCR",
    "PARCERT",
    "REGAGNT",
    "RSSDHCR",
    "SASSER",
    "STALP",
    "STALPHCR",
    "ULTCERT",
    "ZIP",
]


def fetch_fdic_institutions(
    *,
    limit: int = 10000,
    fields: list[str] | None = None,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    requested_fields = fields or DEFAULT_FIELDS
    client = session or requests.Session()
    offset = 0
    total: int | None = None
    rows: list[dict[str, object]] = []

    while total is None or offset < total:
        params = {
            "limit": str(limit),
            "offset": str(offset),
            "fields": ",".join(requested_fields),
        }
        response = client.get(FDIC_INSTITUTIONS_API, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        total = int(payload["meta"]["total"])
        data = payload.get("data", [])
        if not data:
            break
        rows.extend(row.get("data", {}) for row in data)
        offset += len(data)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    renamed = frame.rename(
        columns={
            "ACTIVE": "fdic_active",
            "BKCLASS": "fdic_bank_class",
            "CERT": "fdic_cert",
            "CHARTER": "fdic_charter_code",
            "CITY": "fdic_city",
            "FED": "fdic_federal_member_flag",
            "FED_RSSD": "rssd_id",
            "HCTMULT": "fdic_multi_bank_holding_company_flag",
            "NAME": "fdic_entity_name",
            "NAMEHCR": "fdic_top_parent_name",
            "PARCERT": "fdic_parent_cert",
            "REGAGNT": "fdic_regulator",
            "RSSDHCR": "fdic_top_parent_rssd",
            "SASSER": "fdic_sasser_flag",
            "STALP": "fdic_state",
            "STALPHCR": "fdic_top_parent_state",
            "ULTCERT": "fdic_ultimate_cert",
            "ZIP": "fdic_zip",
        }
    ).copy()

    for column in ["rssd_id", "fdic_cert", "fdic_top_parent_rssd", "fdic_parent_cert", "fdic_ultimate_cert"]:
        if column in renamed.columns:
            renamed[column] = renamed[column].astype("string").str.strip()

    for column in [
        "fdic_entity_name",
        "fdic_top_parent_name",
        "fdic_city",
        "fdic_state",
        "fdic_regulator",
        "fdic_bank_class",
        "fdic_charter_code",
        "fdic_zip",
        "fdic_top_parent_state",
    ]:
        if column in renamed.columns:
            renamed[column] = renamed[column].astype("string").str.strip()

    for column in [
        "fdic_active",
        "fdic_federal_member_flag",
        "fdic_multi_bank_holding_company_flag",
        "fdic_sasser_flag",
    ]:
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce").astype("Int64")

    renamed = renamed.dropna(subset=["rssd_id"]).drop_duplicates("rssd_id", keep="first").reset_index(drop=True)
    return renamed


def build_fdic_institutions_reference(
    *,
    output_path: Path | None = None,
    session: requests.Session | None = None,
) -> Path:
    destination = output_path or derived_data_path("fdic_institutions.csv")
    frame = fetch_fdic_institutions(session=session)
    return write_frame(frame, destination)
