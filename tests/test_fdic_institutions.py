from __future__ import annotations

from pathlib import Path

import pandas as pd

from slr_watch.ingest.fdic_institutions import build_fdic_institutions_reference, fetch_fdic_institutions


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: dict[str, str], timeout: int) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _FakeResponse(self._responses[len(self.calls) - 1])


def test_fetch_fdic_institutions_paginates_and_normalizes_fields() -> None:
    session = _FakeSession(
        [
            {
                "meta": {"total": 3},
                "data": [
                    {"data": {"FED_RSSD": "1001", "CERT": "11111", "RSSDHCR": "9001", "NAME": "Alpha Bank", "NAMEHCR": "Alpha HoldCo", "ACTIVE": "1"}},
                    {"data": {"FED_RSSD": "1002", "CERT": "22222", "RSSDHCR": "9002", "NAME": "Beta Bank", "NAMEHCR": "Beta HoldCo", "ACTIVE": "1"}},
                ],
            },
            {
                "meta": {"total": 3},
                "data": [
                    {"data": {"FED_RSSD": "1003", "CERT": "33333", "RSSDHCR": "9003", "NAME": "Gamma Bank", "NAMEHCR": "Gamma HoldCo", "ACTIVE": "0"}},
                ],
            },
        ]
    )

    frame = fetch_fdic_institutions(limit=2, session=session)

    assert len(session.calls) == 2
    assert session.calls[0]["params"]["offset"] == "0"
    assert session.calls[1]["params"]["offset"] == "2"
    assert list(frame["rssd_id"]) == ["1001", "1002", "1003"]
    assert list(frame["fdic_cert"]) == ["11111", "22222", "33333"]
    assert list(frame["fdic_top_parent_rssd"]) == ["9001", "9002", "9003"]
    assert list(frame["fdic_entity_name"]) == ["Alpha Bank", "Beta Bank", "Gamma Bank"]
    assert list(frame["fdic_top_parent_name"]) == ["Alpha HoldCo", "Beta HoldCo", "Gamma HoldCo"]
    assert frame["fdic_active"].tolist() == [1, 1, 0]


def test_build_fdic_institutions_reference_writes_csv(tmp_path: Path) -> None:
    session = _FakeSession(
        [
            {
                "meta": {"total": 1},
                "data": [
                    {"data": {"FED_RSSD": "1001", "CERT": "11111", "RSSDHCR": "9001", "NAME": "Alpha Bank", "NAMEHCR": "Alpha HoldCo", "ACTIVE": "1"}},
                ],
            }
        ]
    )
    output_path = tmp_path / "fdic_institutions.csv"

    written = build_fdic_institutions_reference(output_path=output_path, session=session)

    assert written == output_path
    frame = pd.read_csv(written, dtype={"rssd_id": "string", "fdic_cert": "string", "fdic_top_parent_rssd": "string"})
    assert len(frame) == 1
    assert frame.loc[0, "rssd_id"] == "1001"
    assert frame.loc[0, "fdic_top_parent_rssd"] == "9001"
