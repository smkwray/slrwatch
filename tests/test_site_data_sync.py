from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from slr_watch.site_data import build_site_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_site_data_matches_generated_report_outputs(tmp_path: Path) -> None:
    root = _repo_root()
    event_study_reports = root / "output" / "reports" / "event_2020" / "sample_ladder.csv"
    if not event_study_reports.exists():
        pytest.skip("Local generated event-study reports are not committed in clean clones.")
    generated_dir = tmp_path / "site_data"
    written = build_site_data(
        reports_root=root / "output" / "reports",
        output_dir=generated_dir,
    )

    assert written
    for generated_path in written:
        committed_path = root / "site" / "assets" / "data" / generated_path.name
        assert committed_path.exists()
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        committed = json.loads(committed_path.read_text(encoding="utf-8"))
        assert committed == generated


def test_public_copy_mentions_2019q4_baseline_nuance() -> None:
    root = _repo_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    site = (root / "site" / "index.html").read_text(encoding="utf-8")

    assert "2019Q4 baseline" in readme
    assert "2019&ndash;Q4 baseline" in site


def test_public_copy_mentions_current_event_study_sample_ladder_counts() -> None:
    root = _repo_root()
    event_study = json.loads((root / "site" / "assets" / "data" / "event_study.json").read_text(encoding="utf-8"))
    samples = event_study["samples"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    site = (root / "site" / "index.html").read_text(encoding="utf-8")

    assert f"{int(samples['descriptive_universe']['entities']):,} insured-bank filers" in readme
    assert f"{int(samples['slr_reporting']['entities'])} entities" in readme
    assert f"{int(samples['primary_core']['entities'])} entities" in readme
    assert f"{int(samples['flagship_primary']['clusters'])} parent clusters" in readme

    assert f"{int(samples['descriptive_universe']['entities']):,} entities" in site
    assert f"{int(samples['slr_reporting']['entities'])} SLR-reporting insured" in site
    assert f"{int(samples['primary_core']['entities'])} fully balanced banks" in site
    assert f"{int(samples['flagship_primary']['clusters'])} parent clusters" in site
