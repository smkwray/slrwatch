from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return package_root()


def config_path(name: str) -> Path:
    return repo_root() / "config" / name


def docs_path(name: str) -> Path:
    return repo_root() / "docs" / name


def data_path(*parts: str) -> Path:
    return repo_root().joinpath("data", *parts)


def raw_data_path(source: str, quarter: str, *parts: str) -> Path:
    return data_path("raw", source, quarter, *parts)


def staging_data_path(source: str, quarter: str, *parts: str) -> Path:
    return data_path("staging", source, quarter, *parts)


def derived_data_path(*parts: str) -> Path:
    return data_path("derived", *parts)


def reference_data_path(*parts: str) -> Path:
    return data_path("reference", *parts)


def reports_path(*parts: str) -> Path:
    return repo_root().joinpath("output", "reports", *parts)
