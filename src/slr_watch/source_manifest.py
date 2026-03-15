from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from .config import config_path


@dataclass(frozen=True)
class SourceManifest:
    raw: dict[str, Any]

    @property
    def sources(self) -> list[dict[str, Any]]:
        return self.raw.get("sources", [])


def load_source_manifest() -> SourceManifest:
    with config_path("source_manifest.yml").open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SourceManifest(raw=raw)


def format_sources_for_cli() -> str:
    manifest = load_source_manifest()
    blocks: list[str] = []
    for source in manifest.sources:
        lines = [
            f"- {source['id']}: {source['title']}",
            f"  owner: {source.get('owner', 'n/a')}",
            f"  cadence: {source.get('cadence', 'n/a')}",
            f"  retrieval_status: {source.get('retrieval_status', 'n/a')}",
            f"  landing_page: {source.get('landing_page', 'n/a')}",
        ]
        if source.get("notes"):
            lines.append(f"  notes: {source['notes']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
