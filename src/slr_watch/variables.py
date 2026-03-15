from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from .config import config_path


@dataclass(frozen=True)
class VariableRegistry:
    raw: dict[str, Any]

    @property
    def variables(self) -> dict[str, Any]:
        return self.raw.get("variables", {})


def load_variable_registry() -> VariableRegistry:
    with config_path("variables.yml").open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return VariableRegistry(raw=raw)
