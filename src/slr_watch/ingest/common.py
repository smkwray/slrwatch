from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def is_probably_text_table(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".txt", ".csv", ".tsv"}
