from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _public_text_paths(root: Path) -> list[Path]:
    return [root / "README.md"]


def test_public_files_exclude_internal_language_and_local_paths() -> None:
    root = _repo_root()
    forbidden_snippets = [
        "/Users/",
        "GPT Pro",
        "handoff prompt",
        "commit/push",
        "v1 release",
        "next-question handoff",
    ]

    for path in _public_text_paths(root):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in text, f"{snippet!r} leaked into {path}"


def test_local_workflow_dirs_remain_ignored_and_unreferenced_publicly() -> None:
    root = _repo_root()
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "do/" in gitignore

    public_text = "\n".join(path.read_text(encoding="utf-8") for path in _public_text_paths(root))
    assert "`do/`" not in public_text
    assert " do/" not in public_text
