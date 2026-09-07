from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _public_text_paths(root: Path) -> list[Path]:
    paths = [root / "README.md", root / "site" / "index.html"]
    paths.extend(sorted((root / "site" / "assets" / "data").glob("*.json")))
    return [path for path in paths if path.exists()]


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


def test_dot_plot_domain_includes_negative_coefficients_and_zero() -> None:
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for chart behavior validation")
    script = _repo_root() / "site" / "assets" / "js" / "main.js"
    check = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const context = {
  localStorage: {getItem: () => null},
  window: {matchMedia: () => ({matches: false})},
  document: {addEventListener: () => {}}
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const rows = [{primary: {coef: -0.06}, expanded: {coef: 0.02},
               flagship_clustered: {coef: -0.12}}];
const domain = context.computeDotPlotDomain(rows);
assert(domain.min < -0.12 && domain.max > 0.02);
const empty = context.computeDotPlotDomain([]);
assert(empty.min < 0 && empty.max > 0);
const zero = context.computeDotPlotDomain([{primary: {coef: 0}}]);
assert(zero.min < 0 && zero.max > 0);
const ticks = context.buildLinearTicks(domain.min, domain.max, 7);
assert(ticks.length === 7 && ticks[0] === domain.min);
assert(Math.abs(ticks[6] - domain.max) < 1e-12);
"""
    subprocess.run([node, "-e", check, str(script)], check=True, capture_output=True, text=True)
