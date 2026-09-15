"""Garante que o workflow de CI existe com lint + pytest."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_FILE = ROOT / ".github" / "workflows" / "ci.yml"


def test_github_actions_workflow_has_lint_and_pytest() -> None:
    assert CI_FILE.is_file()
    content = CI_FILE.read_text(encoding="utf-8")
    assert "ruff check" in content
    assert "pytest" in content
    assert "poetry install" in content
