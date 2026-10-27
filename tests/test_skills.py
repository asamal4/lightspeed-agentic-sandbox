"""Tests for shared skill-root presence detection."""

from pathlib import Path

from lightspeed_agentic.skills import has_skills


def test_has_skills_true_when_skill_md_in_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "my-skill").mkdir()
    (tmp_path / "my-skill" / "SKILL.md").write_text("# skill")
    assert has_skills(str(tmp_path)) is True


def test_has_skills_false_when_cwd_empty(tmp_path: Path) -> None:
    assert has_skills(str(tmp_path)) is False


def test_has_skills_false_when_cwd_missing() -> None:
    assert has_skills("/nonexistent/skills-root") is False


def test_has_skills_false_for_agents_workdir_without_skill_md(tmp_path: Path) -> None:
    """An empty directory without SKILL.md must not enable skills by itself."""
    (tmp_path / ".agents").mkdir()
    assert has_skills(str(tmp_path)) is False


def test_has_skills_false_for_skill_md_at_cwd_root(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("# skill")
    assert has_skills(str(tmp_path)) is False
