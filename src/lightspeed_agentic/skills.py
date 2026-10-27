"""Shared skill-root helpers for provider adapters."""

from __future__ import annotations

import os


def has_skills(cwd: str) -> bool:
    """Return True when at least one subdirectory of *cwd* contains a SKILL.md.

    Matches host discovery used by DeepAgents SkillsMiddleware:
    immediate children of the skills root only. Empty subdirectories
    do not count as skills.
    """
    try:
        for entry in os.listdir(cwd):
            child = os.path.join(cwd, entry)
            if os.path.isdir(child) and os.path.isfile(os.path.join(child, "SKILL.md")):
                return True
    except OSError:
        pass
    return False
