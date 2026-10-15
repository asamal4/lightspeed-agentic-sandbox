# 0002 — SDK-Native DeepAgents Skills Route

- **Jira:** [OLS-4413](https://redhat.atlassian.net/browse/OLS-4413)
- **Status:** Accepted and implemented. The SDK-native route was proposed in [PR #248](https://github.com/openshift/lightspeed-agentic-sandbox/pull/248) and adopted for PR #246.
- **Date:** 2026-10-14
- **Behavior:** [provider-contract.md](../what/provider-contract.md), Skills (rule 23)

## Context

OLS-4413 requires DeepAgents to use scripts shipped with skills. A custom `run_skill_script` tool was considered, but DeepAgents already provides skill discovery and a virtual filesystem route. The model can read skill instructions and resources, then invoke scripts using the ordinary shell tool.

## Decision

When `has_skills(cwd)` finds skills, pass `skills=["/skills/"]` to `create_deep_agent()`. Compose the backend as:

- `LocalShellBackend(root_dir=cwd, virtual_mode=False)` for the default workspace and shell.
- `FilesystemBackend(root_dir=cwd, virtual_mode=True)` mounted at `/skills/` for skill discovery and reads.

The SDK's filesystem middleware supplies the virtual-to-host path mapping needed for shell commands to execute scripts from their physical skill paths. Use standard `execute`; do not add a custom skill-script tool, path validator, prompt rewrite, or separate process runner.

Each `execute` call starts a fresh shell. Commands that depend on shared shell state, such as setup followed by a check using an exported variable, must be composed into one call. Shell output is capped at 32,000 bytes through `LocalShellBackend(max_output_bytes=MAX_TOOL_RETURN_CHARS)`. Composite-backend artifacts are stored under `HOME/.deepagents` (falling back to `/home/agent`).

## Security boundary and consequences

The `/skills/` mapping and its prompt guidance are not path validation or shell confinement. General shell execution remains available, and the model may compose commands. Pod permissions, mounts, and RBAC—not the virtual route—are the security boundary.

The shared `allowed_tools` list is not currently passed to `create_deep_agent()`, so its built-in shell/filesystem tool surface is broader than that list. This decision does not treat the SDK path mapping as a security control.

## Verification

Offline coverage is in [`test_deepagents.py`](../../../tests/test_deepagents.py) and the model-driven skill scenarios in [`skills.feature`](../../../tests/e2e/features/skills.feature). PR #248 reports a live run; this record does not claim a new cluster-backed run for PR #246.
