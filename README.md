# lightspeed-agentic-sandbox

Multi-provider agentic sandbox library for OpenShift Lightspeed.

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture and
[AGENTS.md](AGENTS.md) for development guidance.

Local development uses `uv`. Run `make install` for dev dependencies,
`make install-all` for all providers plus e2e extras, and `make lock` to refresh
`uv.lock` after dependency changes.

## Python dependencies

The retired Google ADK and OpenAI Agents harness libraries are removed from
`pyproject.toml`, `uv.lock`, and the generated Konflux inputs. Use the `deepagents`
extra for Anthropic/Bedrock/Vertex models, `openai` for the shared harness plus
OpenAI/Azure integrations, or `all` for every supported model integration. The obsolete ADK-only `gemini` extra is
removed; Vertex Gemini remains supported through `deepagents`.

Konflux runtime/bootstrap/build requirements and both Tekton binary package
lists are regenerated for the active dependency set. Run `make test`,
`make verify`, and `uv pip check` to check the local environment and hermetic
package coverage. Version skew between the independently resolved graphs is
reported as warnings; missing packages and orphans fail verification.

## Bumping Dependencies

The release image uses RHOAI 3.5 and is built hermetically in Konflux.
Local development uses `uv.lock`; Konflux resolves independently from
`pyproject.toml` against the RHOAI and PyPI indexes, so versions can differ.

Requirements regeneration needs Podman or Docker, access to
`quay.io/syedriko/uv:prefer-index`, Python 3.12+ with `packaging`, and
`pybuild-deps` on `PATH`. Run generation in a Linux environment: build-dependency
markers follow the host OS, so macOS generation omits Linux-only requirements.
RPM regeneration also needs a Red Hat subscription with `ACTIVATION_KEY` and
`ORG_ID` set.

```bash
make bump-deps          # upgrade uv.lock + regenerate .konflux/requirements.* and Tekton package lists
make rpm-lockfile       # regenerate .konflux/rpms.lock.yaml
make verify-hermetic-requirements  # check package coverage; version skew is a warning
```

Review and commit all changed lockfiles, requirements, and Tekton files together.
See [AGENTS.md](AGENTS.md#konflux-hermetic-builds) for the build inputs,
prerequisites, and instructions for adding dependencies.
