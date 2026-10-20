# Project Structure

Package tree (authoritative for agents): see `AGENTS.md` Architecture section.
Do not maintain a duplicate path inventory here.

## Key Entry Points

| Entry point | How invoked |
|---|---|
| `lightspeed_agentic.batch` | Container CMD: `python -m lightspeed_agentic.batch` |
| `batch.main()` | Read `/input/`, run agent, publish Result CR, exit |
| `config.resolve_model_family()` | Resolve family, SDK env mapping, and credentials once per batch run |
| `providers.model_router.resolve_model_route(model_family=...)` | Reuse that family; return an Anthropic or Vertex/Google model route, or legacy OpenAI SDK configuration, without constructing clients |
| `run_readiness_checks(model_family, model_route)` | R1 credential env (+ file paths) and route parameters before LLM (`readiness.py`) |
| `create_provider(name, model_route=None)` | Factory lazy-imports the selected adapter; DeepAgents requires the resolved model route |
| `run_agent_query()` | Shared agent execution (`run_agent.py`) |
| `publish_agent_result()` | Result CR create + status via Kubernetes API |
| `init_tracer` / `shutdown_tracer` | OTel TracerProvider setup/teardown in `batch.main()` |

## Naming Conventions

- **Package:** `lightspeed_agentic` under `src/` (hatchling src-layout).
- **Provider adapters:** `providers/deepagents.py` and `providers/openai.py` export the active provider classes. DeepAgents receives a resolved model route for Anthropic and Vertex/Google; OpenAI remains on its legacy SDK path.
- **Model routing/construction:** `providers/model_router.py` defines `ResolvedModelRoute`, `ModelParameters`, and the `ModelCreator` callable alias. `providers/models/anthropic.py` provides direct, Bedrock, and Vertex Anthropic constructors; `providers/models/gemini.py` constructs the Vertex Gemini model. Optional SDK imports remain lazy.
- **Batch / agent:** `batch.py` (entrypoint + input reading), `run_agent.py` (provider query loop).
- **Publish results:** `publish_results/publish.py`, `publish_results/status.py`.
- **Observability modules:** `audit.py` (span events), `metrics.py` (histograms), `tracing.py` (TracerProvider + traceparent parsing).
- **Config / MCP / readiness:** `config.py` maps `LIGHTSPEED_*` → SDK env; `mcp.py` parses `LIGHTSPEED_MCP_SERVERS`; `readiness.py` runs `run_readiness_checks()` at batch startup (see `health-probes.md`).
- **Test layout:** `tests/` mirrors source. `tests/e2e/` holds batch cluster BDD (feature files, `batch_runner.py`, `suite_setup.py`). Live provider coverage runs via `make e2e` (see [e2e-testing.md](../what/e2e-testing.md)).

## Dependency Organization

The project uses optional extras to gate provider SDKs:

| Extra | Packages |
|---|---|
| `deepagents` | `deepagents`, `langchain-anthropic`, `langchain-google-genai`, `langchain-google-vertexai`, `langchain-aws`, `langchain-mcp-adapters` |
| `gemini` | `google-adk` (declared legacy extra; runtime adapter removed, dependency cleanup deferred) |
| `openai` | `openai-agents` |
| `all` | All three provider extras |
| `dev` | All providers + test/lint tools |
| `eval` | Eval-specific test dependencies |
| `e2e` | BDD test dependencies |

Runtime dependencies include `kubernetes` (Result CR publishing) and `prometheus-client` (metrics histograms). FastAPI and Uvicorn are **not** runtime dependencies.

The `google-adk` extra remains declared for now but is not used by the Vertex Gemini route; its dependency cleanup is deferred. Provider SDK imports are always lazy (inside methods or guarded by the factory) so the base package imports cleanly without any extras installed.
