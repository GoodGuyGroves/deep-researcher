# Deep Researcher

Research-to-knowledge pipeline. Uses iterative web search (Tavily + OpenAI gpt-5.4) to deeply research topics and ingest results into OpenViking.

## Architecture

The pipeline has two entry points:

1. **CLI** -- `research.py` runs the full pipeline for a single topic or batch of topics
2. **MCP server** -- `server.py` exposes the pipeline as HTTP MCP tools on port 8001

Both share the same core flow: research (LangGraph agent) -> evaluate (rubric scoring) -> filter (source relevance) -> save (markdown) -> ingest (OpenViking) -> log (JSONL).

### Key modules

| File | Purpose |
|---|---|
| `research.py` | Orchestration, CLI, monkey-patches, OpenViking ingestion |
| `server.py` | FastMCP HTTP server with 4 tools: `research`, `list_research`, `read_research`, `health` |
| `scripts/evaluate.py` | Quality evaluation via gpt-5.4 rubric (coverage, source quality, specificity, actionability) |
| `scripts/relevance.py` | Source URL filtering via gpt-5.4 before ingestion |

### External dependency: ollama-deep-researcher

The research agent is from [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher), installed via pip from git. It provides a LangGraph `StateGraph` with nodes: `generate_query` -> `web_research` -> `summarize_sources` -> `reflect_on_summary` -> loop or `finalize_summary`.

This package only supports `ollama` and `lmstudio` as LLM providers. We use the `lmstudio` provider pointed at `https://api.openai.com/v1` to reach gpt-5.4. Two monkey-patches in `research.py` make this work:

- `_patch_lmstudio_api_key()` -- injects `OPENAI_API_KEY` into `ChatLMStudio.__init__` (which defaults to `"not-needed-for-local-models"`)
- `_patch_tavily_query_length()` -- truncates queries > 390 chars to stay within Tavily's 400-char API limit

Both patches are fragile -- if the upstream package changes its internals, they break silently. Check them if research starts failing after a dependency update.

## Environment

### Nix shell

`flake.nix` provides Python 3.12, uv, and git. The `.envrc` activates the Nix shell, runs `uv sync`, and loads `.env` via dotenv.

### Required environment variables

All set in `.env` (gitignored):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Used by monkey-patch, evaluate.py, relevance.py |
| `TAVILY_API_KEY` | Tavily search API |
| `LLM_PROVIDER` | Must be `lmstudio` |
| `LOCAL_LLM` | Model name, e.g. `gpt-5.4` |
| `LMSTUDIO_BASE_URL` | `https://api.openai.com/v1` |
| `SEARCH_API` | `tavily` |
| `MAX_WEB_RESEARCH_LOOPS` | Default iterations per run (default: 5) |
| `OPENVIKING_URL` | OpenViking server URL (default: `http://localhost:1933`) |

## MCP server

Runs on port 8001 with Streamable HTTP transport. Registered in the workspace `.mcp.json` as:

```json
"deep-researcher": {
  "type": "http",
  "url": "http://localhost:8001/mcp",
  "headers": {
    "X-OpenViking-URL": "http://localhost:1933/work"
  }
}
```

The `X-OpenViking-URL` header routes ingestion to a specific OpenViking instance per request.

Start with: `python server.py --port 8001`

## Quality gate

Research is scored 1-5 on four dimensions. If the overall average falls below 3.0, a follow-up research run is triggered targeting the weakest dimension, results are merged, and the combined output is re-evaluated. Only one retry is attempted -- if quality is still below threshold, the output is saved with a warning banner.

## Output

- **Markdown files** in `output/` with YAML frontmatter (topic, date). Filename: `YYYYMMDD-<slug>.md`
- **OpenViking resources** at `viking://resources/research/<slug>` with source URLs as children under `viking://resources/research/<slug>/sources/`
- **Run log** in `research_log.jsonl` (one JSON record per run with topic, scores, timing, config)

## Things to know when modifying

- `research.py` imports from `ollama_deep_researcher` at call time (inside `run_research()`) so that env vars are set before the package reads them.
- `server.py` also imports `research` at call time (inside the `research` tool handler) for the same reason, and uses `asyncio.to_thread()` since the underlying pipeline is synchronous.
- The `scripts/` directory is a Python package (has `__init__.py`) -- imports use `from scripts.evaluate import ...`.
- `save_output()` uses relative paths (`Path("output")`), so the server does `os.chdir(PROJECT_ROOT)` around calls to it.
- Source extraction relies on the `* Title : URL` format that local-deep-researcher produces. If the upstream format changes, `extract_sources()` will silently return no URLs.
- The `research` MCP tool is decorated with `@mcp.tool(task=True)` making it a FastMCP background task with progress reporting.
