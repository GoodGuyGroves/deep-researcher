# Deep Researcher

Research-to-knowledge pipeline. Uses iterative web search (Tavily) with an LLM (via litellm) to deeply research topics, with optional ingestion into OpenViking.

## Architecture

The pipeline has two entry points:

1. **CLI** -- `research.py` runs the full pipeline for a single topic or batch of topics
2. **MCP server** -- `server.py` exposes the pipeline as HTTP MCP tools on port 8001

Both share the same core flow: research (engine.py loop) -> evaluate (rubric scoring) -> filter (source relevance) -> save (markdown) -> ingest (OpenViking, optional) -> log (JSONL).

### Key modules

| File | Purpose |
|---|---|
| `engine.py` | Core research loop: generate query -> Tavily search -> summarize -> reflect -> loop. Uses litellm + tavily-python directly. |
| `research.py` | Orchestration, CLI, OpenViking ingestion, quality retry logic |
| `server.py` | FastMCP HTTP server with 4 tools: `research`, `list_research`, `read_research`, `health` |
| `scripts/evaluate.py` | Quality evaluation via litellm rubric (coverage, source quality, specificity, actionability) |
| `scripts/relevance.py` | Source URL filtering via litellm before ingestion |

### Research engine (engine.py)

The research loop is a plain Python implementation with no LangChain/LangGraph dependency. It uses:

- **litellm** for all LLM calls (supports Anthropic, OpenAI, Mistral, and 100+ other providers)
- **tavily-python** for web search (with built-in query truncation for the 400-char API limit)

The loop runs `max_loops` iterations of: search the web -> summarize findings into a running summary -> reflect on knowledge gaps to generate the next query. The final output is a markdown document with a sources section.

`run_research_loop()` is the main entry point. It accepts a `ResearchConfig` (or reads from env vars) and returns `{"running_summary": "..."}`.

## Environment

### Nix shell

`flake.nix` provides Python, uv, and git. The `.envrc` activates the Nix shell, runs `uv sync`, and loads `.env` via dotenv.

### Environment variables

All set in `.env` (gitignored):

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes* | -- | API key for the LLM provider (* or whichever key litellm needs for `LLM_MODEL`) |
| `TAVILY_API_KEY` | Yes | -- | Tavily web search API |
| `LLM_MODEL` | No | `anthropic/claude-sonnet-4-6` | litellm model string |
| `MAX_WEB_RESEARCH_LOOPS` | No | `5` | Research iterations per topic |
| `OPENVIKING_URL` | No | -- | OpenViking server URL for ingestion (skipped if unset) |
| `OPENVIKING_API_KEY` | No | -- | API key for OpenViking when its `auth_mode=api_key`; sent as `X-Api-Key` header |
| `FETCH_FULL_PAGE` | No | `true` | Include full page content from Tavily results |

## Running

The MCP server runs via Docker Compose:

```bash
# Start the server
docker compose up -d

# Check status
docker compose ps

# Follow logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after code changes
docker compose up -d --build
```

The CLI can also be run directly (for local dev or one-off research):
```bash
python research.py "topic"
```

Or via the container:
```bash
docker compose run deep-researcher python research.py "topic"
```

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

The `X-OpenViking-URL` header routes ingestion to a specific OpenViking instance per request. If omitted, research still runs but skips ingestion.

## Quality gate

Research is scored 1-5 on four dimensions. If the overall average falls below 3.0, a follow-up research run is triggered targeting the weakest dimension, results are merged, and the combined output is re-evaluated. Only one retry is attempted -- if quality is still below threshold, the output is saved with a warning banner.

## Output

- **Markdown files** in `output/` with YAML frontmatter (topic, date). Filename: `YYYYMMDD-<slug>.md`
- **OpenViking resources** at `viking://resources/research/<slug>` with source URLs as children under `viking://resources/research/<slug>/sources/`
- **Run log** in `research_log.jsonl` (one JSON record per run with topic, scores, timing, config)

## Things to know when modifying

- `server.py` imports `research` at call time (inside the `research` tool handler) and uses `asyncio.to_thread()` since the underlying pipeline is synchronous.
- The `scripts/` directory is a Python package (has `__init__.py`) -- imports use `from scripts.evaluate import ...`.
- `save_output()` uses relative paths (`Path("output")`), so the server does `os.chdir(PROJECT_ROOT)` around calls to it.
- Source extraction relies on the `* Title : URL` format that `engine.py` produces. If the format changes, `extract_sources()` in research.py will silently return no URLs.
- The `research` MCP tool is decorated with `@mcp.tool(task=True)` making it a FastMCP background task with progress reporting.
- `evaluate.py` and `relevance.py` use `litellm.completion()` directly. They read `LLM_MODEL` for the model name at import time.
