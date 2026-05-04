# Deep Researcher

Research-to-knowledge pipeline that iteratively searches the web, evaluates quality, filters sources, and optionally ingests results into an [OpenViking](https://github.com/GoodGuyGroves/OpenViking) knowledge base.

## How it works

1. **Research** -- an iterative loop searches the web using Tavily, synthesises findings into a running summary, identifies knowledge gaps, and generates targeted follow-up queries. Each loop deepens coverage on the topic. LLM calls go through [litellm](https://docs.litellm.ai/), supporting Anthropic, OpenAI, Mistral, and 100+ other providers.
2. **Evaluate** -- the summary is scored against a four-dimension rubric (coverage, source quality, specificity, actionability). If the score falls below 3.0/5.0, a targeted follow-up research run is triggered automatically.
3. **Filter** -- source URLs are checked for on-topic relevance before ingestion, preventing knowledge base pollution.
4. **Ingest** -- the summary and filtered sources are uploaded to OpenViking, where they are embedded, indexed, and made searchable via MCP. This step is optional.
5. **Log** -- every run is recorded to `research_log.jsonl` with scores, timing, and configuration for analysis.

## Quick start

### Prerequisites

- API keys: an LLM provider key (e.g. `ANTHROPIC_API_KEY`) and `TAVILY_API_KEY`
- Optionally: an OpenViking server for knowledge base ingestion

### Docker (recommended)

```bash
docker compose up -d
```

The server starts on port 8001. Pass API keys via `.env` file or environment variables.

### Local development

```bash
cd deep-researcher

# Option A: Nix + direnv (automatic)
direnv allow   # activates Nix shell, creates venv, loads .env

# Option B: manual
uv sync
source .venv/bin/activate
```

Create a `.env` file:

```
LLM_MODEL=anthropic/claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
MAX_WEB_RESEARCH_LOOPS=5
# Optional: OpenViking server for knowledge base ingestion
# OPENVIKING_URL=http://localhost:1933
# OPENVIKING_API_KEY=...   # required if OpenViking has auth_mode=api_key
```

### Usage

```bash
# Single topic
python research.py "your research topic"

# Batch from file (one topic per line, # for comments)
python research.py --file topics/my-topics.txt

# Override loop count (more loops = deeper research)
python research.py -l 7 "topic"

# Skip OpenViking ingestion (research and save only)
python research.py --no-ingest "topic"

# Target a specific OpenViking instance
python research.py --url http://localhost:1934 "topic"
```

Output is saved to `output/YYYYMMDD-<slug>.md` with YAML frontmatter.

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_MODEL` | No | `anthropic/claude-sonnet-4-6` | [litellm model string](https://docs.litellm.ai/docs/providers) (e.g. `openai/gpt-4.1`, `anthropic/claude-sonnet-4-6`) |
| `ANTHROPIC_API_KEY` | Yes* | -- | API key for your LLM provider (* set whichever key litellm needs for your `LLM_MODEL`) |
| `TAVILY_API_KEY` | Yes | -- | [Tavily](https://tavily.com/) web search API key |
| `MAX_WEB_RESEARCH_LOOPS` | No | `5` | Number of search-summarize-reflect iterations per topic |
| `OPENVIKING_URL` | No | -- | OpenViking server URL for ingestion (skipped if unset) |
| `OPENVIKING_API_KEY` | No | -- | API key for OpenViking when its `auth_mode=api_key`; sent as `X-Api-Key` header |
| `FETCH_FULL_PAGE` | No | `true` | Include full page content from Tavily results |

## MCP server

The pipeline is also available as an MCP server over Streamable HTTP, for integration with Claude Code and other MCP clients.

```bash
python server.py --port 8001
```

### Tools

| Tool | Description |
|---|---|
| `research` | Run the full pipeline on a topic (long-running background task with progress) |
| `list_research` | Browse completed research outputs |
| `read_research` | Read a specific output file |
| `health` | Check server status, OpenViking reachability, and env key presence |

The server reads the `X-OpenViking-URL` HTTP header to route ingestion to a specific OpenViking instance. If the header is not set, research runs and saves output locally without ingestion.

## Project structure

```
engine.py            Core research loop (search, summarize, reflect)
research.py          CLI and pipeline orchestration (evaluate, retry, save, ingest)
server.py            FastMCP HTTP server exposing the pipeline as MCP tools
scripts/
  evaluate.py        Rubric-based quality evaluation
  relevance.py       Source URL relevance filtering
output/              Research output markdown files
topics/              Topic list files for batch processing
research_log.jsonl   Append-only run log (gitignored)
pyproject.toml       Python dependencies (managed with uv)
Dockerfile           Container image
docker-compose.yml   Container orchestration
```

## Dependencies

- **[litellm](https://docs.litellm.ai/)** -- LLM routing to any provider (Anthropic, OpenAI, Mistral, etc.)
- **[tavily-python](https://docs.tavily.com/)** -- Web search with full page content extraction
- **[httpx](https://www.python-httpx.org/)** -- HTTP client for OpenViking API calls
- **[FastMCP](https://github.com/jlowin/fastmcp)** -- MCP server framework with background task support

External services:
- **LLM API** (configurable via litellm) -- for research synthesis, evaluation, and source filtering
- **Tavily API** -- web search
- **OpenViking** (optional) -- knowledge base for semantic storage and retrieval
