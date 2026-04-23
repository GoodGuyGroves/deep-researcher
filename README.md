# Deep Researcher

Research-to-knowledge pipeline that iteratively searches the web, evaluates quality, filters sources, and ingests results into an [OpenViking](../OpenViking/) knowledge base.

## How it works

1. **Research** -- an iterative LangGraph agent (from [local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher)) searches the web using Tavily, synthesises findings into a running summary, identifies knowledge gaps, and generates targeted follow-up queries. Each loop deepens coverage on the topic.
2. **Evaluate** -- the summary is scored against a four-dimension rubric (coverage, source quality, specificity, actionability) using gpt-5.4. If the score falls below 3.0/5.0, a targeted follow-up research run is triggered automatically.
3. **Filter** -- source URLs are checked for on-topic relevance before ingestion, preventing knowledge base pollution.
4. **Ingest** -- the summary and filtered sources are uploaded to OpenViking, where they are embedded (text-embedding-3-large), indexed, and made searchable via MCP.
5. **Log** -- every run is recorded to `research_log.jsonl` with scores, timing, and configuration for analysis.

## Quick start

### Prerequisites

- [Nix](https://nixos.org/download.html) with flakes enabled
- [direnv](https://direnv.net/) (optional, auto-activates the shell)
- API keys: `OPENAI_API_KEY` and `TAVILY_API_KEY`
- OpenViking server running on `localhost:1933` (for ingestion)

### Setup

```bash
cd deep-researcher

# Option A: direnv (automatic)
direnv allow   # activates Nix shell, creates venv, loads .env

# Option B: manual
nix develop
uv sync
source .venv/bin/activate
```

Create a `.env` file:

```
LLM_PROVIDER=lmstudio
LOCAL_LLM=gpt-5.4
LMSTUDIO_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
SEARCH_API=tavily
TAVILY_API_KEY=tvly-...
MAX_WEB_RESEARCH_LOOPS=5
FETCH_FULL_PAGE=True
STRIP_THINKING_TOKENS=False
USE_TOOL_CALLING=False
OPENVIKING_URL=http://localhost:1933
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

The server reads the `X-OpenViking-URL` HTTP header to route ingestion to a specific OpenViking instance, allowing multiple clients to share one server.

## Project structure

```
research.py          Main CLI and pipeline orchestration
server.py            FastMCP HTTP server exposing the pipeline as MCP tools
scripts/
  evaluate.py        Rubric-based quality evaluation (gpt-5.4)
  relevance.py       Source URL relevance filtering (gpt-5.4)
output/              Research output markdown files
topics/              Topic list files for batch processing
research_log.jsonl   Append-only run log (gitignored)
flake.nix            Nix dev shell (Python 3.12, uv, git)
pyproject.toml       Python dependencies (managed with uv)
ARCHITECTURE.md      Detailed architecture and design decisions
```

## Claude Code integration

The pipeline is designed to work with two Claude Code custom skills defined at the workspace level:

- **`/research`** -- decomposes broad topics into focused sub-topics, runs the pipeline for each, and returns a structured summary with quality scores.
- **`/knowledge`** -- queries the OpenViking knowledge base to answer questions grounded in stored research, with citations.

## Dependencies

- **[ollama-deep-researcher](https://github.com/langchain-ai/local-deep-researcher)** -- LangGraph iterative web research agent
- **[httpx](https://www.python-httpx.org/)** -- HTTP client for OpenViking and OpenAI API calls
- **[FastMCP](https://github.com/jlowin/fastmcp)** -- MCP server framework with background task support

External services:
- **OpenAI API** (gpt-5.4) -- LLM for research synthesis, evaluation, and source filtering
- **Tavily API** -- web search with full page content extraction
- **OpenViking** -- local context database for semantic storage and retrieval
