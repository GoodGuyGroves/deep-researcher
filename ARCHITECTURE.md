# Deep Research Pipeline Architecture

## Overview

The deep research pipeline automates the process of researching topics on the web, evaluating the quality of findings, and optionally ingesting the results into a context database (OpenViking). It uses iterative web search with LLM-powered synthesis, self-critique, and quality gating.

The pipeline is invoked either directly via `research.py` (CLI), via the MCP server (`server.py`), or through the `/research` Claude Code skill which decomposes broad topics and manages multi-run sessions.


## Architecture Diagram

```
              +-----------------+-----------------+
              |           research.py             |
              |                                   |
              |  1. run_research(topic)            |
              |     +---------------------------+ |
              |     | engine.py (research loop)  | |
              |     |                           | |
              |     | generate_query       -----+-+---> LLM (via litellm)
              |     |     |                     | |
              |     |     v                     | |
              |     | web_research         -----+-+---> Tavily API
              |     |     |                     | |       (1 result per query)
              |     |     v                     | |
              |     | summarize_sources    -----+-+---> LLM (via litellm)
              |     |     |                     | |
              |     |     v                     | |
              |     | reflect_on_summary   -----+-+---> LLM (via litellm)
              |     |     |                     | |
              |     |     v                     | |
              |     | loop_count < N? ----------+ |
              |     |   yes -> web_research     | |
              |     |   no  -> finalize_summary | |
              |     +---------------------------+ |
              |                                   |
              |  2. Extract source URLs from      |
              |     markdown output               |
              |                                   |
              |  3. evaluate_research()           |
              |     (LLM rubric scoring)          |
              |     If score < 3.0:               |
              |       -> generate follow-up query |
              |       -> re-run research loop     |
              |       -> merge + re-evaluate      |
              |                                   |
              |  4. save_output() -> output/*.md  |
              |                                   |
              |  5. filter_relevant_sources()     |
              |     (LLM relevance check)         |
              |                                   |
              |  6. ingest_to_openviking()        |
              |     (optional)                    |
              |     -> POST summary file          |
              |     -> POST each source URL       |
              |     -> wait for indexing           |
              |                                   |
              |  7. _append_research_log()        |
              |     -> research_log.jsonl          |
              +-----------------------------------+
```


## Component Details

### 1. engine.py (Research Loop)

**What:** A plain-Python iterative research loop that searches the web, synthesises findings, identifies knowledge gaps, and searches again. No LangChain or LangGraph dependency.

**How:** The loop uses litellm for all LLM calls (supporting Anthropic, OpenAI, Mistral, and 100+ other providers) and tavily-python for web search.

| Function | Purpose |
|---|---|
| `generate_query` | Creates an initial search query from the topic using the LLM (JSON mode) |
| `web_research` | Executes a Tavily search, deduplicates, and formats results for the LLM |
| `summarize_sources` | Uses the LLM to integrate new findings into a running summary (free text mode) |
| `reflect_on_summary` | Identifies knowledge gaps and generates a follow-up query (JSON mode) |
| `_finalize_summary` | Deduplicates sources and produces the final markdown output |
| `run_research_loop` | Main entry point — orchestrates the loop and returns `{"running_summary": "..."}` |

The flow: `generate_query -> [web_research -> summarize -> reflect] x N -> finalize`.

Configuration is via `ResearchConfig` dataclass, populated from environment variables or passed programmatically.

Tavily's 400-character query limit is handled natively inside `_tavily_search()` by truncating queries to 390 characters.


### 2. research.py (Orchestration Script)

**What:** The main entry point that ties together the research engine, quality evaluation, source filtering, file output, OpenViking ingestion, and run logging.

**How:** The `process_topic()` function executes the full pipeline for a single topic:

1. **Research** -- calls `run_research(topic)` which delegates to `engine.run_research_loop()`.
2. **Extract sources** -- parses the `* Title : URL` format from the markdown summary to get source URLs.
3. **Evaluate quality** -- calls `evaluate_research()`. If the overall score is below 3.0/5.0, generates and executes a follow-up research run, merges the results, and re-evaluates.
4. **Save** -- writes the summary as a markdown file to `output/` with YAML frontmatter.
5. **Filter sources** -- calls `filter_relevant_sources()` before ingestion.
6. **Ingest** -- uploads the summary file and filtered source URLs to OpenViking via its HTTP API.
7. **Log** -- appends a JSON record to `research_log.jsonl`.

**CLI interface:**
```
python research.py "topic"                       # Single topic
python research.py --file topics/list.txt        # Batch from file
python research.py -l 7 "topic"                  # Override loop count
python research.py --no-ingest "topic"           # Skip OpenViking ingestion
python research.py --url http://host:1934 "topic" # Target specific OpenViking
```


### 3. scripts/evaluate.py (Self-Critique)

**What:** Scores research output against a four-dimension rubric using litellm, and optionally generates a follow-up query targeting the weakest dimension.

| Dimension | What it measures |
|---|---|
| **Coverage** | Did the research address the actual topic or drift off-topic? |
| **Source quality** | Are sources authoritative, or mostly SEO / low-quality content? |
| **Specificity** | Does it contain concrete findings, data, and details? |
| **Actionability** | Can someone use this information to make decisions or build things? |

Each dimension is scored 1-5. If overall < 3.0, generates a follow-up search query targeting the weakest dimension.


### 4. scripts/relevance.py (Source Filtering)

**What:** Filters source URLs for relevance before OpenViking ingestion. Asks the LLM which sources are on-topic and worth keeping. Falls back to keeping all sources if filtering fails.


### 5. Configuration

**`.env`** (gitignored) -- API keys and runtime configuration:
```
LLM_MODEL=anthropic/claude-sonnet-4-6    # litellm model string
ANTHROPIC_API_KEY=sk-ant-...              # API key for the LLM provider
TAVILY_API_KEY=tvly-...                   # Tavily web search
MAX_WEB_RESEARCH_LOOPS=5                  # Iterations per research run
OPENVIKING_URL=http://localhost:1933      # Optional: OpenViking for ingestion
FETCH_FULL_PAGE=true                      # Include full page content
```

**`pyproject.toml`** -- Dependencies: `litellm`, `tavily-python`, `httpx`, `fastmcp[tasks]`, `python-dotenv`. Managed with `uv`.


## Design Decisions

### 1. Why a self-contained engine instead of local-deep-researcher?

The project originally used `langchain-ai/local-deep-researcher` (aka `ollama-deep-researcher`) which provides a LangGraph-based iterative research agent. However, that package only supports local LLMs (Ollama and LMStudio). Using cloud LLMs required monkey-patching `ChatLMStudio._generate()` to route through litellm, and patching Tavily search to handle query length limits. Both patches were fragile and could break silently on upstream updates.

The reimplementation in `engine.py` eliminates both patches, the entire LangChain/LangGraph dependency tree (~15+ transitive packages), the unpinned GitHub dependency, and the confusing env vars (`LLM_PROVIDER=lmstudio`, `LMSTUDIO_BASE_URL`). The research loop logic is ~250 lines of plain Python.

### 2. Why litellm?

Provides a unified API for 100+ LLM providers. Users can switch between Anthropic, OpenAI, Mistral, and others by changing a single `LLM_MODEL` env var. The evaluate and relevance scripts already used litellm directly.

### 3. Why Tavily over DuckDuckGo?

Higher quality results with raw page content extraction. DuckDuckGo caps at 3 results per query and often misses source citations. Tavily provides structured results with `raw_content` (full page text), which is critical because the summarizer needs actual content, not just snippets. With `max_results=1`, every search iteration gets one high-quality, full-content result.

### 4. Why self-critique before ingestion?

Based on the ACR (Autonomous Critique and Refinement) pattern. Naive iterative research can degrade quality: the running summary may drift off-topic, sources may be low-quality SEO content, or findings may be vague. The rubric-based evaluation catches these problems with a structured assessment. When quality is below threshold, the system generates a targeted follow-up query addressing the specific weakest dimension.

### 5. Why source relevance filtering?

Prevents off-topic content from leaking into the knowledge base. Without filtering, tangentially-related search results would be ingested and surface in future semantic searches, degrading retrieval precision.

### 6. Why sequential research runs, not parallel?

Tavily API rate limits and OpenViking's single-worker embedding queue. Concurrent requests to Tavily risk 429 errors. Sequential execution is more reliable and easier to debug.


## Known Limitations

1. **No parallel research runs.** Sequential execution means a session with 6 sub-topics at 5 loops each takes a long time.

2. **Research quality varies by topic.** Some topics get thin coverage despite many loops — either because Tavily returns the same result repeatedly, or because the reflection step generates queries that are too similar to previous ones.

3. **Single retry on low quality.** If the follow-up research run also scores below 3.0, the pipeline flags the output but still saves it. No mechanism for escalation beyond the single retry.

4. **No deduplication across runs.** If the same topic is researched twice, both outputs are saved. The knowledge base will contain redundant content.

5. **Hardcoded quality threshold.** The 3.0/5.0 threshold is not configurable.
