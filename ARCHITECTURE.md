# Deep Research Pipeline Architecture

## Overview

The deep research pipeline automates the process of building a curated knowledge base by iteratively researching topics on the web, evaluating the quality of findings, and ingesting the results into a context database (OpenViking). It exists to serve a two-phase vision:

1. **Now (cloud phase):** Use cloud LLMs (gpt-5.4 via OpenAI API) and professional search APIs (Tavily) to produce high-quality, well-sourced research. A frontier model does the synthesis, self-critique, and quality gating.
2. **Later (local phase):** The knowledge base built during the cloud phase becomes the retrieval substrate for local LLMs. An MCP server exposes semantic search over the accumulated research, so local models can answer questions grounded in vetted content rather than relying on their own parametric knowledge.

The pipeline is invoked either directly via `research.py` or through the `/research` Claude Code skill, which decomposes broad topics and manages multi-run sessions.


## Architecture Diagram

```
                          Claude Code session
                                 |
                      /research skill (SKILL.md)
                                 |
                    +--------------------------+
                    | Decompose broad topic    |
                    | into 2-6 focused         |
                    | sub-topics with loop     |
                    | counts                   |
                    +-----------+--------------+
                                |
                    Spawn single subagent
                    (runs all sub-topics sequentially)
                                |
                    +-----------v--------------+
                    | For each sub-topic:      |
                    |                          |
                    |   research.py -l N       |
                    |   "sub-topic string"     |
                    +-----------+--------------+
                                |
              +-----------------v------------------+
              |           research.py              |
              |                                    |
              |  1. Monkey-patch ChatLMStudio      |
              |     (inject OPENAI_API_KEY)        |
              |                                    |
              |  2. Monkey-patch Tavily search      |
              |     (truncate queries > 390 chars) |
              |                                    |
              |  3. graph.invoke(topic)            |
              |     +---------------------------+  |
              |     | local-deep-researcher     |  |
              |     | (LangGraph)               |  |
              |     |                           |  |
              |     | generate_query            |  |
              |     |     |                     |  |
              |     |     v                     |  |
              |     | web_research (Tavily)  <--+--+---> Tavily API
              |     |     |                     |  |       (1 result per query)
              |     |     v                     |  |
              |     | summarize_sources  <------+--+---> gpt-5.4 (OpenAI)
              |     |     |                     |  |
              |     |     v                     |  |
              |     | reflect_on_summary        |  |
              |     |     |                     |  |
              |     |     v                     |  |
              |     | loop_count <= N? ---------+  |
              |     |   yes -> web_research     |  |
              |     |   no  -> finalize_summary |  |
              |     +---------------------------+  |
              |                                    |
              |  4. Extract source URLs from       |
              |     markdown output                |
              |                                    |
              |  5. evaluate_research()            |
              |     (gpt-5.4 rubric scoring)       |
              |     If score < 3.0:                |
              |       -> generate follow-up query  |
              |       -> re-run graph              |
              |       -> merge + re-evaluate       |
              |                                    |
              |  6. save_output() -> output/*.md   |
              |                                    |
              |  7. filter_relevant_sources()      |
              |     (gpt-5.4 relevance check)      |
              |                                    |
              |  8. ingest_to_openviking()         |
              |     -> POST summary file           |
              |     -> POST each source URL        |
              |     -> wait for indexing            |
              |                                    |
              |  9. _append_research_log()         |
              |     -> research_log.jsonl           |
              +------------------------------------+
                                |
              +-----------------v------------------+
              |           OpenViking               |
              |  (localhost:1933)                   |
              |                                    |
              |  - Embeds content with             |
              |    text-embedding-3-large (3072d)  |
              |  - Auto-generates L0 abstracts     |
              |    and L1 overviews via gpt-5.4    |
              |  - Stores in viking:// namespace   |
              |  - MCP server for retrieval        |
              +------------------------------------+
                                |
                                v
                     /knowledge skill
                     (semantic search + read)
```


## Component Details

### 1. local-deep-researcher (LangGraph Agent)

**What:** An iterative web research agent that searches, synthesizes, identifies knowledge gaps, and searches again in a loop. Published by LangChain AI as `ollama-deep-researcher`.

**Why:** The iterative gap-identification loop is the core value proposition. Each cycle, the agent reflects on what it knows and generates a targeted follow-up query to fill gaps. Building this from scratch would be reinventing a well-tested wheel.

**How:** The agent is a compiled LangGraph `StateGraph` with five nodes:

| Node | Purpose |
|---|---|
| `generate_query` | Creates an initial search query from the topic using the LLM |
| `web_research` | Executes the search via Tavily (or DuckDuckGo/Perplexity/SearXNG) |
| `summarize_sources` | Uses the LLM to integrate new findings into a running summary |
| `reflect_on_summary` | Identifies knowledge gaps and generates a follow-up query |
| `finalize_summary` | Deduplicates sources and produces the final markdown output |

The flow is: `generate_query -> web_research -> summarize_sources -> reflect_on_summary -> [loop or finalize]`. The `route_research` function checks `research_loop_count` against `max_web_research_loops` to decide whether to loop or finalize.

State is tracked in `SummaryState`:
- `research_topic` -- the input topic
- `search_query` -- current query (updated each iteration)
- `web_research_results` -- accumulating list of raw search results
- `sources_gathered` -- accumulating list of formatted source citations
- `research_loop_count` -- iteration counter
- `running_summary` -- the evolving summary document

Configuration is read from environment variables via the `Configuration` pydantic model: `LOCAL_LLM`, `LLM_PROVIDER`, `SEARCH_API`, `MAX_WEB_RESEARCH_LOOPS`, `FETCH_FULL_PAGE`, `STRIP_THINKING_TOKENS`, `USE_TOOL_CALLING`, `LMSTUDIO_BASE_URL`.

**Where:** Installed as a pip dependency from GitHub (`langchain-ai/local-deep-researcher.git`). Package code lives in `.venv/lib/python3.*/site-packages/ollama_deep_researcher/`. Key files: `graph.py`, `state.py`, `configuration.py`, `prompts.py`, `lmstudio.py`, `utils.py`.


### 2. research.py (Orchestration Script)

**What:** The main entry point that ties together the research agent, quality evaluation, source filtering, file output, OpenViking ingestion, and run logging.

**Why:** local-deep-researcher produces a summary but has no concept of quality gating, source filtering, knowledge base ingestion, or run logging. This script wraps it with a production pipeline.

**How:** The `process_topic()` function executes the full pipeline for a single topic:

1. **Research** -- calls `run_research(topic)` which patches the LLM and search modules, then invokes the LangGraph agent.
2. **Extract sources** -- parses the `* Title : URL` format from the markdown summary to get source URLs.
3. **Evaluate quality** -- calls `evaluate_research()` (see below). If the overall score is below 3.0/5.0, generates and executes a follow-up research run, merges the results, and re-evaluates.
4. **Save** -- writes the summary as a markdown file to `output/` with YAML frontmatter (topic, date). Filename format: `YYYYMMDD-<slugified-topic>.md`.
5. **Filter sources** -- calls `filter_relevant_sources()` (see below) before ingestion.
6. **Ingest** -- uploads the summary file and filtered source URLs to OpenViking via its HTTP API.
7. **Log** -- appends a JSON record to `research_log.jsonl` with topic, scores, timing, config, and source counts.

**Monkey-patches applied:**
- `_patch_lmstudio_api_key()` -- Replaces `ChatLMStudio.__init__` to inject `OPENAI_API_KEY` instead of the default `"not-needed-for-local-models"`.
- `_patch_tavily_query_length()` -- Wraps `utils.tavily_search` to truncate queries exceeding 390 characters (Tavily's 400-char API limit, with 10-char safety margin).

**CLI interface:**
```
python research.py "topic"                       # Single topic
python research.py --file topics/list.txt        # Batch from file (one topic per line, # for comments)
python research.py -l 7 "topic"                  # Override loop count
python research.py --no-ingest "topic"           # Skip OpenViking ingestion
```

**Where:** `/Users/russ/Documents/code/Oaasis/deep-researcher/research.py`


### 3. scripts/evaluate.py (Self-Critique)

**What:** Scores research output against a four-dimension rubric using gpt-5.4, and optionally generates a follow-up query targeting the weakest dimension.

**Why:** Based on research into self-feedback patterns (ACR pattern from RefineCoder). Naive iteration can degrade quality -- the LLM may drift off-topic or produce increasingly vague summaries over multiple loops. Rubric-based evaluation catches drift and low-quality output before it pollutes the knowledge base. When quality is poor, a targeted follow-up query addresses the specific weakness rather than just "try again."

**How:** Sends the research summary (truncated to 4000 chars) and source list to gpt-5.4 with a structured rubric prompt. The LLM returns JSON with four scores:

| Dimension | What it measures |
|---|---|
| **Coverage** | Did the research address the actual topic or drift off-topic? |
| **Source quality** | Are sources authoritative, or mostly SEO / low-quality content? |
| **Specificity** | Does it contain concrete findings, data, and details? |
| **Actionability** | Can someone use this information to make decisions or build things? |

Each dimension is scored 1-5. The overall score is the average. If overall < 3.0, the module generates a follow-up search query (under 200 chars) specifically targeting the weakest dimension.

`research.py` uses the follow-up query to re-run the entire research graph, then merges the new findings with the original summary and re-evaluates. If quality is still below 3.0 after retry, a warning banner is prepended to the saved output.

**Where:** `/Users/russ/Documents/code/Oaasis/deep-researcher/scripts/evaluate.py`


### 4. scripts/relevance.py (Source Filtering)

**What:** Filters source URLs for relevance before OpenViking ingestion by asking gpt-5.4 which sources are actually on-topic and worth keeping.

**Why:** Prevents off-topic content from leaking into the knowledge base. This was motivated by a real incident where a research run on "multi-turn LLM refinement" returned hardware verification content that would have polluted OpenViking. Since OpenViking indexes and retrieves based on these sources, bad sources degrade retrieval quality for all future queries.

**How:** Sends the topic, summary excerpt (first 1000 chars), and full source URL list to gpt-5.4. The LLM returns a JSON array of URLs to keep. The module validates that returned URLs exist in the original list (preventing hallucinated URLs) and falls back to keeping all sources if anything fails.

**Where:** `/Users/russ/Documents/code/Oaasis/deep-researcher/scripts/relevance.py`


### 5. OpenViking (Context Database)

**What:** A filesystem-paradigm context database that stores documents, embeds them with `text-embedding-3-large` (3072 dimensions), and provides semantic search via an MCP server.

**Why:** The knowledge base needs to be semantically searchable, not just a directory of markdown files. OpenViking provides tiered storage (L0 abstracts auto-generated from content, L1 overviews synthesized across related documents), embedding-based retrieval, and an MCP server that makes the knowledge base directly accessible to Claude Code and other AI tools. The `viking://` URI scheme gives every piece of content a stable address.

**How:**
- **Server** runs on `localhost:1933`, configured via `ov.conf`.
- **Embedding** uses OpenAI's `text-embedding-3-large` model (3072-dimensional vectors).
- **VLM** (Vision-Language Model, used for abstract/overview generation) uses `gpt-5.4` via OpenAI API.
- **Storage** is at `OpenViking/data/` with subdirectories for the vector DB, viking filesystem, and system metadata.
- **MCP server** (`openviking-mcp`) connects to the running server and exposes tools: `search`, `read_content`, `list_contents`, `add_memory`, `add_resource`, `health_check`, and others.
- **MCP server** runs as a persistent HTTP service managed by `ov-manager`, eliminating the need for stdio wrapper scripts.

Research is stored under the `viking://resources/research/` namespace. Each research run produces:
- `viking://resources/research/<topic-slug>` -- the summary document
- `viking://resources/research/<topic-slug>/sources/` -- individual source URLs as child resources

**Where:** `/Users/russ/Documents/code/Oaasis/OpenViking/`


### 6. /research Skill (Claude Code Orchestration)

**What:** A Claude Code custom skill that orchestrates multi-topic research sessions from within a Claude Code conversation.

**Why:** A single broad topic researched in one long run produces worse results than multiple focused runs. The skill decomposes broad topics, decides loop counts based on topic complexity and existing knowledge, and uses a subagent to keep the main context window clean.

**How:**
1. Checks OpenViking for existing coverage to avoid redundant research.
2. Decomposes the user's topic into 2-6 focused sub-topics, each with a loop count (2-3 for well-documented areas, 5-6 for complex topics, 7-8 for niche/cutting-edge).
3. Presents the plan to the user for confirmation.
4. Spawns a single subagent that runs `research.py` for each sub-topic sequentially.
5. The subagent reads output files after each run and returns a structured summary with key findings per sub-topic.
6. The skill presents findings to the user and offers follow-up research on identified gaps.

The subagent invocation pattern:
```
cd ~/Documents/code/Oaasis/deep-researcher && nix develop --command .venv/bin/python3 research.py -l <N> "<sub-topic>"
```

**Where:** `/Users/russ/Documents/code/Oaasis/.claude/skills/research/SKILL.md`


### 7. /knowledge Skill (Retrieval)

**What:** A Claude Code skill that queries OpenViking to answer questions grounded in stored research.

**Why:** Complements `/research`. Once knowledge has been built, any Claude Code session can query it without re-researching. The skill provides structured retrieval: search with multiple phrasings, read full content for top matches, synthesize a grounded answer with citations, and clearly identify gaps.

**How:**
1. Runs 2-3 semantic searches with different phrasings against `viking://resources/research/`.
2. Reads full content for the top 2-3 results (score minimum 0.35).
3. Synthesizes an answer that cites specific viking:// URIs and distinguishes well-covered areas from gaps.
4. If coverage is thin, suggests running `/research` to fill the gap.

**Where:** `/Users/russ/Documents/code/Oaasis/.claude/skills/knowledge/SKILL.md`


### 8. Configuration

**`.env`** (gitignored) -- API keys and runtime configuration:
```
LLM_PROVIDER=lmstudio          # Tells local-deep-researcher to use ChatLMStudio
LOCAL_LLM=gpt-5.4               # Model name (passed to OpenAI API as-is)
LMSTUDIO_BASE_URL=https://api.openai.com/v1   # The actual OpenAI endpoint
OPENAI_API_KEY=sk-...           # Used by monkey-patch, evaluate.py, relevance.py

SEARCH_API=tavily               # Search provider
TAVILY_API_KEY=tvly-...         # Tavily API credentials

MAX_WEB_RESEARCH_LOOPS=5        # Default iterations per research run
FETCH_FULL_PAGE=True            # Include raw page content in search results
STRIP_THINKING_TOKENS=False     # Don't strip <think> tags (not needed for gpt-5.4)
USE_TOOL_CALLING=False          # Use JSON mode, not tool calling

OPENVIKING_URL=http://localhost:1933
```

**`ov.conf`** (in OpenViking/) -- OpenViking server configuration:
- Embedding: `text-embedding-3-large`, 3072 dimensions, OpenAI provider
- VLM: `gpt-5.4`, OpenAI provider (for auto-generated abstracts and overviews)
- Storage: `./data`

**`flake.nix`** -- Nix dev shell providing Python 3.12, uv, and git.

**`.envrc`** -- direnv configuration that activates the Nix shell, sets up the uv virtualenv, and loads `.env`.

**`pyproject.toml`** -- Python project definition. Dependencies: `ollama-deep-researcher` (from GitHub), `httpx`. Managed with `uv`.


## Design Decisions

### 1. Why local-deep-researcher instead of custom search code?

The iterative gap-identification loop is the key value. Each cycle, the agent reflects on its accumulated knowledge, identifies what is missing, and generates a targeted follow-up query. This is a solved problem in LangGraph with well-tested prompt engineering. Building it from scratch would mean reinventing query generation, source deduplication, running summary management, and the reflection/routing logic -- all of which `local-deep-researcher` handles.

### 2. Why LMStudio provider pointed at OpenAI?

`local-deep-researcher` only supports two LLM providers: `ollama` and `lmstudio`. There is no native OpenAI provider. However, `ChatLMStudio` extends `ChatOpenAI` from langchain-openai and uses an OpenAI-compatible API format. By setting `LMSTUDIO_BASE_URL=https://api.openai.com/v1`, requests go directly to OpenAI's API. The catch: `ChatLMStudio.__init__` defaults `api_key` to `"not-needed-for-local-models"`, which OpenAI rejects. The monkey-patch in `research.py` intercepts `__init__` and injects the real `OPENAI_API_KEY`.

### 3. Why Tavily over DuckDuckGo?

Higher quality results with raw page content extraction. DuckDuckGo caps at 3 results per query and often misses source citations (the `href` field is sometimes empty). Tavily provides structured results with `raw_content` (full page text), which is critical because the summarizer needs actual content, not just snippets. With Tavily configured to `max_results=1` (the default in `web_research`), every search iteration gets one high-quality, full-content result rather than three shallow ones.

### 4. Why topic decomposition in the /research skill?

Multiple focused 5-loop runs produce far deeper knowledge than one 15-loop run. Each run stays on a single facet of the topic, so every search query is targeted. Since Tavily returns 1 result per query, every query counts -- a broad topic causes the agent to bounce between subtopics, wasting queries on context-switching rather than depth. Decomposition also allows different loop counts per sub-topic: well-documented areas need fewer loops, niche areas need more.

### 5. Why self-critique before ingestion?

Based on research into self-feedback patterns (ACR -- Autonomous Critique and Refinement -- from the RefineCoder paper). Naive iterative research can degrade quality: the running summary may drift off-topic, sources may be low-quality SEO content, or findings may be vague. The rubric-based evaluation catches these problems with a structured assessment across four dimensions. When quality is below threshold (3.0/5.0), the system generates a targeted follow-up query addressing the specific weakest dimension, runs another research cycle, merges the results, and re-evaluates. This is a single retry, not an unbounded loop -- if quality is still poor after retry, the output is flagged with a warning rather than discarded.

### 6. Why source relevance filtering?

Prevents off-topic content from leaking into OpenViking. This was motivated by a real incident: a research run on "multi-turn LLM refinement" returned hardware verification content via a tangentially-related search result. Without filtering, that content would have been ingested into OpenViking and would surface in future semantic searches, degrading retrieval precision. The filter asks gpt-5.4 to evaluate each URL against the topic and summary context, keeping only authoritative, on-topic sources.

### 7. Why a subagent in the /research skill?

Keeps Claude Code's main context window clean. A multi-topic research session can involve dozens of research runs, each producing thousands of tokens of output. If all of that were in the main conversation, the context window would fill up quickly. The subagent runs all topics sequentially, reads output files, and returns only a structured summary with key findings. The detailed research lives in OpenViking for any session that needs deeper access later.

### 8. Why sequential research runs, not parallel?

Tavily API rate limits and OpenViking's single-worker embedding queue. Concurrent requests to Tavily risk 429 errors, and concurrent ingestion into OpenViking can overwhelm its embedding pipeline. 2-3 concurrent runs might work technically but sequential is more reliable and easier to debug. The time cost is acceptable since research sessions are not latency-sensitive.

### 9. Why OpenViking?

It provides the three things a knowledge base needs that a directory of markdown files does not:
1. **Semantic search** -- embedding-based retrieval via `text-embedding-3-large` means queries find conceptually related content, not just keyword matches.
2. **Tiered abstraction** -- auto-generated L0 abstracts and L1 overviews mean you can browse at different granularity levels without reading every document.
3. **MCP integration** -- the MCP server makes the knowledge base a first-class tool for Claude Code and other AI agents, accessible through the standard `/knowledge` skill.

The `viking://` URI scheme provides stable addresses for every piece of content, making citations and cross-references possible.

### 10. Why log research runs?

Inspired by the MIRROR framework for self-improving agents: preserve prior run artifacts as evidence for future improvement. The log enables analysis of which topics, decomposition strategies, and loop counts produce the best results. Each record captures the topic, loop count, source counts, evaluation scores, duration, search API, and LLM model. Over time, this data can inform automated tuning of research parameters.


## Known Limitations

1. **Tavily 400-character query limit.** The LLM sometimes generates verbose follow-up queries that exceed Tavily's 400-character API limit. The monkey-patch truncates at 390 characters, which loses query specificity. A better approach would be to constrain query length in the prompt or summarize long queries.

2. **ChatLMStudio API key default.** The upstream `ChatLMStudio` class defaults `api_key` to `"not-needed-for-local-models"`. The monkey-patch works but is fragile -- if the upstream class changes its `__init__` signature, the patch breaks silently. A proper fix would be a PR to `local-deep-researcher` adding an `openai` provider, or supporting API key passthrough in the LMStudio provider.

3. **No parallel research runs.** Sequential execution means a session with 6 sub-topics at 5 loops each takes a long time. Parallelism is limited by Tavily rate limits and OpenViking's embedding queue.

4. **OpenViking has no delete tool in the MCP server.** Content can only be removed via the CLI or direct API calls, not through the MCP tools available to Claude Code. Stale or low-quality content that passes the quality gate stays in the knowledge base until manually cleaned up.

5. **DuckDuckGo sources sometimes missing in tool-calling mode.** When `USE_TOOL_CALLING=True`, the DuckDuckGo search sometimes returns results without the `href` field, causing source extraction to fail silently. This is not an issue with the current configuration (`USE_TOOL_CALLING=False`, `SEARCH_API=tavily`), but would surface if switching to DuckDuckGo.

6. **Research quality varies by topic.** Some topics get thin coverage despite many loops -- either because Tavily returns the same result repeatedly, or because the reflection step generates queries that are too similar to previous ones. The self-critique catches this but can only retry once.

7. **Single retry on low quality.** If the follow-up research run also scores below 3.0, the pipeline flags the output but still saves and ingests it. There is no mechanism for escalation beyond the single retry.

8. **No deduplication across runs.** If the same topic is researched twice, both outputs are ingested into OpenViking as separate resources. The knowledge base will contain redundant content.


## Future Improvements

- **Parallel research with configurable concurrency.** Allow 2-3 concurrent `research.py` processes with rate limiting. Would significantly reduce wall-clock time for multi-topic sessions.

- **Durable execution (Temporal / Restate).** Long research sessions (30+ minutes) are vulnerable to crashes, network interruptions, and process kills. A durable execution framework would provide checkpointing, automatic retry, and crash recovery so that a 6-topic session resumes from where it left off rather than restarting.

- **LoRA distillation of research critique patterns.** Fine-tune a small local model on the accumulated evaluation data to perform quality assessment without calling gpt-5.4. This would reduce cost and enable fully offline quality gating.

- **Automated knowledge base curation.** Detect and flag stale, redundant, or low-quality entries in OpenViking. Could use embedding similarity to find near-duplicates and evaluation scores from the research log to identify content that should be re-researched or removed.

- **Integration with agentskills.io ecosystem.** Package the `/research` and `/knowledge` skills for portable distribution, so other developers can use the same pipeline with their own OpenViking instances.

- **Smarter query diversity.** Track previous queries within a research run and penalize the reflection step for generating similar queries. Would reduce the "Tavily returns the same result" problem.

- **Configurable quality threshold.** The 3.0/5.0 threshold is hardcoded. Make it a CLI argument or environment variable so users can trade quality for speed.
