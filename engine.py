"""Core research engine — iterative web search and summarization loop.

Replaces the upstream ollama-deep-researcher LangGraph pipeline with a
plain-Python loop using litellm for LLM calls and tavily-python for
web search. No LangChain/LangGraph dependency.

Usage:
    from engine import run_research_loop, ResearchConfig

    result = run_research_loop("your topic")
    print(result["running_summary"])
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import litellm
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4
MAX_TAVILY_QUERY_LENGTH = 390  # Tavily API limit is 400; leave 10-char margin


@dataclass
class ResearchConfig:
    """Runtime configuration for a research run."""

    llm_model: str = "anthropic/claude-sonnet-4-6"
    max_loops: int = 5
    max_results_per_search: int = 1
    max_tokens_per_source: int = 1000
    fetch_full_page: bool = True

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        return cls(
            llm_model=os.environ.get("LLM_MODEL", cls.llm_model),
            max_loops=int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", str(cls.max_loops))),
            fetch_full_page=os.environ.get("FETCH_FULL_PAGE", "true").lower() == "true",
        )


# ---------------------------------------------------------------------------
# Prompts (adapted from langchain-ai/local-deep-researcher)
# ---------------------------------------------------------------------------

QUERY_WRITER_PROMPT = """\
Your goal is to generate a targeted web search query.

<CONTEXT>
Current date: {current_date}
Please ensure your queries account for the most current information available as of this date.
</CONTEXT>

<TOPIC>
{research_topic}
</TOPIC>

<FORMAT>
Format your response as a JSON object with these exact keys:
- "query": The actual search query string
- "rationale": Brief explanation of why this query is relevant
</FORMAT>

Provide your response in JSON format:"""

SUMMARIZER_PROMPT = """\
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:
1. Read the existing summary and new search results carefully.
2. Compare the new information with the existing summary.
3. For each piece of new information:
    a. If it's related to existing points, integrate it into the relevant paragraph.
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.
    c. If it's not relevant to the user topic, skip it.
4. Ensure all additions are relevant to the user's topic.
5. Verify that your final output differs from the input summary.
</REQUIREMENTS>

<FORMATTING>
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.
</FORMATTING>

<Task>
Think carefully about the provided Context first. Then generate a summary of the context \
to address the User Input.
</Task>"""

REFLECTION_PROMPT = """\
You are an expert research assistant analyzing a summary about {research_topic}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- "knowledge_gap": Describe what information is missing or needs clarification
- "follow_up_query": Write a specific question to address this gap
</FORMAT>

Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. \
Then, produce your output following this JSON format:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}

Provide your analysis in JSON format:"""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _llm_call(model: str, messages: list[dict], json_mode: bool = False) -> str:
    """Call an LLM via litellm and return the content string.

    When *json_mode* is True, attempts to extract a JSON object from the
    response by finding the first ``{`` and last ``}``.
    """
    response = litellm.completion(model=model, messages=messages, temperature=0)
    content = response.choices[0].message.content or ""

    if json_mode and content:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            candidate = content[start:end]
            try:
                json.loads(candidate)  # validate
                return candidate
            except json.JSONDecodeError:
                pass
    return content


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _tavily_search(
    query: str,
    max_results: int = 1,
    include_raw_content: bool = True,
) -> dict:
    """Run a Tavily web search, truncating long queries to stay within the API limit."""
    if len(query) > MAX_TAVILY_QUERY_LENGTH:
        query = query[:MAX_TAVILY_QUERY_LENGTH]

    client = TavilyClient()
    return client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
    )


def _deduplicate_and_format_sources(
    search_response: dict,
    max_tokens_per_source: int,
    fetch_full_page: bool = False,
) -> str:
    """Deduplicate search results by URL and format them for the LLM."""
    results = search_response.get("results", [])

    seen_urls: set[str] = set()
    formatted = "Sources:\n\n"

    for source in results:
        url = source.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = source.get("title", "")
        snippet = source.get("content", "")
        formatted += f"Source: {title}\n===\n"
        formatted += f"URL: {url}\n===\n"
        formatted += f"Most relevant content from source: {snippet}\n===\n"

        if fetch_full_page:
            char_limit = max_tokens_per_source * CHARS_PER_TOKEN
            raw = source.get("raw_content") or ""
            if len(raw) > char_limit:
                raw = raw[:char_limit] + "... [truncated]"
            formatted += (
                f"Full source content limited to {max_tokens_per_source} tokens: {raw}\n\n"
            )

    return formatted.strip()


def _format_sources(search_response: dict) -> str:
    """Produce a ``* Title : URL`` bullet list for the sources section."""
    return "\n".join(
        f"* {s['title']} : {s['url']}" for s in search_response.get("results", [])
    )


# ---------------------------------------------------------------------------
# Research-loop nodes
# ---------------------------------------------------------------------------


def generate_query(topic: str, config: ResearchConfig) -> str:
    """Generate an initial search query from the research topic."""
    prompt = QUERY_WRITER_PROMPT.format(
        current_date=datetime.now().strftime("%B %d, %Y"),
        research_topic=topic,
    )
    raw = _llm_call(
        config.llm_model,
        [{"role": "user", "content": prompt}],
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        return data.get("query", f"Tell me more about {topic}")
    except (json.JSONDecodeError, AttributeError):
        return f"Tell me more about {topic}"


def web_research(query: str, config: ResearchConfig) -> tuple[str, str]:
    """Execute a web search and return (formatted_text_for_llm, source_bullet_list)."""
    results = _tavily_search(
        query,
        max_results=config.max_results_per_search,
        include_raw_content=config.fetch_full_page,
    )
    formatted = _deduplicate_and_format_sources(
        results, config.max_tokens_per_source, config.fetch_full_page
    )
    sources = _format_sources(results)
    return formatted, sources


def summarize_sources(
    topic: str,
    existing_summary: str | None,
    new_research: str,
    config: ResearchConfig,
) -> str:
    """Create or extend the running summary with new search results."""
    if existing_summary:
        human_msg = (
            f"<Existing Summary>\n{existing_summary}\n</Existing Summary>\n\n"
            f"<New Context>\n{new_research}\n</New Context>\n\n"
            f"Update the Existing Summary with the New Context on this topic:\n\n"
            f"<User Input>\n{topic}\n</User Input>"
        )
    else:
        human_msg = (
            f"<Context>\n{new_research}\n</Context>\n\n"
            f"Create a Summary using the Context on this topic:\n\n"
            f"<User Input>\n{topic}\n</User Input>"
        )

    return _llm_call(
        config.llm_model,
        [
            {"role": "system", "content": SUMMARIZER_PROMPT},
            {"role": "user", "content": human_msg},
        ],
        json_mode=False,
    )


def reflect_on_summary(topic: str, summary: str, config: ResearchConfig) -> str:
    """Identify knowledge gaps and generate a follow-up search query."""
    prompt = REFLECTION_PROMPT.format(research_topic=topic)
    human_msg = (
        f"Reflect on our existing knowledge:\n===\n{summary}\n===\n\n"
        "And now identify a knowledge gap and generate a follow-up web search query:"
    )
    raw = _llm_call(
        config.llm_model,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": human_msg},
        ],
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        return data.get("follow_up_query", f"Tell me more about {topic}")
    except (json.JSONDecodeError, AttributeError):
        return f"Tell me more about {topic}"


def _finalize_summary(summary: str, all_sources: list[str]) -> str:
    """Deduplicate sources and format the final output."""
    seen: set[str] = set()
    unique: list[str] = []
    for source_block in all_sources:
        for line in source_block.split("\n"):
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique.append(line)

    all_sources_text = "\n".join(unique)
    return f"## Summary\n\n{summary}\n\n### Sources:\n{all_sources_text}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_research_loop(
    topic: str,
    config: ResearchConfig | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Run the iterative research loop on a topic.

    Args:
        topic: The research topic to investigate.
        config: Runtime configuration. Defaults to ``ResearchConfig.from_env()``.
        on_progress: Optional callback ``(current_loop, total_loops)`` for
            progress reporting.

    Returns:
        Dict with ``"running_summary"`` key containing the final markdown
        report with sources.
    """
    if config is None:
        config = ResearchConfig.from_env()

    # Step 1: Generate initial search query
    query = generate_query(topic, config)
    summary: str | None = None
    all_sources: list[str] = []

    # Step 2–4: Research loop
    for i in range(config.max_loops):
        # Web research
        search_text, source_list = web_research(query, config)
        all_sources.append(source_list)

        if on_progress:
            on_progress(i + 1, config.max_loops)

        # Summarize
        summary = summarize_sources(topic, summary, search_text, config)

        # Reflect — generate follow-up query for the next iteration
        query = reflect_on_summary(topic, summary, config)

    # Step 5: Finalize
    final = _finalize_summary(summary or "", all_sources)

    return {"running_summary": final}
