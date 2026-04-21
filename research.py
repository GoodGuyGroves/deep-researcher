"""Research-to-knowledge pipeline.

Runs local-deep-researcher on a topic, saves the output, and ingests
the summary + source URLs into OpenViking.

Usage:
    python research.py "your research topic"
    python research.py --file topics/my-topics.txt
    python research.py --file topics/my-topics.txt --no-ingest   # research only, skip OpenViking
    python research.py --url http://localhost:1934 "topic"    # target a specific OpenViking instance
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import httpx

from scripts.evaluate import evaluate_research
from scripts.relevance import filter_relevant_sources

RESEARCH_LOG_PATH = Path(__file__).parent / "research_log.jsonl"


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80].strip("-")


def _patch_lmstudio_api_key():
    """Patch ChatLMStudio to use OPENAI_API_KEY when pointing at OpenAI's API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return

    from ollama_deep_researcher.lmstudio import ChatLMStudio

    _orig_init = ChatLMStudio.__init__

    def _patched_init(self, api_key="not-needed-for-local-models", **kwargs):
        real_key = os.environ.get("OPENAI_API_KEY", api_key)
        _orig_init(self, api_key=real_key, **kwargs)

    ChatLMStudio.__init__ = _patched_init


def _patch_tavily_query_length():
    """Patch Tavily search to truncate queries that exceed the 400-char API limit."""
    try:
        import ollama_deep_researcher.utils as utils

        _orig_tavily = utils.tavily_search

        def _truncated_tavily(query, *args, **kwargs):
            if len(query) > 390:
                query = query[:390]
            return _orig_tavily(query, *args, **kwargs)

        utils.tavily_search = _truncated_tavily
    except (ImportError, AttributeError):
        pass


def run_research(topic: str) -> dict:
    """Run the deep researcher graph on a topic. Returns the full state dict."""
    _patch_lmstudio_api_key()
    _patch_tavily_query_length()
    from ollama_deep_researcher.graph import graph

    print(f"  Researching: {topic}")
    print(f"  LLM: {os.environ.get('LOCAL_LLM', 'llama3.2')}")
    print(f"  Search: {os.environ.get('SEARCH_API', 'duckduckgo')}")
    print(f"  Loops: {os.environ.get('MAX_WEB_RESEARCH_LOOPS', '3')}")
    print()

    result = graph.invoke({"research_topic": topic})
    return result


def extract_sources(summary: str) -> list[str]:
    """Extract source URLs from the markdown summary."""
    urls = []
    for line in summary.split("\n"):
        line = line.strip()
        if line.startswith("*") and ":" in line and "http" in line:
            # Format: * Title : https://url
            match = re.search(r"https?://\S+", line)
            if match:
                urls.append(match.group(0))
    return urls


def save_output(topic: str, summary: str) -> Path:
    """Save research output to a markdown file. Returns the file path."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    slug = slugify(topic)
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{timestamp}-{slug}.md"
    filepath = output_dir / filename

    content = f"---\ntopic: {topic}\ndate: {datetime.now().isoformat()}\n---\n\n{summary}"
    filepath.write_text(content)
    print(f"  Saved: {filepath}")
    return filepath


def ingest_to_openviking(
    filepath: Path, source_urls: list[str], topic: str = "", summary: str = ""
) -> int:
    """Upload the research summary and source URLs to OpenViking.

    Returns the number of source URLs actually ingested.
    """
    ov_url = os.environ.get("OPENVIKING_URL", "http://localhost:1933")

    client = httpx.Client(base_url=ov_url, timeout=120)

    # Check server health
    try:
        resp = client.get("/health")
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError) as e:
        print(f"  OpenViking server not reachable at {ov_url}: {e}")
        print("  Skipping ingestion. Start the server and re-ingest manually.")
        return 0

    # Step 1: Upload the summary file
    print(f"  Uploading summary to OpenViking...")
    with open(filepath, "rb") as f:
        resp = client.post(
            "/api/v1/resources/temp_upload",
            files={"file": (filepath.name, f, "text/markdown")},
        )
    resp.raise_for_status()
    temp_file_id = resp.json()["result"]["temp_file_id"]

    # Add the uploaded file as a resource
    topic_slug = filepath.stem
    resp = client.post(
        "/api/v1/resources",
        json={
            "temp_file_id": temp_file_id,
            "to": f"viking://resources/research/{topic_slug}",
            "wait": False,
        },
    )
    resp.raise_for_status()
    print(f"  Ingested summary as viking://resources/research/{topic_slug}")

    # Step 2: Filter and ingest source URLs
    ingested = 0
    if source_urls:
        # Relevance filtering
        filtered_urls = filter_relevant_sources(topic, source_urls, summary)
        filtered_out = len(source_urls) - len(filtered_urls)
        if filtered_out > 0:
            print(f"  Filtered out {filtered_out}/{len(source_urls)} irrelevant sources")
        print(f"  Ingesting {len(filtered_urls)} source URLs...")

        for url in filtered_urls:
            try:
                resp = client.post(
                    "/api/v1/resources",
                    json={
                        "path": url,
                        "parent": f"viking://resources/research/{topic_slug}/sources",
                        "wait": False,
                    },
                )
                resp.raise_for_status()
                ingested += 1
            except (httpx.HTTPStatusError, httpx.ConnectError) as e:
                print(f"    Failed to ingest {url}: {e}")
        print(f"  Ingested {ingested}/{len(filtered_urls)} source URLs")

    # Wait for background processing
    print("  Waiting for OpenViking to finish indexing...")
    resp = client.post("/api/v1/system/wait", timeout=300)
    print("  Done.")

    client.close()
    return ingested


def _append_research_log(record: dict) -> None:
    """Append a single JSON record to the research log."""
    with open(RESEARCH_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def process_topic(topic: str, ingest: bool = True) -> None:
    """Run the full pipeline for a single topic."""
    print(f"\n{'='*60}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    start = time.time()
    loop_count = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
    search_api = os.environ.get("SEARCH_API", "duckduckgo")
    llm_model = os.environ.get("LOCAL_LLM", "llama3.2")

    # Research
    result = run_research(topic)
    summary = result["running_summary"]
    research_elapsed = time.time() - start
    print(f"\n  Research completed in {research_elapsed:.0f}s")

    # Extract sources
    source_urls = extract_sources(summary)
    print(f"  Found {len(source_urls)} source URLs")

    # Evaluate quality
    print(f"\n  Evaluating research quality...")
    evaluation = evaluate_research(topic, summary, source_urls)
    quality_concern = False

    if evaluation["scores"]:
        scores = evaluation["scores"]
        print(f"  Scores: coverage={scores.get('coverage')}, "
              f"source_quality={scores.get('source_quality')}, "
              f"specificity={scores.get('specificity')}, "
              f"actionability={scores.get('actionability')}")
        print(f"  Overall: {evaluation['overall']}/5.0")
        if evaluation["critique"]:
            print(f"  Critique: {evaluation['critique']}")

        # If score is low, run one retry with the follow-up query
        if evaluation["overall"] < 3.0 and evaluation["follow_up_query"]:
            print(f"\n  Score below 3.0 - running follow-up research...")
            print(f"  Follow-up query: {evaluation['follow_up_query']}")

            retry_result = run_research(evaluation["follow_up_query"])
            retry_summary = retry_result["running_summary"]

            # Merge: append retry findings to original summary
            summary = summary + "\n\n---\n\n## Follow-up Research\n\n" + retry_summary
            retry_sources = extract_sources(retry_summary)
            source_urls = list(dict.fromkeys(source_urls + retry_sources))  # dedupe, preserve order
            print(f"  Total sources after follow-up: {len(source_urls)}")

            # Re-evaluate
            print(f"  Re-evaluating combined research...")
            evaluation = evaluate_research(topic, summary, source_urls)
            if evaluation["scores"]:
                scores = evaluation["scores"]
                print(f"  Scores: coverage={scores.get('coverage')}, "
                      f"source_quality={scores.get('source_quality')}, "
                      f"specificity={scores.get('specificity')}, "
                      f"actionability={scores.get('actionability')}")
                print(f"  Overall: {evaluation['overall']}/5.0")
                if evaluation["overall"] < 3.0:
                    quality_concern = True
                    print(f"  WARNING: Quality still below threshold after retry")
    else:
        print(f"  Evaluation skipped: {evaluation['critique']}")

    # Save (possibly with quality warning)
    if quality_concern:
        summary = (
            "> **Note:** This research scored below the quality threshold "
            "even after a follow-up iteration. Review with care.\n\n" + summary
        )
    filepath = save_output(topic, summary)

    # Ingest
    sources_ingested = 0
    if ingest:
        sources_ingested = ingest_to_openviking(filepath, source_urls, topic, summary)
    else:
        print("  Skipping OpenViking ingestion (--no-ingest)")

    # Log the run
    duration = time.time() - start
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "loop_count": loop_count,
        "sources_found": len(source_urls),
        "sources_ingested": sources_ingested,
        "evaluation": {
            "scores": evaluation.get("scores", {}),
            "overall": evaluation.get("overall", 0.0),
            "critique": evaluation.get("critique", ""),
        },
        "duration_seconds": round(duration, 1),
        "search_api": search_api,
        "llm_model": llm_model,
    }
    _append_research_log(log_record)
    print(f"\n  Run logged ({duration:.0f}s total)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Research topics and ingest into OpenViking"
    )
    parser.add_argument("topic", nargs="?", help="Research topic")
    parser.add_argument(
        "--file", "-f", help="File with one topic per line"
    )
    parser.add_argument(
        "--loops", "-l",
        type=int,
        help="Number of research iterations (overrides MAX_WEB_RESEARCH_LOOPS)",
    )
    parser.add_argument(
        "--url", "-u",
        help="OpenViking server URL (overrides OPENVIKING_URL env var)",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip OpenViking ingestion (research and save only)",
    )
    args = parser.parse_args()

    if args.loops:
        os.environ["MAX_WEB_RESEARCH_LOOPS"] = str(args.loops)

    if args.url:
        os.environ["OPENVIKING_URL"] = args.url

    if not args.topic and not args.file:
        parser.print_help()
        sys.exit(1)

    topics = []
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        topics = [
            line.strip()
            for line in filepath.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.topic:
        topics = [args.topic]

    print(f"\nResearching {len(topics)} topic(s)...\n")

    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}]")
        process_topic(topic, ingest=not args.no_ingest)

    print(f"\nAll done. Researched {len(topics)} topic(s).")


if __name__ == "__main__":
    main()
