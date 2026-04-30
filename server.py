"""FastMCP server for the deep research pipeline.

Exposes the research-to-knowledge pipeline as MCP tools over
Streamable HTTP.

Usage:
    python server.py --port 8001
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastmcp import FastMCP
from fastmcp.server.dependencies import Progress, get_http_headers

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"

mcp = FastMCP(
    "deep-researcher",
    instructions=(
        "Deep research pipeline that iteratively searches the web and ingests "
        "results into OpenViking knowledge bases. Use 'research' to start a "
        "research task, 'list_research' to browse completed research, and "
        "'read_research' to read specific outputs."
    ),
)


@mcp.tool(task=True)
async def research(
    topic: str,
    loops: int = 5,
    openviking_url: str | None = None,
    progress: Progress = Progress(),
) -> dict:
    """Run the full deep research pipeline on a topic.

    Iteratively searches the web, evaluates quality, filters sources,
    and optionally ingests results into an OpenViking knowledge base.
    This is a long-running background task that reports progress.

    Args:
        topic: The research topic to investigate.
        loops: Number of research iterations (more loops = deeper research).
        openviking_url: OpenViking instance URL for ingestion. If not provided,
            falls back to the X-OpenViking-URL header from the HTTP request.
            If neither is set, research runs and saves output locally without ingestion.
    """
    import asyncio

    start = time.time()

    # Resolve OpenViking URL: explicit param > X-OpenViking-URL header
    if openviking_url is None:
        headers = get_http_headers()
        openviking_url = headers.get("x-openviking-url", "")

    ingest = bool(openviking_url)

    # Configure environment before importing research modules
    os.environ["MAX_WEB_RESEARCH_LOOPS"] = str(loops)
    if ingest:
        os.environ["OPENVIKING_URL"] = openviking_url

    # Import at call time to avoid import-time side effects and to
    # pick up the env vars we just set
    from research import (
        extract_sources,
        ingest_to_openviking,
        run_research,
        save_output,
    )
    from scripts.evaluate import evaluate_research

    # -- Step 1: Research --
    total_steps = 4 if ingest else 3
    await progress.set_total(total_steps)
    await progress.set_message(f"Researching: {topic}")

    result = await asyncio.to_thread(run_research, topic)
    summary = result["running_summary"]

    await progress.increment()

    # -- Step 2: Extract sources --
    source_urls = extract_sources(summary)

    # -- Step 3: Evaluate quality --
    await progress.set_message("Evaluating quality...")

    evaluation = await asyncio.to_thread(evaluate_research, topic, summary, source_urls)

    # Handle low-quality retry (same logic as research.py process_topic)
    quality_concern = False
    if evaluation.get("scores") and evaluation["overall"] < 3.0 and evaluation.get("follow_up_query"):
        await progress.set_message("Score below threshold, running follow-up research...")

        retry_result = await asyncio.to_thread(run_research, evaluation["follow_up_query"])
        retry_summary = retry_result["running_summary"]
        summary = summary + "\n\n---\n\n## Follow-up Research\n\n" + retry_summary
        retry_sources = extract_sources(retry_summary)
        source_urls = list(dict.fromkeys(source_urls + retry_sources))

        evaluation = await asyncio.to_thread(evaluate_research, topic, summary, source_urls)
        if evaluation.get("scores") and evaluation["overall"] < 3.0:
            quality_concern = True

    await progress.increment()

    # -- Step 4: Save output --
    await progress.set_message("Filtering sources...")

    if quality_concern:
        summary = (
            "> **Note:** This research scored below the quality threshold "
            "even after a follow-up iteration. Review with care.\n\n" + summary
        )

    # save_output uses relative paths, so we need to run from the project root
    original_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        filepath = await asyncio.to_thread(save_output, topic, summary)
    finally:
        os.chdir(original_cwd)

    await progress.increment()

    # -- Step 5: Ingest into OpenViking (optional) --
    sources_ingested = 0
    if ingest:
        await progress.set_message(f"Ingesting into OpenViking ({openviking_url})...")

        original_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            sources_ingested = await asyncio.to_thread(
                ingest_to_openviking, filepath, source_urls, topic, summary
            )
        finally:
            os.chdir(original_cwd)

        await progress.increment()

    duration = round(time.time() - start, 1)

    return {
        "topic": topic,
        "sources_found": len(source_urls),
        "sources_ingested": sources_ingested,
        "evaluation_scores": evaluation.get("scores", {}),
        "evaluation_overall": evaluation.get("overall", 0.0),
        "evaluation_critique": evaluation.get("critique", ""),
        "output_file": str(filepath),
        "duration_seconds": duration,
    }


@mcp.tool
def list_research(limit: int = 20) -> list[dict]:
    """Browse completed research in the output directory.

    Returns a list of research outputs sorted by date (most recent first),
    with filename, topic, date, and file size for each entry.

    Args:
        limit: Maximum number of results to return.
    """
    if not OUTPUT_DIR.exists():
        return []

    entries = []
    for filepath in sorted(OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        # Parse frontmatter for topic and date
        topic = ""
        date = ""
        try:
            content = filepath.read_text(encoding="utf-8")
            if content.startswith("---"):
                _, frontmatter, _ = content.split("---", 2)
                for line in frontmatter.strip().splitlines():
                    if line.startswith("topic:"):
                        topic = line[len("topic:"):].strip()
                    elif line.startswith("date:"):
                        date = line[len("date:"):].strip()
        except (ValueError, OSError):
            pass

        entries.append({
            "filename": filepath.name,
            "topic": topic,
            "date": date,
            "size_bytes": filepath.stat().st_size,
        })

        if len(entries) >= limit:
            break

    return entries


@mcp.tool
def read_research(filename: str) -> str:
    """Read a specific research output file.

    Args:
        filename: The filename from list_research (e.g. '20260419-my-topic.md').
    """
    filepath = OUTPUT_DIR / filename

    # Prevent path traversal
    if not filepath.resolve().parent == OUTPUT_DIR.resolve():
        return f"Error: invalid filename '{filename}'"

    if not filepath.exists():
        return f"Error: file '{filename}' not found in output directory"

    return filepath.read_text(encoding="utf-8")


@mcp.tool
def health() -> dict:
    """Check server and dependency health.

    Returns server status, OpenViking reachability, .env key presence,
    and output directory stats.
    """
    import httpx

    # Check required .env keys
    required_keys = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    env_status = {}
    for key in required_keys:
        value = os.environ.get(key, "")
        env_status[key] = "set" if value else "missing"

    # Check OpenViking reachability (from X-OpenViking-URL header)
    headers = get_http_headers()
    ov_url = headers.get("x-openviking-url", "")
    openviking_reachable = False
    openviking_error = None
    if ov_url:
        try:
            resp = httpx.get(f"{ov_url}/health", timeout=5)
            resp.raise_for_status()
            openviking_reachable = True
        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException) as e:
            openviking_error = str(e)
    else:
        openviking_error = "No X-OpenViking-URL header set"

    # Output directory stats
    output_file_count = 0
    if OUTPUT_DIR.exists():
        output_file_count = len(list(OUTPUT_DIR.glob("*.md")))

    return {
        "server": "ok",
        "openviking": {
            "url": ov_url,
            "reachable": openviking_reachable,
            "error": openviking_error,
        },
        "env_keys": env_status,
        "output": {
            "path": str(OUTPUT_DIR),
            "file_count": output_file_count,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep researcher MCP server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="HTTP port (default: 8001)",
    )
    args = parser.parse_args()

    mcp.run(transport="streamable-http", host=args.host, port=args.port)
