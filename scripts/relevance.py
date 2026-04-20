"""Source relevance filtering module.

Filters source URLs before OpenViking ingestion by asking gpt-5.4
which sources are actually relevant to the research topic.
"""

import json
import os

import httpx

_MODEL = "gpt-5.4"
_API_URL = "https://api.openai.com/v1/chat/completions"

_FILTER_PROMPT = """\
You are evaluating source URLs from a research report for relevance.

Research topic: {topic}

Research summary (first 1000 chars):
{summary_excerpt}

Source URLs found in the report:
{sources_list}

For each source, decide whether it is relevant enough to ingest into a knowledge base \
about this topic. Keep sources that are authoritative, on-topic, and contain substantial \
information. Remove sources that are off-topic, low-quality SEO content, generic landing \
pages, or paywalled with no useful preview.

Respond with ONLY a JSON array of the URLs to KEEP (no markdown fencing), e.g.:
["https://example.com/good-source", "https://example.com/another-good-one"]
"""


def filter_relevant_sources(
    topic: str, sources: list[str], summary: str
) -> list[str]:
    """Filter source URLs to only those relevant to the topic.

    Returns the filtered list of URLs. If filtering fails, returns the
    original list unchanged.
    """
    if not sources:
        return sources

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return sources

    sources_list = "\n".join(f"- {url}" for url in sources)

    prompt = _FILTER_PROMPT.format(
        topic=topic,
        summary_excerpt=summary[:1000],
        sources_list=sources_list,
    )

    client = httpx.Client(timeout=60)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = client.post(
            _API_URL,
            headers=headers,
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        kept_urls = json.loads(raw)
        if not isinstance(kept_urls, list):
            return sources

        # Only keep URLs that were in the original list
        valid_kept = [url for url in kept_urls if url in sources]
        return valid_kept if valid_kept else sources

    except (httpx.HTTPStatusError, httpx.ConnectError, json.JSONDecodeError, KeyError) as e:
        print(f"    Source filtering failed ({e}), keeping all sources")
        return sources
    finally:
        client.close()
