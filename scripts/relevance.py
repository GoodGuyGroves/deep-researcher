"""Source relevance filtering module.

Filters source URLs before OpenViking ingestion by asking the LLM
(via litellm) which sources are actually relevant to the research topic.
"""

import json
import os

import litellm

_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-6")

_FILTER_PROMPT = """\
You are evaluating source URLs from a research report for relevance and authority.

Research topic: {topic}

Research summary (first 1000 chars):
{summary_excerpt}

Source URLs found in the report:
{sources_list}

# Decision criteria

Keep a source if it is **on-topic AND authoritative AND substantive**. All three.

Source authority — strong default priors (override only with strong evidence):

PRIMARY (usually keep, on-topic):
- Official vendor / project docs (anthropic.com/docs, kubernetes.io, etc.)
- Vendor / project engineering blogs on their own domain
- GitHub repos, source code, RFCs, design documents
- Standards bodies (W3C, IETF, IEEE, ISO)
- Peer-reviewed academic papers (arxiv.org, acm.org, usenix.org)
- Official release notes, security advisories, postmortems on company domains

LOW-PRIORITY (filter aggressively unless the URL clearly contains unique primary material):
- Medium posts (medium.com, *.medium.com, "@author" pages)
- LinkedIn Pulse posts (linkedin.com/pulse)
- dev.to, hashnode.com, freecodecamp.org listicles
- "AI marketing" or vendor-comparison landing pages with no engineering detail
- SEO listicle / round-up content ("top 10", "best X for Y")
- Generic news aggregators (tldr.tech, infoq summaries of other content)
- Personal blogs unless the author is a known practitioner with depth on the topic

A great post on a low-priority domain (e.g. an AWS principal engineer writing on Medium with concrete code) CAN be kept — the prior is rebuttable, not absolute. But the default is to filter.

Also remove: off-topic, paywalled with no useful preview, dead/landing pages, content that's just a vendor's marketing pitch.

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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return sources

    sources_list = "\n".join(f"- {url}" for url in sources)

    prompt = _FILTER_PROMPT.format(
        topic=topic,
        summary_excerpt=summary[:1000],
        sources_list=sources_list,
    )

    try:
        resp = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        raw = resp.choices[0].message.content.strip()
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

    except Exception as e:
        print(f"    Source filtering failed ({e}), keeping all sources")
        return sources
