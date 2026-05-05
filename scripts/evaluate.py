"""Self-critique module for research output quality evaluation.

Scores research output against a rubric using litellm and optionally
generates a follow-up query targeting the weakest dimension.
"""

import json
import os

import litellm

_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-6")

_RUBRIC_PROMPT = """\
You are a research quality evaluator. Be a strict grader; the goal is to surface weak research, not to be encouraging.

Score the following research output on four dimensions (1-5 each).

1. **Coverage** — Did the research address the actual topic or drift off-topic? (Breadth, not depth.)

2. **Source quality** — Are the sources authoritative and credible?
   - 5 = primarily official docs, vendor engineering blogs on the vendor's own domain, GitHub repos, RFCs, peer-reviewed papers
   - 3 = mix of authoritative and secondary sources (industry magazines, well-known practitioner blogs)
   - 1-2 = mostly Medium posts, LinkedIn Pulse, dev.to listicles, "AI marketing" sites, SEO content, generic news aggregators
   Score down hard for Medium / LinkedIn / marketing-shaped domains unless the specific URL clearly carries unique primary material.

3. **Specificity** — Does it contain concrete, verifiable detail?
   - 5 = schemas, code snippets, version numbers, named systems with their actual configuration, specific dates/numbers with citations
   - 3 = mostly named entities and clear claims, but missing the artifacts a reader would need to act
   - 1-2 = vague hand-waving, marketing-shaped language ("leverages", "enables", "robust", "comprehensive"), no schemas/code/versions, sweeping numbers without provenance
   Treat unsourced statistics (e.g. "70% reduction in MTTR") as a specificity warning, not strength.

4. **Actionability** — Could a competent practitioner use this output, alone, to make a decision or write code?
   - 5 = decision-grade: comparisons with concrete trade-offs, runnable examples, named tools with their actual interfaces
   - 3 = directional guidance but the reader still has to do their own legwork
   - 1-2 = aspirational summary; no concrete next step

Research topic: {topic}

Sources found:
{sources}

Research summary:
{summary}

Respond with ONLY valid JSON (no markdown fencing) in this exact format:
{{
  "coverage": <int 1-5>,
  "source_quality": <int 1-5>,
  "specificity": <int 1-5>,
  "actionability": <int 1-5>,
  "overall": <float, average of the four scores>,
  "critique": "<1-2 sentence explanation of the biggest weakness>",
  "weakest_dimension": "<name of the lowest-scoring dimension>"
}}
"""

_FOLLOW_UP_PROMPT = """\
The following research on "{topic}" scored poorly on {weakest_dimension}.

Critique: {critique}

Original summary (first 500 chars):
{summary_excerpt}

Generate a single, focused search query (under 200 characters) that would help \
fill the gap identified in the critique. Target the {weakest_dimension} weakness specifically.

Respond with ONLY the search query text, nothing else.
"""


def evaluate_research(
    topic: str, summary: str, sources: list[str]
) -> dict:
    """Evaluate research quality against a rubric.

    Returns dict with keys: scores, overall, critique, follow_up_query.
    follow_up_query is non-None only when overall < 3.0.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "scores": {},
            "overall": 0.0,
            "critique": "No ANTHROPIC_API_KEY set, skipping evaluation",
            "follow_up_query": None,
        }

    sources_text = "\n".join(f"- {s}" for s in sources) if sources else "(no sources)"

    rubric_message = _RUBRIC_PROMPT.format(
        topic=topic,
        sources=sources_text,
        summary=summary[:4000],
    )

    # Score the research
    resp = litellm.completion(
        model=_MODEL,
        messages=[{"role": "user", "content": rubric_message}],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()
    # Strip markdown fencing if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        eval_data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "scores": {},
            "overall": 0.0,
            "critique": f"Failed to parse evaluation response: {raw[:200]}",
            "follow_up_query": None,
        }

    scores = {
        "coverage": eval_data.get("coverage", 0),
        "source_quality": eval_data.get("source_quality", 0),
        "specificity": eval_data.get("specificity", 0),
        "actionability": eval_data.get("actionability", 0),
    }
    overall = eval_data.get("overall", sum(scores.values()) / max(len(scores), 1))
    critique = eval_data.get("critique", "")
    weakest = eval_data.get("weakest_dimension", "")

    # Trigger a follow-up when the overall score is weak OR when any single
    # dimension is below 2.5 — catches the "broad-but-shallow" pattern where
    # source_quality scores 2 but coverage carries the average above 3.
    weakest_score = min(scores.values()) if scores else 0
    follow_up_query = None
    if (overall < 3.0 or weakest_score < 2.5) and weakest:
        # Generate a targeted follow-up query
        follow_up_message = _FOLLOW_UP_PROMPT.format(
            topic=topic,
            weakest_dimension=weakest,
            critique=critique,
            summary_excerpt=summary[:500],
        )
        resp = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": follow_up_message}],
            temperature=0.3,
        )
        follow_up_query = resp.choices[0].message.content.strip()
        # Truncate if needed
        if len(follow_up_query) > 200:
            follow_up_query = follow_up_query[:200]

    return {
        "scores": scores,
        "overall": round(overall, 2),
        "critique": critique,
        "follow_up_query": follow_up_query,
    }
