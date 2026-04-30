"""Self-critique module for research output quality evaluation.

Scores research output against a rubric using litellm and optionally
generates a follow-up query targeting the weakest dimension.
"""

import json
import os

import litellm

_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-6")

_RUBRIC_PROMPT = """\
You are a research quality evaluator. Score the following research output on four dimensions (1-5 each):

1. **Coverage** - Did the research address the actual topic or drift off-topic?
2. **Source quality** - Are the sources authoritative and credible, or mostly SEO content / low-quality blogs?
3. **Specificity** - Does it contain concrete findings, data, and details, or is it vague hand-waving?
4. **Actionability** - Can someone use this information to make decisions or build things?

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

    follow_up_query = None
    if overall < 3.0 and weakest:
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
