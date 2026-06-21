"""LLM-based fact extraction.

Storing raw conversation turns makes memory noisy ("Remind me — what stack did I
say?" is not a fact worth keeping). An extractor distills a turn into atomic,
standalone, durable facts, which are then auto-placed into the hierarchy. This is
what makes recall clean instead of a transcript dump.

The LLM call is injectable, so extraction is testable without network. The prompt
asks for a strict JSON array of short strings; parsing is defensive (a bad/empty
response yields no facts rather than an error).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

Messages = list[dict[str, str]]

EXTRACTION_SYSTEM = (
    "You extract durable memory from a message. Return ONLY a JSON array of short, "
    "atomic, self-contained facts worth remembering long-term about the user or "
    "their project (preferences, identity, goals, decisions, stable facts). "
    "Rewrite each as a standalone statement (no pronouns like 'it'/'they' without "
    "a referent). Ignore questions, chit-chat, and transient remarks. If nothing is "
    'worth remembering, return []. Example: ["The user prefers FastAPI for backends"].'
)


def parse_facts(raw: str) -> list[str]:
    """Defensively pull a JSON string array out of an LLM response."""
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)  # tolerate prose/code-fence wrappers
    if match:
        raw = match.group(0)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [s.strip() for s in data if isinstance(s, str) and s.strip()]


class LLMFactExtractor:
    def __init__(self, llm: Callable[[Messages], str]) -> None:
        self._llm = llm

    def extract(self, text: str) -> list[str]:
        messages: Messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": text},
        ]
        return parse_facts(self._llm(messages))
