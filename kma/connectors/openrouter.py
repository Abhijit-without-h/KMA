"""OpenRouter agent loop with KMA as the agent's long-term memory.

OpenRouter is an OpenAI-compatible gateway, so this works for any model it
fronts (openai/*, anthropic/*, meta-llama/*, ...). The loop is the canonical
agentic-memory pattern:

    retrieve-before  ->  get_context(user_msg) injected into the system prompt
    generate         ->  one chat completion
    write-after      ->  add(user_msg) auto-placed into the memory hierarchy

The LLM call is injectable (`llm=`), so the whole loop is unit-testable without
network or an API key; the default path calls OpenRouter via stdlib urllib (no
extra dependency).
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable

from kma.memory import AgenticMemory

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_SYSTEM = (
    "You are a helpful assistant with long-term memory of this user. "
    "Use the provided memory when relevant; do not invent facts not supported by it."
)

Messages = list[dict[str, str]]


class OpenRouterAgent:
    def __init__(
        self,
        memory: AgenticMemory | None = None,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        scope: dict | None = None,
        system: str = DEFAULT_SYSTEM,
        remember: bool = True,
        extract_facts: bool = False,
        llm: Callable[[Messages], str] | None = None,
    ) -> None:
        self.memory = memory or AgenticMemory()
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.scope = scope or {}
        self.system = system
        self.remember = remember
        self._llm = llm  # injectable for tests; None -> real OpenRouter call
        # When True, write-after distills the turn into atomic facts via the LLM
        # instead of storing the raw user message.
        self.extractor = None
        if extract_facts:
            from kma.extract import LLMFactExtractor

            self.extractor = LLMFactExtractor(self._call)

    def chat(self, user_message: str) -> tuple[str, str]:
        """One agent turn. Returns (assistant_reply, injected_memory_context)."""
        context = self.memory.get_context(user_message, scope=self.scope)
        system = self.system
        if context:
            system += "\n\n# Memory about this user\n" + context
        messages: Messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        reply = self._call(messages)
        if self.remember:
            if self.extractor is not None:
                self.memory.ingest(user_message, extractor=self.extractor,
                                   scope=self.scope, source="user")
            else:
                self.memory.add(user_message, scope=self.scope, source="user")
        return reply, context

    # --- LLM backend ---------------------------------------------------------
    def _call(self, messages: Messages) -> str:
        if self._llm is not None:
            return self._llm(messages)
        if not self.api_key:
            raise RuntimeError("set OPENROUTER_API_KEY or pass api_key/llm=")
        body = json.dumps({"model": self.model, "messages": messages}).encode()
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/kma",
                "X-Title": "KMA agentic memory",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]
