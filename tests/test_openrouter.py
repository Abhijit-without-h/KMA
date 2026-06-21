"""OpenRouter agentic-memory loop.

The wiring is tested with a fake LLM (no network, runs in CI). The real call is
exercised by a live test that auto-skips unless OPENROUTER_API_KEY is set.
"""

import os

import pytest

from kma.connectors import OpenRouterAgent
from kma.memory import AgenticMemory


def test_loop_injects_memory_and_writes_back():
    mem = AgenticMemory()
    scope = {"user_id": "alice"}
    mem.add("the user prefers python and fastapi for backends", scope=scope)

    captured = {}

    def fake_llm(messages):
        captured["messages"] = messages
        return "Great — I'll scaffold a FastAPI backend."

    agent = OpenRouterAgent(mem, llm=fake_llm, scope=scope)
    reply, context = agent.chat("what backend should I scaffold?")

    # generate: assistant replied
    assert reply.startswith("Great")
    # retrieve-before: prior memory was injected into the system prompt
    system_prompt = captured["messages"][0]["content"]
    assert "fastapi" in system_prompt.lower()
    assert "fastapi" in context.lower()
    # write-after: the new user turn was stored in memory (scoped)
    texts = [n.text for n in mem.engine.store.all()]
    assert any("what backend should I scaffold" in t for t in texts)


def test_remember_flag_disables_writes():
    mem = AgenticMemory()
    agent = OpenRouterAgent(mem, llm=lambda m: "ok", remember=False)
    before = len(mem.engine.store)
    agent.chat("hello there")
    assert len(mem.engine.store) == before


def test_missing_key_raises_without_llm():
    agent = OpenRouterAgent(AgenticMemory(), api_key=None)
    agent.api_key = None
    with pytest.raises(RuntimeError):
        agent.chat("hi")


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"),
                    reason="set OPENROUTER_API_KEY to run the live OpenRouter test")
def test_live_openrouter_roundtrip():
    mem = AgenticMemory()
    mem.add("the user's name is alice and she builds hiking apps",
            scope={"user_id": "alice"})
    agent = OpenRouterAgent(mem, scope={"user_id": "alice"})
    reply, context = agent.chat("what is my name?")
    assert isinstance(reply, str) and reply
    assert "alice" in context.lower()
