"""Live OpenRouter agent with KMA memory across turns.

    export OPENROUTER_API_KEY=sk-or-...
    python examples/openrouter_agent.py

Watch memory accumulate: facts from early turns are recalled in later answers.
"""

from __future__ import annotations

from kma.connectors import OpenRouterAgent
from kma.memory import AgenticMemory


def main() -> None:
    agent = OpenRouterAgent(AgenticMemory(), scope={"user_id": "demo"})

    turns = [
        "Hi! I'm building a travel app for hikers, in Python with FastAPI.",
        "I want offline maps so people can navigate without signal.",
        "Remind me — what stack and key feature did I say I'm using?",
    ]
    for msg in turns:
        reply, context = agent.chat(msg)
        print(f"\nUSER: {msg}")
        if context:
            print(f"[memory injected]\n{context}")
        print(f"ASSISTANT: {reply}")


if __name__ == "__main__":
    main()
